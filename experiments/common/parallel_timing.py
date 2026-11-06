"""Multi-core query timing (complete queries; R11_parallel): how each step of a query scales with the number of cores.

The single-thread timings (timing.py, R4) are the paper's benchmark. This module times the same steps on T physical
cores, one worker per core (no SMT siblings), to show which steps of a query split across cores. src/ is frozen
(method-v2), so the parallel versions here are experiment code built on the public pieces, and each is asserted equal
to its single-thread counterpart, bit for bit, on the warm-up queries:

  qlog_whole   the whole-matrix vector (one p x p eigendecomposition): index_cache.whole_vectorizer(p).vector on T BLAS
               threads (threadpoolctl; OpenBLAS defaults, whose threads spin briefly between calls). Timed in a pass of
               its own over all queries, before the worker pool exists; LAPACK decides how far one dense
               eigendecomposition splits.
  qlog_block   the query's block logs in every niche (shared by Spindle-naive and Spindle): the (niche, block) pairs are
               independent; the pool's T workers take them, balanced by longest-processing-time on each block's measured
               single-thread time, and compute metrics.log_spd(cov[g, g]) * weights exactly as BlockVectorizer.vector.
               Equal to SpindleNaive.query_vectors bit for bit (asserted).
  whole        kernels.naive_scan over the whole-matrix vectors, rows split into T chunks over the pool + stable argsort
               (main process, one core).
  naive        Spindle-naive: kernels.naive_scan over every niche's block-log vectors, rows split into chunks over the
               pool + stable argsort (equal to SpindleNaive.distances, asserted).
  spindle      Spindle.search_vectors (g = 0.25): one call into the C kernel per query, one core whatever T (as R4).
  spindle_par  Spindle's search split by niche (method-v3: FlatSpindle.probe_niche / scan_niche / assemble, the same
               output as search_vectors bit for bit, asserted): the pool's workers probe their niches, the main process
               takes the smallest leaf A^2 and the radius, the workers scan their niches, the main process assembles
               the ranking. Niches go to workers by longest-processing-time on their measured single-thread time, so at
               most n_niche workers (brain: 18) have work. At T = 1 it is the single-core search (spindle_par = spindle).

The pool: T forked worker processes (processes, not threads: a block log is a few tens of microseconds of mostly
GIL-holding numpy calls), each pinned to its own physical core; the main process is pinned to one more core. Inputs
(query covariance, whole-matrix query vector) and outputs (block-log vectors, distances) live in shared memory; the
index arrays are inherited at fork. Workers and the main process busy-wait on a shared sequence number instead of
sleeping: the nodes run the schedutil frequency governor, under which a core that has just woken from a sleep runs
the next few milliseconds at a lower clock (measured: Spindle's search 3x slower right after a 20 ms sleep), which
would charge every T > 1 step for a frequency ramp that T = 1 never pays. Busy-waiting is the best case for dispatch.

T = 1 runs every step in-process exactly as timing.whole_tile (R4) does, and also times the block logs by the
pool's per-block route on one core (qlog_block_blockwise, the baseline of the block logs' scaling). Every timed step is wall time per query
(perf_counter), including the pool's dispatch and synchronisation.
"""

import mmap
import multiprocessing as mp
import os
import socket
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import data
import index_cache
from spindle_dev.kernels import naive_scan
from spindle_dev.metrics import LOG_FLOOR, as_cov, log_spd

STEPS = ["qlog_whole_ms", "qlog_block_ms", "whole_ms", "naive_ms", "spindle_ms", "spindle_par_ms"]
N_WARM = 3
IDLE_S = 1.0
CMD_BLOCK, CMD_WHOLE, CMD_NAIVE, CMD_PROBE, CMD_SCAN, CMD_EXIT = 1, 2, 3, 4, 5, 9
G = 0.25  # Spindle's knob, as R4


# ---------------------------------------------------------------- cores
def physical_cores():
    """One logical CPU per physical core among the CPUs this process may run on, socket by socket."""
    seen, out = set(), []
    for cpu in sorted(os.sched_getaffinity(0)):
        topo = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        core = (int((topo / "physical_package_id").read_text()), int((topo / "core_id").read_text()))
        if core not in seen:
            seen.add(core)
            out.append((core, cpu))
    return [cpu for _, cpu in sorted(out)]


def _pin(cpus):
    os.sched_setaffinity(0, set(cpus))  # pid 0 = the calling thread on Linux


def _pin_all_threads(cpus):
    """Every thread of this process (BLAS threads included) onto ``cpus``: threads inherit the affinity of the thread
    that starts them, so BLAS threads started while the main thread was pinned to one core would all share it."""
    for tid in os.listdir("/proc/self/task"):
        try:
            os.sched_setaffinity(int(tid), set(cpus))
        except OSError:
            pass


def _shared(n, dtype):
    buf = mmap.mmap(-1, max(1, n * np.dtype(dtype).itemsize))  # anonymous MAP_SHARED: inherited across fork
    return np.frombuffer(buf, dtype=dtype, count=n)


def chunks(n, t):
    e = np.linspace(0, n, t + 1).round().astype(int)
    return [(a, b) for a, b in zip(e[:-1], e[1:]) if b > a]


def _lpt(items, n):
    """Longest-processing-time assignment of (cost, item) pairs to n workers: (lists of items, max / mean load)."""
    load, out = [0.0] * n, [[] for _ in range(n)]
    for cost, item in sorted(items, key=lambda t: -t[0]):
        i = int(np.argmin(load))
        load[i] += cost
        out[i].append(item)
    return out, max(load) / (sum(load) / n)


# ---------------------------------------------------------------- the worker pool
class Pool:
    """T forked workers, one per core in ``cores``, for the block logs and both scans of one query at a time."""

    def __init__(self, naive, spindle, X, cores, cost, niche_cost):
        p = next(iter(naive.layouts.values()))[1].p
        T, N = len(cores), len(X)
        self.T, self.p = T, p
        self.cov = _shared(p * p, np.float64)
        self.qw = _shared(X.shape[1], np.float32)
        self.dw, self.dn = _shared(N, np.float64), _shared(N, np.float64)
        self.ctl = _shared(2, np.int64)          # [sequence number, command]
        self.done = _shared(T, np.int64)         # sequence number each worker finished last
        self.seq = 0
        # block logs: (niche, block) tasks into one float32 buffer, niche by niche as SpindleNaive.query_vectors
        tasks, self.views, off = [], {}, 0
        for k, (_, vec) in naive.layouts.items():
            start = off
            for b, ((s, e), iu, w) in enumerate(zip(vec.runs, vec.iu, vec.w)):
                tasks.append((cost[(k, b)], (np.asarray(vec.perm)[s:e], iu, w, off, len(w))))
                off += len(w)
            self.views[k] = (start, off)
        self.out = _shared(off, np.float32)
        block, self.balance = _lpt(tasks, T)
        # whole scan: T equal row chunks; Spindle-naive scan: chunks of about N / T rows inside each niche
        whole = [[(a, b)] for a, b in chunks(N, T)]
        nt = []
        for k, (rows, _) in naive.layouts.items():
            a0 = self.views[k][0]
            m = max(1, round(T * len(rows) / N))
            nt += [(b - a, (k, a, b, a0)) for a, b in chunks(len(rows), m)]
        nchunks, _ = _lpt(nt, T)
        self.n_naive_chunks = len(nt)
        # Spindle's niches: each worker probes and scans its own niches (one private distance cache per worker)
        flat = spindle.flat
        self.flat = flat
        self.qcat = _shared(int(flat.layer_qoff[-1] + flat.layer_dim[-1]), np.float64)
        self.probe_a2 = _shared(flat.n_niche, np.float64)
        self.work_p, self.work_s = _shared(flat.n_niche, np.int64), _shared(flat.n_niche, np.int64)
        self.n_hits = _shared(flat.n_niche, np.int64)
        self.r2 = _shared(1, np.float64)
        self.hit_a2, self.hit_t = _shared(len(flat.rows), np.float64), _shared(len(flat.rows), np.int64)
        sniches, self.niche_balance = _lpt([(niche_cost[k], k) for k in range(flat.n_niche)], T)
        ctx = mp.get_context("fork")
        self.procs = []
        for i, cpu in enumerate(cores):
            pr = ctx.Process(target=self._work, args=(i, cpu, block[i], whole[i] if i < len(whole) else [],
                                                      nchunks[i], sniches[i], naive, X), daemon=True)
            pr.start()
            self.procs.append(pr)
        self._run(0)  # every worker up and spinning

    def _work(self, i, cpu, block, whole, nchunks, sniches, naive, X):
        _pin([cpu])
        flat = self.flat
        dcache = np.empty(len(flat.rad))
        with threadpool_limits(1):
            cov = self.cov.reshape(self.p, self.p)
            rows_all = np.arange(len(X), dtype=np.int64)
            views = {k: self.out[a:b] for k, (a, b) in self.views.items()}
            last = 0
            while True:
                s = int(self.ctl[0])
                if s == last:
                    continue
                cmd = int(self.ctl[1])
                if cmd == CMD_EXIT:
                    return
                if cmd == CMD_BLOCK:
                    for g, iu, w, off, n in block:
                        L = log_spd(cov[np.ix_(g, g)], eps=LOG_FLOOR)
                        self.out[off:off + n] = (L[iu] * w).astype(np.float32)
                elif cmd == CMD_WHOLE:
                    for a, b in whole:
                        naive_scan(X[a:b], self.qw, rows_all[a:b], self.dw)
                elif cmd == CMD_NAIVE:
                    for k, a, b, _ in nchunks:
                        rows = naive.layouts[k][0]
                        naive_scan(naive.X[k][a:b], views[k], rows[a:b], self.dn)
                elif cmd == CMD_PROBE:
                    for k in sniches:
                        self.probe_a2[k], self.work_p[k] = flat.probe_niche(self.qcat, k, dcache)
                elif cmd == CMD_SCAN:
                    r2 = float(self.r2[0])
                    for k in sniches:
                        a, b = flat.niche_tile0[k], flat.niche_tile0[k + 1]
                        h_a2, _, self.work_s[k] = flat.scan_niche(self.qcat, k, r2, dcache, self.hit_a2[a:b],
                                                                  self.hit_t[a:b])
                        self.n_hits[k] = len(h_a2)
                self.done[i] = s
                last = s

    def _run(self, cmd):
        self.seq += 1
        self.ctl[1] = cmd
        self.ctl[0] = self.seq                   # x86 keeps the two stores in order
        while (self.done != self.seq).any():
            pass

    def query_vectors(self, cov):
        self.cov[:] = np.asarray(as_cov(cov), dtype=np.float64).ravel()
        self._run(CMD_BLOCK)
        return {k: self.out[a:b] for k, (a, b) in self.views.items()}

    def whole(self, qw):
        self.qw[:] = qw
        self._run(CMD_WHOLE)
        return self.dw

    def naive(self):
        """Spindle-naive distances for the block-log vectors the last query_vectors left in shared memory."""
        self._run(CMD_NAIVE)
        return self.dn

    def spindle_search(self, qv, g=G):
        """Spindle.search_vectors(qv, g) with the niches on the pool: (rows, scores, A_hat, radius, work)."""
        flat = self.flat
        self.qcat[:] = flat.qcat(qv)
        self._run(CMD_PROBE)
        best = np.inf
        for v in self.probe_a2:                  # as spindle_search: the first strictly smaller value
            if v < best:
                best = float(v)
        a_hat, r, r2 = flat.radius(best, g)
        self.r2[0] = r2
        self._run(CMD_SCAN)
        t0 = flat.niche_tile0
        hits = [(self.hit_a2[t0[k]:t0[k] + self.n_hits[k]], self.hit_t[t0[k]:t0[k] + self.n_hits[k]])
                for k in range(flat.n_niche)]
        return flat.assemble(hits, a_hat, r, int(self.work_p.sum() + self.work_s.sum()))

    def close(self):
        self.seq += 1
        self.ctl[1] = CMD_EXIT
        self.ctl[0] = self.seq
        for pr in self.procs:
            pr.join()


def blockwise_query_vectors(layouts, cov):
    """SpindleNaive.query_vectors by the pool's route on one core: each block's genes taken straight from the query
    covariance (no niche-wide reordering first). Same values, bit for bit (asserted); the T = 1 baseline of the block
    logs' scaling, so that the scaling measures the cores and not the cheaper route."""
    cov = np.asarray(as_cov(cov), dtype=np.float64)
    out = {}
    for k, (_, vec) in layouts.items():
        parts = []
        for (s, e), iu, w in zip(vec.runs, vec.iu, vec.w):
            g = np.asarray(vec.perm)[s:e]
            L = log_spd(cov[np.ix_(g, g)], eps=LOG_FLOOR)
            parts.append((L[iu] * w).astype(np.float32))
        out[k] = np.concatenate(parts)
    return out


def niche_costs(flat, qvs, g=G, reps=3):
    """Single-thread seconds of every niche's probe + scan, the minimum over ``reps`` queries (the radius from the
    whole search, as in the pool)."""
    cost = np.full(flat.n_niche, np.inf)
    dcache = np.empty(len(flat.rad))
    for qv in qvs[:reps]:
        qcat = flat.qcat(qv)
        _, _, _, r, _ = flat.search(qv, g)
        for k in range(flat.n_niche):
            t0 = time.perf_counter()
            flat.probe_niche(qcat, k, dcache)
            flat.scan_niche(qcat, k, r * r, dcache)
            cost[k] = min(cost[k], time.perf_counter() - t0)
    return cost


def block_costs(layouts, covs, reps=3):
    """Single-thread seconds of every (niche, block) log, the minimum over ``reps`` query covariances."""
    cost = {}
    for k, (_, vec) in layouts.items():
        for b, ((s, e), iu, w) in enumerate(zip(vec.runs, vec.iu, vec.w)):
            g = np.asarray(vec.perm)[s:e]
            ts = []
            for cov in covs[:reps]:
                t0 = time.perf_counter()
                L = log_spd(np.asarray(cov, dtype=np.float64)[np.ix_(g, g)], eps=LOG_FLOOR)
                (L[iu] * w).astype(np.float32)
                ts.append(time.perf_counter() - t0)
            cost[(k, b)] = min(ts)
    return cost


# ---------------------------------------------------------------- the timing
def _host():
    return {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID", "")}


def dataset_parallel(key, threads, n_queries, seeds, log=print):
    """Every step of the first n_queries held-out queries of every seed on T cores, T in threads: one row per (T, seed,
    query)."""
    st = data.stamp()
    cores = physical_cores()
    main_core, cores = cores[-1], cores[:-1]  # the main process gets a core of its own
    threads = [t for t in threads if t <= len(cores)]
    log(f"{len(cores)} physical cores for workers {cores}, main on {main_core}; timing T = {threads}")
    covs = data.staged_covs(key)
    W = data.staged_whole(key)
    p = covs.arr.shape[1]
    wvec = index_cache.whole_vectorizer(p)
    out = []
    for seed in seeds:
        c = data.load_cache(key, seed, parts=("naive", "spindle", "queries"))
        naive, spindle = c.naive, c.spindle
        X = np.ascontiguousarray(W[c.train])
        N = len(X)
        rows_all = np.arange(N, dtype=np.int64)
        idx = list(range(min(n_queries, len(c.test))))
        qws = [np.array(W[c.test[j]]) for j in idx]
        qcovs = [np.array(covs[c.test[j]]["cov"]) for j in idx]
        with threadpool_limits(1):  # references and costs on one BLAS thread, as R4 and every cached result
            block_costs(naive.layouts, qcovs)  # untimed first round (cold caches, lazy setup)
            cost = block_costs(naive.layouts, qcovs, reps=5)
            ref = [naive.query_vectors(qcovs[j]) for j in idx[:N_WARM]]
            ref_dn = [naive.distances(v) for v in ref]
            ref_dw = []
            for j in idx[:N_WARM]:
                naive_scan(X, qws[j], rows_all, dw0 := np.empty(N))
                ref_dw.append(dw0)
            ref_sp = [spindle.search_vectors(v, G) for v in ref]
            niche_costs(spindle.flat, ref)  # untimed first round
            ncost = niche_costs(spindle.flat, ref)
        for T in threads:
            # pass 1: the whole-matrix logarithms on T BLAS threads (main unpinned: BLAS threads it starts inherit
            # its affinity); then BLAS back to one thread and IDLE_S for its threads to stop spinning
            _pin_all_threads(cores + [main_core])  # physical cores only
            with threadpool_limits(T, user_api="blas"):
                for j in idx[:N_WARM]:
                    wvec.vector(qcovs[j], np.float32)
                t_whole_log = []
                for j in idx:
                    t0 = time.perf_counter()
                    wvec.vector(qcovs[j], np.float32)
                    t_whole_log.append(1e3 * (time.perf_counter() - t0))
            time.sleep(IDLE_S)
            # pass 2: the other steps
            with threadpool_limits(1):
                pool = None
                if T == 1:
                    dw = np.empty(N)

                    def step_qlog_block(cov):
                        return naive.query_vectors(cov)

                    def step_whole(qw):
                        naive_scan(X, qw, rows_all, dw)
                        return np.argsort(dw, kind="stable"), dw

                    def step_naive(qv):
                        d = naive.distances(qv)
                        return np.argsort(d, kind="stable"), d
                else:
                    pool = Pool(naive, spindle, X, cores[:T], cost, ncost)
                    _pin([main_core])
                    log(f"seed {seed} T {T}: block-log balance {pool.balance:.2f}, Spindle niche balance "
                        f"{pool.niche_balance:.2f} (max / mean load), {pool.n_naive_chunks} Spindle-naive chunks")

                    def step_qlog_block(cov):
                        return pool.query_vectors(cov)

                    def step_whole(qw):
                        d = pool.whole(qw)
                        return np.argsort(d, kind="stable"), d

                    def step_naive(qv):
                        d = pool.naive()
                        return np.argsort(d, kind="stable"), d

                for j in idx[:N_WARM]:  # warm-up, and every parallel step equal to the single-thread one
                    qv = step_qlog_block(qcovs[j])
                    for k, v in ref[j].items():
                        assert np.array_equal(qv[k], v), (seed, T, j, k)
                    assert np.array_equal(step_whole(qws[j])[1], ref_dw[j]), (seed, T, j)
                    assert np.array_equal(step_naive(qv)[1], ref_dn[j]), (seed, T, j)
                    if T == 1:
                        bw = blockwise_query_vectors(naive.layouts, qcovs[j])
                        assert all(np.array_equal(bw[k], v) for k, v in ref[j].items()), (seed, j)
                    a = spindle.search_vectors(qv, G)
                    assert np.array_equal(a[0], ref_sp[j][0]) and a[2] == ref_sp[j][2], (seed, T, j)
                    if pool is not None:
                        rows, sc, a_hat, r, work = pool.spindle_search(qv, G)
                        assert np.array_equal(rows, a[0]) and np.array_equal(sc, a[1]), (seed, T, j)
                        assert (a_hat, r, work) == (a[2]["a_hat"], a[2]["radius"], a[2]["work"]), (seed, T, j)
                for j in idx:
                    t1 = time.perf_counter()
                    qv = step_qlog_block(qcovs[j])
                    t2 = time.perf_counter()
                    step_whole(qws[j])
                    t3 = time.perf_counter()
                    step_naive(qv)
                    t4 = time.perf_counter()
                    rows, _, s = spindle.search_vectors(qv, G)
                    t5 = time.perf_counter()
                    if pool is not None:
                        pool.spindle_search(qv, G)
                    t6 = time.perf_counter()
                    if pool is None:
                        blockwise_query_vectors(naive.layouts, qcovs[j])
                    t7 = time.perf_counter()
                    out.append({"key": key, "threads": T, "seed": seed, "query": j, "test_tile": int(c.test[j]),
                                "N": N, "qlog_whole_ms": t_whole_log[j], "qlog_block_ms": 1e3 * (t2 - t1),
                                "whole_ms": 1e3 * (t3 - t2), "naive_ms": 1e3 * (t4 - t3),
                                "spindle_ms": 1e3 * (t5 - t4),
                                "spindle_par_ms": 1e3 * ((t6 - t5) if pool is not None else (t5 - t4)),
                                "qlog_block_blockwise_ms": 1e3 * (t7 - t6) if pool is None else np.nan,
                                "spindle_found": len(rows), "spindle_work": s["work"],
                                **_host(), "tag": st["tag"], "describe": st["describe"]})
                if pool is not None:
                    pool.close()
                    _pin(cores + [main_core])  # before BLAS goes back to its threads at the end of this block
            m = pd.DataFrame(out[-len(idx):]).median(numeric_only=True)
            log(f"{key} seed {seed} T {T}: {len(idx)} queries; median ms: logs whole {m.qlog_whole_ms:.2f}, block "
                f"{m.qlog_block_ms:.2f}; scans whole {m.whole_ms:.2f}, naive {m.naive_ms:.2f}; spindle "
                f"{m.spindle_ms:.3f} (1 core), {m.spindle_par_ms:.3f} (niches on T cores)")
        del X, c, naive, spindle
    return pd.DataFrame(out)


def summary(tm):
    """(per (dataset, threads, seed) medians, per (dataset, threads) mean and s.d. over seeds).

    Raw: the median of every step and of the end-to-end totals (summed per query), as measured at T.
    At most T cores (columns ``*_best_ms``, the figure's): every step at its best core count up to T, per (dataset,
    seed) the running minimum over T' <= T of the step's median, and end to end = the sum of those steps
    (e2e_whole = whole-matrix logarithm + scan, e2e_naive = block logs + Spindle-naive scan, e2e_spindle = block logs +
    Spindle's niche-split search). Nobody runs a small dense eigendecomposition on more threads than help it, and with
    every core of the allocation busy, spin-synchronised BLAS threads occasionally stall for hundreds of ms when one is
    preempted on the shared node (R11: a random 1-2 of 5 seeds at T = 64); the rule removes that and treats all three
    methods alike. Speed-ups: of the means over seeds, raw and at most T cores."""
    tm = tm.assign(e2e_whole_ms=tm.qlog_whole_ms + tm.whole_ms, e2e_naive_ms=tm.qlog_block_ms + tm.naive_ms,
                   e2e_spindle_1core_ms=tm.qlog_block_ms + tm.spindle_ms,
                   e2e_spindle_ms=tm.qlog_block_ms + tm.spindle_par_ms)  # Spindle end to end: niches on T cores
    cols = STEPS + ["e2e_whole_ms", "e2e_naive_ms", "e2e_spindle_1core_ms", "e2e_spindle_ms"]
    med = tm.groupby(["key", "threads", "seed"])[cols].median().reset_index().sort_values(["key", "seed", "threads"])
    for col in STEPS:
        med[col.replace("_ms", "_best_ms")] = med.groupby(["key", "seed"])[col].cummin()
    med["e2e_whole_best_ms"] = med.qlog_whole_best_ms + med.whole_best_ms
    med["e2e_naive_best_ms"] = med.qlog_block_best_ms + med.naive_best_ms
    med["e2e_spindle_best_ms"] = med.qlog_block_best_ms + med.spindle_par_best_ms
    med = med.sort_values(["key", "threads", "seed"]).reset_index(drop=True)
    vals = [c for c in med.columns if c.endswith("_ms")]
    g = med.groupby(["key", "threads"])[vals]
    t = g.mean().join(g.std(ddof=1), rsuffix="_sd").reset_index()
    t["n_seeds"] = g.size().values
    for col in vals:
        one = t[t.threads == 1].set_index("key")[col]
        t[f"{col.removesuffix('_ms')}_scaling"] = t.key.map(one) / t[col]
    if "qlog_block_blockwise_ms" in tm:  # the block logs' scaling over the same route on one core
        bw = tm[tm.threads == 1].groupby(["key", "seed"]).qlog_block_blockwise_ms.median().groupby("key").mean()
        t["qlog_block_blockwise_1_ms"] = t.key.map(bw)
        t["qlog_block_scaling_blockwise"] = t.qlog_block_blockwise_1_ms / t.qlog_block_ms
    for sfx in ("", "_best"):
        t[f"speedup{sfx}_e2e_spindle_vs_whole"] = t[f"e2e_whole{sfx}_ms"] / t[f"e2e_spindle{sfx}_ms"]
        t[f"speedup{sfx}_e2e_spindle_vs_naive"] = t[f"e2e_naive{sfx}_ms"] / t[f"e2e_spindle{sfx}_ms"]
    t["tag"] = sorted(set(tm.tag))[0]
    return med, t
