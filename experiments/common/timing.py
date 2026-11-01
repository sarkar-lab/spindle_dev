"""Query timings of the compiled searches (Fig 3 whole tile, Fig 4 gene sets), from the index cache and R0's staged
inputs. Ported from experiments/fig3_whole_tile/c_timing.py and experiments/fig4_gene_set/c_timing.py (in git history).

Like for like (CLAUDE.md Benchmarks): one thread (run under OMP_NUM_THREADS=1 etc.), compiled searches only, each step
timed once per query (perf_counter); per seed 0-4 the first N_TIMED held-out tiles in test order (all when fewer),
N_WARM untimed warm-up queries per seed. The held-out covariances are copied into memory before anything is timed, so
no step pays for reading the memory-mapped staged file.

whole_tile(key) -> one row per (seed, query), ms of every step:
  whole    naive_scan over the whole-matrix vectors + argsort (stable)
  naive    SpindleNaive.distances + argsort (stable)       (as SpindleNaive.search with c = None)
  spindle  Spindle.search_vectors, g = 0.25                (returns its tiles sorted)
  Both exhaustive searches store float32 logs of every index tile and run the same C kernel (kernels.naive_scan), so
  they differ only in what they store (whole matrices or the niches' gene blocks). The query's own logs are timed
  separately: qlog_whole = the whole-matrix vector (one p x p eigendecomposition), qlog_block =
  SpindleNaive.query_vectors (the block logs in every niche; shared by Spindle-naive and Spindle).

gene_set(key) -> one row per (seed, query, size), ms of every step:
  naive_partial       SpindleNaivePartial.distances + argsort (stable); its query logs (one per niche and block of S)
                      are computed inside, so they are timed again on their own right after, on the same query
  qlog_naive_partial  gene_set.naive_partial_query_logs: those query logs, as distances computes them (user, 26 Oct)
  naive_partial_search  naive_partial - qlog_naive_partial per query: the Spindle-naive scan without its query logs
  spindle_partial     SpindlePartial.search_vectors, g = 1, from the query pieces computed beforehand
  qlog_spindle_partial  SpindlePartial.query_vectors (the query's piece logs in every niche), reported separately
  padding     fill (gene_set.pad) + SpindleNaive.query_vectors + SpindleNaive.distances + argsort (stable)
  imputation  niche prediction + fill (gene_set.impute) + SpindleNaive.query_vectors + SpindleNaive.distances + argsort
  Queries: the gene-set workload (gene_set.workload, |S| = 4 / 8 / 16 / 32) of the first N_TIMED held-out tiles.
"""

import os
import socket
import time

import numpy as np
import pandas as pd

import data
import gene_set as gs
import index_cache
import spindle_dev as sd
from spindle_dev.kernels import naive_scan

N_TIMED = 200
N_WARM = 3
WHOLE_STEPS = ["qlog_whole_ms", "qlog_block_ms", "whole_ms", "naive_ms", "spindle_ms"]
GENE_SET_STEPS = ["naive_partial_ms", "qlog_naive_partial_ms", "naive_partial_search_ms", "qlog_spindle_partial_ms",
                  "spindle_partial_ms", "pad_fill_ms", "pad_qlog_ms", "pad_scan_ms", "imp_fill_ms", "imp_qlog_ms",
                  "imp_scan_ms"]


def _host():
    return {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID", "")}


def whole_tile(key):
    st = data.stamp()
    covs = data.staged_covs(key)
    W = data.staged_whole(key)
    wvec = index_cache.whole_vectorizer(covs.arr.shape[1])
    out = []
    for seed in data.SEEDS:
        c = data.load_cache(key, seed, parts=("naive", "spindle", "queries"))
        naive, spindle = c.naive, c.spindle
        X = np.ascontiguousarray(W[c.train])
        rows_all = np.arange(len(X), dtype=np.int64)
        dw = np.empty(len(X))
        idx = list(range(min(N_TIMED, len(c.test))))
        qws = [np.array(W[c.test[j]]) for j in idx]
        qcovs = [np.array(covs[c.test[j]]["cov"]) for j in idx]
        for j in idx[:N_WARM]:
            naive_scan(X, qws[j], rows_all, dw)
            naive.distances(c.qvs[j])
            spindle.search_vectors(c.qvs[j])
        for j in idx:
            cov = qcovs[j]
            t0 = time.perf_counter()
            wvec.vector(cov, np.float32)
            t1 = time.perf_counter()
            naive.query_vectors(cov)
            t2 = time.perf_counter()
            naive_scan(X, qws[j], rows_all, dw)
            np.argsort(dw, kind="stable")
            t3 = time.perf_counter()
            np.argsort(naive.distances(c.qvs[j]), kind="stable")
            t4 = time.perf_counter()
            rows, _, s = spindle.search_vectors(c.qvs[j])
            t5 = time.perf_counter()
            out.append({"key": key, "seed": seed, "query": j, "test_tile": int(c.test[j]), "N": len(X),
                        "qlog_whole_ms": 1e3 * (t1 - t0), "qlog_block_ms": 1e3 * (t2 - t1),
                        "whole_ms": 1e3 * (t3 - t2), "naive_ms": 1e3 * (t4 - t3), "spindle_ms": 1e3 * (t5 - t4),
                        "spindle_found": len(rows), "spindle_work": s["work"], **_host(), "tag": st["tag"],
                        "describe": st["describe"]})
        m = pd.DataFrame(out[-len(idx):]).median(numeric_only=True)
        print(f"{key} seed {seed}: {len(idx)} queries; median ms whole {m.whole_ms:.2f}, naive {m.naive_ms:.2f}, "
              f"spindle {m.spindle_ms:.3f}; query logs whole {m.qlog_whole_ms:.1f}, block {m.qlog_block_ms:.1f}",
              flush=True)
        del X, c, naive, spindle
    return pd.DataFrame(out)


def gene_set(key):
    st = data.stamp()
    covs = data.staged_covs(key)
    out = []
    for seed in data.SEEDS:
        c = data.load_cache(key, seed, parts=("layout", "naive", "naive_partial", "spindle_partial", "D"), covs=covs)
        naive, naive_partial, spindle_partial = c.naive, c.naive_partial, c.spindle_partial
        p = naive_partial.p
        means = data.niche_means(key, seed)
        work = [w for w in gs.workload(c.layout, c.D, seed) if w[0] < N_TIMED]
        qs = [sd.tile_query(np.array(covs[c.test[w[0]]]["cov"]), w[5]) for w in work]
        for cov_s, idx in qs[:N_WARM]:
            naive_partial.distances(cov_s, idx)
            gs.naive_partial_query_logs(naive_partial, cov_s, idx)
            spindle_partial.search_vectors(spindle_partial.query_vectors(cov_s, idx))
            naive.distances(naive.query_vectors(gs.pad(cov_s, idx, p)))
        for (j, home, L, b, a, genes), (cov_s, idx) in zip(work, qs):
            t0 = time.perf_counter()
            np.argsort(naive_partial.distances(cov_s, idx), kind="stable")
            t1 = time.perf_counter()
            gs.naive_partial_query_logs(naive_partial, cov_s, idx)
            t1b = time.perf_counter()
            qp = spindle_partial.query_vectors(cov_s, idx)
            t2 = time.perf_counter()
            rows, _, s = spindle_partial.search_vectors(qp)
            t3 = time.perf_counter()
            full = gs.pad(cov_s, idx, p)
            t4 = time.perf_counter()
            qv = naive.query_vectors(full)
            t5 = time.perf_counter()
            np.argsort(naive.distances(qv), kind="stable")
            t6 = time.perf_counter()
            full = gs.impute(cov_s, idx, means[gs.predict_niche(cov_s, idx, means, naive_partial.blocks)])
            t7 = time.perf_counter()
            qv = naive.query_vectors(full)
            t8 = time.perf_counter()
            np.argsort(naive.distances(qv), kind="stable")
            t9 = time.perf_counter()
            out.append({"key": key, "seed": seed, "query": j, "test_tile": int(c.test[j]), "size": L,
                        "N": naive_partial.n, "naive_partial_ms": 1e3 * (t1 - t0),
                        "qlog_naive_partial_ms": 1e3 * (t1b - t1),
                        "naive_partial_search_ms": 1e3 * ((t1 - t0) - (t1b - t1)),
                        "qlog_spindle_partial_ms": 1e3 * (t2 - t1b), "spindle_partial_ms": 1e3 * (t3 - t2),
                        "pad_fill_ms": 1e3 * (t4 - t3), "pad_qlog_ms": 1e3 * (t5 - t4),
                        "pad_scan_ms": 1e3 * (t6 - t5), "imp_fill_ms": 1e3 * (t7 - t6),
                        "imp_qlog_ms": 1e3 * (t8 - t7), "imp_scan_ms": 1e3 * (t9 - t8),
                        "spindle_partial_found": len(rows), "spindle_partial_work": s["work"],
                        **_host(), "tag": st["tag"], "describe": st["describe"]})
        m = pd.DataFrame(out[-len(work):]).median(numeric_only=True)
        print(f"{key} seed {seed}: {len(work)} queries; median ms Spindle-naive {m.naive_partial_ms:.2f} (search "
              f"{m.naive_partial_search_ms:.2f}, query logs {m.qlog_naive_partial_ms:.2f}), Spindle "
              f"{m.spindle_partial_ms:.3f} (pieces {m.qlog_spindle_partial_ms:.2f}); "
              f"padding {m.pad_fill_ms + m.pad_qlog_ms + m.pad_scan_ms:.1f}, "
              f"imputation {m.imp_fill_ms + m.imp_qlog_ms + m.imp_scan_ms:.1f}", flush=True)
        del c, naive, naive_partial, spindle_partial, means
    return pd.DataFrame(out)


# ---------------------------------------------------------------- summaries (fig3 / fig4 c_summary.py)
def one_tag(df):
    tags = set(df["tag"])
    if tags != {data.TAG}:
        raise RuntimeError(f"inputs carry tags {tags}, expected {data.TAG}")
    return data.TAG


def mean_sd(df, by, cols):
    g = df.groupby(by)[cols]
    out = g.mean().join(g.std(ddof=1), rsuffix="_sd").reset_index()
    out["n_seeds"] = df.groupby(by).size().values
    return out


def whole_tile_tables(tm):
    """(per (dataset, seed) medians, per dataset mean and s.d. over seeds with the speed-ups of the means)."""
    tag = one_tag(tm)
    med = tm.groupby(["key", "seed"])[WHOLE_STEPS].median().reset_index()
    med["n_queries"] = tm.groupby(["key", "seed"]).size().values
    med["tag"] = tag
    t = mean_sd(med, ["key"], WHOLE_STEPS)
    t["speedup_naive_vs_whole"] = t.whole_ms / t.naive_ms
    t["speedup_spindle_vs_naive"] = t.naive_ms / t.spindle_ms
    t["speedup_spindle_vs_whole"] = t.whole_ms / t.spindle_ms
    t["tag"] = tag
    return med, t


def gene_set_tables(tm):
    """(per (dataset, seed) medians, per (dataset, seed, size) medians, per dataset mean and s.d. over seeds with the
    speed-ups: search only (Spindle-naive scan without its query logs / Spindle search) and end to end)."""
    tag = one_tag(tm)
    tm = tm.assign(padding_ms=tm.pad_fill_ms + tm.pad_qlog_ms + tm.pad_scan_ms,
                   imputation_ms=tm.imp_fill_ms + tm.imp_qlog_ms + tm.imp_scan_ms)
    cols = GENE_SET_STEPS + ["padding_ms", "imputation_ms", "spindle_partial_found", "spindle_partial_work"]
    res = []
    for by in (["key", "seed"], ["key", "seed", "size"]):
        m = tm.groupby(by)[cols].median().reset_index()
        m["n_queries"] = tm.groupby(by).size().values
        m["host"] = tm.groupby(by).host.agg(lambda h: ",".join(sorted(set(h)))).values
        m["tag"] = tag
        res.append(m)
    t = mean_sd(res[0], ["key"], cols)
    t["speedup_spindle_partial"] = t.naive_partial_search_ms / t.spindle_partial_ms
    t["speedup_e2e"] = t.naive_partial_ms / (t.qlog_spindle_partial_ms + t.spindle_partial_ms)
    t["tag"] = tag
    return res[0], res[1], t
