"""Whole-tile accuracy (Fig 3): Spindle against Spindle-naive on every held-out tile, the typical query of the band
plot, and memory against fidelity. Ported from experiments/fig3_whole_tile/{c_nx,c_memory,c_summary}.py (in git
history); readouts in retrieval.py; the whole-matrix vectors and d_W are staged by R0 (index_cache.stage).

nx_rows(key, seed)  The cached Spindle (beta = 0.10) is searched from the cached query vectors (Spindle.search_vectors,
    compiled) at every g in retrieval.GS; the exact d_B is the cache's D.npy (SpindleNaive.distances). Readouts per
    (query, g): NX5, Overlap(5 %, 5 % N), MX10, tiles returned, short of 10, work (node distances computed) and work /
    all nodes, A_hat, radius, nearest exact d_B.
    Declared before the run (Fig 3B): headline NX5 at g = 0.25; per (dataset, seed) the mean over held-out queries, then
    mean +/- s.d. over seeds 0-4; the text quotes the median over datasets. Expected (19 Oct, seed 73, 100 queries):
    NX5 about 11 % (8.5-14.4 %).
typical(df, D, found)  The typical query of the band plot (seed 0, g = 0.25): among the held-out queries that return
    at least 10 tiles, the one whose NX5 is the median (lower middle when even, ties by test order). Base rate stated
    with it: the share of queries that return fewer than 10 tiles.
memory_rows(key, seed)  Memory against fidelity (Fig 3D: breast and brain): Spindle on block logs against compressed
    indexes over whole-matrix log vectors, each graded against its own exact distance. The competitors index the
    whole-matrix log vector of every index tile, whose exact search is the whole-matrix scan (d_W). HNSW was removed
    (user, 26 Oct): it stores every whole vector plus a graph, so it compresses nothing.
      pca_flat <D>    PCA to D in {16, 64, 256} (randomized SVD of the centred index vectors, random_state = seed),
                      exact scan of the codes; MB = codes N x D + projection + mean.
      pca256_pq <m>   PCA-256, then product quantization (m in {8, 32} sub-quantizers, 8 bits); MB = serialized PQ
                      (codes + codebooks) + projection + mean.
      spindle         Spindle as built (beta = 0.10), g = 0.25; MB = Spindle.nbytes(); graded against d_B (D.npy).
    Reference stores (not graded): naive = SpindleNaive.nbytes(), whole = float32 upper triangles of the whole-matrix
    logs of the index tiles. Readout per method: NX5 (c_eff = min(c, returned)), Overlap(5 %, 5 % N), share of queries
    returning fewer than c; mean over every held-out query. The PQ build uses the job's cores (not a readout).
"""

import gc
import os

import numpy as np
import pandas as pd

import data
import retrieval as rt

PCA_DIMS = [16, 64, 256]
PQ_M = [8, 32]
MEMORY_KEYS = ["breast_cancer", "brain_cancer"]


# ---------------------------------------------------------------- Spindle against Spindle-naive
def nx_rows(key, seed):
    """(per-query DataFrame, rows Spindle returned at g = 0.25 per query, D) of one (dataset, seed)."""
    st = data.stamp()
    c = data.load_cache(key, seed, parts=("spindle", "queries", "D"))
    spindle, D = c.spindle, c.D
    n = D.shape[1]
    c5 = rt.c5(n)
    n_nodes = spindle.n_nodes()
    out, found0 = [], []
    for j, qv in enumerate(c.qvs):
        d = D[j]
        for g in rt.GS:
            rows, _, s = spindle.search_vectors(qv, g=g)
            out.append({"key": key, "seed": seed, "query": j, "test_tile": int(c.test[j]), "g": g, "method": "spindle",
                        "N": n, "c5": c5, "n_found": len(rows), "short10": len(rows) < 10, "short5": len(rows) < c5,
                        "NX5": rt.nx(d, rows, c5), "overlap5": rt.overlap(d, rows, c5), "MX10": rt.mx10(d, rows),
                        "work": s["work"], "work_frac": s["work"] / n_nodes, "a_hat": s["a_hat"],
                        "radius": s["radius"], "d_nearest": float(d.min()), "tag": st["tag"],
                        "describe": st["describe"]})
            if g == rt.G_DEFAULT:
                found0.append(rows)
    return pd.DataFrame(out), found0, D, c.test


def typical(df, D, found, test):
    """(band record, exact d_B of the typical query to every index tile, the rows Spindle returned) of a seed-0
    per-query table (nx_rows)."""
    q = df[df.g == rt.G_DEFAULT].sort_values("query")
    ok = q[q.n_found >= 10].sort_values(["NX5", "query"], kind="stable")
    j = int(ok["query"].iloc[(len(ok) - 1) // 2])
    rec = {"key": df.key.iloc[0], "seed": int(df.seed.iloc[0]), "query": j, "test_tile": int(test[j]),
           "NX5": float(q.NX5.iloc[j]), "n_found": int(q.n_found.iloc[j]), "N": int(q.N.iloc[j]), "queries": len(q),
           "queries_ge10": len(ok), "share_short10": float(1 - len(ok) / len(q)), "tag": df.tag.iloc[0]}
    return rec, D[j].astype(np.float32), found[j]


# ---------------------------------------------------------------- memory against fidelity
class Pca:
    def __init__(self, X, D, seed):
        from sklearn.utils.extmath import randomized_svd
        self.mean = X.mean(axis=0)
        _, _, Vt = randomized_svd(X - self.mean, n_components=D, random_state=seed)
        self.W = np.ascontiguousarray(Vt.T, dtype=np.float32)

    def project(self, X, D=None):
        W = self.W if D is None else self.W[:, :D]
        return np.ascontiguousarray((np.atleast_2d(X) - self.mean) @ W, dtype=np.float32)


def grade(name, param, mb, orders, Dref, c):
    """Readout row of one method: orders[j] = the rows it returns for query j, best first. NX is NaN for a query
    that returned nothing (left out of the mean; their share is ``empty``)."""
    nxs = [rt.nx(Dref[j], o, c) for j, o in enumerate(orders)]
    return {"method": name, "param": param, "MB": mb / 1e6, "NX5": np.nanmean(nxs), "NX5_median": np.nanmedian(nxs),
            "overlap5": np.mean([rt.overlap(Dref[j], o, c) for j, o in enumerate(orders)]),
            "short5": np.mean([len(o) < c for o in orders]), "empty": np.mean([len(o) == 0 for o in orders]),
            "n_queries": len(orders)}


def memory_rows(key, seed):
    """One row per method of one (dataset, seed). Peak memory: the index tiles' whole vectors (float32) twice (PCA
    centring), about 9 GB for brain."""
    import faiss
    st = data.stamp()
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    cc = data.load_cache(key, seed, parts=("naive", "spindle", "queries", "D"))
    DW = np.load(data.cache_dir(key, seed) / "DW.npy")
    W = data.staged_whole(key)
    X = np.ascontiguousarray(W[cc.train])
    Q = np.ascontiguousarray(W[cc.test])
    del W
    gc.collect()
    n = len(X)
    c = rt.c5(n)
    rows = []

    def log(r):
        r.update(key=key, seed=seed, N=n, c5=c, tag=st["tag"], describe=st["describe"])
        rows.append(r)
        print(f"{key} seed {seed}: {r['method']} {r['param']}: {r['MB']:.2f} MB, NX5 {100 * r.get('NX5', 0):.2f} %, "
              f"overlap {r.get('overlap5', 0):.3f}, short5 {r.get('short5', 0):.3f}, empty {r.get('empty', 0):.3f}",
              flush=True)

    # Spindle as built, graded against d_B
    orders = [cc.spindle.search_vectors(qv)[0] for qv in cc.qvs]
    log({**grade("spindle", cc.spindle.beta, cc.spindle.nbytes(), orders, cc.D, c), "nodes": cc.spindle.n_nodes()})
    log({"method": "naive", "param": "", "MB": cc.naive.nbytes() / 1e6})
    log({"method": "whole", "param": "", "MB": X.nbytes / 1e6})
    del cc
    gc.collect()

    # whole-matrix competitors, graded against d_W
    faiss.omp_set_num_threads(threads)
    pca = Pca(X, min(max(PCA_DIMS), n - 1), seed)
    Z, Zq = pca.project(X), pca.project(Q)
    del X
    gc.collect()
    for D in PCA_DIMS:
        if D > Z.shape[1]:
            continue
        orders = [np.argsort(((Z[:, :D] - z[:D]) ** 2).sum(1), kind="stable")[:c] for z in Zq]
        log(grade("pca_flat", D, Z[:, :D].nbytes + pca.W[:, :D].nbytes + pca.mean.nbytes, orders, DW, c))
    for m in PQ_M:
        if Z.shape[1] % m or n < 256:
            continue
        pq = faiss.IndexPQ(Z.shape[1], m, 8)
        pq.train(Z)
        pq.add(Z)
        _, I = pq.search(Zq, c)
        log(grade("pca256_pq", m, faiss.serialize_index(pq).nbytes + pca.W.nbytes + pca.mean.nbytes,
                  [i[i >= 0] for i in I], DW, c))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- summaries (c_summary.py)
def one_tag(df):
    tags = set(df["tag"])
    if tags != {data.TAG}:
        raise RuntimeError(f"inputs carry tags {tags}, expected {data.TAG}")
    return data.TAG


def mean_sd(df, by, cols):
    g = df.groupby(by)[cols]
    m, s = g.mean(), g.std(ddof=1)
    out = m.join(s, rsuffix="_sd").reset_index()
    out["n_seeds"] = df.groupby(by).size().values
    return out


def nx_tables(pq):
    """(nx_seeds, nx): per (dataset, seed, g) means over held-out queries (MX10 over queries with >= 10 tiles; work,
    work fraction, tiles returned, radius / nearest exact d_B as medians); per (dataset, g) mean and s.d. over seeds."""
    tag = one_tag(pq)
    pq = pq.assign(r_over_nn=pq.radius / pq.d_nearest)
    agg = pq.groupby(["key", "seed", "g"]).agg(
        NX5=("NX5", "mean"), overlap5=("overlap5", "mean"), MX10=("MX10", "mean"),
        short10=("short10", "mean"), short5=("short5", "mean"), work=("work", "median"),
        work_frac=("work_frac", "median"), n_found=("n_found", "median"), r_over_nn=("r_over_nn", "median"),
        N=("N", "first"), n_queries=("query", "size")).reset_index()
    agg["tag"] = tag
    cols = ["NX5", "overlap5", "MX10", "short10", "short5", "work", "work_frac", "n_found", "r_over_nn",
            "N", "n_queries"]
    t = mean_sd(agg, ["key", "g"], cols)
    t["tag"] = tag
    return agg, t


def memory_table(mm):
    """Per (dataset, method, param): mean and s.d. over seeds of MB, NX5, Overlap, short5."""
    tag = one_tag(mm)
    mm = mm.assign(param=mm["param"].fillna("").astype(str))
    t = mean_sd(mm, ["key", "method", "param"], ["MB", "NX5", "overlap5", "short5"])
    t["tag"] = tag
    return t


def timing_tables(tm):
    """(timing_seeds, timing) from the whole-tile timings (results/timing/whole_tile.csv): per (dataset, seed) the
    median ms of every timed step; per dataset mean and s.d. over seeds and the speed-ups of the means."""
    tag = one_tag(tm)
    cols = ["qlog_whole_ms", "qlog_block_ms", "whole_ms", "naive_ms", "spindle_ms"]
    med = tm.groupby(["key", "seed"])[cols].median().reset_index()
    med["n_queries"] = tm.groupby(["key", "seed"]).size().values
    med["tag"] = tag
    t = mean_sd(med, ["key"], cols)
    t["speedup_naive_vs_whole"] = t.whole_ms / t.naive_ms
    t["speedup_spindle_vs_naive"] = t.naive_ms / t.spindle_ms
    t["speedup_spindle_vs_whole"] = t.whole_ms / t.spindle_ms
    t["tag"] = tag
    return med, t
