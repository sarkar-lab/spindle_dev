"""Whole-tile accuracy (Fig 2C, Supplementary Fig S5): Spindle against Spindle-naive on every held-out tile and the
typical query of the band plot; the whole-tile timing tables (Fig 2B). Readouts in retrieval.py.

nx_rows(key, seed)  The cached Spindle (beta = 0.10) is searched from the cached query vectors (Spindle.search_vectors,
    compiled) at every g in retrieval.GS; the exact d_B is the cache's D.npy (SpindleNaive.distances). Readouts per
    (query, g): NX5, Overlap(5 %, 5 % N), MX10, tiles returned, short of 10, work (node distances computed) and work /
    all nodes, A_hat, radius, nearest exact d_B.
    Declared before the run (Fig 2C): headline NX5 at g = 0.25; per (dataset, seed) the mean over held-out queries, then
    mean +/- s.d. over seeds 0-4; the text quotes the median over datasets. Expected (19 Oct, seed 73, 100 queries):
    NX5 about 11 % (8.5-14.4 %).
typical(df, D, found)  The typical query of the band plot (seed 0, g = 0.25): among the held-out queries that return
    at least 10 tiles, the one whose NX5 is the median (lower middle when even, ties by test order). Base rate stated
    with it: the share of queries that return fewer than 10 tiles.
"""

import numpy as np
import pandas as pd

import data
import retrieval as rt


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


def timing_tables(tm):
    """(timing_seeds, timing) from the whole-tile timings (results/timing/whole_tile.csv): per (dataset, seed) the
    median ms of every timed step and of the end-to-end totals (query logarithms + search, summed per query: e2e_whole
    = whole-matrix logarithm + scan, e2e_naive / e2e_spindle = block logarithms + search); per dataset mean and s.d.
    over seeds and the speed-ups of the means, search only and end to end."""
    tag = one_tag(tm)
    tm = tm.assign(e2e_whole_ms=tm.qlog_whole_ms + tm.whole_ms, e2e_naive_ms=tm.qlog_block_ms + tm.naive_ms,
                   e2e_spindle_ms=tm.qlog_block_ms + tm.spindle_ms)
    cols = ["qlog_whole_ms", "qlog_block_ms", "whole_ms", "naive_ms", "spindle_ms", "e2e_whole_ms", "e2e_naive_ms",
            "e2e_spindle_ms"]
    med = tm.groupby(["key", "seed"])[cols].median().reset_index()
    med["n_queries"] = tm.groupby(["key", "seed"]).size().values
    med["tag"] = tag
    t = mean_sd(med, ["key"], cols)
    t["speedup_naive_vs_whole"] = t.whole_ms / t.naive_ms
    t["speedup_spindle_vs_naive"] = t.naive_ms / t.spindle_ms
    t["speedup_spindle_vs_whole"] = t.whole_ms / t.spindle_ms
    t["speedup_e2e_spindle_vs_whole"] = t.e2e_whole_ms / t.e2e_spindle_ms
    t["speedup_e2e_spindle_vs_naive"] = t.e2e_naive_ms / t.e2e_spindle_ms
    t["tag"] = tag
    return med, t
