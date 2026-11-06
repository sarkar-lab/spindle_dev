"""Gene-set search against the exact d_B^S on every held-out tile (Fig 4): Spindle, padding and imputation, and the
summary tables. Ported from experiments/fig4_gene_set/c_nx.py and c_summary.py (in git history); workload and
baselines in gene_set.py, readouts in retrieval.py.

    import gene_set_accuracy as gsa
    df, band = gsa.run("breast_cancer", 0)          # one row per (query, size, method); band arrays (seed 0, |S| = 16)

Per (query, size): exact d_B^S of every index tile (SpindleNaivePartial.distances); Spindle from the query pieces
(SpindlePartial.query_vectors + search_vectors, compiled, g = 1); padding and imputation as ranked lists of all index
tiles (exact d_B of the completed query, stable sort). Readouts (graded by the exact d_B^S): NX5, Overlap(5 %, 5 % N);
for Spindle also tiles returned, short of 10, work, A_hat, radius; segments of S in the home niche and the largest
number in any niche.

Declared before the run (Fig 4; user 26 Oct):
- Headline: NX5 of Spindle at g = 1; per (dataset, seed, size) the mean over held-out queries, then the
  mean +/- s.d. over seeds 0-4; per size the median over datasets (with padding and imputation); per dataset
  all sizes pooled. Expected (19 Oct, seed 73, 100 queries): median NX5 about 12 % (8.4-32.8 %, LN5k highest), per
  size 9.5 / 10.6 / 13.3 / 16.5 %; a large mismatch goes to the user.
- Typical query of the band plot: breast_cancer, seed 0, |S| = 16; among the held-out queries whose Spindle list
  holds at least 10 tiles, the one whose NX5 is the median (lower middle when even, ties by test order). Base rate
  stated with it: the share of |S| = 16 queries that return fewer than 10 tiles (typical_query).
"""

import json

import numpy as np
import pandas as pd

import data
import gene_set as gs
import retrieval as rt
import spindle_dev as sd

MB = 1e6


def run(key, seed):
    """(per-query DataFrame, band dict or None). Only the held-out covariances are read (memory-mapped)."""
    st = data.stamp()
    covs = data.staged_covs(key)
    c = data.load_cache(key, seed, parts=("layout", "naive", "naive_partial", "spindle_partial", "D"))
    naive, naive_partial, spindle_partial = c.naive, c.naive_partial, c.spindle_partial
    n, p = c.D.shape[1], naive_partial.p
    c5 = rt.c5(n)
    means = data.niche_means(key, seed)
    out, band_d, band_rows, band_q = [], [], [], []
    for j, home, L, b, a, genes in gs.workload(c.layout, c.D, seed):
        cov_s, idx = sd.tile_query(covs[c.test[j]]["cov"], genes)
        d = naive_partial.distances(cov_s, idx)
        rows, _, s = spindle_partial.search_vectors(spindle_partial.query_vectors(cov_s, idx))
        pad_rows = np.argsort(naive.distances(naive.query_vectors(gs.pad(cov_s, idx, p))), kind="stable")
        k_hat = gs.predict_niche(cov_s, idx, means, naive_partial.blocks)
        imp_rows = np.argsort(naive.distances(naive.query_vectors(gs.impute(cov_s, idx, means[k_hat]))), kind="stable")
        seg = gs.n_segments(spindle_partial, idx)
        base = {"key": key, "seed": seed, "query": j, "test_tile": int(c.test[j]), "size": L, "home": home,
                "block": b, "start": a, "N": n, "c5": c5, "seg_home": seg[home], "seg_max": max(seg.values()),
                "d_nearest": float(d.min()), "tag": st["tag"], "describe": st["describe"]}
        for method, r in (("spindle_partial", rows), ("padding", pad_rows), ("imputation", imp_rows)):
            rec = {**base, "method": method, "NX5": rt.nx(d, r, c5), "overlap5": rt.overlap(d, r, c5)}
            if method == "spindle_partial":
                rec.update(n_found=len(r), short10=len(r) < 10, short5=len(r) < c5, work=s["work"],
                           a_hat=s["a_hat"], radius=s["radius"])
            if method == "imputation":
                rec.update(k_hat=k_hat, k_hat_home=k_hat == home)
            out.append(rec)
        if seed == data.EXAMPLE_SEED and L == gs.BAND_SIZE:
            band_d.append(d.astype(np.float32))
            band_rows.append(rows)
            band_q.append(j)
    band = None
    if band_q:
        band = {"D": np.stack(band_d), "found": np.concatenate(band_rows),
                "off": np.cumsum([0] + [len(r) for r in band_rows]), "query": np.asarray(band_q), "tag": st["tag"]}
    return pd.DataFrame(out), band


# ---------------------------------------------------------------- summaries (c_summary.py)
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


def nx_tables(pq):
    """(nx_seeds, nx, nx_size) from the per-query rows of every (dataset, seed):
    nx_seeds  per (dataset, seed, method, size; size 0 = all sizes pooled): mean NX5 and Overlap over held-out
              queries; Spindle also the share short of 10 and of 5 % N, median tiles returned and work; imputation
              the share whose predicted niche is the home niche
    nx        per (dataset, method, size): mean and s.d. over seeds of every nx_seeds column
    nx_size   per (method, size): per seed the median over datasets, then mean and s.d. over seeds"""
    tag = one_tag(pq)
    pq = pd.concat([pq, pq.assign(size=0)])
    agg = pq.groupby(["key", "seed", "method", "size"]).agg(
        NX5=("NX5", "mean"), overlap5=("overlap5", "mean"), short10=("short10", "mean"), short5=("short5", "mean"),
        n_found=("n_found", "median"), work=("work", "median"), k_hat_home=("k_hat_home", "mean"),
        seg_max=("seg_max", "median"), N=("N", "first"), n_queries=("query", "size")).reset_index()
    agg["tag"] = tag
    cols = ["NX5", "overlap5", "short10", "short5", "n_found", "work", "k_hat_home", "seg_max", "N", "n_queries"]
    t = mean_sd(agg, ["key", "method", "size"], cols)
    t["tag"] = tag
    med = agg.groupby(["method", "size", "seed"])[["NX5", "overlap5", "short10"]].median().reset_index()
    s = mean_sd(med, ["method", "size"], ["NX5", "overlap5", "short10"])
    s["tag"] = tag
    return agg, t, s


def typical_query(band, pq0):
    """The typical |S| = BAND_SIZE query of one dataset (seed 0; rule in the module docstring): (npz dict, row).
    band: the dict run() returned (or its saved npz); pq0: the dataset's seed-0 per-query rows."""
    pq = pq0[(pq0.method == "spindle_partial") & (pq0["size"] == gs.BAND_SIZE)].set_index("query").loc[band["query"]]
    ok = np.flatnonzero(pq.n_found.values >= 10)
    order = ok[np.argsort(pq.NX5.values[ok], kind="stable")]
    i = int(order[(len(order) - 1) // 2])
    found = band["found"][band["off"][i]:band["off"][i + 1]]
    z = {"d": band["D"][i], "found": found, "query": int(band["query"][i]), "tag": data.TAG}
    row = {"query": int(band["query"][i]), "NX5": float(pq.NX5.values[i]), "n_found": len(found),
           "n_queries": len(pq), "short10": float((pq.n_found < 10).mean()), "tag": data.TAG}
    return z, row


def storage_table():
    """Per dataset: mean, s.d. over seeds, min and max over builds of MB: whole covariances, Spindle-naive (gene sets),
    Spindle (gene sets), padding (= Spindle-naive whole-tile block logs), imputation (those + niche means)."""
    rows = []
    for key in data.indexed_keys():
        for seed in data.SEEDS:
            b = json.loads((data.cache_dir(key, seed) / "build.json").read_text())
            means = b["n_niches"] * 4 * b["p"] * (b["p"] + 1) // 2
            rows.append({"key": key, "seed": seed, "whole": b["bytes_whole"] / MB,
                         "naive_partial": b["bytes_naive_partial"] / MB,
                         "spindle_partial": b["bytes_spindle_partial"] / MB, "padding": b["bytes_naive"] / MB,
                         "imputation": (b["bytes_naive"] + means) / MB, "tag": b["tag"]})
    df = pd.DataFrame(rows)
    tag = one_tag(df)
    cols = ["whole", "naive_partial", "spindle_partial", "padding", "imputation"]
    t = mean_sd(df, ["key"], cols)
    g = df.groupby("key")[cols]
    t = t.merge(g.min().add_suffix("_min").reset_index(), on="key").merge(g.max().add_suffix("_max").reset_index(),
                                                                          on="key")
    t["spindle_partial_pct_naive_partial"] = 100 * t.spindle_partial / t.naive_partial
    t["tag"] = tag
    return t


STEPS = ["naive_partial_ms", "qlog_naive_partial_ms", "naive_partial_search_ms", "qlog_spindle_partial_ms",
         "spindle_partial_ms", "pad_fill_ms", "pad_qlog_ms", "pad_scan_ms", "imp_fill_ms", "imp_qlog_ms", "imp_scan_ms"]


def timing_tables(tm):
    """(timing_seeds, timing_size_seeds, timing) from the gene-set timing rows (one per seed, query, size):
    median ms of every step per (dataset, seed) (and per size), the baselines' totals; per dataset mean and s.d. over
    seeds and the speed-ups of the means: search only (Spindle-naive search / Spindle) and end to end
    (Spindle-naive / (piece logs + Spindle))."""
    tag = one_tag(tm)
    tm = tm.assign(padding_ms=tm.pad_fill_ms + tm.pad_qlog_ms + tm.pad_scan_ms,
                   imputation_ms=tm.imp_fill_ms + tm.imp_qlog_ms + tm.imp_scan_ms,
                   e2e_spindle_partial_ms=tm.qlog_spindle_partial_ms + tm.spindle_partial_ms)
    cols = STEPS + ["e2e_spindle_partial_ms", "padding_ms", "imputation_ms", "spindle_partial_found",
                    "spindle_partial_work"]
    out = []
    for by in (["key", "seed"], ["key", "seed", "size"]):
        m = tm.groupby(by)[cols].median().reset_index()
        m["n_queries"] = tm.groupby(by).size().values
        m["host"] = tm.groupby(by).host.agg(lambda h: ",".join(sorted(set(h)))).values
        m["tag"] = tag
        out.append(m)
    t = mean_sd(out[0], ["key"], cols)
    t["speedup_spindle_partial"] = t.naive_partial_search_ms / t.spindle_partial_ms
    t["speedup_e2e"] = t.naive_partial_ms / (t.qlog_spindle_partial_ms + t.spindle_partial_ms)
    t["speedup_e2e_median"] = t.naive_partial_ms / t.e2e_spindle_partial_ms  # totals summed per query
    t["tag"] = tag
    return out[0], out[1], t
