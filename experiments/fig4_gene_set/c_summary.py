"""Small figure inputs of Fig 4 from the compute outputs (login node or sbatch; seconds).

Usage: python experiments/fig4_gene_set/c_summary.py
Reads results/fig4_gene_set/{per_query,timing}/ and the cache's build.json files; writes (tracked):
  nx_seeds.csv     per (dataset, seed, method, size; size 0 = all sizes pooled): mean NX5 and Overlap over held-out
                   queries; Spindle-lite-partial also the share short of 10 and of 5 % N, median tiles returned and
                   work; imputation the share whose predicted niche is the home niche
  nx.csv           per (dataset, method, size): mean and s.d. over seeds of every nx_seeds.csv column
  nx_size.csv      per (method, size): per seed the median over datasets, then mean and s.d. over seeds (Fig 4C)
  timing_seeds.csv per (dataset, seed): median ms of every timed step over the timed queries (all sizes pooled), the
                   baselines' totals; timing_size_seeds.csv the same per size
  timing.csv       per dataset: mean and s.d. over seeds; speed-ups of the means: search only (exact search /
                   Spindle-lite-partial) and end to end (exact / (piece logs + Spindle-lite-partial))
  storage.csv      per dataset: mean, s.d. over seeds, min and max over builds of MB: whole covariances,
                   Spindle-partial, Spindle-lite-partial, padding (= Spindle), imputation (Spindle + niche means)
  band/<key>.npz   the typical |S| = 16 query (seed 0; rule in c_nx.py): its exact d_B^S to every index tile, the rows
                   Spindle-lite-partial returned; band/typical.csv: the pick and its base rate per dataset
Every file keeps the tag column of its inputs; mixed or missing tags raise.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common4 as c4  # noqa: E402
from common4 import data  # noqa: E402

OUT = c4.OUT
MB = 1e6


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


def nx_tables():
    files = sorted((OUT / "per_query").glob("*_seed*.csv"))
    if not files:
        return
    pq = pd.concat([pd.read_csv(f) for f in files])
    tag = one_tag(pq)
    pooled = pq.assign(size=0)
    pq = pd.concat([pq, pooled])
    agg = pq.groupby(["key", "seed", "method", "size"]).agg(
        NX5=("NX5", "mean"), overlap5=("overlap5", "mean"), short10=("short10", "mean"), short5=("short5", "mean"),
        n_found=("n_found", "median"), work=("work", "median"), k_hat_home=("k_hat_home", "mean"),
        seg_max=("seg_max", "median"), N=("N", "first"), n_queries=("query", "size")).reset_index()
    agg["tag"] = tag
    agg.to_csv(OUT / "nx_seeds.csv", index=False)
    cols = ["NX5", "overlap5", "short10", "short5", "n_found", "work", "k_hat_home", "seg_max", "N", "n_queries"]
    t = mean_sd(agg, ["key", "method", "size"], cols)
    t["tag"] = tag
    t.to_csv(OUT / "nx.csv", index=False)
    med = agg.groupby(["method", "size", "seed"])[["NX5", "overlap5", "short10"]].median().reset_index()
    s = mean_sd(med, ["method", "size"], ["NX5", "overlap5", "short10"])
    s["tag"] = tag
    s.to_csv(OUT / "nx_size.csv", index=False)
    print((100 * t[t["size"] == 0].pivot(index="key", columns="method", values="NX5")).round(2).to_string())
    print((100 * s.pivot(index="size", columns="method", values="NX5")).round(2).to_string())


def band():
    dst = OUT / "band"
    dst.mkdir(exist_ok=True)
    rows = []
    for f in sorted((OUT / "per_query").glob("*_band.npz")):
        key = f.name[:-len("_band.npz")]
        z = np.load(f)
        one_tag(pd.DataFrame({"tag": [str(z["tag"])]}))
        pq = pd.read_csv(OUT / "per_query" / f"{key}_seed{data.EXAMPLE_SEED}.csv")
        pq = pq[(pq.method == "lite") & (pq["size"] == c4.BAND_SIZE)].set_index("query").loc[z["query"]]
        ok = np.flatnonzero(pq.n_found.values >= 10)
        order = ok[np.argsort(pq.NX5.values[ok], kind="stable")]
        i = int(order[(len(order) - 1) // 2])
        found = z["found"][z["off"][i]:z["off"][i + 1]]
        np.savez_compressed(dst / f"{key}.npz", d=z["D"][i], found=found, query=int(z["query"][i]), tag=data.TAG)
        rows.append({"key": key, "query": int(z["query"][i]), "NX5": float(pq.NX5.values[i]),
                     "n_found": len(found), "n_queries": len(pq), "short10": float((pq.n_found < 10).mean()),
                     "tag": data.TAG})
    if rows:
        pd.DataFrame(rows).to_csv(dst / "typical.csv", index=False)


STEPS = ["exact_ms", "qlog_exact_ms", "exact_search_ms", "qlog_lite_ms", "lite_ms", "pad_fill_ms", "pad_qlog_ms",
         "pad_scan_ms", "imp_fill_ms", "imp_qlog_ms", "imp_scan_ms"]


def timing():
    files = sorted((OUT / "timing").glob("*.csv"))
    if not files:
        return
    tm = pd.concat([pd.read_csv(f) for f in files])
    tag = one_tag(tm)
    tm["padding_ms"] = tm.pad_fill_ms + tm.pad_qlog_ms + tm.pad_scan_ms
    tm["imputation_ms"] = tm.imp_fill_ms + tm.imp_qlog_ms + tm.imp_scan_ms
    cols = STEPS + ["padding_ms", "imputation_ms", "lite_found", "lite_work"]
    for by, name in ((["key", "seed"], "timing_seeds.csv"), (["key", "seed", "size"], "timing_size_seeds.csv")):
        m = tm.groupby(by)[cols].median().reset_index()
        m["n_queries"] = tm.groupby(by).size().values
        m["host"] = tm.groupby(by).host.agg(lambda h: ",".join(sorted(set(h)))).values
        m["tag"] = tag
        m.to_csv(OUT / name, index=False)
        if name == "timing_seeds.csv":
            t = mean_sd(m, ["key"], cols)
            t["speedup_lite"] = t.exact_search_ms / t.lite_ms
            t["speedup_e2e"] = t.exact_ms / (t.qlog_lite_ms + t.lite_ms)
            t["tag"] = tag
            t.to_csv(OUT / "timing.csv", index=False)
            print(t[["key", "exact_ms", "exact_search_ms", "qlog_exact_ms", "lite_ms", "qlog_lite_ms", "padding_ms",
                     "imputation_ms", "speedup_lite", "speedup_e2e"]].round(3).to_string())


def storage():
    rows = []
    for key in data.indexed_keys():
        for seed in data.SEEDS:
            f = data.cache_dir(key, seed) / "build.json"
            if not f.exists():
                continue
            b = json.loads(f.read_text())
            means = b["n_niches"] * 4 * b["p"] * (b["p"] + 1) // 2
            rows.append({"key": key, "seed": seed, "whole": b["bytes_whole"] / MB, "partial": b["bytes_partial"] / MB,
                         "plite": b["bytes_plite"] / MB, "padding": b["bytes_spindle"] / MB,
                         "imputation": (b["bytes_spindle"] + means) / MB, "tag": b["tag"]})
    df = pd.DataFrame(rows)
    tag = one_tag(df)
    cols = ["whole", "partial", "plite", "padding", "imputation"]
    t = mean_sd(df, ["key"], cols)
    g = df.groupby("key")[cols]
    t = t.merge(g.min().add_suffix("_min").reset_index(), on="key").merge(g.max().add_suffix("_max").reset_index(),
                                                                          on="key")
    t["plite_pct_partial"] = 100 * t.plite / t.partial
    t["tag"] = tag
    t.to_csv(OUT / "storage.csv", index=False)
    print(t[["key"] + cols + ["plite_pct_partial"]].round(2).to_string())


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    nx_tables()
    band()
    timing()
    storage()
