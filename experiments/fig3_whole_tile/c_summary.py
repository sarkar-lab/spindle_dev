"""Small figure inputs of Fig 3 from the compute outputs (login node or sbatch; seconds).

Usage: python experiments/fig3_whole_tile/c_summary.py
Reads results/fig3_whole_tile/{per_query,timing,memory}/ and writes (tracked):
  nx_seeds.csv     per (dataset, seed, g): mean NX5, Overlap, MX10 (queries with >= 10 tiles), share short of
                   10 and of 5 % N over held-out queries; median work, work fraction, tiles returned, radius / nearest
  nx.csv           per (dataset, g): mean and s.d. over seeds of every nx_seeds.csv column
  timing_seeds.csv per (dataset, seed): median ms of every timed step over the timed queries
  timing.csv       per dataset: mean and s.d. over seeds; speed-ups of the means (whole / Spindle, Spindle / lite)
  memory.csv       per (dataset, method, param): mean and s.d. over seeds of MB, NX5, Overlap, short5
  band/<key>.npz   the typical query of the band plot (seed 0, g = 0.25; rule in c_nx.py): its exact d_B to every
                   index tile, the rows Spindle-lite returned; band/typical.csv: the pick and its base rate per dataset
Every file keeps the tag column of its inputs; mixed or missing tags raise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common3 as c3  # noqa: E402
from common3 import data  # noqa: E402

OUT = c3.OUT


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


def nx_tables():
    files = sorted((OUT / "per_query").glob("*_seed*.csv"))
    if not files:
        return
    pq = pd.concat([pd.read_csv(f) for f in files])
    tag = one_tag(pq)
    pq["r_over_nn"] = pq.radius / pq.d_nearest
    agg = pq.groupby(["key", "seed", "g"]).agg(
        NX5=("NX5", "mean"), overlap5=("overlap5", "mean"), MX10=("MX10", "mean"),
        short10=("short10", "mean"), short5=("short5", "mean"), work=("work", "median"),
        work_frac=("work_frac", "median"), n_found=("n_found", "median"), r_over_nn=("r_over_nn", "median"),
        N=("N", "first"), n_queries=("query", "size")).reset_index()
    agg["tag"] = tag
    agg.to_csv(OUT / "nx_seeds.csv", index=False)
    cols = ["NX5", "overlap5", "MX10", "short10", "short5", "work", "work_frac", "n_found", "r_over_nn",
            "N", "n_queries"]
    t = mean_sd(agg, ["key", "g"], cols)
    t["tag"] = tag
    t.to_csv(OUT / "nx.csv", index=False)

    # band plot input: the typical query of every dataset (seed 0), rule declared in c_nx.py
    (OUT / "band").mkdir(exist_ok=True)
    picks = []
    for key in data.indexed_keys():
        f = OUT / "per_query" / f"{key}_band.npz"
        if not f.exists():
            continue
        z = np.load(f)
        q = pq[(pq.key == key) & (pq.seed == data.EXAMPLE_SEED) & (pq.g == c3.G_DEFAULT)].sort_values("query")
        ok = q[q.n_found >= 10].sort_values(["NX5", "query"], kind="stable")
        j = int(ok["query"].iloc[(len(ok) - 1) // 2])
        rows = z["found"][z["off"][j]:z["off"][j + 1]]
        np.savez_compressed(OUT / "band" / f"{key}.npz", d=z["D"][j], found=rows, test_tile=int(z["test"][j]))
        picks.append({"key": key, "seed": data.EXAMPLE_SEED, "query": j, "test_tile": int(z["test"][j]),
                      "NX5": float(q.NX5.iloc[j]), "n_found": int(q.n_found.iloc[j]), "N": int(q.N.iloc[j]),
                      "queries": len(q), "queries_ge10": len(ok), "share_short10": float(1 - len(ok) / len(q)),
                      "tag": tag})
    pd.DataFrame(picks).to_csv(OUT / "band" / "typical.csv", index=False)


def timing_tables():
    files = sorted((OUT / "timing").glob("*.csv"))
    if not files:
        return
    tm = pd.concat([pd.read_csv(f) for f in files])
    tag = one_tag(tm)
    cols = ["qlog_whole_ms", "qlog_block_ms", "whole_ms", "spindle_ms", "lite_ms"]
    med = tm.groupby(["key", "seed"])[cols].median().reset_index()
    med["n_queries"] = tm.groupby(["key", "seed"]).size().values
    med["tag"] = tag
    med.to_csv(OUT / "timing_seeds.csv", index=False)
    t = mean_sd(med, ["key"], cols)
    t["speedup_spindle_vs_whole"] = t.whole_ms / t.spindle_ms
    t["speedup_lite_vs_spindle"] = t.spindle_ms / t.lite_ms
    t["speedup_lite_vs_whole"] = t.whole_ms / t.lite_ms
    t["tag"] = tag
    t.to_csv(OUT / "timing.csv", index=False)


def memory_tables():
    files = sorted((OUT / "memory").glob("*.csv"))
    if not files:
        return
    mm = pd.concat([pd.read_csv(f) for f in files])
    tag = one_tag(mm)
    mm["param"] = mm["param"].fillna("").astype(str)
    t = mean_sd(mm, ["key", "method", "param"], ["MB", "NX5", "overlap5", "short5"])
    t["tag"] = tag
    t.to_csv(OUT / "memory.csv", index=False)


if __name__ == "__main__":
    nx_tables()
    timing_tables()
    memory_tables()
    for f in sorted(OUT.glob("*.csv")):
        print(f.relative_to(data.ROOT), len(pd.read_csv(f)), "rows")
