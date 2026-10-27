"""Small figure inputs of Fig 5 from the compute outputs (login node or sbatch; seconds).

Usage: python experiments/fig5_cross_platform/c_summary.py
Reads results/fig5_cross_platform/per_query/ (c_cross.py); writes (tracked):
  tissue_seeds.csv  per (direction, method, seed): tissue agreement r (query tumour fraction vs the mean of its top
                    10), mean absolute difference (top 10; random = all index tiles), co-located rank (hit@1/5/10,
                    median rank), N
  tissue.csv        per (direction, method): mean and s.d. over seeds 0-4 of every tissue_seeds.csv column
  tissue_seed0.csv  seed 0, spindle_global: per query, its tumour fraction and its top 10's mean (Fig 5C)
  lite.csv          per direction: Spindle-lite against spindle_global (NX5, Overlap(5 %, 5 % N), tiles returned),
                    mean over queries per seed, then mean and s.d. over seeds
  index.csv         per direction: N, query tiles, p, niches and blocks (min-max over seeds), Spindle and Spindle-lite MB
Every file keeps the tag of its inputs; mixed or missing tags raise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common5 as c5  # noqa: E402
from common5 import data  # noqa: E402

PQ = c5.OUT / "per_query"


def load(suffix):
    df = pd.concat([pd.read_csv(PQ / f"seed{s}{suffix}.csv") for s in data.SEEDS], ignore_index=True)
    if set(df["tag"]) != {data.TAG}:
        raise RuntimeError(f"inputs carry tags {set(df['tag'])}, expected {data.TAG}")
    return df


def seed_stats(df, keys):
    num = [c for c in df.columns if c not in keys + ["seed", "tag"]]
    out = df.groupby(keys)[num].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    out["n_seeds"] = df.groupby(keys)["seed"].nunique().to_numpy()
    out["tag"] = data.TAG
    return out


def main():
    pq = load("")
    rows = []
    for (d, m, s), g in pq.groupby(["direction", "method", "seed"]):
        h = g[g.matched_tile >= 0]
        rows.append({"direction": d, "method": m, "seed": s, "N": int(g.N.iloc[0]), "n_query": len(g),
                     "tissue_r": float(np.corrcoef(g.query_tumour, g.hits_tumour)[0, 1]),
                     "mae": float((g.query_tumour - g.hits_tumour).abs().mean()), "mae_random": g.random_mae.mean(),
                     "hit1": h["hit@1"].mean(), "hit5": h["hit@5"].mean(), "hit10": h["hit@10"].mean(),
                     "hit5_random": 5 / g.N.iloc[0], "median_rank": h.matched_rank.median(), "n_matched": len(h),
                     "tag": data.TAG})
    ts = pd.DataFrame(rows)
    ts.to_csv(c5.OUT / "tissue_seeds.csv", index=False)
    tissue = seed_stats(ts, ["direction", "method"])
    tissue.to_csv(c5.OUT / "tissue.csv", index=False)
    s0 = pq[(pq.seed == data.EXAMPLE_SEED) & (pq.method == "spindle_global")]
    s0[["direction", "query", "query_tumour", "hits_tumour", "tag"]].to_csv(c5.OUT / "tissue_seed0.csv", index=False)

    lt = load("_lite").groupby(["direction", "seed"])[["NX5", "overlap5", "n_found", "work"]].mean().reset_index()
    lite = seed_stats(lt, ["direction"])
    lite.to_csv(c5.OUT / "lite.csv", index=False)

    ix = load("_index")
    index = ix.groupby("direction").agg(N=("N", "first"), n_query=("n_query", "first"), p=("p", "first"),
                                        niches_min=("n_niches", "min"), niches_max=("n_niches", "max"),
                                        blocks_min=("n_blocks", "min"), blocks_max=("n_blocks", "max"),
                                        spindle_mb=("bytes_spindle", "mean"), lite_mb=("bytes_lite", "mean"))
    index[["spindle_mb", "lite_mb"]] /= 1e6
    index["tag"] = data.TAG
    index.reset_index().to_csv(c5.OUT / "index.csv", index=False)

    pd.set_option("display.width", 200)
    print(tissue[["direction", "method", "tissue_r_mean", "tissue_r_std", "mae_mean", "mae_random_mean",
                  "hit5_mean", "hit5_random_mean"]].round(3).to_string(index=False))
    print(lite.round(3).to_string(index=False))
    print(index.round(2).to_string())


if __name__ == "__main__":
    main()
