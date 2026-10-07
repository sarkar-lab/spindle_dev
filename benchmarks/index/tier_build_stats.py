"""End-to-end index build time and peak memory per dataset (Fig 2C).

One build = everything from reading the h5ad to both search tiers being ready, exactly as the
saved index was built (benchmarks/index/build_indexes.py: tiles, per-tile covariances, train/test split,
niches + blocks with the 1,000-tile cap), then spindle_dev.tiers: the exact tier (shrunk block
logs) and the DAG (K = 32 node means). The production DAG of build_indexes is not built.

Readouts:
* exact_build_s, build_s           wall time from the h5ad to the exact tier / to both tiers (Fig 2C)
* peak_rss_exact_gb, peak_rss_gb   peak resident memory up to the exact tier / of the whole build
                                   (main process; ru_maxrss). The covariance step's joblib workers
                                   are separate processes and are not included.
* stage times (diagnostic only): tiles_covs_s, niches_blocks_s, exact_s, dag_s
* sizes: n_tiles, n_niches, n_blocks, exact_mb, dag_mb
* check: the niche labels, gene orders and blocks equal the saved index for this seed.

Seeds 0-4 hold out 10 % of the tiles (as the saved multiseed indexes); seed 73 holds out 100.

Outputs (results/tier_build_stats/): <stem>_seed<s>.csv; summary.csv + summary_by_dataset.csv (--aggregate)

  sbatch slurm_jobs/index/run_tier_build_stats.sbatch <dataset> <seed> | --aggregate
"""

import argparse
import platform
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import dag_eval_common as de
import experiment_common as ec
from build_indexes import load_and_split_data, run_index  # type: ignore
from spindle_dev import tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "tier_build_stats"


def same_layout(a, b):
    """Niche labels, per-niche gene order and blocks of two ProcessedData objects agree."""
    la, lb = np.asarray(a.labels).astype(int), np.asarray(b.labels).astype(int)
    if not np.array_equal(la, lb):
        return False
    return all(np.array_equal(np.asarray(a.perm_list[k]), np.asarray(b.perm_list[k]))
               and list(map(tuple, a.block_dict[k])) == list(map(tuple, b.block_dict[k]))
               for k in sorted(set(la.tolist())))


def run(stem, seed):
    path = paths.dataset_path(stem)
    split = dict(n_holdout=100) if seed == ec.PRODUCTION_SEED else dict(test_ratio=0.10)
    T = {}
    t_start = t0 = time.perf_counter()
    adata, genes, train_tiles, train_covs, _, _, _, _ = load_and_split_data(path, seed=seed, **split)
    T["tiles_covs_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    data, _ = run_index(train_tiles, train_covs, genes, adata, resolution=0.2, min_final_size=15,
                        max_niche_size=1000)
    T["niches_blocks_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    exact = tiers.build_tier("exact", data, train_covs)
    T["exact_s"] = time.perf_counter() - t0
    exact_build_s, peak_exact = time.perf_counter() - t_start, de.peak_rss_gb()
    t0 = time.perf_counter()
    dag = tiers.build_tier("dag", data, exact=exact)
    T["dag_s"] = time.perf_counter() - t0
    build_s = time.perf_counter() - t_start
    peak = de.peak_rss_gb()

    sizes = np.bincount(np.asarray(data.labels).astype(int))
    rec = {"dataset": stem, "seed": seed, "cells": adata.n_obs, "n_genes": len(genes),
           "n_tiles": len(data.labels), "n_niches": len(sizes), "niche_max": int(sizes.max()),
           "n_blocks": int(sum(len(v) for v in data.block_dict.values())),
           "exact_build_s": exact_build_s, "build_s": build_s, "peak_rss_exact_gb": peak_exact,
           "peak_rss_gb": peak, **T,
           "exact_mb": exact.nbytes() / de.MB, "dag_mb": dag.nbytes() / de.MB,
           "hostname": platform.node()}
    del exact, dag, train_covs
    data.spd_matrices = []
    rec["layout_matches_saved"] = same_layout(data, ec.load_index(stem, seed)["data"])
    return pd.DataFrame([rec])


def aggregate():
    s = pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob("xenium_*_seed*.csv"))], ignore_index=True)
    order = {v: i for i, v in enumerate(ec.DATASETS.values())}
    s = s.sort_values(["dataset", "seed"], key=lambda c: c.map(order) if c.name == "dataset" else c)
    s.to_csv(OUT_DIR / "summary.csv", index=False)
    multi = s[s.seed.isin(ec.MULTISEED)]
    cols = ["cells", "n_tiles", "exact_build_s", "build_s", "peak_rss_exact_gb", "peak_rss_gb",
            "tiles_covs_s", "niches_blocks_s", "exact_s", "dag_s",
            "exact_mb", "dag_mb"]
    g = multi.groupby("dataset", sort=False)[cols].agg(["mean", "std"])
    g.columns = [f"{a}_{b.replace('std', 'sd')}" for a, b in g.columns]
    g["n_seeds"] = multi.groupby("dataset", sort=False).size()
    g["layouts_match"] = multi.groupby("dataset", sort=False).layout_matches_saved.all()
    g.reset_index().to_csv(OUT_DIR / "summary_by_dataset.csv", index=False)
    pd.set_option("display.width", 250)
    print(g.round(2).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    stem = ec.resolve_stem(args.dataset)
    df = run(stem, args.seed)
    df.to_csv(OUT_DIR / f"{stem}_seed{args.seed}.csv", index=False)
    print(df.round(3).T.to_string())


if __name__ == "__main__":
    main()
