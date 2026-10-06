"""Collect per-dataset index statistics into one CSV (Table S1, Figs. 2, S1, S2).

Sources, per (dataset, seed):
  * results/indexes/<stem>[_seed<n>]_spindle_index.pkl
      training tiles, niche sizes, blocks per niche, DAG node count, genes,
      cells, index-only build time and index_size_mb (the DatasetIndex bundle
      size of the production DAG, alpha = 0.05 / k_min = 8; Fig 2 uses Spindle-DAG
      sizes from results/dag_size_table/ instead).
  * file sizes of the matching *_raw_covariances.pkl / *_interval_index.pkl
      (the raw file holds all tiles; compression uses dense float32 storage of the training tiles, n_train * G^2 * 4 bytes, over index_size_mb).
  * results/index_stats/build_run_logs/<stem>[_seed<n>]_index_build_run_log.json
      end-to-end wall time and peak RSS of the index-build job.
  * results/index_stats/log_index_stats.csv (seeds 0-4 only; an input, not an output)
      low-density tiles dropped, total/held-out tile counts, and niche/block
      counts, parsed from the index-build SLURM logs before they were deleted.

Seed 73 is the single-seed production build (fixed 100 held-out tiles) used by
E1; seeds 0-4 are the E5/E11 builds (10% held out).

Outputs (results/index_stats/):
  index_stats.csv          one row per (dataset, seed)
  index_stats_summary.csv  one row per dataset, mean/sd over seeds 0-4

Loads every index pickle (up to ~330 MB each), so run it through SLURM:
  sbatch slurm_jobs/run_collect_index_stats.sbatch
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

INDEX_DIR = PROJECT_ROOT / "results" / "indexes"
OUT_DIR = PROJECT_ROOT / "results" / "index_stats"
RUN_LOG_DIR = OUT_DIR / "build_run_logs"
LOG_STATS = OUT_DIR / "log_index_stats.csv"

DATASETS = {
    "skin_melanoma": "xenium_human_skin_melanoma",
    "kidney_nondiseased": "xenium_human_kidney_nondiseased",
    "breast_cancer": "xenium_human_breast_cancer",
    "lung_cancer": "xenium_human_lung_cancer",
    "pancreatic_cancer": "xenium_human_pancreatic_cancer",
    "lymph_node": "xenium_human_lymph_node",
    "lymph_node_5k": "xenium_human_lymph_node_5k",
    "brain_cancer": "xenium_human_brain_cancer",
}
MULTISEED = [0, 1, 2, 3, 4]
PRODUCTION_SEED = 73

MB = 1024 ** 2

def file_mb(path):
    return path.stat().st_size / MB if path.exists() else np.nan


def index_row(key, stem, seed):
    tag = stem if seed == PRODUCTION_SEED else f"{stem}_seed{seed}"
    index_path = INDEX_DIR / f"{tag}_spindle_index.pkl"
    with open(index_path, "rb") as fh:
        bundle = pickle.load(fh)
    data = bundle["data"]
    dag_dict = bundle["dag_dict"]

    niches = sorted(dag_dict)
    labels = np.asarray(data.labels)
    niche_sizes = [int((labels == n).sum()) for n in niches]
    blocks_per_niche = [len(dag_dict[n].block_runs) for n in niches]
    nodes_per_niche = [len(dag_dict[n].nodes) for n in niches]
    # Sub-matrices stored without clustering = sum over niches of tiles x blocks;
    # the DAG keeps one centroid per epsilon-cover cluster instead.
    n_submatrices = sum(s * b for s, b in zip(niche_sizes, blocks_per_niche))

    run_log_path = RUN_LOG_DIR / f"{tag}_index_build_run_log.json"
    run_log = json.loads(run_log_path.read_text()) if run_log_path.exists() else {}

    raw_mb = file_mb(INDEX_DIR / f"{tag}_raw_covariances.pkl")
    index_mb = float(bundle.get("index_size_mb", np.nan))
    # *_raw_covariances.pkl holds dense float32 G x G matrices for ALL tiles
    # (training + held-out; its size matches n_tiles_total * G^2 * 4 bytes).
    # The index covers training tiles only, so compression is measured
    # against the dense float32 storage of the training tiles.
    dense_train_mb = len(data.spd_ids) * int(data.num_genes) ** 2 * 4 / MB
    return {
        "dataset": key,
        "stem": stem,
        "seed": seed,
        "cells": int(data.num_spots),
        "genes": int(data.num_genes),
        "n_tiles_train": len(data.spd_ids),
        "n_niches": len(niches),
        "niche_sizes": ",".join(map(str, niche_sizes)),
        "n_blocks": sum(blocks_per_niche),
        "blocks_per_niche": ",".join(map(str, blocks_per_niche)),
        "n_dag_nodes": sum(nodes_per_niche),
        "n_submatrices": n_submatrices,
        "submatrix_compression": n_submatrices / max(sum(nodes_per_niche), 1),
        "index_build_time_s": float(bundle.get("build_time_s", np.nan)),
        "build_wall_time_s": run_log.get("wall_time_s", np.nan),
        "peak_rss_gb": run_log.get("peak_rss_gb", np.nan),
        "index_size_mb": index_mb,
        "index_file_mb": file_mb(index_path),
        "interval_index_file_mb": file_mb(INDEX_DIR / f"{tag}_interval_index.pkl"),
        "raw_cov_file_mb": raw_mb,
        "dense_train_cov_mb": dense_train_mb,
        "compression_ratio": dense_train_mb / index_mb if index_mb else np.nan,
        "run_log_n_holdout": run_log.get("n_holdout", np.nan),
        "hostname": run_log.get("hostname"),
        "cpu_model": run_log.get("cpu_model"),
    }


def summarize(df):
    multi = df[df["seed"].isin(MULTISEED)]
    numeric = [
        "cells", "genes", "n_tiles_train", "n_tiles_query", "n_tiles_dropped_low_density",
        "n_niches", "n_blocks", "n_dag_nodes", "submatrix_compression",
        "index_build_time_s", "build_wall_time_s", "peak_rss_gb",
        "index_size_mb", "raw_cov_file_mb", "dense_train_cov_mb", "compression_ratio",
    ]
    grouped = multi.groupby("dataset", sort=False)[numeric]
    out = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd"))
    out.insert(0, "n_seeds", multi.groupby("dataset", sort=False).size())
    order = [c for pair in zip([f"{c}_mean" for c in numeric], [f"{c}_sd" for c in numeric]) for c in pair]
    return out[["n_seeds"] + order].reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    logs = pd.read_csv(LOG_STATS)

    rows = []
    for key, stem in DATASETS.items():
        for seed in MULTISEED + [PRODUCTION_SEED]:
            print(f"  {key} seed={seed}", flush=True)
            rows.append(index_row(key, stem, seed))
    df = pd.DataFrame(rows)

    # Low-density removal depends only on the tiling, not the seed, so the value
    # logged for seeds 0-4 also applies to the seed-73 build.
    df = df.merge(logs, on=["stem", "seed"], how="left")
    dropped = logs.groupby("stem")["n_tiles_dropped_low_density"].agg(lambda s: s.unique().tolist())
    for stem, values in dropped.items():
        if len(values) != 1:
            raise ValueError(f"{stem}: low-density drop count varies across seeds: {values}")
        df.loc[df["stem"] == stem, "n_tiles_dropped_low_density"] = values[0]
    df["n_tiles_query"] = df["log_n_tiles_query"]
    df.loc[df["seed"] == PRODUCTION_SEED, "n_tiles_query"] = df["run_log_n_holdout"]

    # Cross-check pickle-derived counts against what was logged at build time.
    checked = df.dropna(subset=["log_n_niches"])
    mismatch = checked[
        (checked["n_niches"] != checked["log_n_niches"])
        | (checked["n_blocks"] != checked["log_n_blocks"])
        | (checked["n_tiles_train"] != checked["log_n_tiles_train"])
    ]
    if not mismatch.empty:
        print("WARNING: pickle vs. log mismatch:")
        print(mismatch[["dataset", "seed", "n_niches", "log_n_niches", "n_blocks",
                        "log_n_blocks", "n_tiles_train", "log_n_tiles_train"]])
    else:
        print(f"Pickle and log counts agree for all {len(checked)} logged builds")

    df = df.drop(columns=[c for c in df.columns if c.startswith("log_") and c != "log_file"])
    df.to_csv(args.out_dir / "index_stats.csv", index=False)
    summarize(df).to_csv(args.out_dir / "index_stats_summary.csv", index=False)
    print(f"Wrote {args.out_dir / 'index_stats.csv'} ({len(df)} rows) and index_stats_summary.csv")


if __name__ == "__main__":
    main()
