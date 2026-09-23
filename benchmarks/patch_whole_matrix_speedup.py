"""One-off patch: recompute bf_time_ms/speedup against the WHOLE-matrix
brute-force cost for an already-completed multiseed_holdout unit.

Context: benchmarks/holdout_validation.py's main() originally only computed
the block-diagonalized ground truth, so its bf_time_ms/speedup columns fell
back to the block ground truth's own (cheaper, sampled) brute-force timing
-- diverging from the documented invariant (see
evaluate_against_ground_truth's docstring / EXPERIMENT_PLAN.md) that
bf_time_ms/speedup are always measured against the whole-matrix brute-force
cost, exactly like benchmarks/budget_sweep_holdout.py already does.
holdout_validation.py itself has since been fixed to compute both ground
truths going forward -- this script only patches CSVs already produced by
the unfixed version, without re-running the (already-correct) search or
block-ground-truth work.

For one dataset x seed unit (identified the same way
benchmarks/multiseed_holdout.py's unit mode does -- a seed-suffixed symlink
under results/multiseed_holdout/_dataset_symlinks/), this: loads the
already-built index/covariance pickle, computes (or loads from cache) the
whole-matrix ground truth, and overwrites the bf_time_ms/speedup columns of
both CSV copies (results/holdout_validation/<name>/<name>_query_metrics.csv
and results/multiseed_holdout/<stem>/seed_<n>/query_metrics.csv) in place,
matched by query_idx. Also refreshes that dataset's row in
results/holdout_validation/benchmark_summary.csv if present.
"""

import argparse
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import holdout_validation as hv  # type: ignore
from run_logging import RunLogger  # type: ignore

BASE_INDEXED_DIR = project_root / "results" / "holdout_validation_indexed"
HOLDOUT_VALIDATION_DIR = project_root / "results" / "holdout_validation"
MULTISEED_OUT_DIR = project_root / "results" / "multiseed_holdout"
SYMLINK_DIR = MULTISEED_OUT_DIR / "_dataset_symlinks"
RUN_LOG_DIR = project_root / "results" / "run_logs"
CACHE_DIR = project_root / "results" / "ground_truth_cache"


def _patch_csv(csv_path: Path, per_query_whole: list) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    bf_time_ms = [round(per_query_whole[i]["bf_time_ms"], 4) for i in df["query_idx"]]
    df["bf_time_ms"] = bf_time_ms
    df["speedup"] = [
        round(bf / sp, 2) if sp > 0 else np.nan
        for bf, sp in zip(df["bf_time_ms"], df["spindle_time_ms"])
    ]
    df.to_csv(csv_path, index=False)
    print(f"[patch] Rewrote bf_time_ms/speedup in {csv_path}")
    return df


def patch_unit(dataset_path: Path, seed: int) -> None:
    stem = dataset_path.stem
    seeded_path = SYMLINK_DIR / f"{stem}_seed{seed}.h5ad"
    if not seeded_path.exists():
        raise FileNotFoundError(
            f"Expected symlink from the original run not found: {seeded_path}. "
            "This script only patches units that already completed a multiseed_holdout run."
        )
    dataset_name = seeded_path.stem

    idx_path = BASE_INDEXED_DIR / f"{dataset_name}_spindle_index.pkl"
    covs_path = BASE_INDEXED_DIR / f"{dataset_name}_raw_covariances.pkl"
    if not idx_path.exists():
        raise FileNotFoundError(f"Index pickle not found: {idx_path}")

    with RunLogger(dataset_name=dataset_name, stage="patch_whole_matrix_speedup",
                    out_dir=RUN_LOG_DIR, seed=seed):
        print(f"Loading index for {dataset_name}...")
        with open(idx_path, "rb") as f:
            saved_data = pickle.load(f)

        if "test_tile_covs" in saved_data:
            test_tile_covs = saved_data["test_tile_covs"]
            train_tile_covs = saved_data["train_tile_covs"]
            train_idx = saved_data["train_idx"]
            test_idx = saved_data.get("test_idx")
        else:
            with open(covs_path, "rb") as cf:
                covs_data = pickle.load(cf)
            test_tile_covs = covs_data["test_tile_covs"]
            train_tile_covs = covs_data["train_tile_covs"]
            train_idx = covs_data["train_idx"]
            test_idx = covs_data.get("test_idx")

        print(f"Computing/loading whole-matrix ground truth for {dataset_name}...")
        ground_truth_whole = hv.load_or_compute_ground_truth(
            "whole", dataset_name, test_tile_covs, train_tile_covs, train_idx, test_idx,
            cache_dir=CACHE_DIR, seed=seed,
        )
        per_query_whole = ground_truth_whole["per_query"]

        src_csv = HOLDOUT_VALIDATION_DIR / dataset_name / f"{dataset_name}_query_metrics.csv"
        if not src_csv.exists():
            raise FileNotFoundError(f"Expected query metrics CSV not found: {src_csv}")
        patched_df = _patch_csv(src_csv, per_query_whole)

        dest_csv = MULTISEED_OUT_DIR / stem / f"seed_{seed}" / "query_metrics.csv"
        if dest_csv.exists():
            dest_csv.write_bytes(src_csv.read_bytes())
            print(f"[patch] Synced patched CSV to {dest_csv}")

        summary_csv_path = HOLDOUT_VALIDATION_DIR / "benchmark_summary.csv"
        if summary_csv_path.exists():
            summary_df = pd.read_csv(summary_csv_path)
            if dataset_name in set(summary_df["Dataset"]):
                mean_bf = patched_df["bf_time_ms"].mean()
                mean_sp = patched_df["spindle_time_ms"].mean()
                mean_speedup = mean_bf / mean_sp if mean_sp > 0 else np.nan
                mask = summary_df["Dataset"] == dataset_name
                summary_df.loc[mask, "mean_bf_time_ms"] = round(mean_bf, 4)
                summary_df.loc[mask, "mean_speedup"] = round(mean_speedup, 2)
                summary_df.to_csv(summary_csv_path, index=False)
                print(f"[patch] Refreshed {dataset_name} row in {summary_csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Patch bf_time_ms/speedup to use whole-matrix ground truth.")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the real dataset .h5ad file.")
    parser.add_argument("--seed", type=int, required=True, help="Seed of the unit to patch.")
    args = parser.parse_args()
    patch_unit(Path(args.dataset_path), args.seed)


if __name__ == "__main__":
    main()
