"""Aggregate multi-seed cross-modal search results (Stage C / E12).

Each seed is one ``benchmarks/cross_modal_search.py --seed <n>`` run (one SLURM
job, see ``slurm_jobs/run_cross_modal_search.sbatch``), which writes
``results/cross_modal_search/seed_<n>/{x2v,v2x}_query_metrics.csv``. This
script reduces each seed/direction to its own mean over queries, then reports
the mean +/- s.d. (ddof=1) of those per-seed values across seeds in
``results/cross_modal_search/summary.csv`` -- the same convention as
``multiseed_holdout.py`` / ``multiseed_partial_panel_search.py``. Cheap; runs
on the login node once every seed's job has finished.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
RESULTS_DIR = project_root / "results" / "cross_modal_search"

METRIC_COLUMNS = [
    "recall_at_eps_0.1",
    "recall_at_eps_0.5",
    "overlap_at_eps_0.5",
    "overlap_at_eps_1.0",
]
DIRECTIONS = ["x2v", "v2x"]


def _per_seed_summary(df: pd.DataFrame) -> dict:
    record = {col: df[col].mean() for col in METRIC_COLUMNS}
    mean_sp = df["spindle_time_ms"].mean()
    mean_bf = df["bf_time_ms"].mean()
    record["mean_speedup"] = mean_bf / mean_sp if mean_sp > 0 else np.nan
    record["num_queries"] = len(df)
    for col in ("n_niches", "n_tiles_index", "n_tiles_query"):
        record[col] = int(df[col].iloc[0])
    return record


def aggregate(variant_suffix: str = "") -> pd.DataFrame:
    rows, flags = [], []
    for direction in DIRECTIONS:
        per_seed = []
        for seed_dir in sorted(RESULTS_DIR.glob("seed_*")):
            csv_path = seed_dir / f"{direction}{variant_suffix}_query_metrics.csv"
            if not csv_path.exists():
                print(f"[aggregate] Warning: missing {csv_path}, skipping.")
                continue
            per_seed.append(_per_seed_summary(pd.read_csv(csv_path)))
        if not per_seed:
            continue

        per_seed_df = pd.DataFrame(per_seed)
        row = {"direction": direction, "n_seeds": len(per_seed_df),
               "num_queries": int(per_seed_df["num_queries"].iloc[0]),
               "n_tiles_index": int(per_seed_df["n_tiles_index"].iloc[0]),
               "n_tiles_query": int(per_seed_df["n_tiles_query"].iloc[0]),
               "mean_n_niches": per_seed_df["n_niches"].mean(),
               "sd_n_niches": per_seed_df["n_niches"].std(ddof=1) if len(per_seed_df) > 1 else 0.0}
        for col in METRIC_COLUMNS + ["mean_speedup"]:
            mean_v = per_seed_df[col].mean()
            sd_v = per_seed_df[col].std(ddof=1) if len(per_seed_df) > 1 else 0.0
            row[f"mean_{col}"] = mean_v
            row[f"sd_{col}"] = sd_v
            if mean_v and not np.isnan(mean_v) and sd_v / mean_v > 0.05:
                flags.append((direction, col, sd_v / mean_v))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    out_path = RESULTS_DIR / f"summary{variant_suffix}.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(summary_df.to_string(index=False))
    if flags:
        print("\nRelative s.d. > 5%:")
        for direction, col, rel in flags:
            print(f"  {direction}: {col} rel_sd={rel:.1%}")
    else:
        print("\nNo metric exceeded 5% relative s.d.")
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Aggregate multi-seed cross-modal search results (E12).")
    parser.add_argument("--variant", choices=["all_niche", "single_niche_baseline"], default="all_niche")
    args = parser.parse_args()
    aggregate("" if args.variant == "all_niche" else f"_{args.variant}")


if __name__ == "__main__":
    main()
