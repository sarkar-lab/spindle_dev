"""Multi-seed holdout-validation reruns for error bars (Stage A / E5).

Reruns the ``index_datasets.py`` -> ``holdout_validation.py`` pipeline at a
fixed, already-confirmed production configuration (``budget_multiplier=1.0``,
``--train-test-ratio 0.10``) across several independent train/test holdout
seeds, so that the headline recall/speedup numbers can be reported as
mean +/- s.d. across seeds rather than a single point estimate.

Two modes:

- Unit mode (``--dataset-path``, ``--seed``, optional ``--train-test-ratio``):
  does the work for ONE dataset x seed combination. Meant to be invoked once
  per SLURM job (see ``slurm_jobs/run_multiseed_holdout.sbatch``). Both
  ``index_datasets.py`` and ``holdout_validation.py`` derive their internal
  ``dataset_name`` from ``Path(dataset_path).stem`` -- so this script points
  them at a seed-suffixed *symlink* to the real dataset file, which gives
  each seed its own index pickle, covariance pickle, run log, and
  ground-truth cache entry, with zero changes to either script.

- Aggregate mode (``--aggregate``): reads every unit's
  ``results/multiseed_holdout/<dataset>/seed_<n>/query_metrics.csv``, computes
  each seed's own per-dataset summary, then the mean +/- s.d. of those
  per-seed summaries across seeds, and writes
  ``results/multiseed_holdout/summary.csv``. Cheap -- meant to run on the
  login node once every unit job has finished.
"""

import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from run_logging import RunLogger  # type: ignore

SEEDS = [0, 1, 2, 3, 4]
BUDGET_MULT = 1.0
DEFAULT_TRAIN_TEST_RATIO = 0.10

METRIC_COLUMNS = [
    "recall_at_eps_0.1",
    "recall_at_eps_0.5",
    "overlap_at_eps_0.5",
    "overlap_at_eps_1.0",
]

MULTISEED_OUT_DIR = project_root / "results" / "multiseed_holdout"
SYMLINK_DIR = MULTISEED_OUT_DIR / "_dataset_symlinks"
RUN_LOG_DIR = project_root / "results" / "run_logs"
HOLDOUT_VALIDATION_DIR = project_root / "results" / "holdout_validation"


def make_seed_symlink(dataset_path: Path, seed: int) -> Path:
    """Return a symlink named ``{stem}_seed{n}.h5ad`` pointing at ``dataset_path``.

    Both ``index_datasets.py`` and ``holdout_validation.py`` derive
    ``dataset_name`` from the path stem when given ``--dataset-paths`` --
    a seed-suffixed symlink name is the only change needed to give each
    seed its own index/covariance/run-log/ground-truth-cache artifacts.
    """
    SYMLINK_DIR.mkdir(parents=True, exist_ok=True)
    link_path = SYMLINK_DIR / f"{dataset_path.stem}_seed{seed}.h5ad"
    if link_path.is_symlink() or link_path.exists():
        if link_path.resolve() != dataset_path.resolve():
            link_path.unlink()
            link_path.symlink_to(dataset_path)
    else:
        link_path.symlink_to(dataset_path)
    return link_path


def run_unit(dataset_path: Path, seed: int, train_test_ratio: float) -> None:
    stem = dataset_path.stem
    seeded_path = make_seed_symlink(dataset_path, seed)
    dataset_name = seeded_path.stem  # "{stem}_seed{n}"

    with RunLogger(dataset_name=dataset_name, stage="multiseed_holdout_unit",
                    out_dir=RUN_LOG_DIR, seed=seed, train_test_ratio=train_test_ratio):
        print(f"\n{'=' * 80}\n[multiseed_holdout] {stem} seed={seed}: building index\n{'=' * 80}")
        subprocess.run(
            [sys.executable, str(current_dir / "index_datasets.py"),
             "--dataset-paths", str(seeded_path),
             "--seed", str(seed),
             "--train-test-ratio", str(train_test_ratio)],
            check=True, cwd=str(project_root),
        )

        print(f"\n{'=' * 80}\n[multiseed_holdout] {stem} seed={seed}: holdout validation "
              f"(budget_mult={BUDGET_MULT})\n{'=' * 80}")
        subprocess.run(
            [sys.executable, str(current_dir / "holdout_validation.py"),
             "--dataset-paths", str(seeded_path),
             "--budget-mult", str(BUDGET_MULT),
             "--seed", str(seed),
             "--train-test-ratio", str(train_test_ratio)],
            check=True, cwd=str(project_root),
        )

        src_csv = HOLDOUT_VALIDATION_DIR / dataset_name / f"{dataset_name}_query_metrics.csv"
        if not src_csv.exists():
            raise FileNotFoundError(f"Expected query metrics CSV not found: {src_csv}")

        dest_dir = MULTISEED_OUT_DIR / stem / f"seed_{seed}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_csv = dest_dir / "query_metrics.csv"
        dest_csv.write_bytes(src_csv.read_bytes())
        print(f"[multiseed_holdout] Copied {src_csv} -> {dest_csv}")


def _per_seed_summary(df: pd.DataFrame) -> dict:
    """Reduce one seed's per-query metrics CSV to the same summary fields
    ``evaluate_against_ground_truth`` reports (mean over queries, and
    mean_speedup = mean(bf_time_ms) / mean(spindle_time_ms))."""
    record = {col: df[col].mean() for col in METRIC_COLUMNS}
    mean_sp = df["spindle_time_ms"].mean()
    mean_bf = df["bf_time_ms"].mean()
    record["mean_speedup"] = mean_bf / mean_sp if mean_sp > 0 else np.nan
    record["num_queries"] = len(df)
    return record


def aggregate() -> pd.DataFrame:
    dataset_dirs = sorted(p for p in MULTISEED_OUT_DIR.iterdir() if p.is_dir() and p.name != "_dataset_symlinks")

    rows = []
    high_variance_flags = []
    for ds_dir in dataset_dirs:
        seed_dirs = sorted(ds_dir.glob("seed_*"))
        per_seed_records = []
        for seed_dir in seed_dirs:
            csv_path = seed_dir / "query_metrics.csv"
            if not csv_path.exists():
                print(f"[aggregate] Warning: missing {csv_path}, skipping this seed.")
                continue
            df = pd.read_csv(csv_path)
            per_seed_records.append(_per_seed_summary(df))

        if not per_seed_records:
            print(f"[aggregate] Warning: no seed results found for {ds_dir.name}, skipping.")
            continue

        per_seed_df = pd.DataFrame(per_seed_records)
        row = {"dataset": ds_dir.name, "n_seeds": len(per_seed_df)}
        for col in METRIC_COLUMNS + ["mean_speedup"]:
            mean_v = per_seed_df[col].mean()
            sd_v = per_seed_df[col].std(ddof=1) if len(per_seed_df) > 1 else 0.0
            row[f"mean_{col}"] = mean_v
            row[f"sd_{col}"] = sd_v
            rel_sd = (sd_v / mean_v) if mean_v not in (0, np.nan) and not np.isnan(mean_v) else np.nan
            if rel_sd is not None and not np.isnan(rel_sd) and rel_sd > 0.05:
                high_variance_flags.append((ds_dir.name, col, rel_sd))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = MULTISEED_OUT_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved multi-seed summary to {summary_path}")
    print(summary_df.to_string(index=False))

    if high_variance_flags:
        print("\nDatasets with relative s.d. > 5% on at least one metric:")
        for name, col, rel_sd in high_variance_flags:
            print(f"  {name}: {col} rel_sd={rel_sd:.1%}")
    else:
        print("\nNo dataset exceeded 5% relative s.d. on any reported metric.")

    return summary_df


def main():
    parser = argparse.ArgumentParser(description="Multi-seed holdout-validation reruns (Stage A / E5).")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to one dataset .h5ad file (unit mode).")
    parser.add_argument("--seed", type=int, default=None, help="Holdout split seed for this unit.")
    parser.add_argument("--train-test-ratio", type=float, default=DEFAULT_TRAIN_TEST_RATIO,
                         help=f"Holdout fraction (default {DEFAULT_TRAIN_TEST_RATIO}).")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate mode: compute summary.csv across seeds.")
    args = parser.parse_args()

    if args.aggregate:
        aggregate()
        return

    if args.dataset_path is None or args.seed is None:
        parser.error("Unit mode requires --dataset-path and --seed (or pass --aggregate).")

    run_unit(Path(args.dataset_path), args.seed, args.train_test_ratio)


if __name__ == "__main__":
    main()
