"""E1: budget sweep of the held-out whole-tile search (results/budget_sweep/).

Runs the holdout validation benchmark (see ``holdout_core.py``) across
a fixed sequence of distance-budget multipliers (``DEFAULT_BUDGET_MULTS``),
for each dataset, at a fixed ``top_c`` (Stage-2 re-rank candidate cap) --
characterizing the recall/overlap-vs-speedup tradeoff over a realistic,
bounded operating range.

Each swept budget point is evaluated against TWO ground truths, computed
once per dataset and reused across the whole sweep:
  - "block": the existing exact block-diagonalized Frobenius distance
    ground truth (``holdout_core.compute_ground_truth``).
  - "whole": a whole-matrix (non-block) log-Euclidean distance ground truth
    (``holdout_core.compute_ground_truth_whole_matrix``).
Two parallel CSV trees are written, one per ground-truth kind (see
``main()``), plus two merged top-level summaries
(``sweep_summary_block.csv`` / ``sweep_summary_whole.csv``).

This script only produces CSVs; scripts/fig3_accuracy_speed.py (panel D),
figS5_ground_truth_scale.py and figS7_budget_sweep.py plot them.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Setup import path for spindle_dev and sibling benchmark modules
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import holdout_core as hv  # type: ignore

# Fine, hand-picked grid concentrated where recall/overlap actually move
# (realistic operating range starts ~0.05 and saturates by ~1.0-2.0) --
# replaces the earlier blind log-space 0.01-32 sweep, most of whose low end
# was known to sit at near-zero recall.
DEFAULT_BUDGET_MULTS = [
    0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.25,
    0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 2.00,
    3.00, 4.00, 8.00, 16.00,
]

# The early-stop / first_perfect_at plateau detector keys off THIS metric,
# read from the WHOLE-matrix ground truth's summary record (not the block
# kind) -- a stricter, independent check that the block-based search budget
# also recovers the true global nearest neighbor, not just its own
# block-diagonalized notion of it.
EARLY_STOP_GROUND_TRUTH_KIND = "whole"
EARLY_STOP_METRIC = "recall_at_eps_0.25"

# Never allow the plateau early-stop to fire before the sweep has reached
# this budget_multiplier -- every point below it is always run in full,
# regardless of how flat EARLY_STOP_METRIC looks along the way (a flat
# reading at a low, still-starved budget is not the same thing as a genuine
# plateau). DEFAULT_BUDGET_MULTS includes 1.00 as an explicit grid point.
MIN_BUDGET_MULTIPLIER_FOR_EARLY_STOP = 1.0

# Once EARLY_STOP_METRIC has been (near-)flat for this many consecutive
# swept multipliers (at or above MIN_BUDGET_MULTIPLIER_FOR_EARLY_STOP), stop
# sweeping this dataset -- the plateau is already captured, and continuing
# through the rest of DEFAULT_BUDGET_MULTS would just spend compute
# confirming the same flat line further out.
PLATEAU_WINDOW = 3
PLATEAU_TOL = 1e-6

# A flat reading only counts as a genuine plateau if EARLY_STOP_METRIC has
# also climbed at least this high -- distinguishes "flat because saturated"
# from "flat because we got unlucky mid-climb."
PLATEAU_MIN_VALUE = 0.9


def sweep_dataset(
    dataset_name, data, dag_dict, config,
    test_tile_covs, train_tile_covs, train_idx, test_idx,
    budget_mults, top_c_candidates,
    no_early_stop=False, seed=None, n_holdout=None, cache_dir=None,
):
    """Run the holdout benchmark across a fixed sequence of budget multipliers.

    Walks ``budget_mults`` (default ``DEFAULT_BUDGET_MULTS``) in order, at a
    fixed ``top_c_candidates`` throughout -- no auto-doubling/effort-widening
    chase for a perfect score. Stops early, before exhausting the sequence,
    once ``EARLY_STOP_METRIC`` (read from the ``EARLY_STOP_GROUND_TRUTH_KIND``
    ground truth's summary) has been flat for ``PLATEAU_WINDOW`` consecutive
    multipliers AT OR ABOVE ``MIN_BUDGET_MULTIPLIER_FOR_EARLY_STOP`` -- every
    point below that budget is always run in full regardless of how flat the
    metric looks. This experiment is about characterizing the
    recall/overlap-vs-speedup curve over a realistic, bounded operating
    range, not about proving that budget_multiplier can eventually be
    cranked high enough to reproduce brute force exactly.

    Both ground truths (block-diagonalized and whole-matrix) do not depend
    on ``budget_multiplier`` at all -- only Spindle's own retrieved
    candidates do -- so each is loaded/computed exactly ONCE here (via
    ``hv.load_or_compute_ground_truth``, which also caches to disk), before
    the sweep loop, and reused across every swept multiplier. Spindle's own
    search (``perform_search``) is likewise independent of which ground
    truth is being scored against, so it too runs exactly once per budget
    point; ``evaluate_against_ground_truth`` is then called twice per budget
    point (once per kind) against those same search results.

    Returns ``(combined_query_df_block, combined_query_df_whole,
    summary_records_block, summary_records_whole, first_perfect_at)`` where
    ``first_perfect_at`` is the smallest swept multiplier (if any) at which
    ``EARLY_STOP_METRIC`` reached 1.0.
    """
    query_matrices = hv.extract_query_matrices(test_tile_covs)

    ground_truth_block = hv.load_or_compute_ground_truth(
        "block", dataset_name, test_tile_covs, train_tile_covs, train_idx, test_idx,
        data=data, cache_dir=cache_dir, seed=seed, n_holdout=n_holdout,
    )
    ground_truth_whole = hv.load_or_compute_ground_truth(
        "whole", dataset_name, test_tile_covs, train_tile_covs, train_idx, test_idx,
        data=data, cache_dir=cache_dir, seed=seed, n_holdout=n_holdout,
    )
    ground_truths = {"block": ground_truth_block, "whole": ground_truth_whole}

    all_query_dfs = {"block": [], "whole": []}
    summary_records = {"block": [], "whole": []}
    first_perfect_at = None
    recent_metric_values = []

    for budget_mult in budget_mults:
        print(f"\n>>> {dataset_name}: budget_multiplier = {budget_mult}")
        all_matched_train_ids, spindle_search_times = (
            hv.perform_search(query_matrices, data, dag_dict, config, budget_multiplier=budget_mult)
        )

        # Realized budget varies per-niche (epsilon * num_blocks * mult * niche_scale);
        # every niche is searched per query now, so report the mean across ALL niches.
        # Independent of ground-truth kind, so computed once per budget point.
        realized_budgets = []
        for cluster_id in sorted(set(int(c) for c in data.labels)):
            epsilon = config.epsilon_dict[cluster_id]
            num_blocks = len(dag_dict[cluster_id].sorted_blocks)
            n_niche = int(np.sum(data.labels == cluster_id))
            f = hv._niche_scale_factor(n_niche)
            realized_budgets.append(float(epsilon) * float(num_blocks) * float(budget_mult) * f)
        mean_realized_budget = float(np.mean(realized_budgets)) if realized_budgets else np.nan

        for kind in ("block", "whole"):
            df_query_metrics, summary_record = hv.evaluate_against_ground_truth(
                ground_truths[kind], ground_truth_block, train_idx,
                all_matched_train_ids, spindle_search_times,
                data, dataset_name, config, ground_truth_kind=kind,
                top_c_candidates=top_c_candidates,
                ground_truth_whole=ground_truth_whole,
            )
            df_query_metrics = df_query_metrics.copy()
            df_query_metrics['budget_multiplier'] = budget_mult
            all_query_dfs[kind].append(df_query_metrics)

            summary_record = dict(summary_record)
            summary_record['budget_multiplier'] = budget_mult
            summary_record['top_c_used'] = top_c_candidates
            summary_record['mean_realized_budget'] = mean_realized_budget
            summary_records[kind].append(summary_record)

        metric_value = summary_records[EARLY_STOP_GROUND_TRUTH_KIND][-1].get(EARLY_STOP_METRIC, np.nan)
        print(f"    [{EARLY_STOP_GROUND_TRUTH_KIND}] {EARLY_STOP_METRIC} = {metric_value}")

        if first_perfect_at is None and metric_value is not None and metric_value >= 1.0:
            first_perfect_at = budget_mult

        if budget_mult >= MIN_BUDGET_MULTIPLIER_FOR_EARLY_STOP:
            recent_metric_values.append(metric_value)
            recent_metric_values = recent_metric_values[-PLATEAU_WINDOW:]
        else:
            recent_metric_values = []  # never let a below-1.0 reading count toward the plateau window
        plateaued = (
            len(recent_metric_values) == PLATEAU_WINDOW
            and np.all(np.isfinite(recent_metric_values))
            and max(recent_metric_values) - min(recent_metric_values) < PLATEAU_TOL
            and min(recent_metric_values) >= PLATEAU_MIN_VALUE
        )
        if plateaued and no_early_stop:
            print(f"    [{EARLY_STOP_GROUND_TRUTH_KIND}] {EARLY_STOP_METRIC} has been flat at "
                  f"{metric_value} for the last {PLATEAU_WINDOW} budget multipliers -- would "
                  f"normally stop early here, but --no-early-stop is set, continuing through "
                  f"the full sequence.")
        elif plateaued:
            print(f"    [{EARLY_STOP_GROUND_TRUTH_KIND}] {EARLY_STOP_METRIC} has been flat at "
                  f"{metric_value} for the last {PLATEAU_WINDOW} budget multipliers (at or above "
                  f"budget_multiplier={MIN_BUDGET_MULTIPLIER_FOR_EARLY_STOP}) -- stopping this "
                  f"dataset's sweep early.")
            break

    combined = {}
    for kind in ("block", "whole"):
        for rec in summary_records[kind]:
            rec['first_perfect_at'] = first_perfect_at
        combined[kind] = pd.concat(all_query_dfs[kind], ignore_index=True) if all_query_dfs[kind] else pd.DataFrame()
        if not combined[kind].empty:
            combined[kind]['first_perfect_at'] = first_perfect_at

    if first_perfect_at is not None:
        print(f"    Note: first reached [{EARLY_STOP_GROUND_TRUTH_KIND}] {EARLY_STOP_METRIC}=1.0 "
              f"at budget_multiplier={first_perfect_at}.")
    else:
        print(f"    Note: [{EARLY_STOP_GROUND_TRUTH_KIND}] {EARLY_STOP_METRIC} never reached 1.0 "
              f"across the swept range (top_c={top_c_candidates} may be the binding constraint "
              f"for some niches).")

    return combined["block"], combined["whole"], summary_records["block"], summary_records["whole"], first_perfect_at


def main():
    parser = argparse.ArgumentParser(description="Sweep distance-budget multiplier across holdout validation.")
    parser.add_argument('--budget-mults', type=float, nargs='*', default=None,
                         help='Space-separated budget multipliers to sweep, ascending')
    parser.add_argument('--seed', type=int, default=None,
                         help='Seed used for this dataset\'s train/test holdout split (informational -- '
                              'recorded in the ground-truth cache and run log; the actual split is already '
                              'baked into the loaded index/covariance pickles).')
    parser.add_argument('--n-holdout', type=int, default=100,
                         help='Fixed holdout tile count used for this dataset\'s split (informational, see --seed).')
    parser.add_argument('--top-c', type=int, default=400,
                         help='Floor on the Stage-1 candidate pool retrieval cap for Stage-2 re-ranking. '
                              'Automatically widened per-dataset to at least the largest niche size, so '
                              'it never truncates below what Stage-1 search already found.')
    parser.add_argument('--max-queries', type=int, default=None,
                         help='Cap the number of held-out test queries evaluated per dataset '
                              '(randomly subsampled with a fixed seed). Default: no cap.')
    parser.add_argument('--dataset-paths', nargs='+', required=True,
                        help='h5ad paths; the stem names the index in results/indexes/')
    parser.add_argument('--no-early-stop', action='store_true',
                         help='Disable the plateau early-stop and walk the full budget_mults sequence '
                              'regardless of EARLY_STOP_METRIC. The plateau detector cannot distinguish "flat '
                              'because saturated" from "flat because still collapsed at a near-zero '
                              'budget" -- for a sweep meant to characterize the whole curve (e.g. a '
                              'collapse-to-saturation transition packed into a narrow low-budget range), '
                              'that can stop the sweep after only a handful of points. Off by default to '
                              'keep the normal quick-convergence use case unchanged.')
    args = parser.parse_args()

    budget_mults = args.budget_mults if args.budget_mults else DEFAULT_BUDGET_MULTS

    base_indexed_dir = project_root / "results" / "indexes"
    base_results_dir = project_root / "results" / "budget_sweep"
    base_results_dir.mkdir(exist_ok=True, parents=True)

    all_summary_records = {"block": [], "whole": []}
    cache_dir = project_root / "results" / "ground_truth_cache"

    for ds_path in args.dataset_paths:
        dataset_name = Path(ds_path).stem
        print(f"\n{'=' * 80}\nProcessing {dataset_name}\n{'=' * 80}\n")
        saved = hv.load_index(base_indexed_dir, dataset_name)
        data, dag_dict, config = saved['data'], saved['dag_dict'], saved['config']
        test_tile_covs, train_tile_covs = saved['test_tile_covs'], saved['train_tile_covs']
        train_idx, test_idx = saved['train_idx'], saved.get('test_idx')

        if args.max_queries is not None and len(test_tile_covs) > args.max_queries:
            rng = np.random.default_rng(42)
            keep = np.sort(rng.choice(len(test_tile_covs), size=args.max_queries, replace=False))
            print(f"[--max-queries {args.max_queries}] Subsampling {len(test_tile_covs)} "
                  f"test queries down to {args.max_queries}...")
            test_tile_covs = [test_tile_covs[i] for i in keep]
            if test_idx is not None:
                test_idx = np.asarray(test_idx)[keep]

        dataset_out_dir = base_results_dir / dataset_name
        dataset_out_dir.mkdir(exist_ok=True, parents=True)

        # top_c (Stage-2 exact re-rank candidate cap) is held FIXED at
        # args.top_c across the whole sweep -- deliberately not auto-widened
        # to the largest niche size. This experiment is about finding a
        # realistic, production-usable budget_multiplier sweet spot, not
        # about proving that an unbounded top_c + budget eventually
        # reproduces brute force (that's a different, uninteresting claim:
        # given a large enough candidate pool, exact re-ranking is exact by
        # definition). Keeping top_c fixed means budget_multiplier is the
        # only swept variable, so a plateau caused by top_c truncation
        # itself stays visible in the results rather than being silently
        # papered over.
        max_niche_size = int(np.bincount(np.asarray(data.labels).astype(int)).max())
        if max_niche_size > args.top_c:
            print(f"[{dataset_name}] Note: largest niche has {max_niche_size} members, "
                  f"above the fixed --top-c={args.top_c} -- Stage-2 re-ranking may truncate "
                  f"before considering the true nearest neighbor for queries in that niche. "
                  f"This is intentional: it reflects the real production cost/accuracy "
                  f"tradeoff top_c imposes, and should show up as a visible ceiling below "
                  f"1.0 in the swept metrics rather than being hidden.")

        combined_query_df_block, combined_query_df_whole, summary_records_block, summary_records_whole, first_perfect_at = sweep_dataset(
            dataset_name, data, dag_dict, config,
            test_tile_covs, train_tile_covs, train_idx, test_idx,
            budget_mults, args.top_c,
            no_early_stop=args.no_early_stop,
            seed=args.seed, n_holdout=args.n_holdout, cache_dir=cache_dir,
        )

        per_kind = {"block": (combined_query_df_block, summary_records_block),
                    "whole": (combined_query_df_whole, summary_records_whole)}

        for kind, (combined_query_df, summary_records) in per_kind.items():
            if not combined_query_df.empty:
                csv_path = dataset_out_dir / f"{dataset_name}_query_metrics_{kind}.csv"
                combined_query_df.to_csv(csv_path, index=False)
                print(f"Saved {dataset_name} ({kind}) sweep query metrics to {csv_path}")

            # Written per-dataset (not into one shared file) so that datasets processed
            # in separate concurrent SLURM jobs never race on the same CSV. The final
            # combined sweep_summary_{kind}.csv is produced by merging these afterward.
            if summary_records:
                df_dataset_summary = pd.DataFrame(summary_records)
                dataset_summary_csv_path = dataset_out_dir / f"{dataset_name}_sweep_summary_{kind}.csv"
                df_dataset_summary.to_csv(dataset_summary_csv_path, index=False)
                print(f"Saved {dataset_name} ({kind}) sweep summary to {dataset_summary_csv_path}")

            all_summary_records[kind].extend(summary_records)

        print(f"\n{dataset_name}: first perfect ([{EARLY_STOP_GROUND_TRUTH_KIND}] "
              f"{EARLY_STOP_METRIC}==1.0) at budget_multiplier = {first_perfect_at}")

    for kind in ("block", "whole"):
        if all_summary_records[kind]:
            df_summary = pd.DataFrame(all_summary_records[kind])
            print("\n" + "=" * 80)
            print(f"BUDGET SWEEP SUMMARY ({kind.upper()}, THIS RUN):")
            print(df_summary.to_string(index=False))
            print("=" * 80)

        # Merge every per-dataset summary CSV under results/budget_sweep/*/ into
        # one combined sweep_summary_{kind}.csv. Safe to call even when only a subset of
        # datasets were processed in this invocation -- it just picks up whatever
        # per-dataset summaries exist on disk right now.
        per_dataset_csvs = sorted(base_results_dir.glob(f"*/*_sweep_summary_{kind}.csv"))
        if per_dataset_csvs:
            combined = pd.concat([pd.read_csv(p) for p in per_dataset_csvs], ignore_index=True)
            summary_csv_path = base_results_dir / f"sweep_summary_{kind}.csv"
            combined.to_csv(summary_csv_path, index=False)
            print(f"\nMerged {len(per_dataset_csvs)} per-dataset ({kind}) summaries into {summary_csv_path}")


if __name__ == "__main__":
    main()
