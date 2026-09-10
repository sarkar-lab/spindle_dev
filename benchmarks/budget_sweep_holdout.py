"""Budget-Sweep Holdout Benchmark for Spindle Covariance Indexing.

Runs the holdout validation benchmark (see ``holdout_validation.py``) across
an increasing sequence of distance-budget multipliers, for each dataset,
demonstrating that Spindle's DAG search is deterministic: given enough
budget it always recovers the exact nearest neighbors, at the cost of
progressively lower speedup relative to brute force.

The initial multiplier sequence is fine-grained over a modest range; if the
stop metric still hasn't hit 1.0 by the end of that sequence, the budget
multiplier keeps doubling (open-ended, up to a safety cap) until it does.
If the metric plateaus below 1.0 across several multipliers, that plateau
is diagnosed as the DFS search-effort caps (not the distance budget) being
the bottleneck, so those caps are widened (``effort_multiplier``) and the
sweep retries before concluding the dataset can't reach a perfect score.
For each dataset the sweep stops once the perfect-recovery metric is
reached, running one extra multiplier afterward to confirm the plateau,
then moving to the next dataset.

This script only produces CSVs. A separate script
(``scripts/generate_budget_sweep_figure.py``) reads those CSVs and renders
the presentable figure.
"""

import argparse
from pathlib import Path
import pickle
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

import holdout_validation as hv  # type: ignore

DEFAULT_BUDGET_MULTS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 32.0]

# Once the fixed sequence above is exhausted without hitting a perfect score,
# keep doubling the last multiplier (open-ended) until the stop metric hits
# 1.0. Capped so a dataset that can never reach a perfect score doesn't loop
# forever.
EXTENSION_GROWTH_FACTOR = 2.0
MAX_EXTENSION_STEPS = 20

# When the metric plateaus below 1.0, the bottleneck is usually the DFS
# search-effort caps (max_failed_starts / max_failed_paths / total_paths_limit
# in perform_search's SearchConfig), not the distance budget itself -- those
# caps don't scale with budget_multiplier. So on a plateau, widen them via
# effort_multiplier and retry before concluding the dataset can't reach 1.0.
EFFORT_GROWTH_FACTOR = 2.0
MAX_EFFORT_STEPS = 5


def sweep_dataset(
    dataset_name, data, dag_dict, config,
    test_tile_covs, train_tile_covs, train_idx,
    dataset_out_dir, budget_mults, stop_metric, top_c_candidates,
):
    """Run the holdout benchmark across a sequence of budget multipliers.

    Beyond the end of ``budget_mults``, the multiplier keeps doubling
    (``EXTENSION_GROWTH_FACTOR``, up to ``MAX_EXTENSION_STEPS`` extra steps)
    until ``stop_metric`` hits 1.0. Stops (after one confirmation step) once
    that happens, or once the extension budget is exhausted.
    Returns (per_query_df, per_multiplier_summary_records, first_perfect_at).
    """
    query_matrices = hv.extract_query_matrices(test_tile_covs)

    all_query_dfs = []
    summary_records = []
    first_perfect_at = None
    confirmed_plateau = False

    mults = list(budget_mults)
    extension_steps_used = 0
    effort_multiplier = 1.0
    effort_steps_used = 0
    recent_metric_values = []
    PLATEAU_WINDOW = 3
    PLATEAU_TOL = 1e-6
    i = 0
    while i < len(mults):
        budget_mult = mults[i]
        i += 1
        print(f"\n>>> {dataset_name}: budget_multiplier = {budget_mult}, effort_multiplier = {effort_multiplier}")
        predicted_clusters, all_matched_train_ids, spindle_search_times, assign_time_ms_per_query = (
            hv.perform_search(query_matrices, data, dag_dict, config, budget_multiplier=budget_mult,
                               effort_multiplier=effort_multiplier)
        )

        df_query_metrics, summary_record = hv.evaluate_brute_force_approximation(
            test_tile_covs, train_tile_covs, train_idx, predicted_clusters,
            all_matched_train_ids, spindle_search_times, assign_time_ms_per_query,
            data, dataset_out_dir, dataset_name,
            top_c_candidates=top_c_candidates,
        )

        df_query_metrics = df_query_metrics.copy()
        df_query_metrics['budget_multiplier'] = budget_mult
        all_query_dfs.append(df_query_metrics)

        # Realized budget varies per-niche (epsilon * num_blocks * mult * niche_scale);
        # report the mean actually used across the queried niches for this multiplier.
        realized_budgets = []
        for cluster_id in set(int(c) for c in predicted_clusters):
            epsilon = config.epsilon_dict[cluster_id]
            num_blocks = len(dag_dict[cluster_id].sorted_blocks)
            n_niche = int(np.sum(data.labels == cluster_id))
            f = hv._niche_scale_factor(n_niche)
            realized_budgets.append(float(epsilon) * float(num_blocks) * float(budget_mult) * f)

        summary_record = dict(summary_record)
        summary_record['budget_multiplier'] = budget_mult
        summary_record['effort_multiplier'] = effort_multiplier
        summary_record['top_c_used'] = top_c_candidates
        summary_record['mean_realized_budget'] = float(np.mean(realized_budgets)) if realized_budgets else np.nan
        summary_records.append(summary_record)

        metric_value = summary_record.get(stop_metric, np.nan)
        print(f"    {stop_metric} = {metric_value}")

        if first_perfect_at is None and metric_value is not None and metric_value >= 1.0:
            first_perfect_at = budget_mult
            confirmed_plateau = False
            continue

        if first_perfect_at is not None:
            confirmed_plateau = True
            break

        recent_metric_values.append(metric_value)
        recent_metric_values = recent_metric_values[-PLATEAU_WINDOW:]
        stuck_below_one = (
            len(recent_metric_values) == PLATEAU_WINDOW
            and max(recent_metric_values) - min(recent_metric_values) < PLATEAU_TOL
        )
        if stuck_below_one:
            if effort_steps_used < MAX_EFFORT_STEPS:
                effort_multiplier *= EFFORT_GROWTH_FACTOR
                effort_steps_used += 1
                recent_metric_values = []  # give the new effort level a fresh plateau window
                print(f"    {stop_metric} has been flat at {metric_value} for the last "
                      f"{PLATEAU_WINDOW} multipliers -- distance budget isn't the bottleneck, "
                      f"widening search effort_multiplier to {effort_multiplier} and retrying.")
                if i == len(mults):
                    # Retry at the same budget, now with more search effort.
                    mults.append(budget_mult)
            else:
                print(f"    {stop_metric} has been flat at {metric_value} even after widening "
                      f"effort_multiplier to {effort_multiplier} ({effort_steps_used} step(s)) -- "
                      f"stopping this dataset's sweep without reaching 1.0.")
                break
            continue

        if i == len(mults) and extension_steps_used < MAX_EXTENSION_STEPS:
            next_mult = mults[-1] * EXTENSION_GROWTH_FACTOR
            mults.append(next_mult)
            extension_steps_used += 1
            print(f"    {stop_metric} not yet 1.0 at budget_multiplier={budget_mult} -- "
                  f"extending sweep to {next_mult}")

    for rec in summary_records:
        rec['first_perfect_at'] = first_perfect_at

    combined_query_df = pd.concat(all_query_dfs, ignore_index=True) if all_query_dfs else pd.DataFrame()
    if not combined_query_df.empty:
        combined_query_df['first_perfect_at'] = first_perfect_at

    if first_perfect_at is not None and not confirmed_plateau:
        print(f"    Note: reached {stop_metric}=1.0 at multiplier {first_perfect_at} "
              f"but ran out of configured multipliers before a confirmation step.")
    elif first_perfect_at is None:
        print(f"    Note: {stop_metric} never reached 1.0 even after "
              f"{extension_steps_used} budget extension step(s) and "
              f"{effort_steps_used} search-effort widening step(s) "
              f"(final budget_multiplier={mults[-1]}, effort_multiplier={effort_multiplier}).")

    return combined_query_df, summary_records, first_perfect_at


def main():
    parser = argparse.ArgumentParser(description="Sweep distance-budget multiplier across holdout validation.")
    parser.add_argument('--test', action='store_true', help='Run quick test: 5 queries, short multiplier list')
    parser.add_argument('--budget-mults', type=float, nargs='*', default=None,
                         help='Space-separated budget multipliers to sweep, ascending')
    parser.add_argument('--stop-metric', type=str, default='overlap_at_50',
                         choices=['overlap_at_50', 'recall_at_1'],
                         help='Metric checked for perfect (==1.0) recovery to stop the sweep')
    parser.add_argument('--top-c', type=int, default=400,
                         help='Floor on the Stage-1 candidate pool retrieval cap for Stage-2 re-ranking. '
                              'Automatically widened per-dataset to at least the largest niche size, so '
                              'it never truncates below what Stage-1 search already found.')
    parser.add_argument('--max-queries', type=int, default=None,
                         help='Cap the number of held-out test queries evaluated per dataset '
                              '(randomly subsampled with a fixed seed). Default: no cap.')
    parser.add_argument('--dataset-paths', nargs='*', default=None, help='Paths to the datasets')
    args = parser.parse_args()

    budget_mults = args.budget_mults if args.budget_mults else DEFAULT_BUDGET_MULTS
    if args.test:
        budget_mults = [m for m in budget_mults if m <= 4.0] or budget_mults[:3]

    base_indexed_dir = project_root / "results" / "holdout_validation_indexed"
    base_results_dir = project_root / "results" / "budget_sweep_holdout"
    base_results_dir.mkdir(exist_ok=True, parents=True)

    if args.dataset_paths:
        datasets = {Path(p).stem: Path(p) for p in args.dataset_paths}
    else:
        datasets = {
            "breast_cancer": project_root / "dataset" / "xenium_human_breast_cancer.h5ad",
            "kidney_nondiseased": project_root / "dataset" / "xenium_human_kidney_nondiseased.h5ad",
            "lymph_node": project_root / "dataset" / "xenium_human_lymph_node.h5ad",
            "lung_cancer": project_root / "dataset" / "xenium_human_lung_cancer.h5ad",
            "skin_melanoma": project_root / "dataset" / "xenium_human_skin_melanoma.h5ad",
            "pancreatic_cancer": project_root / "dataset" / "xenium_human_pancreatic_cancer.h5ad",
        }

    indexed_files = []
    for ds_name in datasets.keys():
        idx_path = base_indexed_dir / f"{ds_name}_spindle_index.pkl"
        if not idx_path.exists():
            idx_path = base_indexed_dir / f"{ds_name}_indexed.pkl"
        if idx_path.exists():
            indexed_files.append(idx_path)
        else:
            print(f"Warning: Index file for {ds_name} not found.")

    all_summary_records = []

    for indexed_file in indexed_files:
        print(f"\n{'=' * 80}")
        print(f"Processing indexed file: {indexed_file.name}")
        print(f"{'=' * 80}\n")

        with open(indexed_file, 'rb') as f:
            saved_data = pickle.load(f)

        data = saved_data['data']
        dag_dict = saved_data['dag_dict']
        config = saved_data['config']
        dataset_name = saved_data['dataset_name']

        if 'test_tile_covs' in saved_data:
            test_tile_covs = saved_data['test_tile_covs']
            train_tile_covs = saved_data['train_tile_covs']
            train_idx = saved_data['train_idx']
        else:
            covs_file = base_indexed_dir / f"{dataset_name}_raw_covariances.pkl"
            with open(covs_file, 'rb') as cf:
                covs_data = pickle.load(cf)
            test_tile_covs = covs_data['test_tile_covs']
            train_tile_covs = covs_data['train_tile_covs']
            train_idx = covs_data['train_idx']

        if args.test:
            test_tile_covs = test_tile_covs[:5]
        elif args.max_queries is not None and len(test_tile_covs) > args.max_queries:
            rng = np.random.default_rng(42)
            keep = np.sort(rng.choice(len(test_tile_covs), size=args.max_queries, replace=False))
            print(f"[--max-queries {args.max_queries}] Subsampling {len(test_tile_covs)} "
                  f"test queries down to {args.max_queries}...")
            test_tile_covs = [test_tile_covs[i] for i in keep]

        dataset_out_dir = base_results_dir / dataset_name
        dataset_out_dir.mkdir(exist_ok=True, parents=True)

        # Stage-2 exact re-ranking only ever sees the first `top_c_candidates` of
        # Stage-1's *approximately*-ordered candidate pool (see
        # evaluate_brute_force_approximation). If a niche has more members than
        # that, the true nearest neighbors can be silently discarded before
        # Stage-2 runs regardless of how much budget/effort Stage-1 search is
        # given -- this is what caused the accuracy plateau observed below 1.0.
        # Widen the cap to at least the largest niche size so the sweep never
        # truncates below what Stage-1 already found, which isolates the
        # budget/effort-driven determinism claim this benchmark is meant to test.
        max_niche_size = int(np.bincount(np.asarray(data.labels).astype(int)).max())
        effective_top_c = max(args.top_c, max_niche_size)
        if effective_top_c > args.top_c:
            print(f"[{dataset_name}] Widening --top-c from {args.top_c} to {effective_top_c} "
                  f"(largest niche has {max_niche_size} members) to avoid truncating Stage-2 "
                  f"re-ranking below what Stage-1 search already retrieves.")

        combined_query_df, summary_records, first_perfect_at = sweep_dataset(
            dataset_name, data, dag_dict, config,
            test_tile_covs, train_tile_covs, train_idx,
            dataset_out_dir, budget_mults, args.stop_metric, effective_top_c,
        )

        if not combined_query_df.empty:
            csv_path = dataset_out_dir / f"{dataset_name}_budget_sweep_query_metrics.csv"
            combined_query_df.to_csv(csv_path, index=False)
            print(f"Saved {dataset_name} sweep query metrics to {csv_path}")

        # Written per-dataset (not into one shared file) so that datasets processed
        # in separate concurrent SLURM jobs never race on the same CSV. The final
        # combined `sweep_summary.csv` is produced by merging these afterward, either
        # by re-running this script with --dataset-paths covering everything already
        # done (see the merge step below), or via a separate merge pass.
        if summary_records:
            df_dataset_summary = pd.DataFrame(summary_records)
            dataset_summary_csv_path = dataset_out_dir / f"{dataset_name}_sweep_summary.csv"
            df_dataset_summary.to_csv(dataset_summary_csv_path, index=False)
            print(f"Saved {dataset_name} sweep summary to {dataset_summary_csv_path}")

        all_summary_records.extend(summary_records)
        print(f"\n{dataset_name}: first perfect ({args.stop_metric}==1.0) at budget_multiplier = {first_perfect_at}")

    if all_summary_records:
        df_summary = pd.DataFrame(all_summary_records)
        print("\n" + "=" * 80)
        print("BUDGET SWEEP SUMMARY (THIS RUN):")
        print(df_summary.to_string(index=False))
        print("=" * 80)

    # Merge every per-dataset summary CSV under results/budget_sweep_holdout/*/ into
    # one combined sweep_summary.csv. Safe to call even when only a subset of
    # datasets were processed in this invocation -- it just picks up whatever
    # per-dataset summaries exist on disk right now.
    per_dataset_csvs = sorted(base_results_dir.glob("*/*_sweep_summary.csv"))
    if per_dataset_csvs:
        combined = pd.concat([pd.read_csv(p) for p in per_dataset_csvs], ignore_index=True)
        summary_csv_path = base_results_dir / "sweep_summary.csv"
        combined.to_csv(summary_csv_path, index=False)
        print(f"\nMerged {len(per_dataset_csvs)} per-dataset summaries into {summary_csv_path}")


if __name__ == "__main__":
    main()
