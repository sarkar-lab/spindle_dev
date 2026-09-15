"""Budget-Sweep Holdout Benchmark for Spindle Covariance Indexing.

Runs the holdout validation benchmark (see ``holdout_validation.py``) across
a fixed sequence of distance-budget multipliers (``DEFAULT_BUDGET_MULTS``),
for each dataset, at a fixed ``top_c`` (Stage-2 re-rank candidate cap) --
characterizing the recall/overlap-vs-speedup tradeoff over a realistic,
bounded operating range, and flagging where higher budgets start
over-retrieving the niche rather than doing a meaningful search.

Each swept row also carries the Recall@epsilon / Overlap@epsilon columns
computed in ``holdout_validation.evaluate_brute_force_approximation`` (see
``EPSILON_TOLERANCE_FRACTIONS``), plus a ``true_near_frac_of_niche_1.0``-derived
``over_retrieving_niche`` flag: once that fraction gets large at a given
budget multiplier, Spindle is no longer doing a meaningful search -- it's
retrieving most/all of the niche, which trivially inflates recall/overlap.
That's the degenerate high-budget regime the recall-vs-budget figure needs
to visually separate from the real sweet spot.

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

# Log-spaced (constant-ratio, ~1.27x per step) rather than the old ad hoc
# list -- a constant ratio keeps absolute step sizes small while
# budget_multiplier itself is small (where the recall/overlap transition
# actually happens) and lets them grow automatically at higher multipliers,
# without needing a separate "coarsen later" rule. 30 points was chosen so
# the low-budget transition region gets several points instead of jumping
# straight from a starved budget to an already-saturated one.
DEFAULT_BUDGET_MULTS = [round(float(x), 4) for x in np.geomspace(0.01, 32.0, num=30)]

# Above this fraction of a niche falling within the loosest epsilon-tolerance
# band (see hv.EPSILON_TOLERANCE_FRACTIONS[-1], i.e. true_near_frac_of_niche_1.0),
# Spindle is effectively retrieving the whole niche rather than doing a
# meaningful nearest-neighbor search -- flag rows past this threshold so the
# sweep figure can grey out the degenerate high-budget region.
OVER_RETRIEVAL_NICHE_FRAC_THRESHOLD = 0.8

# Once stop_metric has been (near-)flat for this many consecutive swept
# multipliers, stop sweeping this dataset -- the plateau (whether at 1.0 or
# capped below it by top_c) is already captured, and continuing through the
# rest of DEFAULT_BUDGET_MULTS would just spend compute confirming the same
# flat line further out.
PLATEAU_WINDOW = 3
PLATEAU_TOL = 1e-6


def sweep_dataset(
    dataset_name, data, dag_dict, config,
    test_tile_covs, train_tile_covs, train_idx,
    dataset_out_dir, budget_mults, stop_metric, top_c_candidates,
):
    """Run the holdout benchmark across a fixed sequence of budget multipliers.

    Walks ``budget_mults`` (default ``DEFAULT_BUDGET_MULTS``) in order, at a
    fixed ``top_c_candidates`` throughout -- no auto-doubling/effort-widening
    chase for a perfect score. Stops early, before exhausting the sequence,
    once ``stop_metric`` has been flat for ``PLATEAU_WINDOW`` consecutive
    multipliers (whether that plateau is at 1.0 or capped below it by
    top_c) -- the shape of the curve is already captured at that point, and
    continuing would just re-confirm the same flat line at higher budgets
    for no benefit. This experiment is about characterizing the
    recall/overlap-vs-speedup curve over a realistic, bounded operating
    range (and catching where it starts over-retrieving the niche), not
    about proving that budget_multiplier can eventually be cranked high
    enough to reproduce brute force exactly.
    Returns (per_query_df, per_multiplier_summary_records, first_perfect_at)
    where ``first_perfect_at`` is the smallest swept multiplier (if any) at
    which ``stop_metric`` reached 1.0.
    """
    query_matrices = hv.extract_query_matrices(test_tile_covs)

    all_query_dfs = []
    summary_records = []
    first_perfect_at = None
    recent_metric_values = []

    for budget_mult in budget_mults:
        print(f"\n>>> {dataset_name}: budget_multiplier = {budget_mult}")
        all_matched_train_ids, spindle_search_times = (
            hv.perform_search(query_matrices, data, dag_dict, config, budget_multiplier=budget_mult)
        )

        df_query_metrics, summary_record = hv.evaluate_brute_force_approximation(
            test_tile_covs, train_tile_covs, train_idx,
            all_matched_train_ids, spindle_search_times,
            data, dataset_out_dir, dataset_name, config,
            top_c_candidates=top_c_candidates,
        )

        df_query_metrics = df_query_metrics.copy()
        df_query_metrics['budget_multiplier'] = budget_mult
        loosest_frac = hv.EPSILON_TOLERANCE_FRACTIONS[-1]
        df_query_metrics['over_retrieving_niche'] = (
            df_query_metrics[f'true_near_frac_of_niche_{loosest_frac}'] > OVER_RETRIEVAL_NICHE_FRAC_THRESHOLD
        )
        all_query_dfs.append(df_query_metrics)

        # Realized budget varies per-niche (epsilon * num_blocks * mult * niche_scale);
        # every niche is searched per query now, so report the mean across ALL niches.
        realized_budgets = []
        for cluster_id in sorted(set(int(c) for c in data.labels)):
            epsilon = config.epsilon_dict[cluster_id]
            num_blocks = len(dag_dict[cluster_id].sorted_blocks)
            n_niche = int(np.sum(data.labels == cluster_id))
            f = hv._niche_scale_factor(n_niche)
            realized_budgets.append(float(epsilon) * float(num_blocks) * float(budget_mult) * f)

        summary_record = dict(summary_record)
        summary_record['budget_multiplier'] = budget_mult
        summary_record['top_c_used'] = top_c_candidates
        summary_record['mean_realized_budget'] = float(np.mean(realized_budgets)) if realized_budgets else np.nan
        summary_record['over_retrieving_niche_frac'] = round(
            float(df_query_metrics['over_retrieving_niche'].mean()), 4
        )
        summary_records.append(summary_record)

        metric_value = summary_record.get(stop_metric, np.nan)
        print(f"    {stop_metric} = {metric_value}")

        if first_perfect_at is None and metric_value is not None and metric_value >= 1.0:
            first_perfect_at = budget_mult

        recent_metric_values.append(metric_value)
        recent_metric_values = recent_metric_values[-PLATEAU_WINDOW:]
        plateaued = (
            len(recent_metric_values) == PLATEAU_WINDOW
            and np.all(np.isfinite(recent_metric_values))
            and max(recent_metric_values) - min(recent_metric_values) < PLATEAU_TOL
        )
        if plateaued:
            print(f"    {stop_metric} has been flat at {metric_value} for the last "
                  f"{PLATEAU_WINDOW} budget multipliers -- stopping this dataset's sweep early.")
            break

    for rec in summary_records:
        rec['first_perfect_at'] = first_perfect_at

    combined_query_df = pd.concat(all_query_dfs, ignore_index=True) if all_query_dfs else pd.DataFrame()
    if not combined_query_df.empty:
        combined_query_df['first_perfect_at'] = first_perfect_at

    if first_perfect_at is not None:
        print(f"    Note: first reached {stop_metric}=1.0 at budget_multiplier={first_perfect_at}.")
    else:
        print(f"    Note: {stop_metric} never reached 1.0 across the swept range "
              f"(top_c={top_c_candidates} may be the binding constraint for some niches).")

    return combined_query_df, summary_records, first_perfect_at


def main():
    parser = argparse.ArgumentParser(description="Sweep distance-budget multiplier across holdout validation.")
    parser.add_argument('--test', action='store_true', help='Run quick test: 5 queries, short multiplier list')
    parser.add_argument('--budget-mults', type=float, nargs='*', default=None,
                         help='Space-separated budget multipliers to sweep, ascending')
    parser.add_argument('--stop-metric', type=str, default='recall_at_eps_0.1',
                         choices=['overlap_at_50', 'recall_at_1', 'overlap_at_eps_1.0', 'recall_at_eps_1.0',
                                  'recall_at_eps_0.1'],
                         help='Metric checked for perfect (==1.0) recovery AND for the plateau early-stop. '
                              'Defaults to the strict-tolerance Recall@eps rather than Overlap@eps: overlap '
                              'needs much more budget to plateau on datasets with many niches (each additional '
                              'niche searched adds real Stage-1 cost), so using it as the stop metric burns '
                              'budget chasing a metric that is not the primary one of interest (finding the '
                              'true nearest neighbor).')
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
            "lymph_node_5k": project_root / "dataset" / "xenium_human_lymph_node_5k.h5ad",
            "lung_cancer": project_root / "dataset" / "xenium_human_lung_cancer.h5ad",
            "skin_melanoma": project_root / "dataset" / "xenium_human_skin_melanoma.h5ad",
            "pancreatic_cancer": project_root / "dataset" / "xenium_human_pancreatic_cancer.h5ad",
            "brain_cancer": project_root / "dataset" / "xenium_human_brain_cancer.h5ad",
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

        combined_query_df, summary_records, first_perfect_at = sweep_dataset(
            dataset_name, data, dag_dict, config,
            test_tile_covs, train_tile_covs, train_idx,
            dataset_out_dir, budget_mults, args.stop_metric, args.top_c,
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
