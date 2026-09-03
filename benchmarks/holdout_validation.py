"""Split Test Benchmark for Spindle Covariance Indexing.

Performs blind holdout validation on indexed datasets, comparing Spindle's
two-stage Approximate Nearest Neighbor (ANN) search against ground-truth
exact block-wise Frobenius distance.
"""

import argparse
from pathlib import Path
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

# Setup import path for spindle_dev
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import spindle_dev.search as search


# =====================================================================
# Mathematical & Distance Utilities
# =====================================================================

def log_spd(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Compute the matrix logarithm of a Symmetric Positive Definite matrix."""
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T


def get_ordinal(n: int) -> str:
    """Return the ordinal string representation of an integer (e.g., 1st, 2nd)."""
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffixes = ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"]
    return str(n) + suffixes[n % 10]


def extract_query_matrices(test_tile_covs: list) -> list:
    """Extract raw covariance ndarrays from query dictionary objects."""
    query_matrices = []
    for q_dict in test_tile_covs:
        if isinstance(q_dict, dict):
            query_matrices.append(q_dict.get('cov', q_dict.get('matrix', q_dict)))
        else:
            query_matrices.append(q_dict)
    return query_matrices


# =====================================================================
# Stage 1: Spindle DAG Candidate Retrieval
# =====================================================================

def perform_search(query_matrices: list, data, dag_dict: dict, config, budget_multiplier: float = 2.0):
    """Query the Spindle DAG index to retrieve Stage 1 candidate pools."""
    search_cfg = search.SearchConfig(
        max_results=None,
        debug=False,
        max_failed_starts=10,
        max_failed_paths=20,
        total_paths_limit=300
    )

    print(f"Starting blind holdout validation for {len(query_matrices)} unseen queries...")
    print("-" * 65)

    print("Step 1/2: Assigning queries to Covariance-Niches using latent space...")
    if query_matrices:
        # Warmup call: runs one assignment to trigger any JIT / lazy-init paths so
        # the subsequent timed batch is not penalised by first-call overhead.
        _ = search.assign_clusters_to_new_spds(query_matrices[:1], data)

    assign_start = time.perf_counter()
    predicted_clusters = search.assign_clusters_to_new_spds(query_matrices, data)
    assign_total_time = time.perf_counter() - assign_start
    assign_time_ms_per_query = (assign_total_time / max(1, len(query_matrices))) * 1000
    print(f"Assignment complete in {assign_total_time:.3f}s ({assign_time_ms_per_query:.2f}ms per query)\n")

    print("Step 2/2: Performing distance-budgeted search across DAG...")
    search_start = time.perf_counter()  # use perf_counter throughout for consistent precision
    all_matched_train_ids = []
    spindle_search_times = []

    for j, cluster_id in enumerate(tqdm(predicted_clusters, desc="Querying Index", leave=True)):
        cluster_id = int(cluster_id)
        index_handle = dag_dict[cluster_id]
        epsilon = config.epsilon_dict[cluster_id]
        num_blocks = len(index_handle.sorted_blocks)

        budget = float(epsilon) * float(num_blocks) * float(budget_multiplier)

        q_spd = query_matrices[j]
        perm = data.perm_list[cluster_id]
        q_spd_perm = q_spd[np.ix_(perm, perm)]
        query_block_runs = data.block_dict[cluster_id]

        t0 = time.perf_counter()
        results = search.search_index(
            index_handle,
            q_spd_perm,
            [],
            query_block_runs,
            budget,
            config=search_cfg,
        )
        spindle_search_times.append(time.perf_counter() - t0)

        matched_ids_for_query = []
        seen_matched = set()
        if results.paths:
            for path in results.paths:
                member_sets = []
                for node_id in path.node_path:
                    node = index_handle.nodes[node_id]
                    members = getattr(getattr(node, "metadata", None), "members", [])
                    spd_ids = {int(spd_id) for spd_id, _ in members}
                    member_sets.append(spd_ids)

                intersect_ids = set.intersection(*member_sets) if member_sets else set()
                for spd_id in sorted(intersect_ids):
                    if spd_id not in seen_matched:
                        seen_matched.add(spd_id)
                        matched_ids_for_query.append(spd_id)

        all_matched_train_ids.append(matched_ids_for_query)

    search_time = time.perf_counter() - search_start
    print("-" * 65)
    print(f"Index Querying Complete! Total time: {search_time:.3f}s ({search_time/len(predicted_clusters):.4f}s per query)")

    return predicted_clusters, all_matched_train_ids, spindle_search_times, assign_time_ms_per_query


def summarize_hits(all_matched_train_ids: list, predicted_clusters: list):
    """Print summary statistics of retrieved Stage 1 candidate counts."""
    print("\n" + "=" * 40)
    print("           QUERY HITS SUMMARY")
    print("=" * 40)
    for j, hits in enumerate(all_matched_train_ids):
        target_niche = int(predicted_clusters[j])
        print(f"Query {j:3d} [Niche {target_niche:2d}]: {len(hits):4d} matches found")
    print("=" * 40 + "\n")


# =====================================================================
# Stage 2: Fine Re-ranking & Performance Benchmark
# =====================================================================

def evaluate_brute_force_approximation(
    test_tile_covs, train_tile_covs, train_idx, predicted_clusters,
    all_matched_train_ids, spindle_search_times, assign_time_ms_per_query, data, dataset_out_dir, dataset_name,
    top_c_candidates: int = 100
):
    """Evaluate Spindle search against exact ground-truth nearest neighbors."""
    print("\n" + "=" * 60)
    print(f"TASK 2: Two-Stage ANN Search Benchmark (Top-{top_c_candidates} Candidates)")
    print("=" * 60)

    global_to_local_train_map = {global_id: local_idx for local_idx, global_id in enumerate(train_idx)}

    print("Pre-computing matrix logarithms for training tiles...")
    niche_train_cache = {}
    unique_niches = set(int(lab) for lab in data.labels)
    for niche in unique_niches:
        perm = data.perm_list[niche]
        block_runs = data.block_dict[niche]
        niche_indices = [idx for idx, lab in enumerate(data.labels) if int(lab) == niche]

        cached_logs = {}
        for t_idx in niche_indices:
            t_spd = train_tile_covs[t_idx]
            t_spd = t_spd if not isinstance(t_spd, dict) else t_spd.get('cov')
            t_perm = t_spd[np.ix_(perm, perm)]
            cached_logs[t_idx] = [log_spd(t_perm[s:e, s:e]) for s, e in block_runs]

        niche_train_cache[niche] = (niche_indices, cached_logs)

    exact_dists = []
    spindle_dists = []
    spindle_ranks = []
    query_metrics_list = []

    for i, q_dict in enumerate(test_tile_covs):
        target_niche = int(predicted_clusters[i])
        perm = data.perm_list[target_niche]
        block_runs = data.block_dict[target_niche]
        niche_train_indices, cached_logs = niche_train_cache[target_niche]

        q_spd = q_dict if not isinstance(q_dict, dict) else q_dict.get('cov', q_dict.get('matrix', q_dict))
        q_perm = q_spd[np.ix_(perm, perm)]
        q_blocks_log = [log_spd(q_perm[s:e, s:e]) for s, e in block_runs]

        # Measure true exact brute-force search time (including log_spd on raw candidate tiles).
        # We time a sample of up to 100 tiles from the *same niche* and extrapolate to the full
        # niche size.  Using the total train-set size (across all niches) would over-inflate the
        # BF baseline because a real brute-force search would also be restricted to the predicted
        # niche after cluster assignment.
        bf_start = time.perf_counter()
        raw_dists = []
        sample_indices = niche_train_indices[:min(100, len(niche_train_indices))]
        for t_idx in sample_indices:
            t_raw = train_tile_covs[t_idx] if not isinstance(train_tile_covs[t_idx], dict) else train_tile_covs[t_idx].get('cov', train_tile_covs[t_idx])
            t_perm = t_raw[np.ix_(perm, perm)]
            d_val = sum(
                np.linalg.norm(q_blocks_log[b] - log_spd(t_perm[s:e, s:e]), ord='fro') / np.sqrt(e - s)
                for b, (s, e) in enumerate(block_runs)
            )
            raw_dists.append(d_val)
        bf_time_sample_ms = (time.perf_counter() - bf_start) * 1000

        # Scale sample time to full niche size (not full train set — see comment above).
        bf_time_ms = bf_time_sample_ms * (len(niche_train_indices) / max(1, len(raw_dists)))

        # For fast ranking evaluation, use pre-cached logs within target niche
        distances = []
        for t_idx in niche_train_indices:
            t_logs = cached_logs[t_idx]
            total_block_dist = sum(
                np.linalg.norm(q_blocks_log[b_idx] - t_logs[b_idx], ord='fro') / np.sqrt(end - start)
                for b_idx, (start, end) in enumerate(block_runs)
            )
            distances.append((total_block_dist, t_idx))
        distances.sort(key=lambda x: x[0])
        true_order = [idx for d, idx in distances]
        dist_dict = {idx: d for d, idx in distances}

        spindle_candidates_local = []
        for match_global_idx in all_matched_train_ids[i]:
            local_match_idx = global_to_local_train_map.get(match_global_idx)
            if local_match_idx is not None and local_match_idx in dist_dict:
                spindle_candidates_local.append(local_match_idx)

        stage1_pool = spindle_candidates_local[:top_c_candidates]

        # Time Stage 2 fine re-ranking computation on retrieved Stage 1 candidates
        rerank_start = time.perf_counter()
        spindle_rerank_dists = []
        for cand_local_idx in stage1_pool:
            c_logs = cached_logs[cand_local_idx]
            c_dist = sum(
                np.linalg.norm(q_blocks_log[b_idx] - c_logs[b_idx], ord='fro') / np.sqrt(end - start)
                for b_idx, (start, end) in enumerate(block_runs)
            )
            spindle_rerank_dists.append((c_dist, cand_local_idx))
        spindle_rerank_dists.sort(key=lambda x: x[0])
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000

        stage2_reranked = [idx for _, idx in spindle_rerank_dists]

        closest_dist = distances[0][0] if distances else float('inf')

        if not stage2_reranked:
            print(f"Query {i:3d} (Niche {target_niche}): Exact closest dist = {closest_dist:.3f} | Spindle found NOTHING.")
            exact_dists.append(closest_dist)
            spindle_dists.append(float('inf'))
            spindle_ranks.append(-1)
            spindle_best_dist = np.nan
            spindle_best_rank = -1
        else:
            best_match_idx = stage2_reranked[0]
            spindle_best_dist = dist_dict[best_match_idx]
            spindle_best_rank = true_order.index(best_match_idx) + 1
            print(f"Query {i:3d} (Niche {target_niche}): Exact closest dist = {closest_dist:.3f}, Spindle dist = {spindle_best_dist:.3f} | Spindle found {get_ordinal(spindle_best_rank)} closest neighbor.")
            exact_dists.append(closest_dist)
            spindle_dists.append(spindle_best_dist)
            spindle_ranks.append(spindle_best_rank)

        # Tie-aware recall@1: counts as a hit if Spindle's top result matches the true
        # minimum distance within floating-point tolerance, even if another tile shares
        # that exact distance (ties). For full-tile queries ties are extremely rare, so
        # this is equivalent to the strict rank==1 check in practice, but is consistent
        # with the partial search hit_in_k() definition which uses the same logic.
        recall_1 = 1 if (stage2_reranked and
            abs(dist_dict[stage2_reranked[0]] - closest_dist) < 1e-5) else 0
        dag_time_ms = spindle_search_times[i] * 1000 if i < len(spindle_search_times) else 0.0
        spindle_total_ms = assign_time_ms_per_query + dag_time_ms + rerank_time_ms

        metric_record = {
            'query_idx': i,
            'Dataset': dataset_name,
            'target_niche': target_niche,
            'spindle_search_time': round(spindle_total_ms / 1000.0, 6),
            'brute_force_time': round(bf_time_ms / 1000.0, 6),
            'spindle_time_ms': round(spindle_total_ms, 4),
            'bf_time_ms': round(bf_time_ms, 4),
            'speedup': round(bf_time_ms / spindle_total_ms, 2) if spindle_total_ms > 0 else np.nan,
            'exact_best_dist': round(closest_dist, 4),
            'spindle_best_dist': round(spindle_best_dist, 4) if spindle_best_rank != -1 else np.nan,
            'spindle_best_rank': spindle_best_rank,
            'recall_at_1': recall_1
        }

        # overlap@K: fraction of the true top-K (within the predicted niche) that appear in
        # Spindle's re-ranked top-K candidates.  Both numerator and denominator are intra-niche;
        # tiles assigned to other niches by the cluster step are not evaluated here.
        for K in [5, 10, 20, 30, 50]:
            true_top_k = set(true_order[:K])
            stage2_top_k = set(stage2_reranked[:K])
            denom = max(1, len(true_top_k))
            ov = len(true_top_k.intersection(stage2_top_k)) / float(denom)
            metric_record[f'overlap_at_{K}'] = round(ov, 4)

        query_metrics_list.append(metric_record)

    df_query_metrics = pd.DataFrame(query_metrics_list)
    csv_path = dataset_out_dir / f"{dataset_name}_query_metrics.csv"
    df_query_metrics.to_csv(csv_path, index=False)
    print(f"\nSaved detailed query metrics to {csv_path}")

    mean_sp_time_ms = df_query_metrics['spindle_time_ms'].mean()
    mean_bf_time_ms = df_query_metrics['bf_time_ms'].mean()
    mean_speedup = mean_bf_time_ms / mean_sp_time_ms if mean_sp_time_ms > 0 else np.nan

    summary_record = {
        'Dataset': dataset_name,
        'num_queries': len(df_query_metrics),
        'mean_spindle_time': round(mean_sp_time_ms / 1000.0, 6),
        'mean_brute_force_time': round(mean_bf_time_ms / 1000.0, 6),
        'mean_spindle_time_ms': round(mean_sp_time_ms, 4),
        'mean_brute_force_time_ms': round(mean_bf_time_ms, 4),
        'mean_speedup': round(mean_speedup, 2),
        'recall_at_1': round(df_query_metrics['recall_at_1'].mean(), 4),
        'overlap_at_5': round(df_query_metrics['overlap_at_5'].mean(), 4),
        'overlap_at_10': round(df_query_metrics['overlap_at_10'].mean(), 4),
        'overlap_at_20': round(df_query_metrics['overlap_at_20'].mean(), 4),
        'overlap_at_30': round(df_query_metrics['overlap_at_30'].mean(), 4),
        'overlap_at_50': round(df_query_metrics['overlap_at_50'].mean(), 4),
    }

    print("\n" + "-" * 60)
    print(f"BENCHMARK SUMMARY FOR {dataset_name}:")
    for k, v in summary_record.items():
        if k != 'Dataset': print(f"  {k:22s} : {v}")
    print("-" * 60)

    print(f"\nCompleted evaluation for {dataset_name}.")
    return df_query_metrics, summary_record


# =====================================================================
# Summary & Visualization
# =====================================================================

def generate_rank_distribution_csv(combined_df: pd.DataFrame, results_dir: Path):
    """Generate top-1 match rank distribution summary CSV."""
    print("\nGenerating top-1 match rank distribution CSV...")
    ranks = combined_df['spindle_best_rank'].values
    valid_ranks = ranks[ranks != -1]
    total = len(valid_ranks) if len(valid_ranks) > 0 else 1

    bins = [
        ('1st', valid_ranks == 1),
        ('2nd', valid_ranks == 2),
        ('3rd', valid_ranks == 3),
        ('4-5th', (valid_ranks >= 4) & (valid_ranks <= 5)),
        ('6-10th', (valid_ranks >= 6) & (valid_ranks <= 10)),
        ('>10th', valid_ranks > 10)
    ]

    records = []
    for label, mask in bins:
        count = np.sum(mask)
        pct = round((count / total) * 100.0, 2)
        records.append({'Rank_Category': label, 'Count': int(count), 'Percentage': pct})

    df_rank = pd.DataFrame(records)
    out_csv = results_dir / "top1_rank_distribution.csv"
    df_rank.to_csv(out_csv, index=False)
    print(f"Saved rank distribution CSV to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Run split test on saved indexed datasets.")
    parser.add_argument('--test', action='store_true', help='Run quick test (arg compat)')
    parser.add_argument('--top-c', type=int, default=400, help='Stage 1 candidate pool retrieval cap for Stage 2 re-ranking')
    parser.add_argument('--budget-mult', type=float, default=1.0, help='Distance budget multiplier for DAG search')
    parser.add_argument('--datasets', nargs='*', default=None, help='Optional list of dataset keywords to execute in specific order (e.g. kidney lung breast)')
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    base_indexed_dir = project_root / "results" / "holdout_validation_indexed"
    base_results_dir = project_root / "results" / "holdout_validation"
    base_results_dir.mkdir(exist_ok=True, parents=True)

    if not base_indexed_dir.exists():
        print(f"Directory {base_indexed_dir} not found. Please run index_datasets.py first.")
        return

    indexed_files = list(base_indexed_dir.glob("*_spindle_index.pkl"))
    if not indexed_files:
        indexed_files = list(base_indexed_dir.glob("*_indexed.pkl"))
    if not indexed_files:
        print(f"No indexed files found in {base_indexed_dir}. Please run index_datasets.py first.")
        return

    if args.datasets:
        ordered_files = []
        for kw in args.datasets:
            matched = [f for f in indexed_files if kw.lower() in f.name.lower() and f not in ordered_files]
            if matched:
                ordered_files.extend(matched)
            else:
                print(f"Warning: No indexed file matched dataset keyword '{kw}'")
        indexed_files = ordered_files
    else:
        indexed_files.sort(key=lambda f: f.name)

    all_query_metrics_dfs = []
    summary_records = []

    for indexed_file in indexed_files:
        print(f"\n{'=' * 80}")
        print(f"Processing indexed file: {indexed_file.name}")
        print(f"{'=' * 80}\n")

        print("Loading Spindle index data...")
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
            if not covs_file.exists():
                covs_file = project_root.parent / "Results Backup" / "holdout_validation_indexed" / f"{dataset_name}_raw_covariances.pkl"
            print(f"Loading benchmark raw covariance matrices from {covs_file.name}...")
            with open(covs_file, 'rb') as cf:
                covs_data = pickle.load(cf)
            test_tile_covs = covs_data['test_tile_covs']
            train_tile_covs = covs_data['train_tile_covs']
            train_idx = covs_data['train_idx']

        if args.test:
            print("[--test flag] Truncating test queries to 5 for quick run...")
            test_tile_covs = test_tile_covs[:5]

        dataset_out_dir = base_results_dir / dataset_name
        dataset_out_dir.mkdir(exist_ok=True, parents=True)

        query_matrices = extract_query_matrices(test_tile_covs)
        predicted_clusters, all_matched_train_ids, spindle_search_times, assign_time_ms_per_query = perform_search(
            query_matrices, data, dag_dict, config, budget_multiplier=args.budget_mult
        )
        summarize_hits(all_matched_train_ids, predicted_clusters)

        df_query_metrics, summary_record = evaluate_brute_force_approximation(
            test_tile_covs, train_tile_covs, train_idx, predicted_clusters,
            all_matched_train_ids, spindle_search_times, assign_time_ms_per_query, data, dataset_out_dir, dataset_name,
            top_c_candidates=args.top_c
        )

        if df_query_metrics is not None and not df_query_metrics.empty:
            all_query_metrics_dfs.append(df_query_metrics)
            summary_records.append(summary_record)

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        summary_csv_path = base_results_dir / "benchmark_summary.csv"
        if summary_csv_path.exists() and args.datasets:
            try:
                existing_df = pd.read_csv(summary_csv_path)
                df_summary = pd.concat(
                    [existing_df[~existing_df['Dataset'].isin(df_summary['Dataset'])], df_summary],
                    ignore_index=True
                )
            except Exception as e:
                print(f"Note: Could not merge existing summary ({e})")
        df_summary.to_csv(summary_csv_path, index=False)
        print("\n" + "=" * 80)
        print("OVERALL BENCHMARK SUMMARY ACROSS ALL DATASETS:")
        print(df_summary.to_string(index=False))
        print("=" * 80)
        print(f"\nSaved overall benchmark summary CSV to {summary_csv_path}")

    all_csvs = list(base_results_dir.glob("*/*_query_metrics.csv"))
    if all_csvs:
        try:
            combined_queries_df = pd.concat([pd.read_csv(p) for p in all_csvs], ignore_index=True)
            generate_rank_distribution_csv(combined_queries_df, base_results_dir)
        except Exception as e:
            print(f"Note: Could not generate rank distribution CSV ({e})")


if __name__ == "__main__":
    main()
