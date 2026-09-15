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
import index_datasets   # type: ignore


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

def _niche_scale_factor(n_niche: int, typical: int = 500, min_factor: float = 1.0, max_factor: float = 6.0) -> float:
    """Scale search effort with niche candidate-pool size.

    Sub-linear (sqrt) scaling anchored so niches at or below ``typical`` size
    get a factor of exactly 1.0 (i.e. no change from today's fixed
    constants -- effort/budget must never be reduced below the baseline that
    already works), while much larger niches get a bounded, proportional
    increase in search effort.
    """
    return float(np.clip(np.sqrt(max(n_niche, 1) / typical), min_factor, max_factor))


def perform_search(query_matrices: list, data, dag_dict: dict, config, budget_multiplier: float = 2.0,
                    effort_multiplier: float = 1.0):
    """Query the Spindle DAG index, searching EVERY niche for every query.

    Niches exist to group covariance matrices that share a permutation/
    block-diagonal structure (compression/indexing units), not as a
    search-space-reduction routing mechanism. Routing each query to a
    single predicted niche (via cluster assignment in a latent embedding
    space) was found to misroute a large fraction of queries on datasets
    with many niches -- the embedding used for routing doesn't reliably
    agree with the actual search/distance metric -- making that niche's
    true nearest neighbor permanently unreachable. Searching every niche's
    DAG (each with its own permutation/budget) removes that failure mode
    entirely; niches still provide their compression/block-diagonalization
    benefit, and each niche's budget-pruned DFS is still far cheaper than a
    brute-force scan of that niche, so the aggregate search is still much
    cheaper than a true brute-force scan of the whole dataset.
    """
    unique_niches = sorted(set(int(c) for c in data.labels))
    niche_sizes = {c: int(np.sum(data.labels == c)) for c in unique_niches}
    niche_search_cfgs = {}
    for c, n_niche in niche_sizes.items():
        f = _niche_scale_factor(n_niche) * effort_multiplier
        # Base effort caps (100 / 200 / 3000) are 10x the original constants
        # (10 / 20 / 300). The radius-aware pruning fix in search.py's dfs()
        # correctly widens which branches survive pruning (using a valid
        # distance lower bound instead of the raw distance-to-cluster-mean),
        # which means more of the DAG must actually be traversed to reach a
        # leaf -- the old caps were calibrated against the previous,
        # incorrectly-narrow pruning and were too tight once that was fixed.
        # 10x was empirically calibrated to match unlimited-budget Stage-1
        # candidate coverage (checked against exact brute-force ground truth)
        # across both tight-radius and large-radius datasets. effort_multiplier
        # is an additional knob (used by the budget sweep) to widen these caps
        # further when a stop metric plateaus below 1.0 for reasons unrelated
        # to the distance budget itself.
        niche_search_cfgs[c] = search.SearchConfig(
            max_results=None,
            debug=False,
            max_failed_starts=max(1, round(100 * f)),
            max_failed_paths=max(1, round(200 * f)),
            total_paths_limit=max(1, round(3000 * f)),
        )

    print(f"Starting blind holdout validation for {len(query_matrices)} unseen queries...")
    print(f"Searching all {len(unique_niches)} niches per query (no niche-routing step).")
    print("-" * 65)

    search_start = time.perf_counter()  # use perf_counter throughout for consistent precision
    all_matched_train_ids = []
    spindle_search_times = []

    for q_spd in tqdm(query_matrices, desc="Querying Index", leave=True):
        matched_ids_for_query = []
        seen_matched = set()
        query_search_time = 0.0

        for cluster_id in unique_niches:
            index_handle = dag_dict[cluster_id]
            epsilon = config.epsilon_dict[cluster_id]
            num_blocks = len(index_handle.sorted_blocks)
            f = _niche_scale_factor(niche_sizes[cluster_id])
            budget = float(epsilon) * float(num_blocks) * float(budget_multiplier) * f

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
                config=niche_search_cfgs[cluster_id],
            )
            query_search_time += time.perf_counter() - t0

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
        spindle_search_times.append(query_search_time)

    search_time = time.perf_counter() - search_start
    print("-" * 65)
    print(f"Index Querying Complete! Total time: {search_time:.3f}s "
          f"({search_time / max(1, len(query_matrices)):.4f}s per query, "
          f"{len(unique_niches)} niches searched per query)")

    return all_matched_train_ids, spindle_search_times


def summarize_hits(all_matched_train_ids: list):
    """Print summary statistics of retrieved Stage 1 candidate counts."""
    print("\n" + "=" * 40)
    print("           QUERY HITS SUMMARY")
    print("=" * 40)
    for j, hits in enumerate(all_matched_train_ids):
        print(f"Query {j:3d}: {len(hits):4d} matches found (across all niches)")
    print("=" * 40 + "\n")


# =====================================================================
# Stage 2: Fine Re-ranking & Performance Benchmark
# =====================================================================

# Recall@epsilon / Overlap@epsilon tolerance bands, each expressed as a fraction of
# the query's true-best-niche epsilon (config.epsilon_dict[true_best_niche]) -- the
# same per-cluster distance scale already used to size the search budget
# (budget = epsilon * num_blocks * budget_multiplier * f). Using a niche-relative
# band keeps the tolerance comparable across datasets/niches with different
# absolute distance scales, unlike a single global epsilon.
EPSILON_TOLERANCE_FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def evaluate_brute_force_approximation(
    test_tile_covs, train_tile_covs, train_idx,
    all_matched_train_ids, spindle_search_times, data, dataset_out_dir, dataset_name,
    config, top_c_candidates: int = 100
):
    """Evaluate Spindle search against exact ground-truth nearest neighbors.

    The brute-force baseline scans the ENTIRE indexed training set -- every
    niche, not just one. Each niche's tiles are compared using that niche's
    own permutation/block-decomposition (a property of how the data was
    pre-organized at index-build time), and results are combined into one
    global ranking across all niches for both the ground truth and the
    timed brute-force scan.

    Spindle's own search (``perform_search``) also now searches every
    niche's DAG per query (no niche-routing step -- see its docstring), so
    Stage-1 candidates can come from any niche. Stage-2 re-ranking looks up
    each candidate's own niche to use the matching cached logs/block
    structure -- there is no single "target niche" for a query anymore.
    ``top_c_candidates`` caps how many Stage-1 candidates are re-ranked
    *per niche* (not globally), since each niche contributes its own
    independent candidate pool.
    """
    print("\n" + "=" * 60)
    print(f"TASK 2: Two-Stage ANN Search Benchmark (Top-{top_c_candidates} Candidates, full-dataset brute force)")
    print("=" * 60)

    global_to_local_train_map = {global_id: local_idx for local_idx, global_id in enumerate(train_idx)}
    tile_niche = {idx: int(lab) for idx, lab in enumerate(data.labels)}

    print("Pre-computing matrix logarithms for training tiles...")
    niche_train_cache = {}
    unique_niches = sorted(set(int(lab) for lab in data.labels))
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
        q_spd = q_dict if not isinstance(q_dict, dict) else q_dict.get('cov', q_dict.get('matrix', q_dict))

        # Full-dataset ground truth + brute-force timing: for every niche
        # (not just the predicted one), project the query into that niche's
        # own permutation/block layout, time a from-scratch sample (raw
        # covariance -> log_spd, exactly what a real brute-force scan would
        # have to do, since it has no precomputed logs) extrapolated to that
        # niche's full size, and compute exact distances to every tile in
        # that niche using the pre-cached logs (cheap -- no eigendecomposition,
        # just norm computation). Distances from every niche are pooled into
        # one global ranking.
        global_distances = []
        bf_time_ms = 0.0
        query_blocks_log_by_niche = {}
        for niche in unique_niches:
            perm = data.perm_list[niche]
            block_runs = data.block_dict[niche]
            niche_indices, cached_logs = niche_train_cache[niche]

            q_perm = q_spd[np.ix_(perm, perm)]
            q_blocks_log = [log_spd(q_perm[s:e, s:e]) for s, e in block_runs]
            query_blocks_log_by_niche[niche] = q_blocks_log

            bf_start = time.perf_counter()
            raw_dists = []
            sample_indices = niche_indices[:min(100, len(niche_indices))]
            for t_idx in sample_indices:
                t_raw = train_tile_covs[t_idx] if not isinstance(train_tile_covs[t_idx], dict) else train_tile_covs[t_idx].get('cov', train_tile_covs[t_idx])
                t_perm = t_raw[np.ix_(perm, perm)]
                d_val = sum(
                    np.linalg.norm(q_blocks_log[b] - log_spd(t_perm[s:e, s:e]), ord='fro') / np.sqrt(e - s)
                    for b, (s, e) in enumerate(block_runs)
                )
                raw_dists.append(d_val)
            bf_time_sample_ms = (time.perf_counter() - bf_start) * 1000
            bf_time_ms += bf_time_sample_ms * (len(niche_indices) / max(1, len(raw_dists)))

            for t_idx in niche_indices:
                t_logs = cached_logs[t_idx]
                total_block_dist = sum(
                    np.linalg.norm(q_blocks_log[b_idx] - t_logs[b_idx], ord='fro') / np.sqrt(end - start)
                    for b_idx, (start, end) in enumerate(block_runs)
                )
                global_distances.append((total_block_dist, t_idx))

        global_distances.sort(key=lambda x: x[0])
        true_order = [idx for d, idx in global_distances]
        dist_dict = {idx: d for d, idx in global_distances}

        # Spindle's Stage-1 candidates can now come from ANY niche (every
        # niche's DAG was searched). Group candidates by their own niche and
        # cap each niche's contribution at top_c_candidates independently,
        # since each niche's Stage-1 search is an independent candidate pool.
        spindle_candidates_by_niche: dict = {}
        for match_global_idx in all_matched_train_ids[i]:
            local_match_idx = global_to_local_train_map.get(match_global_idx)
            if local_match_idx is not None and local_match_idx in dist_dict:
                cand_niche = tile_niche[local_match_idx]
                spindle_candidates_by_niche.setdefault(cand_niche, []).append(local_match_idx)

        stage1_pool = []
        for cand_niche, cands in spindle_candidates_by_niche.items():
            stage1_pool.extend(cands[:top_c_candidates])

        # Time Stage 2 fine re-ranking computation on retrieved Stage 1 candidates
        rerank_start = time.perf_counter()
        spindle_rerank_dists = []
        for cand_local_idx in stage1_pool:
            cand_niche = tile_niche[cand_local_idx]
            c_logs = niche_train_cache[cand_niche][1][cand_local_idx]
            c_block_runs = data.block_dict[cand_niche]
            c_q_blocks_log = query_blocks_log_by_niche[cand_niche]
            c_dist = sum(
                np.linalg.norm(c_q_blocks_log[b_idx] - c_logs[b_idx], ord='fro') / np.sqrt(end - start)
                for b_idx, (start, end) in enumerate(c_block_runs)
            )
            spindle_rerank_dists.append((c_dist, cand_local_idx))
        spindle_rerank_dists.sort(key=lambda x: x[0])
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000

        stage2_reranked = [idx for _, idx in spindle_rerank_dists]

        closest_dist = global_distances[0][0] if global_distances else float('inf')
        true_best_niche = tile_niche[true_order[0]] if true_order else None

        if not stage2_reranked:
            print(f"Query {i:3d}: Exact closest dist = {closest_dist:.3f} | Spindle found NOTHING.")
            exact_dists.append(closest_dist)
            spindle_dists.append(float('inf'))
            spindle_ranks.append(-1)
            spindle_best_dist = np.nan
            spindle_best_rank = -1
        else:
            best_match_idx = stage2_reranked[0]
            spindle_best_dist = dist_dict[best_match_idx]
            spindle_best_rank = true_order.index(best_match_idx) + 1
            print(f"Query {i:3d}: Exact closest dist = {closest_dist:.3f}, Spindle dist = {spindle_best_dist:.3f} | Spindle found {get_ordinal(spindle_best_rank)} closest neighbor.")
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
        # No niche-routing/assignment step exists anymore (every niche is
        # searched), so Spindle's total cost is just DAG search + re-rank.
        spindle_total_ms = dag_time_ms + rerank_time_ms

        metric_record = {
            'query_idx': i,
            'Dataset': dataset_name,
            'true_best_niche': true_best_niche,
            'spindle_search_time': round(spindle_total_ms / 1000.0, 6),
            'brute_force_time': round(bf_time_ms / 1000.0, 6),
            'spindle_time_ms': round(spindle_total_ms, 4),
            'bf_time_ms': round(bf_time_ms, 4),
            'speedup': round(bf_time_ms / spindle_total_ms, 2) if spindle_total_ms > 0 else np.nan,
            'exact_best_dist': round(closest_dist, 4),
            'spindle_best_dist': round(spindle_best_dist, 4) if spindle_best_rank != -1 else np.nan,
            'spindle_best_rank': spindle_best_rank,
            'recall_at_1': recall_1,
        }

        # overlap@K: fraction of the true (global) top-K that appear in Spindle's
        # re-ranked top-K candidates, pooled across every niche it searched.
        for K in [5, 10, 20, 30, 50]:
            true_top_k = set(true_order[:K])
            stage2_top_k = set(stage2_reranked[:K])
            denom = max(1, len(true_top_k))
            ov = len(true_top_k.intersection(stage2_top_k)) / float(denom)
            metric_record[f'overlap_at_{K}'] = round(ov, 4)

        # Recall@epsilon / Overlap@epsilon: distance-tolerance-based analogues of
        # recall_at_1/overlap_at_k, tolerant of near-misses instead of requiring an
        # exact (tied) distance match. The tolerance band is a fraction of the
        # TRUE best niche's own epsilon (there's no single "predicted niche"
        # anymore since every niche is searched), so it's comparable across
        # datasets/niches with different distance scales.
        niche_epsilon = float(config.epsilon_dict[true_best_niche]) if true_best_niche is not None else float('nan')
        niche_train_indices = niche_train_cache[true_best_niche][0] if true_best_niche is not None else []
        n_niche = max(1, len(niche_train_indices))
        n_total = max(1, len(true_order))
        for frac in EPSILON_TOLERANCE_FRACTIONS:
            band = frac * niche_epsilon
            recall_eps = 1 if (spindle_best_rank != -1 and
                (spindle_best_dist - closest_dist) <= band) else 0
            true_near_global = {idx for idx, d in dist_dict.items() if (d - closest_dist) <= band}
            spindle_near = {idx for idx in stage2_reranked if (dist_dict[idx] - closest_dist) <= band}
            denom_eps = max(1, len(true_near_global))
            overlap_eps = len(true_near_global.intersection(spindle_near)) / float(denom_eps)

            metric_record[f'recall_at_eps_{frac}'] = recall_eps
            metric_record[f'overlap_at_eps_{frac}'] = round(overlap_eps, 4)
            # Dataset-wide diagnostic: what fraction of the ENTIRE indexed
            # training set (all niches) falls within this tolerance band of
            # the true best -- if this approaches 1.0, the band is
            # essentially meaningless (almost everything qualifies).
            metric_record[f'true_near_frac_of_dataset_{frac}'] = round(len(true_near_global) / n_total, 4)
            # Niche-scoped diagnostic (same definition as before this change):
            # what fraction of the niche Spindle actually searched falls
            # within this band -- if this approaches 1.0, the search budget
            # is effectively retrieving the whole niche rather than a
            # meaningful neighborhood. This is what budget_sweep_holdout.py's
            # over-retrieval flag is based on.
            niche_near = {idx for idx in niche_train_indices if (dist_dict[idx] - closest_dist) <= band}
            metric_record[f'true_near_frac_of_niche_{frac}'] = round(len(niche_near) / n_niche, 4)

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
    for frac in EPSILON_TOLERANCE_FRACTIONS:
        summary_record[f'recall_at_eps_{frac}'] = round(df_query_metrics[f'recall_at_eps_{frac}'].mean(), 4)
        summary_record[f'overlap_at_eps_{frac}'] = round(df_query_metrics[f'overlap_at_eps_{frac}'].mean(), 4)
        summary_record[f'true_near_frac_of_niche_{frac}'] = round(df_query_metrics[f'true_near_frac_of_niche_{frac}'].mean(), 4)
        summary_record[f'true_near_frac_of_dataset_{frac}'] = round(df_query_metrics[f'true_near_frac_of_dataset_{frac}'].mean(), 4)

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
    parser.add_argument('--dataset-paths', nargs='*', default=None, help='Paths to the datasets')
    parser.add_argument('--max-queries', type=int, default=None,
                         help='Cap the number of held-out test queries evaluated per dataset '
                              '(randomly subsampled with a fixed seed). Default: no cap.')
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    base_indexed_dir = project_root / "results" / "holdout_validation_indexed"
    base_results_dir = project_root / "results" / "holdout_validation"
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
            "pancreatic_cancer": project_root / "dataset" / "xenium_human_pancreatic_cancer.h5ad"
        }

    # Indexing (including the train/test split, via --train-test-ratio) is handled
    # separately in Phase 1 (index_datasets.py) — this script only loads the
    # already-built index and covariance pickles.
    # index_datasets.run_indexing_for_datasets(datasets, is_test=args.test, train_test_ratio=0.05)

    indexed_files = []
    for ds_name in datasets.keys():
        idx_path = base_indexed_dir / f"{ds_name}_spindle_index.pkl"
        if not idx_path.exists():
            idx_path = base_indexed_dir / f"{ds_name}_indexed.pkl"
        if idx_path.exists():
            indexed_files.append(idx_path)
        else:
            print(f"Warning: Index file for {ds_name} not found.")


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
        elif args.max_queries is not None and len(test_tile_covs) > args.max_queries:
            rng = np.random.default_rng(42)
            keep = np.sort(rng.choice(len(test_tile_covs), size=args.max_queries, replace=False))
            print(f"[--max-queries {args.max_queries}] Subsampling {len(test_tile_covs)} "
                  f"test queries down to {args.max_queries}...")
            test_tile_covs = [test_tile_covs[i] for i in keep]

        dataset_out_dir = base_results_dir / dataset_name
        dataset_out_dir.mkdir(exist_ok=True, parents=True)

        query_matrices = extract_query_matrices(test_tile_covs)
        all_matched_train_ids, spindle_search_times = perform_search(
            query_matrices, data, dag_dict, config, budget_multiplier=args.budget_mult
        )
        summarize_hits(all_matched_train_ids)

        df_query_metrics, summary_record = evaluate_brute_force_approximation(
            test_tile_covs, train_tile_covs, train_idx,
            all_matched_train_ids, spindle_search_times, data, dataset_out_dir, dataset_name,
            config, top_c_candidates=args.top_c
        )

        if df_query_metrics is not None and not df_query_metrics.empty:
            all_query_metrics_dfs.append(df_query_metrics)
            summary_records.append(summary_record)

    if summary_records:
        df_summary = pd.DataFrame(summary_records)
        summary_csv_path = base_results_dir / "benchmark_summary.csv"
        if summary_csv_path.exists() and args.dataset_paths:
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
