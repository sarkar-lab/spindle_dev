import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import time
import random
from tqdm.auto import tqdm

# Set up sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import spindle_dev
import spindle_dev.index as index
import spindle_dev.preprocessing as preprocessing
import spindle_dev.typing as typing
import spindle_dev.interval_index as interval_index

def prepare_to_index(adata, max_pts=200, top_genes=800):
    """
    Prepare standard data object for indexing.
    """
    coords = adata.obsm["spatial"]
    tiles = preprocessing.build_quadtree_tiles(coords, max_pts=max_pts, min_side=0.0, max_depth=40)
    num_genes = adata.n_vars
    genes_work, gene_idx = spindle_dev.preprocessing.topvar_genes(adata, G=min(top_genes, num_genes))
    tile_covs = spindle_dev.preprocessing.build_tile_covs_full(adata, tiles, gene_idx, n_jobs=8, eps=1e-6)

    return tiles, tile_covs, genes_work

def load_and_split_data(adata_path, test_ratio=0.02, seed=42, max_pts=200, top_genes=800):
    print(f"Reading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    if 'Cluster' in adata.obs.columns:
        adata = adata[adata.obs.loc[adata.obs.Cluster != "Unlabeled"].index, :].copy()

    print(f"Preparing data for indexing (max_pts={max_pts}, top_genes={top_genes})...")
    tiles, tile_covs, genes_work = prepare_to_index(adata, max_pts=max_pts, top_genes=top_genes)

    np.random.seed(seed) 
    num_total_tiles = len(tiles)
    num_test = int(num_total_tiles * test_ratio)

    all_indices = np.arange(num_total_tiles)
    test_idx = np.random.choice(all_indices, size=num_test, replace=False)
    train_idx = np.setdiff1d(all_indices, test_idx)

    train_tiles = [tiles[i] for i in train_idx]
    train_tile_covs = [tile_covs[i] for i in train_idx]
    test_tiles = [tiles[i] for i in test_idx]
    test_tile_covs = [tile_covs[i] for i in test_idx]

    print(f"Total tiles: {num_total_tiles} | Training/Indexed: {len(train_tiles)} | Held out/Testing: {len(test_tiles)}")

    return adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx

def run_index(tiles, tile_covs, genes_work, adata, resolution=0.2, min_final_size=20, max_niche_size=1000):
    """
    Run indexing workflow.
    """
    data = index.ProcessedData(tiles, tile_covs, genes_work, adata.n_obs)
    data.reduce_dim(num_pca_components=30, n_components=2, do_umap=True)
    data.cluster_spds(
        cluster_distance="tree", cluster_method="leiden", resolution=resolution,
        adaptive_resolution=True, max_niche_size=max_niche_size
    )
    data.assign_label_to_spots()
    data.get_corr_mean_by_cluster()
    out_dict = data.get_adaptive_runs(find_blocks=True, with_size_guard=True, min_final_size=min_final_size, max_final_size=100)
    return data, out_dict

def configure_and_build_dag(data):
    print("Configuring adaptive epsilons for blocks...")
    epsilon_block_wise_dict = {}
    epsilon_dict = {}
    for cluster_id in set(data.labels):
        eps_per_block, eps_elbow_per_block, eps = index.choose_adaptive_epsilons(data, cluster_id, k_target_per_block=64)
        epsilon_block_wise_dict[int(cluster_id)] = eps_per_block
        epsilon_dict[int(cluster_id)] = eps

    # Floor each niche's budget-sizing epsilon at the dataset-wide median
    # across niches -- prevents small/homogeneous niches from being starved
    # of search budget regardless of budget_multiplier (see
    # benchmarks/index_datasets.py's configure_and_build_dag for the full
    # rationale and the confirmed failure case this fixes).
    if epsilon_dict:
        median_eps = float(np.median(list(epsilon_dict.values())))
        epsilon_dict = {k: max(v, median_eps) for k, v in epsilon_dict.items()}

    config = typing.IndexConfig()
    config.epsilon_dict = epsilon_dict
    config.epsilon_block_wise_dict = epsilon_block_wise_dict
    config.threshold_type = 'block_wise'
    config.kmean_method = 'epsilon_net'

    print("Creating index DAG...")
    dag_dict, stat, dist_list = index.index_spds(data, config=config)
    
    return dag_dict, config

def extract_query_matrices(test_tile_covs):
    query_matrices = []
    for q_dict in test_tile_covs:
        if isinstance(q_dict, dict):
            query_matrices.append(q_dict.get('cov', q_dict.get('matrix', q_dict)))
        else:
            query_matrices.append(q_dict)
    return query_matrices

def search_all_clusters_spindle(interval_index_obj, data, perm_ivl, q_spd, top_k=20):
    genes_set = set(perm_ivl)
    valid_len = len(perm_ivl)
    total_scale = np.sqrt(valid_len) if valid_len > 1 else 1.0
    
    all_cluster_ranked = []
    
    for c_id in set(data.labels):
        piece_scores = []
        piece_scales = []
        
        for b_idx, (block_start, block_end) in enumerate(data.block_dict.get(c_id, [])):
            block_perm = data.perm_list[c_id][block_start:block_end]
            
            indices = [i for i, g in enumerate(block_perm) if g in genes_set]
            if not indices:
                continue
                
            ranges_c = []
            start = indices[0]
            prev = indices[0]
            for i in indices[1:]:
                if i == prev + 1:
                    prev = i
                else:
                    ranges_c.append((start, prev + 1))
                    start = i
                    prev = i
            ranges_c.append((start, prev + 1))
            
            q_block_c = q_spd[np.ix_(block_perm, block_perm)]
            
            for a, b in ranges_c:
                pieces = interval_index.decompose_to_dyadic(a, b)
                for pa, pb in pieces:
                    q_sub = q_block_c[pa:pb, pa:pb]
                    p = pb - pa
                    scale = np.sqrt(p) if p > 1 else 1.0
                    
                    piece_results = interval_index.query_interval_index(
                        interval_index_obj, c_id, b_idx, (pa, pb), q_sub, top_k=None
                    )
                    
                    if not piece_results:
                        piece_scores.append({})
                        piece_scales.append(scale)
                        continue
                        
                    scores = {}
                    for dist, members in piece_results:
                        for m in members:
                            if m not in scores or dist < scores[m]:
                                scores[m] = dist
                    piece_scores.append(scores)
                    piece_scales.append(scale)
                    
        if not piece_scores:
            continue
            
        common = set(piece_scores[0].keys())
        for ps in piece_scores[1:]:
            common &= ps.keys()
            if not common:
                break
                
        if not common:
            continue
            
        for tid in common:
            total_sq = 0.0
            for i, ps in enumerate(piece_scores):
                unscaled = ps[tid] * piece_scales[i]
                total_sq += unscaled ** 2
            combined_dist = np.sqrt(total_sq) / total_scale
            all_cluster_ranked.append((combined_dist, tid))
            
    all_cluster_ranked = sorted(all_cluster_ranked, key=lambda x: x[0])
    return [(dist, [tid]) for dist, tid in all_cluster_ranked[:top_k]]

def run_benchmark_suite(queries, data, ivl_idx, search_budget, config):
    """Run brute-force + Spindle interval search + metrics for a list of
    fully pre-built query specs (see partial_panel_search.py::build_binned_queries).

    Unlike the original version of this function, gene-range construction
    (choosing a target length, picking a qualifying block, drawing the
    contiguous/non-contiguous sub-range) happens entirely upstream, in the
    caller -- each ``queries`` entry already carries a concrete ``perm_ivl``.
    This lets the caller *guarantee* coverage of specific query-length bins
    (including rare long-query bins that pure round-robin block sampling
    would leave empty) rather than discovering whatever length distribution
    falls out of round-robin niche/block sampling after the fact.
    """
    benchmark_results_log = []

    all_spd_matrices = np.asarray(data.spd_matrices)
    all_spd_ids = np.asarray(data.spd_ids)
    sid_to_idx = {int(all_spd_ids[k]): k for k in range(len(all_spd_ids))}
    median_block_epsilon = (
        float(np.median(list(config.epsilon_dict.values()))) if config.epsilon_dict else float('nan')
    )
    # Partial-gene-interval distances are far less discriminative than the
    # full-block/whole-matrix distances holdout_validation.py's
    # recall_at_eps_{0.1,0.5}/overlap_at_eps_{0.5,1.0} convention was tuned
    # for -- a fixed absolute radius on a handful of genes sweeps in a huge
    # fraction of the dataset (verified: at frac=0.5 the true-near set already
    # covers 7-40% of all training tiles, vs. Spindle's fixed top_k-sized
    # retrieval pool), so overlap collapses on band size alone rather than
    # search quality. Use smaller fractions here so the near-set stays in a
    # range a bounded retrieval pool can meaningfully cover.
    eps_fracs = [0.05, 0.1, 0.25]

    for q_info in tqdm(queries, desc="Benchmarking"):
        q_spd = q_info['q_spd']
        perm_ivl = q_info['perm_ivl']
        valid_len = len(perm_ivl)
        block_size = q_info['block_size']

        # The block-level epsilon was calibrated (choose_adaptive_epsilons)
        # against full-block distances (fixed size = block_size, both
        # normalized by sqrt(block_size)). Interval queries here are a
        # sub-range of that block (valid_len <= block_size), and their
        # distance is normalized by sqrt(valid_len) instead -- so the raw
        # block epsilon must be rescaled by sqrt(valid_len / block_size)
        # to stay on the same distance scale as what's actually compared,
        # otherwise short intervals get an artificially loose band and
        # long intervals an artificially tight one.
        query_band_epsilon = median_block_epsilon * np.sqrt(valid_len / block_size)

        # =========================================================
        # VECTORIZED BRUTE FORCE
        # =========================================================
        bf_start_time = time.perf_counter()
        q_sub = q_spd[np.ix_(perm_ivl, perm_ivl)]
        q_log = interval_index._log_spd(q_sub)
        scale = np.sqrt(valid_len) if valid_len > 1 else 1.0

        t_subs = all_spd_matrices[:, perm_ivl, :][:, :, perm_ivl]
        t_logs = np.array([interval_index._log_spd(t_sub) for t_sub in t_subs])

        diffs = t_logs - q_log
        dists = np.linalg.norm(diffs, axis=(1, 2)) / scale
        true_min_dist = np.min(dists)

        rounded_dists = np.round(dists, 5)
        unique_dists = np.sort(np.unique(rounded_dists))
        exact_partial_map = {int(all_spd_ids[i]): rounded_dists[i] for i in range(len(dists))}
        bf_time_ms = (time.perf_counter() - bf_start_time) * 1000
        # =========================================================
        # SPINDLE INTERVAL SEARCH
        # =========================================================
        spindle_start = time.perf_counter()
        results = search_all_clusters_spindle(
            ivl_idx, data, perm_ivl, q_spd, top_k=search_budget
        )
        dag_time_ms = (time.perf_counter() - spindle_start) * 1000

        flat_results = []
        for err, ids in results:
            for i in ids:
                flat_results.append((err, i))

        retrieved_sids = [sid for _, sid in flat_results]

        # Real exact re-ranking computation on retrieved candidate spots
        rerank_start = time.perf_counter()
        rerank_scores = []
        for sid in retrieved_sids:
            idx_k = sid_to_idx.get(int(sid))
            if idx_k is not None:
                t_sub = all_spd_matrices[idx_k][np.ix_(perm_ivl, perm_ivl)]
                t_log = interval_index._log_spd(t_sub)
                diff = t_log - q_log
                dist = np.linalg.norm(diff, ord='fro') / scale
                rerank_scores.append((dist, sid))

        rerank_scores.sort(key=lambda x: x[0])
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
        spindle_time_ms = dag_time_ms + rerank_time_ms

        spindle_retrieved_ids_all = [sid for _, sid in rerank_scores]

        spindle_best_rank = -1
        spindle_best_partial_dist = float('inf')

        if spindle_retrieved_ids_all:
            spindle_top1_id = spindle_retrieved_ids_all[0]
            spindle_best_partial_dist = exact_partial_map.get(spindle_top1_id, float('inf'))
            if spindle_best_partial_dist != float('inf'):
                spindle_best_rank = np.searchsorted(unique_dists, np.round(spindle_best_partial_dist, 5)) + 1

        record = {
            'Case': q_info['case_name'],
            'Tile': q_info['tile_id'],
            'Niche': q_info['cluster_id'],
            'Block_Index': q_info['block_index'],
            'Length_Bin': q_info['length_bin'],
            'Query_Length': valid_len,
            'rank': spindle_best_rank,
            'dist_gap': (spindle_best_partial_dist - true_min_dist) if spindle_best_rank != -1 else float('inf'),
            'spindle_time_ms': round(spindle_time_ms, 4),
            'brute_force_time_ms': round(bf_time_ms, 4),
            'speedup': round(bf_time_ms / spindle_time_ms, 2) if spindle_time_ms > 0 else np.nan,
            'retrieved': len(flat_results),
        }

        # Recall@epsilon / Overlap@epsilon: distance-tolerance-based hit
        # metrics, matching holdout_validation.py's convention. There is
        # no single "true niche" for these queries (they're sampled across
        # every niche/block), so the tolerance band is anchored on the
        # dataset-wide median epsilon, rescaled per query to this interval's
        # length (see query_band_epsilon above).
        for frac in eps_fracs:
            band = frac * query_band_epsilon
            recall_eps = 1 if (spindle_best_rank != -1 and
                (spindle_best_partial_dist - true_min_dist) <= band) else 0
            true_near_ids = {sid for sid, d in exact_partial_map.items() if (d - true_min_dist) <= band}
            spindle_near_ids = {
                sid for sid in spindle_retrieved_ids_all
                if (exact_partial_map.get(sid, float('inf')) - true_min_dist) <= band
            }
            denom_eps = max(1, len(true_near_ids))
            overlap_eps = len(true_near_ids.intersection(spindle_near_ids)) / float(denom_eps)

            record[f'recall_at_eps_{frac}'] = recall_eps
            record[f'overlap_at_eps_{frac}'] = round(overlap_eps, 4)

        benchmark_results_log.append(record)

    return benchmark_results_log

def generate_performance_report(df, results_dir, labels):
    report_path = results_dir / "performance_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Interval Index Partial Search Performance Report\n\n")
        f.write("This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.\n\n")
        
        for case_name in df['Case'].unique():
            f.write(f"## {case_name}\n")
            case_df = df[df['Case'] == case_name]
            
            f.write(f"| Query Size | Count | Recall@eps=0.05 (%) | Recall@eps=0.1 (%) | Overlap@eps=0.1 (%) | Overlap@eps=0.25 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |\n")
            f.write(f"|:----------:|:-----:|:-------------------:|:------------------:|:-------------------:|:--------------------:|:--------:|:------------:|:-------:|:-----------:|\n")

            for bin_label in labels:
                sub_df = case_df[(case_df['Length_Bin'] == bin_label) & (case_df['rank'] != -1)]
                if not sub_df.empty:
                    r005 = sub_df['recall_at_eps_0.05'].mean() * 100
                    r01 = sub_df['recall_at_eps_0.1'].mean() * 100
                    olap01 = sub_df['overlap_at_eps_0.1'].mean() * 100
                    olap025 = sub_df['overlap_at_eps_0.25'].mean() * 100
                    rank = sub_df['rank'].mean()
                    sp_tms = sub_df['spindle_time_ms'].mean()
                    bf_tms = sub_df['brute_force_time_ms'].mean()
                    speedup = bf_tms / sp_tms if sp_tms > 0 else np.nan
                    f.write(f"| {bin_label} | {len(sub_df)} | {r005:.1f} | {r01:.1f} | {olap01:.1f} | {olap025:.1f} | {rank:.1f} | {sp_tms:.2f} | {bf_tms:.2f} | {speedup:.2f} |\n")
            
            f.write("\n")
            
    with open(report_path, "r") as f:
        print("\n\n" + f.read())
        
    print(f"-> Full report saved to {report_path}")

def generate_visual_plots(df, results_dir, labels):
    print("Visual plots disabled (CSV-only export mode).")

def generate_overall_dataset_plots(combined_df, results_dir):
    print("Overall dataset plots disabled (CSV-only export mode).")
