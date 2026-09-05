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

def prepare_to_index(adata):
    """
    Prepare standard data object for indexing.
    """
    coords = adata.obsm["spatial"]
    tiles = preprocessing.build_quadtree_tiles(coords, max_pts=200, min_side=0.0, max_depth=40)
    num_genes = adata.n_vars
    genes_work, gene_idx = spindle_dev.preprocessing.topvar_genes(adata, G=min(800, num_genes))
    tile_covs = spindle_dev.preprocessing.build_tile_covs_full(adata, tiles, gene_idx, n_jobs=8, eps=1e-6)

    return tiles, tile_covs, genes_work

def load_and_split_data(adata_path, test_ratio=0.02, seed=42):
    print(f"Reading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    if 'Cluster' in adata.obs.columns:
        adata = adata[adata.obs.loc[adata.obs.Cluster != "Unlabeled"].index, :].copy()

    print("Preparing data for indexing...")
    tiles, tile_covs, genes_work = prepare_to_index(adata)

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

def run_index(tiles, tile_covs, genes_work, adata, resolution=0.2, min_final_size=20):
    """
    Run indexing workflow.
    """
    data = index.ProcessedData(tiles, tile_covs, genes_work, adata.n_obs)
    data.reduce_dim(num_pca_components=30, n_components=2, do_umap=True)
    data.cluster_spds(cluster_distance="tree", cluster_method="leiden", resolution=resolution)
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
        epsilon_block_wise_dict[int(cluster_id)] = eps_elbow_per_block
        epsilon_dict[int(cluster_id)] = eps

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

def run_benchmark_suite(queries, data, ivl_idx, search_budget, test_cases, test_iterations_per_query=5):
    benchmark_results_log = []
    
    all_spd_matrices = np.asarray(data.spd_matrices)
    all_spd_ids = np.asarray(data.spd_ids)
    sid_to_idx = {int(all_spd_ids[k]): k for k in range(len(all_spd_ids))}
    
    for case_name, case_type in test_cases:
        print(f"\n[{case_name.upper()}] Starting Interval Benchmark Suite...")
        
        for q_info in tqdm(queries, desc=f"Benchmarking {case_name}"):
            cluster_id = q_info['cluster_id']
            q_spd = q_info['q_spd']
            
            block_index = 0
            block_start, block_end = data.block_dict[cluster_id][block_index]
            block_size = block_end - block_start
            
            if block_size < 8:
                continue
                
            perm = data.perm_list[cluster_id]
            block_perm = perm[block_start:block_end]
            q_block = q_spd[np.ix_(block_perm, block_perm)]
            
            labels = np.asarray(data.labels)
            mask = (labels == cluster_id)
            spd_matrices_cluster = np.asarray(data.spd_matrices)[mask]
            spd_ids_cluster = np.asarray(data.spd_ids)[mask]
            
            for test_idx in range(test_iterations_per_query):
                if case_type == 'contiguous':
                    length = random.randint(max(4, block_size // 4), block_size)
                    a = random.randint(0, block_size - length)
                    b = a + length
                    ranges = [(a, b)]
                else:
                    split_point = random.randint(3, block_size - 4)
                    len1 = random.randint(2, split_point)
                    a1 = random.randint(0, split_point - len1)
                    b1 = a1 + len1
                    
                    len2 = random.randint(2, block_size - split_point)
                    a2 = random.randint(split_point, block_size - len2)
                    b2 = a2 + len2
                    ranges = [(a1, b1), (a2, b2)]
                
                valid_len = sum(b - a for a, b in ranges)
                if valid_len < 2:
                    continue
                
                perm_ivl = []
                for a, b in ranges:
                    perm_ivl.extend(block_perm[a:b])
                
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
                
                top100_indices = np.argsort(dists)[:100]
                true_top100_ids = set([int(all_spd_ids[i]) for i in top100_indices])
                true_top50_ids = set([int(all_spd_ids[i]) for i in top100_indices[:50]])
                true_top20_ids = set([int(all_spd_ids[i]) for i in top100_indices[:20]])
                true_top10_ids = set([int(all_spd_ids[i]) for i in top100_indices[:10]])
                true_top5_ids = set([int(all_spd_ids[i]) for i in top100_indices[:5]])
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
                        
                spindle_top100_ids = set(spindle_retrieved_ids_all[:100])
                overlap_100 = len(true_top100_ids.intersection(spindle_top100_ids)) / 100.0
                
                spindle_top50_ids = set(spindle_retrieved_ids_all[:50])
                overlap_50 = len(true_top50_ids.intersection(spindle_top50_ids)) / 50.0
                
                spindle_top20_ids = set(spindle_retrieved_ids_all[:20])
                overlap_20 = len(true_top20_ids.intersection(spindle_top20_ids)) / 20.0
                
                spindle_top10_ids = set(spindle_retrieved_ids_all[:10])
                overlap_10 = len(true_top10_ids.intersection(spindle_top10_ids)) / 10.0
                
                spindle_top5_ids = set(spindle_retrieved_ids_all[:5])
                overlap_5 = len(true_top5_ids.intersection(spindle_top5_ids)) / 5.0
                
                def hit_in_k(k):
                    k_ids = spindle_retrieved_ids_all[:k]
                    for sid in k_ids:
                        if sid in exact_partial_map and abs(exact_partial_map[sid] - true_min_dist) < 1e-5:
                            return 1
                    return 0
                    
                benchmark_results_log.append({
                    'Case': case_name,
                    'Tile': q_info['id'],
                    'Iter': test_idx,
                    'Query_Length': valid_len,
                    'recall_at_1': hit_in_k(1),
                    'recall_at_5': hit_in_k(5),
                    'recall_at_10': hit_in_k(10),
                    'recall_at_20': hit_in_k(20),
                    'overlap_at_5': overlap_5,
                    'overlap_at_10': overlap_10,
                    'overlap_at_20': overlap_20,
                    'overlap_at_50': overlap_50,
                    'overlap_at_100': overlap_100,
                    'rank': spindle_best_rank,
                    'dist_gap': (spindle_best_partial_dist - true_min_dist) if spindle_best_rank != -1 else float('inf'),
                    'spindle_time_ms': round(spindle_time_ms, 4),
                    'brute_force_time_ms': round(bf_time_ms, 4),
                    'speedup': round(bf_time_ms / spindle_time_ms, 2) if spindle_time_ms > 0 else np.nan,
                    'retrieved': len(flat_results)
                })

    return benchmark_results_log

def generate_performance_report(df, results_dir, labels):
    report_path = results_dir / "performance_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Interval Index Partial Search Performance Report\n\n")
        f.write("This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.\n\n")
        
        for case_name in df['Case'].unique():
            f.write(f"## {case_name}\n")
            case_df = df[df['Case'] == case_name]
            
            f.write(f"| Query Size | Count | Recall@1 (%) | Recall@5 (%) | Overlap@10 (%) | Overlap@20 (%) | Overlap@50 (%) | Avg Rank | Spindle (ms) | BF (ms) | Speedup (x) |\n")
            f.write(f"|:----------:|:-----:|:------------:|:------------:|:--------------:|:--------------:|:--------------:|:--------:|:------------:|:-------:|:-----------:|\n")
            
            for bin_label in labels:
                sub_df = case_df[(case_df['Length_Bin'] == bin_label) & (case_df['rank'] != -1)]
                if not sub_df.empty:
                    r1 = sub_df['recall_at_1'].mean() * 100
                    r5 = sub_df['recall_at_5'].mean() * 100
                    olap10 = sub_df['overlap_at_10'].mean() * 100
                    olap20 = sub_df['overlap_at_20'].mean() * 100
                    olap50 = sub_df['overlap_at_50'].mean() * 100
                    rank = sub_df['rank'].mean()
                    sp_tms = sub_df['spindle_time_ms'].mean()
                    bf_tms = sub_df['brute_force_time_ms'].mean()
                    speedup = bf_tms / sp_tms if sp_tms > 0 else np.nan
                    f.write(f"| {bin_label} | {len(sub_df)} | {r1:.1f} | {r5:.1f} | {olap10:.1f} | {olap20:.1f} | {olap50:.1f} | {rank:.1f} | {sp_tms:.2f} | {bf_tms:.2f} | {speedup:.2f} |\n")
            
            f.write("\n")
            
    with open(report_path, "r") as f:
        print("\n\n" + f.read())
        
    print(f"-> Full report saved to {report_path}")

def generate_visual_plots(df, results_dir, labels):
    print("Visual plots disabled (CSV-only export mode).")

def generate_overall_dataset_plots(combined_df, results_dir):
    print("Overall dataset plots disabled (CSV-only export mode).")
