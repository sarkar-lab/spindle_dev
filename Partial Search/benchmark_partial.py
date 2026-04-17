import sys
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

# Setup paths to ensure we can import spindle_dev and the previous tests
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'
split_test_path = project_root / 'Split_test'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(split_test_path) not in sys.path:
    sys.path.insert(0, str(split_test_path))

import spindle_dev.search as search
import hbreast_wo_unlabeled_03172026 as hbreast
from spindle_dev.partial_search import pad_query_identity, impute_query_niche_mean

def calc_corr_partial(mat1: np.ndarray, mat2: np.ndarray, perm: list, block_runs: list, valid_indices: list) -> float:
    """Biological Preservation test: Calculates block-diagonalized correlation strictly only for the overlapping genes."""
    valid_set = set(valid_indices)
    v1_all = []
    v2_all = []
    
    for (start, end) in block_runs:
        block_genes_orig = [perm[i] for i in range(start, end)]
        valid_block_genes = [g for g in block_genes_orig if g in valid_set]
        
        if len(valid_block_genes) < 2:
            continue
            
        m1_block = mat1[np.ix_(valid_block_genes, valid_block_genes)]
        m2_block = mat2[np.ix_(valid_block_genes, valid_block_genes)]
        
        iu = np.triu_indices(m1_block.shape[0], k=1)
        v1_all.extend(m1_block[iu])
        v2_all.extend(m2_block[iu])
        
    v1_arr = np.array(v1_all)
    v2_arr = np.array(v2_all)
    
    if len(v1_arr) > 1 and np.std(v1_arr) > 0 and np.std(v2_arr) > 0:
        c = np.corrcoef(v1_arr, v2_arr)[0, 1]
        return float(c) if not np.isnan(c) else 0.0
    return 0.0

def degrade_query(q_spd: np.ndarray, degradation_type: str, block_runs: list, perm: list):
    """Simulates real-world partial sequencing drop-outs."""
    p = q_spd.shape[0]
    valid_indices = list(range(p))
    
    if degradation_type.startswith('scatter_'):
        ratio = float(degradation_type.split('_')[1]) / 100.0
        drop_count = int(p * ratio)
        drop_idx = np.random.choice(p, drop_count, replace=False)
        valid_indices = [i for i in range(p) if i not in drop_idx]
        
    elif degradation_type == 'catastrophe':
        drop_count = int(p * 0.50)
        drop_idx = np.random.choice(p, drop_count, replace=False)
        valid_indices = [i for i in range(p) if i not in drop_idx]
        
    elif degradation_type.startswith('module_'):
        blk_idx = int(degradation_type.split('_')[1])
        if block_runs and blk_idx < len(block_runs):
            start, end = block_runs[blk_idx]
            dropped_genes = set([perm[i] for i in range(start, end)])
            valid_indices = [i for i in range(p) if i not in dropped_genes]
            
    q_degraded = q_spd[np.ix_(valid_indices, valid_indices)]
    return sorted(valid_indices), q_degraded

def find_pristine_ground_truth(test_tile_covs, data, dag_dict, config, num_queries=100):
    print(f"Finding up to {num_queries} pristine queries to establish Ground Truth...")
    pristine_queries = []
    
    query_matrices = hbreast.extract_query_matrices(test_tile_covs)
    predicted_clusters = search.assign_clusters_to_new_spds(query_matrices, data)
    
    search_cfg = search.SearchConfig(max_results=30, max_failed_starts=20, max_failed_paths=100, total_paths_limit=5000)
    
    for j, (q_dict, cluster_id) in enumerate(zip(test_tile_covs, predicted_clusters)):
        if len(pristine_queries) >= num_queries:
            break
            
        cluster_id = int(cluster_id)
        index_handle = dag_dict[cluster_id]
        epsilon = config.epsilon_dict[cluster_id]
        num_blocks = len(index_handle.sorted_blocks)
        target_budget = float(epsilon) * float(num_blocks) * 2.0
        
        q_spd = query_matrices[j]
        perm = data.perm_list[cluster_id]
        q_spd_perm = q_spd[np.ix_(perm, perm)]
        query_block_runs = data.block_dict[cluster_id]
        
        results = search.search_index(
            index_handle, q_spd_perm, [], query_block_runs, target_budget, config=search_cfg
        )
        
        if results.paths:
            top_path = results.paths[0]
            member_sets = []
            for node_id in top_path.node_path:
                node = index_handle.nodes[node_id]
                members = getattr(getattr(node, "metadata", None), "members", [])
                spd_ids_in_node = {int(spd_id) for spd_id, _ in members}
                member_sets.append(spd_ids_in_node)
            top_spd_ids = set.intersection(*member_sets) if member_sets else set()
            
            pristine_queries.append({
                'original_idx': j,
                'q_spd': q_spd,             
                'cluster_id': cluster_id,    
                'gt_spd_ids': top_spd_ids,   
                'gt_path_dist': top_path.total_distance
            })
    print(f"-> Secured {len(pristine_queries)} pristine success queries.")
    return pristine_queries

def run_benchmark(pristine_queries, data, dag_dict, config, train_tile_covs, train_idx, degradation_configs):
    global_to_local_train_map = {global_id: local_idx for local_idx, global_id in enumerate(train_idx)}
    benchmark_results = {}
    methods = ['Padding', 'Imputation']
    
    print("Pre-calculating Niche Means for Imputation Strategy...")
    niche_means = {}
    for cluster_id in set(data.labels):
        cluster_id = int(cluster_id)
        indices = [i for i, lab in enumerate(data.labels) if int(lab) == cluster_id]
        if indices:
            covs = [train_tile_covs[i] for i in indices]
            covs = [c['cov'] if isinstance(c, dict) else c for c in covs]
            niche_means[cluster_id] = np.mean(covs, axis=0)

    # Pre-extract all training covariance matrices into a 3D NumPy array for fast vectorized slicing
    print("Pre-loading training matrices for fast brute-force calculation...")
    t_covs_array = np.array([t['cov'] if isinstance(t, dict) else t for t in train_tile_covs])
            
    for degrad_type in degradation_configs:
        benchmark_results[degrad_type] = {m: [] for m in methods}
        print(f"\n[{degrad_type.upper()}] Starting Benchmark Suite...")
        
        for q_info in tqdm(pristine_queries, desc=f"Benchmarking {degrad_type}", leave=False):
            original_cluster = q_info['cluster_id']
            full_spd = q_info['q_spd']
            p = full_spd.shape[0]
            
            pristine_block_runs = data.block_dict[original_cluster]
            pristine_perm = data.perm_list[original_cluster]
            
            # 1. Degrade the Query
            valid_indices, q_degraded = degrade_query(full_spd, degrad_type, pristine_block_runs, pristine_perm)
            
            # =========================================================
            # VECTORIZED BRUTE FORCE: Mathematical Minimum in Partial Space
            # =========================================================
            q_log = hbreast.log_spd(q_degraded)
            p_valid = len(valid_indices)
            
            # Slice all 2000 training matrices down to the valid genes instantly
            t_degraded_array = t_covs_array[:, valid_indices, :][:, :, valid_indices]
            
            # Compute log_spd for the sliced training matrices
            t_logs = np.array([hbreast.log_spd(mat) for mat in t_degraded_array])
            
            # Vectorized Frobenius norm across all matrices
            diffs = t_logs - q_log
            dists = np.linalg.norm(diffs, axis=(1, 2)) / np.sqrt(p_valid)
            
            # Sort to find absolute closest partial neighbors
            exact_partial_dists = sorted([(d, i) for i, d in enumerate(dists)])
            true_min_dist = exact_partial_dists[0][0]
            # =========================================================
            
            for method in methods:
                start_time = time.time()
                
                # We always route Phase 1 using basic padding
                q_routed_pad = pad_query_identity(q_degraded, valid_indices, p)
                predicted_cluster = int(search.assign_clusters_to_new_spds([q_routed_pad], data)[0])
                
                index_handle = dag_dict[predicted_cluster]
                epsilon = config.epsilon_dict[predicted_cluster]
                num_blocks = len(index_handle.sorted_blocks)
                
                search_cfg = search.SearchConfig(max_results=5, max_failed_starts=20, max_failed_paths=500, total_paths_limit=5000)
                base_budget = float(epsilon) * float(num_blocks)
                perm = data.perm_list[predicted_cluster]
                
                if method == 'Padding':
                    q_final = pad_query_identity(q_degraded, valid_indices, p)
                    budget = base_budget * 2.5  
                elif method == 'Imputation':
                    mean_spd = niche_means.get(predicted_cluster, np.eye(p))
                    q_final = impute_query_niche_mean(q_degraded, valid_indices, mean_spd)
                    budget = base_budget * 1.5  
                    
                q_perm = q_final[np.ix_(perm, perm)]
                active_runs = data.block_dict[predicted_cluster]
                    
                # Perform the DAG search inside index 
                if active_runs and budget > 0:
                    results = search.search_index(
                        index_handle, q_perm, [], active_runs, budget, config=search_cfg
                    )
                else:
                    results = search.SearchResults(paths=[])
                    
                search_time = time.time() - start_time
                
                matched_ids = set()
                paths_explored = len(results.paths)
                
                if results.paths:
                    for path in results.paths:
                        member_sets = []
                        for node_id in path.node_path:
                            node = index_handle.nodes[node_id]
                            members = getattr(getattr(node, "metadata", None), "members", [])
                            spd_ids_in_node = {int(spd_id) for spd_id, _ in members}
                            member_sets.append(spd_ids_in_node)
                        intersect_ids = set.intersection(*member_sets) if member_sets else set()
                        matched_ids.update(intersect_ids)
                        
                # =========================================================
                # EVALUATE EXACT RANK AND DISTANCE GAP
                # =========================================================
                spindle_best_rank = -1
                spindle_best_partial_dist = float('inf')
                
                for global_id in matched_ids:
                    local_match_idx = global_to_local_train_map.get(global_id)
                    if local_match_idx is not None:
                        # Find this match's rank and exact distance in the brute-force list
                        for r, (d, tr_idx) in enumerate(exact_partial_dists):
                            if tr_idx == local_match_idx:
                                if d < spindle_best_partial_dist:
                                    spindle_best_partial_dist = d
                                    spindle_best_rank = r + 1
                                break
                
                dist_gap = (spindle_best_partial_dist - true_min_dist) if spindle_best_rank != -1 else float('inf')
                exact_hit = 1 if spindle_best_rank == 1 else 0
                
                # Metric 2: Preserved Biological Niche Correlation
                best_corr = -1.0 # Set to -1.0 instead of 0 to capture truly bad correlations
                eval_block_runs = data.block_dict[original_cluster]
                eval_perm = data.perm_list[original_cluster]
                
                for global_id in matched_ids:
                    local_match_idx = global_to_local_train_map.get(global_id)
                    if local_match_idx is not None:
                        t_spd = train_tile_covs[local_match_idx]
                        t_spd = t_spd if not isinstance(t_spd, dict) else t_spd.get('cov')
                        c = calc_corr_partial(full_spd, t_spd, eval_perm, eval_block_runs, valid_indices) 
                        if c > best_corr:
                            best_corr = c
                
                benchmark_results[degrad_type][method].append({
                    'exact_hit': exact_hit,
                    'rank': spindle_best_rank,
                    'dist_gap': dist_gap,
                    'corr': best_corr,
                    'time': search_time,
                    'paths_returned': paths_explored
                })

    print("\n" + "="*95)
    print("                      ALGORITHMIC PARTIAL SEARCH BENCHMARK SUMMARY                     ")
    print("="*95)
    print(f"{'Degradation':<15} | {'Method':<12} | {'Exact Top-1':<12} | {'Avg Rank':<10} | {'Avg Dist Gap':<12} | {'Time (ms)':<10}")
    print("-" * 95)
    
    for degrad_type in degradation_configs:
        for method in methods:
            metrics_list = benchmark_results[degrad_type][method]
            if not metrics_list: continue
            
            # Filter to only queries where a match was found for averaging
            valid_results = [x for x in metrics_list if x['rank'] != -1]
            found_pct = (len(valid_results) / len(metrics_list)) * 100
            
            if valid_results:
                pct_exact = np.mean([x['exact_hit'] for x in valid_results]) * 100
                avg_rank = np.mean([x['rank'] for x in valid_results])
                avg_gap = np.mean([x['dist_gap'] for x in valid_results])
                avg_time = np.mean([x['time'] for x in valid_results]) * 1000 
                
                print(f"{degrad_type:<15} | {method:<12} | {pct_exact:>5.1f}% ({found_pct:3.0f}%) | {avg_rank:<10.1f} | +{avg_gap:<11.4f} | {avg_time:<10.1f}")
            else:
                print(f"{degrad_type:<15} | {method:<12} | FAILED TO FIND ANY MATCHES")
        print("-" * 95)


def main():
    np.random.seed(42)
    adata_path = project_root / "dataset" / "adata.h5ad"
    
    print("Preparing the Index Base Dataset...")
    adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = hbreast.load_and_split_data(adata_path)
    
    print("\nBuilding Index DAGs (This acts as the target for our degraded queries)...")
    data, out_dict = hbreast.run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)
    dag_dict, config = hbreast.configure_and_build_dag(data)
    
    pristine_queries = find_pristine_ground_truth(test_tile_covs, data, dag_dict, config, num_queries=50)
    
    if not pristine_queries:
        print("Fatal: Could not find perfectly matching pristine queries to test degrading against.")
        return
        
    print("\nExecuting Controlled Degradation Experiment...")
    degradation_configs = ['scatter_5', 'scatter_20', 'module_1', 'catastrophe']
    
    run_benchmark(pristine_queries, data, dag_dict, config, train_tile_covs, train_idx, degradation_configs)

if __name__ == "__main__":
    main()