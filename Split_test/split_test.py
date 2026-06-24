import sys
import time
from pathlib import Path
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import argparse
from tqdm.auto import tqdm

# Set up sys.path so that 'src/spindle_dev' is importable as 'spindle_dev'
# Use the directory of the current script to support running from any CWD
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import spindle_dev.search as search

def log_spd(M, eps=1e-6):
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T

def calc_corr(mat1, mat2, perm, block_runs):
    m1_perm = mat1[np.ix_(perm, perm)]
    m2_perm = mat2[np.ix_(perm, perm)]
    
    # Arrays to hold the concatenated block data
    v1_all = []
    v2_all = []
    
    for (start, end) in block_runs:
        b1 = m1_perm[start:end, start:end]
        b2 = m2_perm[start:end, start:end]
        
        # Get upper triangle excluding the diagonal
        iu = np.triu_indices(b1.shape[0], k=1) 
        v1_all.extend(b1[iu])
        v2_all.extend(b2[iu])
        
    v1_arr = np.array(v1_all)
    v2_arr = np.array(v2_all)
    
    if len(v1_arr) > 1 and np.std(v1_arr) > 0 and np.std(v2_arr) > 0:
        c = np.corrcoef(v1_arr, v2_arr)[0, 1]
        return c if not np.isnan(c) else 0.0
    return 0.0

def evaluate_block_diagonalized_correlation(test_tile_covs, train_tile_covs, train_idx, predicted_clusters, all_matched_train_ids, data, dataset_out_dir):
    print("\n" + "="*50)
    print("TASK 1: Block-Diagonalized Correlation Permutation Test")
    print("="*50)
    
    overall_p_values = []
    global_to_local_train_map = {global_id: local_idx for local_idx, global_id in enumerate(train_idx)}
    
    for i, q_dict in enumerate(test_tile_covs):
        q_spd = q_dict if not isinstance(q_dict, dict) else q_dict.get('cov', q_dict.get('matrix', q_dict))
        target_niche = int(predicted_clusters[i])
        
        perm = data.perm_list[target_niche]
        block_runs = data.block_dict[target_niche]

        if not all_matched_train_ids[i]:
            print(f"Query {i:3d}: Spindle found NOTHING (0 candidates).")
            continue

        true_match_corr = -float('inf')
        for match_global_idx in all_matched_train_ids[i]:
            local_match_idx = global_to_local_train_map.get(match_global_idx)
            if local_match_idx is None:
                continue

            t_spd = train_tile_covs[local_match_idx]
            t_spd = t_spd if not isinstance(t_spd, dict) else t_spd.get('cov')
            corr = calc_corr(q_spd, t_spd, perm, block_runs)
            if corr > true_match_corr:
                true_match_corr = corr

        if true_match_corr == -float('inf'):
            print(f"Query {i:3d}: Spindle found matches, but none were valid in train_idx.")
            continue
        
        random_indices = np.random.choice(len(train_tile_covs), size=100, replace=False)
        random_corrs = []
        for idx in random_indices:
            r_spd = train_tile_covs[idx]
            r_spd = r_spd if not isinstance(r_spd, dict) else r_spd.get('cov')
            random_corrs.append(calc_corr(q_spd, r_spd, perm, block_runs))
            
        avg_random_corr = np.mean(random_corrs)
        p_value = np.sum(np.array(random_corrs) >= true_match_corr) / 100.0
        overall_p_values.append(p_value)
        
        # Plot Background Distribution vs. True Match
        import matplotlib.pyplot as plt
        
        plot_dir = dataset_out_dir / "correlation_plots"
        plot_dir.mkdir(exist_ok=True, parents=True)
        
        plt.figure(figsize=(8, 5))
        plt.hist(random_corrs, bins=20, color='lightgray', edgecolor='black', alpha=0.7, label='Random Background (n=100)')
        plt.axvline(true_match_corr, color='red', linestyle='dashed', linewidth=2, label=f'Spindle Best Match ({true_match_corr:.2f})')
        plt.title(f'Query {i} (Niche {target_niche}): Block-Diagonalized Correlation\nEmpirical p-value = {p_value:.3f}')
        plt.xlabel('Pearson Correlation (Block Sum)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        
        # Save plot to disk
        plt.savefig(plot_dir / f'query_{i:03d}_corr_distribution.png', dpi=150)
        plt.close()
        
        print(f"Query {i:3d}: True Match Corr = {true_match_corr:.3f}, Avg Random Corr = {avg_random_corr:.3f}, p-value = {p_value:.3f}")
    if overall_p_values:
        mean_pval = np.mean(overall_p_values)
        print(f"\nMean p-value across queries: {mean_pval:.3f}")
        
        plt.figure(figsize=(8, 5))
        plt.hist(overall_p_values, bins=np.linspace(0, 1, 21), color='skyblue', edgecolor='black')
        plt.title(f'Distribution of Empirical p-values\nMean p-value = {mean_pval:.3f}')
        plt.xlabel('p-value')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(plot_dir / 'overall_p_value_distribution.png', dpi=150)
        plt.close()
        print(f"Saved summary p-value distribution plot to {plot_dir / 'overall_p_value_distribution.png'}")


def get_ordinal(n):
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    suffixes = ["th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th"]
    return str(n) + suffixes[n % 10]


def evaluate_brute_force_approximation(test_tile_covs, train_tile_covs, train_idx, predicted_clusters, all_matched_train_ids, data, dataset_out_dir, dataset_name):
    print("\n" + "="*60)
    print("TASK 2:Brute-Force Approximation Benchmark")
    print("="*60)
    
    global_to_local_train_map = {global_id: local_idx for local_idx, global_id in enumerate(train_idx)}
    
    exact_dists = []
    spindle_dists = []
    spindle_ranks = []
    overlap_metrics = []
    
    for i, q_dict in enumerate(test_tile_covs):
        if not all_matched_train_ids[i]:
            continue
            
        target_niche = int(predicted_clusters[i])
        
        # 1. Get the architecture for this specific niche
        perm = data.perm_list[target_niche]
        block_runs = data.block_dict[target_niche]
        
        # 2. Extract, permute, and pre-compute logs for the query blocks
        q_spd = q_dict if not isinstance(q_dict, dict) else q_dict.get('cov', q_dict.get('matrix', q_dict))
        q_perm = q_spd[np.ix_(perm, perm)]
        
        q_blocks_log = []
        for (start, end) in block_runs:
            q_blocks_log.append(log_spd(q_perm[start:end, start:end]))
            
        # 3. Restrict brute force search to ONLY the training tiles in this niche
        niche_train_indices = [idx for idx, lab in enumerate(data.labels) if int(lab) == target_niche]
        
        distances = []
        
        # 4. Calculate Block-Wise distance for the niche
        for t_idx in niche_train_indices:
            t_spd = train_tile_covs[t_idx]
            t_spd = t_spd if not isinstance(t_spd, dict) else t_spd.get('cov')
            t_perm = t_spd[np.ix_(perm, perm)]
            
            total_block_dist = 0.0
            for b_idx, (start, end) in enumerate(block_runs):
                t_block = t_perm[start:end, start:end]
                t_block_log = log_spd(t_block)
                
                # Calculate Frobenius norm and normalize by block size (sqrt(p))
                diff = q_blocks_log[b_idx] - t_block_log
                p_block = t_block.shape[0]
                total_block_dist += np.linalg.norm(diff, ord='fro') / np.sqrt(p_block)
                
            distances.append((total_block_dist, t_idx))
            
        # Sort to find the true Block-Wise nearest neighbors in this Niche
        distances.sort(key=lambda x: x[0]) 
        
        true_order = [idx for d, idx in distances]
        dist_dict = {idx: d for d, idx in distances}
        
        spindle_candidates_local = []
        for match_global_idx in all_matched_train_ids[i]:
            local_match_idx = global_to_local_train_map.get(match_global_idx)
            if local_match_idx is not None and local_match_idx in dist_dict:
                spindle_candidates_local.append(local_match_idx)
                
        spindle_candidates_with_dist = [(dist_dict[idx], idx) for idx in spindle_candidates_local]
        spindle_candidates_with_dist.sort(key=lambda x: x[0])
        spindle_ranked_local = [idx for d, idx in spindle_candidates_with_dist]
        
        overlaps = {}
        for K in [10, 20, 30, 50]:
            if len(true_order) < K:
                overlaps[f'overlap_{K}'] = float('nan')
            else:
                true_top_k = set(true_order[:K])
                spindle_top_k = set(spindle_ranked_local[:K])
                overlap = len(true_top_k.intersection(spindle_top_k)) / K
                overlaps[f'overlap_{K}'] = overlap
                
        overlap_metrics.append({
            'Query': i,
            'Dataset': dataset_name,
            **overlaps
        })
        
        # 5. Evaluate Spindle's BEST match from its search pool
        closest_dist = distances[0][0] if len(distances) > 0 else float('inf')
        
        if not all_matched_train_ids[i]:
            print(f"Query {i:3d} (Niche {target_niche}): Exact closest dist = {closest_dist:.3f} | Spindle found NOTHING.")
            exact_dists.append(closest_dist)
            spindle_dists.append(float('inf'))
            spindle_ranks.append(-1)
            continue

        spindle_best_dist = float('inf')
        spindle_best_rank = -1
        
        for match_global_idx in all_matched_train_ids[i]:
            local_match_idx = global_to_local_train_map.get(match_global_idx)
            if local_match_idx is None:
                continue
                
            match_rank = -1
            match_dist = -1
            for r, (d, idx) in enumerate(distances):
                if idx == local_match_idx:
                    match_rank = r + 1
                    match_dist = d
                    break
                    
            if match_dist != -1 and match_dist < spindle_best_dist:
                spindle_best_dist = match_dist
                spindle_best_rank = match_rank
                
        if spindle_best_rank == -1:
            print(f"Query {i:3d} (Niche {target_niche}): Exact closest dist = {closest_dist:.3f} | Spindle found NOTHING in this niche.")
            exact_dists.append(closest_dist)
            spindle_dists.append(float('inf'))
            spindle_ranks.append(-1)
        else:
            print(f"Query {i:3d} (Niche {target_niche}): Exact closest dist = {closest_dist:.3f}, Spindle dist = {spindle_best_dist:.3f} | Spindle found {get_ordinal(spindle_best_rank)} closest neighbor.")
            exact_dists.append(closest_dist)
            spindle_dists.append(spindle_best_dist)
            spindle_ranks.append(spindle_best_rank)
            
    import matplotlib.pyplot as plt
    from pathlib import Path
    
    plot_dir = dataset_out_dir / "performance_plots"
    plot_dir.mkdir(exist_ok=True, parents=True)
    
    # Plot 1: Spindle Match Ranks
    plt.figure(figsize=(8, 5))
    ranks = np.array(spindle_ranks)
    valid_ranks = ranks[ranks != -1]
    
    bins = [1, 2, 3, 4, 6, 11, 21, 51, 100]
    labels = ['1st', '2nd', '3rd', '4-5th', '6-10th', '11-20th', '21-50th', '51-99th']
    counts = []
    
    for k in range(len(bins)-1):
        counts.append(np.sum((valid_ranks >= bins[k]) & (valid_ranks < bins[k+1])))
    
    counts.append(np.sum(valid_ranks >= bins[-1]))
    labels.append('>=100')
    
    counts.append(np.sum(ranks == -1))
    labels.append('Not Found')
    
    plt.bar(labels, counts, color='lightgreen', edgecolor='black')
    plt.title('Spindle Match Rank Distribution')
    plt.ylabel('Number of Queries')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / 'spindle_match_ranks.png', dpi=150)
    plt.close()
    
    # Plot 2: Exact vs Spindle Distances
    valid_mask = ranks != -1
    if np.sum(valid_mask) > 0:
        v_exact = np.array(exact_dists)[valid_mask]
        v_spindle = np.array(spindle_dists)[valid_mask]
        
        plt.figure(figsize=(6, 6))
        plt.scatter(v_exact, v_spindle, alpha=0.6, c='blue', edgecolors='w', s=50)
        
        min_val = min(np.min(v_exact), np.min(v_spindle))
        max_val = max(np.max(v_exact), np.max(v_spindle))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (Perfect Match)')
        
        plt.title('Exact Distance vs Spindle Distance')
        plt.xlabel('Exact Distance to 1st NN')
        plt.ylabel('Spindle Distance to Best Match')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / 'distance_scatter.png', dpi=150)
        plt.close()
        
    print(f"\nSaved performance summary plots to {plot_dir}")

    import pandas as pd
    df_overlaps = pd.DataFrame(overlap_metrics)
    
    if not df_overlaps.empty:
        # Plot overlaps
        plt.figure(figsize=(8, 5))
        means = []
        labels_k = ['overlap_10', 'overlap_20', 'overlap_30', 'overlap_50']
        for k in labels_k:
            means.append(df_overlaps[k].mean() * 100)
            
        plt.bar(['10', '20', '30', '50'], means, color='coral', edgecolor='black')
        plt.title('Neighborhood Overlap@K')
        plt.xlabel('K')
        plt.ylabel('Overlap (%)')
        plt.ylim(0, 105)
        plt.tight_layout()
        plt.savefig(plot_dir / 'overlap_at_k.png', dpi=150)
        plt.close()
        
    return df_overlaps


def extract_query_matrices(test_tile_covs):
    query_matrices = []
    for q_dict in test_tile_covs:
        if isinstance(q_dict, dict):
            query_matrices.append(q_dict.get('cov', q_dict.get('matrix', q_dict)))
        else:
            query_matrices.append(q_dict)
    return query_matrices


def perform_search(query_matrices, data, dag_dict, config, budget_multiplier=0.75):
    search_cfg = search.SearchConfig(max_results=None, debug=False, max_failed_starts=20, max_failed_paths=50, total_paths_limit=5000)

    print(f"Starting blind holdout validation for {len(query_matrices)} unseen queries...")
    print("-" * 65)

    print("Step 1/2: Assigning queries to Covariance-Niches using latent space...")
    assign_start = time.time()
    predicted_clusters = search.assign_clusters_to_new_spds(query_matrices, data)
    print(f"Assignment complete in {time.time() - assign_start:.3f}s\n")

    print("Step 2/2: Performing distance-budgeted search across DAG...")
    search_start = time.time()
    all_matched_train_ids = []

    for j, cluster_id in enumerate(tqdm(predicted_clusters, desc="Querying Index", leave=True)):
        cluster_id = int(cluster_id)
        index_handle = dag_dict[cluster_id]
        epsilon = config.epsilon_dict[cluster_id]
        num_blocks = len(index_handle.sorted_blocks)
        
        # Budget computation
        budget = float(epsilon) * float(num_blocks) * float(budget_multiplier)

        q_spd = query_matrices[j]
        perm = data.perm_list[cluster_id]
        q_spd_perm = q_spd[np.ix_(perm, perm)]
        query_block_runs = data.block_dict[cluster_id]

        results = search.search_index(
            index_handle,
            q_spd_perm,
            [],
            query_block_runs,
            budget,
            config=search_cfg,
        )

        matched_ids_for_query = []
        if results.paths:
            for rank, path in enumerate(results.paths, start=1):
                member_sets = []
                for node_id in path.node_path:
                    node = index_handle.nodes[node_id]
                    members = getattr(getattr(node, "metadata", None), "members", [])
                    spd_ids = {int(spd_id) for spd_id, _ in members}
                    member_sets.append(spd_ids)

                intersect_ids = set.intersection(*member_sets) if member_sets else set()
                matched_ids_for_query.extend(sorted(intersect_ids))
            
        all_matched_train_ids.append(matched_ids_for_query)

    search_time = time.time() - search_start
    print("-" * 65)
    print(f"Index Querying Complete! Total time: {search_time:.3f}s ({search_time/len(predicted_clusters):.4f}s per query)")
    
    return predicted_clusters, all_matched_train_ids


def summarize_hits(all_matched_train_ids, predicted_clusters):
    print("\n" + "="*40)
    print("           QUERY HITS SUMMARY")
    print("="*40)
    for j, hits in enumerate(all_matched_train_ids):
        target_niche = int(predicted_clusters[j])
        print(f"Query {j:3d} [Niche {target_niche:2d}]: {len(hits):4d} matches found")
    print("="*40 + "\n")


def generate_overall_dataset_plots(combined_df, results_dir):
    print("\nGenerating overall dataset comparison plots...")
    import warnings
    warnings.filterwarnings("ignore")
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.3)
    
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    metrics = ['overlap_10', 'overlap_20', 'overlap_30', 'overlap_50']
    
    for idx, metric in enumerate(metrics):
        combined_df[f'{metric} (%)'] = combined_df[metric] * 100
        sns.barplot(data=combined_df, x='Dataset', y=f'{metric} (%)', ax=axes[idx], palette="muted", errorbar=None)
        axes[idx].set_title(f"Average {metric.replace('_', '@')} per Dataset", fontweight='bold')
        axes[idx].set_ylim(0, 105)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].set_xlabel("")
        axes[idx].set_ylabel("Overlap (%)")
        
    sns.despine(fig=fig)
    plt.tight_layout()
    plot_path = results_dir / "overall_overlap_performance.png"
    plt.savefig(str(plot_path), dpi=300, bbox_inches='tight')
    print(f"Saved overall benchmark plots to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Run split test on saved indexed datasets.")
    parser.add_argument('--test', action='store_true', help='Run a quick test (not used, purely for arg compat)')
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    base_indexed_dir = project_root / "results" / "split_test_indexed"
    base_results_dir = project_root / "results" / "split_test"
    base_results_dir.mkdir(exist_ok=True, parents=True)

    all_overlap_dfs = []

    if not base_indexed_dir.exists():
        print(f"Directory {base_indexed_dir} not found. Please run index_datasets.py first.")
        return

    indexed_files = list(base_indexed_dir.glob("*_indexed.pkl"))
    if not indexed_files:
        print(f"No indexed files found in {base_indexed_dir}. Please run index_datasets.py first.")
        return

    for indexed_file in indexed_files:
        print(f"\n{'='*80}")
        print(f"Processing indexed file: {indexed_file.name}")
        print(f"{'='*80}\n")
        
        print("Loading data...")
        with open(indexed_file, 'rb') as f:
            saved_data = pickle.load(f)
            
        test_tile_covs = saved_data['test_tile_covs']
        train_tile_covs = saved_data['train_tile_covs']
        train_idx = saved_data['train_idx']
        test_idx = saved_data['test_idx']
        data = saved_data['data']
        dag_dict = saved_data['dag_dict']
        config = saved_data['config']
        dataset_name = saved_data['dataset_name']

        dataset_out_dir = base_results_dir / dataset_name
        dataset_out_dir.mkdir(exist_ok=True, parents=True)

        # 4. Extract queries
        query_matrices = extract_query_matrices(test_tile_covs)

        # 5. Perform search
        predicted_clusters, all_matched_train_ids = perform_search(query_matrices, data, dag_dict, config)

        # 6. Summarize hits
        summarize_hits(all_matched_train_ids, predicted_clusters)

        # 7. Evaluate
        # evaluate_block_diagonalized_correlation(test_tile_covs, train_tile_covs, train_idx, predicted_clusters, all_matched_train_ids, data, dataset_out_dir)
        df_overlaps = evaluate_brute_force_approximation(test_tile_covs, train_tile_covs, train_idx, predicted_clusters, all_matched_train_ids, data, dataset_out_dir, dataset_name)
        
        if df_overlaps is not None and not df_overlaps.empty:
            all_overlap_dfs.append(df_overlaps)

    if all_overlap_dfs:
        import pandas as pd
        combined_overlaps = pd.concat(all_overlap_dfs, ignore_index=True)
        generate_overall_dataset_plots(combined_overlaps, base_results_dir)

if __name__ == "__main__":
    main()
