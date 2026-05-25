import sys
import time
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

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
import spindle_dev.interval_index as interval_index

def search_disjoint_intervals(interval_index_obj, cluster_id, block_index, disjoint_intervals, q_block, top_k=20):
    pieces = []
    for a, b in disjoint_intervals:
        pieces.extend(interval_index.decompose_to_dyadic(a, b))
        
    piece_scores = []
    piece_scales = []
    for pa, pb in pieces:
        q_sub = q_block[pa:pb, pa:pb]
        p = pb - pa
        scale = np.sqrt(p) if p > 1 else 1.0
        piece_results = interval_index.query_interval_index(
            interval_index_obj, cluster_id, block_index, (pa, pb), q_sub, top_k=None
        )
        if not piece_results:
            return []
        scores = {}
        for dist, members in piece_results:
            for m in members:
                scores[m] = dist
        piece_scores.append(scores)
        piece_scales.append(scale)
        
    if not piece_scores:
        return []
        
    common = set(piece_scores[0].keys())
    for ps in piece_scores[1:]:
        common &= ps.keys()
        
    if not common:
        return []
        
    valid_len = sum((pb - pa) for pa, pb in pieces)
    total_scale = np.sqrt(valid_len) if valid_len > 1 else 1.0
        
    ranked = []
    for tid in common:
        total_sq = 0.0
        for i, ps in enumerate(piece_scores):
            unscaled = ps[tid] * piece_scales[i]
            total_sq += unscaled ** 2
        combined_dist = np.sqrt(total_sq) / total_scale
        ranked.append((combined_dist, tid))
        
    ranked = sorted(ranked, key=lambda x: x[0])
    return [(dist, [tid]) for dist, tid in ranked[:top_k]]


def main():
    np.random.seed(42)
    random.seed(42)
    adata_path = project_root.parent / "dataset" / "2023_xenium_human_breast_cancer" / "adata.h5ad"
    
    print("Preparing the Index Base Dataset...")
    adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = hbreast.load_and_split_data(adata_path)
    
    print("\nBuilding Standard Index Data...")
    data, out_dict = hbreast.run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)
    
    dag_dict, config = hbreast.configure_and_build_dag(data)
    
    print("\nBuilding Dyadic Interval Index...")
    config.use_interval_index = True
    config.interval_mode = "dyadic"
    config.interval_max_iters = 5
    
    ivl_idx = interval_index.build_all_interval_indices(data, config)
    
    print("\nExtracting pristine test queries (unseen tiles)...")
    query_matrices = hbreast.extract_query_matrices(test_tile_covs)
    predicted_clusters = search.assign_clusters_to_new_spds(query_matrices, data)
    
    queries = []
    for j, (q_spd, cluster_id) in enumerate(zip(query_matrices, predicted_clusters)):
        if len(queries) >= 100:
            break
        cluster_id = int(cluster_id)
        block_runs = data.block_dict.get(cluster_id, [])
        if len(block_runs) > 0:
            queries.append({
                'id': j,
                'q_spd': q_spd,
                'cluster_id': cluster_id
            })
            
    print(f"-> Secured {len(queries)} valid test queries.")
    
    test_cases = [
        ('Contiguous Random', 'contiguous'),
        ('Non-Contiguous Random', 'non_contiguous')
    ]
    
    benchmark_results_log = []
    top100_dists_log = []
    search_budget = 2000
    top_k = 100
    test_iterations_per_query = 5
    
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
                q_sub = q_spd[np.ix_(perm_ivl, perm_ivl)]
                q_log = interval_index._log_spd(q_sub)
                scale = np.sqrt(valid_len) if valid_len > 1 else 1.0
                
                t_subs = spd_matrices_cluster[:, perm_ivl, :][:, :, perm_ivl]
                t_logs = np.array([interval_index._log_spd(t_sub) for t_sub in t_subs])
                
                diffs = t_logs - q_log
                dists = np.linalg.norm(diffs, axis=(1, 2)) / scale
                true_min_dist = np.min(dists)
                
                rounded_dists = np.round(dists, 5)
                unique_dists = np.sort(np.unique(rounded_dists))
                exact_partial_map = {int(spd_ids_cluster[i]): rounded_dists[i] for i in range(len(dists))}
                
                top100_indices = np.argsort(dists)[:100]
                true_top100_ids = set([int(spd_ids_cluster[i]) for i in top100_indices])
                true_top50_ids = set([int(spd_ids_cluster[i]) for i in top100_indices[:50]])
                true_top20_ids = set([int(spd_ids_cluster[i]) for i in top100_indices[:20]])
                true_top10_ids = set([int(spd_ids_cluster[i]) for i in top100_indices[:10]])
                true_top5_ids = set([int(spd_ids_cluster[i]) for i in top100_indices[:5]])
                
                true_top100_dists = np.sort(dists)[:100]
                top100_dists_log.extend([{
                    'Case': case_name,
                    'Query_Length': valid_len,
                    'Distance': float(d)
                } for d in true_top100_dists])
            
                # =========================================================
                # SPINDLE INTERVAL SEARCH
                # =========================================================
                start_time = time.time()
                results = search_disjoint_intervals(
                    ivl_idx, cluster_id, block_index, ranges, q_block, top_k=search_budget
                )
                
                flat_results = []
                for err, ids in results:
                    for i in ids:
                        flat_results.append((err, i))
                        
                spindle_retrieved_ids_all = [sid for _, sid in flat_results]
                
                # Exact Re-ranking
                spindle_retrieved_ids_all = sorted(
                    spindle_retrieved_ids_all, 
                    key=lambda sid: exact_partial_map.get(sid, float('inf'))
                )
                search_time = time.time() - start_time
                
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
                    'hit_top_1': hit_in_k(1),
                    'hit_top_5': hit_in_k(5),
                    'hit_top_10': hit_in_k(10),
                    'hit_top_20': hit_in_k(20),
                    'overlap_5': overlap_5,
                    'overlap_10': overlap_10,
                    'overlap_20': overlap_20,
                    'overlap_50': overlap_50,
                    'overlap_100': overlap_100,
                    'rank': spindle_best_rank,
                    'dist_gap': (spindle_best_partial_dist - true_min_dist) if spindle_best_rank != -1 else float('inf'),
                    'time_ms': search_time * 1000,
                    'retrieved': len(flat_results)
                })

    df = pd.DataFrame(benchmark_results_log)
    df.to_csv(current_dir / "benchmark_interval_metrics.csv", index=False)
    
    # ---------------------------------------------------------
    # Performance Bins and File Writing
    # ---------------------------------------------------------
    bins = [0, 6, 12, 16, 1000]
    labels = ['<=6 genes', '7-12 genes', '13-16 genes', '>16 genes']
    df['Length_Bin'] = pd.cut(df['Query_Length'], bins=bins, labels=labels)
    
    report_path = current_dir / "performance_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Interval Index Partial Search Performance Report\n\n")
        f.write("This report details benchmark retrievals using independent dyadic intersections for contiguous and non-contiguous intervals decoupled by sequence query lengths.\n\n")
        
        for case_name in df['Case'].unique():
            f.write(f"## {case_name}\n")
            case_df = df[df['Case'] == case_name]
            
            f.write(f"| Query Size | Count | Top-1 (%) | Top-5 (%) | Top-10 (%) | Top-20 (%) | Overlap-5 (%) | Overlap-10 (%) | Overlap-20 (%) | Overlap-50 (%) | Overlap-100 (%) | Avg Rank | Time (ms) |\n")
            f.write(f"|:----------:|:-----:|:---------:|:---------:|:----------:|:----------:|:-------------:|:--------------:|:--------------:|:--------------:|:---------------:|:--------:|:---------:|\n")
            
            for bin_label in labels:
                sub_df = case_df[(case_df['Length_Bin'] == bin_label) & (case_df['rank'] != -1)]
                if not sub_df.empty:
                    top1 = sub_df['hit_top_1'].mean() * 100
                    top5 = sub_df['hit_top_5'].mean() * 100
                    top10 = sub_df['hit_top_10'].mean() * 100
                    top20 = sub_df['hit_top_20'].mean() * 100
                    olap5 = sub_df['overlap_5'].mean() * 100
                    olap10 = sub_df['overlap_10'].mean() * 100
                    olap20 = sub_df['overlap_20'].mean() * 100
                    olap50 = sub_df['overlap_50'].mean() * 100
                    olap100 = sub_df['overlap_100'].mean() * 100
                    rank = sub_df['rank'].mean()
                    tms = sub_df['time_ms'].mean()
                    f.write(f"| {bin_label} | {len(sub_df)} | {top1:.1f} | {top5:.1f} | {top10:.1f} | {top20:.1f} | {olap5:.1f} | {olap10:.1f} | {olap20:.1f} | {olap50:.1f} | {olap100:.1f} | {rank:.1f} | {tms:.2f} |\n")
            
            f.write("\n")
            
    # Also print to terminal
    with open(report_path, "r") as f:
        print("\n\n" + f.read())
        
    print(f"-> Full report saved to {report_path}")
    
    # ---------------------------------------------------------
    # Visual Plots
    # ---------------------------------------------------------
    print("\nGenerating visual report plots...")
    import warnings
    warnings.filterwarnings("ignore")
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Top-20 Hit Rate by Query Length
    hit_data = []
    for case_name in df['Case'].unique():
        for bin_label in labels:
            sub = df[(df['Case'] == case_name) & (df['Length_Bin'] == bin_label)]
            if not sub.empty:
                hit_data.append({
                    'Case': case_name,
                    'Query Size': bin_label,
                    'Top-20 Hit Rate (%)': sub['hit_top_20'].mean() * 100
                })
                
    if hit_data:
        hit_df = pd.DataFrame(hit_data)
        sns.barplot(data=hit_df, x='Query Size', y='Top-20 Hit Rate (%)', hue='Case', ax=axes[0])
        axes[0].set_title("Top-20 Hit Rate by Sequence Length", fontsize=14, fontweight='bold')
        axes[0].set_ylim(0, 105)

    # 2. KDE Distribution of Ground Truth Ranks
    valid_ranks_df = df[df['rank'] != -1]
    if not valid_ranks_df.empty:
         sns.boxplot(data=valid_ranks_df, x='Length_Bin', y='rank', hue='Case', ax=axes[1], fliersize=2)
         axes[1].set_title("Distribution of Mathematical Optimality Ranks", fontsize=14, fontweight='bold')
         axes[1].set_xlabel("Query Size")
         axes[1].set_ylabel("Spindle Rank in Ground Truth (Lower is Better)")
         axes[1].set_yscale('log')
         
    plt.tight_layout()
    plot_path = current_dir / "interval_benchmark_plots.png"
    plt.savefig(str(plot_path), dpi=300)
    print(f"Saved benchmark plots to {plot_path}")
    
    # 3. Distribution of Distances for Top-100 Closest Results
    top100_df = pd.DataFrame(top100_dists_log)
    top100_df['Length_Bin'] = pd.cut(top100_df['Query_Length'], bins=bins, labels=labels)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if not top100_df.empty:
        sns.violinplot(data=top100_df, x='Length_Bin', y='Distance', hue='Case', ax=ax2, inner='quartile')
    ax2.set_title("True Distance Distribution for Top-100 Closest Results by Query Size", fontsize=14, fontweight='bold')
    plt.tight_layout()
    dist_plot_path = current_dir / "interval_distribution_plot.png"
    fig2.savefig(str(dist_plot_path), dpi=300)
    print(f"Saved distance distribution plots to {dist_plot_path}")
    
    # Specific 16-gene plot if requested
    query_16_df = top100_df[top100_df['Query_Length'] == 16]
    if not query_16_df.empty:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        sns.histplot(data=query_16_df, x='Distance', hue='Case', element='step', common_norm=False, ax=ax3)
        ax3.set_title("Distance Distribution of Top 100 Neighbors (Exactly 16-Gene Queries)", fontsize=14, fontweight='bold')
        fig3.tight_layout()
        dist_16_path = current_dir / "distance_dist_16_genes.png"
        fig3.savefig(str(dist_16_path), dpi=300)
        print(f"Saved 16-gene specific distance plot to {dist_16_path}")

if __name__ == "__main__":
    main()
