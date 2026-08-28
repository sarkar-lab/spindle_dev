import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

_this_dir = Path(__file__).resolve().parent
sys.path.append(str(_this_dir.parent / "src"))
from spindle_dev.preprocessing import prepare_multiple_adatas
# IndexConfig is imported from spindle_dev.index (not spindle_dev.typing) — this is the
# dataclass used by index_spds() and must match the one stored in the saved index files.
from spindle_dev.index import ProcessedData, index_spds, IndexConfig
from spindle_dev.search import search_index, SearchConfig, assign_clusters_to_new_spds

# Shared SPD math utilities — import from the existing spindle_dev.utils module.
from spindle_dev.utils import log_spd, exp_spd

def evaluate_brute_force_approximation(query_covs, corrected_queries_by_id, train_covs, assigned_labels, all_matched_train_ids, data, direction="x2v"):  # noqa: E501
    import csv
    print("\n" + "=" * 60)
    print("TASK 2: Brute-Force Approximation Benchmark")
    print("=" * 60)
    
    exact_dists = []
    spindle_dists = []
    spindle_ranks = []
    results_for_plot = []
    metrics_records = []
    
    for i, q_spd in enumerate(query_covs):
        target_niche = int(assigned_labels[i])
        perm = data.perm_list[target_niche]
        block_runs = data.block_dict[target_niche]
        q_corr_blocks_log = corrected_queries_by_id.get(i, [])
        if not q_corr_blocks_log: continue
        
        # NOTE: Ground-truth "exact" nearest neighbor is restricted to the
        # predicted niche only (not a global brute-force across all train tiles).
        # If the query is mis-assigned, the true global NN may be missed.
        # This matches the design of split_test.py and must be noted in paper results.
        niche_train_indices = [idx for idx, lab in enumerate(data.labels) if int(lab) == target_niche]
        distances = []
        
        for t_idx in niche_train_indices:
            t_spd = train_covs[t_idx]
            t_perm = t_spd[np.ix_(perm, perm)]
            total_block_dist = 0.0
            for b_idx, (start, end) in enumerate(block_runs):
                t_block = t_perm[start:end, start:end]
                t_block_log = log_spd(t_block)
                diff = q_corr_blocks_log[b_idx] - t_block_log
                p_block = t_block.shape[0]
                total_block_dist += np.linalg.norm(diff, ord='fro') / np.sqrt(p_block)
            distances.append((total_block_dist, t_idx))
            
        distances.sort(key=lambda x: x[0]) 
        closest_dist = distances[0][0] if len(distances) > 0 else float('inf')
        
        best_exact_idx = distances[0][1] if len(distances) > 0 else -1
        
        # Stage 2: Spindle performs exact re-ranking on retrieved candidate tiles
        spindle_reranked = []
        for match_global_idx in all_matched_train_ids[i]:
            t_spd = train_covs[match_global_idx]
            t_perm = t_spd[np.ix_(perm, perm)]
            cand_dist = 0.0
            for b_idx, (start, end) in enumerate(block_runs):
                t_block = t_perm[start:end, start:end]
                t_block_log = log_spd(t_block)
                diff = q_corr_blocks_log[b_idx] - t_block_log
                p_block = t_block.shape[0]
                cand_dist += np.linalg.norm(diff, ord='fro') / np.sqrt(p_block)
            spindle_reranked.append((cand_dist, match_global_idx))
            
        spindle_reranked.sort(key=lambda x: x[0])
        
        spindle_best_dist = spindle_reranked[0][0] if spindle_reranked else float('inf')
        best_spindle_idx = spindle_reranked[0][1] if spindle_reranked else -1
        spindle_best_rank = -1
        if best_spindle_idx != -1:
            for r, (d, idx) in enumerate(distances):
                if idx == best_spindle_idx:
                    spindle_best_rank = r + 1
                    break
            if spindle_best_rank == -1:
                # Spindle's best candidate is not in the predicted niche's train set —
                # assign a rank beyond last place and log a warning.
                spindle_best_rank = len(distances) + 1
                print(f"  [WARN] Query {i}: Spindle best idx {best_spindle_idx} not found "
                      f"in niche {target_niche} ({len(niche_train_indices)} members). "
                      f"Assigning rank {spindle_best_rank}.")
                
        if not spindle_reranked:
            print(f"Query {i:3d} (Niche {target_niche}): Exact closest dist = {closest_dist:.3f} | Spindle found NOTHING.")
            exact_dists.append(closest_dist)
            spindle_dists.append(float('inf'))
            spindle_ranks.append(-1)
        else:
            print(f"Query {i:3d} (Niche {target_niche}): Exact dist = {closest_dist:.3f}, Spindle dist = {spindle_best_dist:.3f} | Rank {spindle_best_rank}")
            exact_dists.append(closest_dist)
            spindle_dists.append(spindle_best_dist)
            spindle_ranks.append(spindle_best_rank)
            
        results_for_plot.append({
            'query_idx': i,
            'exact_idx': best_exact_idx,
            'spindle_idx': best_spindle_idx,
            'spindle_rank': spindle_best_rank
        })
        
        bf_ranking = [idx for d, idx in distances]
        spindle_ranking = [idx for _, idx in spindle_reranked]
        
        # Tie-aware recall@1: counts as a hit if Spindle's top-result distance matches
        # the exact best distance within floating-point tolerance, consistent with
        # split_test.py.
        recall_1 = 1 if (spindle_reranked and
                         abs(spindle_reranked[0][0] - closest_dist) < 1e-5) else 0
        
        def calc_overlap(k):
            if not bf_ranking: return 0.0
            # Cap k to the actual niche size to avoid ill-defined overlap when
            # the niche has fewer than k members.
            effective_k = min(k, len(bf_ranking))
            if effective_k == 0: return 0.0
            bf_top = set(bf_ranking[:effective_k])
            sp_top = set(spindle_ranking[:effective_k])
            denom = max(1, len(bf_top))
            return len(bf_top.intersection(sp_top)) / float(denom)
            
        overlap_5 = calc_overlap(5)
        overlap_10 = calc_overlap(10)
        overlap_20 = calc_overlap(20)
        
        metrics_records.append({
            'query_idx': i,
            'target_niche': target_niche,
            'exact_best_dist': round(closest_dist, 4),
            'spindle_best_dist': round(spindle_best_dist, 4) if spindle_best_rank != -1 else np.nan,
            'spindle_best_rank': spindle_best_rank,
            'recall@1': recall_1,
            'overlap@5': round(overlap_5, 4),
            'overlap@10': round(overlap_10, 4),
            'overlap@20': round(overlap_20, 4)
        })
            
    # Use an absolute path derived from the project root so outputs go to the correct location
    # regardless of the working directory from which this script is invoked.
    _project_root = _this_dir.parent
    _out_dir = _project_root / "results" / "cross_modal_search"
    _out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = str(_out_dir / f"{direction}_query_metrics.csv")
    if metrics_records:
        fieldnames = list(metrics_records[0].keys())
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(metrics_records)
        print(f"Saved detailed query metrics to {csv_path}")
        
        mean_r1 = np.mean([m['recall@1'] for m in metrics_records])
        mean_o5 = np.mean([m['overlap@5'] for m in metrics_records])
        mean_o10 = np.mean([m['overlap@10'] for m in metrics_records])
        mean_o20 = np.mean([m['overlap@20'] for m in metrics_records])
        
        print("\n" + "-" * 60)
        print(f"BENCHMARK OVERLAP METRICS SUMMARY ({direction}):")
        print(f"  Recall@1   : {mean_r1:.4f}")
        print(f"  Overlap@5  : {mean_o5:.4f}")
        print(f"  Overlap@10 : {mean_o10:.4f}")
        print(f"  Overlap@20 : {mean_o20:.4f}")
        print("-" * 60)
        
        summary_dict = {
            'direction': direction,
            'recall@1': round(mean_r1, 4),
            'overlap@5': round(mean_o5, 4),
            'overlap@10': round(mean_o10, 4),
            'overlap@20': round(mean_o20, 4)
        }
        summary_csv_path = str(_out_dir / "benchmark_summary.csv")
        existing_rows = []
        if os.path.exists(summary_csv_path):
            with open(summary_csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
                
        updated = False
        for row in existing_rows:
            if row.get('direction') == direction:
                row.update({k: str(v) for k, v in summary_dict.items() if k != 'direction'})
                updated = True
                break
        if not updated:
            existing_rows.append({k: str(v) for k, v in summary_dict.items()})
            
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['direction', 'recall@1', 'overlap@5', 'overlap@10', 'overlap@20'])
            writer.writeheader()
            writer.writerows(existing_rows)
        print(f"Updated benchmark summary metrics in {summary_csv_path}")

    print(f"\nCompleted evaluation for {direction}.")
    return results_for_plot


def plot_search_results(results_for_plot, coords_query, coords_index, tiles_query, tiles_index, query_name, index_name, direction):
    print("Side-by-side search visualization disabled in CSV-only mode.")

def run_search_pipeline(direction, n_queries, covs_xe, covs_vi, tiles_xe, tiles_vi, total_spots_xe, total_spots_vi, coords_xe, coords_vi, common_genes, args):
    if direction == "x2v":
        index_name = "Visium"
        query_name = "Xenium"
        tiles_index, covs_index, total_spots_index = tiles_vi, covs_vi, total_spots_vi
        tiles_query, covs_query, total_spots_query = tiles_xe, covs_xe, total_spots_xe
        coords_index, coords_query = coords_vi, coords_xe
    else:
        index_name = "Xenium"
        query_name = "Visium"
        tiles_index, covs_index, total_spots_index = tiles_xe, covs_xe, total_spots_xe
        tiles_query, covs_query, total_spots_query = tiles_vi, covs_vi, total_spots_vi
        coords_index, coords_query = coords_xe, coords_vi
        
    print(f"Generated {len(covs_query)} {query_name} query SPDs.")
    
    if n_queries is not None and n_queries < len(covs_query):
        query_covs = [t["cov"] for t in covs_query[:n_queries]]
    else:
        query_covs = [t["cov"] for t in covs_query]
        n_queries = len(query_covs)
        
    print(f"Selecting {n_queries} {query_name} queries for search.")
    
    print(f"\n[3/6] Building ProcessedData for {index_name}...")
    processed_index = ProcessedData(tiles=tiles_index, tile_stats=covs_index, genes_work=common_genes, num_spots=total_spots_index)
    
    print(f"Reducing dimensions and clustering SPDs ({index_name})...")
    processed_index.reduce_dim(cluster_distance="tree", num_pca_components=30, random_state=42)
    # Use Leiden clustering (adaptive resolution) to match the split_test.py methodology.
    processed_index.cluster_spds(cluster_distance="tree", cluster_method="leiden", resolution=0.2)
    
    print("Computing mean correlations and finding block diagonal order...")
    processed_index.get_corr_mean_by_cluster()
    processed_index.get_adaptive_runs(find_blocks=True, min_final_size=5, max_final_size=50)

    print(f"\n[4/6] Building DAG index for {index_name}...")
    from spindle_dev.index import choose_adaptive_epsilons
    config = IndexConfig()
    # Use data-driven adaptive epsilons (elbow method) to match split_test.py methodology.
    for cluster_id in set(processed_index.labels):
        _, _, eps = choose_adaptive_epsilons(processed_index, int(cluster_id), k_target_per_block=64)
        config.epsilon_dict[int(cluster_id)] = eps
        
    try:
        dag_dict, stats, global_dist_list = index_spds(processed_index, config)
    except Exception as e:
        print(f"Error building index: {e}")
        return

    print(f"\n[5/6] Aligning {query_name} SPDs to {index_name} index clusters...")
    assigned_labels = assign_clusters_to_new_spds(query_covs, processed_index, strategy="knn_majority", n_neighbors=5)
    
    print("Calculating tangent space block means and stds for modality bias correction...")
    cluster_means = {}
    train_covs = [t["cov"] for t in covs_index]
    
    for cluster_id in set(processed_index.labels):
        perm = processed_index.perm_list[cluster_id]
        block_runs = processed_index.block_dict[cluster_id]
        
        idx_train = [idx for idx, lab in enumerate(processed_index.labels) if int(lab) == cluster_id]
        idx_query = [idx for idx, lab in enumerate(assigned_labels) if int(lab) == cluster_id]
        
        block_means_train = []
        block_means_query = []
        block_stds_train = []
        block_stds_query = []
        
        for b_idx, (start, end) in enumerate(block_runs):
            # Index Mean & Std
            t_logs = []
            for i in idx_train:
                t_spd = train_covs[i]
                t_perm = t_spd[np.ix_(perm, perm)]
                t_logs.append(log_spd(t_perm[start:end, start:end]))
            if t_logs:
                block_means_train.append(np.mean(t_logs, axis=0))
                block_stds_train.append(np.std(t_logs))
            else:
                block_means_train.append(0)
                block_stds_train.append(1.0)
            
            # Query Mean & Std
            q_logs = []
            for i in idx_query:
                q_spd = query_covs[i]
                q_perm = q_spd[np.ix_(perm, perm)]
                q_logs.append(log_spd(q_perm[start:end, start:end]))
            if q_logs:
                block_means_query.append(np.mean(q_logs, axis=0))
                std_q = np.std(q_logs)
                block_stds_query.append(std_q if std_q > 1e-6 else 1.0)
            else:
                block_means_query.append(0)
                block_stds_query.append(1.0)
            
        cluster_means[cluster_id] = {
            'train': block_means_train, 'query': block_means_query,
            'std_train': block_stds_train, 'std_query': block_stds_query
        }

    print("\n[6/6] Running cross-modal search with bias correction...")
    # SearchConfig aligned with split_test.py for direct comparability.
    search_cfg = SearchConfig(
        max_results=None,
        debug=False,
        max_failed_starts=10,
        max_failed_paths=20,
        total_paths_limit=300
    )
    
    total_hits = 0
    total_time = 0.0
    valid_searches = 0
    all_matched_train_ids = []
    corrected_queries_by_id = {}
    
    for i, q_spd in enumerate(query_covs):
        cluster_id = int(assigned_labels[i])
        index_handle = dag_dict.get(cluster_id)
        
        if index_handle is None:
            all_matched_train_ids.append([])
            continue
            
        perm = processed_index.perm_list[cluster_id]
        block_runs = processed_index.block_dict[cluster_id]
        
        q_spd_perm = q_spd[np.ix_(perm, perm)]
        q_corr_blocks_log = []
        q_spd_corr_perm = np.zeros_like(q_spd_perm)
        
        for b_idx, (start, end) in enumerate(block_runs):
            q_block = q_spd_perm[start:end, start:end]
            q_log = log_spd(q_block)
            
            raw_scale = cluster_means[cluster_id]['std_train'][b_idx] / cluster_means[cluster_id]['std_query'][b_idx]
            # Cap the scale at 1.0 to prevent over-correction: when the index (train) modality
            # has a larger spread than the query modality we still shrink the gap, but we never
            # amplify the query variance beyond its natural level.  This is intentionally
            # conservative — it means x2v (Xenium query, Visium index) benefits more from
            # correction than v2x when Xenium std is larger, which accounts for part of the
            # observed recall asymmetry between the two directions.
            scale = min(1.0, raw_scale)
            
            L_corr = (q_log - cluster_means[cluster_id]['query'][b_idx]) * scale + cluster_means[cluster_id]['train'][b_idx]
            q_corr_blocks_log.append(L_corr)
            
            C_corr = exp_spd(L_corr)
            q_spd_corr_perm[start:end, start:end] = C_corr
            
        corrected_queries_by_id[i] = q_corr_blocks_log
        
        epsilon = config.epsilon_dict.get(cluster_id, 0.5) if hasattr(config, 'epsilon_dict') else 0.5
        budget = float(epsilon) * float(len(block_runs)) * float(args.budget_mult)

        if i == 0 or (i == 1 and direction):
            print(f"  [Budget] Niche {cluster_id}: epsilon={epsilon:.4f}, "
                  f"blocks={len(block_runs)}, mult={args.budget_mult}, budget={budget:.3f}")
        
        # Pass the permuted corrected SPD directly — block_runs are defined in permuted
        # gene space, so search_index must receive q_spd_corr_perm (not the inverse-permuted
        # version). Passing the un-permuted matrix caused a coordinate mismatch bug where
        # block slices indexed the wrong rows/columns during DAG traversal.
        t0 = time.perf_counter()
        results = search_index(
            index_handle=index_handle,
            query_spd=q_spd_corr_perm,
            query_indices=[],
            query_block_runs=block_runs,
            budget=budget,
            config=search_cfg
        )
        total_time += (time.perf_counter() - t0)
        
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
                # Deduplicate: a candidate appearing in multiple paths should only be
                # counted once, matching the split_test.py implementation.
                for spd_id in sorted(intersect_ids):
                    if spd_id not in seen_matched:
                        seen_matched.add(spd_id)
                        matched_ids_for_query.append(spd_id)
            
        all_matched_train_ids.append(matched_ids_for_query)
        
        valid_searches += 1
        total_hits += len(matched_ids_for_query)
        
        step = max(1, n_queries//10)
        if (i+1) % step == 0:
            print(f"Processed {i+1}/{n_queries} queries...")

    nothing_count = sum(1 for ids in all_matched_train_ids if len(ids) == 0)
    print("\n" + "=" * 40)
    print(f"Cross-Modal Search Results ({query_name} -> {index_name}):")
    print(f"Total Valid Queries : {valid_searches}")
    print(f"Total Paths Found   : {total_hits}")
    print(f"Queries with NOTHING: {nothing_count}/{n_queries} "
          f"({'budget too tight — try higher --budget-mult' if nothing_count > n_queries * 0.2 else 'OK'})")
    print(f"Avg Time Per Query  : {total_time / valid_searches:.4f}s" if valid_searches else "No valid searches")
    print("=" * 40)

    plot_results = evaluate_brute_force_approximation(query_covs, corrected_queries_by_id, train_covs, assigned_labels, all_matched_train_ids, processed_index, direction=direction)
    # Search visualization plotting disabled in CSV-only mode

def main():
    parser = argparse.ArgumentParser(description="Cross-Modal SPD Index Search")
    parser.add_argument("--direction", type=str, choices=["x2v", "v2x", "both"], default="both",
                        help="Search direction: 'x2v', 'v2x', or 'both'")
    parser.add_argument("--n_queries", type=int, default=50,
                        help="Number of queries to run if query dataset is large.")
    parser.add_argument("--budget-mult", type=float, default=1.5,
                        help=(
                            "Distance budget multiplier for DAG search. "
                            "Cross-modal search requires a higher value than same-modality "
                            "split-test (default 1.0) because residual inter-modality distribution "
                            "shift after bias correction inflates block-level distances. "
                            "Default=3.0, empirically calibrated for Xenium/Visium cross-modal."
                        ))
    args = parser.parse_args()

    print("Loading datasets...")
    adata_vi = sc.read_h5ad(r"/home/asus/spindle_dev/dataset/opt_brca/brca/visium_rotated.h5ad")
    adata_xe = sc.read_h5ad(r"/home/asus/spindle_dev/dataset/opt_brca/brca/xenium_rotated.h5ad")

    adata_vi.var_names_make_unique()
    adata_xe.var_names_make_unique()

    common_genes = sorted(list(set(adata_vi.var_names).intersection(adata_xe.var_names)))
    print(f"Found {len(common_genes)} common genes.")

    if not common_genes:
        print("No common genes found!")
        return

    from spindle_dev.preprocessing import build_quadtree_tiles, QuadTile, build_tile_covs_full

    print("\n[1/6] Building quadtree tiles on Xenium...")
    adata_xe_sub = adata_xe[:, common_genes].copy()
    coords_xe = adata_xe_sub.obsm["spatial"]
    tiles_xe = build_quadtree_tiles(coords_xe, max_pts=2000, min_side=0.0, max_depth=40)
    
    print("\n[2/6] Overlaying tiles on Visium...")
    adata_vi_sub = adata_vi[:, common_genes].copy()
    coords_vi = adata_vi_sub.obsm["spatial"]
    
    tiles_vi = []
    for t in tiles_xe:
        x0, y0, x1, y1 = t.bbox
        mask = (coords_vi[:, 0] >= x0) & (coords_vi[:, 0] < x1) & \
               (coords_vi[:, 1] >= y0) & (coords_vi[:, 1] < y1)
        child_idx = np.where(mask)[0]
        if len(child_idx) >= 10:  
            tiles_vi.append(QuadTile(t.id, t.bbox, child_idx))
            
    for i, t in enumerate(tiles_xe): t.id = i
    for i, t in enumerate(tiles_vi): t.id = i
            
    print(f"Built {len(tiles_xe)} Xenium tiles and {len(tiles_vi)} Visium tiles.")

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "results" / "cross_modal_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    box_records = []
    for t in tiles_xe:
        x0, y0, x1, y1 = t.bbox
        box_records.append({'modality': 'Xenium', 'id': t.id, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
    for t in tiles_vi:
        x0, y0, x1, y1 = t.bbox
        box_records.append({'modality': 'Visium', 'id': t.id, 'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1})
    pd.DataFrame(box_records).to_csv(out_dir / "tile_overlay_boxes.csv", index=False)
    print(f"Exported tile overlay bounding boxes CSV to {out_dir / 'tile_overlay_boxes.csv'}")

    coord_records = []
    np.random.seed(42)
    sample_xe = coords_xe[np.random.choice(len(coords_xe), size=min(5000, len(coords_xe)), replace=False)]
    sample_vi = coords_vi[np.random.choice(len(coords_vi), size=min(5000, len(coords_vi)), replace=False)]
    for pt in sample_xe:
        coord_records.append({'modality': 'Xenium', 'x': pt[0], 'y': pt[1]})
    for pt in sample_vi:
        coord_records.append({'modality': 'Visium', 'x': pt[0], 'y': pt[1]})
    pd.DataFrame(coord_records).to_csv(out_dir / "spatial_coords_sample.csv", index=False)
    print(f"Exported background spatial coordinates sample CSV to {out_dir / 'spatial_coords_sample.csv'}")

    print("\n[3/6] Computing tile covariance matrices...")
    covs_xe = build_tile_covs_full(adata_xe_sub, tiles_xe, gene_idx=None, n_jobs=8, eps=1e-6)
    covs_vi = build_tile_covs_full(adata_vi_sub, tiles_vi, gene_idx=None, n_jobs=8, eps=1e-6)

    total_spots_xe = adata_xe_sub.n_obs
    total_spots_vi = adata_vi_sub.n_obs

    directions = ["x2v", "v2x"] if args.direction == "both" else [args.direction]
    for dir_str in directions:
        print("\n" + "#" * 60)
        print(f"### RUNNING SEARCH PIPELINE FOR DIRECTION: {dir_str.upper()}")
        print("#" * 60)
        run_search_pipeline(dir_str, args.n_queries, covs_xe, covs_vi, tiles_xe, tiles_vi, total_spots_xe, total_spots_vi, coords_xe, coords_vi, common_genes, args)

if __name__ == "__main__":
    main()
