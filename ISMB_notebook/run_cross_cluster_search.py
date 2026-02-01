#!/usr/bin/env python
"""
Cross-cluster search performance analysis for Spindle
"""
import sys
from pathlib import Path

# Set up sys.path
project_root = '/data/sarkar_lab/Projects/spindle_dev'
src_path = Path(project_root) / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import spindle_dev
import spindle_dev.metrics as metrics
import spindle_dev.index as index
import spindle_dev.preprocessing as preprocessing
import spindle_dev.plotting as plotting
import spindle_dev.test as test
import spindle_dev.search as search
import spindle_dev.typing as typing

import scanpy as sc
import glob
import time
import numpy as np
import pandas as pd


def create_index(adata, resolution=0.5, min_final_size=10, top_genes=200, n_queries=2000, all_genes=False):
    """Create Spindle index and perform cross-cluster search"""
    start_time = time.time()
    
    coords = adata.obsm["spatial"]
    tiles = preprocessing.build_quadtree_tiles(coords, max_pts=200, min_side=0.0, max_depth=40)
    # remove tiles with less than 5 spots
    tiles = [tile for tile in tiles if len(tile.idx) >= 5]
    tiles = preprocessing.reindex_tiles(tiles)
    
    if all_genes:
        num_genes = adata.n_vars
    else:
        num_genes = top_genes
        
    genes_work, gene_idx = spindle_dev.preprocessing.topvar_genes(adata, G=num_genes)  
    tile_covs = spindle_dev.preprocessing.build_tile_covs_full_serial(adata, tiles, gene_idx, eps=1e-6)
    data = index.ProcessedData(tiles, tile_covs, genes_work, adata.n_obs)
    
    if 'pca' not in data.latent:
        data.reduce_dim(num_pca_components=30, n_components=2, do_umap=True)
        
    data.cluster_spds(cluster_distance="tree", cluster_method="leiden", resolution=resolution)
    data.assign_label_to_spots()
    data.get_corr_mean_by_cluster()
    out_dict = data.get_adaptive_runs(find_blocks=True, with_size_guard=True, min_final_size=min_final_size, max_final_size=100)
    
    epsilon_block_wise_dict = {}
    epsilon_dict = {}
    for cluster_id in set(data.labels):
        eps_per_block, eps_elbow_per_block, eps = index.choose_adaptive_epsilons(data, cluster_id, k_target_per_block=64)
        epsilon_block_wise_dict[int(cluster_id)] = eps_elbow_per_block
        epsilon_dict[int(cluster_id)] = eps

    # Create indices config
    config = typing.IndexConfig()
    config.epsilon_dict = epsilon_dict
    config.epsilon_block_wise_dict = epsilon_block_wise_dict
    config.threshold_type = 'constant'
    config.kmean_method = 'epsilon_net' 
    
    dag_dict, stat, dist_list = index.index_spds(data, config=config)

    seed = 40
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(data.spd_matrices))
    valid_clusters = list(dag_dict.keys())
    mask = np.isin(data.labels, valid_clusters)
    candidate_indices = all_indices[mask]
    
    query_indices = rng.choice(candidate_indices, size=n_queries, replace=False)

    gt_paths = test.create_ground_truth_paths(dag_dict)
    query_matrices = [data.spd_matrices[i] for i in query_indices]

    true_clusters = [int(data.labels[i]) for i in query_indices]
    predicted_clusters = search.assign_clusters_to_new_spds(query_matrices, data)
    predicted_df = pd.DataFrame({'True Cluster': true_clusters, 'Predicted Cluster': predicted_clusters})
    
    print(f"Done with search - took {time.time() - start_time:.2f}s")
    return predicted_df


def main():
    # Get all h5ad files
    h5ad_files = glob.glob("/data/sarkar_lab/insitupy_demo_data_xenium/*.h5ad")
    
    print(f"Found {len(h5ad_files)} h5ad files")
    
    all_df = []
    for h5ad_file in h5ad_files:
        if '5k' not in h5ad_file:
            print(f"\nProcessing: {h5ad_file}")
            adata = sc.read_h5ad(h5ad_file)
            experiment_name = h5ad_file.split("/")[-1].replace(".h5ad", "")
            print(f"Experiment: {experiment_name}")
            
            predicted_df = create_index(adata, all_genes=True, n_queries=200)
            predicted_df['experiment_name'] = experiment_name
            all_df.append(predicted_df)
            
            del adata
    
    # Merge all dataframes and save to CSV
    merged_df = pd.concat(all_df, ignore_index=True)
    
    output_dir = Path('/data/sarkar_lab/Projects/spindle_dev/results/search_results')
    
    # Create results directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / 'cross_cluster_search_performance.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Saved results to {output_path}")
    print(f"Total rows: {len(merged_df)}")
    print(f"{'='*60}")
    print("\nFirst few rows:")
    print(merged_df.head())


if __name__ == "__main__":
    main()
