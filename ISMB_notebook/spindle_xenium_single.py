# Set up sys.path so that 'src/spindle_dev' is importable as 'spindle_dev'
import sys
import importlib
from pathlib import Path

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
# Reload to pick up code changes without restarting the kernel

import time
import scanpy as sc
import glob
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from joblib import Parallel, delayed
import argparse

def create_index(adata, index_path, resolution=0.5, min_final_size=10, top_genes=200, all_genes=False, max_queries=100):
    # start time 
    
    # Create index directory if it doesn't exist
    Path(index_path).mkdir(parents=True, exist_ok=True)

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
    out_dict = data.get_adaptive_runs(find_blocks=True, with_size_guard=True,min_final_size=min_final_size,max_final_size=100)
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
    config.threshold_type = 'block_wise'
    config.kmean_method = 'epsilon_net'

    dag_dict, stat, dist_list = index.index_spds(data, config=config)
    end_time = time.time()
    index_time = end_time - start_time
    # Dump a json file with index creation time and memory
    with open(index_path + '/index_stats.txt', 'w') as f:
        f.write(f"{index_time}\n")
    

    search_cfg = search.SearchConfig(max_results=2, debug=False, max_failed_starts=5, max_failed_paths=10)
    index.save_index(data, dag_dict, index_path + '/spindle.pkl')
    test_results = test.run_sanity_search(data, dag_dict, config, search_cfg, max_queries=max_queries)
    
    test_df = pd.DataFrame(test_results['records'])
    test_df.to_csv(index_path + '/sanity_test_results.csv', index=False)

def process_file(h5ad_file, resolution=0.5, min_final_size=15, top_genes=800, all_genes=True, max_queries=10):
    """Wrapper function to load and process a single file"""
    try:
        
        adata = sc.read_h5ad(h5ad_file)
        print(f"Processing file: {h5ad_file} with {adata.n_obs} spots and {adata.n_vars} genes.")
        index_path = h5ad_file.replace('.h5ad', '_index')
        create_index(adata, index_path, resolution=resolution, min_final_size=min_final_size, 
                    top_genes=top_genes, all_genes=all_genes, max_queries=max_queries)
        print(f"Completed: {h5ad_file}")
        return h5ad_file, True, None
    except Exception as e:
        print(f"Error processing {h5ad_file}: {e}")
        return h5ad_file, False, str(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a single h5ad file with Spindle indexing")
    parser.add_argument("h5ad_file", help="Path to the h5ad file to process")
    parser.add_argument("--resolution", type=float, default=0.5, help="Clustering resolution (default: 0.5)")
    parser.add_argument("--min-final-size", type=int, default=15, help="Minimum final cluster size (default: 15)")
    parser.add_argument("--top-genes", type=int, default=800, help="Number of top genes to use (default: 800)")
    parser.add_argument("--all-genes", action="store_true", help="Use all genes instead of top genes")
    parser.add_argument("--max-queries", type=int, default=500, help="Maximum number of sanity check queries (default: 500)")
    
    args = parser.parse_args()
    
    h5ad_file = args.h5ad_file
    h5ad_file_name, success, error = process_file(
        h5ad_file, 
        resolution=args.resolution,
        min_final_size=args.min_final_size,
        top_genes=args.top_genes,
        all_genes=args.all_genes,
        max_queries=args.max_queries
    )
    
    if success:
        print(f"✓ Successfully processed: {h5ad_file}")
    else:
        print(f"✗ Failed to process {h5ad_file}: {error}")


