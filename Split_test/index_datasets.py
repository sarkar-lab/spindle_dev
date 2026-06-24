import sys
import time
from pathlib import Path
import pickle
import numpy as np
import scanpy as sc
import matplotlib
matplotlib.use('Agg')
import argparse

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import spindle_dev
import spindle_dev.metrics as metrics
import spindle_dev.index as index
import spindle_dev.preprocessing as preprocessing
import spindle_dev.plotting as plotting
import spindle_dev.typing as typing
import spindle_dev.test as test
import spindle_dev.search as search


def prepare_to_index(adata):
    """
    Prepare standard data object for indexing.
    """
    coords = adata.obsm["spatial"]
    tiles = preprocessing.build_quadtree_tiles(coords, max_pts=200, min_side=0.0, max_depth=40)
    num_genes = adata.n_vars
    genes_work, gene_idx = spindle_dev.preprocessing.topvar_genes(adata, G=num_genes)  
    tile_covs = spindle_dev.preprocessing.build_tile_covs_full(adata, tiles, gene_idx, n_jobs=8, eps=1e-6)

    return tiles, tile_covs, genes_work


def run_index(tiles, tile_covs, genes_work, adata, resolution=0.2, min_final_size=20):
    """
    Run indexing workflow.
    """
    data = index.ProcessedData(tiles, tile_covs, genes_work, adata.n_obs)
    num_pca = min(30, len(tiles) - 1)
    if num_pca < 2:
        num_pca = 2
    data.reduce_dim(num_pca_components=num_pca, n_components=2, do_umap=True)
    data.cluster_spds(cluster_distance="tree", cluster_method="leiden", resolution=resolution)
    data.assign_label_to_spots()
    data.get_corr_mean_by_cluster()
    out_dict = data.get_adaptive_runs(find_blocks=True, with_size_guard=True, min_final_size=min_final_size, max_final_size=100)
    return data, out_dict


def load_and_split_data(adata_path, test_ratio=0.05, seed=42, n_subsample=None):
    print(f"Reading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    if 'Cluster' in adata.obs.columns:
        adata = adata[adata.obs.loc[adata.obs.Cluster != "Unlabeled"].index, :].copy()

    if n_subsample is not None and n_subsample < adata.n_obs:
        print(f"Subsampling to {n_subsample} spots for quick test...")
        sc.pp.subsample(adata, n_obs=n_subsample, random_state=seed)

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


def main():
    parser = argparse.ArgumentParser(description="Index datasets and save to disk")
    parser.add_argument('--test', action='store_true', help='Run a quick test on a subset of data (10k spots)')
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    
    datasets = {
        # "brain_cancer": project_root.parent / "dataset" / "xenium_human_brain_cancer.h5ad",
        # "breast_cancer": project_root.parent / "dataset" / "xenium_human_breast_cancer.h5ad",
        # "kidney_nondiseased": project_root.parent / "dataset" / "xenium_human_kidney_nondiseased.h5ad",
        # "lymph_node": project_root.parent / "dataset" / "xenium_human_lymph_node.h5ad",
        # "lung_cancer": project_root.parent / "dataset" / "xenium_human_lung_cancer.h5ad",
        "skin_melanoma": project_root.parent / "dataset" / "xenium_human_skin_melanoma.h5ad",
        # "pancreatic_cancer": project_root.parent / "dataset" / "xenium_human_pancreatic_cancer.h5ad"
        # "lymph_node_5k": project_root.parent / "dataset" / "xenium_human_lymph_node_5k.h5ad"
    }

    base_results_dir = project_root / "results" / "split_test_indexed"
    base_results_dir.mkdir(exist_ok=True, parents=True)

    for dataset_name, adata_path in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        save_path = base_results_dir / f"{dataset_name}_indexed.pkl"
        if save_path.exists():
            print(f"Index for {dataset_name} already exists. Skipping.")
            continue

        if not adata_path.exists():
            print(f"Dataset not found at {adata_path}. Skipping.")
            continue

        n_subsample = 10000 if args.test else None
        # 1. Load and split data
        adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = load_and_split_data(adata_path, n_subsample=n_subsample)

        # 2. Run index
        print("Running index...")
        data, out_dict = run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)

        # 3. Configure and build DAG
        dag_dict, config = configure_and_build_dag(data)

        # Save to disk
        print(f"Saving indexed data to {save_path}...")
        save_data = {
            'test_tile_covs': test_tile_covs,
            'train_tile_covs': train_tile_covs,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'data': data,
            'dag_dict': dag_dict,
            'config': config,
            'dataset_name': dataset_name
        }

        import gc
        del adata, train_tiles, test_tiles, genes_work
        gc.collect()

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Save complete.")

        import matplotlib.pyplot as plt
        plt.close('all')
        del save_data, train_tile_covs, test_tile_covs, data, dag_dict, config
        gc.collect()

if __name__ == "__main__":
    main()
