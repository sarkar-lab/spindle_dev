import sys
import time
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
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


def run_indexing_for_datasets(datasets, is_test=False, train_test_ratio=0.05):
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    base_results_dir = project_root / "results" / "holdout_validation_indexed"
    base_results_dir.mkdir(exist_ok=True, parents=True)

    summary_out_dir = project_root / "results" / "holdout_validation"
    summary_out_dir.mkdir(exist_ok=True, parents=True)

    # Fallback/known values for dataset display names and benchmark timings if skipping already
    # indexed files.  Cell counts are post-filter (after removing 'Unlabeled' clusters), matching
    # the actual adata.n_obs values stored inside the index files and reported in
    # results/holdout_validation/index_scalability_summary.csv.
    ds_display_map = {
        'skin_melanoma': ('Skin', 87499, 26.89, 19.41),
        'kidney_nondiseased': ('Kidney', 97560, 30.27, 24.21),
        'breast_cancer': ('Breast', 159226, 90.70, 23.25),
        'lung_cancer': ('Lung', 162254, 58.41, 20.25),
        'lymph_node': ('Lymph Node', 377985, 140.90, 22.83),
        'pancreatic_cancer': ('Pancreas', 190965, 198.37, 30.86),
    }

    scalability_records = []

    for dataset_name, adata_path in datasets.items():
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_name}")
        print(f"{'='*80}\n")
        
        index_save_path = base_results_dir / f"{dataset_name}_spindle_index.pkl"
        covs_save_path = base_results_dir / f"{dataset_name}_raw_covariances.pkl"
        
        backup_index = project_root.parent / "Results Backup" / "holdout_validation_indexed" / f"{dataset_name}_spindle_index.pkl"
        backup_covs = project_root.parent / "Results Backup" / "holdout_validation_indexed" / f"{dataset_name}_raw_covariances.pkl"
        
        legacy_save_path = base_results_dir / f"{dataset_name}_indexed.pkl"
        legacy_backup_path = project_root.parent / "Results Backup" / "holdout_validation_indexed" / f"{dataset_name}_indexed.pkl"

        load_index_path = index_save_path if index_save_path.exists() else (backup_index if backup_index.exists() else None)
        load_covs_path = covs_save_path if covs_save_path.exists() else (backup_covs if backup_covs.exists() else None)
        legacy_path = legacy_save_path if legacy_save_path.exists() else (legacy_backup_path if legacy_backup_path.exists() else None)

        disp_name, fb_cells, fb_time, fb_size = ds_display_map.get(dataset_name, (dataset_name, 0, 0.0, 0.0))

        # Auto-migrate legacy file into separate files if needed
        if load_index_path is None and legacy_path is not None:
            print(f"Migrating legacy combined file {legacy_path.name} into separate index and covariance files...")
            try:
                with open(legacy_path, 'rb') as pf:
                    saved_legacy = pickle.load(pf)
                
                d = saved_legacy['data']
                bundle = typing.DatasetIndex(
                    dag_dict=saved_legacy['dag_dict'],
                    metadata=getattr(d, 'metadata', {}),
                    latent=getattr(d, 'latent', {}),
                    labels=getattr(d, 'labels', None),
                    pca_model=getattr(d, 'pca_model', None),
                )
                size_mb = round(len(pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024), 2)
                d.spd_matrices = []
                d.U_list = None
                
                index_data = {
                    'data': d,
                    'dag_dict': saved_legacy['dag_dict'],
                    'config': saved_legacy['config'],
                    'dataset_name': saved_legacy['dataset_name'],
                    'build_time_s': saved_legacy.get('build_time_s', fb_time),
                    'index_size_mb': size_mb
                }
                covs_data = {
                    'test_tile_covs': saved_legacy['test_tile_covs'],
                    'train_tile_covs': saved_legacy['train_tile_covs'],
                    'train_idx': saved_legacy['train_idx'],
                    'test_idx': saved_legacy['test_idx'],
                    'dataset_name': saved_legacy['dataset_name']
                }
                
                with open(index_save_path, 'wb') as f:
                    pickle.dump(index_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open(covs_save_path, 'wb') as f:
                    pickle.dump(covs_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print(f"Successfully migrated {dataset_name} to {index_save_path.name} ({size_mb} MB) and {covs_save_path.name}")
                load_index_path = index_save_path
            except Exception as e:
                print(f"Error migrating {legacy_path}: {e}")

        if load_index_path is not None:
            print(f"Index for {dataset_name} already exists at {load_index_path}. Extracting scalability info from file.")
            try:
                with open(load_index_path, 'rb') as pf:
                    saved = pickle.load(pf)
                num_cells = getattr(saved.get('data', None), 'num_spots', fb_cells)
                build_t = saved.get('build_time_s', fb_time)
                size_mb = saved.get('index_size_mb', fb_size)
            except Exception as e:
                print(f"Error loading {load_index_path}: {e}")
                num_cells, build_t, size_mb = fb_cells, fb_time, fb_size

            scalability_records.append({
                'Dataset': disp_name,
                'Cells': num_cells,
                'Build Time (s)': round(build_t, 2),
                'Index Size (MB)': size_mb
            })
            continue

        if not adata_path.exists():
            print(f"Dataset not found at {adata_path}. Skipping.")
            continue

        n_subsample = 10000 if is_test else None
        # 1. Load and split data
        adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = load_and_split_data(adata_path, test_ratio=train_test_ratio, n_subsample=n_subsample)
        num_cells = adata.n_obs

        # 2. Run index & measure time
        print("Running index...")
        t0 = time.perf_counter()
        data, out_dict = run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)

        # 3. Configure and build DAG
        dag_dict, config = configure_and_build_dag(data)
        build_time_s = time.perf_counter() - t0

        # Calculate true Spindle index size (DatasetIndex bundle without raw covariance matrices)
        index_bundle = typing.DatasetIndex(
            dag_dict=dag_dict,
            metadata=data.metadata,
            latent=data.latent,
            labels=data.labels,
            pca_model=getattr(data, "pca_model", None),
        )
        size_mb = round(len(pickle.dumps(index_bundle, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024), 2)

        # Save to separate disk files
        print(f"Saving lightweight Spindle index to {index_save_path}...")
        data.spd_matrices = []  # Clear duplicate raw covariance matrices
        data.U_list = None      # Clear redundant 1.5 GB intermediate ultrametric matrices
        index_save_data = {
            'data': data,
            'dag_dict': dag_dict,
            'config': config,
            'dataset_name': dataset_name,
            'build_time_s': build_time_s,
            'index_size_mb': size_mb
        }
        with open(index_save_path, 'wb') as f:
            pickle.dump(index_save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Saving benchmark raw covariance matrices to {covs_save_path}...")
        covs_save_data = {
            'test_tile_covs': test_tile_covs,
            'train_tile_covs': train_tile_covs,
            'train_idx': train_idx,
            'test_idx': test_idx,
            'dataset_name': dataset_name
        }
        with open(covs_save_path, 'wb') as f:
            pickle.dump(covs_save_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        import gc
        del adata, train_tiles, test_tiles, genes_work, index_bundle
        gc.collect()
        
        print(f"Save complete. True Spindle Index Size: {size_mb} MB, Build Time: {build_time_s:.2f} s")

        scalability_records.append({
            'Dataset': disp_name,
            'Cells': num_cells,
            'Build Time (s)': round(build_time_s, 2),
            'Index Size (MB)': size_mb
        })

        import matplotlib.pyplot as plt
        plt.close('all')
        del index_save_data, covs_save_data, train_tile_covs, test_tile_covs, data, dag_dict, config
        gc.collect()

    if scalability_records:
        df_scale = pd.DataFrame(scalability_records)
        csv_scale_path = summary_out_dir / "index_scalability_summary.csv"
        df_scale.to_csv(csv_scale_path, index=False)
        print(f"\nExported index scalability summary CSV to {csv_scale_path}")
        print(df_scale.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Index datasets and save to disk")
    parser.add_argument('--test', action='store_true', help='Run a quick test on a subset of data (10k spots)')
    parser.add_argument('--dataset-paths', nargs='*', default=None, help='Paths to the datasets')
    parser.add_argument('--train-test-ratio', type=float, default=0.05, help='Train test ratio')
    args = parser.parse_args()

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent

    if args.dataset_paths:
        datasets = {Path(p).stem: Path(p) for p in args.dataset_paths}
    else:
        datasets = {
            "breast_cancer": project_root/ "dataset" / "xenium_human_breast_cancer.h5ad",
            "kidney_nondiseased": project_root / "dataset" / "xenium_human_kidney_nondiseased.h5ad",
            "lymph_node": project_root / "dataset" / "xenium_human_lymph_node.h5ad",
            "lung_cancer": project_root / "dataset" / "xenium_human_lung_cancer.h5ad",
            "skin_melanoma": project_root / "dataset" / "xenium_human_skin_melanoma.h5ad",
            "pancreatic_cancer": project_root / "dataset" / "xenium_human_pancreatic_cancer.h5ad"
        }

    run_indexing_for_datasets(datasets, is_test=args.test, train_test_ratio=args.train_test_ratio)

if __name__ == "__main__":
    main()
