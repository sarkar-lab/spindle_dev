"""Build and pickle one Spindle index per dataset (results/indexes/).

For each dataset: quadtree tiles, top-variance genes, per-tile covariances, a
seeded held-out split (--n-holdout tiles, or --train-test-ratio), niche
clustering, block detection, adaptive epsilons and the block DAG. Writes
``<name>_spindle_index.pkl`` (index + config) and ``<name>_raw_covariances.pkl``
(train/test covariances and split indices), and a RunLogger JSON (wall time,
peak RSS) to results/index_stats/build_run_logs/. Existing indexes are skipped.
The production build is seed 73 with 100 held-out tiles
(slurm_jobs/submit_build_indexes.sh); the per-seed builds come from
holdout_search.py.
"""

import argparse
import gc
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import scanpy as sc

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import paths
import spindle_dev
import spindle_dev.index as index
import spindle_dev.preprocessing as preprocessing
import spindle_dev.typing as typing
from run_logging import RunLogger

INDEX_DIR = project_root / "results" / "indexes"
RUN_LOG_DIR = project_root / "results" / "index_stats" / "build_run_logs"


def prepare_to_index(adata, max_genes=paths.DEFAULT_MAX_GENES):
    """
    Prepare standard data object for indexing (the top min(max_genes, n_genes) variable genes).
    """
    coords = adata.obsm["spatial"]
    tiles = preprocessing.build_quadtree_tiles(coords, max_pts=200, min_side=0.0, max_depth=40)
    num_genes = adata.n_vars
    genes_work, gene_idx = spindle_dev.preprocessing.topvar_genes(adata, G=min(max_genes, num_genes))
    tile_covs = spindle_dev.preprocessing.build_tile_covs_full(adata, tiles, gene_idx, n_jobs=8, eps=1e-6)

    return tiles, tile_covs, genes_work


def run_index(tiles, tile_covs, genes_work, adata, resolution=0.2, min_final_size=20, max_niche_size=1000,
              random_state=0):
    """
    Run indexing workflow.

    ``random_state`` seeds PCA/UMAP and Leiden clustering (default 0 keeps the
    production behaviour; cross_modal_search.py varies it per seed).
    """
    data = index.ProcessedData(tiles, tile_covs, genes_work, adata.n_obs)
    num_pca = min(30, len(tiles) - 1)
    if num_pca < 2:
        num_pca = 2
    data.reduce_dim(num_pca_components=num_pca, n_components=2, do_umap=True, random_state=random_state)
    data.cluster_spds(
        cluster_distance="tree", cluster_method="leiden", resolution=resolution,
        adaptive_resolution=True, max_niche_size=max_niche_size, random_state=random_state
    )
    data.assign_label_to_spots()
    data.get_corr_mean_by_cluster()
    out_dict = data.get_adaptive_runs(find_blocks=True, with_size_guard=True, min_final_size=min_final_size, max_final_size=100)
    return data, out_dict


def load_and_split_data(adata_path, test_ratio=0.05, seed=42, n_holdout=None):
    print(f"Reading data from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    if 'Cluster' in adata.obs.columns:
        adata = adata[adata.obs.loc[adata.obs.Cluster != "Unlabeled"].index, :].copy()

    max_genes = paths.dataset_max_genes(str(adata_path))  # datasets.yaml
    print(f"Preparing data for indexing (top {min(max_genes, adata.n_vars)} genes)...")
    tiles, tile_covs, genes_work = prepare_to_index(adata, max_genes=max_genes)

    np.random.seed(seed)
    num_total_tiles = len(tiles)
    if n_holdout is not None:
        # Fixed-count holdout takes priority over test_ratio -- e.g. "hold out
        # exactly 100 tiles" rather than a percentage of a dataset-dependent
        # tile count. Capped so tiny datasets always keep at least one training tile.
        num_test = min(n_holdout, max(0, num_total_tiles - 1))
        if num_test < n_holdout:
            print(f"Warning: requested n_holdout={n_holdout} exceeds available tiles "
                  f"({num_total_tiles}); capping holdout at {num_test}.")
        elif num_test > 0.3 * num_total_tiles:
            print(f"Warning: n_holdout={n_holdout} is {num_test / num_total_tiles:.1%} of "
                  f"this dataset's {num_total_tiles} tiles -- unusually large a holdout fraction.")
    else:
        num_test = int(num_total_tiles * test_ratio)

    all_indices = np.arange(num_total_tiles)
    test_idx = np.random.choice(all_indices, size=num_test, replace=False)
    train_idx = np.setdiff1d(all_indices, test_idx)

    train_tiles = [tiles[i] for i in train_idx]
    train_tile_covs = [tile_covs[i] for i in train_idx]
    test_tiles = [tiles[i] for i in test_idx]
    test_tile_covs = [tile_covs[i] for i in test_idx]

    print(f"Total tiles: {num_total_tiles} | Training/Indexed: {len(train_tiles)} | Held out/Testing: {len(test_tiles)} "
          f"(seed={seed}, n_holdout={n_holdout})")

    return adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx


def configure_and_build_dag(data):
    print("Configuring adaptive epsilons for blocks...")
    epsilon_block_wise_dict = {}
    epsilon_dict = {}
    for cluster_id in set(data.labels):
        eps_per_block, eps_elbow_per_block, eps = index.choose_adaptive_epsilons(data, cluster_id, k_target_per_block=64)
        epsilon_block_wise_dict[int(cluster_id)] = eps_per_block
        epsilon_dict[int(cluster_id)] = eps

    # Floor each niche's budget-sizing epsilon (config.epsilon_dict, used for
    # `budget = epsilon * num_blocks * budget_multiplier * f` in
    # holdout_core.py, and for the Recall@eps/Overlap@eps tolerance
    # bands) at the dataset-wide median across niches. epsilon_dict reflects
    # how tightly a niche's OWN members cluster together -- for a small,
    # homogeneous niche that can be far smaller than the typical distance a
    # query actually has to travel (in log-Euclidean space) to reach that
    # niche's cluster region, which has nothing to do with intra-niche
    # spread. Left unfloored, such a niche's search budget collapses to
    # near-zero regardless of budget_multiplier, and search_index() returns
    # literally zero candidates for it -- confirmed directly on
    # lung_cancer's 22-tile niche (epsilon=0.94 vs ~7-8 for its other
    # niches) and lymph_node's 15-tile niche (epsilon=2.33 vs ~6-8), where
    # every query whose true nearest neighbor fell in that niche failed
    # unrecoverably at every swept budget. The median (not max) is used so
    # well-calibrated niches aren't dragged up by one unusually loose
    # outlier niche, while still guaranteeing no niche is starved below the
    # dataset's typical scale.
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


def run_indexing_for_datasets(datasets, train_test_ratio=0.05, seed=42, n_holdout=None):
    INDEX_DIR.mkdir(exist_ok=True, parents=True)

    for dataset_name, adata_path in datasets.items():
        print(f"\n{'=' * 80}\nProcessing dataset: {dataset_name}\n{'=' * 80}\n")
        index_save_path = INDEX_DIR / f"{dataset_name}_spindle_index.pkl"
        covs_save_path = INDEX_DIR / f"{dataset_name}_raw_covariances.pkl"

        if index_save_path.exists():
            print(f"Index for {dataset_name} already exists at {index_save_path}; skipping.")
            continue
        if not adata_path.exists():
            print(f"Dataset not found at {adata_path}. Skipping.")
            continue

        with RunLogger(dataset_name=dataset_name, stage="index_build", out_dir=RUN_LOG_DIR,
                       seed=seed, n_holdout=n_holdout):
            adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = load_and_split_data(
                adata_path, test_ratio=train_test_ratio, seed=seed, n_holdout=n_holdout
            )

            print("Running index...")
            t0 = time.perf_counter()
            data, out_dict = run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15, max_niche_size=1000)
            dag_dict, config = configure_and_build_dag(data)
            build_time_s = time.perf_counter() - t0

            # Spindle index size: the DatasetIndex bundle, without raw covariance matrices
            index_bundle = typing.DatasetIndex(
                dag_dict=dag_dict,
                metadata=data.metadata,
                latent=data.latent,
                labels=data.labels,
                pca_model=getattr(data, "pca_model", None),
            )
            size_mb = round(len(pickle.dumps(index_bundle, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024), 2)

            print(f"Saving lightweight Spindle index to {index_save_path}...")
            data.spd_matrices = []  # raw covariances are saved separately below
            data.U_list = None      # intermediate ultrametric matrices (~1.5 GB), not needed for search
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

            print(f"Save complete. Spindle index size: {size_mb} MB, build time: {build_time_s:.2f} s")

        del adata, train_tiles, test_tiles, genes_work, index_bundle
        del index_save_data, covs_save_data, train_tile_covs, test_tile_covs, data, dag_dict, config
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Index datasets and save to disk")
    parser.add_argument('--dataset-paths', nargs='+', required=True,
                        help='h5ad paths; the file stem names the index')
    parser.add_argument('--train-test-ratio', type=float, default=0.05, help='Held-out fraction (if no --n-holdout)')
    parser.add_argument('--seed', type=int, required=True, help='Seed for the train/test tile split (production: 73)')
    parser.add_argument('--n-holdout', type=int, default=None,
                        help='Fixed number of tiles to hold out per dataset. Takes priority '
                             'over --train-test-ratio when set.')
    args = parser.parse_args()

    datasets = {Path(p).stem: Path(p) for p in args.dataset_paths}
    run_indexing_for_datasets(datasets, train_test_ratio=args.train_test_ratio,
                              seed=args.seed, n_holdout=args.n_holdout)


if __name__ == "__main__":
    main()
