
## Background

Sprindle is a library for indexing and searching symmetric positive definite (SPD)
sub-matrices derived from spatial-omics datasets. Please refer to the (documentation page)[www.hiraksarkar.com/spindle_dev/].

The core idea is to build a block-structured DAG index over SPD
matrices (or their correlation equivalents) and then perform
budget-pruned search over this graph for fast matching of query
sub-matrices.

## Modules

- src/spindle_dev/preprocessing.py – interfaces only; define how raw spatial data is
  converted into a SpatialDataset with points and SPD matrices.
- src/spindle_dev/index.py – builds a DAG index over block clusters of SPD matrices
  according to the spec in .github/copilot-instructions.md.
- src/spindle_dev/search.py – traverses the index given a query SPD sub-matrix and a
  distance budget, returning matching SPD IDs and paths.
- src/spindle_dev/metrics.py – implements log-Euclidean distance and SPD
  ↔ correlation conversions.
- src/spindle_dev/utils.py – serialization helpers (save_index / load_index),
  deterministic config, and logging utilities.

## Public API

The intended "front door" for users and higher-level code is:

- `build_index(spatial_data, config) -> IndexHandle` (from `index.py`)
- `query_index(index_handle, query_spd, budget, config) -> SearchResults`
  (from `search.py`)

All other functions and data structures are considered internal
implementation details.


## **Index And Search**

- **Prerequisites:** Ensure you run from the project root or add `src` to `PYTHONPATH` so `spindle_dev` is importable. Install runtime deps (e.g. `scanpy`, `numpy`, `pandas`, `scikit-learn`, `umap-learn`).

- **Quick CLI (recommended for single H5AD files):** Use the helper script to build an index from an AnnData `.h5ad` file and run a small sanity search. Example:

```bash
# from project root
python ISMB_notebook/spindle_xenium_single.py path/to/sample.h5ad \
  --top-genes 800 --all-genes --max-queries 100
```

- **Programmatic: Create index from an `AnnData` object**

```python
import scanpy as sc
from ISMB_notebook.spindle_xenium_single import create_index

# load your AnnData
adata = sc.read_h5ad('path/to/sample.h5ad')
index_path = 'path/to/sample_index'

# create_index will build the tiles, compute covariances, cluster, choose epsilons,
# save a serialized index at `<index_path>/spindle.pkl` and run a small sanity test.
create_index(adata, index_path, resolution=0.5, min_final_size=15,
             top_genes=800, all_genes=True, max_queries=100)
```

- **Programmatic: Load saved index and run a search**

```python
import numpy as np
import spindle_dev
from spindle_dev import index as sd_index, search as sd_search

# load the saved DatasetIndex bundle
bundle = sd_index.load_index('path/to/sample_index/spindle.pkl')

# inspect available cluster ids (each maps to an IndexHandle)
print('cluster ids:', list(bundle.dag_dict.keys()))

# pick an IndexHandle (e.g. first key) and prepare a SPD query matrix
cluster_id = list(bundle.dag_dict.keys())[0]
index_handle = bundle.dag_dict[cluster_id]

# Replace the following with a real SPD query (e.g. a tile covariance or correlation)
query_spd = np.eye(bundle.pca_model.components_.shape[1])  # placeholder; use a real SPD
budget = 0.5
cfg = sd_search.SearchConfig(max_results=5)

results = sd_search.query_index(index_handle, query_spd, budget, config=cfg)
print(results)
```

- **Notes:**
  - The script <a href="ISMB_notebook/spindle_xenium_single.py">ISMB_notebook/spindle_xenium_single.py</a> contains a convenient `create_index` wrapper that builds the index and runs a sanity check. Use it for quick experiments.
  - Saved indexes are written with `spindle.pkl` and can be loaded with `spindle_dev.index.load_index` (returns a `DatasetIndex` bundle containing `dag_dict` with `IndexHandle` objects).
  - `sd_search.query_index` expects an `IndexHandle`, a SPD matrix (`numpy.ndarray`) and a numeric `budget`.

## **Detailed: create_index steps**

- **Entry point:** `create_index` (defined in <a href="ISMB_notebook/spindle_xenium_single.py">ISMB_notebook/spindle_xenium_single.py</a>).
- **Load coordinates:** extract `coords = adata.obsm['spatial']` and build spatial tiles with `preprocessing.build_quadtree_tiles(coords, ...)`.
- **Filter & reindex tiles:** remove tiny tiles, then call `preprocessing.reindex_tiles(tiles)`.
- **Select genes:** choose `top_genes` (or all genes) via `preprocessing.topvar_genes(adata, G=num_genes)` producing `genes_work, gene_idx`.
- **Compute per-tile covariances:** `preprocessing.build_tile_covs_full_serial(adata, tiles, gene_idx, eps=1e-6)` returns `tile_covs` used to create `index.ProcessedData(tiles, tile_covs, genes_work, adata.n_obs)`.
- **Dimensionality reduction:** if PCA/UMAP not present, call `data.reduce_dim(num_pca_components=30, n_components=2, do_umap=True)` to compute latent features and store `pca_model`.
- **Clustering:** call `data.cluster_spds(cluster_distance='tree', cluster_method='leiden', resolution=resolution)` to assign `data.labels` and compute per-cluster consensus trees and permutations.
- **Assign spot labels:** `data.assign_label_to_spots()` maps original spot indices to cluster labels.
- **Cluster means / correlation means:** `data.get_corr_mean_by_cluster()` computes `data.R_mean_list` used for block detection.
- **Adaptive block detection:** `data.get_adaptive_runs(find_blocks=True, with_size_guard=True, min_final_size=min_final_size, max_final_size=100)` returns candidate block runs.
- **Per-cluster epsilon selection:** for each cluster call `index.choose_adaptive_epsilons(data, cluster_id, k_target_per_block=64)` to get `eps_per_block, eps_elbow_per_block, eps` and populate `IndexConfig` fields `epsilon_dict` and `epsilon_block_wise_dict`.
- **Index construction:** call `index.index_spds(data, config=config)` to produce `dag_dict, stat, dist_list` (the block-DAG index).
- **Serialize index:** `index.save_index(data, dag_dict, index_path + '/spindle.pkl')` writes the dataset bundle.
- **Sanity test & artifacts:** run `test.run_sanity_search(data, dag_dict, config, search_cfg, max_queries=max_queries)`, write `index_stats.txt` with timing, and save `sanity_test_results.csv` in the index folder.

- **What to inspect after building:**
  - `index_path/spindle.pkl` — load with `spindle_dev.index.load_index` to get a `DatasetIndex` bundle containing `dag_dict` and `IndexHandle` objects.
  - `index_path/index_stats.txt` — index build time.
  - `index_path/sanity_test_results.csv` — sanity-check search records.

