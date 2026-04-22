# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Spindle** is a Python library for indexing and searching symmetric positive definite (SPD) sub-matrices derived from spatial-omics datasets. It builds a block-structured DAG index over SPD covariance matrices computed from spatial tiles, enabling fast budget-pruned nearest-neighbor search.

Published docs: https://www.hiraksarkar.com/spindle_dev/

## Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

## Common Commands

### Build documentation locally
```bash
python -m sphinx -b html docs_src docs/_build/html -a
```

### Run a typical workflow (end-to-end)
```bash
python ISMB_notebook/spindle_xenium_single.py path/to/sample.h5ad \
  --top-genes 800 --all-genes --max-queries 100
```

No formal test suite exists; sanity testing is done via `spindle_dev.test.run_sanity_search()`.

## Architecture

### Public API (`src/spindle_dev/__init__.py`)

The two main entry points:
- **`build_index(spatial_data, config) -> IndexHandle`** — constructs the DAG index
- **`query_index(index_handle, query_spd, budget, config) -> SearchResults`** — searches the index

### Core Modules

**`typing.py`** — Dataclasses for the entire system:
- `IndexConfig` — index construction parameters (epsilons, ordering, clustering method)
- `BlockClusterNode` — a node in the DAG (block_index, global_node_id, children)
- `DatasetIndex` — serialized bundle: dag_dict + metadata + PCA model + embeddings
- `SearchConfig` — search parameters (max_results, budget limits, debug flags)

**`preprocessing.py`** — Spatial data preparation:
- `build_quadtree_tiles(coords)` — recursively tiles spatial coordinates into `QuadTile` objects
- `build_tile_covs_full_serial()` — computes per-tile SPD covariance matrices
- `topvar_genes()` — selects top-variance genes

**`metrics.py`** — SPD distance math:
- `log_euclidean_distance()` — primary metric between SPD matrices (normalized by √p)
- `log_spd()` — matrix log via eigendecomposition
- `spd_to_correlation()` / `correlation_to_spd()` — SPD ↔ correlation conversions
- `build_ultrametrics()` / `consensus_tree_from_ultrametrics()` — hierarchical clustering support

**`index.py`** — Index construction (largest module, ~2400 lines):
- `ProcessedData` — central data container; holds tiles, SPD matrices, cluster assignments, block structure
  - `.reduce_dim()` — PCA + optional UMAP on flattened SPD features
  - `.cluster_spds()` — Leiden clustering in SPD feature space
  - `.assign_label_to_spots()` — maps original spatial spots → cluster labels
  - `.get_adaptive_runs()` — detects block boundaries adaptively
- `choose_adaptive_epsilons()` — per-block distance thresholds via k-means on target cluster size
- `index_spds(data, config)` — **core function**: builds the block-DAG, returns `dag_dict`
- `save_index()` / `load_index()` — pickle serialization

**`search.py`** — DAG traversal (~1030 lines):
- `search_index()` — best-first traversal with budget pruning; uses binary search over sorted block-cluster means

**`test.py`** — Sanity checks:
- `run_sanity_search()` — end-to-end validation: builds ground-truth paths, queries the index, verifies recovery

**`plotting.py`** — Visualization utilities for index structure and results

**`go_score.py`** — Gene ontology scoring/analysis

### Typical Workflow

```
1. Load AnnData (sc.read_h5ad)
2. build_quadtree_tiles(coords)          → QuadTile list
3. topvar_genes(adata, G=800)            → gene subset
4. build_tile_covs_full_serial(...)      → per-tile SPD matrices
5. ProcessedData(tiles, covs, genes)
6. data.reduce_dim(num_pca_components=30)
7. data.cluster_spds(method='leiden')
8. data.assign_label_to_spots()
9. data.get_adaptive_runs(find_blocks=True)
10. choose_adaptive_epsilons(data, ...)
11. index_spds(data, config)             → dag_dict
12. save_index(data, dag_dict, path)
13. run_sanity_search(data, dag_dict, ...) → validation CSV
```

### Key Design Concepts

- **Block-structured DAG**: SPD matrices are grouped into blocks (detected via adaptive consensus clustering). Each block becomes a DAG layer with parent-child relationships across layers.
- **Per-block epsilons**: Distance thresholds differ per block/cluster, chosen to hit a target neighborhood size (`k_target_per_block`).
- **Budget-pruned search**: `search_index()` tracks a query budget and prunes subtrees when the budget is exhausted, trading recall for speed.
- **Determinism**: `DeterministicConfig` / `configure_determinism()` set NumPy seeds for reproducible clustering runs.

## Documentation

Sphinx + MyST (Markdown). Source in `docs_src/`, built output in `docs/`. CI auto-deploys to GitHub Pages on push to `main` (`.github/workflows/sphinx_docs.yml`).
