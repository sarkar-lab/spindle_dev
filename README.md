# SPINDLE

**SPINDLE** is a library for fast, approximate nearest-neighbor search over **Symmetric Positive Definite (SPD) covariance matrices** derived from spatial transcriptomics datasets. It builds a block-structured DAG index and performs distance-budgeted search for sub-matrix matching.

---

## Repository Structure

```
spindle_dev/
├── src/spindle_dev/       # Core Python package (index, search, metrics, preprocessing)
├── benchmarks/            # Paper benchmark scripts
│   ├── index_datasets.py          # Pre-build and serialize dataset indices
│   ├── holdout_validation.py      # Blind holdout split-test benchmark (ANN accuracy & speed)
│   ├── partial_panel_search.py    # Partial gene panel search benchmark
│   ├── cross_modal_search.py      # Cross-platform Xenium ↔ Visium search benchmark
│   ├── gene_signature_search.py   # Gene signature–driven niche discovery benchmark
│   └── data_helpers.py            # Shared data loading, indexing, and evaluation utilities
├── examples/              # End-to-end runnable examples
│   ├── run_single_dataset.py      # Build index & run sanity check on a single H5AD file
│   └── run_batch_datasets.py      # Batch processing of multiple datasets
├── scripts/               # Figure generation scripts
│   └── generate_figure_1.py
├── notebooks/             # Analysis notebooks
│   └── figure_1_walkthrough.ipynb
├── results/               # Benchmark output CSVs and figures (generated, not committed)
├── figures/               # Final paper figures
├── dataset/               # Raw .h5ad data files (gitignored — store externally)
├── paper/                 # LaTeX source, PDF draft (gitignored — track in Overleaf)
└── hpc/                   # HPC/SLURM job scripts (gitignored)
```

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Core Package (`src/spindle_dev/`)

| Module | Purpose |
|---|---|
| `preprocessing.py` | Spatial tiling (quadtree), per-tile covariance estimation |
| `index.py` | Block-cluster DAG index construction |
| `search.py` | Budget-pruned best-first search over the DAG |
| `interval_index.py` | Dyadic interval index for partial gene-panel queries |
| `metrics.py` | Log-Euclidean distance, SPD ↔ correlation utilities |
| `partial_search.py` | Partial-query padding and imputation strategies |
| `go_score.py` | GO/pathway enrichment scoring utilities |
| `test.py` | Sanity-check search helpers |
| `utils.py` | Serialization, deterministic config, SPD log/exp |
| `plotting.py` | Visualization utilities |
| `typing.py` | Shared type definitions and dataclasses |

---

## Benchmark Scripts (`benchmarks/`)

All benchmark scripts can be run from the **project root**. Each expects pre-built index files in `results/split_test_indexed/` (produced by `index_datasets.py`).

### 1. Build dataset indices first

```bash
python benchmarks/index_datasets.py --datasets kidney breast lung
```

This reads `.h5ad` files from `dataset/`, builds the Spindle index, and saves `.pkl` files to `results/split_test_indexed/`.

### 2. Holdout validation (ANN accuracy + speed)

```bash
python benchmarks/holdout_validation.py --datasets kidney lung breast
```

Outputs per-query metrics to `results/split_test/<dataset>/` with canonical columns:
`recall_at_1`, `overlap_at_5`, `overlap_at_10`, `overlap_at_20`, `spindle_time_ms`, `brute_force_time_ms`, `speedup`

### 3. Partial gene panel search

```bash
python benchmarks/partial_panel_search.py --top-k 50 --num-queries 5
```

Outputs `results/partial_search/<dataset>/benchmark_interval_metrics.csv` with columns:
`recall_at_1`, `recall_at_5`, `overlap_at_10`, `overlap_at_20`, `spindle_time_ms`, `brute_force_time_ms`

### 4. Cross-modal search (Xenium ↔ Visium)

```bash
python benchmarks/cross_modal_search.py
```

Outputs `results/cross_modal_search/benchmark_summary.csv` with columns:
`recall_at_1`, `overlap_at_5`, `overlap_at_10`, `overlap_at_20`

### 5. Gene signature–driven niche discovery

```bash
python benchmarks/gene_signature_search.py
```

Outputs `results/gene_list_search/benchmark_metrics.csv` with columns:
`background_score`, `enrichment_score_at_5/10/20/50`, `enrichment_lift_at_5/10/20/50`

---

## Metric Definitions

All benchmark CSVs use a **standardized set of metric column names**. The `@` notation is for paper text only — CSV columns use `_at_`.

| Metric | CSV Column | Definition |
|---|---|---|
| Recall@1 | `recall_at_1` | Fraction of queries where the true nearest neighbor is the #1 returned result |
| Recall@K | `recall_at_{K}` | Fraction of queries where the true NN appears in the top-K results |
| Overlap@K | `overlap_at_{K}` | `|spindle_top_K ∩ exact_top_K| / K` — overlap between Spindle's top-K and the brute-force ground-truth top-K |
| Spindle Time | `spindle_time_ms` | Query time (ms): cluster assignment + DAG search + re-ranking |
| Brute-Force Time | `brute_force_time_ms` | Time (ms) for exhaustive log-Euclidean scan in the predicted niche |
| Speedup | `speedup` | `brute_force_time_ms / spindle_time_ms` |
| Enrichment Score@K | `enrichment_score_at_{K}` | Mean pathway score of cells in the top-K retrieved tiles |
| Background Score | `background_score` | Dataset-wide mean pathway score (baseline) |
| Enrichment Lift@K | `enrichment_lift_at_{K}` | `enrichment_score_at_K / background_score` |

---

## Quick Start (Programmatic)

```python
import scanpy as sc
import sys
sys.path.insert(0, 'src')
from examples.run_single_dataset import create_index

adata = sc.read_h5ad('dataset/xenium_human_breast_cancer.h5ad')
create_index(adata, 'my_index/', resolution=0.5, min_final_size=15,
             top_genes=800, all_genes=True, max_queries=100)
```

```python
import numpy as np
import sys
sys.path.insert(0, 'src')
import spindle_dev
from spindle_dev import index as sd_index, search as sd_search

bundle = sd_index.load_index('my_index/spindle.pkl')
cluster_id = list(bundle.dag_dict.keys())[0]
index_handle = bundle.dag_dict[cluster_id]

query_spd = np.eye(bundle.pca_model.components_.shape[1])  # replace with real SPD
budget = 0.5
results = sd_search.query_index(index_handle, query_spd, budget, config=sd_search.SearchConfig(max_results=5))
print(results)
```

---

## Interval Index (Partial Gene Panel Queries)

The interval index enables searching with a **subset of genes** against a full-panel reference atlas.

```python
from spindle_dev.interval_index import build_all_interval_indices, query_interval_index
from spindle_dev.typing import IndexConfig

config = IndexConfig(
    epsilon_dict={...},
    epsilon_block_wise_dict={...},
    use_interval_index=True,
    interval_mode='dyadic',   # 'all' | 'dyadic' | 'fixed'
    interval_max_iters=5,
)

ivl_idx = build_all_interval_indices(data, config)

hits = query_interval_index(
    ivl_idx,
    cluster_id=cluster_id,
    block_index=0,
    interval=(2, 8),          # gene indices [2, 8) in permuted block space
    query_spd_sub=A_interval,
    top_k=5,
)
```
