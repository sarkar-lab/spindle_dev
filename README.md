# SPINDLE

**SPINDLE** is a library for fast, approximate nearest-neighbor search over **Symmetric Positive Definite (SPD) covariance matrices** derived from spatial transcriptomics datasets. It builds a block-structured DAG index and performs distance-budgeted search for sub-matrix matching.

---

## Repository Structure

```
spindle_dev/
├── src/spindle_dev/   # Core Python package (index, search, metrics, preprocessing)
├── benchmarks/        # Paper experiments; each writes results/<same name>/
│   ├── paths.py       #   dataset paths (datasets.yaml) + sys.path setup for every benchmarks/ folder
│   ├── common/        #   shared helpers (loaders, tiers glue, ground truth, retrieval metrics)
│   ├── index/         #   index builds and Fig 2 (storage, build cost, cell/gene ladders)
│   ├── whole_tile/    #   Fig 3 (whole-tile search)
│   ├── partial/       #   Fig 4 (partial gene-set search)
│   ├── cross_platform/#   Fig 5 (Xenium <-> Visium)
│   ├── biology/       #   Figs 6–7 inputs (breast niches, composition, gene signatures)
│   ├── checks/        #   one-off checks behind recorded decisions
│   └── legacy/        #   old E4 ANN baselines (until Fig 1 is redrawn)
├── scripts/
│   ├── figure_data/   #   pickles -> small CSVs for the figures (+ bio_configs/)
│   └── figures/       #   one script per paper figure (CSV in, PDF/PNG out) + figstyle.py
├── slurm_jobs/        # Same topic folders as benchmarks/; README.md gives the run order
├── results/           # Result CSVs (README.md maps folder -> experiment -> figure); index pickles not committed
├── figures/           # Figures from scripts/figures/: pdf/ and png/ copies, each with panels/<figure>/<panel>.*
├── examples/          # End-to-end runnable examples
└── dataset/           # Cross-platform .h5ad pair (gitignored)
```

The manuscript lives in a separate Overleaf project; nothing in this repository writes into it.

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

## Paper experiments (`benchmarks/`, `slurm_jobs/`, `scripts/`)

Run everything from the project root, through SLURM (see `slurm_jobs/README.md` for the order):

**Dataset paths** live only in `datasets.yaml` (repository root). Python code reads them through
`benchmarks/paths.py` (`dataset_path(key)`), and slurm scripts through `python benchmarks/paths.py <key | stem | file>`.
To move the data, or add a dataset, edit that file only.


1. `benchmarks/index/build_indexes.py` builds one index per dataset into `results/indexes/`
   (production build: seed 73, 100 held-out tiles; `slurm_jobs/index/submit_build_indexes.sh`).
2. Each experiment script writes `results/<script name>/` (paths under `benchmarks/`):

   | Script | Experiment |
   |---|---|
   | `index/dag_size_table.py`, `tier_build_stats.py`, `dag_cell_ladder.py`, `dag_gene_ladder.py` | storage, build cost, cell and gene ladders (Fig 2) |
   | `whole_tile/dag_holdout.py` | Spindle-DAG retrieval vs Spindle-Exact, c90, K sweep (Fig 3B–C) |
   | `whole_tile/exact_vs_whole.py` | query time: whole-matrix exact vs Spindle-Exact vs Spindle-DAG (Fig 3A) |
   | `whole_tile/whole_cov_baselines.py` | HNSW / PCA + flat / PCA + PQ on whole-matrix logs: memory vs fidelity (Fig 3D) |
   | `whole_tile/neighbour_biology.py` | block vs whole-matrix neighbours against biology (Fig S12) |
   | `partial/partial_search_final.py` | gene-set queries: interval index, padding, imputation vs Spindle-Exact on S (Fig 4, S8) |
   | `cross_platform/cross_platform_tiers.py` | Xenium ↔ Visium search on the tiers: tissue agreement, co-located tile, correction ablation, bias PCA (Fig 5, S9) |
   | `biology/niche_concordance.py`, `composition_concordance.py`, `gene_signature_search.py` | breast biology (E10-lite, E13, E9-lite) |

3. `scripts/figure_data/` (`extract_*.py`, `collect_index_stats.py`, `bio_modules.py`) turns pickles into small CSVs;
   `slurm_jobs/figures/run_make_figures.sbatch` then draws every figure into `figures/`.

---

## Metric Definitions

| Metric | CSV column | Definition |
|---|---|---|
| Recall@ε | `recall_at_eps_{f}` | 1 if Spindle's top hit is within `f × ε` of the exact nearest distance (ε = the true-best niche's ε), averaged over queries |
| Overlap@ε | `overlap_at_eps_{f}` | fraction of the tiles within `f × ε` of the exact nearest distance that Spindle's candidate pool contains |
| Speedup | `speedup`, `mean_speedup` | mean brute-force time / mean Spindle time; brute force recomputes matrix logs per query over the whole matrix |

Every niche's DAG is searched for every query (no niche routing); Stage-1 candidates (up to 400 per
niche) are re-ranked by the exact block log-Euclidean distance. The production search budget
multiplier is 1.0.

---

## Quick Start (Programmatic)

```python
import scanpy as sc
import sys
sys.path.insert(0, 'src')
from examples.run_single_dataset import create_index

adata = sc.read_h5ad('xenium_human_breast_cancer.h5ad')
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
