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
# data: dataset/cross_modal/{visium,xenium}_rotated.h5ad (or --visium-path/--xenium-path)
python benchmarks/cross_modal_search.py --seed 0            # one seed, both directions
./slurm_jobs/submit_cross_modal_search.sh                   # seeds 0-4 via SLURM
python benchmarks/multiseed_cross_modal_search.py           # aggregate -> summary.csv
```

Outputs `results/cross_modal_search/seed_<n>/{x2v,v2x}_query_metrics.csv` and the
mean ± s.d. `results/cross_modal_search/summary.csv` with columns
`mean_/sd_` × `recall_at_eps_{0.1,0.5}`, `overlap_at_eps_{0.5,1.0}`, `mean_speedup`

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
| Recall@ε | `recall_at_eps_{frac}` | Fraction of queries where Spindle's top-1 result is within `frac × niche_epsilon` of the true nearest-neighbor distance (niche-relative distance tolerance, not an exact/tied match like `recall_at_1`) |
| Overlap@ε | `overlap_at_eps_{frac}` | `|spindle_near ∩ true_near| / |true_near|`, where `true_near`/`spindle_near` are the sets of tiles within `frac × niche_epsilon` of the true best distance — a distance-window analogue of `overlap_at_K` |
| Niche-coverage diagnostic | `true_near_frac_of_niche_{frac}` | Fraction of the query's true-best-match niche that falls within the `frac × niche_epsilon` band — if this approaches 1.0, the search budget is effectively covering the whole niche rather than a meaningful neighborhood |
| Dataset-coverage diagnostic | `true_near_frac_of_dataset_{frac}` | Fraction of the *entire indexed dataset* (all niches) that falls within the band — if this approaches 1.0, the tolerance band itself is essentially meaningless |

`frac` ranges over `EPSILON_TOLERANCE_FRACTIONS = [0.1, 0.25, 0.5, 1.0]` (see `benchmarks/holdout_validation.py`). `niche_epsilon` is `config.epsilon_dict[true_best_niche]` — the query's true-best-match niche's own per-niche distance scale, the same one already used to size that niche's search budget (`budget = epsilon * num_blocks * budget_multiplier * niche_scale_factor`), so the tolerance is comparable across datasets/niches with different absolute distance scales. (There is no single "predicted niche" per query anymore — see below.)

### Search design: every niche is searched, not just a predicted one

Earlier versions of this benchmark routed each query to a single niche via `search.assign_clusters_to_new_spds()` (a KNN-vote in a latent PCA/ultrametric embedding) and searched only that niche's DAG. Diagnosis found this embedding disagrees substantially with the actual search/distance metric on datasets with many niches — e.g. 60% of `lymph_node` queries and 86% of `brain_cancer` queries had their true nearest neighbor in a *different* niche than the one routed to, an unrecoverable recall failure under single-niche search. Neither widening top-K niche-probing nor increasing PCA embedding dimensionality fixed this (more PCA dims made separability *worse* — a curse-of-dimensionality effect), and coarsening the clustering enough to fix it would have gutted per-niche search efficiency.

Since niches exist to group SPD matrices that share a block-diagonal permutation (a compression/indexing property), not to reduce the search space, `benchmarks/holdout_validation.py`'s `perform_search()` now searches **every niche's DAG for every query** (each with its own permutation/budget) and merges Stage-1 candidates before Stage-2 re-ranking, which is niche-aware per candidate. This removes the misrouting failure mode entirely (`recall_at_1` on a 5-query breast_cancer smoke test went from 0.4 to 1.0) while still being far cheaper than true brute force, since each niche's DFS is still budget-pruned. The brute-force baseline itself was also corrected to scan the *entire* indexed dataset (previously it was restricted to the same predicted niche, which handed it an unearned shortcut and understated Spindle's real speedup — by 4-16× depending on how many niches a dataset has).

### Budget multiplier

All benchmark runs use a single, shared `budget_multiplier = 0.015` for every dataset, passed via `--budget-mult`/`--budget-mults` to `benchmarks/holdout_validation.py` / `benchmarks/budget_sweep_holdout.py`. This was chosen from a recall/overlap-vs-speedup sweep (`benchmarks/budget_sweep_holdout.py`, log-spaced `budget_multiplier` grid, `DEFAULT_BUDGET_MULTS`, at a fixed `top_c=400` Stage-2 re-rank cap per niche) across all 8 datasets; see `results/budget_sweep_holdout/sweep_summary.csv` and `figures/fig_budget_sweep.{pdf,png}` / `figures/fig_budget_sweep_candidate_counts.{pdf,png}` (`scripts/generate_budget_sweep_figure.py`) for the full tradeoff curves.

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
