# Spindle

## Installation

Python 3.9 or later. gcc is needed once, to build the search kernels. Create and activate a virtual environment, then
install the dependencies and the package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Background

Spindle is a library for indexing and searching the gene–gene covariance matrices of spatial-omics tiles: given a
query, it finds the tiles of a section whose genes co-vary most like it. Please refer to the
[documentation page](https://www.hiraksarkar.com/spindle_dev/).

A query can be a whole tile, a tile restricted to a gene set, or a gene list alone. Two indexes answer them:

- **Spindle** (`Spindle`, `SpindlePartial`): the fast index, which returns close matches in a fraction of the time
  and memory of an exact search.
- **Spindle-naive** (`SpindleNaive`, `SpindleNaivePartial`): the exact search over every tile, slower and used as the
  reference.

## Repository

```
src/spindle_dev/            the package
  preprocessing.py            tiling, gene selection, shrunk tile covariances, train / held-out split
  layout.py                   covariance niches, gene order and blocks per niche (shared by all indexes)
  spindle.py                  Spindle, complete queries
  spindle_partial.py          Spindle, partial queries (SpindlePartial)
  naive.py                    Spindle-naive (SpindleNaive, SpindleNaivePartial)
  queries.py                  partial queries from gene names (gene_indices, template_query, tile_query)
  kernels.c, kernels.py       compiled searches
  metrics.py, utils.py        shrinkage, matrix logarithms; save / load
examples/
  spindle_quickstart.ipynb    build both indexes on one section, run the three query types, map the results
  run_single_dataset.py       build every index on one dataset and query it
  check_indexes.py            sanity checks (exact self-queries, compiled == reference, lower bounds)
experiments/                the paper's results (see "Reproducing the paper")
  common/                     shared modules and Slurm scripts
  compute/                    R0-R11: result notebooks, writing to results/
  figures/                    figure and table notebooks, writing to figures/
results/<topic>/            CSV / NPZ inputs of the figures, with a manifest.json per topic
figures/<figure>/           each figure (PDF + PNG) and its panels; supp_tables/ holds the LaTeX tables
datasets.yaml, paths.py     dataset locations (the only place a path is written) and their reader
```

## Use

```python
import spindle_dev as sd

adata = sd.load_adata("section.h5ad")              # raw counts in X, cell centroids in obsm["spatial"]
tiles, covs, genes = sd.prepare_tiles(adata, max_genes=800)
train, test = sd.split_tiles(len(tiles), seed=0, test_ratio=0.1)
train_covs = [covs[i] for i in train]
data = sd.build_layout([tiles[i] for i in train], train_covs, genes, adata.n_obs)

# complete queries
naive = sd.SpindleNaive(data, train_covs)
spindle = sd.Spindle.from_naive(naive, data)
query = covs[test[0]]
rows, scores, seconds = naive.search(query, c=10)   # exact top 10 by d_B
rows, scores, seconds = spindle.search(query)       # Spindle's tiles, best first (g = 0.25)

# partial queries: a held-out tile's covariance on a gene subset
naive_partial = sd.SpindleNaivePartial(data, train_covs)
spindle_partial = sd.SpindlePartial.from_naive_partial(naive_partial, data)
idx, found, missing = sd.gene_indices(genes, ["CD68", "CD163", "CD14", "AIF1", "MNDA"])
cov_s, idx = sd.tile_query(query, idx)
rows, scores, seconds = naive_partial.search(cov_s, idx, c=10)   # exact top 10 by d_B^S
rows, scores, seconds = spindle_partial.search(cov_s, idx)       # g = 1

# partial queries from a gene list alone: the co-expression template
omega = sd.variance_quantiles(train_covs)           # keep it beside the index
cov_s, idx = sd.template_query(omega, idx)          # correlation 0.5, 90th-percentile variances
rows, scores, seconds = spindle_partial.search(cov_s, idx)
```

`rows` are positions among the index tiles (`train[rows]` gives tile numbers). A query covariance built elsewhere must
be shrunk first (`metrics.shrink_cov`), as the index tiles are. `sd.save_index` / `sd.load_index` pickle an index or a
layout. `python examples/run_single_dataset.py <key | file.h5ad>` runs all of the above on one dataset and reports
Spindle's accuracy against Spindle-naive. `examples/spindle_quickstart.ipynb` does the same in a notebook, with maps of what
each query (whole tile, gene set from a tile, gene list) returns; set `DATASET` in its first cell.

## Reproducing the paper

The datasets are the public 10x Xenium sections listed in `datasets.yaml` (eight indexed sections, a tonsil section
for cell scaling, and a registered Xenium / Visium breast pair). Point the `dirs` entries of `datasets.yaml` at your
copies. The experiments also need `pandas`, `matplotlib`, `threadpoolctl` and `jupyter_client`, with the environment
registered as the Jupyter kernel `spindle_env`. The method is frozen at git tag `method-v3`, and every result row
carries the tag it was made with.

1. **Index cache** (`experiments/compute/R0_index_cache.ipynb`): builds the layout and all four indexes for every
   dataset and seeds 0–4 (10 % of tiles held out at random), times the builds, and stages the per-cell inputs into
   `results/_cache/` (not tracked). This is the only step that needs a cluster: it submits Slurm jobs, and brain
   cancer needs about 128 GB.
2. **Results** (`experiments/compute/R1`–`R11`): each notebook writes CSVs and a manifest to `results/<topic>/` and
   skips outputs that already exist (`FORCE = True` redoes them). R1–R3 and R5–R10 run in 16 GB through
   `sbatch experiments/common/run_nb.sbatch experiments/compute/<notebook>.ipynb`. The timings must run alone on an
   idle node: R4 (one core) through `experiments/common/timing_nb.sbatch`, R11 (1–64 cores) through
   `experiments/common/parallel_nb.sbatch`.
3. **Figures and tables** (`experiments/figures/`): `python experiments/common/run_nb.py experiments/figures/<notebook>.ipynb`
   writes each panel and the assembled figure to `figures/`.

| Manuscript | Notebook | Inputs |
|---|---|---|
| Fig 1F, G | `F1_overview_f` | R0, R2, R4 |
| Fig 2 (storage, time, accuracy) | `F2_cost` | R1, R2, R3, R4 |
| Fig 3 (cross-platform, within-platform queries) | `F3_cross_platform` | R5, R6 |
| Fig 4 (cross-dataset, partial and gene-list queries) | `F4_skin` | R7, R8, R9 |
| Supp. Fig S1 (block-diagonal structure) | `S1_block` | R10 |
| Supp. Fig S2 (breast niches and blocks) | `S2_breast` | R6 |
| Supp. Fig S3 (scaling) | `S3_scaling` | R1 |
| Supp. Fig S4 (multi-core queries) | `S8_parallel` | R11 |
| Supp. Fig S5 (accuracy) | `S4_accuracy` | R2, R3 |
| Supp. Fig S6 (cross-platform) | `S5_cross_platform` | R5 |
| Supp. Fig S7 (consensus gene order) | `S6_blocks` | R10 |
| Supp. Fig S8 (dyadic intervals) | `S7_dyadic` | R10 |
| Supp. Tables | `T_supplement` | R1, R4, R7, R8, R9 |

Fig 1A–E is a schematic and is not generated by code. The topics in `results/` are `index_cost` (R1), `whole_tile` (R2),
`gene_set` (R3), `timing` (R4), `cross_platform` (R5), `breast_programs` (R6), `queries` (R7), `pan_cancer` (R8),
`skin_maps` (R9), `block_structure` (R10) and `parallel` (R11).

`examples/check_indexes.py <key>` (through sbatch on large sections) checks one dataset's indexes: exact self-queries,
compiled searches equal to their references bit for bit, and the lower bound and safety of both Spindle indexes.
