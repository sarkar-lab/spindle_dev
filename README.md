# Spindle

Search over the covariance matrices of spatial-omics tiles. A section is cut into tiles of ≤ 200 cells;
each tile's gene–gene covariance is indexed under a shared layout (covariance niches, and a gene order with
contiguous gene blocks per niche). Three indexes sit on that layout:

- **Spindle**: an exact scan over stored block logs, for whole-tile and gene-set queries.
- **Spindle-lite**: a layered block DAG per niche with a budgeted depth-first search.
- **Interval index**: dyadic gene intervals with ε-bounded node means, for gene-set queries.

## Layout

```
spindle_dev/
├── src/spindle_dev/     # the package
│   ├── preprocessing.py #   tiling, gene selection, tile covariances, train/held-out split
│   ├── layout.py        #   niches, gene order and blocks (shared by all indexes)
│   ├── spindle.py       #   Spindle (exact block-log scan; whole tile and gene set)
│   ├── lite.py          #   Spindle-lite (layered DAG, budgeted DFS), save/load
│   ├── interval.py      #   interval index
│   ├── metrics.py       #   shrinkage, matrix logs, correlation
│   └── utils.py         #   logging
├── examples/            # run_single_dataset.py (usage), check_indexes.py (sanity checks)
├── paths.py             # dataset paths, read from datasets.yaml
└── datasets.yaml        # the only place a dataset path is written
```

## Install

```bash
pip install -r requirements.txt
pip install -e .
```

## Use

```python
import spindle_dev as sd

adata = sd.load_adata("section.h5ad")
tiles, covs, genes = sd.prepare_tiles(adata, max_genes=800)
train, test = sd.split_tiles(len(tiles), seed=73, n_holdout=100)
train_covs = [covs[i] for i in train]
data = sd.build_layout([tiles[i] for i in train], train_covs, genes, adata.n_obs)

spindle = sd.Spindle(data, train_covs)
rows, scores, seconds = spindle.search(covs[test[0]], c=10)        # exact d_B top 10

lite = sd.build_lite(data)
rows, seconds = sd.search_lite(lite, covs[test[0]])                 # Spindle-lite set

partial = sd.SpindlePartial(data, train_covs)
interval = sd.build_interval_index(data, block_covs=partial.covs)
rows, scores, seconds = sd.search_interval_index(interval, covs[test[0]], gene_idx, c=10)
```

Every tile covariance is shrunk once when it is computed (`metrics.SHRINK_ALPHA` = 0.1), and every
distance is d_B (d_B^S for gene sets). Queries built elsewhere must be shrunk first (`metrics.shrink_cov`).
Rows are positions of the training tiles (`data.labels` order). `python examples/run_single_dataset.py
skin_melanoma` runs all three on one dataset.
