# Spindle

Search over the covariance matrices of spatial-omics tiles. A section is cut into tiles of ≤ 200 cells;
each tile's gene–gene covariance is indexed under a shared layout (covariance niches, and a gene order with
contiguous gene blocks per niche). Two indexes, each for whole tiles and gene sets, sit on that layout:

- **Spindle**: an exact scan over stored block logs, for whole-tile and gene-set queries.
- **Spindle-lite**: a layered block DAG per niche, searched with a radius set by the query's own best match
  (`paper/SPINDLE_LITE.md`); for gene sets, the same design on dyadic gene intervals (Spindle-lite partial,
  `paper/SPINDLE_PARTIAL.md`).

## Layout

```
spindle_dev/
├── src/spindle_dev/     # the package
│   ├── preprocessing.py #   tiling, gene selection, tile covariances, train/held-out split
│   ├── layout.py        #   niches, gene order and blocks (shared by all indexes)
│   ├── spindle.py       #   Spindle (exact block-log scan; whole tile and gene set)
│   ├── lite.py          #   Spindle-lite (SpindleLite; paper/SPINDLE_LITE.md)
│   ├── lite_partial.py  #   Spindle-lite partial (SpindleLitePartial; paper/SPINDLE_PARTIAL.md)
│   ├── queries.py       #   gene-set queries from gene names (template, tile)
│   ├── kernels.c/.py    #   compiled searches behind Spindle and both Spindle-lites (C, AVX2, ctypes)
│   ├── metrics.py       #   shrinkage, matrix logs, correlation
│   └── utils.py         #   logging, save/load
├── examples/            # run_single_dataset.py (usage), check_indexes.py (sanity checks)
├── experiments/         # the paper's figures: common/ pipeline, one folder per figure (run.sh lists the steps)
├── results/             # figure inputs and numbers.json per figure (the index cache _cache/ is gitignored)
├── figures/             # atoms, edited pptx and rendered PDF/PNG per figure, all_figures.{pptx,pdf}
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

lite = sd.SpindleLite.from_spindle(spindle, data)
rows, scores, seconds = lite.search(covs[test[0]])                  # Spindle-lite set (g = 0.25), best first

partial = sd.SpindlePartial(data, train_covs)
rows, scores, seconds = partial.search(covs[test[0]], gene_idx, c=10)  # exact d_B^S top 10

plite = sd.SpindleLitePartial.from_partial(partial, data)
rows, scores, seconds = plite.search(covs[test[0]], gene_idx)        # Spindle-lite partial set (g = 1), best first

# a gene list by names: the co-expression template (Methods Eq. signature)
omega = sd.variance_quantiles(train_covs)                           # (p,), keep it beside the index
idx, found, missing = sd.gene_indices(genes, ["CD68", "CD163", "CD14", "AIF1", "MNDA"])
cov_s, idx = sd.template_query(omega, idx)                          # rho = 0.5, 90th-percentile variances
rows, scores, seconds = partial.search(cov_s, idx, c=10)
rows, scores, seconds = plite.search(cov_s, idx)
```

Every tile covariance is shrunk once when it is computed (`metrics.SHRINK_ALPHA` = 0.1), and every
distance is d_B (d_B^S for gene sets). Queries built elsewhere must be shrunk first (`metrics.shrink_cov`).
Rows are positions of the training tiles (`data.labels` order). Spindle's whole-tile scan and both Spindle-lite searches run compiled C kernels
(built with gcc on first use into `src/spindle_dev/_build/`), bit-identical to their numpy references. `python examples/run_single_dataset.py
skin_melanoma` runs all of them on one dataset.
