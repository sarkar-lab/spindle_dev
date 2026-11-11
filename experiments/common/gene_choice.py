"""Lymph node 5k with another gene choice: the production build (tiles, covariances, layout) on the top-``n`` genes by
mean expression instead of by variance (supplementary Fig S12, R14).

Same calls as ``prepare_tiles`` + ``index_cache.build``'s layout stage, with only the gene list changed:
``build_quadtree_tiles`` (as prepare_tiles), ``build_tile_covs_full`` on the chosen genes, the seed-0 split and
``build_layout`` on the index tiles. Fig S1's readouts then come from ``block_structure`` (``tile_means``,
``niche_example``), as for the cached (top-variance) build.

    import gene_choice as gc
    b = gc.build("lymph_node_5k", 400, "mean")    # {genes, layout, train_covs, ...}
"""

import numpy as np
import scipy.sparse as sp

import data
import spindle_dev as sd
from spindle_dev.preprocessing import build_quadtree_tiles, build_tile_covs_full, topvar_genes


def top_expressed(adata, G):
    """Names and column indices of the ``G`` genes with the highest mean count (as topvar_genes, by mean)."""
    X = adata.X
    m = np.asarray(X.mean(0)).ravel() if sp.issparse(X) else X.mean(0)
    keep = np.argsort(m)[::-1][:G]
    return np.array(adata.var_names)[keep], keep


def build(key, n_genes, by, seed=data.EXAMPLE_SEED, n_jobs=8):
    """The production layout of ``key`` on the top-``n_genes`` genes ``by`` 'var' (as the index) or 'mean'."""
    adata = sd.load_adata(data.paths.dataset_path(key))
    tiles = build_quadtree_tiles(adata.obsm["spatial"], max_pts=200, min_side=0.0, max_depth=40)
    pick = topvar_genes if by == "var" else top_expressed
    genes, gidx = pick(adata, min(n_genes, adata.n_vars))
    covs = build_tile_covs_full(adata, tiles, gidx, n_jobs=n_jobs, eps=1e-6)
    train, test = data.split(len(tiles), seed)
    tr_covs = [covs[i] for i in train]
    layout = sd.build_layout([tiles[i] for i in train], tr_covs, genes, adata.n_obs)
    X = adata.X[:, gidx]
    mean = np.asarray(X.mean(0)).ravel()
    var = np.asarray(X.multiply(X).mean(0)).ravel() - mean ** 2 if sp.issparse(X) else X.var(0)
    return {"genes": np.asarray(genes).astype(str), "mean": mean, "var": var, "layout": layout,
            "train": train, "train_covs": tr_covs}
