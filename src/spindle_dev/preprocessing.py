"""Tiling, gene selection, per-tile covariances and the train/held-out split."""

from __future__ import annotations

import math

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, delayed

from .utils import get_logger

LOGGER = get_logger(__name__)


class QuadTile:
    __slots__ = ("id", "bbox", "idx", "centroid", "side")

    def __init__(self, id, bbox, idx):
        self.id = id          # int
        self.bbox = bbox      # (xmin, ymin, xmax, ymax)
        self.idx = idx        # spot indices
        self.side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
        self.centroid = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])


def _split_bbox(b):
    xm = (b[0] + b[2]) / 2
    ym = (b[1] + b[3]) / 2
    return [(b[0], b[1], xm, ym), (xm, b[1], b[2], ym), (b[0], ym, xm, b[3]), (xm, ym, b[2], b[3])]


def reindex_tiles(tiles, sort=True):
    """Renumber tile ids 0..K-1 (optionally after sorting by bbox)."""
    if sort:
        tiles = sorted(tiles, key=lambda t: (t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]))
    for new_id, t in enumerate(tiles):
        t.id = new_id
    return tiles


def _low_density_tile_mask(tiles, mad_multiplier=6.0):
    """Tiles whose log cell density is more than ``mad_multiplier`` robust s.d. below the median."""
    n = len(tiles)
    if n == 0:
        return np.zeros(0, dtype=bool)
    areas = np.array([max((t.bbox[2] - t.bbox[0]) * (t.bbox[3] - t.bbox[1]), 1e-9) for t in tiles])
    counts = np.array([t.idx.size for t in tiles], dtype=float)
    log_density = np.log(np.maximum(counts / areas, 1e-12))
    median = np.median(log_density)
    mad = np.median(np.abs(log_density - median))
    if mad == 0:
        return np.zeros(n, dtype=bool)
    return log_density < median - mad_multiplier * 1.4826 * mad


def build_quadtree_tiles(coords, max_pts=200, min_side=0.0, max_depth=12,
                         filter_low_density=True, density_mad_multiplier=6.0):
    """Quadtree tiles of at most ``max_pts`` spots each; low-density tiles are dropped by default."""
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    xmin, ymin = coords.min(0)
    xmax, ymax = coords.max(0)
    tiles, stack, next_id = [], [((xmin, ymin, xmax, ymax), np.arange(n), 0)], 0

    while stack:
        bbox, idx, depth = stack.pop()
        idx = np.asarray(idx, dtype=int)
        side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        if (side <= min_side) or (depth >= max_depth):
            for chunk in np.array_split(idx, max(1, math.ceil(idx.size / max_pts))):
                tiles.append(QuadTile(next_id, bbox, chunk))
                next_id += 1
            continue

        if min_side <= 0 and idx.size <= max_pts:
            tiles.append(QuadTile(next_id, bbox, idx))
            next_id += 1
            continue

        child_nodes = []
        for child in _split_bbox(bbox):
            x0, y0, x1, y1 = child
            mask = ((coords[idx, 0] >= x0) & (coords[idx, 0] < x1) &
                    (coords[idx, 1] >= y0) & (coords[idx, 1] < y1))
            if mask.any():
                child_nodes.append((child, idx[mask]))

        # All points in one child: split by count instead.
        if len(child_nodes) <= 1:
            for chunk in np.array_split(idx, math.ceil(idx.size / max_pts)):
                tiles.append(QuadTile(next_id, bbox, chunk))
                next_id += 1
            continue

        for child_bbox, child_idx in child_nodes:
            stack.append((child_bbox, child_idx, depth + 1))

    if filter_low_density and tiles:
        drop = _low_density_tile_mask(tiles, mad_multiplier=density_mad_multiplier)
        if drop.any():
            LOGGER.info("build_quadtree_tiles: dropping %d low-density tile(s) (%d spots)",
                        int(drop.sum()), sum(t.idx.size for t, d in zip(tiles, drop) if d))
            tiles = reindex_tiles([t for t, d in zip(tiles, drop) if not d], sort=False)
    return tiles


def topvar_genes(adata, G=800):
    """Names and column indices of the ``G`` highest-variance genes."""
    X = adata.X
    if sp.issparse(X):
        m = np.asarray(X.mean(0)).ravel()
        m2 = np.asarray(X.multiply(X).mean(0)).ravel()
    else:
        m = X.mean(0)
        m2 = (X ** 2).mean(0)
    keep = np.argsort(m2 - m ** 2)[::-1][:G]
    return np.array(adata.var_names)[keep], keep


def _cov_ml(Xslice, eps=1e-6, cast32=True):
    """Maximum-likelihood covariance of the rows of ``Xslice`` plus ``eps`` on the diagonal."""
    n = Xslice.shape[0]
    if n == 0:
        return None
    if sp.issparse(Xslice):
        mu = np.asarray(Xslice.mean(0)).ravel()
        XtX = (Xslice.T @ Xslice).toarray()
    else:
        X = np.asarray(Xslice)
        mu = X.mean(0)
        XtX = X.T @ X
    Sig = XtX / max(n, 1) - np.outer(mu, mu)
    Sig.flat[::Sig.shape[0] + 1] += eps
    return Sig.astype(np.float32) if cast32 else Sig


def _tile_cov(X, t, eps):
    return {"tile_id": t.id, "cov": _cov_ml(X[t.idx, :], eps=eps, cast32=True), "n": len(t.idx)}


def build_tile_covs_full(adata, tiles, gene_idx=None, n_jobs=8, eps=1e-6, prefer="threads"):
    """One {'tile_id', 'cov', 'n'} per tile, computed in parallel."""
    X = adata.X if gene_idx is None else adata.X[:, gene_idx]
    return Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(_tile_cov)(X, t, eps) for t in tiles)


def build_tile_covs_full_serial(adata, tiles, gene_idx=None, eps=1e-6):
    """Serial version of ``build_tile_covs_full``."""
    X = adata.X if gene_idx is None else adata.X[:, gene_idx]
    return [_tile_cov(X, t, eps) for t in tiles]


def load_adata(path):
    """Read an .h5ad; cells with ``Cluster == "Unlabeled"`` (breast annotation) are dropped."""
    import scanpy as sc

    adata = sc.read_h5ad(path)
    if "Cluster" in adata.obs.columns:
        adata = adata[adata.obs.loc[adata.obs.Cluster != "Unlabeled"].index, :].copy()
    return adata


def prepare_tiles(adata, max_genes=800, n_jobs=8):
    """Tiles (<= 200 cells), the top min(max_genes, n_genes) variable genes and every tile's covariance."""
    tiles = build_quadtree_tiles(adata.obsm["spatial"], max_pts=200, min_side=0.0, max_depth=40)
    genes, gene_idx = topvar_genes(adata, G=min(max_genes, adata.n_vars))
    tile_covs = build_tile_covs_full(adata, tiles, gene_idx, n_jobs=n_jobs, eps=1e-6)
    return tiles, tile_covs, genes


def split_tiles(n_tiles, seed, n_holdout=None, test_ratio=0.05):
    """(train_idx, test_idx): ``n_holdout`` tiles (or a ``test_ratio`` share) held out at random."""
    np.random.seed(seed)
    if n_holdout is not None:
        num_test = min(n_holdout, max(0, n_tiles - 1))
    else:
        num_test = int(n_tiles * test_ratio)
    all_idx = np.arange(n_tiles)
    test_idx = np.random.choice(all_idx, size=num_test, replace=False)
    return np.setdiff1d(all_idx, test_idx), test_idx
