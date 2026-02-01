"""Preprocessing interfaces for SPD indexing.

This module intentionally only defines interfaces and placeholders.
Actual preprocessing (tiling, SPD estimation, etc.) is expected to live
in a separate project and be plugged in via these interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence
import numpy as np
import math
import scipy.sparse as sp
from joblib import Parallel, delayed

from .utils import get_logger

LOGGER = get_logger(__name__)


class QuadTile:
    __slots__ = ("id","bbox","idx","centroid","side")
    def __init__(self, id, bbox, idx):
        self.id = id                      # int
        self.bbox = bbox                  # (xmin, ymin, xmax, ymax)
        self.idx = idx                    # np.array of spot indices
        self.side = max(bbox[2]-bbox[0], bbox[3]-bbox[1])
        self.centroid = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])


def _split_bbox(b):
    xm = (b[0]+b[2])/2; ym = (b[1]+b[3])/2
    return [(b[0], b[1], xm,   ym  ),
            (xm,   b[1], b[2], ym  ),
            (b[0], ym,   xm,   b[3]),
            (xm,   ym,   b[2], b[3])]


def build_quadtree_tiles_old(coords, max_pts=200, min_side=0.0, max_depth=12):
    """
    coords: (n,2) array, typically adata.obsm['spatial'] (x,y in same units)
    Returns: list[QuadTile] with spot membership per tile.
    """
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    xmin, ymin = coords.min(0); xmax, ymax = coords.max(0)
    root_bbox = (xmin, ymin, xmax, ymax)
    tiles, stack, next_id = [], [(root_bbox, np.arange(n), 0)], 0

    while stack:
        bbox, idx, depth = stack.pop()
        side = max(bbox[2]-bbox[0], bbox[3]-bbox[1])
        if (idx.size <= max_pts) or (side <= min_side) or (depth >= max_depth):
            tiles.append(QuadTile(next_id, bbox, idx))
            next_id += 1
            continue
        # split
        for child in _split_bbox(bbox):
            x0,y0,x1,y1 = child
            mask = (coords[idx,0] >= x0) & (coords[idx,0] < x1) & \
                   (coords[idx,1] >= y0) & (coords[idx,1] < y1)
            child_idx = idx[mask]
            if child_idx.size == 0: 
                continue
            stack.append((child, child_idx, depth+1))
    return tiles


def reindex_tiles(tiles, sort=True):
    """
    Reassign tile_id to 0..K-1 after filtering.
    If sort=True, sort deterministically by (depth proxy, bbox coords) so IDs are stable.
    """
    if sort:
        # stable ordering by bbox; add more keys if you want
        tiles = sorted(tiles, key=lambda t: (t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]))

    for new_id, t in enumerate(tiles):
        t.id = new_id   # or t.id depending on your QuadTile class

    return tiles


def build_quadtree_tiles(coords, max_pts=200, min_side=0.0, max_depth=12):
    """
    coords: (n,2) array, typically adata.obsm['spatial'] (x,y in same units)
    Returns: list[QuadTile] with spot membership per tile.
    Guarantees: each tile has at most `max_pts` points.
    """
    coords = np.asarray(coords, float)
    n = coords.shape[0]
    xmin, ymin = coords.min(0); xmax, ymax = coords.max(0)
    root_bbox = (xmin, ymin, xmax, ymax)
    tiles, stack, next_id = [], [(root_bbox, np.arange(n), 0)], 0

    while stack:
        bbox, idx, depth = stack.pop()
        idx = np.asarray(idx, dtype=int)
        side = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

        # If this node is small enough, make it a leaf.
        if idx.size <= max_pts:
            tiles.append(QuadTile(next_id, bbox, idx))
            next_id += 1
            continue

        # From here on, idx.size > max_pts so we MUST split somehow.

        # If we cannot meaningfully split spatially (too deep / too small),
        # fall back to non-spatial chunking to enforce the cap.
        if (side <= min_side) or (depth >= max_depth):
            n_chunks = math.ceil(idx.size / max_pts)
            for chunk in np.array_split(idx, n_chunks):
                tiles.append(QuadTile(next_id, bbox, chunk))
                next_id += 1
            continue

        # Try spatial split
        child_nodes = []
        for child in _split_bbox(bbox):
            x0, y0, x1, y1 = child
            mask = (
                (coords[idx, 0] >= x0) & (coords[idx, 0] < x1) &
                (coords[idx, 1] >= y0) & (coords[idx, 1] < y1)
            )
            child_idx = idx[mask]
            if child_idx.size > 0:
                child_nodes.append((child, child_idx))

        # Degenerate case: all points end up in one child, or no effective split.
        if len(child_nodes) <= 1:
            n_chunks = math.ceil(idx.size / max_pts)
            for chunk in np.array_split(idx, n_chunks):
                tiles.append(QuadTile(next_id, bbox, chunk))
                next_id += 1
            continue

        # Normal case: push children for further processing
        for child_bbox, child_idx in child_nodes:
            stack.append((child_bbox, child_idx, depth + 1))

    return tiles


def topvar_genes(adata, G=800):
    X = adata.X
    if sp.issparse(X):
        m = np.asarray(X.mean(0)).ravel()
        m2 = np.asarray(X.multiply(X).mean(0)).ravel()
    else:
        m = X.mean(0); m2 = (X**2).mean(0)
    var = m2 - m**2
    keep = np.argsort(var)[::-1][:G]
    return np.array(adata.var_names)[keep], keep


def _cov_ml(Xslice, eps=1e-6, cast32=True):
    n = Xslice.shape[0]
    if n == 0: return None
    if sp.issparse(Xslice):
        mu = np.asarray(Xslice.mean(0)).ravel()
        XtX = (Xslice.T @ Xslice).toarray()
    else:
        X = np.asarray(Xslice)
        mu = X.mean(0)
        XtX = X.T @ X
    Sig = XtX / max(n,1) - np.outer(mu, mu)
    g = Sig.shape[0]; Sig.flat[::g+1] += eps
    return Sig.astype(np.float32) if cast32 else Sig


def build_tile_covs_full(adata, tiles, gene_idx=None, n_jobs=8, eps=1e-6, prefer="threads"):
    if gene_idx is None:
        X = adata.X
    else:
        X = adata.X[:, gene_idx]  # slice once
    def one(t):
        idx = t.idx if hasattr(t,"idx") else t["spot_indices"]
        Sig = _cov_ml(X[idx, :], eps=eps, cast32=True)
        return {"tile_id": (t.id if hasattr(t,"id") else t.get("tile_id")),
                "cov": Sig, "n": len(idx)}
    return Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(one)(t) for t in tiles)


def build_tile_covs_full_serial(adata, tiles, gene_idx=None, eps=1e-6):
    """
    Serial (non-parallel) version of build_tile_covs_full.
    Computes covariance per tile sequentially and returns the same structure.
    """
    if gene_idx is None:
        X = adata.X
    else:
        X = adata.X[:, gene_idx]  # slice once

    out = []
    for t in tiles:
        idx = t.idx if hasattr(t, "idx") else t["spot_indices"]
        Sig = _cov_ml(X[idx, :], eps=eps, cast32=True)
        out.append({
            "tile_id": (t.id if hasattr(t, "id") else t.get("tile_id")),
            "cov": Sig,
            "n": len(idx),
        })
    return out

class SpatialDataset(Protocol):
    """Protocol describing the minimal spatial dataset needed for indexing.

    Implementations are expected to provide:
    - a set of spatial point coordinates (e.g., shape (N, D)), and
    - a collection of SPD sub-matrices derived from these points.
    """

    @property
    def points(self) -> np.ndarray:
        """Array of spatial coordinates, shape (N, dim)."""
        ...

    @property
    def spd_matrices(self) -> Sequence[np.ndarray]:
        """Sequence of SPD matrices associated with the spatial data."""
        ...


@dataclass
class PreprocessingConfig:
    """Configuration stub for preprocessing.

    This is kept deliberately generic; concrete projects should subclass
    or extend this in their own codebase if they need additional fields.
    """

    description: str | None = None


def build_spatial_dataset(raw_data: Any, config: PreprocessingConfig | None = None) -> SpatialDataset:
    """Hook for user-provided preprocessing.

    This function is intentionally *not* implemented in this repo.
    Downstream projects should provide an implementation that converts
    arbitrary raw spatial data into a ``SpatialDataset`` instance that
    can be passed to ``index.build_index``.
    """

    raise NotImplementedError(
        "Preprocessing is out-of-scope for this package; "
        "provide your own `build_spatial_dataset` implementation in the "
        "host project."
    )

