"""Interval index for partial (gene-subset) queries.

Index
-----
For every niche C_j and block beta of the niche's gene order sigma_j, the
positions 0..n_beta-1 of the block are covered by the *dyadic* intervals
I(s, a) = {a, ..., a+s-1} with s = 2^k <= max_len, s | a and a + s <= n_beta.
For each interval, the tiles' interval matrices X_m^I = log Sigma_m[g(I)]
(logarithm of the covariance restricted to the interval's genes) are
partitioned so that every member lies within eps_{beta,s} of its node *mean*
(farthest-first cover at eps/2, optional Lloyd iterations, violators re-covered
at eps/2).  eps_{beta,s} is set so that a length-s interval has about
K_s = k_target * s^k_alpha nodes.  The index stores, per interval, the node
means and one node code per tile.

Distances are unnormalized Frobenius distances between matrix logarithms.
Symmetric matrices are stored as upper-triangle vectors with off-diagonals
scaled by sqrt 2, so their Euclidean norm is the Frobenius norm.

Search
------
A query gene set S is cut, in every niche, into segments (maximal runs of
consecutive positions within one block), and each segment into the fewest
dyadic pieces (from each position, the longest indexed interval that fits).
A tile's score is

    sqrt( sum over pieces I of ||Q^I - mu_{node of the tile on I}||_F^2 ),

so covariances between genes of a piece are used and those between pieces are
not.  Because every member lies within eps of its node mean, the score differs
from the same sum with the tile's own interval logs by at most
sqrt(sum_I eps_{beta,|I|}^2) (``error_bound``).  Tiles of all niches are
ranked by the score.

Defaults are the paper's configuration: k_target = 32, k_alpha = 0.5,
max_len = 16, no Lloyd iterations.

Public API
----------
build_interval_index(data, train_covs=None, ...) -> IntervalIndex
search_interval_index(index, genes, cov_s, score="mean") -> IntervalSearchResult
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

LOG_FLOOR = 1e-6  # eigenvalue floor of the ground-truth block distances


# ---------------------------------------------------------------------------
# Matrix helpers
# ---------------------------------------------------------------------------

def batched_log_spd(mats: np.ndarray, floor: float = LOG_FLOOR) -> np.ndarray:
    """Matrix logarithm of a stack (..., s, s) of SPD matrices, eigenvalues floored."""
    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    w, V = np.linalg.eigh(mats)
    w = np.log(np.maximum(w, floor))
    return (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)


def sym_to_vec(L: np.ndarray) -> np.ndarray:
    """(..., s, s) symmetric -> (..., s(s+1)/2) with ||vec|| = ||L||_F."""
    iu = np.triu_indices(L.shape[-1])
    return L[..., iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def _sqdist(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances between the rows of X (n, d) and C (k, d)."""
    d = (X * X).sum(1)[:, None] - 2.0 * X @ C.T + (C * C).sum(1)[None, :]
    return np.maximum(d, 0.0)


def _assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return np.argmin(_sqdist(X, C), axis=1)


# ---------------------------------------------------------------------------
# Clustering (Algorithm IntervalCluster)
# ---------------------------------------------------------------------------

class FarthestFirst:
    """Resumable farthest-first traversal from row 0 (Gonzalez).

    ``radii[k]`` is the covering radius with the first k+1 centres, so the cover
    at radius rho is the shortest prefix whose radius is <= rho.
    """

    def __init__(self, X: np.ndarray):
        self.X = X
        self.centres = [0]
        self.mind = np.sqrt(((X - X[0]) ** 2).sum(1))
        self.radii = [float(self.mind.max())]

    def extend_to(self, k: Optional[int] = None, radius: Optional[float] = None) -> None:
        while len(self.centres) < len(self.X):
            if k is not None and len(self.centres) >= k:
                break
            if radius is not None and self.radii[-1] <= radius:
                break
            i = int(np.argmax(self.mind))
            self.centres.append(i)
            self.mind = np.minimum(self.mind, np.sqrt(((self.X - self.X[i]) ** 2).sum(1)))
            self.radii.append(float(self.mind.max()))

    def cover(self, radius: float) -> np.ndarray:
        """Centres (row indices) of the cover at ``radius``."""
        self.extend_to(radius=radius)
        k = next((i + 1 for i, r in enumerate(self.radii) if r <= radius), len(self.radii))
        return np.asarray(self.centres[:k])


def interval_cluster(X: np.ndarray, eps: float, lloyd_iters: int,
                     ff: Optional[FarthestFirst] = None) -> np.ndarray:
    """eps-bounded mean partition of the rows of X; returns one code per row.

    (i) farthest-first cover at eps/2 and nearest-centre assignment;
    (ii) at most ``lloyd_iters`` Lloyd iterations (empty clusters discarded);
    (iii) every cluster with a member farther than eps from its mean is
    replaced by the nearest-centre clusters of its own farthest-first cover at
    eps/2, whose members lie within eps of their means (Lemma mean_ball).
    """
    ff = ff if ff is not None else FarthestFirst(X)
    codes = _assign(X, X[ff.cover(eps / 2.0)])
    for _ in range(lloyd_iters):
        _, codes = np.unique(codes, return_inverse=True)
        means = np.zeros((codes.max() + 1, X.shape[1]))
        np.add.at(means, codes, X)
        means /= np.bincount(codes)[:, None]
        new = _assign(X, means)
        if np.array_equal(new, codes):
            break
        codes = new
    _, codes = np.unique(codes, return_inverse=True)

    out = np.empty(len(X), dtype=np.int64)
    nxt = 0
    for c in range(codes.max() + 1):
        mem = np.flatnonzero(codes == c)
        sub = X[mem]
        if np.sqrt(((sub - sub.mean(0)) ** 2).sum(1)).max() <= eps:
            out[mem] = nxt
            nxt += 1
            continue
        _, sub_codes = np.unique(_assign(sub, sub[FarthestFirst(sub).cover(eps / 2.0)]), return_inverse=True)
        out[mem] = nxt + sub_codes
        nxt += sub_codes.max() + 1
    return out


# ---------------------------------------------------------------------------
# Index data structures
# ---------------------------------------------------------------------------

@dataclass
class IntervalNodes:
    """Nodes of one dyadic interval I(s, a) of one block."""
    s: int
    a: int
    means: np.ndarray            # (K, s(s+1)/2) float32 node means
    codes: np.ndarray            # (n_j,) node of every niche tile

    @property
    def n_nodes(self) -> int:
        return len(self.means)


@dataclass
class BlockIntervals:
    """The dyadic intervals of one block and their nodes."""
    genes: np.ndarray                                  # global gene index of every block position
    eps: Dict[int, float]                              # radius per interval length
    intervals: Dict[Tuple[int, int], IntervalNodes]    # (s, a) -> nodes
    tile_logs: Optional[Dict[Tuple[int, int], np.ndarray]] = None  # (s,a) -> (n_j, d) exact X_m^I

    def starting_at(self, a: int, end_max: int) -> List[Tuple[int, int]]:
        """Interval keys starting at ``a`` that end no later than ``end_max``."""
        keys, s = [], 1
        while a % s == 0 and a + s <= end_max:
            if (s, a) in self.intervals:
                keys.append((s, a))
            s *= 2
        return keys


@dataclass
class NicheIntervalIndex:
    niche: int
    rows: np.ndarray              # row of every niche tile in data.spd_ids order
    tile_ids: np.ndarray          # spd_id of every niche tile
    blocks: List[BlockIntervals]
    gene_block: np.ndarray        # (p,) block index of every global gene in this niche
    gene_pos: np.ndarray          # (p,) position of every global gene inside its block

    @property
    def n_tiles(self) -> int:
        return len(self.rows)


@dataclass
class IntervalIndex:
    niches: Dict[int, NicheIntervalIndex]
    n_tiles: int
    params: dict
    build_seconds: float = 0.0

    def n_nodes(self) -> int:
        return sum(iv.n_nodes for ni in self.niches.values() for b in ni.blocks for iv in b.intervals.values())

    def nbytes(self) -> Dict[str, int]:
        """Deployed footprint: float32 node means and one code per (tile, interval)."""
        means = codes = 0
        for ni in self.niches.values():
            for b in ni.blocks:
                for iv in b.intervals.values():
                    means += iv.means.size * 4
                    codes += iv.codes.size * (1 if iv.n_nodes <= 256 else 2 if iv.n_nodes <= 65536 else 4)
        return {"means": means, "codes": codes, "total": means + codes}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def dyadic_intervals(n: int, max_len: Optional[int] = None) -> List[Tuple[int, int]]:
    """(s, a) of every dyadic interval of a block with n positions, by increasing s."""
    out, s = [], 1
    while s <= n and (max_len is None or s <= max_len):
        out.extend((s, a) for a in range(0, n - s + 1, s))
        s *= 2
    return out


def _build_block(block_covs: np.ndarray, genes: np.ndarray, *, eps, k_target, k_alpha, eps_scale, lloyd_iters,
                 max_len, floor, keep_tile_logs) -> BlockIntervals:
    """block_covs: (n_j, b, b) covariances of the niche's tiles on the block's genes, in block order."""
    by_len: Dict[int, List[int]] = {}
    for s, a in dyadic_intervals(len(genes), max_len):
        by_len.setdefault(s, []).append(a)

    intervals: Dict[Tuple[int, int], IntervalNodes] = {}
    eps_used: Dict[int, float] = {}
    tile_logs = {} if keep_tile_logs else None
    for s, starts in by_len.items():
        X = {}
        for a in starts:
            X[a] = sym_to_vec(batched_log_spd(block_covs[:, a:a + s, a:a + s], floor))
        ffs = {a: FarthestFirst(X[a]) for a in starts}
        if eps is not None:
            e = float(eps(s) if callable(eps) else eps[s] if isinstance(eps, dict) else eps)
        else:
            # radius rule: eps/2 = eps_scale * median over this length's intervals of the
            # farthest-first covering radius with k_target centres
            k_s = max(1, int(round(k_target * s ** k_alpha)))
            radii = []
            for a in starts:
                ffs[a].extend_to(k=k_s)
                radii.append(ffs[a].radii[min(k_s, len(ffs[a].radii)) - 1])
            e = 2.0 * eps_scale * float(np.median(radii))
        e = max(e, 1e-12)
        eps_used[s] = e
        for a in starts:
            codes = interval_cluster(X[a], e, lloyd_iters, ff=ffs[a])
            K = int(codes.max()) + 1
            means = np.zeros((K, X[a].shape[1]))
            np.add.at(means, codes, X[a])
            means /= np.bincount(codes, minlength=K)[:, None]
            intervals[(s, a)] = IntervalNodes(s, a, means.astype(np.float32), codes.astype(np.int32))
            if keep_tile_logs:
                tile_logs[(s, a)] = X[a].astype(np.float32)
    return BlockIntervals(np.asarray(genes), eps_used, intervals, tile_logs)


def build_interval_index(data, train_covs=None, *, eps=None, k_target: int = 32, eps_scale: float = 1.0,
                         k_alpha: float = 0.5, lloyd_iters: int = 0, max_len: Optional[int] = 16,
                         floor: float = LOG_FLOOR, keep_tile_logs: bool = False,
                         niches: Optional[Sequence[int]] = None, block_covs=None) -> IntervalIndex:
    """Build the interval index of every niche of ``data``.

    Parameters
    ----------
    data : ProcessedData (``labels``, ``spd_ids``, ``perm_list``, ``block_dict``;
        ``spd_matrices`` unless ``train_covs`` is given).
    train_covs : optional training covariances (arrays or dicts with 'cov') in ``data.spd_ids`` order.
    eps : radius eps_{beta,s}: a float, a dict {s: eps} or a callable s -> eps, shared by all
        blocks.  When None, eps_{beta,s} = 2 * eps_scale * the median over the block's length-s
        intervals of the farthest-first covering radius with ``k_target`` centres, so an
        interval gets about ``k_target`` nodes at eps_scale = 1.
    lloyd_iters : T, the number of Lloyd iterations.
    max_len : longest interval indexed (None: every dyadic length).
    keep_tile_logs : also keep every tile's exact interval vectors (evaluation only).
    block_covs : optional {niche: [(n_j, b, b) covariance stack per block, in block order]} used instead of
        full covariances (rows in the niche's ``data.labels`` order); avoids holding p x p matrices.
    """
    t0 = time.perf_counter()
    covs_all = train_covs if train_covs is not None else getattr(data, "spd_matrices", None)
    labels = np.asarray(data.labels).astype(int)
    spd_ids = np.asarray(data.spd_ids)
    out: Dict[int, NicheIntervalIndex] = {}
    for k in sorted(set(labels.tolist())):
        if niches is not None and k not in niches:
            continue
        rows = np.flatnonzero(labels == k)
        if block_covs is None:
            covs = np.stack([np.asarray(covs_all[r].get("cov", covs_all[r]) if isinstance(covs_all[r], dict)
                                        else covs_all[r], dtype=np.float64) for r in rows])
            p = covs.shape[1]
        else:
            p = len(data.metadata["genes"]) if hasattr(data, "metadata") else len(data.perm_list[k])
        perm = np.asarray(data.perm_list[k])
        gene_block = np.full(p, -1, dtype=np.int32)
        gene_pos = np.full(p, -1, dtype=np.int32)
        blocks = []
        for b, (start, end) in enumerate(data.block_dict[k]):
            genes = perm[start:end]
            gene_block[genes] = b
            gene_pos[genes] = np.arange(end - start)
            bc = covs[:, genes[:, None], genes[None, :]] if block_covs is None else block_covs[k][b]
            blocks.append(_build_block(bc, genes, eps=eps, k_target=k_target, k_alpha=k_alpha, eps_scale=eps_scale,
                                       lloyd_iters=lloyd_iters, max_len=max_len, floor=floor,
                                       keep_tile_logs=keep_tile_logs))
        out[k] = NicheIntervalIndex(k, rows, spd_ids[rows], blocks, gene_block, gene_pos)
        logger.info("Interval index niche %d: %d tiles, %d blocks", k, len(rows), len(blocks))
    params = dict(eps=eps, k_target=k_target, k_alpha=k_alpha, eps_scale=eps_scale, lloyd_iters=lloyd_iters,
                  max_len=max_len, floor=floor)
    return IntervalIndex(out, len(labels), params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    block: int
    a: int
    b: int


def query_segments(ni: NicheIntervalIndex, genes: Sequence[int]) -> List[Segment]:
    """Segments of the gene set in this niche, ordered by block then start."""
    g = np.asarray(genes)
    blk, pos = ni.gene_block[g], ni.gene_pos[g]
    if np.any(blk < 0):
        raise ValueError("query gene outside the niche's blocks")
    segs = []
    for b in np.unique(blk):
        ps = np.sort(pos[blk == b])
        cuts = np.flatnonzero(np.diff(ps) > 1)
        starts = np.r_[ps[0], ps[cuts + 1]]
        ends = np.r_[ps[cuts], ps[-1]] + 1
        segs.extend(Segment(int(b), int(s), int(e)) for s, e in zip(starts, ends))
    return segs


def _query_pieces(blk: BlockIntervals, keys, cov_s, col_of_gene, floor) -> Dict[Tuple[int, int], np.ndarray]:
    """Q^I as Frobenius vectors for every interval key, batched by length."""
    out, by_len = {}, {}
    for s, a in keys:
        by_len.setdefault(s, []).append(a)
    for s, starts in by_len.items():
        cols = np.stack([col_of_gene[blk.genes[a:a + s]] for a in starts])
        vecs = sym_to_vec(batched_log_spd(cov_s[cols[:, :, None], cols[:, None, :]], floor))
        out.update({(s, a): v for a, v in zip(starts, vecs)})
    return out


@dataclass
class IntervalSearchResult:
    rows: np.ndarray          # tiles (rows of data.spd_ids), by increasing score
    tile_ids: np.ndarray
    scores: np.ndarray
    stats: dict


def coarsest_decomposition(blk: BlockIntervals, seg: Segment) -> List[Tuple[int, int]]:
    """The decomposition with the fewest pieces: from each position, the longest indexed dyadic interval that fits."""
    keys, c = [], seg.a
    while c < seg.b:
        key = blk.starting_at(c, seg.b)[-1]
        keys.append(key)
        c += key[0]
    return keys


def search_interval_index(index: IntervalIndex, genes: Sequence[int], cov_s: np.ndarray, budget: float = np.inf,
                          score: str = "mean", niches: Optional[Sequence[int]] = None) -> IntervalSearchResult:
    """Score every tile of every niche for the gene set ``genes`` with covariance ``cov_s`` (rows/cols in
    ``genes`` order): sqrt of the sum, over the pieces of the fewest-piece decomposition of each segment,
    of the squared piece distance.

    score  : "mean"  -- ||Q^I - mu_v|| to the mean of the tile's node (the method);
             "exact" -- ||Q^I - X_m^I|| to the tile's own interval log (needs keep_tile_logs; the
                        index-free limit of the method, for evaluation).
    budget : tiles with a score above it are not returned.
    """
    if score not in ("mean", "exact"):
        raise ValueError(score)
    t0 = time.perf_counter()
    genes = np.asarray(genes)
    cov_s = np.asarray(cov_s, dtype=np.float64)
    p = next(iter(index.niches.values())).gene_block.shape[0]
    col_of_gene = np.full(p, -1, dtype=np.int64)
    col_of_gene[genes] = np.arange(len(genes))
    floor = index.params.get("floor", LOG_FLOOR)

    rows, scores, n_pieces = [], [], {}
    for k, ni in index.niches.items():
        if niches is not None and k not in niches:
            continue
        sq = np.zeros(ni.n_tiles)
        n_pieces[k] = 0
        for seg in query_segments(ni, genes):
            blk = ni.blocks[seg.block]
            keys = coarsest_decomposition(blk, seg)
            n_pieces[k] += len(keys)
            q = _query_pieces(blk, keys, cov_s, col_of_gene, floor)
            for key in keys:
                if score == "exact":
                    sq += ((blk.tile_logs[key] - q[key]) ** 2).sum(1)
                else:
                    iv = blk.intervals[key]
                    sq += ((iv.means.astype(np.float64) - q[key]) ** 2).sum(1)[iv.codes]
        total = np.sqrt(sq)
        idx = np.flatnonzero(total <= budget)
        rows.append(ni.rows[idx])
        scores.append(total[idx])
    rows = np.concatenate(rows)
    scores = np.concatenate(scores)
    order = np.argsort(scores, kind="stable")
    rows, scores = rows[order], scores[order]
    tile_ids = np.empty(len(rows), dtype=next(iter(index.niches.values())).tile_ids.dtype)
    for ni in index.niches.values():
        hit = np.isin(rows, ni.rows)
        tile_ids[hit] = ni.tile_ids[np.searchsorted(ni.rows, rows[hit])]
    return IntervalSearchResult(rows, tile_ids, scores,
                                {"seconds": time.perf_counter() - t0, "pieces": n_pieces})


def error_bound(index: IntervalIndex, genes: Sequence[int]) -> Dict[int, float]:
    """Per niche, sqrt(sum of eps_{beta,s}^2 over the pieces): |mean score - exact score| <= this for every tile."""
    out = {}
    for k, ni in index.niches.items():
        tot = 0.0
        for seg in query_segments(ni, genes):
            blk = ni.blocks[seg.block]
            tot += sum(blk.eps[s] ** 2 for s, _ in coarsest_decomposition(blk, seg))
        out[k] = float(np.sqrt(tot))
    return out


__all__ = [
    "IntervalIndex", "NicheIntervalIndex", "BlockIntervals", "IntervalNodes", "IntervalSearchResult",
    "build_interval_index", "search_interval_index", "coarsest_decomposition", "query_segments", "dyadic_intervals",
    "interval_cluster", "batched_log_spd", "sym_to_vec", "error_bound",
]
