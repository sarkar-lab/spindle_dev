"""Interval index for gene-set (partial) queries.

Index
-----
For every niche and block of the niche's gene order, the block positions 0..n-1 are covered by the
dyadic intervals I(s, a) = {a, ..., a+s-1} with s = 2^k <= max_len, s | a and a + s <= n. For each
interval, the tiles' interval logs X_m^I = log Sigma_m[g(I)] (shrunk covariances) are partitioned so
that every member lies within eps_{beta,s} of its node mean (farthest-first cover at eps/2,
nearest-centre assignment, clusters that break the bound re-covered at eps/2). eps_{beta,s} is set so
that a length-s interval has about K_s = k_target * s^k_alpha nodes. The index stores, per interval, the
float32 node means and one node code per tile.

Distances are unnormalised Frobenius distances between matrix logarithms; symmetric matrices are stored
as upper-triangle vectors with off-diagonals scaled by sqrt 2.

Search
------
A gene set S is cut, in every niche, into segments (maximal runs of consecutive positions in one block),
and each segment into the fewest dyadic pieces (from each position, the longest indexed interval that
fits). A tile's score is

    sqrt( sum over pieces I of ||Q^I - mu_{node of the tile on I}||_F^2 ),

which differs from the same sum over the tile's own interval logs by at most sqrt(sum_I eps^2)
(``error_bound``). Tiles of all niches are ranked by the score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import LOG_FLOOR, batched_log_spd, query_cov_s, sym_to_vec
from .spindle import block_cov_stacks
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------------------------- clustering
def _assign(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Nearest row of C for every row of X."""
    d = (X * X).sum(1)[:, None] - 2.0 * X @ C.T + (C * C).sum(1)[None, :]
    return np.argmin(np.maximum(d, 0.0), axis=1)


class FarthestFirst:
    """Resumable farthest-first traversal from row 0; ``radii[k]`` is the cover radius with k+1 centres."""

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
        """Centres of the shortest prefix whose cover radius is <= ``radius``."""
        self.extend_to(radius=radius)
        k = next((i + 1 for i, r in enumerate(self.radii) if r <= radius), len(self.radii))
        return np.asarray(self.centres[:k])


def interval_cluster(X: np.ndarray, eps: float, ff: Optional[FarthestFirst] = None) -> np.ndarray:
    """One node code per row of X such that every row lies within ``eps`` of its node mean."""
    ff = ff if ff is not None else FarthestFirst(X)
    _, codes = np.unique(_assign(X, X[ff.cover(eps / 2.0)]), return_inverse=True)
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


# ---------------------------------------------------------------------------------------------- index
@dataclass
class IntervalNodes:
    """Nodes of one dyadic interval I(s, a) of one block."""
    s: int
    a: int
    means: np.ndarray            # (K, s(s+1)/2) float32 node means
    codes: np.ndarray            # (n_niche,) node of every niche tile

    @property
    def n_nodes(self) -> int:
        return len(self.means)


@dataclass
class BlockIntervals:
    """The dyadic intervals of one block and their nodes."""
    genes: np.ndarray                                  # global gene index of every block position
    eps: Dict[int, float]                              # radius per interval length
    intervals: Dict[Tuple[int, int], IntervalNodes]    # (s, a) -> nodes

    def starting_at(self, a: int, end_max: int) -> List[Tuple[int, int]]:
        """Interval keys starting at ``a`` that end no later than ``end_max``, shortest first."""
        keys, s = [], 1
        while a % s == 0 and a + s <= end_max:
            if (s, a) in self.intervals:
                keys.append((s, a))
            s *= 2
        return keys


@dataclass
class NicheIntervalIndex:
    niche: int
    rows: np.ndarray              # training row of every niche tile
    blocks: List[BlockIntervals]
    gene_block: np.ndarray        # (p,) block of every global gene in this niche
    gene_pos: np.ndarray          # (p,) position of every global gene inside its block

    @property
    def n_tiles(self) -> int:
        return len(self.rows)


@dataclass
class IntervalIndex:
    niches: Dict[int, NicheIntervalIndex]
    n_tiles: int
    p: int
    params: dict
    build_seconds: float = 0.0

    def n_nodes(self) -> int:
        return sum(iv.n_nodes for ni in self.niches.values() for b in ni.blocks for iv in b.intervals.values())

    def nbytes(self) -> int:
        """float32 node means plus one code per (tile, interval) of the smallest sufficient width."""
        total = 0
        for ni in self.niches.values():
            for b in ni.blocks:
                for iv in b.intervals.values():
                    total += iv.means.size * 4
                    total += iv.codes.size * (1 if iv.n_nodes <= 256 else 2 if iv.n_nodes <= 65536 else 4)
        return total


def dyadic_intervals(n: int, max_len: Optional[int] = None) -> List[Tuple[int, int]]:
    """(s, a) of every dyadic interval of a block with n positions, by increasing s."""
    out, s = [], 1
    while s <= n and (max_len is None or s <= max_len):
        out.extend((s, a) for a in range(0, n - s + 1, s))
        s *= 2
    return out


def _build_block(block_covs: np.ndarray, genes: np.ndarray, k_target, k_alpha, max_len) -> BlockIntervals:
    """block_covs: (n_niche, b, b) prepared covariances of the niche's tiles on the block's genes."""
    by_len: Dict[int, List[int]] = {}
    for s, a in dyadic_intervals(len(genes), max_len):
        by_len.setdefault(s, []).append(a)

    intervals: Dict[Tuple[int, int], IntervalNodes] = {}
    eps_used: Dict[int, float] = {}
    for s, starts in by_len.items():
        X = {a: sym_to_vec(batched_log_spd(block_covs[:, a:a + s, a:a + s], LOG_FLOOR)) for a in starts}
        ffs = {a: FarthestFirst(X[a]) for a in starts}
        # eps/2 = median over this length's intervals of the farthest-first radius with K_s centres
        k_s = max(1, int(round(k_target * s ** k_alpha)))
        radii = []
        for a in starts:
            ffs[a].extend_to(k=k_s)
            radii.append(ffs[a].radii[min(k_s, len(ffs[a].radii)) - 1])
        e = max(2.0 * float(np.median(radii)), 1e-12)
        eps_used[s] = e
        for a in starts:
            codes = interval_cluster(X[a], e, ff=ffs[a])
            K = int(codes.max()) + 1
            means = np.zeros((K, X[a].shape[1]))
            np.add.at(means, codes, X[a])
            means /= np.bincount(codes, minlength=K)[:, None]
            intervals[(s, a)] = IntervalNodes(s, a, means.astype(np.float32), codes.astype(np.int32))
    return BlockIntervals(np.asarray(genes), eps_used, intervals)


def build_interval_index(data, train_covs=None, *, block_covs=None, k_target: int = 32, k_alpha: float = 0.5,
                         max_len: Optional[int] = 16) -> IntervalIndex:
    """The interval index of every niche of ``data``.

    Pass the raw training covariances (``data.labels`` order), or ``block_covs`` from
    ``spindle.block_cov_stacks`` / ``SpindlePartial.covs`` to reuse them.
    """
    t0 = time.perf_counter()
    if block_covs is None:
        block_covs = block_cov_stacks(data, train_covs)
    labels = np.asarray(data.labels).astype(int)
    p = len(data.metadata["genes"])
    out: Dict[int, NicheIntervalIndex] = {}
    for k in sorted(set(labels.tolist())):
        perm = np.asarray(data.perm_list[k])
        gene_block = np.full(p, -1, dtype=np.int32)
        gene_pos = np.full(p, -1, dtype=np.int32)
        blocks = []
        for b, (start, end) in enumerate(data.block_dict[k]):
            genes = perm[start:end]
            gene_block[genes] = b
            gene_pos[genes] = np.arange(end - start)
            blocks.append(_build_block(block_covs[k][b], genes, k_target, k_alpha, max_len))
        out[k] = NicheIntervalIndex(k, np.flatnonzero(labels == k), blocks, gene_block, gene_pos)
        logger.info("Interval index niche %d: %d tiles, %d blocks", k, out[k].n_tiles, len(blocks))
    params = dict(k_target=k_target, k_alpha=k_alpha, max_len=max_len)
    return IntervalIndex(out, len(labels), p, params, time.perf_counter() - t0)


# ---------------------------------------------------------------------------------------------- search
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


def coarsest_decomposition(blk: BlockIntervals, seg: Segment) -> List[Tuple[int, int]]:
    """The fewest-piece decomposition: from each position, the longest indexed interval that fits."""
    keys, c = [], seg.a
    while c < seg.b:
        key = blk.starting_at(c, seg.b)[-1]
        keys.append(key)
        c += key[0]
    return keys


def _query_pieces(blk: BlockIntervals, keys, cov_s, col_of_gene) -> Dict[Tuple[int, int], np.ndarray]:
    """Q^I as Frobenius vectors for every interval key, batched by length."""
    out, by_len = {}, {}
    for s, a in keys:
        by_len.setdefault(s, []).append(a)
    for s, starts in by_len.items():
        cols = np.stack([col_of_gene[blk.genes[a:a + s]] for a in starts])
        vecs = sym_to_vec(batched_log_spd(cov_s[cols[:, :, None], cols[:, None, :]], LOG_FLOOR))
        out.update({(s, a): v for a, v in zip(starts, vecs)})
    return out


def search_interval_index(index: IntervalIndex, query_cov, genes: Sequence[int], c: Optional[int] = None):
    """Rank every training tile for the gene set ``genes``; ``query_cov`` is the full p x p query or the
    |S| x |S| one in ``genes`` order. Returns (rows best first, scores, seconds)."""
    t0 = time.perf_counter()
    genes = np.asarray(genes)
    cov_s = query_cov_s(query_cov, genes, index.p)
    col_of_gene = np.full(index.p, -1, dtype=np.int64)
    col_of_gene[genes] = np.arange(len(genes))

    rows, scores = [], []
    for ni in index.niches.values():
        sq = np.zeros(ni.n_tiles)
        for seg in query_segments(ni, genes):
            blk = ni.blocks[seg.block]
            keys = coarsest_decomposition(blk, seg)
            q = _query_pieces(blk, keys, cov_s, col_of_gene)
            for key in keys:
                iv = blk.intervals[key]
                sq += ((iv.means.astype(np.float64) - q[key]) ** 2).sum(1)[iv.codes]
        rows.append(ni.rows)
        scores.append(np.sqrt(sq))
    rows = np.concatenate(rows)
    scores = np.concatenate(scores)
    order = np.argsort(scores, kind="stable")
    if c is not None:
        order = order[:c]
    return rows[order], scores[order], time.perf_counter() - t0


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
