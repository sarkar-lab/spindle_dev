"""Spindle-lite partial, final configuration (paper/SPINDLE_PARTIAL.md): gene-set queries on dyadic gene intervals,
searched with a radius set by the query's own best match.

Distances are unnormalised Frobenius distances between matrix logs (symmetric matrices as upper triangles with
off-diagonals times sqrt(2)); a tile's score is divided by sqrt(|S|) only at the end, in d_B^S units.

Build, per niche (n tiles, its gene order and blocks from the layout):
  1. intervals: every aligned dyadic interval I(s, j) of every block (s = 1, 2, 4, ... up to the block length,
     s | j, j + s <= m), in (block, s, j) order; a tile's interval vector is the float32 upper triangle of
     log Sigma_t[I, I];
  2. spread s_I = mean over tiles of ||v_t - mean v||^2;
  3. per interval, farthest-first from tile 0 (lite.farthest_first, at most 256 centres); node counts: every K_I
     starts at 1 and the K_I with the largest (r(K)^2 - r(K')^2) / s_I / added bytes is doubled (K' = min(2K, curve
     length); ties to the earlier interval) until the node bytes reach BETA (0.10) times the niche's Spindle bytes;
  4. nodes = the first K_I centres (float32 vectors), each tile joins its nearest (first on ties), radius = largest
     member distance (float32, rounded up); each tile stores its node per interval (one byte). No edges are stored:
     two nodes of consecutive pieces are linked when a tile points to both, which the node index already says.
Search (``search``), one gene set S, knob g (default 1):
  1. pieces: per niche, S is cut into segments (maximal runs of consecutive positions inside one block) and each
     segment into the fewest aligned dyadic pieces, left to right; the query's piece vectors are logs of Q[I, I];
  2. greedy probe in every niche: the closest first-piece node, then at each next piece the closest node that shares
     tiles with the path; A_hat^2 = the smallest sum of d^2 over niches;
  3. radius: pruning compares squares with r^2 = g^2 * A_hat^2 (no square-root round trip);
  4. depth-first search: first-piece nodes, then at each piece the nodes that the path's tiles point to, by
     increasing lb = max(0, d - radius); a path stops once its sum of lb^2 exceeds r^2;
  5. output: every tile reached, ranked by A = sqrt(sum over pieces of d^2 / |S|).
Every node distance is computed at most once per query (the probe's are reused). Because lb <= d(query piece, the
tile's own interval log) for every member, every tile whose piece distance is <= r is returned.

``search`` runs the compiled search (``kernels.FlatPartial``: the same search breadth-first over the node index,
bit-identical output); ``search_vectors_dfs`` is the Python depth-first reference. Time ``search_vectors``.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kernels import FlatPartial
from .lite import K_MAX, farthest_first, node_bytes
from .metrics import LOG_FLOOR, batched_log_spd, query_cov_s, sym_to_vec
from .spindle import block_cov_stacks
from .utils import get_logger

logger = get_logger(__name__)

BETA = 0.10       # node bytes per niche, as a share of the niche's Spindle block-log bytes
G_DEFAULT = 1.0   # search knob: r = g * A_hat


# ---------------------------------------------------------------------------------------------- intervals
def aligned_intervals(m: int) -> List[Tuple[int, int]]:
    """(s, j) of every aligned dyadic interval of a block with m positions, by increasing s, then j."""
    out, s = [], 1
    while s <= m:
        out.extend((s, j) for j in range(0, m - s + 1, s))
        s *= 2
    return out


def dyadic_pieces(a: int, e: int) -> List[Tuple[int, int]]:
    """The fewest aligned dyadic intervals (s, j) covering positions a .. e-1, left to right."""
    out, c = [], a
    while c < e:
        s = 1
        while c % (2 * s) == 0 and c + 2 * s <= e:
            s *= 2
        out.append((s, c))
        c += s
    return out


def interval_vectors(covs: np.ndarray, s: int, j: int) -> np.ndarray:
    """sym_to_vec(log cov[j:j+s, j:j+s]) of a stack of block covariances (float64)."""
    sub = np.asarray(covs[..., j:j + s, j:j + s], dtype=np.float64)
    return sym_to_vec(batched_log_spd(sub, LOG_FLOOR))


def allocate_relative(curves: List[np.ndarray], dims: List[int], spread: List[float], target: float) -> List[int]:
    """Node count per interval: start at 1; double the count of the interval with the largest
    (r(K)^2 - r(K')^2) / s_I / added bytes (ties: the earlier interval; zero-spread intervals keep one node) until
    the node bytes reach ``target``."""
    K = [1] * len(curves)
    used = sum(node_bytes(d) for d in dims)

    def step(b):
        r, k = curves[b], K[b]
        k2 = min(2 * k, len(r))
        if k2 <= k or spread[b] <= 0:
            return None
        return (r[k - 1] ** 2 - r[k2 - 1] ** 2) / spread[b] / ((k2 - k) * node_bytes(dims[b])), k2

    heap = []
    for b in range(len(curves)):
        st = step(b)
        if st:
            heapq.heappush(heap, (-st[0], b, st[1]))
    while heap and used < target:
        _, b, k2 = heapq.heappop(heap)
        used += (k2 - K[b]) * node_bytes(dims[b])
        K[b] = k2
        st = step(b)
        if st:
            heapq.heappush(heap, (-st[0], b, st[1]))
    return K


# ---------------------------------------------------------------------------------------------- index
@dataclass
class PartialNiche:
    """One niche's interval nodes. Tile position t (0..n-1) is the niche's t-th training tile, row ``rows[t]``."""
    rows: np.ndarray                      # training rows of the niche's tiles
    perm: np.ndarray                      # the niche's gene order (global gene ids)
    block_starts: List[int]               # start of each block in ``perm``
    gene_block: np.ndarray                # (p,) block of every global gene in this niche
    gene_pos: np.ndarray                  # (p,) position of every global gene inside its block
    keys: List[Tuple[int, int, int]]      # (block, s, j) of each interval
    spread: List[float]                   # s_I of each interval
    centres: List[np.ndarray]             # (K_I, s(s+1)/2) float32 centre vectors
    radius: List[np.ndarray]              # (K_I,) float32 radii, rounded up
    node_index: List[np.ndarray]          # (n,) uint8: each tile's node in each interval
    spindle_bytes: int = 0
    pos: Dict[Tuple[int, int, int], int] = field(default_factory=dict)

    def __post_init__(self):
        self.pos = {key: i for i, key in enumerate(self.keys)}

    @property
    def n(self) -> int:
        return len(self.rows)

    def nbytes(self) -> int:
        """Nodes (vector + radius) and the node index (one byte per tile and interval)."""
        nodes = sum(len(C) * node_bytes(C.shape[1]) for C in self.centres)
        return int(nodes + self.n * len(self.centres))

    def pieces(self, genes: np.ndarray) -> List[Tuple[int, int, int]]:
        """(block, s, j) of the gene set's pieces in this niche, left to right (block, then position)."""
        blk, pos = self.gene_block[genes], self.gene_pos[genes]
        if np.any(blk < 0):
            raise ValueError("query gene outside the niche's blocks")
        out = []
        for b in np.unique(blk):
            ps = np.sort(pos[blk == b])
            cut = np.flatnonzero(np.diff(ps) > 1)
            for a, e in zip(np.r_[ps[0], ps[cut + 1]], np.r_[ps[cut], ps[-1]] + 1):
                out.extend((int(b), s, j) for s, j in dyadic_pieces(int(a), int(e)))
        return out


def build_niche(covs: List[np.ndarray], perm, block_runs, rows, p: int, beta: float = BETA) -> PartialNiche:
    """One niche's interval nodes from its tiles' block covariances (``spindle.block_cov_stacks`` order)."""
    n = len(rows)
    ms = [e - s for s, e in block_runs]
    spindle_bytes = 4 * n * sum(m * (m + 1) // 2 for m in ms)
    keys, dims, spread, curves, ffs, vecs = [], [], [], [], [], []
    for b, m in enumerate(ms):
        for s, j in aligned_intervals(m):
            v = interval_vectors(covs[b], s, j).astype(np.float32)
            L = v.astype(np.float64)
            order, r, cols = farthest_first(L, K_MAX)
            keys.append((b, s, j))
            dims.append(v.shape[1])
            spread.append(float(((L - L.mean(0)) ** 2).sum(1).mean()))
            curves.append(r)
            ffs.append((order, cols))
            vecs.append(v)
    K = allocate_relative(curves, dims, spread, beta * spindle_bytes)
    centres, radius, node_index = [], [], []
    for (order, cols), v, k in zip(ffs, vecs, K):
        d = cols[:, :k]
        a = np.argmin(d, axis=1)
        rad = np.zeros(k)
        np.maximum.at(rad, a, d[np.arange(n), a])
        rad32 = rad.astype(np.float32)
        low = rad32.astype(np.float64) < rad
        rad32[low] = np.nextafter(rad32[low], np.float32(np.inf))
        centres.append(v[order[:k]].copy())
        radius.append(rad32)
        node_index.append(a.astype(np.uint8))
    perm = np.asarray(perm)
    gene_block = np.full(p, -1, dtype=np.int64)
    gene_pos = np.full(p, -1, dtype=np.int64)
    for b, (s0, e0) in enumerate(block_runs):
        gene_block[perm[s0:e0]] = b
        gene_pos[perm[s0:e0]] = np.arange(e0 - s0)
    return PartialNiche(np.asarray(rows), perm, [int(s0) for s0, _ in block_runs], gene_block, gene_pos, keys,
                        spread, centres, radius, node_index, spindle_bytes)


@dataclass
class QueryPieces:
    """A gene-set query's pieces: per niche, [(interval position in the niche, float32 query piece vector)]."""
    n_genes: int
    layers: Dict[int, List[Tuple[int, np.ndarray]]]


# ---------------------------------------------------------------------------------------------- reference search
class _Query:
    """One query's lazy node distances d(query piece, centre), each computed at most once; ``work`` counts them."""

    def __init__(self, niches: Dict[int, PartialNiche], qp: QueryPieces):
        self.niches = niches
        self.layers = qp.layers
        self.q = {k: [np.asarray(v, dtype=np.float64) for _, v in qp.layers[k]] for k in niches}
        self.cache = {k: [np.full(len(niches[k].centres[iv]), np.nan) for iv, _ in qp.layers[k]] for k in niches}
        self.work = 0

    def dist(self, k, l, idx):
        c = self.cache[k][l]
        miss = idx[np.isnan(c[idx])]
        if len(miss):
            iv = self.layers[k][l][0]
            diff = self.niches[k].centres[iv][miss].astype(np.float64) - self.q[k][l]
            c[miss] = np.sqrt((diff * diff).sum(1))
            self.work += len(miss)
        return c


def greedy_probe(qy: _Query) -> float:
    """The smallest sum of d^2 over niches of the greedy path (unnormalised A_hat^2)."""
    best = np.inf
    for k, nc in qy.niches.items():
        layers = qy.layers[k]
        if not layers:
            continue
        iv0 = layers[0][0]
        d = qy.dist(k, 0, np.arange(len(nc.centres[iv0])))
        j = int(np.argmin(d))
        a2 = float(d[j]) * float(d[j])
        mask = nc.node_index[iv0] == j
        for l in range(1, len(layers)):
            a = nc.node_index[layers[l][0]]
            cand = np.unique(a[mask])
            d = qy.dist(k, l, cand)
            j = int(cand[np.argmin(d[cand])])
            a2 += float(d[j]) * float(d[j])
            mask &= a == j
        best = min(best, a2)
    return float(best)


def radius_search(qy: _Query, r2: float) -> List[Tuple[float, int, int]]:
    """Every tile whose path keeps sum of lb^2 <= r2, as (A^2 unnormalised, niche, tile position), sorted (depth-first
    over the pieces; a node's children are the next piece's nodes that the path's tiles point to)."""
    found = []
    for k, nc in qy.niches.items():
        layers = qy.layers[k]
        if not layers:
            continue

        def dfs(l, mask, l2, a2):
            if l == len(layers):
                found.extend((a2, k, int(t)) for t in np.flatnonzero(mask))
                return
            iv = layers[l][0]
            a = nc.node_index[iv]
            ch = np.unique(a[mask])
            d = qy.dist(k, l, ch)
            rad = nc.radius[iv].astype(np.float64)
            lbs = [(max(0.0, float(d[c]) - float(rad[c])), int(c)) for c in ch]
            for lb, c in sorted(lbs, key=lambda x: x[0]):
                nl = l2 + lb * lb
                if nl > r2:
                    break
                dfs(l + 1, mask & (a == c), nl, a2 + float(d[c]) * float(d[c]))

        iv0 = layers[0][0]
        d = qy.dist(k, 0, np.arange(len(nc.centres[iv0])))
        lb = np.maximum(0.0, d - nc.radius[iv0].astype(np.float64))
        a0 = nc.node_index[iv0]
        for j in np.argsort(lb, kind="stable").tolist():
            if float(lb[j]) * float(lb[j]) > r2:
                break
            dfs(1, a0 == j, float(lb[j]) * float(lb[j]), float(d[j]) * float(d[j]))
    found.sort()
    return found


# ---------------------------------------------------------------------------------------------- index class
class SpindleLitePartial:
    """Spindle-lite partial (final configuration): build from the layout and the training covariances, search a gene
    set with knob g."""

    def __init__(self, data, train_covs, beta: float = BETA):
        self._build(data, block_cov_stacks(data, train_covs), beta)

    @classmethod
    def from_partial(cls, partial, data, beta: float = BETA) -> "SpindleLitePartial":
        """Build from a ``SpindlePartial`` index's block covariances (same layout and covariances)."""
        obj = cls.__new__(cls)
        obj._build(data, partial.covs, beta)
        return obj

    def _build(self, data, block_covs, beta):
        labels = np.asarray(data.labels).astype(int)
        self.beta = float(beta)
        self.n = len(labels)
        self.p = len(data.metadata["genes"])
        self.tile_ids = np.asarray(data.spd_ids)
        self.niches: Dict[int, PartialNiche] = {}
        for k in sorted(set(labels.tolist())):
            t0 = time.perf_counter()
            nc = build_niche(block_covs[k], data.perm_list[k], data.block_dict[k], np.flatnonzero(labels == k),
                             self.p, beta)
            self.niches[k] = nc
            logger.info("Niche %d: %d tiles, %d intervals, %d nodes, %.1fs", k, nc.n, len(nc.keys),
                        sum(len(C) for C in nc.centres), time.perf_counter() - t0)
        self.spindle_bytes = int(sum(nc.spindle_bytes for nc in self.niches.values()))
        self._flat = FlatPartial(self.niches)

    @property
    def flat(self) -> FlatPartial:
        """The flat arrays of the compiled search (derived from the niches; rebuilt after unpickling)."""
        if getattr(self, "_flat", None) is None:
            self._flat = FlatPartial(self.niches)
        return self._flat

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_flat"] = None
        return state

    def nbytes(self) -> int:
        return int(sum(nc.nbytes() for nc in self.niches.values()))

    def n_nodes(self) -> int:
        return int(sum(len(C) for nc in self.niches.values() for C in nc.centres))

    def query_vectors(self, query_cov, genes: Sequence[int]) -> QueryPieces:
        """The gene set's pieces in every niche and the query's float32 piece vectors. ``query_cov`` is the full
        p x p query (shrunk over all p genes, as the tiles) or the |S| x |S| one in ``genes`` order."""
        genes = np.asarray(genes)
        cov_s = query_cov_s(query_cov, genes, self.p)
        col = np.full(self.p, -1, dtype=np.int64)
        col[genes] = np.arange(len(genes))
        layers = {}
        for k, nc in self.niches.items():
            out = []
            for b, s, j in nc.pieces(genes):
                cq = col[nc.perm[nc.block_starts[b] + j:nc.block_starts[b] + j + s]]
                v = sym_to_vec(batched_log_spd(cov_s[np.ix_(cq, cq)][None], LOG_FLOOR))[0].astype(np.float32)
                out.append((nc.pos[(b, s, j)], v))
            layers[k] = out
        return QueryPieces(len(genes), layers)

    def _stats(self, g, best_a2, work, n_found, n_genes):
        a_hat = float(np.sqrt(best_a2 / n_genes))
        return {"g": float(g), "a_hat": a_hat, "radius": float(g) * a_hat, "work": int(work), "n_found": int(n_found)}

    def search_vectors(self, qp: QueryPieces, g: float = G_DEFAULT):
        """Search with precomputed query pieces (compiled). Returns (training rows by increasing A, A, stats)."""
        rows, scores, best_a2, work = self.flat.search(qp, g)
        return rows, scores, self._stats(g, best_a2, work, len(rows), qp.n_genes)

    def search_vectors_many(self, qps: List[QueryPieces], g: float = G_DEFAULT, threads: Optional[int] = None):
        """``search_vectors`` for many queries, in parallel threads (default: all cores)."""
        return [(rows, scores, self._stats(g, best_a2, work, len(rows), qp.n_genes))
                for qp, (rows, scores, best_a2, work) in zip(qps, self.flat.search_many(qps, g, threads))]

    def search_vectors_dfs(self, qp: QueryPieces, g: float = G_DEFAULT):
        """The Python depth-first reference of ``search_vectors`` (same output, much slower)."""
        qy = _Query(self.niches, qp)
        best_a2 = greedy_probe(qy)
        found = radius_search(qy, (g * g) * best_a2)
        rows = np.asarray([self.niches[k].rows[t] for _, k, t in found], dtype=np.int64)
        scores = np.sqrt(np.asarray([a2 for a2, _, _ in found], dtype=np.float64) / qp.n_genes)
        return rows, scores, self._stats(g, best_a2, qy.work, len(found), qp.n_genes)

    def search(self, query_cov, genes: Sequence[int], g: float = G_DEFAULT):
        """Training rows best first (by A), their scores A (d_B^S units) and the seconds taken; ``last_stats`` holds
        A_hat, the radius, the node distances computed (``work``) and the number of tiles returned."""
        t0 = time.perf_counter()
        qp = self.query_vectors(query_cov, genes)
        t1 = time.perf_counter()
        rows, scores, stats = self.search_vectors(qp, g)
        t2 = time.perf_counter()
        self.last_stats = {**stats, "query_s": t1 - t0, "search_s": t2 - t1}
        return rows, scores, t2 - t0
