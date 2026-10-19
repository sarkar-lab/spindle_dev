"""Spindle-lite, final configuration (paper/SPINDLE_LITE.md): a layered block DAG per niche, searched with a radius
set by the query's own best match.

Vectors are Spindle's (``spindle.BlockVectorizer``): a block's float32 upper-triangle log, off-diagonals times
sqrt(2), divided by sqrt(p), so the Euclidean distance of two block slices is d_b and of two whole vectors is d_B.

Build, per niche:
  1. layers = the niche's blocks by decreasing spread s_b = mean over its tiles of d_b(tile, mean block log)^2
     (ties in gene order);
  2. per block, farthest-first from tile 0 (argmax, first index on ties) gives the centre order and the cover
     radius r_b(K) after K centres (K <= 256);
  3. node counts: every K_b starts at 1; the K_b of the block with the largest s_b * (r_b(K)^2 - r_b(K')^2) / bytes
     of the added nodes (K' = min(2K, curve length)) is doubled until the node bytes reach BETA (0.10) times the
     niche's Spindle bytes;
  4. nodes = the first K_b centres; every tile joins its nearest centre (first on ties); a node stores its centre's
     vector (float32) and its radius (largest member distance, float32 rounded up); each tile stores its node per
     layer (one byte);
  5. node u (layer b) has child v (layer b + 1) when some tile is in both.
Search (``SpindleLite.search``), with the knob g (default 0.25):
  1. greedy probe in every niche: the first-layer node with the smallest d_b, then at each layer the child that
     shares tiles with the path with the smallest d_b; A_hat = the smallest A = sqrt(sum_b d_b^2) of these leaves;
  2. radius r = g * A_hat, the same in every niche;
  3. depth-first search of every niche: first-layer nodes and children by increasing lb_b = max(0, d_b - radius_b),
     children only where the path still shares tiles; a path stops once sum_b lb_b^2 > r^2;
  4. output: every tile reached, ranked by A(t) = sqrt(sum_b d_b(query, centre of t's node)^2).
Every node distance is computed at most once per query (the probe's are reused). Because lb_b <= d_b(query, tile)
for every member, every tile with d_B(query, tile) <= r is returned.

``search`` runs the compiled search (``kernels.FlatLite``: the same search breadth-first over the node index,
bit-identical output); ``search_vectors_dfs`` is the Python depth-first reference. Time ``search`` / ``search_vectors``.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .kernels import FlatLite
from .metrics import as_cov
from .spindle import BlockVectorizer, niche_layouts
from .utils import get_logger

logger = get_logger(__name__)

BETA = 0.10        # node bytes per niche, as a share of the niche's Spindle block-log bytes
K_MAX = 256        # most nodes per block (the node index fits in one byte)
G_DEFAULT = 0.25   # search knob: r = g * A_hat


def node_bytes(dim: int) -> int:
    """Stored bytes of one node: its float32 vector and its float32 radius."""
    return 4 * dim + 4


def _dist_to(L: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Euclidean distances of the rows of L to x (float64)."""
    diff = L - x
    return np.sqrt((diff * diff).sum(1))


def farthest_first(L: np.ndarray, k_max: int = K_MAX):
    """Farthest-first from tile 0 (argmax, first index on ties) until k_max centres or every tile is a centre's
    duplicate. Returns (centre order, r with r[K-1] = cover radius after K centres, (n, len(order)) distances of every
    tile to each centre)."""
    n = len(L)
    order = [0]
    cols = [_dist_to(L, L[0])]
    delta = cols[0].copy()
    r = [float(delta.max())]
    while len(order) < min(k_max, n):
        i = int(np.argmax(delta))
        if delta[i] <= 0.0:
            break
        order.append(i)
        cols.append(_dist_to(L, L[i]))
        delta = np.minimum(delta, cols[-1])
        r.append(float(delta.max()))
    return np.asarray(order), np.asarray(r), np.stack(cols, axis=1)


def allocate(curves: List[np.ndarray], dims: List[int], spread: List[float], target: float) -> List[int]:
    """Node count per layer: start at 1; double the count of the layer with the largest
    spread * (r(K)^2 - r(K')^2) / added bytes (ties: the earlier layer) until the node bytes reach ``target``."""
    K = [1] * len(curves)
    used = sum(node_bytes(d) for d in dims)

    def step(b):
        r, k = curves[b], K[b]
        k2 = min(2 * k, len(r))
        if k2 <= k:
            return None
        return spread[b] * (r[k - 1] ** 2 - r[k2 - 1] ** 2) / ((k2 - k) * node_bytes(dims[b])), k2

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


@dataclass
class LiteNiche:
    """One niche's DAG. Tile position t (0..n-1) is the niche's t-th training tile, row ``rows[t]``."""
    rows: np.ndarray                      # training rows (positions in data.labels) of the niche's tiles
    vectorizer: BlockVectorizer           # the niche's gene order and blocks (Spindle's)
    layer_blocks: List[int]               # block index (niche gene order) of each layer
    layer_slices: List[Tuple[int, int]]   # that block's slice of the niche's block-log vector
    spread: List[float]                   # s_b of each layer
    centre_tiles: List[np.ndarray]        # tile positions of each layer's centres (farthest-first order)
    centres: List[np.ndarray]             # (K_b, dim_b) float32 centre vectors
    radius: List[np.ndarray]              # (K_b,) float32 radii, rounded up
    node_index: List[np.ndarray]          # (n,) uint8: each tile's node in each layer
    children: List[List[np.ndarray]]      # layer b < B-1, node u: sorted nodes of layer b+1 sharing a tile with u
    _masks: Optional[List[List[int]]] = field(default=None, repr=False)

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def n_layers(self) -> int:
        return len(self.centres)

    def masks(self) -> List[List[int]]:
        """Per layer and node, the bitmask of its tiles (bit t = tile position t); derived from ``node_index``."""
        if self._masks is None:
            out = []
            for a, C in zip(self.node_index, self.centres):
                m = [0] * len(C)
                for t, j in enumerate(a.tolist()):
                    m[j] |= 1 << t
                out.append(m)
            self._masks = out
        return self._masks

    def nbytes(self) -> int:
        """Nodes (vector + radius), child edges (4 bytes each) and the node index (1 byte, 2 above 256 nodes)."""
        nodes = sum(len(C) * node_bytes(C.shape[1]) for C in self.centres)
        edges = 4 * sum(len(ch) for layer in self.children for ch in layer)
        index = sum(self.n * (1 if len(C) <= 256 else 2) for C in self.centres)
        return int(nodes + edges + index)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_masks"] = None
        return state


def build_niche(X: np.ndarray, vectorizer: BlockVectorizer, rows: np.ndarray, beta: float = BETA) -> LiteNiche:
    """One niche's DAG from its tiles' float32 block-log vectors X (n x dim, rows in niche order)."""
    n = len(X)
    widths = [len(w) for w in vectorizer.w]
    starts = np.cumsum([0] + widths)
    slices = [(int(starts[b]), int(starts[b + 1])) for b in range(len(widths))]
    Ls = [X[:, s:e].astype(np.float64) for s, e in slices]
    spread = [float(((L - L.mean(0)) ** 2).sum(1).mean()) for L in Ls]
    layer_blocks = [int(b) for b in np.argsort([-v for v in spread], kind="stable")]
    ffs = [farthest_first(Ls[b]) for b in layer_blocks]
    dims = [widths[b] for b in layer_blocks]
    s = [spread[b] for b in layer_blocks]
    K = allocate([r for _, r, _ in ffs], dims, s, beta * 4 * X.size)

    centre_tiles, centres, radius, node_index = [], [], [], []
    for b, (order, _, cols) in zip(layer_blocks, ffs):
        k = K[len(centres)]
        d = cols[:, :k]
        a = np.argmin(d, axis=1)
        rad = np.zeros(k)
        np.maximum.at(rad, a, d[np.arange(n), a])
        rad32 = rad.astype(np.float32)
        low = rad32.astype(np.float64) < rad
        rad32[low] = np.nextafter(rad32[low], np.float32(np.inf))
        centre_tiles.append(order[:k].copy())
        centres.append(X[order[:k], slices[b][0]:slices[b][1]].copy())
        radius.append(rad32)
        node_index.append(a.astype(np.uint8 if k <= 256 else np.uint16))
    children = []
    for b in range(len(layer_blocks) - 1):
        pairs = np.unique(np.stack([node_index[b].astype(np.int64), node_index[b + 1].astype(np.int64)], axis=1),
                          axis=0)
        kids = [[] for _ in range(len(centres[b]))]
        for u, v in pairs.tolist():
            kids[u].append(v)
        children.append([np.asarray(c, dtype=np.int64) for c in kids])
    return LiteNiche(np.asarray(rows), vectorizer, layer_blocks, [slices[b] for b in layer_blocks], s, centre_tiles,
                     centres, radius, node_index, children)


def _tiles_of(mask: int) -> List[int]:
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


class _Query:
    """One query's lazy node distances d_b(query, centre), each computed at most once; ``work`` counts them."""

    def __init__(self, niches: Dict[int, LiteNiche], qv: Dict[int, np.ndarray]):
        self.niches = niches
        self.qb = {k: [np.asarray(qv[k][s:e], dtype=np.float64) for s, e in nc.layer_slices] for k, nc in niches.items()}
        self.cache = {k: [np.full(len(C), np.nan) for C in nc.centres] for k, nc in niches.items()}
        self.masks = {k: nc.masks() for k, nc in niches.items()}
        self.work = 0

    def dist(self, k, b, idx):
        c = self.cache[k][b]
        miss = idx[np.isnan(c[idx])]
        if len(miss):
            diff = self.niches[k].centres[b][miss] - self.qb[k][b]
            c[miss] = np.sqrt((diff * diff).sum(1))
            self.work += len(miss)
        return c[idx]

    def children(self, k, b, node, mask):
        """[(child, path mask, d_b)] of node's children in layer b + 1 that still share tiles with the path."""
        tm = self.masks[k][b + 1]
        sel = [(ch, mask & tm[ch]) for ch in self.niches[k].children[b][node].tolist()]
        sel = [x for x in sel if x[1]]
        if not sel:
            return []
        d = self.dist(k, b + 1, np.fromiter((ch for ch, _ in sel), dtype=np.int64, count=len(sel)))
        return [(ch, m, float(dd)) for (ch, m), dd in zip(sel, d)]


def greedy_probe(qy: _Query) -> float:
    """A_hat: the smallest A over niches of the greedy leaf (closest first-layer node, then the closest child)."""
    best = np.inf
    for k, nc in qy.niches.items():
        d0 = qy.dist(k, 0, np.arange(len(nc.centres[0])))
        j = int(np.argmin(d0))
        a2, mask = float(d0[j]) ** 2, qy.masks[k][0][j]
        for b in range(nc.n_layers - 1):
            j, mask, dd = min(qy.children(k, b, j, mask), key=lambda x: x[2])
            a2 += dd * dd
        best = min(best, a2)
    return float(np.sqrt(best))


def radius_search(qy: _Query, r: float) -> List[Tuple[float, int, int]]:
    """Every tile whose path lower bound sqrt(sum_b lb_b^2) <= r, as (A^2, niche, tile position) by increasing A."""
    found = []
    r2 = r * r
    for k, nc in qy.niches.items():
        masks = qy.masks[k]

        def dfs(b, j, l2, a2, mask):
            if b == nc.n_layers - 1:
                found.extend((a2, k, t) for t in _tiles_of(mask))
                return
            rad = nc.radius[b + 1]
            ch = [(c, m, dd, max(0.0, dd - float(rad[c]))) for c, m, dd in qy.children(k, b, j, mask)]
            for c, m, dd, lb in sorted(ch, key=lambda x: x[3]):
                nl = l2 + lb * lb
                if nl > r2:
                    break
                dfs(b + 1, c, nl, a2 + dd * dd, m)

        d = qy.dist(k, 0, np.arange(len(nc.centres[0])))
        lb = np.maximum(0.0, d - nc.radius[0].astype(np.float64))
        for j in np.argsort(lb, kind="stable").tolist():
            if lb[j] ** 2 > r2:
                break
            dfs(0, j, float(lb[j]) ** 2, float(d[j]) ** 2, masks[0][j])
    found.sort()
    return found


class SpindleLite:
    """Spindle-lite (final configuration): build from the layout and the training covariances, search with knob g."""

    def __init__(self, data, train_covs, beta: float = BETA):
        layouts = niche_layouts(data)
        X = {k: np.stack([vec.vector(as_cov(train_covs[t]), np.float32) for t in rows])
             for k, (rows, vec) in layouts.items()}
        self._build(layouts, X, beta, data)

    @classmethod
    def from_spindle(cls, spindle, data, beta: float = BETA) -> "SpindleLite":
        """Build from a ``Spindle`` index's vectors (same layout and covariances), without recomputing logs."""
        obj = cls.__new__(cls)
        obj._build(spindle.layouts, spindle.X, beta, data)
        return obj

    def _build(self, layouts, X, beta, data):
        self.beta = float(beta)
        self.n = len(data.labels)
        self.tile_ids = np.asarray(data.spd_ids)
        self.spindle_bytes = int(sum(4 * x.size for x in X.values()))
        self.niches: Dict[int, LiteNiche] = {}
        for k, (rows, vec) in layouts.items():
            t0 = time.perf_counter()
            self.niches[k] = build_niche(np.asarray(X[k], dtype=np.float32), vec, rows, beta)
            nc = self.niches[k]
            logger.info("Niche %d: %d tiles, %d layers, %d nodes, %.1fs", k, nc.n, nc.n_layers,
                        sum(len(C) for C in nc.centres), time.perf_counter() - t0)
        self._flat = FlatLite(self.niches)

    @property
    def flat(self) -> FlatLite:
        """The flat arrays of the compiled search (derived from the niches; rebuilt after unpickling)."""
        if getattr(self, "_flat", None) is None:
            self._flat = FlatLite(self.niches)
        return self._flat

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_flat"] = None
        return state

    def nbytes(self) -> int:
        return int(sum(nc.nbytes() for nc in self.niches.values()))

    def n_nodes(self) -> int:
        return int(sum(len(C) for nc in self.niches.values() for C in nc.centres))

    def query_vectors(self, query_cov) -> Dict[int, np.ndarray]:
        """Per-niche float32 block-log vectors of a query covariance (as ``Spindle.query_vectors``)."""
        cov = as_cov(query_cov)
        return {k: nc.vectorizer.vector(cov, np.float32) for k, nc in self.niches.items()}

    def search_vectors(self, qv: Dict[int, np.ndarray], g: float = G_DEFAULT):
        """Search with precomputed query vectors (compiled). Returns (training rows by increasing A, A, stats)."""
        rows, scores, a_hat, r, work = self.flat.search(qv, g)
        stats = {"g": float(g), "a_hat": a_hat, "radius": r, "work": work, "n_found": len(rows)}
        return rows, scores, stats

    def search_vectors_many(self, qvs: List[Dict[int, np.ndarray]], g: float = G_DEFAULT,
                            threads: Optional[int] = None):
        """``search_vectors`` for many queries, in parallel threads (default: all cores)."""
        return [(rows, scores, {"g": float(g), "a_hat": a_hat, "radius": r, "work": work, "n_found": len(rows)})
                for rows, scores, a_hat, r, work in self.flat.search_many(qvs, g, threads)]

    def search_vectors_dfs(self, qv: Dict[int, np.ndarray], g: float = G_DEFAULT):
        """The Python depth-first reference of ``search_vectors`` (same output, much slower)."""
        qy = _Query(self.niches, qv)
        a_hat = greedy_probe(qy)
        r = g * a_hat
        found = radius_search(qy, r)
        rows = np.asarray([self.niches[k].rows[t] for _, k, t in found], dtype=np.int64)
        scores = np.sqrt(np.asarray([a2 for a2, _, _ in found], dtype=np.float64))
        stats = {"g": float(g), "a_hat": a_hat, "radius": r, "work": qy.work, "n_found": len(found)}
        return rows, scores, stats

    def search(self, query_cov, g: float = G_DEFAULT):
        """Training rows best first (by A), their scores A and the seconds taken; ``last_stats`` holds the probe's
        A_hat, the radius, the node distances computed (``work``) and the number of tiles returned."""
        t0 = time.perf_counter()
        qv = self.query_vectors(query_cov)
        t1 = time.perf_counter()
        rows, scores, stats = self.search_vectors(qv, g)
        t2 = time.perf_counter()
        self.last_stats = {**stats, "query_s": t1 - t0, "search_s": t2 - t1}
        return rows, scores, t2 - t0
