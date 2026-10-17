"""Spindle-lite: a layered block DAG per niche with budgeted depth-first search.

Distances are Spindle's: block b contributes d_b(A, B) = ||log A_b - log B_b||_F / sqrt(p) (p = all genes,
shrunk covariances, eigenvalue floor metrics.LOG_FLOOR), and a path adds them in L2, so the full path
distance to a tile is d_B.

Build. In every niche, layer b holds the tiles' block-b logs (niche gene order). Each layer is covered by an
epsilon-net (farthest-first centres from tile 0 until every tile is within the block's epsilon under d_b);
each tile joins its nearest centre. A node stores its centre (``mean``), its members' mean log
(``representative_mean``), its members and its radius (largest member d_b to the centre). Node u in layer b
points to node v in layer b+1 when they share a tile. Per-block epsilons are the farthest-first cover radius
at k = max(64, 0.05 n) centres (``choose_adaptive_epsilons``); each niche's budget epsilon is the L2 sum of
the blocks' elbow radii, floored at the median over niches.

Search. Every niche is searched with budget = epsilon_niche * budget_multiplier * f
(f = clip(sqrt(n_niche / 500), 1, 6)). From the first-layer nodes, closest first, the DFS follows children
that still share tiles with the path, closest first. Each layer adds lb_b = max(0, d_b(query, centre) - radius)
and a path is pruned when sqrt(sum_b lb_b^2) exceeds the budget. By the triangle inequality lb_b <= d_b(query,
tile) for every member, so no tile with d_B(query, tile) <= budget is pruned by the budget (the effort caps
can still stop the search). Each leaf reached returns the tiles shared by every node on its path.
"""

from __future__ import annotations

import pickle
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .metrics import as_cov, log_spd
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------------------------- DAG types
@dataclass
class BlockClusterMetadata:
    members: List[Tuple[int, int]]        # (spd_id, block index)
    mean: np.ndarray                      # epsilon-net centre (log)
    representative_mean: np.ndarray       # members' mean log
    radius: float


@dataclass
class BlockClusterNode:
    global_node_id: int                   # position in IndexHandle.nodes
    block_index: int
    metadata: BlockClusterMetadata
    children: List[int] = field(default_factory=list)


@dataclass
class IndexHandle:
    """One niche's DAG."""
    nodes: List[BlockClusterNode]
    block_runs: List[Tuple[int, int]]
    perm: np.ndarray
    block_to_node_indices: Dict[int, List[int]]
    sorted_blocks: List[int]


@dataclass
class LiteIndex:
    dag_dict: Dict[int, IndexHandle]
    epsilon_dict: Dict[int, float]                     # niche -> budget epsilon
    epsilon_block_wise_dict: Dict[int, Dict[int, float]]
    niche_sizes: Dict[int, int]
    row_of_spd_id: Dict[int, int]


# ---------------------------------------------------------------------------------------------- epsilons
def _block_dist(A, B, p):
    """d_b in d_B units: ||A - B||_F / sqrt(p), p = all genes."""
    return float(np.linalg.norm(A - B, ord="fro") / np.sqrt(p))


def farthest_first_eps_for_k(L_list, p, k_target, seed=0):
    """Cover radius of farthest-first with ``k_target`` centres (random first centre), and the radius curve."""
    n = len(L_list)
    if n == 0:
        return 0.0, []
    first = int(np.random.default_rng(seed).integers(0, n))
    n_centres = 1
    delta = np.array([_block_dist(L_list[i], L_list[first], p) for i in range(n)], dtype=float)
    delta[first] = 0.0
    deltas_max = [float(delta.max())]
    while n_centres < min(k_target, n):
        i_star = int(np.argmax(delta))
        if delta[i_star] <= 0.0:
            break
        n_centres += 1
        for i in range(n):
            d = _block_dist(L_list[i], L_list[i_star], p)
            if d < delta[i]:
                delta[i] = d
        deltas_max.append(float(delta.max()))
    return deltas_max[min(len(deltas_max), k_target) - 1], deltas_max


def choose_eps_from_curve(deltas_max):
    """Elbow of the radius curve: largest second difference of its log."""
    if not deltas_max:
        return 0.0
    y = np.log(np.array(deltas_max) + 1e-12)
    if len(y) < 3:
        return float(deltas_max[-1])
    t = int(np.argmax(y[:-2] - 2 * y[1:-1] + y[2:])) + 2
    return float(deltas_max[t - 1])


def _effective_k_target(n: int, base: int, alpha: float = 0.05, k_min: int = 8, k_max: int = 256) -> int:
    return int(min(max(round(alpha * n), base, k_min), k_max))


def choose_adaptive_epsilons(data, cluster_id, k_target_per_block, alpha=0.05, k_min=8, k_max=256):
    """(eps_per_block, eps_elbow_per_block, eps) of one niche.

    eps_per_block: farthest-first cover radius at k = _effective_k_target (drives the epsilon-net);
    eps: sqrt of the sum of the squared elbow radii (sizes the search budget). All in d_B units.
    """
    indices = np.where(data.labels == cluster_id)[0]
    block_runs = data.block_dict[cluster_id]
    permutation = data.perm_list[cluster_id]
    p = len(permutation)
    block_log = defaultdict(list)
    for i in indices:
        spd_i = as_cov(data.spd_matrices[i])
        for block_idx, (start, end) in enumerate(block_runs):
            idx = permutation[start:end]
            block_log[block_idx].append(log_spd(spd_i[np.ix_(idx, idx)]))

    eps_per_block, eps_elbow_per_block = {}, {}
    for block_idx, L_list in block_log.items():
        k = _effective_k_target(len(L_list), base=k_target_per_block, alpha=alpha, k_min=k_min, k_max=k_max)
        eps_hat, deltas_max = farthest_first_eps_for_k(L_list, p, k_target=k, seed=42)
        eps_per_block[block_idx] = eps_hat
        eps_elbow_per_block[block_idx] = choose_eps_from_curve(deltas_max)
    eps = float(np.sqrt(sum(eps_elbow_per_block[b] ** 2 for b in range(len(block_runs)))))
    return eps_per_block, eps_elbow_per_block, eps


# ---------------------------------------------------------------------------------------------- build
def _epsilon_net_layer(L_list, spd_ids, epsilon, p):
    """Epsilon-net clusters of one layer: [{'spd_ids', 'mean', 'representative_mean', 'radius'}]."""
    n = len(L_list)
    centres = [0]
    delta = np.array([_block_dist(L_list[i], L_list[0], p) for i in range(n)], dtype=float)
    while True:
        i_star = int(np.argmax(delta))
        if delta[i_star] <= epsilon:
            break
        centres.append(i_star)
        for i in range(n):
            d = _block_dist(L_list[i], L_list[i_star], p)
            if d < delta[i]:
                delta[i] = d

    clusters = [{"spd_ids": [], "log_sum": np.zeros_like(L_list[ci]), "count": 0, "radius": 0.0} for ci in centres]
    for i in range(n):
        best_k, best_d = None, float("inf")
        for k, ci in enumerate(centres):
            d = _block_dist(L_list[i], L_list[ci], p)
            if d < best_d:
                best_d, best_k = d, k
        cl = clusters[best_k]
        cl["spd_ids"].append(spd_ids[i])
        cl["log_sum"] += L_list[i]
        cl["count"] += 1
        cl["radius"] = max(float(cl["radius"]), float(best_d))
    for cl, ci in zip(clusters, centres):
        if cl["count"] > 0:
            cl["mean"] = L_list[ci].copy()
            cl["representative_mean"] = cl["log_sum"] / float(cl["count"])
        else:
            cl["mean"] = cl["representative_mean"] = np.zeros_like(L_list[ci])
    return clusters


def index_spds(data, epsilon_dict, epsilon_block_wise_dict) -> Dict[int, IndexHandle]:
    """One DAG per niche (block-wise epsilons; a missing block keeps the previous epsilon)."""
    dag_dict: Dict[int, IndexHandle] = {}
    for cluster_id in sorted(set(int(c) for c in data.labels)):
        rows = np.flatnonzero(data.labels == cluster_id)
        mats = [as_cov(data.spd_matrices[r]) for r in rows]
        ids = np.array(data.spd_ids)[rows]
        perm, block_runs = data.perm_list[cluster_id], data.block_dict[cluster_id]
        epsilon = epsilon_dict[cluster_id]
        layers = []
        for block_index, (start, end) in enumerate(block_runs):
            epsilon = epsilon_block_wise_dict.get(cluster_id, {}).get(block_index, epsilon)
            g = perm[start:end]
            L_list = [log_spd(C[np.ix_(g, g)]) for C in mats]
            layers.append(_epsilon_net_layer(L_list, ids, epsilon, len(perm)))
            logger.info("Niche %d block %d: %d nodes", cluster_id, block_index, len(layers[-1]))

        first_id = np.cumsum([0] + [len(layer) for layer in layers])
        nodes: List[BlockClusterNode] = []
        for b, layer in enumerate(layers):
            next_of_spd: Dict[int, List[int]] = {}
            if b + 1 < len(layers):
                for j, cl in enumerate(layers[b + 1]):
                    for s in cl["spd_ids"]:
                        next_of_spd.setdefault(s, []).append(int(first_id[b + 1] + j))
            for cl in layer:
                children = set()
                for s in cl["spd_ids"]:
                    children.update(next_of_spd.get(s, ()))
                meta = BlockClusterMetadata([(s, b) for s in cl["spd_ids"]], cl["mean"], cl["representative_mean"],
                                            cl["radius"])
                nodes.append(BlockClusterNode(len(nodes), b, meta, list(children)))
        block_to_node_indices = {b: list(range(first_id[b], first_id[b + 1])) for b in range(len(layers))}
        dag_dict[cluster_id] = IndexHandle(nodes, block_runs, perm, block_to_node_indices, list(range(len(layers))))
    return dag_dict


def build_lite(data, k_target_per_block: int = 64) -> LiteIndex:
    """Spindle-lite over ``data``'s layout and training covariances (``data.spd_matrices``)."""
    epsilon_block_wise_dict, epsilon_dict = {}, {}
    for cluster_id in sorted(set(int(c) for c in data.labels)):
        eps_per_block, _, eps = choose_adaptive_epsilons(data, cluster_id, k_target_per_block=k_target_per_block)
        epsilon_block_wise_dict[cluster_id] = eps_per_block
        epsilon_dict[cluster_id] = eps
    # A small homogeneous niche has a tiny elbow epsilon, which would starve its search budget.
    median_eps = float(np.median(list(epsilon_dict.values())))
    epsilon_dict = {k: max(v, median_eps) for k, v in epsilon_dict.items()}
    dag_dict = index_spds(data, epsilon_dict, epsilon_block_wise_dict)
    labels = np.asarray(data.labels).astype(int)
    return LiteIndex(dag_dict, epsilon_dict, epsilon_block_wise_dict,
                     {k: int((labels == k).sum()) for k in dag_dict},
                     {int(s): i for i, s in enumerate(data.spd_ids)})


# ---------------------------------------------------------------------------------------------- search
@dataclass
class SearchConfig:
    max_results: int | None = None        # stop after this many leaf paths
    max_failed_starts: int | None = None  # stop after this many first-layer nodes give no new leaf
    max_failed_paths: int | None = None   # stop after this many branches pruned by the budget
    total_paths_limit: int | None = 1000  # stop after this many leaves reached


@dataclass
class SearchPath:
    node_path: List[int]
    total_distance: float                 # sqrt(sum_b lb_b^2): lower bound on d_B to every path tile (prunes)
    path_score: Optional[float] = None    # sqrt(sum_b d_b^2) to the node centres
    _mask: Optional[int] = field(default=None, repr=False, compare=False)
    _decoder: Optional[object] = field(default=None, repr=False, compare=False)

    @property
    def member_ids(self) -> List[int]:
        """spd_ids shared by every node on the path, ascending."""
        if self._mask is None or self._decoder is None:
            return []
        return self._decoder.decode(self._mask)


@dataclass
class SearchResults:
    paths: List[SearchPath]


class _FastHandle:
    """Per-DAG search precomputation (cached, never pickled): member bitmasks, children by block,
    node centres, radii and sqrt(p) (p = all genes)."""

    def __init__(self, index_handle):
        nodes = index_handle.nodes
        self.n_nodes = len(nodes)
        ids = sorted({int(s) for n in nodes for s, _ in n.metadata.members})
        self.ids = np.asarray(ids, dtype=np.int64)
        pos = {s: j for j, s in enumerate(ids)}
        self.nbytes = (len(ids) + 7) // 8
        self.mask = []
        for n in nodes:
            bits = np.zeros(len(ids), dtype=bool)
            bits[[pos[int(s)] for s, _ in n.metadata.members]] = True
            self.mask.append(int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little"))
        self.global_id = [n.global_node_id for n in nodes]
        self.block = [n.block_index for n in nodes]
        self.mean = [n.metadata.mean for n in nodes]
        self.radius = [n.metadata.radius for n in nodes]
        self.sqrt_p = np.sqrt(len(index_handle.perm))
        self.children = []
        for n in nodes:
            by_block: Dict[int, List[int]] = {}
            for c in n.children:
                if 0 <= c < self.n_nodes:
                    by_block.setdefault(nodes[c].block_index, []).append(c)
            self.children.append(by_block)

    def decode(self, m: int) -> List[int]:
        bits = np.unpackbits(np.frombuffer(m.to_bytes(self.nbytes, "little"), dtype=np.uint8), bitorder="little")
        return self.ids[np.flatnonzero(bits[:len(self.ids)])].tolist()

    def distances(self, query_blocks_log):
        """Lazy per-query node -> (d_b to the centre, max(0, d_b - radius))."""
        raw: List[Optional[float]] = [None] * self.n_nodes
        lb: List[Optional[float]] = [None] * self.n_nodes
        mean, radius, sqrt_p, block = self.mean, self.radius, self.sqrt_p, self.block

        def dist(i):
            if raw[i] is None:
                d = np.linalg.norm(query_blocks_log[block[i]] - mean[i], ord="fro") / sqrt_p
                raw[i] = d
                lb[i] = max(0.0, d - radius[i])
            return raw[i], lb[i]
        return dist


_FAST_CACHE: Dict[int, Tuple[object, _FastHandle]] = {}


def _fast_handle(index_handle) -> _FastHandle:
    key = id(index_handle)
    hit = _FAST_CACHE.get(key)
    if hit is not None:
        ref, fh = hit
        if (ref() if callable(ref) else ref) is index_handle and fh.n_nodes == len(index_handle.nodes):
            return fh
    fh = _FastHandle(index_handle)
    try:
        ref = weakref.ref(index_handle, lambda _r, k=key: _FAST_CACHE.pop(k, None))
    except TypeError:
        ref = index_handle
    _FAST_CACHE[key] = (ref, fh)
    return fh


def search_index(index_handle: IndexHandle, query_spd: np.ndarray, budget: float,
                 config: SearchConfig | None = None) -> SearchResults:
    """Budgeted DFS over one niche's DAG; ``query_spd`` is the whole query in the niche's gene order.
    Paths are returned by increasing ``total_distance``."""
    config = config or SearchConfig()
    first_layer_nodes = index_handle.block_to_node_indices.get(0, []) if index_handle.sorted_blocks else []
    if not first_layer_nodes:
        return SearchResults(paths=[])
    query_blocks_log = [log_spd(query_spd[s:e, s:e]) for s, e in index_handle.block_runs]
    budget_sq = budget * budget

    fh = _fast_handle(index_handle)
    dist = fh.distances(query_blocks_log)
    children, mask, gid = fh.children, fh.mask, fh.global_id
    last = len(index_handle.sorted_blocks) - 1
    max_results, max_failed_paths, total_paths_limit = config.max_results, config.max_failed_paths, config.total_paths_limit

    best_paths: Dict[Tuple[int, ...], SearchPath] = {}
    done = False
    failed_paths = 0
    total_paths_explored = 0

    def dfs(layer_idx, node_idx, total_sq, raw_sq, path_indices, valid):
        nonlocal done, failed_paths, total_paths_explored
        if done:
            return
        if layer_idx == last:
            total_paths_explored += 1
            if total_paths_limit is not None and total_paths_explored >= total_paths_limit:
                done = True
                return
            path_key = tuple(gid[i] for i in path_indices)
            if path_key in best_paths:
                return
            best_paths[path_key] = SearchPath(node_path=list(path_key), total_distance=float(np.sqrt(total_sq)),
                                              path_score=float(np.sqrt(raw_sq)), _mask=valid, _decoder=fh)
            if max_results is not None and len(best_paths) >= max_results:
                done = True
            return

        child_dists = []
        for c in children[node_idx].get(layer_idx + 1, ()):
            m = valid & mask[c]
            if not m:
                continue
            r, d = dist(c)
            child_dists.append((c, d, r, m))
        child_dists.sort(key=lambda x: x[1])

        for c, d, r, m in child_dists:
            new_sq = total_sq + d * d
            if new_sq > budget_sq:
                failed_paths += 1
                if max_failed_paths is not None and failed_paths >= max_failed_paths:
                    done = True
                    return
                continue
            dfs(layer_idx + 1, c, new_sq, raw_sq + r * r, path_indices + [c], m)
            if done:
                return

    start_candidates = sorted(((i, *dist(i)[::-1]) for i in first_layer_nodes), key=lambda x: x[1])
    failed_starts = 0
    for node_idx, d, r in start_candidates:
        if d > budget:
            break
        before = len(best_paths)
        valid = mask[node_idx]
        if valid:
            dfs(0, node_idx, d * d, r * r, [node_idx], valid)
        if len(best_paths) == before:
            failed_starts += 1
            if config.max_failed_starts is not None and failed_starts >= config.max_failed_starts:
                break
        if done:
            break

    paths_sorted = sorted(best_paths.values(), key=lambda p: p.total_distance)
    if max_results is not None:
        paths_sorted = paths_sorted[:max_results]
    return SearchResults(paths=paths_sorted)


def _niche_scale_factor(n_niche: int, typical: int = 500, min_factor: float = 1.0, max_factor: float = 6.0) -> float:
    return float(np.clip(np.sqrt(max(n_niche, 1) / typical), min_factor, max_factor))


def search_lite(lite: LiteIndex, query_cov, budget_multiplier: float = 1.0, effort_multiplier: float | None = 1.0):
    """Search every niche; returns (training rows in the order found, seconds).

    Effort caps per niche: 100 failed starts, 200 failed branches, 3000 leaves, each times f * effort_multiplier;
    ``effort_multiplier=None`` lifts the caps (only the budget prunes).
    """
    q = as_cov(query_cov)
    found, seen = [], set()
    seconds = 0.0
    for c in sorted(lite.dag_dict):
        handle = lite.dag_dict[c]
        f = _niche_scale_factor(lite.niche_sizes[c])
        if effort_multiplier is None:
            config = SearchConfig(total_paths_limit=None)
        else:
            fe = f * effort_multiplier
            config = SearchConfig(max_failed_starts=max(1, round(100 * fe)), max_failed_paths=max(1, round(200 * fe)),
                                  total_paths_limit=max(1, round(3000 * fe)))
        budget = float(lite.epsilon_dict[c]) * float(budget_multiplier) * f
        t0 = time.perf_counter()
        results = search_index(handle, q[np.ix_(handle.perm, handle.perm)], budget, config)
        seconds += time.perf_counter() - t0
        for path in results.paths:
            for spd_id in path.member_ids:
                if spd_id not in seen:
                    seen.add(spd_id)
                    found.append(lite.row_of_spd_id[spd_id])
    return np.asarray(found, dtype=np.int64), seconds


# ---------------------------------------------------------------------------------------------- io
def save_index(obj, path) -> None:
    """Pickle any index or layout."""
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)
