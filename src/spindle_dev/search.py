"""Search traversal over the SPD block-DAG index.

Implements best-first / budget-pruned traversal over the block-cluster
DAG produced by :mod:`index`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Dict, List, Tuple, Optional
import weakref

import numpy as np

from .typing import IndexHandle, BlockClusterNode
from .metrics import (
    log_euclidean_distance,
    log_spd,
    build_ultrametrics,
    spd_tree_feature_matrix,
    fit_cca_alignment,
    project_with_cca,
    fit_supervised_cca_alignment,
    assign_with_supervised_cca,
)
from .utils import DeterministicConfig, configure_determinism, get_logger


logger = get_logger(__name__)


@dataclass
class SearchConfig:
    """Configuration for search traversal.

    Attributes
    ----------
    max_results:
        Optional upper bound on number of SPD IDs to return.
    max_failed_starts:
        Optional upper bound on how many starting candidates in the
        first block may be explored without yielding any leaf paths
        before aborting the search early. This helps cap work for
        queries that have no good matches.
    max_failed_paths:
        Optional upper bound on how many individual DFS branches may
        be pruned due to budget exhaustion (no leaf reached) before we
        stop the search entirely. This further limits work when the
        DAG is large but the query has no feasible paths.
    deterministic:
        Controls global deterministic settings.
    debug:
        If True, emit detailed debug logs about traversal and budget
        usage.
    """
    max_results: int | None = None
    max_failed_starts: int | None = None
    max_failed_paths: int | None = None
    total_paths_limit: int = 1000
    deterministic: DeterministicConfig = field(default_factory=DeterministicConfig)
    debug: bool = False


@dataclass
class SearchPath:
    """Represents a single path (leaf) through the block-DAG.

    ``node_path`` is the path of global node IDs and ``total_distance`` its
    accumulated radius-adjusted lower bound (the quantity the search orders and
    prunes by). ``path_score`` is the accumulated raw distance to the node means.
    ``member_ids`` lists the SPD IDs shared by every node on the path (decoded on
    first access).
    """

    node_path: List[int]
    total_distance: float
    path_score: Optional[float] = None
    _mask: Optional[int] = field(default=None, repr=False, compare=False)
    _decoder: Optional[object] = field(default=None, repr=False, compare=False)

    @property
    def member_ids(self) -> List[int]:
        if self._mask is None or self._decoder is None:
            return []
        return self._decoder.decode(self._mask)


@dataclass
class SearchResults:
    """Results returned by :func:`query_index`.

    paths:
        Detailed per-hit paths including the sequence of node IDs and
        accumulated log-Euclidean distance.
    """
    paths: List[SearchPath]


def query_index(index_handle: IndexHandle, query_spd: np.ndarray, budget: float, config: SearchConfig | None = None) -> SearchResults:
    """Query an index with a SPD sub-matrix and distance budget.

    The algorithm matches the spec from .github/copilot-instructions.md:

    - Binary-search over the sorted block-cluster means for the first
      block to find nearest candidates.
    - Recursively traverse children for subsequent blocks, decreasing
      the remaining budget by the distance to each chosen block-mean.
    - If remaining budget < 0, backtrack. Implemented as a best-first
      search with budget pruning.

    This implementation currently assumes that nodes in ``index_handle``
    are already organized by block in ``block_to_nodes`` and that each
    node's ``metadata.members`` contains (spd_id, block_id) pairs.

    For now, if the index is empty, an empty ``SearchResults`` is
    returned. A more complete implementation should fill in the search
    traversal once full index construction is available.
    """

    if config is None:
        config = SearchConfig()

    configure_determinism(config.deterministic)

    if not index_handle.nodes:
        logger.info("Index is empty; returning no search results.")
        return SearchResults(paths=[])

    # Placeholder best-first traversal skeleton.
    # Full implementation should:
    #   * identify the first block index present in block_to_nodes
    #   * compute distances from query block to each block-cluster mean
    #   * order candidates by distance (binary-search over sorted list)
    #   * traverse children while tracking remaining budget.

    logger.warning(
        "query_index called with a non-empty index, but traversal is not yet fully implemented."
    )
    return SearchResults(paths=[])


def _align_query(index_handle, query_spd, query_indices, query_block_runs):
    """Choose the index layers a query covers and slice its blocks.

    Returns ``(used_blocks, query_blocks)`` or ``None`` when the query cannot be searched
    (the caller then returns an empty result).
    """
    nodes = index_handle.nodes
    block_to_nodes = index_handle.block_to_nodes
    if not nodes or not block_to_nodes:
        logger.info("Index is empty; returning no search results.")
        return None

    sorted_blocks: List[int] = index_handle.sorted_blocks  # type: ignore[assignment]
    if not sorted_blocks:
        logger.info("Index has no block layers; returning no search results.")
        return None

    # True alignment: choose index layers that overlap query runs
    index_block_runs = index_handle.block_runs  # type: ignore[attr-defined]

    if len(query_indices) == 0 and len(query_block_runs) == 0:
        logger.info("No query indices or block runs provided; returning no search results.")
        return None

    if len(query_block_runs) == 0:
        logger.info("Using query_indices to align blocks.")
        logger.warning("Not implemented: returning no search results.")
        return None
    elif len(query_indices) == 0:
        # first check if they are same
        if len(query_block_runs) == len(index_block_runs):
            used_blocks = [b for b in sorted_blocks]
        elif len(query_block_runs) > len(index_block_runs):
            logger.warning("Query block runs longer than index block runs; using index block runs to align.")
            logger.info("Not implemented: returning no search results.")
            return None
        else:
            logger.info("Using query_block_runs to align blocks.")
            used_blocks: List[int] = determine_active_blocks(query_block_runs, index_block_runs)
    else:
        logger.info("Both query_indices and query_block_runs provided; using indices to align blocks.")
        used_blocks = find_matching_blocks(query_indices, index_block_runs)
    used_blocks = [b for b in used_blocks if b in sorted_blocks]
    if any(used_blocks[i] + 1 != used_blocks[i + 1] for i in range(len(used_blocks) - 1)):
        logger.warning("Non-contiguous blocks detected in used_blocks; search may be suboptimal.")
        logger.info("Not implemented: returning no search results.")
        return None
    if not used_blocks:
        logger.info("No overlapping blocks between query and index; returning no search results.")
        return None

    num_layers = len(used_blocks)
    # Pre-extract aligned query blocks corresponding to each used index layer.
    is_spd_matrix_full = query_spd.shape[0] == index_block_runs[sorted_blocks[0]][1]
    query_blocks: List[np.ndarray] = []
    offset_start = 0
    for layer_idx in range(num_layers):
        start, end = query_block_runs[layer_idx]
        if is_spd_matrix_full:
            query_blocks.append(query_spd[start:end, start:end])
        else:
            offset_size = end - start
            query_blocks.append(query_spd[offset_start:offset_start + offset_size, offset_start:offset_start + offset_size])
            offset_start += offset_size
    if not query_blocks:
        logger.info("No query blocks align to index layers; returning no search results.")
        return None
    return used_blocks, query_blocks


class _FastHandle:
    """Per-``IndexHandle`` search precomputation, built once and cached (never pickled).

    - every node's member spd_ids as a Python-int bitmask (bit j = ``ids[j]``), so testing
      whether a child still shares members with the path is a single AND;
    - every node's children grouped by block, in stored order;
    - node means, radii and sqrt(p), and the global node ids.
    """

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
        self.sqrt_p = [np.sqrt(n.metadata.mean.shape[0]) for n in nodes]
        self.children = []
        for n in nodes:
            by_block: Dict[int, List[int]] = {}
            for c in n.children:
                # By construction global_node_id equals the index in ``nodes``; guard anyway.
                if 0 <= c < self.n_nodes:
                    by_block.setdefault(nodes[c].block_index, []).append(c)
            self.children.append(by_block)

    def decode(self, m: int) -> List[int]:
        """spd_ids in a member bitmask."""
        bits = np.unpackbits(np.frombuffer(m.to_bytes(self.nbytes, "little"), dtype=np.uint8), bitorder="little")
        return self.ids[np.flatnonzero(bits[:len(self.ids)])].tolist()

    def distances(self, query_blocks_log, layer_of_block):
        """Lazy per-query cache node -> (raw distance to mean, radius-adjusted lower bound).

        Same arithmetic as the original traversal, so the values are bit-identical.
        """
        raw: List[Optional[float]] = [None] * self.n_nodes
        lb: List[Optional[float]] = [None] * self.n_nodes
        mean, radius, sqrt_p, block = self.mean, self.radius, self.sqrt_p, self.block

        def dist(i):
            if raw[i] is None:
                diff = query_blocks_log[layer_of_block[block[i]]] - mean[i]
                d = np.linalg.norm(diff, ord='fro') / sqrt_p[i]
                raw[i] = d
                # Triangle-inequality lower bound (distance to the cluster's mean minus its
                # radius): the query's true nearest member can be up to `radius` closer than
                # the mean, so pruning on the raw distance-to-mean would wrongly discard
                # branches that contain it.
                lb[i] = max(0.0, d - radius[i])
            return raw[i], lb[i]
        return dist


_FAST_CACHE: Dict[int, Tuple[object, _FastHandle]] = {}


def _fast_handle(index_handle) -> _FastHandle:
    """The cached ``_FastHandle`` of ``index_handle`` (keyed by object identity)."""
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


def search_index(
    index_handle: IndexHandle,
    query_spd: np.ndarray,
    query_indices: List[int],
    query_block_runs: List[Tuple[int, int]],
    budget: float,
    config: SearchConfig | None = None,
) -> SearchResults:
    """Query an index with a multi-block SPD sub-matrix and distance budget.

    The traversal proceeds layer by layer through the block-DAG:

    1. In the first layer (block), compute the distance from the query
       block to each block-cluster mean and sort clusters by distance.
       Starting from the closest cluster, spend from the distance
       budget and recurse into the DAG.
    2. At each subsequent layer, follow the node's children, again
       ordered by distance between the next query block and each child
       block-cluster mean. The remaining budget is decreased by this
       distance.
    3. If the remaining budget would go negative, the path is pruned
       and recursion backtracks to explore alternative branches.
    4. When a path reaches the final block within budget, all SPD IDs
       in the leaf node's metadata are recorded as hits, together with
       the path of global node IDs and the total accumulated distance.

    Distances used for ordering and pruning are radius-adjusted lower bounds
    (distance to the node mean minus the node radius). Each returned path also
    carries ``path_score`` -- the raw sum of distances to the node means -- and
    ``member_ids``, the SPD IDs shared by every node on the path.

    Implementation: node member sets are held as bitmasks and node distances are
    computed at most once per query (see ``_FastHandle``); the visiting order,
    pruning, limits and returned paths are those of the original set-based
    traversal. The function returns ``SearchResults`` for a searchable query and
    a singleton list ``[SearchResults(paths=[])]`` on its early exits.
    """

    if config is None:
        config = SearchConfig()

    configure_determinism(config.deterministic)

    debug = config.debug

    aligned = _align_query(index_handle, query_spd, query_indices, query_block_runs)
    if aligned is None:
        return [SearchResults(paths=[])]
    used_blocks, query_blocks = aligned
    num_layers = len(used_blocks)

    if debug:
        logger.info("Starting search: budget=%.4f, num_layers=%d, blocks=%s", budget, num_layers, used_blocks)

    query_blocks_log = [log_spd(qb) for qb in query_blocks]

    first_layer_nodes = index_handle.block_to_node_indices.get(used_blocks[0], [])
    if not first_layer_nodes:
        logger.info("No nodes in first block; returning no search results.")
        return [SearchResults(paths=[])]

    fh = _fast_handle(index_handle)
    layer_of_block = {b: l for l, b in enumerate(used_blocks)}
    dist = fh.distances(query_blocks_log, layer_of_block)
    children, mask, gid = fh.children, fh.mask, fh.global_id
    next_block = used_blocks[1:] + [None]
    last = num_layers - 1
    max_results = config.max_results
    max_failed_paths = config.max_failed_paths
    total_paths_limit = config.total_paths_limit

    # Best path per leaf (keyed by the path of global node ids).
    best_paths: Dict[Tuple[int, ...], SearchPath] = {}
    done = False
    failed_paths = 0
    total_paths_explored = 0

    # Depth-first recursive traversal with budget-based backtracking.
    def dfs(layer_idx, node_idx, remaining_budget, total_dist, raw_total, path_indices, valid):
        nonlocal done, failed_paths, total_paths_explored
        if done:
            return

        # If we've reached the last layer, record a single path for this leaf.
        if layer_idx == last:
            total_paths_explored += 1
            if total_paths_explored >= total_paths_limit:
                if len(best_paths) == 0:
                    logger.info(
                        "Stopping search early after reaching total paths limit of %d. Increase budget or total_paths_limit to get any results.",
                        total_paths_limit,
                    )
                done = True
                return
            path_key = tuple(gid[i] for i in path_indices)
            # If we've already recorded this exact path, we don't need to refine it.
            if path_key in best_paths:
                return
            best_paths[path_key] = SearchPath(node_path=list(path_key), total_distance=total_dist,
                                              path_score=raw_total, _mask=valid, _decoder=fh)
            # If we've reached the requested number of leaf paths, signal completion.
            if max_results is not None and len(best_paths) >= max_results:
                done = True
            return

        # Children in the next block that still share members with the path, closest first.
        child_dists = []
        for c in children[node_idx].get(next_block[layer_idx], ()):
            m = valid & mask[c]
            if not m:
                continue
            r, d = dist(c)
            child_dists.append((c, d, r, m))
        child_dists.sort(key=lambda x: x[1])

        for c, d, r, m in child_dists:
            new_total = total_dist + d
            new_remaining = remaining_budget - d
            if new_remaining < 0 or new_total > budget:
                # Count this as a failed DFS branch (no leaf reached because budget is exhausted).
                failed_paths += 1
                if max_failed_paths is not None and failed_paths >= max_failed_paths:
                    if debug:
                        logger.info("Stopping search early after %d failed DFS branches.", failed_paths)
                    done = True
                    return
                continue
            dfs(layer_idx + 1, c, new_remaining, new_total, raw_total + r, path_indices + [c], m)
            if done:
                return

    # --- Seed the DFS from the first layer ---
    start_candidates = []
    for node_idx in first_layer_nodes:
        r, d = dist(node_idx)
        start_candidates.append((node_idx, d, r))
    # Explore starting nodes in order of increasing distance.
    start_candidates.sort(key=lambda x: x[1])

    failed_starts = 0
    for node_idx, d, r in start_candidates:
        if d > budget:
            break
        before = len(best_paths)
        valid = mask[node_idx]
        if valid:
            dfs(0, node_idx, budget - d, d, r, [node_idx], valid)
        # If this start contributed no new leaf paths, count it as a failed attempt. For
        # hard / false-positive queries this provides a hard cap on search effort.
        if len(best_paths) == before:
            failed_starts += 1
            if config.max_failed_starts is not None and failed_starts >= config.max_failed_starts:
                if debug:
                    logger.info("Stopping search early after %d failed start candidates.", failed_starts)
                break
        if done:
            break

    # Assemble final SearchResults, ordered by increasing total_distance.
    paths_sorted = sorted(best_paths.values(), key=lambda p: p.total_distance)
    if debug:
        logger.info("Search complete: found %d hits within budget %.4f", len(paths_sorted), budget)
    if max_results is not None:
        paths_sorted = paths_sorted[:max_results]

    return SearchResults(paths=paths_sorted)


def search_top_c(
    queries: List[Tuple[IndexHandle, np.ndarray, List[Tuple[int, int]]]],
    c: int,
    budgets: Optional[List[float]] = None,
    return_query_logs: bool = False,
):
    """The ``c`` best SPD IDs across one or more indexes, by raw DAG path score.

    ``queries`` holds one ``(index_handle, query_spd, query_block_runs)`` per index to search
    (e.g. one per niche, each query already permuted into that niche's gene order). A path's
    score is the sum over its blocks of the distance from the query block's log to the node
    mean (no radius subtracted). One best-first search over every index pops partial paths in
    order of their accumulated score; scores never decrease along a path, so leaves come out in
    exact score order and the search stops as soon as ``c`` SPD IDs have been emitted.
    ``budgets`` (optional, one per query) drops partial paths whose radius-adjusted lower bound
    exceeds it, as ``search_index`` does. SPD IDs of one leaf share a score and are emitted in
    id order.

    Returns ``[(path_score, query_position, spd_id)]``, at most ``c`` entries, best first; with
    ``return_query_logs=True`` also the per-query list of query block logs the search used
    (``None`` for a query that could not be aligned), e.g. for an exact re-ranking.
    """
    heap = []
    counter = 0
    state = []
    all_logs = []
    for qpos, (index_handle, query_spd, query_block_runs) in enumerate(queries):
        aligned = _align_query(index_handle, query_spd, [], query_block_runs)
        if aligned is None:
            state.append(None)
            all_logs.append(None)
            continue
        used_blocks, query_blocks = aligned
        query_blocks_log = [log_spd(qb) for qb in query_blocks]
        all_logs.append(query_blocks_log)
        fh = _fast_handle(index_handle)
        dist = fh.distances(query_blocks_log, {b: l for l, b in enumerate(used_blocks)})
        state.append((fh, dist, used_blocks[1:] + [None], len(used_blocks) - 1))
        budget = None if budgets is None else budgets[qpos]
        for node_idx in index_handle.block_to_node_indices.get(used_blocks[0], []):
            if fh.mask[node_idx]:
                r, d = dist(node_idx)
                if budget is None or d <= budget:
                    heap.append((r, counter, qpos, 0, node_idx, d, fh.mask[node_idx]))
                    counter += 1
    heapq.heapify(heap)

    out: List[Tuple[float, int, int]] = []
    seen = set()
    while heap and len(out) < c:
        score, _, qpos, layer_idx, node_idx, lb_total, m = heapq.heappop(heap)
        fh, dist, next_block, last = state[qpos]
        if layer_idx == last:
            for sid in fh.decode(m):
                if (qpos, sid) not in seen and len(out) < c:
                    seen.add((qpos, sid))
                    out.append((score, qpos, sid))
            continue
        budget = None if budgets is None else budgets[qpos]
        mask = fh.mask
        for ch in fh.children[node_idx].get(next_block[layer_idx], ()):
            mc = m & mask[ch]
            if mc:
                r, d = dist(ch)
                if budget is None or lb_total + d <= budget:
                    heapq.heappush(heap, (score + r, counter, qpos, layer_idx + 1, ch, lb_total + d, mc))
                    counter += 1
    return (out, all_logs) if return_query_logs else out


from bisect import bisect_right

def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    lo = max(a0, b0)
    hi = min(a1, b1)
    return max(0, hi - lo)


def determine_active_blocks(
    query_block_runs: List[Tuple[int, int]],
    block_runs: List[Tuple[int, int]],
) -> List[int]:
    """Return index block indices that overlap the query, in index order.

    Keeps a simple, deterministic behavior:
    - Iterate index `block_runs` in their natural order.
    - Select blocks that have any positive overlap with any query run.
    - Stop once as many blocks are selected as there are query runs.
    """
    # dictionary of block index to (start, end) runs
    # in order of block index
    # sort the  query runs by start
    sorted_query_runs = sorted(query_block_runs, key=lambda x: x[0])
    block_ids = list(range(len(block_runs)))
    starts    = [r[0] for r in block_runs]
    ends      = [r[1] for r in block_runs]
    
    chosen = set()
    active: List[int] = []
    for q_start, q_end in sorted_query_runs:
        if q_end <= q_start:
            continue

        # Find rightmost block with start <= q_start
        i = bisect_right(starts, q_start) - 1
        if i < 0:
            i = 0

        best_j = None
        best_olap = 0

        # Scan forward while blocks might still overlap (start < q_end)
        j = i
        while j < len(starts) and starts[j] < q_end:
            olap = _overlap_len(q_start, q_end, starts[j], ends[j])
            if olap > best_olap:
                best_olap = olap
                best_j = j
            # stable tie-break: keep earliest j (do nothing on ==)
            j += 1

        # If nothing overlapped from i forward, try the immediate predecessor (rare boundary case)
        if best_j is None and i > 0:
            olap = _overlap_len(q_start, q_end, starts[i - 1], ends[i - 1])
            if olap > 0:
                best_j = i - 1
                best_olap = olap

        if best_j is not None:
            bid = block_ids[best_j]
            if bid not in chosen:
                chosen.add(bid)
                active.append(bid)

    active.sort()
    return active


from bisect import bisect_right
from typing import Iterable, List, Sequence, Tuple, Set

def _normalize_query_indices(query_indices: Iterable[int]) -> List[int]:
    # deterministic; remove duplicates
    return sorted(set(int(x) for x in query_indices))

def _map_index_to_block_partition(idx: int, starts: Sequence[int]) -> int:
    """
    For partitioned blocks with starts sorted ascending, return the block id
    containing idx. Assumes full coverage and non-overlap.
    """
    j = bisect_right(starts, idx) - 1
    return 0 if j < 0 else j

def find_matching_blocks(
    query_indices: Iterable[int],
    block_runs: Sequence[Tuple[int, int]]
) -> List[int]:
    """
    Given row indices and a non-overlapping partition of blocks (full coverage),
    return block ids that match.

    Under the partition assumption:
      - mode="cover_all" returns all blocks that contain any query index.
      - mode="greedy" is identical to cover_all (kept for API symmetry).
    """
    qs = _normalize_query_indices(query_indices)
    if not qs:
        return []

    starts = [s for (s, _) in block_runs]

    chosen: Set[int] = set()
    for idx in qs:
        bid = _map_index_to_block_partition(idx, starts)
        chosen.add(bid)

    # deterministic index order
    result = sorted(chosen)

    
    return result


def log_euclidean_distance_for_SPD(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> float:
    """Compute the log-Euclidean distance between two SPD matrices A and B."""
    def log_spd(M: np.ndarray) -> np.ndarray:
        M = 0.5 * (M + M.T)
        w, V = np.linalg.eigh(M)
        w = np.maximum(w, eps)  # clamp for numerical safety
        return (V * np.log(w)) @ V.T

    L_A = log_spd(A)
    L_B = log_spd(B)
    p = A.shape[0]
    diff = L_A - L_B
    dist = np.linalg.norm(diff, ord='fro') / np.sqrt(p)
    return dist


def search_brute_force(
    candidates: List[np.array],
    query_matrix: np.array
):
    # Do brute force search to find out the least distance 
    logger.info("Starting brute force")
    dists = []
    for cid in range(len(candidates)):
        d = log_euclidean_distance_for_SPD(query_matrix, candidates[cid])
        dists.append((cid, float(d)))
    dists.sort(key=lambda x: x[1])
    baseline_top_id, baseline_top_dist = dists[0]
    return baseline_top_id, baseline_top_dist


def search_lsh_index(
    lsh_index: Dict,
    query_spd: np.ndarray,
    
) -> List[Tuple[int, float]]:
    """Query an LSH index with a SPD matrix.

    The function computes the hash keys for the query SPD matrix
    using the stored random projections and retrieves candidate SPD IDs
    from the corresponding buckets.

    Returns a list of (SPD ID, estimated distance) tuples.
    """

    projections = lsh_index["projections"]
    buckets = lsh_index["buckets"]

    if not projections or not buckets:
        logger.info("LSH index is empty; returning no search results.")
        return []

    n_tables = len(projections)
    hash_keys = []
    for t in range(n_tables):
        proj = projections[t]
        flat_query = query_spd.flatten()
        hash_key = tuple(int((flat_query @ proj[:, i]) > 0) for i in range(proj.shape[1]))
        hash_keys.append(hash_key)

    candidate_ids = set()
    for t in range(n_tables):
        h = hash_keys[t]
        bucket = buckets.get(t, {}).get(h, [])
        candidate_ids.update(bucket)

    logger.info(f"Number of candidates retrieved: {len(candidate_ids)}")

    # For simplicity, we return candidate IDs with a placeholder distance of 0.0
    results = [(cid, 0.0) for cid in candidate_ids]
    return results


def lsh_query(
    index: Dict[str, object],
    query_spd: np.ndarray,
    *,
    min_collisions: Optional[int] = None,
) -> np.ndarray:
    """Return candidate SPD IDs that collide in >= min_collisions tables."""
    projections = index["projections"]
    buckets = index["buckets"]
    use_log_eigenvalues = index["use_log_eigenvalues"]
    eig_floor = index["eig_floor"]
    eig_cap = index["eig_cap"]
    normalize_repr = index["normalize_repr"]

    if min_collisions is None:
        min_collisions = index["min_collisions"]

    # Build query representation
    w = np.linalg.eigvalsh(query_spd)

    w = np.maximum(w, eig_floor)
    if eig_cap is not None:
        w = np.minimum(w, eig_cap)

    if use_log_eigenvalues:
        w = np.log(w)

    if normalize_repr:
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm

    def _bits_to_int(bits: np.ndarray) -> int:
        key = 0
        for b in range(bits.shape[0]):
            if bits[b]:
                key |= (1 << b)
        return int(key)

    # Count how many tables each candidate collides in
    counts: Dict[int, int] = {}

    n_tables = projections.shape[0]
    for t in range(n_tables):
        proj = projections[t] @ w
        h = _bits_to_int(proj >= 0.0)
        ids = buckets[t].get(h)
        if not ids:
            continue
        for spd_id in ids:
            counts[spd_id] = counts.get(spd_id, 0) + 1

    # Keep only those with enough collisions
    cand = [spd_id for spd_id, c in counts.items() if c >= int(min_collisions)]
    return np.asarray(cand, dtype=int)


import numpy as np
from sklearn.neighbors import NearestNeighbors

def assign_clusters_to_new_spds(
    new_spds,
    indexed_data,
    strategy="knn_majority",
    n_neighbors=20,
):
    """
    new_spds    : list/array of SPD matrices, each (p, p)
    indexed_data: ProcessedData used to build the index (e.g. data_subset)
                  must have .pca_model and .latent['pca'] and .labels
    strategy    : 'knn_majority', 'knn_weighted', 'centroid', or 'consensus'
    n_neighbors : number of neighbors for KNN-based strategies
    """
    if indexed_data.pca_model is None:
        raise ValueError("indexed_data.pca_model is None; run reduce_dim() before indexing.")

    # 1) Embed new SPDs into the same PCA space as the index.
    U_list, _ = build_ultrametrics(new_spds)
    feats_new = spd_tree_feature_matrix(U_list)
    Z_new = indexed_data.pca_model.transform(feats_new)          # (n_new, n_pca)

    Z_index = np.asarray(indexed_data.latent["pca"])             # (n_index, n_pca)
    labels_index = np.asarray(indexed_data.labels)

    # Precompute KNN structure once.
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    knn.fit(Z_index)
    knn_dist, knn_idx = knn.kneighbors(Z_new)                    # (n_new, k)

    # Strategy 1: simple majority vote among nearest neighbors.
    def _knn_majority():
        out = []
        for neigh_idx in knn_idx:
            neigh_labels = labels_index[neigh_idx]
            uniq, counts = np.unique(neigh_labels, return_counts=True)
            out.append(int(uniq[np.argmax(counts)]))
        return np.array(out, dtype=int)

    # Strategy 2: distance-weighted vote among nearest neighbors.
    def _knn_weighted():
        out = []
        eps = 1e-8
        for dists, neigh_idx in zip(knn_dist, knn_idx):
            neigh_labels = labels_index[neigh_idx]
            weights = 1.0 / (dists + eps)
            # aggregate weights per label
            label_scores = {}
            for lab, w in zip(neigh_labels, weights):
                label_scores[lab] = label_scores.get(lab, 0.0) + float(w)
            best_lab = max(label_scores.items(), key=lambda x: x[1])[0]
            out.append(int(best_lab))
        return np.array(out, dtype=int)

    # Strategy 3: nearest cluster centroid in PCA space.
    def _centroid():
        uniq = np.unique(labels_index)
        centroids = []
        for c in uniq:
            centroids.append(Z_index[labels_index == c].mean(axis=0))
        centroids = np.vstack(centroids)                         # (n_clusters, n_pca)
        # squared distances to centroids
        d2 = ((Z_new[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        return uniq[np.argmin(d2, axis=1)].astype(int)

    # Strategy 4: simple consensus between KNN-majority and centroid.
    # If they disagree, fall back to weighted KNN.
    def _consensus():
        maj = _knn_majority()
        cen = _centroid()
        agree = maj == cen
        if agree.all():
            return maj
        w_knn = _knn_weighted()
        out = np.where(agree, maj, w_knn)
        return out.astype(int)

    if strategy == "knn_majority":
        return _knn_majority()
    elif strategy == "knn_weighted":
        return _knn_weighted()
    elif strategy == "centroid":
        return _centroid()
    elif strategy == "consensus":
        return _consensus()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def assign_clusters_to_new_spds_cca(
    new_spds,
    indexed_data,
    *,
    n_components: int = 10,
    n_neighbors: int = 20,
):
    """Assign clusters to unseen SPDs using CCA-based alignment.

    This function fits a CCA model between an alternative feature
    representation of the *indexed* SPDs (the "source" view) and the
    PCA latent space used for indexing (the "target" view), then
    projects new SPDs into the same canonical space and assigns
    clusters via KNN majority vote.

    Parameters
    ----------
    new_spds:
        List/array of unseen SPD matrices, each of shape (p, p).
    indexed_data:
        The :class:`ProcessedData` instance used to build the index
        (e.g. ``data_subset``). Must expose ``latent['pca']`` and
        ``labels``, and ideally ``U_list`` from ``reduce_dim``; if
        ``U_list`` is missing it will be recomputed from
        ``indexed_data.spd_matrices``.
    n_components:
        Number of canonical components for CCA (capped internally by
        dimensionality and sample size).
    n_neighbors:
        Number of neighbors for the KNN majority vote in canonical
        space.

    Returns
    -------
    assigned_clusters:
        1D numpy array of integer cluster IDs, one per SPD in
        ``new_spds``.
    """

    if "pca" not in getattr(indexed_data, "latent", {}):
        raise ValueError("indexed_data.latent['pca'] not found; run reduce_dim() before indexing.")
    if getattr(indexed_data, "labels", None) is None:
        raise ValueError("indexed_data.labels is None; run cluster_spds() before alignment.")

    # --- Build source/target features for the indexed SPDs ---
    # Prefer the cached U_list from reduce_dim; fall back to recomputing.
    U_list_index = getattr(indexed_data, "U_list", None)
    if U_list_index is None:
        U_list_index, _ = build_ultrametrics(indexed_data.spd_matrices)

    X_source = spd_tree_feature_matrix(U_list_index)           # (n_index, m)
    Y_target = np.asarray(indexed_data.latent["pca"])        # (n_index, d_pca)

    cca, Xc_index, Yc_index = fit_cca_alignment(
        X_source,
        Y_target,
        n_components=n_components,
    )

    # --- Project new SPDs into the same canonical space ---
    U_list_new, _ = build_ultrametrics(new_spds)
    X_source_new = spd_tree_feature_matrix(U_list_new)
    Xc_new = project_with_cca(cca, X_source_new)              # (n_new, n_comp)

    labels_index = np.asarray(indexed_data.labels)

    # KNN majority vote in canonical space.
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    knn.fit(Xc_index)
    _, knn_idx = knn.kneighbors(Xc_new)

    assigned = []
    for neigh_idx in knn_idx:
        neigh_labels = labels_index[neigh_idx]
        uniq, counts = np.unique(neigh_labels, return_counts=True)
        assigned.append(int(uniq[np.argmax(counts)]))

    return np.array(assigned, dtype=int)


def assign_clusters_supervised_cca(
    new_spds,
    indexed_data,
    *,
    target_mode: str = "onehot_centroid",
    n_components: int = 10,
    strategy: str = "hybrid",
    n_neighbors: int = 20,
):
    """Assign clusters to unseen SPDs using label-aware (supervised) CCA.

    This is an improved CCA alignment that explicitly uses cluster labels
    from the indexed data to learn discriminative projections. The target
    view for CCA is constructed from label information (one-hot encoding,
    class centroids, or both), making the canonical space naturally
    cluster-aware.

    Parameters
    ----------
    new_spds:
        List/array of unseen SPD matrices, each of shape (p, p).
    indexed_data:
        The :class:`ProcessedData` instance used to build the index.
        Must expose ``labels`` and either ``U_list`` or ``spd_matrices``.
    target_mode:
        How to construct the label-informed target view for CCA:

        - ``"onehot"``: one-hot encoding of labels.
        - ``"centroid"``: class centroids in feature space.
        - ``"onehot_centroid"`` (default): concatenate both.
        - ``"onehot_aux"``: one-hot + PCA latents (requires
          ``indexed_data.latent['pca']``).
    n_components:
        Number of canonical components.
    strategy:
        Assignment strategy for new samples:

        - ``"centroid"``: nearest class centroid in canonical space.
        - ``"knn_majority"``: KNN majority vote.
        - ``"knn_weighted"``: distance-weighted KNN.
        - ``"hybrid"`` (default): centroid if confident, else KNN.
    n_neighbors:
        Number of neighbors for KNN-based strategies.

    Returns
    -------
    assigned_clusters:
        1D numpy array of assigned cluster IDs.
    distances:
        1D numpy array of distances to the assigned class (centroid
        distance or average neighbor distance).
    """

    if getattr(indexed_data, "labels", None) is None:
        raise ValueError("indexed_data.labels is None; run cluster_spds() first.")

    labels_index = np.asarray(indexed_data.labels)

    # Build source features for indexed SPDs.
    U_list_index = getattr(indexed_data, "U_list", None)
    if U_list_index is None:
        U_list_index, _ = build_ultrametrics(indexed_data.spd_matrices)

    X_source = spd_tree_feature_matrix(U_list_index)

    # Optional auxiliary view (PCA latents).
    Y_aux = None
    if target_mode == "onehot_aux":
        if "pca" not in getattr(indexed_data, "latent", {}):
            raise ValueError(
                "target_mode='onehot_aux' requires indexed_data.latent['pca']."
            )
        Y_aux = np.asarray(indexed_data.latent["pca"])

    # Fit supervised CCA.
    cca, Xc_train, class_centroids, unique_labels = fit_supervised_cca_alignment(
        X_source,
        labels_index,
        Y_aux=Y_aux,
        target_mode=target_mode,
        n_components=n_components,
    )

    # Build source features for new SPDs.
    U_list_new, _ = build_ultrametrics(new_spds)
    X_source_new = spd_tree_feature_matrix(U_list_new)

    # Assign using the specified strategy.
    assigned, distances = assign_with_supervised_cca(
        cca,
        class_centroids,
        unique_labels,
        X_source_new,
        Xc_train=Xc_train,
        labels_train=labels_index,
        strategy=strategy,
        n_neighbors=n_neighbors,
    )

    return assigned, distances
