"""Search traversal over the SPD block-DAG index.

Implements best-first / budget-pruned traversal over the block-cluster
DAG produced by :mod:`index`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

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

    This stores only the path of node IDs and the total distance; the
    SPD members for a leaf can always be recovered from the
    corresponding leaf node's metadata.
    """

    node_path: List[int]
    total_distance: float


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

    The function currently processes a *single* SPD (``query_spd`` with
    its ``query_block_runs``) and returns a singleton list
    ``[SearchResults]`` for compatibility with potential batching.
    """

    if config is None:
        config = SearchConfig()

    configure_determinism(config.deterministic)

    debug = config.debug

    nodes: List[BlockClusterNode] = index_handle.nodes
    block_to_nodes = index_handle.block_to_nodes

    if not nodes or not block_to_nodes:
        logger.info("Index is empty; returning no search results.")
        return [SearchResults(hits=[], paths=[])]

    # # Prepare and cache index views once (id_to_idx, block_to_node_indices, sorted_blocks)
    # if len(index_handle.id_to_idx) == 0:
    #     index_handle.id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}  # type: ignore[attr-defined]
    # if len(index_handle.block_to_node_indices) == 0:
    #     index_handle.block_to_node_indices = {  # type: ignore[attr-defined]
    #         b_idx: [index_handle.id_to_idx[nid] for nid in node_ids if nid in index_handle.id_to_idx]  # type: ignore[index]
    #         for b_idx, node_ids in block_to_nodes.items()
    #     }
    # if len(index_handle.sorted_blocks) == 0:
    #     index_handle.sorted_blocks = sorted(index_handle.block_to_node_indices.keys())  # type: ignore[attr-defined]

    block_to_node_indices: Dict[int, List[int]] = index_handle.block_to_node_indices  # type: ignore[assignment]
    sorted_blocks: List[int] = index_handle.sorted_blocks  # type: ignore[assignment]

    if not sorted_blocks:
        logger.info("Index has no block layers; returning no search results.")
        return [SearchResults(hits=[], paths=[])]

    # True alignment: choose index layers that overlap query runs
    # Normalize index block runs from index_handle
    index_block_runs = index_handle.block_runs  # type: ignore[attr-defined]


    if len(query_indices) == 0 and len(query_block_runs) == 0:
        logger.info("No query indices or block runs provided; returning no search results.")
        return [SearchResults(hits=[], paths=[])]
    
    if len(query_block_runs) == 0:
        logger.info("Using query_indices to align blocks.")
        logger.warning("Not implemented: returning no search results.")
        return [SearchResults(hits=[], paths=[])]
        # used_blocks = find_matching_blocks(query_indices, index_block_runs)
    elif len(query_indices) == 0:
        # first check if they are same 
        if len(query_block_runs) == len(index_block_runs):
            used_blocks = [b for b in sorted_blocks]
        elif len(query_block_runs) > len(index_block_runs):
            logger.warning("Query block runs longer than index block runs; using index block runs to align.")
            logger.info("Not implemented: returning no search results.")
            return [SearchResults(hits=[], paths=[])]
        else:
            logger.info("Using query_block_runs to align blocks.")
            used_blocks: List[int] = determine_active_blocks(query_block_runs, index_block_runs)
    else:
        logger.info("Both query_indices and query_block_runs provided; using indices to align blocks.")
        used_blocks = find_matching_blocks(query_indices, index_block_runs)
    # Intersect with available blocks
    # Is this necessary given sorted_blocks should contain ALL blocks?
    used_blocks = [b for b in used_blocks if b in sorted_blocks]
    # Check if used blocks are contiguous
    if any(used_blocks[i] + 1 != used_blocks[i + 1] for i in range(len(used_blocks) - 1)):
        logger.warning("Non-contiguous blocks detected in used_blocks; search may be suboptimal.")
        logger.info("Not implemented: returning no search results.")
        return [SearchResults(hits=[], paths=[])]

    if not used_blocks:
        logger.info("No overlapping blocks between query and index; returning no search results.")
        return [SearchResults(hits=[], paths=[])]

    num_layers = len(used_blocks)

    if debug:
        logger.info(
            "Starting search: budget=%.4f, num_layers=%d, blocks=%s",
            budget,
            num_layers,
            used_blocks,
        )

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
            query_blocks.append(query_spd[offset_start:offset_start+offset_size, offset_start:offset_start+offset_size])
            offset_start += offset_size 

    if not query_blocks:
        logger.info("No query blocks align to index layers; returning no search results.")
        return [SearchResults(hits=[], paths=[])]

    first_block = used_blocks[0]
    first_layer_nodes = block_to_node_indices.get(first_block, [])
    if not first_layer_nodes:
        logger.info("No nodes in first block; returning no search results.")
        return [SearchResults(hits=[], paths=[])]

    # Best path per leaf (keyed by leaf global_node_id).
    best_paths: Dict[int, SearchPath] = {}
    done: bool = False
    failed_starts: int = 0
    failed_paths: int = 0
    total_paths_explored: int = 0

    # Depth-first recursive traversal with budget-based backtracking.
    def dfs(layer_idx: int, node_idx: int, remaining_budget: float, total_dist: float, path_indices: List[int], valid_spds: set) -> None:
        nonlocal best_paths, done, failed_paths, total_paths_explored

        if done:
            return

        node = nodes[node_idx]

        if debug:
            logger.debug(
                "DFS visit: layer=%d node_idx=%d global_id=%d total_dist=%.4f remaining_budget=%.4f",
                layer_idx,
                node_idx,
                node.global_node_id,
                total_dist,
                remaining_budget,
            )

        # If we've reached the last layer, record a single path for this leaf.
        if layer_idx == num_layers - 1:
            total_paths_explored += 1
            if debug:
                logger.info(
                    "Leaf node reached: layer=%d node_idx=%d global_id=%d total_paths_explored=%d",
                    layer_idx,
                    node_idx,
                    node.global_node_id,
                    total_paths_explored,
                )
            if (total_paths_explored >= config.total_paths_limit):
                if len(best_paths) == 0:
                    logger.info(
                        "Stopping search early after reaching total paths limit of %d. Increase budget or total_paths_limit to get any results.",
                        config.total_paths_limit,
                    )
                done = True
                return
            leaf_id = node.global_node_id
            # If we've already recorded a path for this leaf, we don't
            # need to refine it; just treat this as another way to
            # reach the same leaf and stop.
            if leaf_id in best_paths:
                return

            node_path_global = [nodes[i].global_node_id for i in path_indices]
            candidate = SearchPath(
                node_path=node_path_global,
                total_distance=total_dist,
            )

            best_paths[leaf_id] = candidate
            if debug:
                logger.info("Leaf reached: right now we have %d distinct leaf paths", len(best_paths))

            # If we've reached the requested number of leaf paths, signal completion.
            if (config.max_results is not None and len(best_paths) >= config.max_results):
                if debug and (total_paths_explored >= config.total_paths_limit):
                    logger.info(
                        "Stopping search early after reaching total paths limit of %d.",
                        config.total_paths_limit,
                    )
    
                done = True
                return

            if debug:
                logger.info(
                    "Leaf reached: path=%s total_dist=%.4f remaining_budget=%.4f len(spd_ids)=%s",
                    node_path_global,
                    total_dist,
                    budget - total_dist,
                    len(node.metadata.members),
                )

            return

        next_layer_idx = layer_idx + 1
        if next_layer_idx >= num_layers:
            return

        next_block = used_blocks[next_layer_idx]
        query_block_next = query_blocks[next_layer_idx]

        # Collect children that belong to the next block and order them by distance.
        child_dists: List[Tuple[int, float, set]] = []
        for child_global_id in node.children:
            # By construction, global_node_id is equal to the index in
            # the ``nodes`` list, but guard against out-of-range values
            # for robustness.
            child_idx = child_global_id
            if child_idx < 0 or child_idx >= len(nodes):
                continue
            child = nodes[child_idx]
            if child.block_index != next_block:
                continue
            
            # --- NEW: Intersect members ---
            child_spds = {int(spd_id) for spd_id, _ in child.metadata.members}
            new_valid_spds = valid_spds.intersection(child_spds)
            if not new_valid_spds:
                continue
            
            # dist = log_euclidean_distance(query_block_next, child.metadata.mean, normalize=True)
            # this is log distance
            p = child.metadata.mean.shape[0]
            L_block = log_spd(query_block_next)
            diff = L_block - child.metadata.mean
            dist = np.linalg.norm(diff, ord='fro') / np.sqrt(p)
            child_dists.append((child_idx, dist, new_valid_spds))

        # Explore children from closest to farthest.
        child_dists.sort(key=lambda x: x[1])

        for child_idx, dist, new_valid_spds in child_dists:
            new_total = total_dist + dist
            new_remaining = remaining_budget - dist
            if new_remaining < 0 or new_total > budget:
                # Count this as a failed DFS branch (no leaf reached
                # because budget is exhausted).
                failed_paths += 1
                if debug:
                    logger.debug(
                        "Prune child: layer=%d child_idx=%d dist=%.4f new_total=%.4f new_remaining=%.4f (failed_paths=%d)",
                        next_layer_idx,
                        child_idx,
                        dist,
                        new_total,
                        new_remaining,
                        failed_paths,
                    )
                if config.max_failed_paths is not None and failed_paths >= config.max_failed_paths:
                    if debug:
                        logger.info(
                            "Stopping search early after %d failed DFS branches.",
                            failed_paths,
                        )
                    done = True
                    return
                continue

            if debug:
                logger.debug(
                    "Recurse to child: layer=%d child_idx=%d dist=%.4f new_total=%.4f new_remaining=%.4f",
                    next_layer_idx,
                    child_idx,
                    dist,
                    new_total,
                    new_remaining,
                )

            dfs(next_layer_idx, child_idx, new_remaining, new_total, path_indices + [child_idx], new_valid_spds)

            # Optional early stopping if we've already collected enough results.
            if done:
                return

    # --- Seed the DFS from the first layer ---

    query_block_0 = query_blocks[0]
    start_candidates: List[Tuple[int, float]] = []
    if debug:
        logger.info(
            "Evaluating %d start candidates in first block %d",
            len(first_layer_nodes),
            first_block,
        )

    for node_idx in first_layer_nodes:
        node = nodes[node_idx]
        # dist = log_euclidean_distance(query_block_0, node.metadata.mean, normalize=True)
        p = node.metadata.mean.shape[0]
        L_block = log_spd(query_block_0)
        diff = L_block - node.metadata.mean
        dist = np.linalg.norm(diff, ord='fro') / np.sqrt(p)
        start_candidates.append((node_idx, dist))

    # Explore starting nodes in order of increasing distance.
    start_candidates.sort(key=lambda x: x[1])

    if debug:
        logger.info("Start candidates (node_idx, dist): %s", start_candidates)

    for node_idx, dist in start_candidates:
        if dist > budget:
            if debug:
                logger.debug(
                    "Skip start node_idx=%d dist=%.4f > budget=%.4f",
                    node_idx,
                    dist,
                    budget,
                )
            break
        remaining = budget - dist
        if debug:
            logger.debug(
                "Seed DFS from node_idx=%d dist=%.4f remaining_budget=%.4f",
                node_idx,
                dist,
                remaining,
            )
        before = len(best_paths)
        
        start_node = nodes[node_idx]
        valid_spds = {int(spd_id) for spd_id, _ in start_node.metadata.members}
        if valid_spds:
            dfs(0, node_idx, remaining, dist, [node_idx], valid_spds)

        # If this start contributed no new leaf paths, count it as a
        # failed attempt. For hard / false-positive queries this
        # provides a hard cap on search effort.
        if len(best_paths) == before:
            failed_starts += 1
            if debug and config.max_failed_starts is not None:
                logger.debug(
                    "Start node_idx=%d produced no leaves; failed_starts=%d/%d",
                    node_idx,
                    failed_starts,
                    config.max_failed_starts,
                )
            if config.max_failed_starts is not None and failed_starts >= config.max_failed_starts:
                if debug:
                    logger.info(
                        "Stopping search early after %d failed start candidates.",
                        failed_starts,
                    )
                break

        if done:
            break

    # Assemble final SearchResults, ordered by increasing total_distance.
    paths_sorted = sorted(best_paths.values(), key=lambda p: p.total_distance)
    if debug:
        logger.info("Search complete: found %d hits within budget %.4f", len(paths_sorted), budget)
    if config.max_results is not None:
        paths_sorted = paths_sorted[: config.max_results]

    # Flatten members from all kept leaves, preserving path order.
    # hits: List[int] = []
    # seen: set[int] = set()
    # for p in paths_sorted:
    #     # Leaf node is the last node in the path.
    #     leaf_global_id = p.node_path[-1]
    #     leaf_node = nodes[leaf_global_id]
    #     for spd_id, _ in leaf_node.metadata.members:
    #         sid = int(spd_id)
    #         if sid not in seen:
    #             seen.add(sid)
    #             hits.append(sid)

    return SearchResults(paths=paths_sorted)


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
