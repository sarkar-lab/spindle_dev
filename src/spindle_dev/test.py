"""Lightweight sanity helpers for the SPD block-DAG index.

This module is not a full test-suite. It provides two small utilities
you can call from a notebook or REPL to quickly sanity-check that:

* ground-truth paths can be reconstructed from a built index, and
* the search traversal can recover those paths for random queries.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from .utils import get_logger
from .metrics import log_euclidean_distance_for_SPD
from . import search


logger = get_logger(__name__)


def create_ground_truth_paths(dag_dict: Dict[int, Any]) -> Dict[int, Dict[int, List[int]]]:
    """Build canonical block-DAG paths for each SPD in each cluster.

    Parameters
    ----------
    dag_dict:
        Mapping ``cluster_id -> IndexHandle`` as returned by
        :func:`index_spds`.

    Returns
    -------
    Dict[int, Dict[int, List[int]]]
        ``ground_truth_paths[cluster_id][spd_id] -> [global_node_id, ...]``
        giving one canonical path per SPD across all blocks.
    """

    ground_truth_paths: Dict[int, Dict[int, List[int]]] = {}

    for cluster_id, index_handle in dag_dict.items():
        dag_nodes = index_handle.nodes
        block_to_nodes = index_handle.block_to_nodes

        # Ordered list of block indices for this cluster.
        sorted_blocks = sorted(block_to_nodes.keys())

        # Map from node_id to index in dag_nodes (block_to_nodes stores node_ids).
        id_to_idx = {node.node_id: i for i, node in enumerate(dag_nodes)}

        # Collect all SPD ids that appear in this cluster.
        spd_ids = set()
        for node in dag_nodes:
            for spd_id, _ in node.metadata.members:
                spd_ids.add(int(spd_id))

        spd_paths: Dict[int, List[int]] = {}

        # For each SPD, walk blocks in order and find the unique node
        # whose metadata.members contains (spd_id, block_index).
        for spd_id in sorted(spd_ids):
            path: List[int] = []
            valid = True
            for block_idx in sorted_blocks:
                found_idx = None
                for node_local_id in block_to_nodes[block_idx]:
                    node_idx = id_to_idx[node_local_id]
                    node = dag_nodes[node_idx]
                    if any(int(sid) == spd_id and blk == block_idx for sid, blk in node.metadata.members):
                        found_idx = node_idx
                        path.append(node.global_node_id)
                        break
                if found_idx is None:
                    valid = False
                    break
            if valid and path:
                spd_paths[spd_id] = path

        ground_truth_paths[cluster_id] = spd_paths
        logger.info(
            "Built ground-truth paths for %d SPDs in cluster %d across %d blocks.",
            len(spd_paths),
            cluster_id,
            len(sorted_blocks),
        )

    return ground_truth_paths


def search_ground_truth_matrices(
    data: Any,
    dag_dict: Dict[int, Any],
    config: Any,
    ground_truth_paths: Dict[int, Dict[int, List[int]]],
    search_cfg: Any = None,
    *,
    max_queries: int = 2000,
    seed: int = 42,
    skip_baseline = False,
) -> List[Dict[str, Any]]:
    """Randomly probe the index and check path recovery.

    This mirrors the notebook logic but is parameterized so it can be
    called directly from code.

    Parameters
    ----------
    data:
        ProcessedData instance with ``spd_matrices``, ``labels``,
        ``perm_list`` and ``block_dict`` populated.
    dag_dict:
        Mapping ``cluster_id -> IndexHandle`` as returned by
        :func:`index_spds`.
    config:
        IndexConfig used to build the index; must expose
        ``epsilon_dict``.
    ground_truth_paths:
        Output of :func:`create_ground_truth_paths`.
    max_queries:
        Maximum number of random queries to issue.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    List[dict]
        One record per query with basic match statistics.
    """

    from tqdm import tqdm

    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(data.spd_matrices))
    valid_clusters = list(dag_dict.keys())
    mask = np.isin(data.labels, valid_clusters)
    candidate_indices = all_indices[mask]
    n_queries = min(max_queries, len(candidate_indices))
    query_indices = rng.choice(candidate_indices, size=n_queries, replace=False)

    records: List[Dict[str, Any]] = []
    logger.info("Running sanity search with %d queries.", n_queries)

    for q_idx in tqdm(query_indices):
        q_idx = int(q_idx)
        cluster_id = int(data.labels[q_idx])
        index_handle = dag_dict.get(cluster_id)
        if index_handle is None:
            continue

        # Query SPD in original (gene) order.
        q_spd = data.spd_matrices[q_idx]

        # Prepare search.
        perm = data.perm_list[cluster_id]
        q_spd_perm = q_spd[np.ix_(perm, perm)]
        query_block_runs = data.block_dict[cluster_id]
        num_blocks = len(query_block_runs) if query_block_runs is not None else 1

        if not skip_baseline:
            cluster_mask = data.labels == cluster_id
            candidate_ids = np.where(cluster_mask)[0]
            t0 = time.perf_counter()
            dists = []
            for cid in candidate_ids:
                cid_int = int(cid)
                # d = metrics.log_euclidean_distance(q_spd, data.spd_matrices[cid_int], normalize=True)
                d = log_euclidean_distance_for_SPD(q_spd, data.spd_matrices[cid_int])
                dists.append((cid_int, float(d)))
            dists.sort(key=lambda x: x[1])
            baseline_time = time.perf_counter() - t0
        else:
            baseline_time = 0

        # Heuristic budget: scale per-cluster epsilon by number of blocks.
        epsilon = config.epsilon_dict[cluster_id]
        budget = float(epsilon) * float(num_blocks) * 1.5

        t1 = time.perf_counter()
        if not search_cfg:
            search_cfg = search.SearchConfig(max_results=2, debug=False, max_failed_starts=5, max_failed_paths=10)
        results = search.search_index(
            index_handle,
            q_spd_perm,
            [],
            query_block_runs,
            budget,
            config=search_cfg,
        )
        search_time = time.perf_counter() - t1

        # Check if any returned path matches the full ground-truth path.
        
        # FIX: Extract the true global ID that the DAG actually stored
        # Safely handle different dataclass structures (e.g. .id or .spd_id)
        tile_obj = data.metadata["tiles"][q_idx]
        true_spd_id = int(getattr(tile_obj, 'spd_id', getattr(tile_obj, 'id', q_idx)))
        
        gt_path = ground_truth_paths.get(cluster_id, {}).get(true_spd_id)
        
        if gt_path is None:
            logger.warning("Ground-truth path not found for TRUE tile id %d (loop idx %d) in cluster %d.", true_spd_id, q_idx, cluster_id)
            continue

        matched = False
        matched_leaf = False
        matched_budget = None
        for path in results.paths:
            if path.node_path == gt_path:
                matched = True
                matched_leaf = True
                matched_budget = path.total_distance
                break

        # If no exact path match, see if any result shares the same leaf.
        if not matched:
            gt_leaf_node = gt_path[-1]
            for path in results.paths:
                if path.node_path and path.node_path[-1] == gt_leaf_node:
                    matched_leaf = True
                    matched_budget = path.total_distance
                    break

        records.append(
            {
                "query_idx": q_idx,
                "cluster_id": cluster_id,
                "budget": budget,
                "matched_gt": matched,
                "matched_leaf": matched_leaf,
                "matched_budget": matched_budget,
                "search_time": search_time,
                "baseline_time": baseline_time 
            }
        )

    return records


def run_sanity_search(
    data: Any,
    dag_dict: Dict[int, Any],
    config: Any,
    search_cfg: Any = None,
    *,
    max_queries: int = 2000,
    seed: int = 42,
    skip_baseline=False
) -> Dict[str, Any]:
    """End-to-end sanity check in a single call.

    This convenience wrapper:

    1. Builds ``ground_truth_paths`` from ``dag_dict``.
    2. Runs ``search_ground_truth_matrices`` with the given ``data`` and
       ``config``.

    Returns a dict with both the ground truth and the per-query records:

    ``{"ground_truth_paths": ..., "records": ...}``.
    """

    gt_paths = create_ground_truth_paths(dag_dict)
    logger.info("Created ground-truth paths for all clusters.")
    logger.info("Search config: %s", search_cfg)
    records = search_ground_truth_matrices(
        data=data,
        dag_dict=dag_dict,
        config=config,
        ground_truth_paths=gt_paths,
        search_cfg=search_cfg,
        max_queries=max_queries,
        seed=seed,
        skip_baseline=skip_baseline
    )

    # Print a short summary to the log.
    n = len(records)
    matched_gt = sum(1 for r in records if r.get("matched_gt"))
    matched_leaf = sum(1 for r in records if r.get("matched_leaf"))
    mean_time = (
        float(sum(r.get("search_time", 0.0) for r in records)) / n
        if n > 0
        else 0.0
    )
    logger.info(
        "Sanity search: %d queries, %d exact path matches, %d leaf matches, mean search_time=%.4fs",
        n,
        matched_gt,
        matched_leaf,
        mean_time,
    )

    return {"ground_truth_paths": gt_paths, "records": records}
