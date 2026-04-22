"""Optional per-(niche, layer) interval sub-matrix index for Spindle.

Builds and queries an interval index alongside (but independent of) the
standard block-level DAG.  Interval submatrices are clustered directly in
log-Euclidean space per (niche j, layer ℓ, interval I), so no guarantee from
full-block clusters is assumed or inherited.

Public API
----------
build_all_interval_indices(data, config)   -> IntervalIndexData
query_interval_index(idx_data, cluster_id, block_index, interval, query_spd_sub, top_k=5)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .utils import get_logger

logger = get_logger(__name__)

# Type alias: cluster_id -> block_index -> (a,b) -> List[IntervalCluster]
IntervalIndexData = Dict[int, Dict[int, Dict[Tuple[int, int], List["IntervalCluster"]]]]


@dataclass
class IntervalCluster:
    """One cluster within a (niche, layer, interval) cell.

    centroid : log-mean of the cluster, shape (interval_size, interval_size).
    members  : spd_ids belonging to this cluster.
    radius   : max normalized LE distance from any member to the centroid.
    """
    centroid: np.ndarray
    members: List[int]
    radius: float


# ---------------------------------------------------------------------------
# Interval enumeration
# ---------------------------------------------------------------------------

def generate_intervals(d: int, mode: str = "dyadic") -> List[Tuple[int, int]]:
    """Return contiguous half-open intervals [a, b) for a block of size d.

    mode="all"    : every (a,b) with 0<=a<b<=d — O(d^2) intervals.
    mode="dyadic" : intervals of length 2^k anchored at multiples of 2^k.
    mode="fixed"  : only the single interval (0, d).
    """
    if mode == "all":
        return [(a, b) for a in range(d) for b in range(a + 1, d + 1)]

    if mode == "dyadic":
        ivls: List[Tuple[int, int]] = []
        k = 1
        while k <= d:
            a = 0
            while a + k <= d:
                ivls.append((a, a + k))
                a += k
            k *= 2
        return ivls

    if mode == "fixed":
        return [(0, d)]

    raise ValueError(f"Unknown interval_mode: {mode!r}. Choose 'all', 'dyadic', or 'fixed'.")


def decompose_to_dyadic(a: int, b: int) -> List[Tuple[int, int]]:
    """Decompose arbitrary [a, b) into non-overlapping dyadic intervals.

    Any contiguous interval decomposes into at most 2·log₂(b-a) dyadic
    pieces, all of which are present in a dyadic-mode interval index.

    Examples
    --------
    >>> decompose_to_dyadic(3, 11)
    [(3, 4), (4, 8), (8, 10), (10, 11)]
    >>> decompose_to_dyadic(0, 8)
    [(0, 8)]
    """
    pieces: List[Tuple[int, int]] = []
    cur = a
    while cur < b:
        # largest power-of-2 step that keeps us aligned and within [cur, b)
        k = 1
        while k * 2 <= (b - cur) and (cur % (k * 2)) == 0:
            k *= 2
        pieces.append((cur, cur + k))
        cur += k
    return pieces


# ---------------------------------------------------------------------------
# Introspection helpers
# ---------------------------------------------------------------------------

def describe_intervals(d: int, mode: str = "dyadic") -> None:
    """Print a visual map of every interval that would be built for a block of size d.

    Each row is one interval [a, b).  The bar shows which gene positions it
    covers so you can immediately see the coverage pattern.

    Examples
    --------
    >>> describe_intervals(8, "dyadic")
    Block size: 8  mode: dyadic  →  7 intervals

    len  [a, b)   coverage (gene positions 0–7)
    ---  ------   --------------------------------
      1  [0, 1)   █░░░░░░░
      1  [1, 2)   ░█░░░░░░
      1  [2, 3)   ░░█░░░░░
      1  [3, 4)   ░░░█░░░░
      2  [0, 2)   ██░░░░░░
      2  [2, 4)   ░░██░░░░
      4  [0, 4)   ████░░░░
    ...
    """
    ivls = generate_intervals(d, mode)
    by_len: dict = {}
    for (a, b) in ivls:
        by_len.setdefault(b - a, []).append((a, b))

    print(f"Block size: {d}  mode: {mode}  →  {len(ivls)} intervals\n")
    print(f"{'len':>4}  {'[a, b)':7}   coverage (gene positions 0–{d-1})")
    print(f"{'---':>4}  {'------':7}   {'─' * d}")
    for length in sorted(by_len):
        for (a, b) in by_len[length]:
            bar = "░" * a + "█" * (b - a) + "░" * (d - b)
            print(f"{length:>4}  [{a:2d},{b:2d})   {bar}")
    print()


def summarize_interval_index(ivl_idx: "IntervalIndexData") -> None:
    """Print a compact per-(niche, block) breakdown of a built interval index.

    Shows interval count, cluster count, and member/radius statistics so you
    can quickly judge whether epsilon is too tight or too loose.
    """
    for niche in sorted(ivl_idx):
        for blk in sorted(ivl_idx[niche]):
            cells = ivl_idx[niche][blk]
            if not cells:
                print(f"  niche={niche} block={blk}: skipped (block too large)")
                continue
            all_clusters = [c for cs in cells.values() for c in cs]
            if not all_clusters:
                continue
            sizes  = [len(c.members) for c in all_clusters]
            radii  = [c.radius       for c in all_clusters]
            # group clusters by interval length for a length-wise view
            by_len: dict = {}
            for (a, b), cs in cells.items():
                by_len.setdefault(b - a, []).extend(cs)
            len_summary = "  ".join(
                f"len{length}:{len(cs)}cl" for length, cs in sorted(by_len.items())
            )
            print(
                f"  niche={niche} block={blk}: "
                f"{len(cells)} intervals  {len(all_clusters)} clusters  "
                f"members [{min(sizes)}–{max(sizes)} avg {sum(sizes)/len(sizes):.1f}]  "
                f"radius [{min(radii):.3f}–{max(radii):.3f}]"
            )
            print(f"    by length → {len_summary}")


# ---------------------------------------------------------------------------
# Log-space helpers (mirrors _log_spd in index.py — kept local to avoid
# circular imports)
# ---------------------------------------------------------------------------

def _log_spd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T


# ---------------------------------------------------------------------------
# Epsilon-constrained greedy clustering in log space
# (mirrors the online k-means in index.py; kept self-contained here)
# ---------------------------------------------------------------------------

def _cluster_log_blocks(
    items: List[Tuple[int, np.ndarray]],   # (spd_id, L_sub) — already in log space
    epsilon: float,
    max_iters: int = 5,
) -> List[IntervalCluster]:
    """Greedy + Lloyd epsilon-constrained clustering on log-space blocks.

    Returns a list of IntervalCluster objects whose radius <= epsilon/2.
    Any cluster that still violates the constraint after refinement is
    split into singletons.
    """
    if not items:
        return []

    p = items[0][1].shape[0]
    scale = np.sqrt(p) if p > 1 else 1.0
    log_map = {spd_id: L for spd_id, L in items}
    sids = [spd_id for spd_id, _ in items]

    # ---- greedy initialization ----
    clusters: List[dict] = []
    for spd_id, L in items:
        best_cid, best_d = -1, float("inf")
        for cid, c in enumerate(clusters):
            diff = L - c["log_mean"]
            d = float(np.linalg.norm(diff, ord="fro") / scale)
            r = float(c["radius"])
            if (r + d) / 2.0 <= epsilon / 2.0 and d < best_d:
                best_d, best_cid = d, cid
        if best_cid >= 0:
            c = clusters[best_cid]
            c["spd_ids"].append(spd_id)
            c["log_sum"] += L
            c["count"] += 1
            c["log_mean"] = c["log_sum"] / float(c["count"])
            diff_new = L - c["log_mean"]
            c["radius"] = max(float(c["radius"]), float(np.linalg.norm(diff_new, ord="fro") / scale))
        else:
            clusters.append({"spd_ids": [spd_id], "log_sum": L.copy(),
                              "count": 1, "log_mean": L.copy(), "radius": 0.0})

    # ---- Lloyd refinement ----
    for _ in range(max_iters):
        assign = []
        for _, L in items:
            best_cid, best_d = 0, float("inf")
            for cid, c in enumerate(clusters):
                if c["log_mean"] is None:
                    continue
                d = float(np.linalg.norm(L - c["log_mean"], ord="fro") / scale)
                if d < best_d:
                    best_d, best_cid = d, cid
            assign.append(best_cid)

        new_clusters = [{"spd_ids": [], "log_sum": np.zeros_like(items[0][1]),
                         "count": 0, "log_mean": None, "radius": 0.0}
                        for _ in clusters]
        for i, (spd_id, L) in enumerate(items):
            cid = assign[i]
            new_clusters[cid]["spd_ids"].append(spd_id)
            new_clusters[cid]["log_sum"] += L
            new_clusters[cid]["count"] += 1
        for c in new_clusters:
            if c["count"] > 0:
                c["log_mean"] = c["log_sum"] / float(c["count"])
        if all(new_clusters[i]["spd_ids"] == clusters[i]["spd_ids"] for i in range(len(clusters))):
            clusters = new_clusters
            break
        clusters = new_clusters

    # ---- compute radii; split violators into singletons ----
    result: List[IntervalCluster] = []
    for c in clusters:
        if not c["spd_ids"] or c["log_mean"] is None:
            continue
        mean = c["log_mean"]
        r = max(float(np.linalg.norm(log_map[sid] - mean, ord="fro") / scale)
                for sid in c["spd_ids"])
        if r <= epsilon / 2.0:
            result.append(IntervalCluster(centroid=mean.copy(), members=list(c["spd_ids"]), radius=r))
        else:
            for sid in c["spd_ids"]:
                L = log_map[sid]
                result.append(IntervalCluster(centroid=L.copy(), members=[sid], radius=0.0))
    return result


# ---------------------------------------------------------------------------
# Per-(niche, layer) interval index build
# ---------------------------------------------------------------------------

def _build_block_interval_index(
    log_spd_cache: List[Tuple[int, np.ndarray]],   # (spd_id, L_block) in log space
    block_size: int,
    epsilon: float,
    mode: str = "dyadic",
    max_iters: int = 5,
) -> Dict[Tuple[int, int], List[IntervalCluster]]:
    """Build interval clusters for a single (niche, layer) pair."""
    result: Dict[Tuple[int, int], List[IntervalCluster]] = {}
    for (a, b) in generate_intervals(block_size, mode):
        sub_items = [(sid, L[a:b, a:b]) for sid, L in log_spd_cache]
        result[(a, b)] = _cluster_log_blocks(sub_items, epsilon, max_iters=max_iters)
    return result


def build_all_interval_indices(data, config) -> IntervalIndexData:
    """Build interval indices for every (niche, layer) pair in *data*.

    Parameters
    ----------
    data : ProcessedData
        Must have ``spd_matrices``, ``labels``, ``spd_ids``, ``perm_list``,
        ``block_dict``, and ``block_dict[cluster_id]`` populated.
    config : IndexConfig
        Must have ``use_interval_index=True``; reads ``interval_mode``,
        ``interval_eps``, ``interval_max_layer_size``, ``interval_max_iters``.

    Returns
    -------
    IntervalIndexData
        Nested dict ``[cluster_id][block_index][(a,b)] -> List[IntervalCluster]``.
        Returns an empty dict when ``config.use_interval_index`` is False.
    """
    if not getattr(config, "use_interval_index", False):
        return {}

    mode = getattr(config, "interval_mode", "dyadic")
    interval_eps = getattr(config, "interval_eps", None)
    max_iters = getattr(config, "interval_max_iters", 5)
    max_layer_size = getattr(config, "interval_max_layer_size", None) or 64

    spd_matrices = np.asarray(data.spd_matrices)
    spd_ids_all = np.asarray(data.spd_ids)
    labels = np.asarray(data.labels)
    interval_index: IntervalIndexData = {}

    for cluster_id in sorted(set(int(c) for c in labels)):
        logger.info("Building interval index for niche %d", cluster_id)
        interval_index[cluster_id] = {}

        epsilon = float(interval_eps if interval_eps is not None
                        else config.epsilon_dict.get(cluster_id, 0.5))

        mask = labels == cluster_id
        spd_matrices_cluster = spd_matrices[mask]
        spd_ids_cluster = spd_ids_all[mask]
        permutation = data.perm_list[cluster_id]
        block_runs = data.block_dict[cluster_id]

        for block_index, (block_start, block_end) in enumerate(block_runs):
            block_size = block_end - block_start

            if block_size > max_layer_size:
                logger.info(
                    "Niche %d block %d: size %d > max %d — skipping interval index.",
                    cluster_id, block_index, block_size, max_layer_size,
                )
                interval_index[cluster_id][block_index] = {}
                continue

            perm_idx = permutation[block_start:block_end]
            log_spd_cache: List[Tuple[int, np.ndarray]] = [
                (int(spd_ids_cluster[i]),
                 _log_spd(spd_matrices_cluster[i][np.ix_(perm_idx, perm_idx)]))
                for i in range(len(spd_matrices_cluster))
            ]

            interval_index[cluster_id][block_index] = _build_block_interval_index(
                log_spd_cache, block_size, epsilon, mode=mode, max_iters=max_iters,
            )
            n_ivl = len(interval_index[cluster_id][block_index])
            logger.info("Niche %d block %d: %d intervals indexed.", cluster_id, block_index, n_ivl)

    return interval_index


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_interval_index(
    interval_index: IntervalIndexData,
    cluster_id: int,
    block_index: int,
    interval: Tuple[int, int],
    query_spd_sub: np.ndarray,
    top_k: int = 5,
) -> List[Tuple[float, List[int]]]:
    """Search one (niche, layer, interval) cell of the interval index.

    Pass ``top_k=None`` to return all clusters (used by
    :func:`query_arbitrary_interval` when intersecting pieces).

    Parameters
    ----------
    interval_index : built by :func:`build_all_interval_indices`.
    cluster_id : niche id (j).
    block_index : layer id (ℓ).
    interval : (a, b) half-open interval within the permuted block.
    query_spd_sub : raw SPD principal sub-matrix of shape (b-a, b-a).
        The log is computed internally.
    top_k : return at most this many (distance, member_list) pairs.

    Returns
    -------
    List of ``(distance, spd_ids)`` sorted by ascending distance.
    Falls back to an empty list when the cell is absent.
    """
    cell = (interval_index.get(cluster_id) or {}).get(block_index) or {}
    clusters = cell.get(interval)
    if not clusters:
        return []

    p = query_spd_sub.shape[0]
    scale = np.sqrt(p) if p > 1 else 1.0
    L_q = _log_spd(query_spd_sub)

    scored = [
        (float(np.linalg.norm(L_q - c.centroid, ord="fro") / scale), c.members)
        for c in clusters
    ]
    scored.sort(key=lambda x: x[0])
    return scored if top_k is None else scored[:top_k]


def query_arbitrary_interval(
    interval_index: "IntervalIndexData",
    cluster_id: int,
    block_index: int,
    a: int,
    b: int,
    query_spd_block: np.ndarray,
    top_k: int = 10,
) -> List[Tuple[float, List[int]]]:
    """Search an arbitrary [a, b) interval via dyadic decomposition.

    Decomposes [a, b) into non-overlapping dyadic pieces, queries each piece
    independently, intersects the candidate sets, and ranks survivors by the
    sum of per-piece distances.

    Parameters
    ----------
    interval_index : built by :func:`build_all_interval_indices`.
    cluster_id : niche id.
    block_index : block (layer) index within the niche.
    a, b : half-open interval within the permuted block (0-indexed).
    query_spd_block : raw SPD sub-matrix for the **full block**, shape
        (block_size, block_size).  The function slices [a:b, a:b] for each
        dyadic piece internally.
    top_k : number of results to return.

    Returns
    -------
    List of ``(total_distance, spd_ids)`` sorted by ascending distance.
    Empty when [a, b) cannot be covered (missing dyadic cells).
    """
    pieces = decompose_to_dyadic(a, b)

    # query each dyadic piece → {spd_id: distance}
    piece_scores: List[Dict[int, float]] = []
    for pa, pb in pieces:
        q_sub = query_spd_block[pa:pb, pa:pb]
        piece_results = query_interval_index(
            interval_index, cluster_id, block_index, (pa, pb), q_sub, top_k=None
        )
        if not piece_results:
            # a required piece is missing — can't cover the interval
            return []
        scores: Dict[int, float] = {}
        for dist, members in piece_results:
            for m in members:
                scores[m] = dist
        piece_scores.append(scores)

    # intersect: keep only tiles present in every piece
    common = set(piece_scores[0].keys())
    for ps in piece_scores[1:]:
        common &= ps.keys()

    if not common:
        return []

    # sum distances across pieces as combined score
    ranked = sorted(
        ((sum(ps[tid] for ps in piece_scores), tid) for tid in common),
        key=lambda x: x[0],
    )
    # group ties into member lists (each tile is its own entry here)
    results = [(dist, [tid]) for dist, tid in ranked[:top_k]]
    return results


__all__ = [
    "IntervalCluster",
    "IntervalIndexData",
    "generate_intervals",
    "decompose_to_dyadic",
    "describe_intervals",
    "summarize_interval_index",
    "build_all_interval_indices",
    "query_interval_index",
    "query_arbitrary_interval",
]
