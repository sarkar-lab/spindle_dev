"""Index construction for SPD sub-matrices from spatial datasets.

This module builds a block-structured DAG index over SPD matrices as
specified in .github/copilot-instructions.md. The implementation here
focuses on clear data structures and interfaces; higher-level projects
can extend or optimize clustering strategies as needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path

import numpy as np
import pickle
import os

from .metrics import (
    log_euclidean_distance,
    spd_to_correlation,
    build_ultrametrics,
    spd_tree_feature_matrix,
    leiden_clustering_latent,
    consensus_tree_from_ultrametrics,
    blocks_from_fcluster,
    dp_group_runs,
    block_metrics_over_t,
    knee_from_num_blocks,
    pick_t_knee_with_size_guard

)
from .preprocessing import SpatialDataset, QuadTile
from .utils import DeterministicConfig, configure_determinism, get_logger
from .typing import (
    IndexHandle,
    IndexConfig,
    BlockClusterNode,
    BlockClusterMetadata,
    DatasetIndex
)

PathLike = Union[str, os.PathLike]

from umap import UMAP 
logger = get_logger(__name__)



class ProcessedData:
    """Holds processed SPD matrices and related info from spatial data.
    Inputs:
    tiles:
        List of QuadTile objects representing spatial tiling.
    tile_stats:
        List of dicts with per-tile statistics, including 'cov' for
        the SPD covariance matrix and 'tile_id' for the tile identifier.
    genes_work:
        List of gene names corresponding to the rows/columns of the
        SPD matrices.
    
    """
    def __init__(self, tiles: List[QuadTile], tile_stats: List[Dict], genes_work: List[str], num_spots: int):
        self.spd_matrices = [t["cov"] for t in tile_stats]
        self.spd_ids = [t["tile_id"] for t in tile_stats]

        self.metadata = {
            "tiles": tiles,
            "genes": np.array(genes_work),
        }
        self.num_genes = len(genes_work)
        self.num_spots = num_spots
        self.latent = {}
        # Models fitted during dimensionality reduction / embedding.
        # These are useful for aligning unseen SPDs into the same
        # latent space (e.g., for assigning clusters to new tiles).
        self.pca_model = None
        self.labels = None
        self.spot_label = None
        self.U_list = None
        self.U_mean_list = {}
        self.R_mean_list = {}
        self.perm_list = {}
        self.Z_list = {}
        self.block_initial_dict = {}
        self.block_dict = {}
    
    
    def reduce_dim(
        self,
        cluster_distance: str = "tree",
        num_pca_components: int = 30,
        n_components: int = 2,
        random_state: int = 0,
        do_umap: bool = False
    ):
        logger.info("Clustering SPD-s using '%s' distance.", cluster_distance)
        try:
            from sklearn.decomposition import PCA  # type: ignore
            from sklearn.cluster import KMeans  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "scikit-learn is required for ProcessedData.cluster_spds "
                "when cluster_distance='tree'. Install scikit-learn or "
                "change clustering options."
            ) from exc

        logger.info("Building ultrametric features from SPD matrices.")
        U_list, Z_list = build_ultrametrics(self.spd_matrices)
        logger.info("Computing latent features from the tree representations.")
        latent_feat = spd_tree_feature_matrix(U_list)
        pca = PCA(n_components=num_pca_components, random_state=random_state)
        logger.info("Reducing latent features to %d dimensions using PCA.", num_pca_components)
        Z_pca = pca.fit_transform(latent_feat)
        logger.info("Explained variance ratios by PCA components: %s", pca.explained_variance_ratio_)
        self.latent['pca'] = Z_pca
        # Store the fitted PCA model so we can embed unseen SPDs into
        # the same latent space later without refitting.
        self.pca_model = pca
        self.U_list = U_list
        logger.info("Reducing latent features to %d dimensions using UMAP.", n_components)
        umap = UMAP(
            n_components=n_components,
            random_state=random_state,
            metric="euclidean",      # or something SPD-aware if you define it
            n_neighbors=15,          # tune these
            min_dist=0.1,
        )
        Z_umap = umap.fit_transform(Z_pca)
        self.latent['umap'] = Z_umap

    
    def cluster_spds(
        self,
        cluster_distance: str = "tree",
        cluster_method: str = "kmeans",
        n_components: int = 2,
        random_state: int = 0,
        n_clusters: int = 2,
        resolution: float = 0.1,
        block_diagonalize: bool = True
    ) -> None:
        """Group SPD IDs by cluster labels.

        Returns a mapping from cluster label to list of SPD IDs
        belonging to that cluster.
        """
        if cluster_distance == "tree":
            # Placeholder: implement hierarchical clustering and assign labels
            # First convert covariance matrices to correlation 
            logger.info("Clustering SPD-s using '%s' distance.", cluster_distance)
            # corrs_np = np.array([spd_to_correlation(C) for C in self.spd_matrices])
            # corrs_np = np.stack(corrs_np)

            # Lazy import heavy sklearn dependencies only when clustering is used.
            try:
                from sklearn.decomposition import PCA  # type: ignore
                from sklearn.cluster import KMeans  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "scikit-learn is required for ProcessedData.cluster_spds "
                    "when cluster_distance='tree'. Install scikit-learn or "
                    "change clustering options."
                ) from exc

            if 'pca' in self.latent:
                latent_feat = self.latent['pca']
                U_list = self.U_list
            else:
                logger.error("Latent features not found. Please run reduce_dim() before clustering.")
            # logger.info("Building ultrametric features from SPD matrices.")
            # U_list, Z_list = build_ultrametrics(self.spd_matrices)
            # logger.info("Computing latent features from the tree representations.")
            # latent_feat = spd_tree_feature_matrix(U_list)
            # pca = PCA(n_components=n_components, random_state=random_state)
            # logger.info("Reducing latent features to %d dimensions using PCA.", n_components)
            # Z_pca = pca.fit_transform(latent_feat)
            # logger.info("Explained variance ratios by PCA components: %s", pca.explained_variance_ratio_)
            # self.latent['pca'] = Z_pca
            # self.U_list = U_list
            

            if cluster_method == 'kmeans':
                logger.info("Clustering SPD matrices into %d clusters using KMeans.", n_clusters)
                km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
                labels = km.fit_predict(latent_feat) 
            elif cluster_method == 'leiden':
                logger.info("Clustering SPD matrices using Leiden clustering with resolution %.2f.", resolution)
                labels, _, _ = leiden_clustering_latent(
                    latent_feat, 
                    k_neighbors=10, 
                    resolution=resolution
                )
            else:
                raise ValueError(f"Unknown cluster_method: {cluster_method}")
            self.labels = labels
            # Assigning point

        else:
            raise ValueError(f"Unknown cluster_distance: {cluster_distance}")
        
        if cluster_distance == 'tree':
            logger.info("Since clustering method is tree, I am going to find global order per cluster")
            for cluster_id in set(labels):
                logger.info("Finding consensus tree for cluster %d", cluster_id)
                U_mean, Z, perm = consensus_tree_from_ultrametrics(U_list, labels, cluster_id)
                self.U_mean_list[cluster_id] = U_mean
                self.perm_list[cluster_id] = perm
                self.Z_list[cluster_id] = Z
        

    def assign_label_to_spots(self):
        # self.spot_label = np.full(self.num_spots, -1, dtype=int)
        # make it a dictionary
        self.spot_label = {}
        for i,t in enumerate(self.metadata['tiles']):
            c = self.labels[i]
            for id in t.idx:
                self.spot_label[id] = c


    def get_spd_corr_matrices(self) -> List[np.ndarray]:
        """Get list of SPD correlation matrices corresponding to the
        processed SPD covariance matrices.
        """
        return [spd_to_correlation(C) for C in self.spd_matrices]
    

    def get_corr_mean_by_cluster(self):
        """Get the mean correlation matrix for a given cluster ID."""
        if not type(self.spd_matrices) is np.ndarray:
            self.spd_matrices = np.array(self.spd_matrices)
        
        for cluster_id in set(self.labels):
            logger.info("Computing mean correlation matrix for cluster %d", cluster_id)
            idxs = np.where(self.labels == cluster_id)[0]
            corrs = [spd_to_correlation(self.spd_matrices[i]) for i in idxs]
            corrs_stack = np.stack(corrs, axis=0)
            corr_mean = corrs_stack.mean(axis=0)
            self.R_mean_list[cluster_id] = corr_mean
        

    def get_block_diagonal_order(self, t: float = 0.92, min_size: int =10) -> None:
        """Get block-diagonal ordering for each cluster."""
        for cluster_id in set(self.labels):
            logger.info("Finding block-diagonal order for cluster %d", cluster_id)
            R_mean = self.R_mean_list[cluster_id]
            perm = self.perm_list[cluster_id]
            Z = self.Z_list[cluster_id]
            _, runs_fcluster = blocks_from_fcluster(Z, perm, t=t)
            runs = dp_group_runs(R_mean, perm, runs_fcluster, min_size=min_size, lam=0.0)
            self.block_dict[cluster_id] = runs

    def get_adaptive_runs(self, find_blocks=False, with_size_guard=False, min_size=1, max_size=50, lam=0, min_final_size=5, max_final_size=50, step_size=5):
        out_dict = {}
        for cluster_id in set(self.labels):
            logger.info("Finding adaptive block runs for cluster %d", cluster_id)
            Z = self.Z_list[cluster_id]
            perm = self.perm_list[cluster_id]
            out = block_metrics_over_t(Z, perm)
            out_dict[cluster_id] = out
            if find_blocks:
                if with_size_guard:
                    res = pick_t_knee_with_size_guard(Z, perm, out['t'], min_size=min_size, max_size=max_size)
                    t_choice = res['t']
                    
                else:
                    t_knee, t_choice, k, j, dy, d2y = knee_from_num_blocks(out["t"], out["num_blocks"])
                    logger.info(f" Chose t={t_choice} with knee at t={t_knee}, resulting in {k} blocks.")
                _, run_fcluster = blocks_from_fcluster(Z, perm, t=t_choice)
                self.block_initial_dict[cluster_id] = run_fcluster
                _, try_default = blocks_from_fcluster(Z, perm, t=0.92)
                logger.info(f" Chose t={t_choice} resulting in {len(run_fcluster)} blocks instead of {len(try_default)} blocks would have gotten by default")
                
                runs = dp_group_runs(self.R_mean_list[cluster_id], perm, run_fcluster, min_size=min_final_size, max_size=max_final_size, lam=lam)
                # Do binary search if the conditions satisfied
                bad_runs = ((len(runs) < len(perm) / max_final_size) or (len(runs) >  len(perm) / min_final_size))
                num_tries = 1
                new_max_size = max_final_size
                while bad_runs:
                    new_max_size += step_size
                    # Think how can we increase max_final_size to get sensible runs
                    if new_max_size > self.R_mean_list[cluster_id].shape[0]:
                        logger.error(f"Cannot find sensible runs for cluster {cluster_id} even with step_size={step_size}. Giving up. Try with smaller step size")
                        break
                    runs = dp_group_runs(self.R_mean_list[cluster_id], perm, run_fcluster, min_size=min_final_size, max_size=new_max_size, lam=lam)
                    bad_runs = ((len(runs) < len(perm) / new_max_size) or (len(runs) >  len(perm) / min_final_size))
                    logger.info(f"First run failed trying with {len(runs)}... this is {num_tries}-th try with limit {len(perm) // new_max_size}, {len(perm) // min_final_size}")
                    num_tries += 1

                logger.info(f" Final block runs for cluster {cluster_id}: {len(runs)} blocks.")
                self.block_dict[cluster_id] = runs

        return out_dict

    def subset_data(self, percentage: float, random_state: Optional[int] = 0) -> "ProcessedData":
        """Return a new ProcessedData with a per-cluster subset of tiles.

        Parameters
        ----------
        percentage:
            Fraction of SPDs to keep *within each cluster*, e.g. ``0.7``
            keeps 70% of the SPDs from every cluster (at least one per
            non-empty cluster).
        random_state:
            Optional seed for the RNG to make the subsampling
            deterministic.  If ``None``, use NumPy's global RNG.

        Notes
        -----
        - Requires ``self.labels`` to be populated (run ``cluster_spds``
          first).
        - Only low-level, per-SPD fields are copied (SPD matrices,
          IDs, tiles, latent features, labels).  Higher-level derived
          structures such as ``U_mean_list``, ``R_mean_list``,
          ``perm_list``, and ``block_dict`` are *not* copied and should
          be recomputed on the subset if needed.
        """

        if self.labels is None:
            raise ValueError("subset_data requires 'labels' to be set; run cluster_spds() first.")

        if not (0.0 < percentage <= 1.0):
            raise ValueError("percentage must be in (0, 1]. For 70%, use 0.7.")

        labels = np.asarray(self.labels)
        n = labels.shape[0]
        if n == 0:
            raise ValueError("No SPDs available to subset.")

        # Choose RNG
        rng = np.random.default_rng(random_state) if random_state is not None else np.random.default_rng()

        # Collect indices per cluster, then sample within each.
        selected_indices: List[int] = []
        unique_clusters = np.unique(labels)
        for cid in unique_clusters:
            cluster_idx = np.where(labels == cid)[0]
            if cluster_idx.size == 0:
                continue
            k = max(1, int(np.floor(cluster_idx.size * percentage)))
            # Deterministic shuffle within this cluster.
            perm = rng.permutation(cluster_idx)
            chosen = np.sort(perm[:k])
            selected_indices.extend(int(i) for i in chosen)

        if not selected_indices:
            raise ValueError("Subsampling produced an empty selection; check 'percentage' and labels.")

        selected_indices = sorted(selected_indices)

        # Build the minimal tile_stats needed by ProcessedData.__init__.
        tiles: List[QuadTile] = self.metadata["tiles"]
        genes_work = self.metadata["genes"].tolist()

        sub_tiles: List[QuadTile] = [tiles[i] for i in selected_indices]
        sub_tile_stats: List[Dict[str, Any]] = [
            {"cov": self.spd_matrices[i], "tile_id": self.spd_ids[i]} for i in selected_indices
        ]

        subset = ProcessedData(sub_tiles, sub_tile_stats, genes_work, self.num_spots)

        # Carry over basic per-SPD structures aligned with the subset.
        subset.labels = labels[selected_indices]

        # Latent features (PCA, UMAP, etc.) are kept if present and
        # indexable along the first axis.
        subset.latent = {}
        for key, arr in self.latent.items():
            try:
                subset.latent[key] = np.asarray(arr)[selected_indices]
            except Exception:
                # If the latent representation is not indexable in this
                # way, skip it; callers can recompute if needed.
                logger.warning("Skipping latent representation '%s' during subsetting.", key)

        # U_list, perm_list, R_mean_list, block_dict, etc. are left at
        # their default empty values; they should be recomputed on the
        # subset if required.

        return subset

def _apply_permutation(mat: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return mat[np.ix_(perm, perm)]


def _slice_block(mat: np.ndarray, block_start: int, block_size: int) -> np.ndarray:
    i0 = block_start
    i1 = block_start + block_size
    return mat[i0:i1, i0:i1]


def _log_spd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """SPD matrix logarithm via eigendecomposition (fast + stable for SPD)."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)  # clamp for numerical safety
    return (V * np.log(w)) @ V.T



def offline_k_means(
    items: List[Tuple[int, np.ndarray]],   # (spd_id, L_block)
    epsilon: float,
    max_iters: int = 10,
):
    """Offline, epsilon-constrained k-means-style clustering in log space.

    This performs multiple Lloyd-style iterations:

    1. Create pairwise distance matrix in log-Euclidean space.
    2. Choose k via a greedy radius-cover heuristic driven by ``epsilon``.
    3. Initialize k centers using a deterministic k-means++-style (farthest-point) seeding.
    4. Run Lloyd iterations (recompute means, reassign points), then enforce the
       ``epsilon`` constraint by splitting any over-radius cluster into
       singletons.
    """

    clusters: Dict[int, dict] = {}

    if not items:
        return clusters

    # All blocks are square of the same size.
    n = len(items)
    p = items[0][1].shape[0]
    scale = np.sqrt(p)

    # Map from index -> (spd_id, L_block) and sid -> L_block.
    sids = [int(sid) for sid, _ in items]
    logs = [L for _, L in items]
    log_map: Dict[int, np.ndarray] = {int(sid): L for sid, L in items}

    # --- Step 1: pairwise distance matrix in log space ---
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        Li = logs[i]
        for j in range(i + 1, n):
            Lj = logs[j]
            diff = Li - Lj
            d = float(np.linalg.norm(diff, ord="fro") / scale)
            D[i, j] = d
            D[j, i] = d

    radius_thr = epsilon / 2.0

    # --- Step 2: choose k via greedy radius cover (ball radius = epsilon/2) ---
    uncovered = set(range(n))
    cover_seeds: List[int] = []
    while uncovered:
        i = min(uncovered)  # deterministic pivot
        cover_seeds.append(i)
        to_remove = [j for j in uncovered if D[i, j] <= radius_thr]
        for j in to_remove:
            uncovered.discard(j)

    k = max(1, len(cover_seeds))

    # --- Step 3: deterministic k-means++-style seeding (farthest-point) ---
    centers_idx: List[int] = []
    # Start from the first cover seed for determinism.
    centers_idx.append(cover_seeds[0])

    while len(centers_idx) < k:
        best_idx = None
        best_dist = -1.0
        for i in range(n):
            if i in centers_idx:
                continue
            # distance to nearest existing center
            d_near = min(D[i, c] for c in centers_idx)
            if d_near > best_dist:
                best_dist = d_near
                best_idx = i
        if best_idx is None:
            break
        centers_idx.append(int(best_idx))

    # Initialize centers as means located at the chosen indices.
    centers = [logs[i].copy() for i in centers_idx]

    # --- Step 4: standard Lloyd iterations (without epsilon in the E-step) ---
    assignments = np.zeros(n, dtype=int)

    for it in range(max_iters):
        changed = False
        # Assign each point to the nearest center.
        for i in range(n):
            Li = logs[i]
            best_c = 0
            best_d = float("inf")
            for c_id, C in enumerate(centers):
                diff = Li - C
                d = float(np.linalg.norm(diff, ord="fro") / scale)
                if d < best_d:
                    best_d = d
                    best_c = c_id
            if assignments[i] != best_c:
                assignments[i] = best_c
                changed = True

        # Recompute centers from current assignments.
        new_centers: List[np.ndarray] = []
        for c_id in range(len(centers)):
            idxs = [i for i in range(n) if assignments[i] == c_id]
            if not idxs:
                # Keep the old center if cluster is empty.
                new_centers.append(centers[c_id])
            else:
                acc = np.zeros_like(logs[0])
                for i in idxs:
                    acc += logs[i]
                new_centers.append(acc / float(len(idxs)))

        centers = new_centers
        if not changed:
            break

    # --- Build cluster dicts and enforce epsilon via exact radii ---
    raw_clusters: Dict[int, dict] = {}
    for i in range(n):
        cid = int(assignments[i])
        sid = sids[i]
        L = logs[i]
        if cid not in raw_clusters:
            raw_clusters[cid] = {
                "spd_ids": [],
                "log_sum": np.zeros_like(L),
                "count": 0,
                "log_mean": None,
                "radius": 0.0,
            }
        c = raw_clusters[cid]
        c["spd_ids"].append(sid)
        c["log_sum"] += L
        c["count"] += 1

    # finalize mean and exact radius
    for cid, c in raw_clusters.items():
        if c["count"] == 0:
            continue
        mean = c["log_sum"] / float(c["count"])
        c["log_mean"] = mean
        max_d = 0.0
        for sid in c["spd_ids"]:
            L = log_map[sid]
            d = float(np.linalg.norm(L - mean, ord="fro") / scale)
            if d > max_d:
                max_d = d
        c["radius"] = max_d

    # Enforce epsilon: split any cluster with radius > epsilon/2 into singletons.
    final_clusters: Dict[int, dict] = {}
    cid_out = 0
    for _, c in sorted(raw_clusters.items(), key=lambda x: x[0]):
        if c["count"] == 0:
            continue
        if c["radius"] <= radius_thr:
            final_clusters[cid_out] = c
            cid_out += 1
        else:
            for sid in c["spd_ids"]:
                L = log_map[sid]
                final_clusters[cid_out] = {
                    "spd_ids": [sid],
                    "log_sum": L.copy(),
                    "count": 1,
                    "log_mean": L.copy(),
                    "radius": 0.0,
                }
                cid_out += 1

    return final_clusters

def stable_k_means(
    items: List[Tuple[int, np.ndarray]],   # (spd_id, L_block)
    epsilon: float,
    max_iters: int = 10,
):
    """Offline, epsilon-constrained k-means-style clustering in log space.

    This performs multiple Lloyd-style iterations:

    1. Initialize clusters greedily.
    2. Recompute means from all members.
    3. Reassign each SPD to the nearest cluster whose *previous* radius
       would remain ``<= epsilon / 2`` when adding that SPD (approximate
       radius test in the E-step).
    4. Rebuild clusters and exact radii from the new assignments.

    After convergence (or ``max_iters``), any cluster whose exact radius
    still exceeds ``epsilon / 2`` is conservatively split into singleton
    clusters, which trivially satisfy the epsilon constraint.

    Returns a mapping ``cluster_id -> {spd_ids, log_sum, count, log_mean, radius}``.
    """

    clusters: Dict[int, dict] = {}

    if not items:
        return clusters

    # All blocks are square of the same size.
    p = items[0][1].shape[0]
    scale = np.sqrt(p)

    # Map from spd_id to its log-block matrix for quick lookup.
    log_map: Dict[int, np.ndarray] = {int(sid): L for sid, L in items}

    sids_ordered = [int(sid) for sid, _ in items]

    # --- Helper to (re)build cluster stats from assignments ---
    def _rebuild_clusters(assignments: Dict[int, int]) -> Dict[int, dict]:
        new_clusters: Dict[int, dict] = {}
        for sid, cid in assignments.items():
            L = log_map[sid]
            if cid not in new_clusters:
                new_clusters[cid] = {
                    "spd_ids": [],
                    "log_sum": np.zeros_like(L),
                    "count": 0,
                    "log_mean": None,
                    "radius": 0.0,
                }
            c = new_clusters[cid]
            c["spd_ids"].append(sid)
            c["log_sum"] += L
            c["count"] += 1

        # finalize means and radii
        for cid, c in new_clusters.items():
            if c["count"] == 0:
                continue
            mean = c["log_sum"] / float(c["count"])
            c["log_mean"] = mean
            # exact radius
            max_d = 0.0
            for sid in c["spd_ids"]:
                L = log_map[sid]
                d = float(np.linalg.norm(L - mean, ord="fro") / scale)
                if d > max_d:
                    max_d = d
            c["radius"] = max_d
        return new_clusters

    # --- Initialization: greedy seeding (single pass) ---
    assignments: Dict[int, int] = {}
    next_cid = 0

    for sid, L in items:
        sid_int = int(sid)
        if not clusters:
            clusters[next_cid] = {
                "spd_ids": [sid_int],
                "log_sum": L.copy(),
                "count": 1,
                "log_mean": L.copy(),
                "radius": 0.0,
            }
            assignments[sid_int] = next_cid
            next_cid += 1
            continue

        # Try existing clusters by current mean.
        best_cid = None
        best_dist = float("inf")
        for cid, c in clusters.items():
            mean = c["log_mean"]
            diff = L - mean
            dist = float(np.linalg.norm(diff, ord="fro") / scale)
            # conservative radius check using current mean
            new_radius_est = max(float(c.get("radius", 0.0)), dist)
            if new_radius_est <= epsilon / 2.0 and dist < best_dist:
                best_dist = dist
                best_cid = cid

        if best_cid is None:
            # start new cluster
            cid = next_cid
            next_cid += 1
            clusters[cid] = {
                "spd_ids": [sid_int],
                "log_sum": L.copy(),
                "count": 1,
                "log_mean": L.copy(),
                "radius": 0.0,
            }
            assignments[sid_int] = cid
        else:
            c = clusters[best_cid]
            c["spd_ids"].append(sid_int)
            c["log_sum"] += L
            c["count"] += 1
            c["log_mean"] = c["log_sum"] / float(c["count"])
            # update radius lower bound for initialization
            diff_new = L - c["log_mean"]
            d_new = float(np.linalg.norm(diff_new, ord="fro") / scale)
            c["radius"] = max(float(c.get("radius", 0.0)), d_new)
            assignments[sid_int] = best_cid

    # --- Offline refinement: Lloyd-style iterations ---
    for _ in range(max_iters):
        # Rebuild clusters to get exact means and radii.
        clusters = _rebuild_clusters(assignments)

        prev_assignments = assignments.copy()
        assignments = {}

        # Freeze cluster stats for this E-step.
        frozen_stats = {
            cid: {
                "mean": c["log_mean"],
                "radius": float(c.get("radius", 0.0)),
            }
            for cid, c in clusters.items()
            if c["count"] > 0
        }

        # Existing cluster ids are reused; new ones (if needed) are added above the max.
        if frozen_stats:
            next_cid = max(frozen_stats.keys()) + 1
        else:
            next_cid = 0

        for sid in sids_ordered:
            L = log_map[sid]
            best_cid = None
            best_dist = float("inf")

            # Consider only frozen clusters from previous iteration.
            for cid, info in frozen_stats.items():
                mean = info["mean"]
                if mean is None:
                    continue
                diff = L - mean
                dist = float(np.linalg.norm(diff, ord="fro") / scale)
                new_radius_est = max(info["radius"], dist)
                if new_radius_est <= epsilon / 2.0 and dist < best_dist:
                    best_dist = dist
                    best_cid = cid

            if best_cid is None:
                # No acceptable existing cluster; open a new one.
                cid = next_cid
                next_cid += 1
                assignments[sid] = cid
            else:
                assignments[sid] = best_cid

        if assignments == prev_assignments:
            break

    # Final rebuild with exact radii.
    clusters = _rebuild_clusters(assignments)

    # Enforce epsilon strictly by splitting any cluster whose radius
    # still exceeds epsilon/2 into singletons.
    final_clusters: Dict[int, dict] = {}
    cid_out = 0
    for _, c in sorted(clusters.items(), key=lambda x: x[0]):
        if c["count"] == 0:
            continue
        if c["radius"] <= epsilon / 2.0:
            final_clusters[cid_out] = c
            cid_out += 1
        else:
            # Split into singletons.
            for sid in c["spd_ids"]:
                L = log_map[sid]
                final_clusters[cid_out] = {
                    "spd_ids": [sid],
                    "log_sum": L.copy(),
                    "count": 1,
                    "log_mean": L.copy(),
                    "radius": 0.0,
                }
                cid_out += 1

    return final_clusters
    


def refine_block_clusters_phase2_delete_empty(
    items: List[Tuple[int, np.ndarray]],   # (spd_id, L_block)
    clusters: Dict[int, dict],             # cluster_within_block from phase1
    *,
    p: int,
    block_index: int,
    max_iters: int = 5,
    tol: float = 1e-2,
    normalize: bool = True,
    debug: bool = False
):
    """
    Phase 2: Lloyd refinement in log space.
    - Reassign each item to nearest log-mean.
    - Recompute log-means.
    - Repeat a few iterations.
    - Delete empty clusters and renumber ids to 0..K-1 at the end.
    """

    scale = np.sqrt(p) if normalize else 1.0
    n = len(items)

    # stable initial ordering of cluster ids
    old_ids = sorted(clusters.keys())

    # initialize means
    mean_log = {cid: clusters[cid]["log_mean"] for cid in old_ids}

    # build initial assignment from phase1 labels if present
    id_to_idx = {spd_id: i for i, (spd_id, _) in enumerate(items)}
    assign = np.full(n, -1, dtype=int)
    for cid in old_ids:
        for spd_id in clusters[cid]["spd_ids"]:
            if spd_id in id_to_idx:
                assign[id_to_idx[spd_id]] = cid

    # if any unassigned (shouldn't happen), assign to nearest mean once
    if np.any(assign == -1):
        for i, (_, L) in enumerate(items):
            if assign[i] != -1:
                continue
            best_cid, best_d = None, float("inf")
            for cid in old_ids:
                d = float(np.linalg.norm(L - mean_log[cid], ord="fro") / scale)
                if d < best_d:
                    best_d, best_cid = d, cid
            assign[i] = int(best_cid)  # type: ignore[arg-type]

    

    for it in range(max_iters):
        # ----- reassignment -----
        new_assign = assign.copy()
        for i, (_, L) in enumerate(items):
            best_cid, best_d = None, float("inf")
            for cid in old_ids:
                d = float(np.linalg.norm(L - mean_log[cid], ord="fro") / scale)
                if d < best_d:
                    best_d, best_cid = d, cid
            new_assign[i] = int(best_cid)  # type: ignore[arg-type]

        changed = int(np.sum(new_assign != assign))
        assign = new_assign

        # ----- recompute means -----
        # accumulate
        sum_log = {cid: np.zeros_like(next(iter(mean_log.values()))) for cid in old_ids}
        count = {cid: 0 for cid in old_ids}
        members = {cid: [] for cid in old_ids}

        for i, (spd_id, L) in enumerate(items):
            # TODO: remove the line below
            debug_print = False
            if debug_print:
                print(f"{spd_id} spd got assigned to cluster {cid} in phase 2")
            cid = int(assign[i])
            members[cid].append(spd_id)
            sum_log[cid] += L
            count[cid] += 1

        # update means; track max movement
        max_move = 0.0
        for cid in old_ids:
            if count[cid] == 0:
                continue  # leave mean_log as-is for now; we’ll delete empties at end
            new_mean = sum_log[cid] / float(count[cid])
            max_move = max(max_move, float(np.linalg.norm(new_mean - mean_log[cid], ord="fro")))
            mean_log[cid] = new_mean
            # write back
            clusters[cid]["spd_ids"] = members[cid]
            clusters[cid]["log_sum"] = sum_log[cid]
            clusters[cid]["count"] = count[cid]
            clusters[cid]["log_mean"] = new_mean

        if changed == 0 or max_move < tol:
            if debug:
                logger.info(f"Converged in {block_index} after {it} iterations: changed={changed}, max_move={max_move:.6f}")
            break

    # ----- delete empty clusters -----
    nonempty_old_ids = [cid for cid in old_ids if clusters[cid]["count"] > 0]

    # ----- renumber to 0..K-1 -----
    new_clusters: Dict[int, dict] = {}
    old_to_new = {old_cid: new_cid for new_cid, old_cid in enumerate(nonempty_old_ids)}

    for old_cid in nonempty_old_ids:
        new_cid = old_to_new[old_cid]
        new_clusters[new_cid] = clusters[old_cid]

    return new_clusters, old_to_new


def refine_block_clusters_phase2_with_epsilon_constraint(
    items: List[Tuple[int, np.ndarray]],   # (spd_id, L_block)
    clusters: Dict[int, dict],             # cluster_within_block from phase1
    *,
    p: int,
    block_index: int,
    epsilon: float,
    max_iters: int = 5,
    tol: float = 1e-2,
    normalize: bool = True,
    debug: bool = False
):
    """Phase 2: Lloyd refinement in log space with an epsilon radius guard.

    We perform a few rounds of standard Lloyd reassignment + mean updates
    starting from the phase-1 clusters, but *only* accept the refined
    clusters if every cluster's radius (max distance from mean) is
    ``<= epsilon``. If any refined cluster violates this radius
    constraint, we revert to the original phase-1 clusters.
    """

    import copy

    scale = np.sqrt(p) if normalize else 1.0
    n = len(items)

    # Keep a pristine copy of the phase-1 clusters so we can revert.
    phase1_clusters = copy.deepcopy(clusters)

    # Work on a separate copy so callers' dict is never mutated.
    clusters_work = copy.deepcopy(phase1_clusters)

    # stable initial ordering of cluster ids
    old_ids = sorted(clusters_work.keys())

    # initialize means
    mean_log = {cid: clusters_work[cid]["log_mean"] for cid in old_ids}

    # build initial assignment from phase1 labels if present
    id_to_idx = {spd_id: i for i, (spd_id, _) in enumerate(items)}
    assign = np.full(n, -1, dtype=int)
    for cid in old_ids:
        for spd_id in clusters_work[cid]["spd_ids"]:
            if spd_id in id_to_idx:
                assign[id_to_idx[spd_id]] = cid

    # if any unassigned (shouldn't happen), assign to nearest mean once
    if np.any(assign == -1):
        for i, (_, L) in enumerate(items):
            if assign[i] != -1:
                continue
            best_cid, best_d = None, float("inf")
            for cid in old_ids:
                d = float(np.linalg.norm(L - mean_log[cid], ord="fro") / scale)
                if d < best_d:
                    best_d, best_cid = d, cid
            assign[i] = int(best_cid)  # type: ignore[arg-type]

    # Precompute mapping from spd_id to its log-block for radius checks.
    id_to_log = {int(spd_id): L for spd_id, L in items}

    # Lloyd refinement on the working copy.
    for it in range(max_iters):
        # ----- reassignment -----
        new_assign = assign.copy()
        for i, (_, L) in enumerate(items):
            best_cid, best_d = None, float("inf")
            for cid in old_ids:
                d = float(np.linalg.norm(L - mean_log[cid], ord="fro") / scale)
                if d < best_d:
                    best_d, best_cid = d, cid
            new_assign[i] = int(best_cid)  # type: ignore[arg-type]

        changed = int(np.sum(new_assign != assign))
        assign = new_assign

        # ----- recompute means -----
        # accumulate
        sum_log = {cid: np.zeros_like(next(iter(mean_log.values()))) for cid in old_ids}
        count = {cid: 0 for cid in old_ids}
        members = {cid: [] for cid in old_ids}

        for i, (spd_id, L) in enumerate(items):
            cid = int(assign[i])
            members[cid].append(int(spd_id))
            sum_log[cid] += L
            count[cid] += 1

        # update means; track max movement
        max_move = 0.0
        for cid in old_ids:
            if count[cid] == 0:
                continue  # leave mean_log as-is for now; we’ll delete empties at end
            new_mean = sum_log[cid] / float(count[cid])
            max_move = max(max_move, float(np.linalg.norm(new_mean - mean_log[cid], ord="fro")))
            mean_log[cid] = new_mean
            # write back into working clusters
            clusters_work[cid]["spd_ids"] = members[cid]
            clusters_work[cid]["log_sum"] = sum_log[cid]
            clusters_work[cid]["count"] = count[cid]
            clusters_work[cid]["log_mean"] = new_mean

        if changed == 0 or max_move < tol:
            if debug:
                logger.info(
                    f"Converged in block {block_index} after {it} iterations: "
                    f"changed={changed}, max_move={max_move:.6f}"
                )
            break

    # ----- compute exact radii in the working clusters -----
    max_radius = 0.0
    for cid in old_ids:
        c = clusters_work[cid]
        if c["count"] == 0:
            continue
        mean = c["log_mean"]
        r = 0.0
        for spd_id in c["spd_ids"]:
            L = id_to_log[int(spd_id)]
            d = float(np.linalg.norm(L - mean, ord="fro") / scale)
            if d > r:
                r = d
        c["radius"] = r
        if r > max_radius:
            max_radius = r

    # Decide whether to accept the refinement.
    if max_radius > epsilon:
        if debug:
            logger.info(
                f"Rejecting phase-2 refinement for block {block_index}: "
                f"max radius {max_radius:.4f} exceeds epsilon={epsilon:.4f}. "
                "Reverting to phase-1 clusters."
            )
        clusters_final = phase1_clusters
    else:
        clusters_final = clusters_work

    # ----- delete empty clusters -----
    nonempty_old_ids = [cid for cid in sorted(clusters_final.keys()) if clusters_final[cid]["count"] > 0]

    # ----- renumber to 0..K-1 -----
    new_clusters: Dict[int, dict] = {}
    old_to_new = {old_cid: new_cid for new_cid, old_cid in enumerate(nonempty_old_ids)}

    for old_cid in nonempty_old_ids:
        new_cid = old_to_new[old_cid]
        new_clusters[new_cid] = clusters_final[old_cid]

    return new_clusters, old_to_new


def _fro_dist_norm(A, B, p):
    return float(np.linalg.norm(A - B, ord="fro") / np.sqrt(p))


from collections import defaultdict


def farthest_first_eps_for_k(L_list, p, k_target, seed=0):
    """
    Return epsilon (cover radius) achieved by farthest-first with k_target centers,
    plus the delta curve if you want to inspect it.

    L_list: list/array of (p,p) log-block matrices
    distance: normalized Frobenius on logs
    """
    n = len(L_list)
    if n == 0:
        return 0.0, []

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, n))
    centers = [first]

    # delta[i] = distance to nearest selected center so far
    delta = np.array([_fro_dist_norm(L_list[i], L_list[first], p) for i in range(n)], dtype=float)
    delta[first] = 0.0

    deltas_max = [float(delta.max())]  # coverage radius after 1 center

    # add centers until k_target or fully covered
    while len(centers) < min(k_target, n):
        i_star = int(np.argmax(delta))
        if delta[i_star] <= 0.0:  # fully covered (numerical)
            break
        centers.append(i_star)

        c_new = L_list[i_star]
        for i in range(n):
            d = _fro_dist_norm(L_list[i], c_new, p)
            if d < delta[i]:
                delta[i] = d

        deltas_max.append(float(delta.max()))

    # epsilon = achieved radius with k_target centers (or fewer if n < k_target)
    eps = deltas_max[min(len(deltas_max), k_target) - 1]
    return eps, deltas_max


def choose_eps_from_curve(deltas_max, mode="elbow", quantile=0.9):
    """
    Optional if you don't want k_target. Simple heuristics:
    - elbow: pick t where relative drop slows (very rough)
    - quantile: pick a quantile of deltas_max (monotone decreasing curve)
    """
    if not deltas_max:
        return 0.0

    if mode == "quantile":
        # deltas_max decreases with t; quantile of values corresponds to some t
        return float(np.quantile(np.array(deltas_max), quantile))

    # crude elbow: maximize second finite difference on log scale
    y = np.log(np.array(deltas_max) + 1e-12)
    
    if len(y) < 3:
        return float(deltas_max[-1])
    second = (y[:-2] - 2*y[1:-1] + y[2:])  # curvature
    t = int(np.argmax(second)) + 2  # elbow around this index
    # plt.plot(deltas_max)
    # plt.axvline(x=t, color='r', linestyle='--')
    return float(deltas_max[t-1])

from collections import defaultdict
def _log_spd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """SPD matrix logarithm via eigendecomposition (fast + stable for SPD)."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)  # clamp for numerical safety
    return (V * np.log(w)) @ V.T

def choose_adaptive_epsilons(data: ProcessedData, cluster_id: int, k_target_per_block):
    """Choose adaptive epsilons for each block in the given cluster."""
    import numpy as np
    from collections import defaultdict

    indices = np.where(data.labels == cluster_id)[0]
    block_runs = data.block_dict[cluster_id]
    permutation = data.perm_list[cluster_id]
    spd_matrices = np.stack([data.spd_matrices[i] for i in indices])
    # Precompute log-blocks for sampled matrices (avoid pairwise loops)
    block_log = defaultdict(list)   # block_idx -> list of L_block (p,p)
    block_p = {}                    # block_idx -> p

    for i in range(len(spd_matrices)):
        spd_i = spd_matrices[i]
        for block_idx, (block_start, block_end) in enumerate(block_runs):
            idx = permutation[block_start:block_end]
            A_blk = spd_i[np.ix_(idx, idx)]
            p_blk = A_blk.shape[0]
            block_p[block_idx] = p_blk
            block_log[block_idx].append(_log_spd(A_blk))
    eps_per_block = {}
    eps_elbow_per_block = {}
    k_target_per_block = 16  # <-- put in config, or function of block size

    for block_idx, L_list in block_log.items():
        p_blk = block_p[block_idx]
        eps_hat, deltas_max = farthest_first_eps_for_k(L_list, p_blk, k_target=k_target_per_block, seed=42)
        eps_elbow = choose_eps_from_curve(deltas_max, mode="elbow")
        eps_per_block[block_idx] = eps_hat
        eps_elbow_per_block[block_idx] = eps_elbow
    total_eps = 0.0
    for blc_idx in range(len(block_runs)):
        total_eps += (eps_elbow_per_block[blc_idx] ** 2) * block_p[blc_idx]
    eps = np.sqrt(total_eps / sum(block_p.values()))
    return eps_per_block, eps_elbow_per_block, eps


def index_spds(data: ProcessedData, config: IndexConfig):
    """Build an index over SPD sub-matrices derived from spatial data.

    This is a placeholder function. The detailed implementation should
    follow the algorithm specified in .github/copilot-instructions.md.
    """
    # Placeholder: in a complete implementation, the steps described in
    # the algorithm spec would be carried out here. For now, we construct
    # an empty index that can be populated by downstream code.
    
    
    spd_matrices = np.asarray(data.spd_matrices)
    debug = config.debug
    global_dist_list: List[Tuple[int, float]] = []
    # Also store the number of clusters created per block per cluster
    stats = []
    # global tree
    # tree_block = {}
    # # global dictionary of block_id, clust_id -> node_id
    # look_ahead_map = {}
    dag_dict: Dict[int, IndexHandle] = {}
    for cluster_id in set(data.labels):
        logger.info("Processing cluster %d", cluster_id)
        
        epsilon = config.epsilon_dict[cluster_id]
        logger.info("Building SPD index with epsilon=%s", epsilon)
        logger.info("Step 1: Cluster blocks within each class of SPD matrices.")
        
        spd_matrices_cluster = spd_matrices[data.labels == cluster_id]
        spd_ids_cluster = np.array(data.spd_ids)[data.labels == cluster_id]
        permutation = data.perm_list[cluster_id]
        block_runs = data.block_dict[cluster_id]
        num_blocks = len(block_runs)
        logger.info(f"Cluster {cluster_id}: {len(spd_matrices_cluster)} SPDs, {num_blocks} blocks")
        tree_block_clusters = []
        # local dictionary of block_id, clust_id -> node_id
        local_look_ahead_map: Dict[Tuple[int, int], int] = {}
        node_counter = 0
        # measure time
        
        import time 
        for block_index, (block_start, block_end) in enumerate(block_runs):
            start_time = time.time()
            # logger.info("Processing block %d of size %d", block_index, block_end - block_start)
            if config.threshold_type == 'block_wise':
                try:
                    epsilon = config.epsilon_block_wise_dict[cluster_id][block_index]
                except KeyError:
                    logger.error(f"Block-wise epsilon not found for cluster {cluster_id}, block {block_index}. Using default epsilon {config.epsilon_dict[cluster_id]}")
            
            block_size = block_end - block_start
            # Extract block sub-matrices for all SPDs in this cluster and
            # cluster them greedily based on log-Euclidean distance between
            # block means.

            # For each block we maintain a mapping from cluster id to a
            # small state dict:
            #   {
            #       "spd_ids": [int, ...],   # SPD ids assigned to this block-cluster
            #       "sum": np.ndarray,       # running sum of block-SPD matrices
            #       "count": int,            # number of members
            #       "mean": np.ndarray,      # current mean block-SPD
            #   }
            # The BlockClusterMetadata built later will reconstruct
            # (spd_id, block_index) pairs from these spd_ids.
            log_spd_cache: List[Tuple[int, np.ndarray]] = []
            cluster_within_block = {}
            # Make adaptive epsilon smaller for small blocks
            choose_epsilon_strategy = None
            if choose_epsilon_strategy == 'sampling':
                # Create a small sample of random SPD matrices of size block_size
                num_sample = min(50, len(spd_matrices_cluster))
                # randomly sample SPD matrices
                sample_indices = np.random.choice(len(spd_matrices_cluster), num_sample, replace=False)
                sample_dists = []
                for i in sample_indices:
                    C = spd_matrices_cluster[i]
                    C_block = C[np.ix_(permutation[block_start:block_end], permutation[block_start:block_end])]
                    L_block = _log_spd(C_block)
                    for j in sample_indices:
                        if i >= j:
                            continue
                        C2 = spd_matrices_cluster[j]
                        C2_block = C2[np.ix_(permutation[block_start:block_end], permutation[block_start:block_end])]
                        L2_block = _log_spd(C2_block)
                        diff = L_block - L2_block
                        dist = np.linalg.norm(diff, ord='fro') / np.sqrt(block_size)
                        sample_dists.append(dist)
            
                if len(sample_dists) > 0:
                    adaptive_epsilon = np.percentile(sample_dists, 50)
                else:
                    adaptive_epsilon = epsilon
            elif choose_epsilon_strategy == 'scaling':
                adaptive_epsilon = epsilon * np.sqrt(block_size) 
                if adaptive_epsilon < epsilon:
                    logger.info(f" Using adaptive epsilon={adaptive_epsilon:.4f} for block {block_index} of size {block_size}")
                else:
                    adaptive_epsilon = epsilon
            else:
                adaptive_epsilon = epsilon

            if config.kmean_method == 'stable' or config.kmean_method == 'offline':
                logger.info(" Using stable k-means clustering for block %d", block_index)
                # Precompute all block sub-matrices in log space
                for i, C in enumerate(spd_matrices_cluster):
                    C_block = C[np.ix_(permutation[block_start:block_end], permutation[block_start:block_end])]
                    # p = C_block.shape[0]
                    L_block = _log_spd(C_block) # log(C_block) expensive operation
                    spd_id = spd_ids_cluster[i]
                    log_spd_cache.append((spd_id, L_block))
                # Run stable k-means
                if config.kmean_method == 'stable':
                    cluster_within_block = stable_k_means(
                        log_spd_cache,
                        epsilon=adaptive_epsilon,
                        max_iters=config.max_iter,
                    )
                elif config.kmean_method == 'offline':
                    cluster_within_block = offline_k_means(
                        log_spd_cache,
                        epsilon=adaptive_epsilon,
                        max_iters=config.max_iter,
                    )
                else:
                    raise ValueError(f"Unknown kmean_method: {config.kmean_method}")
                # Map local cluster ids to global node ids
                for local_clust_id in cluster_within_block.keys():
                    local_look_ahead_map[(block_index, local_clust_id)] = node_counter
                    node_counter += 1
            elif config.kmean_method == 'online':
                for i, C in enumerate(spd_matrices_cluster):
                    # Extract block sub-matrix
                    C_block = C[np.ix_(permutation[block_start:block_end], permutation[block_start:block_end])]
                    p = C_block.shape[0]
                    L_block = _log_spd(C_block) # log(C_block) expensive operation
                    spd_id = spd_ids_cluster[i]
                    log_spd_cache.append((spd_id, L_block))
                    # TODO: remove the line below
                    debug_print = (spd_id == 143) and debug
                    #print(f" SPD id: {spd_id}, block shape: {C_block.shape}")
                    if not cluster_within_block:
                        # First cluster in this block.
                        cluster_within_block[0] = {
                            "spd_ids": [spd_id],
                            # "sum": C_block.copy(),
                            "log_sum": L_block.copy(),
                            "count": 1,
                            # "mean": C_block.copy(),
                            "log_mean": L_block.copy(),
                            "radius": 0.0,
                        }
                        if debug_print:
                            print(f"Created {spd_id} is assigned to 0 within block {block_index} with node_id {node_counter}")
                        # print(f" Created first cluster 0 for block {block_index} -> node_id {node_counter}")
                        local_look_ahead_map[(block_index, 0)] = node_counter
                        node_counter += 1
                        continue

                    # compute distance to existing cluster means
                    dists: List[Tuple[int, float]] = []
                    for clust_id, clust in cluster_within_block.items():
                        mean_mat = clust["log_mean"]  # type: ignore[assignment]
                        # dist = log_euclidean_distance(C_block, mean_mat, normalize=True)  # type: ignore[arg-type]
                        diff = L_block - mean_mat
                        dist = np.linalg.norm(diff, ord='fro') / np.sqrt(p)
                        dists.append((clust_id, dist))
                        global_dist_list.append((cluster_id, dist))
                    dists.sort(key=lambda x: x[1])
                    closest_clust_id, closest_dist = dists[0]
                    clust = cluster_within_block[closest_clust_id]
                    use_strict_pairwise = True
                    if use_strict_pairwise:
                        r = float(clust.get("radius", 0.0))
                        accept = closest_dist <= (adaptive_epsilon - r) / 2.0
                    else:
                        accept = closest_dist < adaptive_epsilon

                    if accept:
                        # assign to this cluster and update running mean
                        clust = cluster_within_block[closest_clust_id]
                        # clust_spd_ids = clust["spd_ids"]  # type: ignore[assignment]
                        # clust_sum = clust["sum"]          # type: ignore[assignment]
                        # count = clust["count"]            # type: ignore[assignment]

                        # clust_spd_ids.append(spd_id)       # type: ignore[arg-type]
                        # clust_sum += C_block               # type: ignore[operator]
                        # count += 1
                        # clust["count"] = count
                        # clust["mean"] = clust_sum / float(count)
                        if debug_print:
                            print(f"Appended {spd_id} is assigned to {closest_clust_id} within block {block_index} with node_id {local_look_ahead_map[(block_index, closest_clust_id)]} with dist: {closest_dist} with radius {r}")

                        clust["spd_ids"].append(spd_id)  # type: ignore[arg-type]
                        clust["log_sum"] += L_block      # type: ignore[operator]
                        clust["count"] += 1              # type: ignore[operator]
                        mean_log_new = clust["log_sum"] / float(clust["count"])
                        clust["log_mean"] = mean_log_new
                        if use_strict_pairwise:
                            # update radius lower-bounded by this member's distance to new mean
                            # (exact max would require storing all member logs; this is a safe/cheap proxy)
                            diff_new = L_block - mean_log_new
                            d_new = float(np.linalg.norm(diff_new, ord="fro") / np.sqrt(p))
                            clust["radius"] = max(float(clust.get("radius", 0.0)), d_new)
                    else: 
                        # create new cluster
                        new_clust_id = len(cluster_within_block)
                        cluster_within_block[new_clust_id] = {
                                "spd_ids": [spd_id],
                                # "sum": C_block.copy(),
                                "log_sum": L_block.copy(),
                                "count": 1,
                                # "mean": C_block.copy(),
                                "log_mean": L_block.copy(),
                                "radius": 0.0,
                        }
                        if debug_print:
                            print(f"Created {spd_id} is assigned to {new_clust_id} within block {block_index} with node_id {node_counter}")
                        # print(f" Created new cluster {new_clust_id} for block {block_index} -> node_id {node_counter}")
                        local_look_ahead_map[(block_index, new_clust_id)] = node_counter
                        node_counter += 1

                do_phase_two_lloyd = config.stabilize_block_cluster
                max_iters = config.max_iter
                if do_phase_two_lloyd:
                    cluster_within_block, old_to_new = refine_block_clusters_phase2_with_epsilon_constraint(
                        items=log_spd_cache,
                        clusters=cluster_within_block,
                        p=block_size,
                        block_index=block_index,
                        epsilon=adaptive_epsilon,
                        max_iters=max_iters,
                        tol=1e-2,
                        normalize=True,
                        debug=debug
                    )
                    new_local_look_ahead_map = {}
                    for (bidx, old_cid), node_id in local_look_ahead_map.items():
                        if bidx != block_index:
                            new_local_look_ahead_map[(bidx, old_cid)] = node_id
                            continue
                        if old_cid in old_to_new:
                            new_local_look_ahead_map[(block_index, old_to_new[old_cid])] = node_id
                    local_look_ahead_map = new_local_look_ahead_map
            elif config.kmean_method == 'epsilon_net':
                logger.info(" Using epsilon-net clustering for block %d", block_index)
                keep_farthest_distances = []
                L_list = []
                for i, C in enumerate(spd_matrices_cluster):
                    C_block = C[np.ix_(permutation[block_start:block_end], permutation[block_start:block_end])]
                    p = C_block.shape[0]
                    L_block = _log_spd(C_block)
                    spd_id = spd_ids_cluster[i]
                    log_spd_cache.append((spd_id, L_block))
                    L_list.append(L_block)
                n = len(L_list)
                assert n == len(log_spd_cache)

                centers_idx = [0]  # pick any start (0). you can randomize if you want.
                delta = np.empty(n, dtype=float)
                for i in range(n):
                    delta[i] = _fro_dist_norm(L_list[i], L_list[0], block_size)
                while True:
                    i_star = int(np.argmax(delta))
                    keep_farthest_distances.append(delta[i_star])
                    if delta[i_star] <= epsilon:
                        break
                    centers_idx.append(i_star)
                    c_new = L_list[i_star]
                    # update nearest-center distance cache
                    for i in range(n):
                        d = _fro_dist_norm(L_list[i], c_new, block_size)
                        if d < delta[i]:
                            delta[i] = d
                cluster_within_block = {}
                for k, ci in enumerate(centers_idx):
                    Lc = L_list[ci]
                    spd_id_c = spd_ids_cluster[ci]
                    cluster_within_block[k] = {
                        "spd_ids": [],                # filled below
                        "center": Lc.copy(),          # fixed representative (epsilon-net center)
                        "log_sum": np.zeros_like(Lc),
                        "count": 0,
                        "log_mean": np.zeros_like(Lc),
                        "actual_log_mean": np.zeros_like(Lc),  # optional: keep the actual mean same as center
                        "radius": 0.0,
                        "center_spd_id": spd_id_c,    # optional: keep which SPD was the rep
                    }
                
                for i in range(n):
                    Li = L_list[i]
                    spd_id = spd_ids_cluster[i]

                    # nearest fixed center
                    best_k, best_d = None, float("inf")
                    for k, ci in enumerate(centers_idx):
                        d = _fro_dist_norm(Li, L_list[ci], p)
                        if d < best_d:
                            best_d, best_k = d, k

                    # By construction, best_d should be <= epsilon (except numerical issues)
                    if best_d > epsilon + 1e-12:
                        logger.warning(f"[WARN] spd_id={spd_id} best_d={best_d:.6f} > epsilon={epsilon:.6f}")

                    clust = cluster_within_block[best_k]
                    clust["spd_ids"].append(spd_id)
                    clust["log_sum"] += Li
                    clust["count"] += 1
                    clust["radius"] = max(float(clust["radius"]), float(best_d))

                # finalize log_means
                for k, clust in cluster_within_block.items():
                    if clust["count"] > 0:
                        clust["actual_log_mean"] = clust["log_sum"] / float(clust["count"])
                        clust["log_mean"] = clust["center"]

                for local_clust_id in cluster_within_block.keys():
                    local_look_ahead_map[(block_index, local_clust_id)] = node_counter
                    node_counter += 1
            else:
                raise ValueError(f"Unknown kmean_method: {config.kmean_method}")

            tree_block_clusters.append(cluster_within_block)
            end_time = time.time()
            stats.append({'epsilon': adaptive_epsilon, 'cluster_id': cluster_id, 'block_index': block_index, 'num_clusters': len(cluster_within_block), 'block_size': block_size})
            logger.info(f" Finished block {block_index} in {end_time - start_time:.2f} seconds, found {len(cluster_within_block)} clusters.")
        logger.info("Step 2: Build DAG connections between block clusters.")
        # Build DAG connections between block clusters
        logger.info("Step 2.1: For each layer order the block-clusters by")
        logger.info("Not implemented: ordering block-clusters ? How to order them?")
        logger.info("We will use triangle inequality to order clusters.")
        logger.info("Step 2.2: Connect block-clusters between layers based on co-occurrence in SPDs.")
        nodes: List[BlockClusterNode] = []
        block_to_nodes: Dict[int, List[int]] = {}
        for layer, block_cluster_dict in enumerate(tree_block_clusters[:-1]):
            next_block_cluster_dict = tree_block_clusters[layer + 1]
            # build a mapping from spd_id to block cluster ids in next layer
            spd_to_next_clusters: Dict[int, List[int]] = {}
            for next_clust_id, next_clust in next_block_cluster_dict.items():
                for spd_id in next_clust["spd_ids"]:  # type: ignore[index]
                    if spd_id not in spd_to_next_clusters:
                        spd_to_next_clusters[spd_id] = []
                    next_global_id = local_look_ahead_map[(layer + 1, next_clust_id)]
                    spd_to_next_clusters[spd_id].append(next_global_id)
            # Now connect current layer clusters to next layer clusters
            for clust_id, clust in block_cluster_dict.items():
                global_node_id = local_look_ahead_map[(layer, clust_id)]
                node_id = len(nodes)
                # Rebuild (spd_id, block_index) membership tuples for metadata
                member_spd_ids = clust["spd_ids"]  # type: ignore[index]
                metadata_members: List[Tuple[int, int]] = [
                    (spd_id, layer) for spd_id in member_spd_ids  # type: ignore[assignment]
                ]
                if config.kmean_method == 'epsilon_net':
                    metadata = BlockClusterMetadata(
                        members=metadata_members,
                        mean=clust["log_mean"],  # type: ignore[arg-type]
                        representative_mean=clust["actual_log_mean"],  # type: ignore[arg-type]
                        radius=clust["radius"]
                    )
                metadata = BlockClusterMetadata(
                    members=metadata_members,
                    mean=clust["log_mean"],  # type: ignore[arg-type]
                    representative_mean=clust["log_mean"],  # type: ignore[arg-type]
                    radius=clust["radius"]
                )
                children_set = set()
                for spd_id in member_spd_ids:
                    if spd_id in spd_to_next_clusters:
                        children_set.update(spd_to_next_clusters[spd_id])
                children_list = list(children_set)
                node = BlockClusterNode(
                    global_node_id=global_node_id,
                    node_id=node_id,
                    block_index=layer,
                    block_cluster_id=clust_id,
                    order_id=clust_id,  # Placeholder for order_id
                    metadata=metadata,
                    children=children_list
                )
                nodes.append(node)
        # Create nodes for the last layer (leaf nodes)
        last_layer = len(tree_block_clusters) - 1
        last_block_cluster_dict = tree_block_clusters[last_layer]
        for clust_id, clust in last_block_cluster_dict.items():
            global_node_id = local_look_ahead_map[(last_layer, clust_id)]
            node_id = len(nodes)
            # Rebuild (spd_id, block_index) membership tuples for metadata
            member_spd_ids = clust["spd_ids"]  # type: ignore[index]
            metadata_members: List[Tuple[int, int]] = [
                (spd_id, last_layer) for spd_id in member_spd_ids  # type: ignore[assignment]
            ]
            if config.kmean_method == 'epsilon_net':
                metadata = BlockClusterMetadata(
                    members=metadata_members,
                    mean=clust["log_mean"],  # type: ignore[arg-type]
                    representative_mean=clust["actual_log_mean"],  # type: ignore[arg-type]
                    radius=clust["radius"]
                )
            else:
                metadata = BlockClusterMetadata(
                    members=metadata_members,
                    mean=clust["log_mean"],  # type: ignore[arg-type]
                    representative_mean=clust["log_mean"],  # type: ignore[arg-type]
                    radius=clust["radius"]
                )
            # Leaf nodes have no children
            children_list: List[int] = []
            node = BlockClusterNode(
                global_node_id=global_node_id,
                node_id=node_id,
                block_index=last_layer,
                block_cluster_id=clust_id,
                order_id=clust_id,  # Placeholder for order_id
                metadata=metadata,
                children=children_list
            )
            nodes.append(node)

        # Populate block_to_nodes: map each block index to the list of node_ids
        # in that block.
        logger.info("Check if node global_node_id matches index in nodes list")
        for i, node in enumerate(nodes):
            assert node.global_node_id == i, f"Node at index {i} has global_node_id {node.global_node_id}, expected {i}"
            block_to_nodes.setdefault(node.block_index, []).append(node.node_id)

        # Use the log-Euclidean distance (a metric satisfying the
        # triangle inequality) to define a deterministic 1D order of
        # block-clusters within each layer.  For each block index, we:
        #   1. Pick a reference cluster (smallest node_id).
        #   2. Compute its distance to every cluster mean in that block.
        #   3. Sort clusters by this distance and assign consecutive
        #      order_id values.
        #   4. Sort block_to_nodes[b_idx] by the new order_id.
        
        logger.info("Step 2.3: Ordering block-clusters within each layer using log-Euclidean distances.")
        if config.is_ordered:
            for b_idx, node_ids in block_to_nodes.items():
                # Deterministic reference: node with smallest node_id in this block.
                ref_nid = min(node_ids)
                ref_mean = nodes[ref_nid].metadata.mean

                # Compute distances from reference to each cluster mean.
                dist_list: List[Tuple[int, float]] = []
                for nid in node_ids:
                    mean_mat = nodes[nid].metadata.mean
                    dist = log_euclidean_distance(ref_mean, mean_mat, normalize=True)
                    dist_list.append((nid, dist))

                # Sort by distance and assign order_id consecutively.
                dist_list.sort(key=lambda x: x[1])
                for order, (nid, _) in enumerate(dist_list):
                    nodes[nid].order_id = order

                # Finally ensure block_to_nodes[b_idx] reflects this ordering.
                node_ids.sort(key=lambda nid: nodes[nid].order_id)
        # Add additional index infor that helps searching
        id_to_idx = {n.node_id: i for i, n in enumerate(nodes)}
        block_to_node_indices = {  # type: ignore[attr-defined]
            b_idx: [id_to_idx[nid] for nid in node_ids if nid in id_to_idx]  # type: ignore[index]
            for b_idx, node_ids in block_to_nodes.items()
        }
        sorted_blocks = sorted(block_to_node_indices.keys())
        dag_dict[cluster_id] = IndexHandle(
            nodes=nodes, 
            block_to_nodes=block_to_nodes, 
            block_runs=block_runs, 
            perm=permutation, 
            config=config,
            block_to_node_indices=block_to_node_indices,
            sorted_blocks=sorted_blocks
        )
    
    # return dict
    return dag_dict, stats, global_dist_list


import pandas as pd
def get_dag_stats(dag_dict):
    dag_stats = []
    for cluster_id, cluster_dag in dag_dict.items():
        block_to_node = cluster_dag.block_to_node_indices
        for node in cluster_dag.nodes:
            
            num_spds = len(node.metadata.members)
            block_size = node.metadata.mean.shape[0]
            block_id = node.block_index
            
            #print(f"Cluster_id {cluster_id} Node id {node.global_node_id} Block {block_id}: {num_spds} spds")
            dag_stats.append({
                "cluster_id": cluster_id,
                "node_id": node.global_node_id,
                "block_id": block_id,
                "block_cluster_id": node.block_cluster_id,
                "num_spds": num_spds,
                "block_size": block_size
            })
    dag_stats_df = pd.DataFrame(dag_stats)
    return dag_stats_df


# Create an LSH based index where each SPD matrix will be
# eigen-decomposed and LSH buckets are used as clusters.
#
# The idea is:
#   * Represent each SPD by its eigenvalue vector (optionally
#     log-transformed for stability).
#   * Draw a small number of random projection matrices and use the
#     sign pattern of the projected vector as a binary hash.
#   * Use the resulting integer hash keys as bucket IDs.
#
# A simple query procedure is:
#   1. Compute the eigenvalues of the query SPD.
#   2. For each table, apply the stored projections and compute
#      the hash key.
#   3. Retrieve candidate SPD IDs from the corresponding buckets
#      and optionally re-rank them using a true SPD distance.
def build_lsh_index(
    spd_matrices: List[np.ndarray] | np.ndarray,
    *,
    n_tables: int = 4,
    n_bits: int = 16,
    random_state: int = 0,
    use_log_eigenvalues: bool = True,
) -> Dict[str, object]:
    """Build a simple LSH index over SPD matrices.

    Parameters
    ----------
    spd_matrices:
        Sequence or array of SPD matrices of shape (n, p, p).
    n_tables:
        Number of independent hash tables (projection families).
    n_bits:
        Number of bits per table; controls the number of buckets.
    random_state:
        Seed for the random number generator to keep the index
        deterministic.
    use_log_eigenvalues:
        If True, take the logarithm of eigenvalues before hashing,
        which typically stabilizes the representation for SPD.

    Returns
    -------
    Dict[str, object]
        A dictionary with the following keys:

        - "projections": np.ndarray of shape (n_tables, n_bits, p)
          containing the random projection matrices.
        - "buckets": list of length ``n_tables`` where each element
          is a dict mapping integer hash keys to lists of SPD indices.
        - "spd_ids": np.ndarray of shape (n,) with the indices of
          the input SPD matrices.
        - "random_state": int seed used to generate projections.
    """
    logger.info("Starting LSH based index")
    mats = np.asarray(spd_matrices)
    if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
        raise ValueError("spd_matrices must have shape (n, p, p) for some p")

    n, p, _ = mats.shape
    spd_ids = np.arange(n, dtype=int)

    # Compute eigenvalue-based representations for all SPDs.
    eig_reprs = np.empty((n, p), dtype=float)
    for i in range(n):
        # eigh for symmetric / SPD matrices; returns sorted eigenvalues.
        w = np.linalg.eigvalsh(mats[i])
        if use_log_eigenvalues:
            # Clamp for numerical safety before log.
            w = np.maximum(w, 1e-8)
            w = np.log(w)
        eig_reprs[i] = w

    # Create random projections for each table.
    rng = np.random.default_rng(random_state)
    projections = rng.standard_normal(size=(n_tables, n_bits, p))

    # Helper to convert a boolean bit-vector to an integer key.
    def _bits_to_int(bits: np.ndarray) -> int:
        key = 0
        for b, flag in enumerate(bits.astype(bool)):
            if flag:
                key |= 1 << b
        return int(key)

    buckets: List[Dict[int, List[int]]] = [dict() for _ in range(n_tables)]

    # Populate buckets for each table.
    for idx in range(n):
        v = eig_reprs[idx]
        for t in range(n_tables):
            proj = projections[t] @ v  # shape (n_bits,)
            bits = proj >= 0.0
            h = _bits_to_int(bits)
            bucket = buckets[t].setdefault(h, [])
            bucket.append(int(spd_ids[idx]))

    logger.info(f"Number of buckets {len(buckets)}")
    return {
        "projections": projections,
        "buckets": buckets,
        "spd_ids": spd_ids,
        "random_state": int(random_state),
    }


def build_random_lsh_index(
    spd_matrices: List[np.ndarray] | np.ndarray,
    *,
    n_tables: int = 4,
    n_bits: int = 16,
    random_state: int = 0,
    use_log_eigenvalues: bool = True,
    # --- NEW knobs for stability / closeness semantics ---
    eig_floor: float = 1e-8,           # floor before log
    eig_cap: Optional[float] = None,   # optional cap before log
    normalize_repr: bool = True,       # normalize vectors so SRP uses angle
    min_collisions: int = 2,           # candidates must collide in >= this many tables
) -> Dict[str, object]:
    """Build an SRP-LSH index over stabilized eigenvalue (or log-eigenvalue) vectors."""
    logger.info("Starting LSH based index")
    mats = np.asarray(spd_matrices)
    if mats.ndim != 3 or mats.shape[1] != mats.shape[2]:
        raise ValueError("spd_matrices must have shape (n, p, p) for some p")

    n, p, _ = mats.shape
    spd_ids = np.arange(n, dtype=int)

    # Compute eigenvalue-based representations for all SPDs.
    eig_reprs = np.empty((n, p), dtype=np.float64)
    for i in range(n):
        w = np.linalg.eigvalsh(mats[i])  # sorted eigenvalues

        # Stabilize spectrum (prevents log blow-up and wild scales)
        if eig_floor is not None:
            w = np.maximum(w, eig_floor)
        if eig_cap is not None:
            w = np.minimum(w, eig_cap)

        if use_log_eigenvalues:
            w = np.log(w)

        # Optional: normalize so SRP-LSH corresponds to angular similarity
        if normalize_repr:
            norm = np.linalg.norm(w)
            if norm > 0:
                w = w / norm

        eig_reprs[i] = w

    rng = np.random.default_rng(random_state)
    projections = rng.standard_normal(size=(n_tables, n_bits, p)).astype(np.float64)

    def _bits_to_int(bits: np.ndarray) -> int:
        # bits: (n_bits,) bool
        key = 0
        # little-endian bit packing
        for b in range(bits.shape[0]):
            if bits[b]:
                key |= (1 << b)
        return int(key)

    buckets: List[Dict[int, List[int]]] = [dict() for _ in range(n_tables)]

    for idx in range(n):
        v = eig_reprs[idx]
        for t in range(n_tables):
            proj = projections[t] @ v
            bits = proj >= 0.0
            h = _bits_to_int(bits)
            buckets[t].setdefault(h, []).append(int(spd_ids[idx]))

    # NOTE: len(buckets) is just n_tables; logging bucket *population* is more meaningful
    nonempty = sum(len(bt) for bt in buckets)
    logger.info(f"Built {n_tables} tables; total non-empty buckets across tables = {nonempty}")

    return {
        "projections": projections,
        "buckets": buckets,
        "spd_ids": spd_ids,
        "random_state": int(random_state),
        # NEW: store representation details + collision rule for query-time behavior
        "use_log_eigenvalues": bool(use_log_eigenvalues),
        "eig_floor": float(eig_floor),
        "eig_cap": (None if eig_cap is None else float(eig_cap)),
        "normalize_repr": bool(normalize_repr),
        "min_collisions": int(min_collisions),
        "p": int(p),
    }



def build_index(spatial_data: SpatialDataset, config: IndexConfig | None = None) -> IndexHandle:
    """Build an index over SPD sub-matrices derived from spatial data.

    This function is responsible for:

    1. Tiling spatial points (quadtree or generic tiling).
    2. Computing SPD matrices per tile (or tile pair) and converting
       them to correlation matrices.
    3. Clustering matrices into top-level clusters and computing a
       block-diagonal permutation per cluster.
    4. For each cluster, building a DAG index over blocks, grouping
       block-SPDs into clusters with pairwise log-Euclidean distance
       < ``epsilon`` and linking co-occurring block-clusters.

    The detailed algorithm is specified in .github/copilot-instructions.md.

    Note: This implementation provides the structural skeleton and
    deterministic wiring. The actual tiling, SPD computation, and
    clustering strategies should be implemented in project-specific
    extensions while respecting this interface.
    """

    if config is None:
        config = IndexConfig()

    configure_determinism(config.deterministic)
    logger.info("Building SPD index with epsilon=%s", config.epsilon)

    # Placeholder: in a complete implementation, the steps described in
    # the algorithm spec would be carried out here. For now, we construct
    # an empty index that can be populated by downstream code.

    nodes: List[BlockClusterNode] = []
    block_to_nodes: Dict[int, List[int]] = {}

    # The presence of an empty index handle still allows tests and
    # serialization to be written against the public API without
    # constraining concrete tiling or clustering details yet.

    return IndexHandle(nodes=nodes, block_to_nodes=block_to_nodes, config=config)


def save_index(data: Any, dag_dict: Any, path: PathLike) -> None:
    """Serialize an index handle to disk using pickle.

    ``index_handle`` is expected to be a graph-like object built in
    ``index.build_index``. Using dataclasses for index structures makes
    them naturally pickleable.
    """
    # To make the bundle robust to repeated reloads in notebooks, avoid
    # pickling live ``IndexConfig`` objects (which can come from stale
    # class definitions).  Instead, snapshot each handle's config into a
    # plain dict before pickling.

    serial_dag_dict: Dict[int, Any] = {}
    for cid, handle in dag_dict.items():
        cfg = getattr(handle, "config", None)
        cfg_dict = None
        if cfg is not None:
            # Convert to a simple, pickle-friendly dict.  We do not rely
            # on the exact class identity here, only on field values.
            cfg_dict = {
                "epsilon_dict": getattr(cfg, "epsilon_dict", {}),
                "epsilon_block_wise_dict": getattr(cfg, "epsilon_block_wise_dict", {}),
                "is_ordered": getattr(cfg, "is_ordered", False),
                "debug": getattr(cfg, "debug", False),
                "kmean_method": getattr(cfg, "kmean_method", "online"),
                "stabilize_block_cluster": getattr(cfg, "stabilize_block_cluster", False),
                "max_iter": getattr(cfg, "max_iter", 5),
                # DeterministicConfig is itself a small dataclass; turn it
                # into a plain dict if present.
                "deterministic": getattr(cfg, "deterministic", None).__dict__
                if getattr(cfg, "deterministic", None) is not None
                else None,
            }

        # Re-wrap in a lightweight IndexHandle-like struct but with a
        # dict config so that no stale classes are pickled.
        serial_dag_dict[cid] = IndexHandle(
            nodes=handle.nodes,
            block_to_nodes=handle.block_to_nodes,
            block_runs=handle.block_runs,
            perm=handle.perm,
            config=cfg_dict,  # type: ignore[arg-type]
            block_to_node_indices=handle.block_to_node_indices,
            sorted_blocks=handle.sorted_blocks,
        )

    bundle = DatasetIndex(
        dag_dict=serial_dag_dict,
        metadata=data.metadata,
        latent=data.latent,
        labels=data.labels,
        pca_model=getattr(data, "pca_model", None),
    )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: PathLike) -> Any:
    """Load a previously saved index handle from disk."""

    src = Path(path)
    with src.open("rb") as f:
        bundle = pickle.load(f)

    # For bundles created by ``save_index`` we may have stored
    # ``IndexHandle.config`` as a plain dict to avoid pickling live
    # class objects in notebook workflows.  Here we reconstruct proper
    # ``IndexConfig`` (and nested ``DeterministicConfig``) instances so
    # that downstream code can rely on the usual API.

    if isinstance(bundle, DatasetIndex):
        for cid, handle in bundle.dag_dict.items():
            cfg = getattr(handle, "config", None)
            # Already a real IndexConfig – nothing to do.
            if isinstance(cfg, IndexConfig):
                continue
            if isinstance(cfg, dict):
                det_dict = cfg.get("deterministic") or {}
                if isinstance(det_dict, dict):
                    det_cfg = DeterministicConfig(**det_dict)
                else:
                    det_cfg = DeterministicConfig()

                handle.config = IndexConfig(
                    epsilon_dict=cfg.get("epsilon_dict", {}),
                    epsilon_block_wise_dict=cfg.get("epsilon_block_wise_dict", {}),
                    is_ordered=cfg.get("is_ordered", False),
                    debug=cfg.get("debug", False),
                    kmean_method=cfg.get("kmean_method", "online"),
                    stabilize_block_cluster=cfg.get("stabilize_block_cluster", False),
                    max_iter=cfg.get("max_iter", 5),
                    deterministic=det_cfg,
                )

    return bundle


import networkx as nx
from scipy.linalg import expm
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
import tqdm
from spindle_dev import metrics

def expm_sym(A, eps=1e-10):
    A = 0.5*(A + A.T)
    w, V = np.linalg.eigh(A)
    # exp of symmetric matrix
    S = (V * np.exp(w)) @ V.T
    return 0.5*(S + S.T)

def _topk_graph_from_corr(R, genes, k=30):
    np.fill_diagonal(R, 0.0)
    W = np.abs(R)
    p = R.shape[0]
    kk = min(k, p - 1)
    G = nx.Graph()
    G.add_nodes_from(genes)

    for i in range(p):
        nbrs = np.argpartition(W[i], -kk)[-kk:]
        gi = genes[i]
        for j in nbrs:
            if j == i:
                continue
            w = float(W[i, j])
            if w <= 0:
                continue
            gj = genes[j]
            # keep max if repeated
            if G.has_edge(gi, gj):
                if w > G[gi][gj]["weight"]:
                    G[gi][gj]["weight"] = w
            else:
                G.add_edge(gi, gj, weight=w)
    return G

def score_nodes_from_a_cluster(data, cluster_id, dag_dict, take_representative_mean = False):
    gene_list = data.metadata['genes']
    gene_array = np.array(gene_list)
    dag = dag_dict[cluster_id]
    perm = dag.perm
    nodes = dag.nodes
    gene_array_perm = gene_array[np.ix_(perm)]
    score_dict = []
    for node in tqdm.tqdm(nodes):
        if take_representative_mean:
            log_spd_mean = node.metadata.representative_mean
        else:
            log_spd_mean = node.metadata.mean
        spd_mean = expm_sym(log_spd_mean)
        num_spds = len(node.metadata.members)
        block_index = node.block_index
        block_start, block_end = dag.block_runs[block_index]
        block_size = block_end - block_start
        R = metrics.spd_to_correlation(spd_mean)
        gene_subset = gene_array_perm[block_start:block_end]
        p = R.shape[0]
        np.fill_diagonal(R, 0.0)
        E = float(np.linalg.norm(R, ord="fro") / np.sqrt(p))

        G = _topk_graph_from_corr(R, gene_subset)
        if G.number_of_edges() == 0:
            score_dict.append({
                "cluster_id": cluster_id, 
                "block_id": block_index,
                "node_id": node.global_node_id, 
                "node_score": 0.0, 
                "num_spds": num_spds,
                "block_size": block_size,
                "E": E, 
                "Q": 0.0, 
                "S": 1.0, 
                "R": R, 
                "G": G, 
                "modules": []
            })
            continue
        comms = list(greedy_modularity_communities(G, weight="weight"))
        Q = float(modularity(G, comms, weight="weight")) if len(comms) > 1 else 0.0

        # spectral concentration on |R|
        evals = np.linalg.eigvalsh(np.abs(R))
        evals = np.maximum(evals, 0.0)
        Sconc = float(evals[-1] / (evals.sum() + 1e-9))

        node_score = float(E * (Q + 1e-6) * (1.0 - Sconc))

        modules = [sorted(list(c)) for c in comms]
        score_dict.append({
            "cluster_id": cluster_id, 
            "block_id": block_index,
            "node_id": node.global_node_id, 
            "node_score": node_score, 
            "num_spds": num_spds,
            "block_size": block_size,
            "E": E, 
            "Q": Q, 
            "S": Sconc, 
            "R": R, 
            "G": G, 
            "modules": modules
        })
    return score_dict


def get_node_log_ref(index_handle, block_id, use_representative_mean=False):
    nodes_for_block = index_handle.block_to_node_indices[block_id]
    num_mats = 0
    # create a 0 matrix of dimension for 
    block_size = index_handle.nodes[nodes_for_block[0]].metadata.mean.shape[0]
    L_sum = np.zeros((block_size, block_size))
    for n in nodes_for_block:
        if use_representative_mean is True:
            L_sum += index_handle.nodes[n].metadata.representative_mean
        else:
            L_sum += index_handle.nodes[n].metadata.mean
        num_mats += 1
    L_mean = L_sum / float(num_mats)
    return L_mean


def _gene_indices(all_genes, module_genes):
    """Return indices of module_genes that are present in all_genes, preserving order."""
    g2i = {g: i for i, g in enumerate(all_genes)}
    idx = [g2i[g] for g in module_genes if g in g2i]
    missing = [g for g in module_genes if g not in g2i]
    return np.array(idx, dtype=int), missing


def tile_module_scores_from_reference(data, module_genes, index_handle, block_id, L_ref, cluster_id, min_genes=3, eps=1e-8):
    """Score tiles by comparing block-wise covariance to a reference log-SPD block.
    
    For each tile, extracts the block sub-matrix (using the cluster's permutation),
    subsets to module genes, converts to log-SPD, and computes normalized Frobenius
    distance to L_ref.
    
    Parameters
    ----------
    data : ProcessedData
        Processed spatial data with spd_matrices, spd_ids, and metadata.
    module_genes : list of str
        Gene names of interest (subset of data.metadata['genes']).
    index_handle : IndexHandle
        Index with perm (permutation) and block_runs (block boundaries).
    block_id : int
        Block index in block_runs.
    L_ref : np.ndarray
        Reference log-SPD matrix, shape (k, k) where k is block size.
        Must be in the permuted gene order of the block.
    cluster_id : int
        Cluster ID to which the block belongs.
    min_genes : int, optional
        Minimum number of module genes required to score (default 3).
    eps : float, optional
        Floor for eigenvalues in _log_spd (default 1e-8).
    
    Returns
    -------
    scores : dict[tile_id -> float]
        LE distance between each tile's block covariance and L_ref.
    info : dict
        Metadata: kept_genes, missing_genes, k (number of genes kept).
    """
    gene_array = np.array(data.metadata["genes"])
    perm = index_handle.perm
    block_runs = index_handle.block_runs
    block_start, block_end = block_runs[block_id]
    
    # Genes in the permuted block order
    ordered_genes = gene_array[perm]
    block_genes = list(ordered_genes[block_start:block_end])
    
    # Find which module genes appear in the block and their indices within the block
    idx, missing = _gene_indices(block_genes, module_genes)
    
    if idx.size < min_genes:
        # Not enough genes to score reliably
        scores = {tile_id: np.nan for tile_id in data.spd_ids}
        info = {"kept_genes": [block_genes[i] for i in idx], "missing_genes": missing, "k": int(idx.size)}
        return scores, info

    # Verify reference matrix dimensions match the block size
    assert L_ref.shape[0] == len(block_genes), \
        f"Reference matrix size {L_ref.shape[0]} does not match block gene count {len(block_genes)}. " \
        f"Gene order mismatch: L_ref should be in permuted block order."
    assert L_ref.shape[1] == len(block_genes), \
        f"Reference matrix shape {L_ref.shape} not square or mismatched with block size {len(block_genes)}"
    
    # Verify gene names match by slicing ordered_genes array
    block_genes_slice = ordered_genes[block_start:block_end]
    assert np.array_equal(block_genes_slice, np.array(block_genes)), \
        f"Gene name mismatch: slice from ordered_genes doesn't match block_genes. " \
        f"This indicates a coordinate system error."

    scores = {}
    perm_block_indices = perm[block_start:block_end]  # global indices for this block

    spd_matrices = np.asarray(data.spd_matrices)
    spd_matrices_cluster = spd_matrices[data.labels == cluster_id]
    spd_ids_cluster = np.array(data.spd_ids)[data.labels == cluster_id]
    for tile_id, S in zip(spd_ids_cluster, spd_matrices_cluster):
        # Extract block from full SPD using permutation
        S_block = S[np.ix_(perm_block_indices, perm_block_indices)]
        
        # Subset to module genes within this block
        S_block_sub = S_block[np.ix_(idx, idx)]
        
        # Convert to log-SPD space
        L_tile_block = _log_spd(S_block_sub, eps=eps)
        
        # Compute LE distance
        L_ref_sub = L_ref[np.ix_(idx, idx)]
        scores[tile_id] = _fro_dist_norm(L_tile_block, L_ref_sub, p=idx.size)

    info = {"kept_genes": [block_genes[i] for i in idx], "missing_genes": missing, "k": int(idx.size)}
    return scores, info


def _mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    m = np.median(x)
    return np.median(np.abs(x - m))


def robust_z(x, eps=1e-12):
    """
    Robust z within a node (bounded if x is bounded).
    z_i = (x_i - median(x)) / (1.4826*MAD(x) + eps)
    """
    x = np.asarray(x, float)
    good = np.isfinite(x)
    if good.sum() < 3:
        return np.full_like(x, np.nan, dtype=float), {"median": np.nan, "mad": np.nan, "scale": np.nan, "n": int(good.sum())}
    med = np.median(x[good])
    mad = _mad(x[good])
    scale = max(1.4826 * mad, eps)
    z = (x - med) / scale
    return z, {"median": float(med), "mad": float(mad), "scale": float(scale), "n": int(good.sum())}


def node_scores_from_reference(
    data,
    index_handle,
    cluster_id,
    module_genes=None,
    representative_mean=False,
    eps=1e-8,
    alpha=1.0,
    min_tiles=2,
):
    cluster_mask = (np.asarray(data.labels) == cluster_id)
    cluster_tile_ids = set(np.asarray(data.spd_ids)[cluster_mask].tolist())

    node_rows = []
    per_node_tile_scores = {}
    tile_z = {}
    for block_id in index_handle.sorted_blocks:
        nodes_in_block = index_handle.block_to_node_indices[block_id]
        num_of_nodes = len(nodes_in_block)        
        for node_idx in nodes_in_block:
            node = index_handle.nodes[node_idx]
            node_id = node.global_node_id
            L_ref = node.metadata.representative_mean

            member_tile_ids = set([m[0] for m in node.metadata.members])
            num_tiles_in_node = len(member_tile_ids)

            # tile distance from reference
            if representative_mean:
                L_ref = node.metadata.representative_mean
            else:
                L_ref = node.metadata.mean

            # Find the distance vector for tiles within this cluster and the mean
            # for module genes
            # first fix the indices
            gene_array = np.array(data.metadata["genes"])
            perm = index_handle.perm
            block_runs = index_handle.block_runs
            block_start, block_end = block_runs[block_id]
            block_size = block_end - block_start

            ordered_genes = gene_array[perm]
            block_genes = list(ordered_genes[block_start:block_end])
            perm_block_indices = perm[block_start:block_end]

            if module_genes is not None:
                idx, missing = _gene_indices(block_genes, module_genes)
            else:
                idx = np.arange(block_size, dtype=int)
                missing = []
            
            assert len(data.labels) == len(data.spd_ids) == len(data.spd_matrices)
            L_ref_sub = L_ref[np.ix_(idx, idx)]
            d_map = {}
            for tile_id, S in zip(data.spd_ids, data.spd_matrices):
                if tile_id not in member_tile_ids:
                    continue
                S_block = S[np.ix_(perm_block_indices, perm_block_indices)]
                S_block_sub = S_block[np.ix_(idx, idx)]
                L_tile_block = _log_spd(S_block_sub, eps=eps)
                d_map[tile_id] = _fro_dist_norm(L_tile_block, L_ref_sub, p=idx.size)
            
            d_vals = np.array(list(d_map.values()), float)
            d_vals = d_vals[np.isfinite(d_vals)]
            if d_vals.size < min_tiles:
                continue
            med = float(np.median(d_vals))
            mad = float(_mad(d_vals))
            quality = float(np.exp(-med) * np.exp(-mad))
            prevalence = float(num_tiles_in_node / len(member_tile_ids)) if len(member_tile_ids) > 0 else np.nan

            interest = float((prevalence ** alpha) * quality) if np.isfinite(prevalence) else np.nan
            tids = list(d_map.keys())
            d_arr = np.array([d_map[t] for t in tids], float)

            # one robust normalization per node
            z_arr, z_stats = robust_z(d_arr)
            info = {"kept_genes": [block_genes[i] for i in idx], "missing_genes": missing, "k": int(idx.size)}
            tile_z[node_id] = {
                t: (float(z_arr[i]) if np.isfinite(z_arr[i]) else np.nan)
                for i, t in enumerate(tids)
            }

            node_rows.append({
                "block_id": int(block_id),
                "cluster_id": int(cluster_id),
                "node_id": node_id,
                "node_index": int(node_idx),
                "prevalence": prevalence,
                "median_dist": med,
                "mad_dist": mad,
                "quality": quality,
                "interest": interest,
                "k_genes": int(info["k"]),
                "z_median": z_stats["median"],
                "z_mad": z_stats["mad"],
                "z_scale": z_stats["scale"],
            })
        
            # 
            per_node_tile_scores[node_id] = d_map

    return node_rows, per_node_tile_scores, tile_z
            
            

