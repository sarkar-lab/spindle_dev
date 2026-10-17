"""The layout shared by all three indexes: tiles -> covariance niches -> a gene order and blocks per niche.

Niches are Leiden clusters of the tiles' ultrametric (correlation-tree) features. Each niche gets a
consensus gene order (the leaves of the average-linkage tree of its mean ultrametric) and contiguous
gene blocks cut from that tree and merged by dynamic programming on the niche's mean correlation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.cluster.hierarchy import cophenet, dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist, squareform

from .metrics import spd_to_correlation
from .preprocessing import QuadTile
from .utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------------------------- tree features
def spd_to_ultrametric(C, link_method="average"):
    """Cophenetic (ultrametric) matrix and linkage of the gene tree on 1 - |corr(C)|."""
    D = 1.0 - np.abs(spd_to_correlation(C))
    np.fill_diagonal(D, 0.0)
    Y = squareform(D, checks=False)
    Z = linkage(Y, method=link_method)
    return squareform(cophenet(Z, Y)[1]), Z


def build_ultrametrics(C_list, link_method="average"):
    U_list, Z_list = [], []
    for C in C_list:
        U, Z = spd_to_ultrametric(C, link_method=link_method)
        U_list.append(U)
        Z_list.append(Z)
    return U_list, Z_list


def spd_tree_feature_matrix(U_list):
    """(K, p(p-1)/2) matrix of the ultrametrics' upper triangles."""
    return np.vstack([squareform(U, checks=False) for U in U_list])


def leiden_clustering_latent(latent_feat, k_neighbors=10, resolution=1.0, metric="euclidean", random_state=0):
    """Leiden (RB configuration) on a kNN graph of the latent features with Gaussian-kernel weights."""
    import igraph as ig
    import leidenalg

    D = squareform(pdist(latent_feat, metric=metric))
    S = np.exp(-D / (D.mean() + 1e-12))
    K = latent_feat.shape[0]
    edges, weights = [], []
    for i in range(K):
        for j in np.argsort(D[i])[:k_neighbors + 1]:
            if i != j:
                edges.append((i, j))
                weights.append(S[i, j])
    g = ig.Graph()
    g.add_vertices(K)
    g.add_edges(edges)
    g.es["weight"] = weights
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights="weight",
                                    resolution_parameter=resolution, seed=random_state)
    return np.array(part.membership)


def split_oversized_niches(latent_feat: np.ndarray, labels: np.ndarray, cap: int, random_state: int = 0) -> np.ndarray:
    """Split every niche larger than ``cap`` by k-means (k = ceil(size / cap)) until none exceeds it."""
    from sklearn.cluster import KMeans

    labels = np.asarray(labels).astype(int).copy()
    while True:
        over = np.flatnonzero(np.bincount(labels) > cap)
        if len(over) == 0:
            break
        nxt = labels.max() + 1
        for k in over:
            members = np.flatnonzero(labels == k)
            n_parts = int(np.ceil(len(members) / cap))
            parts = KMeans(n_clusters=n_parts, n_init=10, random_state=random_state).fit_predict(latent_feat[members])
            for j in range(1, n_parts):
                labels[members[parts == j]] = nxt
                nxt += 1
            logger.info("Split niche %d (%d tiles) into %s", k, len(members), np.bincount(parts).tolist())
    return np.unique(labels, return_inverse=True)[1]


def consensus_tree_from_ultrametrics(U_list, labels, cluster_id, link_method="average"):
    """Mean ultrametric of a niche, its linkage and the leaf order (the niche's gene order)."""
    idx = np.where(labels == cluster_id)[0]
    U_mean = np.stack([U_list[i] for i in idx], axis=0).mean(axis=0)
    Z = linkage(squareform(U_mean, checks=False), method=link_method)
    return U_mean, Z, np.array(dendrogram(Z, no_plot=True)["leaves"])


# ---------------------------------------------------------------------------------------------- blocks
def runs_and_sizes_from_t(Z, perm, t):
    """Contiguous runs (start, end) in ``perm`` order of the tree cut at ``t``, and their sizes."""
    labp = fcluster(Z, t, criterion="distance")[np.asarray(perm)]
    runs, start = [], 0
    for i in range(1, len(labp) + 1):
        if i == len(labp) or labp[i] != labp[start]:
            runs.append((start, i))
            start = i
    return runs, np.array([b - a for a, b in runs], dtype=int)


def candidate_thresholds_from_Z(Z):
    d = np.unique(Z[:, 2])
    return 0.5 * (d[:-1] + d[1:])


def _smooth_ma(x, w=7):
    x_arr = np.asarray(x, float)
    if w is None or w <= 1:
        return x_arr
    w = min(int(w), len(x_arr))
    if w <= 1:
        return x_arr
    return np.convolve(x_arr, np.ones(w) / w, mode="same")


def knee_from_num_blocks(ts, num_blocks, *, smooth=9, left_margin=2, eps=1e-12):
    """Index of the curvature knee of log(num_blocks(t)), moved ``left_margin`` steps left."""
    t = np.asarray(ts, float)
    if len(t) < 3:
        return 0
    y = _smooth_ma(np.log(np.asarray(num_blocks, float) + eps), w=smooth)
    d2y = np.gradient(np.gradient(y, t), t)
    return max(0, int(np.argmin(d2y)) - int(left_margin))


def pick_t_knee_with_size_guard(Z, perm, ts, *, min_size=None, max_size=None, smooth=9, left_margin=2, eps=1e-12):
    """The knee threshold among cuts whose blocks all have sizes in [min_size, max_size]."""
    ts = np.asarray(ts, float)
    num_blocks = np.empty(len(ts), dtype=int)
    max_block = np.empty(len(ts), dtype=int)
    min_block = np.empty(len(ts), dtype=int)
    for i, t in enumerate(ts):
        _, sizes = runs_and_sizes_from_t(Z, perm, t)
        num_blocks[i], max_block[i], min_block[i] = len(sizes), sizes.max(), sizes.min()

    ok = np.ones(len(ts), dtype=bool)
    if min_size is not None:
        ok &= min_block >= int(min_size)
    if max_size is not None:
        ok &= max_block <= int(max_size)
    idx_ok = np.where(ok)[0]
    if len(idx_ok) == 0:
        return float(ts[int(np.argmin(max_block))])
    j = knee_from_num_blocks(ts[idx_ok], num_blocks[idx_ok], smooth=smooth, left_margin=left_margin, eps=eps)
    return float(ts[int(idx_ok[j])])


def dp_group_runs(R_mean, perm, runs_fine, min_size=20, max_size=None, lam=0.0, eps=1e-12):
    """Merge consecutive fine runs into blocks of size in [min_size, max_size] maximising the summed
    within-block density of squared off-diagonal correlation (``lam`` penalises each extra block)."""
    Rp = R_mean[np.ix_(perm, perm)].copy()
    np.fill_diagonal(Rp, 0.0)
    W = Rp ** 2
    n = W.shape[0]
    P = np.zeros((n + 1, n + 1), dtype=W.dtype)
    P[1:, 1:] = W.cumsum(0).cumsum(1)

    starts = [s for s, _ in runs_fine]
    ends = [e for _, e in runs_fine]
    m = len(runs_fine)
    prefix_len = np.concatenate([[0], np.cumsum([e - s for s, e in runs_fine])])

    cost = np.full((m, m), np.inf)
    for i in range(m):
        for j in range(i, m):
            L = int(prefix_len[j + 1] - prefix_len[i])
            if L < min_size or (max_size is not None and L > max_size):
                continue
            a, b = starts[i], ends[j]
            within = float(P[b, b] - P[a, b] - P[b, a] + P[a, a])
            cost[i, j] = -within / (L * L + eps)

    dp = np.full(m + 1, np.inf)
    prev = np.full(m + 1, -1, dtype=int)
    dp[0] = 0.0
    for j in range(1, m + 1):
        best, best_i = np.inf, -1
        for i in range(j):
            if np.isinf(cost[i, j - 1]):
                continue
            val = dp[i] + cost[i, j - 1] + lam
            if val < best:
                best, best_i = val, i
        dp[j], prev[j] = best, best_i

    super_runs, j = [], m
    while j > 0 and prev[j] >= 0:
        i = prev[j]
        super_runs.append((starts[i], ends[j - 1]))
        j = i
    return super_runs[::-1]


# ---------------------------------------------------------------------------------------------- layout
class ProcessedData:
    """Training tiles, their covariances and the niche/block layout.

    After ``build_layout``: ``labels`` (niche of every tile), ``perm_list[k]`` (gene order of niche k)
    and ``block_dict[k]`` (block runs (start, end) in that order). Rows of ``spd_matrices``/``labels``
    are the training tiles in input order; ``spd_ids`` are their tile ids.
    """

    def __init__(self, tiles: List[QuadTile], tile_stats: List[Dict], genes_work, num_spots: int):
        self.spd_matrices = [t["cov"] for t in tile_stats]
        self.spd_ids = [t["tile_id"] for t in tile_stats]
        self.metadata = {"tiles": tiles, "genes": np.array(genes_work)}
        self.num_genes = len(genes_work)
        self.num_spots = num_spots
        self.latent = {}
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

    def reduce_dim(self, num_pca_components: int = 30, n_components: int = 2, random_state: int = 0):
        """Ultrametric features -> PCA (``latent['pca']``, used for niches) -> UMAP (``latent['umap']``, plots)."""
        from sklearn.decomposition import PCA
        from umap import UMAP

        U_list, _ = build_ultrametrics(self.spd_matrices)
        pca = PCA(n_components=num_pca_components, random_state=random_state)
        self.latent["pca"] = pca.fit_transform(spd_tree_feature_matrix(U_list))
        self.pca_model = pca
        self.U_list = U_list
        umap = UMAP(n_components=n_components, random_state=random_state, metric="euclidean",
                    n_neighbors=15, min_dist=0.1)
        self.latent["umap"] = umap.fit_transform(self.latent["pca"])

    def cluster_spds(self, resolution: float = 0.1, max_niche_size: Optional[int] = 1000,
                     resolution_step: float = 0.1, max_resolution_tries: int = 15, random_state: int = 0) -> None:
        """Leiden niches, raising the resolution until no niche exceeds ``max_niche_size`` (k-means splits
        what is still too large), then each niche's consensus gene order."""
        latent_feat = self.latent["pca"]
        target_max = max_niche_size if max_niche_size is not None else max(200, round(0.15 * latent_feat.shape[0]))
        res = resolution
        for _ in range(max_resolution_tries):
            labels = leiden_clustering_latent(latent_feat, k_neighbors=10, resolution=res, random_state=random_state)
            sizes = np.bincount(labels)
            if sizes.max() <= target_max:
                break
            res += resolution_step
        else:
            logger.warning("Max niche size still %d > %d at resolution %.2f; splitting with k-means.",
                           sizes.max(), target_max, res)
            labels = split_oversized_niches(latent_feat, labels, target_max, random_state)
            sizes = np.bincount(labels)
        logger.info("Leiden resolution %.2f: %d niches, sizes %s", res, len(sizes), sorted(sizes.tolist(), reverse=True))
        self.labels = labels
        for cluster_id in set(labels):
            U_mean, Z, perm = consensus_tree_from_ultrametrics(self.U_list, labels, cluster_id)
            self.U_mean_list[cluster_id] = U_mean
            self.perm_list[cluster_id] = perm
            self.Z_list[cluster_id] = Z

    def assign_label_to_spots(self):
        """``spot_label[spot] = niche`` for every spot of every training tile."""
        self.spot_label = {}
        for i, t in enumerate(self.metadata["tiles"]):
            for spot in t.idx:
                self.spot_label[spot] = self.labels[i]

    def get_corr_mean_by_cluster(self):
        """``R_mean_list[k]``: mean correlation matrix of niche k."""
        if type(self.spd_matrices) is not np.ndarray:
            self.spd_matrices = np.array(self.spd_matrices)
        for cluster_id in set(self.labels):
            idxs = np.where(self.labels == cluster_id)[0]
            corr_sum = None
            for i in idxs:
                corr = spd_to_correlation(self.spd_matrices[i])
                corr_sum = corr.copy() if corr_sum is None else corr_sum + corr
            self.R_mean_list[cluster_id] = corr_sum / len(idxs)

    def get_adaptive_runs(self, min_size=1, max_size=50, lam=0, min_final_size=5, max_final_size=50, step_size=5):
        """``block_dict[k]``: cut niche k's tree at the size-guarded knee, then merge the runs into blocks of
        [min_final_size, max_final_size] genes (max_final_size grows by ``step_size`` until the block count
        is sensible)."""
        for cluster_id in set(self.labels):
            Z, perm, R_mean = self.Z_list[cluster_id], self.perm_list[cluster_id], self.R_mean_list[cluster_id]
            t_choice = pick_t_knee_with_size_guard(Z, perm, candidate_thresholds_from_Z(Z),
                                                   min_size=min_size, max_size=max_size)
            run_fcluster, _ = runs_and_sizes_from_t(Z, perm, t_choice)
            self.block_initial_dict[cluster_id] = run_fcluster

            def bad(runs, max_sz):
                return len(runs) < len(perm) / max_sz or len(runs) > len(perm) / min_final_size

            new_max_size = max_final_size
            runs = dp_group_runs(R_mean, perm, run_fcluster, min_size=min_final_size, max_size=new_max_size, lam=lam)
            while bad(runs, new_max_size):
                new_max_size += step_size
                if new_max_size > R_mean.shape[0]:
                    logger.error("Niche %s: no sensible block runs up to max size %d.", cluster_id, new_max_size)
                    break
                runs = dp_group_runs(R_mean, perm, run_fcluster, min_size=min_final_size, max_size=new_max_size,
                                     lam=lam)
            logger.info("Niche %s: %d blocks", cluster_id, len(runs))
            self.block_dict[cluster_id] = runs


def build_layout(tiles, tile_covs, genes, num_spots, resolution=0.2, min_final_size=15, max_final_size=100,
                 max_niche_size=1000, random_state=0) -> ProcessedData:
    """The production layout of the training tiles (``tile_covs``: one {'tile_id', 'cov'} per tile).

    ``random_state`` seeds PCA, UMAP and Leiden.
    """
    data = ProcessedData(tiles, tile_covs, genes, num_spots)
    data.reduce_dim(num_pca_components=max(2, min(30, len(tiles) - 1)), n_components=2, random_state=random_state)
    data.cluster_spds(resolution=resolution, max_niche_size=max_niche_size, random_state=random_state)
    data.assign_label_to_spots()
    data.get_corr_mean_by_cluster()
    data.get_adaptive_runs(min_final_size=min_final_size, max_final_size=max_final_size)
    return data
