"""SPD / correlation metrics and helpers.

This module provides log-Euclidean distance utilities and conversions
between SPD covariance matrices and correlation matrices.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import logm
from scipy.spatial.distance import squareform, pdist
from scipy.cluster.hierarchy import linkage, cophenet, dendrogram, fcluster


def _as_array(matrix: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(matrix, dtype=np.float64)


def is_spd(matrix: ArrayLike, *, atol: float = 1e-8) -> bool:
    """Return True if ``matrix`` is (numerically) symmetric positive definite.

    Checks symmetry and that all eigenvalues are strictly positive up to
    a small tolerance ``atol``.
    """

    m = _as_array(matrix)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        return False
    if not np.allclose(m, m.T, atol=atol):
        return False
    eigvals = np.linalg.eigvalsh(m)
    return np.all(eigvals > atol)


def log_spd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """SPD matrix logarithm via eigendecomposition (fast + stable for SPD)."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)  # clamp for numerical safety
    return (V * np.log(w)) @ V.T


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


def log_euclidean_distance(
    a: ArrayLike,
    b: ArrayLike,
    *,
    normalize: bool = True
) -> float:
    """Compute the (optionally normalized) log-Euclidean distance between two SPD matrices.

    The distance is defined as:
        d_LE(A, B) = || log(A) - log(B) ||_F

    If normalize=True, the distance is normalized by sqrt(p), where
    p is the matrix dimension, yielding an RMS log-deviation that is
    comparable across block sizes.
    """

    A = _as_array(a)
    B = _as_array(b)

    if A.shape != B.shape:
        raise ValueError("SPD matrices must have the same shape.")

    p = A.shape[0]

    log_A = logm(A)
    log_B = logm(B)

    diff = log_A - log_B
    dist = np.linalg.norm(diff, ord="fro")

    if normalize:
        dist /= np.sqrt(p)

    return float(dist)


def add_spd_noise(spd: ArrayLike, noise_level: float = 0.1, seed: int = None) -> NDArray[np.float64]:
    """Add noise to an SPD matrix while preserving the SPD property.
    
    Uses eigenvalue decomposition to add noise in the log-space of eigenvalues,
    which preserves positive definiteness and symmetry.
    
    Parameters
    ----------
    spd : ArrayLike
        Input SPD matrix.
    noise_level : float, default=0.1
        Standard deviation of Gaussian noise added to log-eigenvalues.
        Higher values produce more perturbation.
    seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    NDArray[np.float64]
        Noisy SPD matrix.
    """
    A = _as_array(spd)
    
    if A.shape[0] != A.shape[1]:
        raise ValueError("Input must be a square matrix.")
    
    rng = np.random.default_rng(seed)
    
    # Symmetrize to ensure numerical symmetry
    A = 0.5 * (A + A.T)
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(A)
    
    # Ensure positive eigenvalues
    eigvals = np.maximum(eigvals, 1e-8)
    
    # Add noise in log space (preserves positivity)
    log_eigvals = np.log(eigvals)
    noise = rng.normal(0, noise_level, size=log_eigvals.shape)
    noisy_log_eigvals = log_eigvals + noise
    noisy_eigvals = np.exp(noisy_log_eigvals)
    
    # Reconstruct the matrix
    A_noisy = eigvecs @ np.diag(noisy_eigvals) @ eigvecs.T
    
    # Ensure symmetry (numerical stability)
    A_noisy = 0.5 * (A_noisy + A_noisy.T)
    
    return A_noisy



# def spd_to_correlation(spd: ArrayLike) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
#     """Convert an SPD covariance matrix to a correlation matrix.

#     Returns a tuple ``(corr, stddevs)`` where ``stddevs`` are the
#     standard deviations (square root of the diagonal) used in the
#     normalization. This allows round-tripping via ``correlation_to_spd``.
#     """

#     cov = _as_array(spd)
#     if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
#         raise ValueError("SPD matrix must be square.")

#     diag = np.diag(cov)
#     if np.any(diag <= 0):
#         raise ValueError("Diagonal entries must be positive to form a correlation matrix.")

#     stddevs = np.sqrt(diag)
#     denom = np.outer(stddevs, stddevs)
#     corr = cov / denom
#     # Numerical guard: enforce ones on the diagonal
#     np.fill_diagonal(corr, 1.0)
#     return corr, stddevs

def spd_to_correlation(C):
    """
    Convert SPD matrix C to a correlation-like matrix R.
    Ensures diag(R) = 1 and is scale-invariant.
    """
    d = np.sqrt(np.diag(C))
    # Guard against zeros on the diagonal
    d[d == 0] = 1e-12
    D_inv = np.diag(1.0 / d)
    R = D_inv @ C @ D_inv
    # Numerical clean-up: clip to [-1, 1]
    R = np.clip(R, -1.0, 1.0)
    return R


def correlation_to_spd(corr: ArrayLike, stddevs: ArrayLike) -> NDArray[np.float64]:
    """Reconstruct an SPD covariance matrix from a correlation matrix.

    ``stddevs`` should be the vector returned from ``spd_to_correlation``.
    """

    c = _as_array(corr)
    s = _as_array(stddevs)
    if c.ndim != 2 or c.shape[0] != c.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if s.ndim != 1 or s.shape[0] != c.shape[0]:
        raise ValueError("stddevs must be a 1D vector matching correlation size.")

    cov = c * np.outer(s, s)
    return cov


def spd_to_ultrametric(C, link_method="average"):
    """
    For one SPD matrix:
    - convert to correlation
    - build distance = 1 - |corr|
    - run linkage
    - return the full cophenetic distance matrix U (ultrametric)
    """
    R = spd_to_correlation(C)
    # distance between indices
    D = 1.0 - np.abs(R)
    # ensure diagonal is exactly 0
    np.fill_diagonal(D, 0.0)
    # condensed form
    Y = squareform(D, checks=False)
    # hierarchical clustering
    Z = linkage(Y, method=link_method)
    # cophenetic distances (condensed)
    coph_dist = cophenet(Z, Y)[1]
    # back to full matrix
    U = squareform(coph_dist)
    return U, Z  # Z is the dendrogram for this SPD if you want it


def spd_tree_feature_matrix(U_list):
    """
    Turn each ultrametric U^(k) into a feature vector: upper-tri entries.
    Returns a (K, m) feature matrix, where m = n*(n-1)/2.
    """
    feats = []
    for U in U_list:
        # use condensed upper-triangular part as features
        y = squareform(U, checks=False)
        feats.append(y)
    return np.vstack(feats)  # shape: (K, m)


def build_ultrametrics(C_list, link_method="average"):
    """
    For a list of SPD matrices C_list, compute their ultrametric matrices.
    Returns:
      U_list: list of (n, n) ultrametric matrices
      Z_list: list of linkage matrices for each SPD (optional)
    """
    U_list = []
    Z_list = []
    for C in C_list:
        U, Z = spd_to_ultrametric(C, link_method=link_method)
        U_list.append(U)
        Z_list.append(Z)
    return U_list, Z_list


def leiden_clustering_latent(latent_feat, 
                             k_neighbors=10, 
                             resolution=1.0, 
                             metric="euclidean", 
                             random_state=0):
    """Run Leiden clustering on latent ultrametric features.

    Imports ``leidenalg`` and ``igraph`` lazily so that this module can
    be imported and reloaded without those optional dependencies
    installed. They are only required when this function is called.
    """

    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "leidenalg and igraph are required for leiden_clustering_latent; "
            "install them to use this clustering routine."
        ) from exc
    # 1. pairwise distances
    D = squareform(pdist(latent_feat, metric=metric))   # (K,K)
    
    # convert to similarity
    S = np.exp(-D / (D.mean() + 1e-12))   # Gaussian kernel-like
    
    # 2. build kNN graph
    K = latent_feat.shape[0]
    edges = []
    weights = []
    for i in range(K):
        nbrs = np.argsort(D[i])[:k_neighbors+1]
        for j in nbrs:
            if i != j:
                edges.append((i, j))
                weights.append(S[i, j])

    # 3. build igraph graph
    g = ig.Graph()
    g.add_vertices(K)
    g.add_edges(edges)
    g.es['weight'] = weights

    # 4. Leiden
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=random_state
    )

    labels = np.array(part.membership)
    return labels, part, g


def consensus_tree_from_ultrametrics(U_list, labels, cluster_id, link_method="average"):
    idx = np.where(labels == cluster_id)[0]
    U_stack = np.stack([U_list[i] for i in idx], axis=0)  # (Nc, n, n)
    U_mean = U_stack.mean(axis=0)

    Y = squareform(U_mean, checks=False)
    Z = linkage(Y, method=link_method)
    perm = dendrogram(Z, no_plot=True)["leaves"]

    return U_mean, Z, np.array(perm)


def bdi_banded(C, perm, w):
    """
    Banded Block-Diagonality Index for a single SPD matrix C.
    Higher BDI means more block-diagonal.
    """
    # convert to correlation-like matrix
    R = spd_to_correlation(C)

    # apply permutation
    Rp = R[np.ix_(perm, perm)]
    
    # remove diagonal
    Rp_off = Rp.copy()
    np.fill_diagonal(Rp_off, 0.0)

    # total off-diagonal energy
    E_tot = np.sum(Rp_off**2)
    if E_tot == 0:
        return 1.0

    # mask for entries OUTSIDE the band |i-j| <= w
    n = Rp.shape[0]
    i_idx, j_idx = np.indices((n, n))
    mask_between = (np.abs(i_idx - j_idx) > w)

    # energy outside the band
    E_between = np.sum(Rp_off[mask_between]**2)

    # BDI
    return 1.0 - (E_between / E_tot)


def pick_w_by_energy_quantile(R_mean, perm, quantile=0.9):
    Rp = R_mean[np.ix_(perm, perm)].copy()
    np.fill_diagonal(Rp, 0.0)

    n = Rp.shape[0]
    i_idx, j_idx = np.indices((n, n))
    dist = np.abs(i_idx - j_idx).ravel()
    weights = (Rp**2).ravel()

    max_d = dist.max()
    energy_per_d = np.bincount(dist, weights=weights, minlength=max_d+1)

    total = energy_per_d.sum()
    cum = np.cumsum(energy_per_d)
    target = quantile * total
    return energy_per_d,int(np.searchsorted(cum, target))


def blocks_from_fcluster(Z, perm, t):
    """
    Z      : linkage (e.g., Z_cons)
    perm   : order (array length n)
    t      : threshold for fcluster
    """
    perm = np.asarray(perm)

    # flat clusters (block labels) from dendrogram cut
    lab = fcluster(Z, t, criterion="distance")   # length n, labels for original indices
    labp = lab[perm]                              # labels in perm order

    # contiguous blocks as (start, end) in perm-space
    runs = []
    start = 0
    for i in range(1, len(labp) + 1):
        if i == len(labp) or labp[i] != labp[start]:
            runs.append((start, i))  # [start, i)
            start = i

    return labp, runs


def candidate_thresholds_from_Z(Z):
    d = np.unique(Z[:, 2])
    return 0.5 * (d[:-1] + d[1:])


def block_metrics_over_t(Z, perm):
    perm = np.asarray(perm)
    ts = candidate_thresholds_from_Z(Z)

    out = {
        "t": [],
        "num_blocks": [],
        "max_frac": [],
        "entropy": [],
    }

    n = len(perm)

    for t in ts:
        lab = fcluster(Z, t, criterion="distance")
        labp = lab[perm]

        runs = []
        start = 0
        for i in range(1, len(labp) + 1):
            if i == len(labp) or labp[i] != labp[start]:
                runs.append((start, i))
                start = i

        sizes = np.array([b - a for a, b in runs], dtype=float)
        p = sizes / sizes.sum()

        out["t"].append(t)
        out["num_blocks"].append(len(sizes))
        out["max_frac"].append(p.max())
        out["entropy"].append(-(p * np.log(p + 1e-12)).sum())

    for k in out:
        out[k] = np.array(out[k])

    return out


def _smooth_ma(x, w=7):
    x_arr = np.asarray(x, float)
    if w is None or w <= 1:
        return x_arr
    w = int(w)
    if len(x_arr) < w:
        w = len(x_arr)
    if w <= 1:
        return x_arr
    ker = np.ones(w) / w
    return np.convolve(x_arr, ker, mode="same")


def knee_from_num_blocks(ts, num_blocks, *, smooth=9, left_margin=2, eps=1e-12):
    """
    Detect collapse onset using log(num_blocks(t)) curvature.
    Returns: (t_knee, t_choice, idx_knee, idx_choice, slopes, curv)
    """
    t = np.asarray(ts, float)
    k = np.asarray(num_blocks, float)

    if len(t) < 3:
        idx_knee = 0
        idx_choice = 0
        return (float(t[idx_knee]), float(t[idx_choice]),
                idx_knee, idx_choice, np.zeros_like(t), np.zeros_like(t))

    y = np.log(k + eps)
    y = _smooth_ma(y, w=smooth)

    # slope and curvature
    dy = np.gradient(y, t)        # dy/dt (negative)
    d2y = np.gradient(dy, t)      # d2y/dt2 (more negative around steepening)

    # "knee" where slope steepens fastest: most negative curvature
    idx_knee = int(np.argmin(d2y))

    idx_choice = max(0, idx_knee - int(left_margin))
    return (float(t[idx_knee]), float(t[idx_choice]),
            idx_knee, idx_choice, dy, d2y)


def runs_and_sizes_from_t(Z, perm, t):
    perm = np.asarray(perm)
    lab = fcluster(Z, t, criterion="distance")
    labp = lab[perm]

    runs = []
    start = 0
    for i in range(1, len(labp) + 1):
        if i == len(labp) or labp[i] != labp[start]:
            runs.append((start, i))
            start = i

    sizes = np.array([b - a for a, b in runs], dtype=int)
    return runs, sizes


def pick_t_knee_with_size_guard(Z, perm, ts, *,
                                min_size=None, max_size=None,
                                smooth=9, left_margin=2, eps=1e-12):
    """
    Pick t using curvature-knee on log(num_blocks), but only among t values
    whose induced blocks satisfy size constraints.
    """
    ts = np.asarray(ts, float)

    # gather metrics across all candidates
    num_blocks = np.empty(len(ts), dtype=int)
    max_block  = np.empty(len(ts), dtype=int)
    min_block  = np.empty(len(ts), dtype=int)

    for i, t in enumerate(ts):
        _, sizes = runs_and_sizes_from_t(Z, perm, t)
        num_blocks[i] = len(sizes)
        max_block[i]  = sizes.max()
        min_block[i]  = sizes.min()

    # feasibility mask
    ok = np.ones(len(ts), dtype=bool)
    if min_size is not None:
        ok &= (min_block >= int(min_size))
    if max_size is not None:
        ok &= (max_block <= int(max_size))

    idx_ok = np.where(ok)[0]
    if len(idx_ok) == 0:
        # nothing feasible: safest fallback is smallest t (most blocks, smallest max block)
        j = int(np.argmin(max_block))
        return {
            "t": float(ts[j]),
            "idx": j,
            "reason": "no feasible t; picked minimal max_block",
            "num_blocks": int(num_blocks[j]),
            "max_block": int(max_block[j]),
            "min_block": int(min_block[j]),
        }

    # restrict arrays to feasible region
    ts_f = ts[idx_ok]
    nb_f = num_blocks[idx_ok]

    # your knee detector on feasible subsequence
    t_knee, t_choice, k_f, j_f, dy_f, d2y_f = knee_from_num_blocks(
        ts_f, nb_f, smooth=smooth, left_margin=left_margin, eps=eps
    )

    # map back to original index
    idx_choice = int(idx_ok[j_f])

    return {
        "t": float(ts[idx_choice]),
        "idx": idx_choice,
        "reason": "knee within size-feasible region",
        "num_blocks": int(num_blocks[idx_choice]),
        "max_block": int(max_block[idx_choice]),
        "min_block": int(min_block[idx_choice]),
        "debug": {
            "t_knee_feasible": float(t_knee),
            "t_choice_feasible": float(t_choice),
            "idx_ok_range": (int(idx_ok[0]), int(idx_ok[-1])),
        }
    }


def dp_group_runs(R_mean, perm, runs_fine, min_size=20, max_size=None, lam=0.0, eps=1e-12):
    Rp = R_mean[np.ix_(perm, perm)].copy()
    np.fill_diagonal(Rp, 0.0)
    W = Rp**2

    n = W.shape[0]
    P = np.zeros((n+1, n+1), dtype=W.dtype)
    P[1:, 1:] = W.cumsum(0).cumsum(1)

    def rect_sum(a,b,c,d):
        return float(P[b,d] - P[a,d] - P[b,c] + P[a,c])

    starts = [s for (s,_) in runs_fine]
    ends   = [e for (_,e) in runs_fine]
    m = len(runs_fine)

    run_len = np.array([e-s for (s,e) in runs_fine], dtype=int)
    prefix_len = np.concatenate([[0], np.cumsum(run_len)])

    def union_size(i, j):  # i..j inclusive
        return int(prefix_len[j+1] - prefix_len[i])

    cost = np.full((m, m), np.inf)
    for i in range(m):
        for j in range(i, m):
            L = union_size(i, j)
            if L < min_size: 
                continue
            if max_size is not None and L > max_size:
                continue
            a, b = starts[i], ends[j]
            within = rect_sum(a, b, a, b)
            density = within / (L*L + eps)
            cost[i, j] = -density  # minimize negative density == maximize density

    dp = np.full(m+1, np.inf)
    prev = np.full(m+1, -1, dtype=int)
    dp[0] = 0.0

    for j in range(1, m+1):
        best, best_i = np.inf, -1
        for i in range(0, j):
            if np.isinf(cost[i, j-1]): 
                continue
            val = dp[i] + cost[i, j-1] + lam  # lam penalizes more blocks
            if val < best:
                best, best_i = val, i
        dp[j] = best
        prev[j] = best_i

    super_runs = []
    j = m
    while j > 0 and prev[j] >= 0:
        i = prev[j]
        super_runs.append((starts[i], ends[j-1]))
        j = i
    super_runs.reverse()
    return super_runs


def fit_cca_alignment(
    X_source: np.ndarray,
    Y_target: np.ndarray,
    *,
    n_components: int = 10,
    max_iter: int = 500,
    scale: bool = True,
    random_state: int | None = 0,
):
    """Fit a CCA model to align two latent spaces.

    This is a thin, explicitly-typed wrapper around
    :class:`sklearn.cross_decomposition.CCA` that keeps the dependency
    optional (lazy import) and surfaces the canonical projections.

    Parameters
    ----------
    X_source:
        Array of shape (n_samples, d_x); e.g. an alternative feature
        representation for the indexed SPDs (such as ultrametric-based
        features from :func:`spd_tree_feature_matrix`).
    Y_target:
        Array of shape (n_samples, d_y); e.g. the PCA latent features
        used when building the index (``data.latent['pca']``).
    n_components:
        Number of canonical components to keep.
    max_iter:
        Maximum number of iterations for the CCA solver.
    scale:
        Whether to standardize each view before fitting (mirrors the
        default behavior of ``sklearn``).
    random_state:
        Optional seed for the underlying solver; set to ``None`` for
        non-deterministic behavior.

    Returns
    -------
    cca:
        The fitted CCA estimator.
    X_c:
        Canonical representation of ``X_source`` of shape
        (n_samples, n_components).
    Y_c:
        Canonical representation of ``Y_target`` of shape
        (n_samples, n_components).
    """

    try:
        from sklearn.cross_decomposition import CCA  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "scikit-learn is required for fit_cca_alignment; "
            "install scikit-learn to use CCA-based alignment."
        ) from exc

    X_source = np.asarray(X_source, dtype=float)
    Y_target = np.asarray(Y_target, dtype=float)
    if X_source.shape[0] != Y_target.shape[0]:
        raise ValueError("X_source and Y_target must have the same number of rows (samples).")

    n = X_source.shape[0]
    if n < 2:
        raise ValueError("CCA requires at least two samples.")

    n_comp_eff = min(n_components, X_source.shape[1], Y_target.shape[1])
    if n_comp_eff <= 0:
        raise ValueError("n_components must be >= 1 and <= min(d_x, d_y, n_samples).")

    cca = CCA(
        n_components=n_comp_eff,
        max_iter=max_iter,
        scale=scale,
    )

    X_c, Y_c = cca.fit_transform(X_source, Y_target)
    return cca, X_c, Y_c


def project_with_cca(
    cca,
    X_source_new: np.ndarray,
):
    """Project new source features into CCA canonical space.

    Parameters
    ----------
    cca:
        A fitted CCA instance returned by :func:`fit_cca_alignment`.
    X_source_new:
        New samples in the *source* feature space, shape
        (n_new, d_x). These are projected into the canonical space that
        was learned between (X_source, Y_target).

    Returns
    -------
    X_c_new:
        Canonical representation of the new samples, shape
        (n_new, n_components).
    """

    X_source_new = np.asarray(X_source_new, dtype=float)
    if X_source_new.ndim != 2:
        raise ValueError("X_source_new must be a 2D array of shape (n_new, d_x).")

    # sklearn's CCA.transform returns (X_c, Y_c); for new samples we
    # only have X, so we ignore the second output.
    X_c_new, _ = cca.transform(X_source_new)
    return X_c_new


def fit_supervised_cca_alignment(
    X_source: np.ndarray,
    labels: np.ndarray,
    Y_aux: np.ndarray | None = None,
    *,
    target_mode: str = "onehot_centroid",
    n_components: int = 10,
    max_iter: int = 500,
    scale: bool = True,
    random_state: int | None = 0,
):
    """Fit a label-aware CCA for discriminative alignment.

    Unlike vanilla CCA between two unsupervised views, this function
    constructs a *label-informed* target view so that the learned
    canonical projections capture class structure.

    Parameters
    ----------
    X_source:
        Array of shape (n_samples, d_x); e.g. ultrametric features from
        :func:`spd_tree_feature_matrix`.
    labels:
        1D integer array of cluster labels for the training samples,
        shape (n_samples,).
    Y_aux:
        Optional auxiliary features (e.g. PCA latents) to concatenate
        with the label-derived target. Shape (n_samples, d_aux). If
        ``None``, only the label-derived view is used.
    target_mode:
        How to construct the label-informed target view:

        - ``"onehot"``: one-hot encoding of labels (n_samples, n_classes).
        - ``"centroid"``: replace each sample with its class centroid in
          X_source space (n_samples, d_x).
        - ``"onehot_centroid"`` (default): concatenate one-hot and
          centroid representations.
        - ``"onehot_aux"``: concatenate one-hot with Y_aux (requires
          Y_aux).
    n_components:
        Number of canonical components.
    max_iter:
        Maximum iterations for the CCA solver.
    scale:
        Whether to standardize views before fitting.
    random_state:
        Optional seed for reproducibility.

    Returns
    -------
    cca:
        Fitted CCA estimator.
    Xc_train:
        Canonical representation of X_source, shape (n_samples, n_comp).
    class_centroids_canonical:
        Per-class centroids in canonical space, shape (n_classes, n_comp).
        Useful for nearest-centroid assignment of new samples.
    unique_labels:
        Sorted unique class labels, aligned with rows of
        ``class_centroids_canonical``.
    """

    try:
        from sklearn.cross_decomposition import CCA  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for fit_supervised_cca_alignment."
        ) from exc

    X_source = np.asarray(X_source, dtype=float)
    labels = np.asarray(labels).ravel()
    n = X_source.shape[0]
    if labels.shape[0] != n:
        raise ValueError("labels must have the same length as X_source rows.")

    unique_labels = np.unique(labels)
    n_classes = len(unique_labels)
    label_to_idx = {int(lab): i for i, lab in enumerate(unique_labels)}

    # --- Build label-informed target view ---
    # One-hot encoding
    onehot = np.zeros((n, n_classes), dtype=float)
    for i, lab in enumerate(labels):
        onehot[i, label_to_idx[int(lab)]] = 1.0

    # Centroid representation: each sample replaced by its class centroid
    centroids_source = np.zeros((n_classes, X_source.shape[1]), dtype=float)
    for lab in unique_labels:
        mask = labels == lab
        centroids_source[label_to_idx[int(lab)]] = X_source[mask].mean(axis=0)
    centroid_repr = np.vstack([centroids_source[label_to_idx[int(lab)]] for lab in labels])

    if target_mode == "onehot":
        Y_target = onehot
    elif target_mode == "centroid":
        Y_target = centroid_repr
    elif target_mode == "onehot_centroid":
        Y_target = np.hstack([onehot, centroid_repr])
    elif target_mode == "onehot_aux":
        if Y_aux is None:
            raise ValueError("target_mode='onehot_aux' requires Y_aux.")
        Y_aux = np.asarray(Y_aux, dtype=float)
        if Y_aux.shape[0] != n:
            raise ValueError("Y_aux must have the same number of rows as X_source.")
        Y_target = np.hstack([onehot, Y_aux])
    else:
        raise ValueError(f"Unknown target_mode: {target_mode}")

    # Effective number of components
    n_comp_eff = min(n_components, X_source.shape[1], Y_target.shape[1], n - 1)
    if n_comp_eff <= 0:
        raise ValueError("n_components must be >= 1.")

    cca = CCA(
        n_components=n_comp_eff,
        max_iter=max_iter,
        scale=scale,
    )
    Xc_train, _ = cca.fit_transform(X_source, Y_target)

    # Compute per-class centroids in canonical space
    class_centroids_canonical = np.zeros((n_classes, n_comp_eff), dtype=float)
    for lab in unique_labels:
        mask = labels == lab
        class_centroids_canonical[label_to_idx[int(lab)]] = Xc_train[mask].mean(axis=0)

    return cca, Xc_train, class_centroids_canonical, unique_labels


def assign_with_supervised_cca(
    cca,
    class_centroids_canonical: np.ndarray,
    unique_labels: np.ndarray,
    X_source_new: np.ndarray,
    Xc_train: np.ndarray | None = None,
    labels_train: np.ndarray | None = None,
    *,
    strategy: str = "centroid",
    n_neighbors: int = 20,
):
    """Assign cluster labels to new samples using supervised CCA projections.

    Parameters
    ----------
    cca:
        Fitted CCA from :func:`fit_supervised_cca_alignment`.
    class_centroids_canonical:
        Per-class centroids in canonical space.
    unique_labels:
        Sorted unique labels aligned with centroids.
    X_source_new:
        New samples in the source feature space, shape (n_new, d_x).
    Xc_train:
        Training samples in canonical space (needed for KNN strategies).
    labels_train:
        Training labels (needed for KNN strategies).
    strategy:
        Assignment strategy:

        - ``"centroid"``: nearest class centroid in canonical space.
        - ``"knn_majority"``: KNN majority vote in canonical space.
        - ``"knn_weighted"``: distance-weighted KNN vote.
        - ``"hybrid"``: centroid if confident, else KNN.
    n_neighbors:
        Number of neighbors for KNN strategies.

    Returns
    -------
    assigned:
        1D array of assigned cluster labels.
    distances:
        1D array of distances to the assigned centroid (or average
        neighbor distance for KNN).
    """

    X_source_new = np.asarray(X_source_new, dtype=float)
    Xc_new, _ = cca.transform(X_source_new)

    n_new = Xc_new.shape[0]
    n_classes = class_centroids_canonical.shape[0]

    # Squared distances to each centroid
    d2_centroids = (
        (Xc_new[:, None, :] - class_centroids_canonical[None, :, :]) ** 2
    ).sum(axis=2)  # (n_new, n_classes)

    def _centroid_assign():
        best_idx = np.argmin(d2_centroids, axis=1)
        assigned = unique_labels[best_idx]
        dists = np.sqrt(d2_centroids[np.arange(n_new), best_idx])
        return assigned, dists

    def _knn_assign(weighted: bool):
        if Xc_train is None or labels_train is None:
            raise ValueError("KNN strategies require Xc_train and labels_train.")
        from sklearn.neighbors import NearestNeighbors

        knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
        knn.fit(Xc_train)
        knn_dist, knn_idx = knn.kneighbors(Xc_new)

        assigned = []
        avg_dists = []
        eps = 1e-8
        for i in range(n_new):
            neigh_labels = labels_train[knn_idx[i]]
            if weighted:
                weights = 1.0 / (knn_dist[i] + eps)
                scores = {}
                for lab, w in zip(neigh_labels, weights):
                    scores[lab] = scores.get(lab, 0.0) + float(w)
                best_lab = max(scores.items(), key=lambda x: x[1])[0]
            else:
                uniq, counts = np.unique(neigh_labels, return_counts=True)
                best_lab = uniq[np.argmax(counts)]
            assigned.append(int(best_lab))
            avg_dists.append(float(knn_dist[i].mean()))
        return np.array(assigned, dtype=int), np.array(avg_dists)

    def _hybrid_assign():
        cen_assigned, cen_dists = _centroid_assign()
        # Confidence: ratio of distance to nearest vs second-nearest centroid
        sorted_d2 = np.sort(d2_centroids, axis=1)
        ratio = np.sqrt(sorted_d2[:, 0]) / (np.sqrt(sorted_d2[:, 1]) + 1e-8)
        confident = ratio < 0.7  # threshold for confident assignment

        if confident.all():
            return cen_assigned, cen_dists

        knn_assigned, knn_dists = _knn_assign(weighted=True)
        final_assigned = np.where(confident, cen_assigned, knn_assigned)
        final_dists = np.where(confident, cen_dists, knn_dists)
        return final_assigned.astype(int), final_dists

    if strategy == "centroid":
        return _centroid_assign()
    elif strategy == "knn_majority":
        return _knn_assign(weighted=False)
    elif strategy == "knn_weighted":
        return _knn_assign(weighted=True)
    elif strategy == "hybrid":
        return _hybrid_assign()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")