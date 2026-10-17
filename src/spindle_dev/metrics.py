"""SPD matrix helpers shared by the three indexes."""

from __future__ import annotations

import numpy as np

# Every tile covariance is shrunk toward a scaled identity once, when it is computed
# (preprocessing._tile_cov): S -> (1 - a) S + a (tr S / p) I. The layout and all three indexes take the
# shrunk covariances as given and never shrink again. Set to 0.0 to turn shrinkage off everywhere.
SHRINK_ALPHA = 0.1
LOG_FLOOR = 1e-6  # eigenvalue floor of every matrix log


def shrink_cov(S, alpha: float | None = None) -> np.ndarray:
    """(1 - alpha) S + alpha * mean(diag S) * I; alpha defaults to SHRINK_ALPHA."""
    S = np.asarray(S, dtype=np.float64)
    a = SHRINK_ALPHA if alpha is None else alpha
    if a == 0:
        return S
    return (1.0 - a) * S + a * (np.trace(S) / S.shape[0]) * np.eye(S.shape[0])


def raw_cov(entry) -> np.ndarray:
    """The covariance of a stored entry (an array, or a dict with 'cov')."""
    return entry if not isinstance(entry, dict) else entry.get("cov", entry.get("matrix", entry))


def as_cov(entry) -> np.ndarray:
    """A stored (already shrunk) covariance as float64."""
    return np.asarray(raw_cov(entry), dtype=np.float64)


def query_cov_s(query_cov, genes, p: int) -> np.ndarray:
    """A gene-set query's |S| x |S| covariance in ``genes`` order: a full p x p query (shrunk over all p
    genes, as the tiles) is cut; an |S| x |S| query is used as given (the caller shrinks it)."""
    genes = np.asarray(genes)
    cov = as_cov(query_cov)
    if cov.shape[0] == p:
        return cov[np.ix_(genes, genes)]
    if cov.shape[0] != len(genes):
        raise ValueError(f"query covariance is {cov.shape[0]} x {cov.shape[0]}; expected {p} (all genes) "
                         f"or {len(genes)} (the gene set)")
    return cov


def log_spd(A: np.ndarray, eps: float = LOG_FLOOR) -> np.ndarray:
    """SPD matrix logarithm via eigendecomposition, eigenvalues floored at ``eps``."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T


def batched_log_spd(mats: np.ndarray, floor: float = LOG_FLOOR) -> np.ndarray:
    """Matrix logarithm of a stack (..., s, s) of SPD matrices, eigenvalues floored."""
    mats = 0.5 * (mats + np.swapaxes(mats, -1, -2))
    w, V = np.linalg.eigh(mats)
    w = np.log(np.maximum(w, floor))
    return (V * w[..., None, :]) @ np.swapaxes(V, -1, -2)


def sym_to_vec(L: np.ndarray) -> np.ndarray:
    """(..., s, s) symmetric -> (..., s(s+1)/2) upper triangle with ||vec|| = ||L||_F."""
    iu = np.triu_indices(L.shape[-1])
    return L[..., iu[0], iu[1]] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


def spd_to_correlation(C: np.ndarray) -> np.ndarray:
    """Correlation matrix of a covariance (zero variances guarded, entries clipped to [-1, 1])."""
    d = np.sqrt(np.diag(C))
    d[d == 0] = 1e-12
    D_inv = np.diag(1.0 / d)
    return np.clip(D_inv @ C @ D_inv, -1.0, 1.0)
