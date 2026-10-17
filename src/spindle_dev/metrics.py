"""SPD matrix helpers shared by the three indexes."""

from __future__ import annotations

import numpy as np

# Spindle and the interval index shrink every covariance toward a scaled identity before any log:
# S -> (1 - a) S + a (tr S / p) I. Set to 0.0 to turn shrinkage off. Spindle-lite uses raw covariances.
SHRINK_ALPHA = 0.1
LOG_FLOOR = 1e-6  # eigenvalue floor of the Spindle and interval-index logs


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


def prepare_cov(entry) -> np.ndarray:
    """float64 covariance with the configured shrinkage applied."""
    return shrink_cov(np.asarray(raw_cov(entry), dtype=np.float64))


def query_cov_s(query_cov, genes, p: int) -> np.ndarray:
    """A gene-set query's prepared |S| x |S| covariance in ``genes`` order.

    A full p x p query is shrunk over all p genes and then cut, as the tiles are. An |S| x |S| query
    is shrunk on its own.
    """
    genes = np.asarray(genes)
    cov = np.asarray(raw_cov(query_cov), dtype=np.float64)
    if cov.shape[0] == p:
        return prepare_cov(cov)[np.ix_(genes, genes)]
    if cov.shape[0] != len(genes):
        raise ValueError(f"query covariance is {cov.shape[0]} x {cov.shape[0]}; expected {p} (all genes) "
                         f"or {len(genes)} (the gene set)")
    return shrink_cov(cov)


def log_spd(A: np.ndarray, eps: float = 1e-8) -> np.ndarray:
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
