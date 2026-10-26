"""Gene-set queries from gene names (Methods, partial queries).

Every constructor returns ``(cov_s, idx)``: the |S| x |S| query covariance in ``idx`` order and the global gene
indices of S, accepted unchanged by ``SpindlePartial.search`` / ``distances`` and ``SpindleLitePartial.search`` /
``query_vectors``. Queries are on the shrunk scale of the index tiles and are never shrunk again.

- ``gene_indices``: gene names -> global indices; unknown genes dropped, duplicates removed (both reported).
- ``tile_query``: a covariance over all p genes (e.g. a tile's, already shrunk) cut to S.
- ``template_query``: a gene list alone, Methods Eq. (signature):
  Q_S = Omega_S^1/2 [(1 - rho) I + rho 11^T] Omega_S^1/2, rho = 0.5, Omega_S = the q = 0.9 quantile over the index
  tiles of each gene's (shrunk) variance. ``variance_quantiles`` computes that p-vector once from the index tiles'
  covariances; keep it beside the index (Spindle-lite partial stores no tile covariances).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from .metrics import as_cov, query_cov_s

RHO = 0.5
VAR_QUANTILE = 0.9


def gene_indices(panel_genes: Sequence[str], names: Sequence[str], *, case_sensitive: bool = True
                 ) -> Tuple[np.ndarray, List[str], List[str]]:
    """(idx, found, missing): global indices of ``names`` in ``panel_genes`` (the index genes), in the order given,
    each gene once; ``found`` are the panel's names for ``idx``, ``missing`` the names not in the panel."""
    fold = (lambda g: str(g)) if case_sensitive else (lambda g: str(g).upper())
    pos = {}
    for i, g in enumerate(panel_genes):
        pos.setdefault(fold(g), i)
    idx, missing = [], []
    for g in names:
        i = pos.get(fold(g))
        if i is None:
            if g not in missing:
                missing.append(g)
        elif i not in idx:
            idx.append(i)
    return np.asarray(idx, dtype=np.int64), [str(panel_genes[i]) for i in idx], missing


def tile_query(cov, idx) -> Tuple[np.ndarray, np.ndarray]:
    """A p x p covariance (shrunk, as the index tiles) cut to the genes ``idx``."""
    idx = np.asarray(idx, dtype=np.int64)
    cov = as_cov(cov)
    return query_cov_s(cov, idx, cov.shape[0]), idx


def variance_quantiles(index_covs, q: float = VAR_QUANTILE) -> np.ndarray:
    """The q-quantile over the index tiles of every gene's variance (diagonal of the shrunk covariances), (p,)."""
    return np.quantile(np.stack([np.diag(as_cov(c)) for c in index_covs]), q, axis=0)


def template_query(omega, idx, rho: float = RHO) -> Tuple[np.ndarray, np.ndarray]:
    """Methods Eq. (signature) on the genes ``idx``: variances ``omega[idx]`` (``variance_quantiles``), every pair
    at correlation ``rho``. SPD for -1/(|S| - 1) < rho < 1 and positive variances (checked)."""
    idx = np.asarray(idx, dtype=np.int64)
    s = len(idx)
    if s == 0:
        raise ValueError("empty gene set")
    if not (s == 1 or -1.0 / (s - 1) < rho < 1.0):
        raise ValueError(f"rho = {rho} does not give an SPD template on {s} genes")
    sd = np.sqrt(np.asarray(omega, dtype=np.float64)[idx])
    if not np.all(sd > 0):
        raise ValueError("template variances must be positive")
    R = (1.0 - rho) * np.eye(s) + rho * np.ones((s, s))
    cov_s = sd[:, None] * R * sd[None, :]
    np.linalg.cholesky(cov_s)
    return cov_s, idx
