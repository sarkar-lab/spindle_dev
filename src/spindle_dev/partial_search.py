"""Index-free baselines for partial (gene-set) queries: fill in the missing genes, then search whole tiles.

A partial query is a covariance Q on a gene set S (|S| x |S|, rows in ``genes`` order, already prepared
by tiers.prepare_cov / query_cov_s). Both baselines turn it into a p x p covariance and hand it to the
whole-tile Spindle-Exact tier (d_B over all p genes); neither shrinks again.

* Padding: Q on S x S, a scaled identity v I on the missing genes (v = mean diagonal of Q, the target that
  shrinkage uses), zero covariance between S and the missing genes.
* Imputation: the Gaussian conditional given Q, with a niche-mean covariance M as the prior (R = missing):

      Sigma_SS = Q,  Sigma_RS = A Q,  Sigma_RR = M_RR - A M_SR + A Q A^T,  A = M_RS M_SS^-1,

  i.e. M_RR - M_RS M_SS^-1 (M_SS - Q) M_SS^-1 M_SR. It is PSD whenever M is PD and Q is PSD.
  M is the mean (shrunk) covariance of the predicted niche: the niche whose mean is nearest Q by d_B^S
  in that niche's own blocks (predict_niche).
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .interval_dp import LOG_FLOOR, batched_log_spd, sym_to_vec


def pad_query_scaled_identity(cov_s: np.ndarray, genes: Sequence[int], p: int) -> np.ndarray:
    """p x p query: ``cov_s`` on genes x genes, mean(diag cov_s) * I on the other genes, zero cross terms."""
    genes = np.asarray(genes)
    cov_s = np.asarray(cov_s, dtype=np.float64)
    if cov_s.shape != (len(genes), len(genes)):
        raise ValueError(f"cov_s is {cov_s.shape}; expected {(len(genes), len(genes))}")
    out = np.eye(p) * float(np.mean(np.diag(cov_s)))
    out[np.ix_(genes, genes)] = cov_s
    return out


def impute_query_conditional(cov_s: np.ndarray, genes: Sequence[int], mean_cov: np.ndarray) -> np.ndarray:
    """p x p query: the Gaussian conditional of the missing genes given ``cov_s`` on genes, prior ``mean_cov``."""
    genes = np.asarray(genes)
    cov_s = np.asarray(cov_s, dtype=np.float64)
    M = np.asarray(mean_cov, dtype=np.float64)
    p = M.shape[0]
    rest = np.setdiff1d(np.arange(p), genes)
    out = np.empty((p, p))
    out[np.ix_(genes, genes)] = cov_s
    if len(rest):
        M_RS = M[np.ix_(rest, genes)]
        A = cho_solve(cho_factor(M[np.ix_(genes, genes)]), M_RS.T).T  # M_RS M_SS^-1 (M_SS symmetric)
        S_RS = A @ cov_s
        S_RR = M[np.ix_(rest, rest)] - A @ M_RS.T + S_RS @ A.T
        out[np.ix_(rest, genes)] = S_RS
        out[np.ix_(genes, rest)] = S_RS.T
        out[np.ix_(rest, rest)] = 0.5 * (S_RR + S_RR.T)
    return out


def predict_niche(cov_s: np.ndarray, genes: Sequence[int], niche_means: Dict[int, np.ndarray],
                  blocks: Dict[int, Sequence[np.ndarray]], floor: float = LOG_FLOOR) -> Tuple[int, Dict[int, float]]:
    """The niche whose mean covariance is nearest the query by d_B^S in that niche's blocks.

    ``blocks[k]`` lists niche k's blocks as arrays of global gene indices (ExactPartialTier.blocks).
    -> (best niche, {niche: d_B^S to its mean}).
    """
    genes = np.asarray(genes)
    cov_s = np.asarray(cov_s, dtype=np.float64)
    col = np.full(len(next(iter(niche_means.values()))), -1, dtype=np.int64)
    col[genes] = np.arange(len(genes))
    d = {}
    for k, M in niche_means.items():
        tot = 0.0
        for g in blocks[k]:
            gs = g[col[g] >= 0]
            if len(gs) == 0:
                continue
            cq = col[gs]
            pair = np.stack([cov_s[np.ix_(cq, cq)], M[np.ix_(gs, gs)]])
            lq, lm = sym_to_vec(batched_log_spd(pair, floor))
            tot += float(((lq - lm) ** 2).sum())
        d[k] = np.sqrt(tot / len(genes))
    return min(d, key=d.get), d


__all__ = ["pad_query_scaled_identity", "impute_query_conditional", "predict_niche"]
