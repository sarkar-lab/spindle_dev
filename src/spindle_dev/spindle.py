"""Spindle: exact search over stored block logs.

Every tile is scored in its own niche's gene order and blocks; scores are pooled over niches:

    d_B(q, t)   = sqrt(sum_b ||log q_b - log t_b||_F^2) / sqrt(p)
    d_B^S(q, t) = sqrt(sum_b ||log q[S cap b] - log t[S cap b]||_F^2) / sqrt(|S|)

Covariances arrive shrunk (preprocessing) and are used as given. ``Spindle`` stores the float32 upper
triangles of every training tile's block logs; ``SpindlePartial`` also stores the block covariances and
eigendecomposes the sub-blocks S cap b at query time.

``search`` returns training rows (positions in ``data.labels``) best first, their scores and the seconds
taken; ``last_timing`` splits those into the query's own logs (``query_s``) and the scan (``scan_s``).
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .metrics import LOG_FLOOR, as_cov, batched_log_spd, log_spd, query_cov_s, sym_to_vec


class BlockVectorizer:
    """One niche's layout: block logs -> a vector whose squared L2 norm is d_B^2."""

    def __init__(self, perm, block_runs):
        self.perm = np.asarray(perm)
        self.runs = [(int(s), int(e)) for s, e in block_runs]
        self.p = sum(e - s for s, e in self.runs)
        self.iu = [np.triu_indices(e - s) for s, e in self.runs]
        self.w = [np.where(i == j, 1.0, np.sqrt(2.0)) / np.sqrt(self.p) for i, j in self.iu]
        self.dim = sum(len(w) for w in self.w)

    def block_logs(self, cov):
        c = np.asarray(cov, dtype=np.float64)[np.ix_(self.perm, self.perm)]
        return [log_spd(c[s:e, s:e], eps=LOG_FLOOR) for s, e in self.runs]

    def vector(self, cov, dtype=np.float64):
        return np.concatenate([(L[iu] * w).astype(dtype) for L, iu, w in zip(self.block_logs(cov), self.iu, self.w)])


def niche_layouts(data):
    """{niche: (rows of its tiles, BlockVectorizer)} in sorted niche order."""
    labels = np.asarray(data.labels).astype(int)
    return {k: (np.flatnonzero(labels == k), BlockVectorizer(data.perm_list[k], data.block_dict[k]))
            for k in sorted(set(labels.tolist()))}


def block_cov_stacks(data, train_covs):
    """{niche: [(n_niche, b, b) covariances of its tiles on each block's genes]}, rows in niche order."""
    labels = np.asarray(data.labels).astype(int)
    out = {}
    for k in sorted(set(labels.tolist())):
        rows = np.flatnonzero(labels == k)
        blocks = [np.asarray(data.perm_list[k])[s:e] for s, e in data.block_dict[k]]
        stacks = [np.empty((len(rows), len(g), len(g))) for g in blocks]
        for i, r in enumerate(rows):
            cov = as_cov(train_covs[r])
            for stack, g in zip(stacks, blocks):
                stack[i] = cov[np.ix_(g, g)]
        out[k] = stacks
    return out


def _top(scores, c):
    order = np.argsort(scores, kind="stable")
    return order if c is None else order[:c]


class Spindle:
    """Whole-tile queries: exact d_B scan over float32 block-log vectors."""

    def __init__(self, data, train_covs):
        self.layouts = niche_layouts(data)
        self.n = len(data.labels)
        self.X = {k: np.stack([vec.vector(as_cov(train_covs[t]), np.float32) for t in idx])
                  for k, (idx, vec) in self.layouts.items()}

    def nbytes(self):
        return int(sum(x.nbytes for x in self.X.values()))

    def query_vectors(self, query_cov):
        """Per-niche float32 block-log vectors of a query covariance."""
        cov = as_cov(query_cov)
        return {k: vec.vector(cov, np.float32) for k, (_, vec) in self.layouts.items()}

    def distances(self, qv):
        """d_B from the query vectors to every training tile."""
        d = np.empty(self.n)
        for k, (idx, _) in self.layouts.items():
            d[idx] = np.sqrt(np.maximum(((self.X[k] - qv[k]) ** 2).sum(1, dtype=np.float64), 0.0))
        return d

    def search(self, query_cov, c: Optional[int] = None):
        t0 = time.perf_counter()
        qv = self.query_vectors(query_cov)
        t1 = time.perf_counter()
        d = self.distances(qv)
        rows = _top(d, c)
        t2 = time.perf_counter()
        self.last_timing = {"query_s": t1 - t0, "scan_s": t2 - t1}
        return rows, d[rows], t2 - t0


class SpindlePartial:
    """Gene-set queries: exact d_B^S from stored block covariances and block logs."""

    def __init__(self, data, train_covs):
        labels = np.asarray(data.labels).astype(int)
        self.p = len(data.perm_list[labels[0]])
        self.niches = sorted(set(labels.tolist()))
        self.rows = {k: np.flatnonzero(labels == k) for k in self.niches}
        self.blocks = {k: [np.asarray(data.perm_list[k])[s:e] for s, e in data.block_dict[k]] for k in self.niches}
        self.covs = block_cov_stacks(data, train_covs)
        self.logs = {k: [sym_to_vec(batched_log_spd(s, LOG_FLOOR)).astype(np.float32) for s in self.covs[k]]
                     for k in self.niches}
        self.n = len(labels)

    def nbytes(self):
        """float32 upper triangles of the block logs plus those of the block covariances."""
        tri = sum(len(self.rows[k]) * sum(len(g) * (len(g) + 1) // 2 for g in self.blocks[k]) for k in self.niches)
        return 8 * tri

    def distances(self, cov_s, genes):
        """d_B^S to every training tile; ``cov_s`` is the |S| x |S| query in ``genes`` order."""
        genes = np.asarray(genes)
        col = np.full(self.p, -1, dtype=np.int64)
        col[genes] = np.arange(len(genes))
        d = np.empty(self.n)
        for k in self.niches:
            tot = np.zeros(len(self.rows[k]))
            for g, cov, logv in zip(self.blocks[k], self.covs[k], self.logs[k]):
                pos = np.flatnonzero(col[g] >= 0)
                if len(pos) == 0:
                    continue
                cq = col[g[pos]]
                lq = sym_to_vec(batched_log_spd(cov_s[np.ix_(cq, cq)][None], LOG_FLOOR))[0]
                if len(pos) == len(g):
                    lt = logv
                else:
                    lt = sym_to_vec(batched_log_spd(cov[:, pos[:, None], pos[None, :]], LOG_FLOOR))
                tot += ((lt - lq) ** 2).sum(1)
            d[self.rows[k]] = np.sqrt(tot / len(genes))
        return d

    def search(self, query_cov, genes, c: Optional[int] = None):
        """``query_cov`` is the full p x p query or the |S| x |S| one in ``genes`` order."""
        t0 = time.perf_counter()
        cov_s = query_cov_s(query_cov, genes, self.p)
        t1 = time.perf_counter()
        d = self.distances(cov_s, genes)
        rows = _top(d, c)
        t2 = time.perf_counter()
        self.last_timing = {"query_s": t1 - t0, "scan_s": t2 - t1}
        return rows, d[rows], t2 - t0
