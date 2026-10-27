"""Shared by the Fig 4 compute scripts (no plotting; safe in sbatch): the gene-set workload, the two whole-tile
baselines and the retrieval readouts (common3: NX5, Overlap, the same definitions as Fig 3 and 19 Oct).

Workload (SPINDLE_PARTIAL.md §6, as 19 Oct, user 26 Oct: contiguous sets only): per held-out tile and size
|S| = 4 / 8 / 16 / 32, a run of consecutive genes in one block of the query's home niche (the niche of its exact-d_B
nearest index tile, cache D.npy): a block drawn uniformly among the niche's blocks with at least |S| genes (the size
is skipped when there is none), then a uniform start. One generator per (dataset, seed), seeded by the seed. The
query is the held-out tile's shrunk covariance cut to S (queries.tile_query); every niche is searched. Ground truth:
the exact d_B^S (SpindlePartial.distances).

Baselines (experiment code, ported from the pre-cleanup src/spindle_dev/partial_search.py, in git history): the gene-set
query is completed to all p genes, then searched as a whole tile with Spindle (Spindle.query_vectors +
Spindle.distances, exact d_B); no second shrinkage.
  padding     Q_S on S x S, the mean of Q_S's variances times I on the other genes, no cross terms.
  imputation  the Gaussian conditional of the other genes given Q_S, prior the mean index covariance of the
              predicted niche (the niche whose mean covariance is nearest the query in d_B^S on its own blocks).
Stored bytes of the baselines: padding = Spindle's block logs; imputation = those plus one float32 upper triangle of
a p x p mean covariance per niche.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig3_whole_tile"))
import common3 as c3  # noqa: E402,F401  (NX5, Overlap; puts src/ on sys.path)
from common3 import data  # noqa: E402,F401
from spindle_dev.metrics import LOG_FLOOR, as_cov, batched_log_spd, sym_to_vec  # noqa: E402

OUT = data.RESULTS / "fig4_gene_set"
SIZES = [4, 8, 16, 32]
BAND_SIZE = 16


def workload(layout, D, seed):
    """[(query j, home niche, size, block, start, genes)] over the held-out tiles in test order (D rows)."""
    labels = np.asarray(layout.labels).astype(int)
    rng = np.random.default_rng(seed)
    out = []
    for j in range(len(D)):
        k = int(labels[int(np.argmin(D[j]))])
        runs = layout.block_dict[k]
        perm = np.asarray(layout.perm_list[k])
        for L in SIZES:
            ok = [b for b, (s, e) in enumerate(runs) if e - s >= L]
            if not ok:
                continue
            b = int(ok[rng.integers(len(ok))])
            s0, e0 = runs[b]
            a = int(rng.integers(e0 - s0 - L + 1))
            out.append((j, k, L, b, a, perm[s0 + a:s0 + a + L].astype(np.int64)))
    return out


def niche_means(layout, covs, train):
    """{niche: mean shrunk covariance of its index tiles (p x p, float64)}."""
    labels = np.asarray(layout.labels).astype(int)
    return {k: np.mean([as_cov(covs[train[r]]) for r in np.flatnonzero(labels == k)], axis=0)
            for k in sorted(set(labels.tolist()))}


def mean_bytes(means):
    return int(sum(4 * m.shape[0] * (m.shape[0] + 1) // 2 for m in means.values()))


def pad(cov_s, genes, p):
    out = np.eye(p) * float(np.mean(np.diag(cov_s)))
    out[np.ix_(genes, genes)] = cov_s
    return out


def predict_niche(cov_s, genes, means, blocks):
    """The niche whose mean covariance is nearest the query in d_B^S on that niche's blocks (blocks: partial.blocks)."""
    col = np.full(len(next(iter(means.values()))), -1, dtype=np.int64)
    col[genes] = np.arange(len(genes))
    best, best_d = None, np.inf
    for k, M in means.items():
        tot = 0.0
        for g in blocks[k]:
            gs = g[col[g] >= 0]
            if len(gs) == 0:
                continue
            cq = col[gs]
            lq, lm = sym_to_vec(batched_log_spd(np.stack([cov_s[np.ix_(cq, cq)], M[np.ix_(gs, gs)]]), LOG_FLOOR))
            tot += float(((lq - lm) ** 2).sum())
        if tot < best_d:
            best, best_d = k, tot
    return best


def impute(cov_s, genes, M):
    p = M.shape[0]
    rest = np.setdiff1d(np.arange(p), genes)
    out = np.empty((p, p))
    out[np.ix_(genes, genes)] = cov_s
    M_RS = M[np.ix_(rest, genes)]
    A = cho_solve(cho_factor(M[np.ix_(genes, genes)]), M_RS.T).T
    S_RS = A @ cov_s
    S_RR = M[np.ix_(rest, rest)] - A @ M_RS.T + S_RS @ A.T
    out[np.ix_(rest, genes)] = S_RS
    out[np.ix_(genes, rest)] = S_RS.T
    out[np.ix_(rest, rest)] = 0.5 * (S_RR + S_RR.T)
    return out


def exact_query_logs(partial, cov_s, genes):
    """The query's block logs as SpindlePartial.distances computes them inside its scan (spindle.py:139-150): one per
    (niche, block meeting S). Timed on their own (c_timing.py), they split the exact gene-set time into query logs and
    search; the per-block bookkeeping is counted with the logs, so the search time is, if anything, underestimated."""
    col = np.full(partial.p, -1, dtype=np.int64)
    col[genes] = np.arange(len(genes))
    out = []
    for k in partial.niches:
        for g in partial.blocks[k]:
            pos = np.flatnonzero(col[g] >= 0)
            if len(pos) == 0:
                continue
            cq = col[g[pos]]
            out.append(sym_to_vec(batched_log_spd(cov_s[np.ix_(cq, cq)][None], LOG_FLOOR))[0])
    return out


def n_segments(plite, genes):
    """{niche: number of segments of S} (runs of consecutive positions inside one block)."""
    out = {}
    for k, nc in plite.niches.items():
        blk, pos = nc.gene_block[genes], nc.gene_pos[genes]
        out[k] = int(sum(1 + int(np.sum(np.diff(np.sort(pos[blk == b])) > 1)) for b in np.unique(blk)))
    return out
