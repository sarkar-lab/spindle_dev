"""Four breast gene programs searched as queries in five tumour sections (Fig 8; R8_pan_cancer.ipynb), co-variation
framing. Ported from experiments/fig8_pan_cancer/{common8,c_xtumour,c_gradient,c_summary}.py (in git history); section
loader and statistics in sections.py.

Programs: signatures.PAN_CANCER_MODULES (cytotoxic T, myeloid, stromal / perivascular, B / plasma), every gene an index
gene of all five sections (checked; a missing gene stops the run). Sections: breast, lung, pancreas, brain, skin.

RULES (fixed before the run):
- EXEMPLAR, one per program, fixed for every section and seed: among breast's seed-0 index tiles, the tile with the
  largest robust co-variation c on S (robust_c, the gene-set pilot's rule: c = mean off-diagonal correlation of S over
  the tile's cells, raw counts; >= 80 cells, >= 0.5 S counts per cell, no cell > 15 % of the S counts, dropping the top
  3 cells keeps >= 70 % of c). No robust tile -> the program is dropped (reported). R_S = the correlation of its shrunk
  covariance cut to S; q_g = the share of breast seed-0 index tiles whose shrunk variance of gene g is <= the
  exemplar's.
- QUERY in section X, seed s (quantile-matched scale, Fig 8 decision of 9 Oct): sigma_g = sqrt of the q_g quantile
  of gene g's shrunk variance over X's seed-s index tiles; Q = diag(sigma) R_S diag(sigma).
- HITS = the first c = 5 % of the index tiles (n_hits): Spindle-naive ("naive_partial", exact d_B^S); Spindle
  ("spindle_partial", the first c of search_vectors at the default g; fewer if it returns fewer, recorded).
  Controls, computed and reported, never claimed against: "D" = diag(sigma^2) (variances only, Spindle-naive) and
  "X" = the c index tiles with the highest tile score. The exemplar tile is never a hit and is left out of every
  background.
- READOUTS per (program, section, seed, method), against the other index tiles (background):
  gain = median over the background of the correlation gap minus its median over the hits (gap = RMS difference
  between a tile's S x S correlation, from its shrunk covariance, and R_S, off-diagonal pairs; > 0 = the hits
  co-vary more like the exemplar than a typical tile) [headline];
  delta = Cliff's delta of the tile score (mean over S of the z-scored tile-mean log-normalised expression) of the
  hits vs the background [delta <= 0 = programme absent];
  lift = median over the hits minus median over all index tiles of c without the robust filter (Figs 6-7);
  Jaccard with naive_partial, D and X.
- GRADIENT (seed 0): every index tile's exact d_B^S (Spindle-naive) to each programme query, with tile centres and
  sides, for the smoothed maps.
"""

import numpy as np
import pandas as pd

import data
import sections as sx
import spindle_dev as sd
from signatures import PAN_CANCER_MODULES

SOURCE = "breast_cancer"
KEYS = ["breast_cancer", "lung_cancer", "pancreatic_cancer", "brain_cancer", "skin_melanoma"]
PROGRAMS = list(PAN_CANCER_MODULES)
HIT_FRAC = 0.05


def n_hits(n):
    """Hits = 5 % of the index tiles (as Fig 3's c = 5 % N)."""
    return max(1, round(HIT_FRAC * n))


def program_idx(genes, prog):
    idx, found, missing = sd.gene_indices(genes, PAN_CANCER_MODULES[prog]["genes"])
    if missing:
        raise RuntimeError(f"{prog}: {missing} not index genes")
    return idx, found


def robust_c(X):
    """The gene-set pilot's exemplar rule (its c_pilot.py was removed 27 Oct; in git history): mean off-diagonal
    correlation c of S over the tile's cells (raw counts), NaN unless >= 80 cells, >= 0.5 S counts per cell, no cell >
    15 % of the S counts, and dropping the top 3 cells keeps >= 70 % of c."""
    c = sx.mean_offdiag_corr(X)
    tot = X.sum(1)
    if len(X) < 80 or tot.sum() / len(X) < 0.5 or tot.max() > 0.15 * tot.sum() or not np.isfinite(c) or c <= 0:
        return np.nan
    c3 = sx.mean_offdiag_corr(X[np.argsort(tot)[:-3]])
    return c if np.isfinite(c3) and c3 >= 0.7 * c else np.nan


def sub_covs(covs, tile_ids, idx):
    """Shrunk S x S covariances of the tiles, (len(tile_ids), |S|, |S|) float64: the values of sd.tile_query(covs[t],
    idx) (float32 covariance as float64, cut to S), cut straight from the memory-mapped covariances so that only the
    S x S entries are read."""
    idx = np.asarray(idx, dtype=np.int64)
    return np.asarray(covs.arr[np.ix_(np.asarray(tile_ids, dtype=np.int64), idx, idx)], dtype=np.float64)


def to_corr(C):
    d = np.sqrt(np.einsum("...ii->...i", C))
    return C / (d[..., :, None] * d[..., None, :])


def corr_gap(subs, R):
    """Per tile: RMS difference between its S x S correlation and R over the off-diagonal pairs (lower = co-varies
    more like the exemplar)."""
    iu = np.triu_indices(R.shape[0], 1)
    C = to_corr(subs)
    return np.sqrt(((C[:, iu[0], iu[1]] - R[iu]) ** 2).mean(1))


def query_scale(subs, q):
    """sigma of the quantile-matched query: sqrt of the q_g quantile of gene g's shrunk variance over the index tiles."""
    V = np.einsum("nii->ni", subs)
    return np.sqrt([np.quantile(V[:, j], q[j]) for j in range(len(q))])


# ---------------------------------------------------------------- exemplars (breast seed 0)
def exemplars():
    """(exemplar table, arrays of exemplar.npz)."""
    S = sx.load(SOURCE, data.EXAMPLE_SEED)
    c, covs = S["c"], S["covs"]
    rows, arrays = [], {}
    for prog in PROGRAMS:
        idx, found = program_idx(S["genes"], prog)
        cs = np.array([robust_c(S["X"][S["tiles"][t].idx][:, idx]) for t in c.train])
        if not np.isfinite(cs).any():
            print(f"{prog}: no robust tile, dropped", flush=True)
            continue
        r = int(np.nanargmax(cs))
        ex = int(c.train[r])
        subs = sub_covs(covs, c.train, idx)
        V = np.einsum("nii->ni", subs)
        R = to_corr(subs[r])
        q = (V <= V[r]).mean(0)
        iu = np.triu_indices(len(idx), 1)
        rows.append({"program": prog, "genes": " ".join(found), "n_genes": len(idx), "tile": ex,
                     "n_cells": len(S["tiles"][ex].idx), "c": float(cs[r]), "mean_r": float(R[iu].mean()),
                     "n_robust": int(np.isfinite(cs).sum()), "q": " ".join(f"{v:.3f}" for v in q),
                     "tag": S["stamp"]["tag"]})
        arrays[f"R_{prog}"], arrays[f"q_{prog}"] = R, q
        arrays[f"bbox_{prog}"] = np.asarray(S["tiles"][ex].bbox, dtype=np.float64)
        print(f"{prog}: tile {ex} ({rows[-1]['n_cells']} cells), c {cs[r]:.3f}, mean r {R[iu].mean():.3f}, "
              f"{rows[-1]['n_robust']} robust tiles, q {np.round(q, 2)}", flush=True)
    arrays.update(programs=np.array([r["program"] for r in rows]), xy=S["xy"])
    return pd.DataFrame(rows), arrays


# ---------------------------------------------------------------- queries in one section
def section(S, seed, E, Z):
    """(programs table, hits table) of one (section, seed); E, Z = exemplar table and arrays."""
    key, c, covs, st = S["key"], S["c"], S["covs"], S["stamp"]
    n = len(c.train)
    k = n_hits(n)
    mu = sx.tile_means(S, c.train)
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    cells = [S["tiles"][t].idx for t in c.train]
    rows, hits = [], []
    for _, e in E.iterrows():
        prog = e.program
        idx, found = program_idx(S["genes"], prog)
        R, q = Z[f"R_{prog}"], Z[f"q_{prog}"]
        subs = sub_covs(covs, c.train, idx)
        sig = query_scale(subs, q)
        Q = sig[:, None] * R * sig[None, :]
        np.linalg.cholesky(Q)
        valid = np.ones(n, bool)
        if key == SOURCE:
            valid[c.train == int(e.tile)] = False
        score = z[:, idx].mean(1)
        gap = corr_gap(subs, R)
        cov_all = np.array([sx.mean_offdiag_corr(S["X"][cl][:, idx]) for cl in cells])

        def first(order):
            return np.asarray([r for r in order if valid[r]][:k], dtype=int)

        tops = {}
        for m, Qm in (("naive_partial", Q), ("D", np.diag(sig ** 2))):
            tops[m] = first(np.argsort(c.naive_partial.distances(Qm, idx), kind="stable"))
        spindle_rows, _, _ = c.spindle_partial.search_vectors(c.spindle_partial.query_vectors(Q, idx))
        tops["spindle_partial"] = first(spindle_rows)
        tops["X"] = first(np.argsort(-score, kind="stable"))
        for m, top in tops.items():
            back = np.setdiff1d(np.flatnonzero(valid), top)
            rows.append({"key": key, "seed": seed, "program": prog, "n_genes": len(idx), "method": m, "n_index": n,
                         "c": k, "n_returned": len(top), "spindle_partial_returned": len(spindle_rows),
                         "gap_hits": float(np.median(gap[top])), "gap_back": float(np.median(gap[back])),
                         "delta": float(sx.cliffs_delta(score[top], score[back])),
                         "cov_top": float(np.nanmedian(cov_all[top])), "cov_all": float(np.nanmedian(cov_all[valid])),
                         **{f"jacc_{o}": sx.jaccard(top, tops[o]) for o in ("naive_partial", "D", "X")},
                         "tag": st["tag"]})
            hits += [{"key": key, "seed": seed, "program": prog, "method": m, "rank": i + 1, "tile": int(c.train[r]),
                      "gap": float(gap[r]), "score": float(score[r])} for i, r in enumerate(top)]
    T = pd.DataFrame(rows)
    T["gain"] = T.gap_back - T.gap_hits
    T["lift"] = T.cov_top - T.cov_all
    return T, pd.DataFrame(hits)


def gradient(S, E, Z):
    """Arrays of gradient.npz: cells, index tile centres / sides, d_<programme> = exact d_B^S of every index tile to
    the query (index-tile order; the breast exemplar tile included)."""
    c, covs = S["c"], S["covs"]
    box = np.array([S["tiles"][t].bbox for t in c.train], dtype=np.float64)
    out = {"xy": S["xy"], "tiles": c.train, "centre": (box[:, :2] + box[:, 2:]) / 2,
           "side": np.sqrt((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])), "programs": E.program.to_numpy(),
           "tag": S["stamp"]["tag"]}
    for prog in E.program:
        idx, _ = program_idx(S["genes"], prog)
        R, q = Z[f"R_{prog}"], Z[f"q_{prog}"]
        sig = query_scale(sub_covs(covs, c.train, idx), q)
        out[f"d_{prog}"] = c.naive_partial.distances(sig[:, None] * R * sig[None, :], idx)
        print(f"{S['key']} {prog}: d {np.min(out[f'd_{prog}']):.3f}-{np.max(out[f'd_{prog}']):.3f}", flush=True)
    return out


def summary(g):
    """Per section, programme and method: mean and s.d. over seeds; absent = Spindle-naive's mean delta <= 0."""
    num = ["gain", "delta", "lift", "gap_hits", "gap_back", "jacc_naive_partial", "jacc_D", "jacc_X", "n_returned",
           "spindle_partial_returned", "c", "n_index"]
    by = ["key", "program", "method"]
    out = g.groupby(by, sort=False)[num].mean().join(g.groupby(by, sort=False)[num].std(), rsuffix="_sd")
    out = out.reset_index()
    p = out[out.method == "naive_partial"].set_index(["key", "program"]).delta
    out["absent"] = [bool(p.loc[(k, pr)] <= 0) for k, pr in zip(out.key, out.program)]
    out["tag"] = g.tag.iloc[0]
    return out
