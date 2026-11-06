"""Cross-dataset partial queries (Fig 4A; R8_pan_cancer.ipynb): breast gene programmes searched in other sections.
Section loader and statistics in sections.py.

Programs: signatures.PAN_CANCER_MODULES (cytotoxic T, myeloid, stromal / perivascular, B / plasma); every gene must be
an index gene of the section searched (a missing gene stops the run).

RULES (fixed before the run):
- EXEMPLAR, one per program, fixed for every section and seed: among breast's seed-0 index tiles, the tile with the
  largest robust co-variation c on S (robust_c: c = mean off-diagonal correlation of S over the tile's cells, raw
  counts; >= 80 cells, >= 0.5 S counts per cell, no cell > 15 % of the S counts, dropping the top 3 cells keeps
  >= 70 % of c). No robust tile -> the program is dropped (reported). R_S = the correlation of its shrunk covariance
  cut to S; q_g = the share of breast seed-0 index tiles whose shrunk variance of gene g is <= the exemplar's.
- QUERY in section X, seed s (quantile-matched scale): sigma_g = sqrt of the q_g quantile of gene g's shrunk variance
  over X's seed-s index tiles; Q = diag(sigma) R_S diag(sigma).
- GRADIENT (skin, seed 0): every index tile's exact d_B^S (Spindle-naive) to each programme query, with tile centres
  and sides, for the smoothed closeness maps of Fig 4A.
"""

import numpy as np
import pandas as pd

import data
import sections as sx
import spindle_dev as sd
from signatures import PAN_CANCER_MODULES

SOURCE = "breast_cancer"
GRADIENT_KEY = "skin_melanoma"
PROGRAMS = list(PAN_CANCER_MODULES)


def program_idx(genes, prog):
    idx, found, missing = sd.gene_indices(genes, PAN_CANCER_MODULES[prog]["genes"])
    if missing:
        raise RuntimeError(f"{prog}: {missing} not index genes")
    return idx, found


def robust_c(X):
    """The exemplar rule: mean off-diagonal correlation c of S over the tile's cells (raw counts), NaN unless >= 80
    cells, >= 0.5 S counts per cell, no cell > 15 % of the S counts, and dropping the top 3 cells keeps >= 70 % of c."""
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

