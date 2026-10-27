"""Shared by the Fig 8 compute scripts (no plotting; safe in sbatch): the five tumour sections, the four pan-cancer
programs and the statistics of the panels. Sections, builds, expression and the statistics come from Figs 6-7
(experiments/fig7_skin/common7.py, experiments/fig6_breast/common6.py), imported unchanged.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig7_skin"))
import common7 as c7  # noqa: E402  (puts experiments/common, fig6_breast and src/ on sys.path)
from common7 import c6, data, sd  # noqa: E402
from signatures import PAN_CANCER_MODULES  # noqa: E402

SOURCE = "breast_cancer"
KEYS = ["breast_cancer", "lung_cancer", "pancreatic_cancer", "brain_cancer", "skin_melanoma"]
PROGRAMS = list(PAN_CANCER_MODULES)
OUT = data.RESULTS / "fig8_pan_cancer"
HIT_FRAC = 0.05


def task(t):
    """(key, seed) of array task t = 5 * dataset + seed."""
    return KEYS[t // len(data.SEEDS)], data.SEEDS[t % len(data.SEEDS)]


def out_dir(key, seed):
    d = OUT / key / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


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
    correlation c of S over the tile's cells (raw counts), NaN unless >= 80 cells, >= 0.5 S counts per cell, no cell > 15 % of the S counts,
    and dropping the top 3 cells keeps >= 70 % of c."""
    c = c6.mean_offdiag_corr(X)
    tot = X.sum(1)
    if len(X) < 80 or tot.sum() / len(X) < 0.5 or tot.max() > 0.15 * tot.sum() or not np.isfinite(c) or c <= 0:
        return np.nan
    c3 = c6.mean_offdiag_corr(X[np.argsort(tot)[:-3]])
    return c if np.isfinite(c3) and c3 >= 0.7 * c else np.nan


def sub_covs(covs, tile_ids, idx):
    """Shrunk S x S covariances of the tiles, (len(tile_ids), |S|, |S|)."""
    return np.stack([sd.tile_query(covs[t], idx)[0] for t in tile_ids])


def to_corr(C):
    d = np.sqrt(np.einsum("...ii->...i", C))
    return C / (d[..., :, None] * d[..., None, :])


def corr_gap(subs, R):
    """Per tile: RMS difference between its S x S correlation and R over the off-diagonal pairs (lower = co-varies
    more like the exemplar)."""
    iu = np.triu_indices(R.shape[0], 1)
    C = to_corr(subs)
    return np.sqrt(((C[:, iu[0], iu[1]] - R[iu]) ** 2).mean(1))
