"""Shared by the Fig 3 compute scripts (no plotting; safe in sbatch): the whole-matrix log vectors and the retrieval
readouts, as declared on 19 Oct (the normalised-excess metric), so the earlier numbers can be reproduced.

  NX(c)    = (mean exact d of the returned top c - mean exact top c) / (median exact d over all index tiles - mean
             exact top c); c_eff = min(c, tiles returned) (a short list is graded against the exact closest set of
             its own size). NX5 = NX at c = 5 % of N.
  Overlap  = share of the exact 5 %-near set {t: d(t) <= 1.05 x nearest d} in the returned top c, c = 5 % of N
             (secondary; short lists scored on what they returned).
  MX10     = (mean exact d of the returned top 10 - mean exact top 10) / mean exact top 10 (secondary; queries
             returning >= 10 tiles).
Whole-matrix vector of a covariance: the float32 upper triangle of its p x p log (eigenvalue floor 1e-6), off-diagonals
x sqrt 2, / sqrt p, i.e. Spindle's block vector with one block of all genes in panel order; its L2 distance is the
whole-matrix log-Euclidean distance d_W, on the same scale as d_B.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402,F401  (puts src/ on sys.path)
from spindle_dev.spindle import BlockVectorizer  # noqa: E402

OUT = data.RESULTS / "fig3_whole_tile"
G_DEFAULT = 0.25
GS = [0.25, 0.354, 0.5, 0.75, 1.0, 1.5]     # the g trade-off of SPINDLE_LITE.md §5


def whole_vectorizer(p):
    return BlockVectorizer(np.arange(p), [(0, p)])


def whole_path(key):
    return data.CACHE / key / "whole.npy"


def c5(n):
    return max(1, round(0.05 * n))


def nx(d, rows, c):
    """NX at c_eff = min(c, len(rows)); NaN when nothing was returned."""
    c = min(c, len(rows))
    if c == 0:
        return np.nan
    ex = np.sort(d)[:c].mean()
    return float((d[rows[:c]].mean() - ex) / (np.median(d) - ex))


def overlap(d, rows, c, delta=0.05):
    near = np.flatnonzero(d <= (1 + delta) * d.min())
    return float(np.isin(near, rows[:c]).mean())


def mx10(d, rows):
    if len(rows) < 10:
        return np.nan
    ex = np.sort(d)[:10].mean()
    return float((d[rows[:10]].mean() - ex) / ex)
