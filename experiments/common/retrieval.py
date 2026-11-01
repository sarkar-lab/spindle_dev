"""Retrieval readouts shared by every accuracy result (Figs 3-5), as declared on 19 Oct (the normalised-excess metric).
Ported from experiments/fig3_whole_tile/common3.py (in git history).

  NX(c)    = (mean exact d of the returned top c - mean exact top c) / (median exact d over all index tiles - mean
             exact top c); c_eff = min(c, tiles returned) (a short list is graded against the exact closest set of
             its own size). NX5 = NX at c = 5 % of N.
  Overlap  = share of the exact 5 %-near set {t: d(t) <= 1.05 x nearest d} in the returned top c, c = 5 % of N
             (secondary; short lists scored on what they returned).
  MX10     = (mean exact d of the returned top 10 - mean exact top 10) / mean exact top 10 (secondary; queries
             returning >= 10 tiles).
"""

import numpy as np

G_DEFAULT = 0.25
GS = [0.25, 0.354, 0.5, 0.75, 1.0, 1.5]     # the whole-tile g trade-off of SPINDLE_LITE.md §5


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
