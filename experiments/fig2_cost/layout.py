"""Fig 2 draft layout (build_figure.py): atoms, panel letters and labels on the 180 mm page.

Row 1 (A-C) shares one row per dataset (f_datasets.py: same atom height and vertical margins); row 2 is the two
ladders (f_ladder.py), three atoms stacked each. Fold labels (B) and column ranges (A) come from
results/fig2_cost/numbers.json, so they never drift from the caption.
"""

import importlib.util
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("f_datasets", HERE / "f_datasets.py")
fd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fd)
_lspec = importlib.util.spec_from_file_location("f_ladder", HERE / "f_ladder.py")
fl = importlib.util.module_from_spec(_lspec)
_lspec.loader.exec_module(fl)
N = json.loads((HERE.parents[1] / "results" / "fig2_cost" / "numbers.json").read_text())

# row 1
Y1 = 3.0
W_A, W_B, A_LEFT, B_LEFT = fd.W_A, fd.W_B, fd.A_LEFT, fd.B_LEFT
X_A = 0.0
X_B = X_A + W_A
X_C1 = X_B + W_B
X_C2 = X_C1 + fd.W_C1
X_C3 = X_C2 + fd.W_C2
# row 2
Y2 = Y1 + fd.H + 5.0
X_D, X_E = 0.0, 92.0
Y_LEG = Y2 + 2 * fl.H_TOP + fl.H_BOTTOM + 1.5
HEIGHT_MM = Y_LEG + 6.0

ATOMS = [
    ("datasets", X_A, Y1, 1), ("storage", X_B, Y1, 1), ("build_time", X_C1, Y1, 1), ("build_adds", X_C2, Y1, 1),
    ("peak_memory", X_C3, Y1, 1),
]
for x, ladder in ((X_D, "cells"), (X_E, "genes")):
    ATOMS += [(f"{ladder}_storage", x, Y2, 1), (f"{ladder}_time", x, Y2 + fl.H_TOP, 1),
              (f"{ladder}_memory", x, Y2 + 2 * fl.H_TOP, 1)]
ATOMS.append(("legend_tiers", 1.0, Y_LEG, 1))


def row_y(i):
    """Centre (mm from the slide top) of dataset row i in the row-1 atoms."""
    pitch = (fd.H - fd.TOP - fd.BOTTOM) / len(fd.ps.DATASET_ORDER)
    return Y1 + fd.TOP + (i + 0.5) * pitch, pitch


def storage_x(mb):
    lo, hi = (math.log10(v) for v in fd.STORAGE_XLIM)
    ax_w = W_B - 2 * B_LEFT
    return X_B + B_LEFT + ax_w * (math.log10(mb) - lo) / (hi - lo)


def fmt(v):
    return f"{v:,.0f}" if v >= 1000 else f"{v:g}"


def k_cells(v):
    return f"{v / 1000:.0f}k" if v < 1e6 else f"{v / 1e6:.2f}M"


TEXTS = [("A", 0.0, 0.0, 4, 4, "letter"), ("B", X_B, 0.0, 4, 4, "letter"), ("C", X_C1, 0.0, 4, 4, "letter"),
         ("D", X_D, Y2 - 3.5, 4, 4, "letter"), ("E", X_E, Y2 - 3.5, 4, 4, "letter")]

# A: range across datasets below each column (seed 0)
r = N["range"]
col_w = (W_A - A_LEFT - 1.0) / 5
ranges = [f"{k_cells(r['cells_seed0'][0])}–{k_cells(r['cells_seed0'][1])}", f"{fmt(r['genes'][0])}–{fmt(r['genes'][1])}",
          f"{fmt(r['index_tiles_seed0'][0])}–{fmt(r['index_tiles_seed0'][1])}",
          f"{fmt(r['niches_seed0'][0])}–{fmt(r['niches_seed0'][1])}", f"{fmt(r['blocks_seed0'][0])}–{fmt(r['blocks_seed0'][1])}"]
for j, t in enumerate(ranges):  # one line, centred under its column
    TEXTS.append((t, X_A + A_LEFT + j * col_w, Y1 + fd.H - fd.BOTTOM + 1.0, col_w, 3, "small_c"))

# B: fold labels above the whole-tile row (whole -> Spindle, Spindle -> Spindle-lite), at the log midpoints
for i, key in enumerate(fd.ps.DATASET_ORDER):
    d = N["datasets"][key]
    y, pitch = row_y(i)
    y_top = y - fd.ROW_OFF * pitch - 2.6
    for a, b, f in (("spindle", "whole", d["fold_whole_spindle"]), ("lite", "spindle", d["fold_spindle_lite"])):
        xm = storage_x(math.sqrt(d["mb"][a] * d["mb"][b]))
        TEXTS.append((f"{f:.0f}×" if f >= 10 else f"{f:.1f}×", xm - 3.0, y_top, 6.0, 2.5, "small"))

# C: what the column shows
TEXTS += [("Spindle (full build)", X_C1 + 1.5, Y1 + 2.4, 23, 3, "small"),
          ("added over Spindle", X_C2 + 1.5, Y1 + 2.4, 23, 3, "small"),
          ("all four indexes", X_C3 + 1.5, Y1 + 2.4, 20, 3, "small")]

# D/E titles
cl, gl = N["cell_ladder"], N["gene_ladder"]
TEXTS += [(f"Tonsil, {cl['full']['cells'] / 1e6:.2f}M cells, {cl['full']['genes']} genes, subsampled",
           X_D + fl.LEFT, Y2 - 3.5, 70, 3, "label"),
          (f"Lymph node 5k, {gl['index_tiles']:,} index tiles", X_E + fl.LEFT, Y2 - 3.5, 70, 3, "label")]
