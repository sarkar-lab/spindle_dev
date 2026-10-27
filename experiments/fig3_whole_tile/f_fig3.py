"""Fig 3 atoms from the small inputs of c_summary.py (login node).

Usage: python experiments/fig3_whole_tile/f_fig3.py
Atoms (figures/fig3_whole_tile/atoms/), datasets top to bottom in style.DATASET_ORDER, so the three row plots line up:
  timing        search time per query (ms, log scale), search only: whole-matrix scan, Spindle, Spindle-lite (bars,
                mean over seeds of the per-seed median; whiskers s.d.); the query's own logs as open markers (whole
                matrix: grey diamond on the whole bar; block logs: blue circle on the Spindle bar)
  nx            Spindle-lite NX at c = 5 % of N (%), g = 0.25; mean +/- s.d. over seeds
  band          the typical breast query (rule in c_nx.py): exact d_B of every index tile (grey) and of the tiles
                Spindle-lite returned (colour), same bins
  memory_<key>  breast, brain: NX5 (%) against stored MB (log scale); Spindle-lite (as built, one point) against
                d_B; PCA + flat, PCA-256 + PQ against d_W; dotted: Spindle and whole-matrix stores
  legend_*      one legend atom per panel
Numbers that go next to plots (speed-ups, parameter labels) are text boxes in the pptx: their positions (mm in the
atom) go to figures/fig3_whole_tile/annotations.json, read by layout.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import style as ps  # noqa: E402

FIG = "fig3_whole_tile"
RES = ps.ROOT / "results" / FIG
ANN = ps.FIGURES / FIG / "annotations.json"
KEYS = ps.DATASET_ORDER
ROW_H = 70.0                         # mm, the three row plots
LEFT_A, RIGHT_A, BOTTOM, TOP = 17.0, 17.0, 9.0, 1.5
G = 0.25
LITE, SPINDLE, WHOLE = ps.TIER_COLORS["lite"], ps.TIER_COLORS["spindle"], ps.TIER_COLORS["whole"]
COMP = {"pca_flat": "#117733", "pca256_pq": "#999933"}
COMP_LABEL = {"pca_flat": "PCA + flat (whole matrix)",
              "pca256_pq": "PCA-256 + PQ (whole matrix)"}
annotations = {}


def to_mm(fig, ax, x, y):
    """(x_mm, y_mm from the atom's top-left) of data point (x, y)."""
    fx, fy = fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    w, h = fig.get_size_inches() / ps.MM
    return float(fx * w), float((1 - fy) * h)


def note(atom, text, x_mm, y_mm):
    annotations.setdefault(atom, []).append([text, round(x_mm, 2), round(y_mm, 2)])


def rows_axis(ax, labels):
    ax.set_ylim(len(KEYS) - 0.5, -0.5)
    ax.set_yticks(range(len(KEYS)))
    ax.set_yticklabels([ps.DATASET_LABELS[k] for k in KEYS] if labels else [])
    ax.tick_params(axis="y", length=0)


def timing():
    t = pd.read_csv(RES / "timing.csv").set_index("key")
    w = 110.0
    fig, ax = ps.atom(w, ROW_H, left=LEFT_A, right=RIGHT_A, bottom=BOTTOM, top=TOP)
    bh = 0.26
    for i, k in enumerate(KEYS):
        if k not in t.index:
            continue
        r = t.loc[k]
        for off, col, c in ((-bh, "whole_ms", WHOLE), (0, "spindle_ms", SPINDLE), (bh, "lite_ms", LITE)):
            ax.barh(i + off, r[col], height=bh * 0.92, color=c, lw=0, zorder=2)
            ax.errorbar(r[col], i + off, xerr=r[f"{col}_sd"], fmt="none", ecolor=ps.INK, elinewidth=0.5, zorder=3)
        ax.plot(r.qlog_whole_ms, i - bh, "D", ms=2.4, mfc="white", mec=ps.INK, mew=0.6, zorder=4)
        ax.plot(r.qlog_block_ms, i, "o", ms=2.6, mfc="white", mec=SPINDLE, mew=0.7, zorder=4)
    ax.set_xscale("log")
    ps.plain_log(ax, "x", subs=(1.0,))
    ps.log_grid(ax, "x")
    lo = min(t.lite_ms.min(), t.spindle_ms.min()) / 2
    hi = max(t.whole_ms.max(), t.qlog_whole_ms.max()) * 2
    ax.set_xlim(lo, hi)
    rows_axis(ax, True)
    ax.set_xlabel("Time per query (ms, log scale)")
    for i, k in enumerate(KEYS):
        if k in t.index:
            r = t.loc[k]
            xr = to_mm(fig, ax, hi, i)[0] + 1.5
            note("timing", f"{r.speedup_spindle_vs_whole:.0f}×", xr, to_mm(fig, ax, hi, i - bh / 2)[1])
            note("timing", f"{r.speedup_lite_vs_spindle:.1f}×", xr + 7.5, to_mm(fig, ax, hi, i + bh / 2)[1])
    ps.save(fig, FIG, "timing", source="results/fig3_whole_tile/timing.csv")


def dot_rows(name, cols, xlabel, src, w=70.0):
    t = pd.read_csv(RES / src)
    t = t[t.g == G].set_index("key")
    fig, ax = ps.atom(w, ROW_H, left=2.0, right=3.0, bottom=BOTTOM, top=TOP)
    for i, k in enumerate(KEYS):
        if k not in t.index:
            continue
        r = t.loc[k]
        for j, (col, filled) in enumerate(cols):
            y = i + (j - (len(cols) - 1) / 2) * 0.22
            ax.errorbar(100 * r[col], y, xerr=100 * r[f"{col}_sd"], fmt="o", ms=2.8, color=LITE,
                        mfc=LITE if filled else "white", mec=LITE, mew=0.7, elinewidth=0.5, zorder=3)
    rows_axis(ax, False)
    ax.set_xlim(0, None)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(xlabel)
    ps.save(fig, FIG, name, source=f"results/fig3_whole_tile/{src}")


def band():
    z = np.load(RES / "band" / "breast_cancer.npz")
    d, found = z["d"], z["found"]
    fig, ax = ps.atom(56, 52, left=10.0, right=2.0, bottom=BOTTOM, top=TOP)
    bins = np.linspace(d.min(), np.quantile(d, 0.995), 50)
    ax.hist(d, bins=bins, color=ps.LIGHT, lw=0, zorder=2)
    ax.hist(d[found], bins=bins, color=LITE, lw=0, zorder=3)
    ax.set_xlabel(r"Block distance $d_{\mathcal{B}}$ to the query")
    ax.set_ylabel("Index tiles")
    ax.set_xlim(bins[0], bins[-1])
    ps.save(fig, FIG, "band", source="results/fig3_whole_tile/band/breast_cancer.npz")


def memory(key):
    m = pd.read_csv(RES / "memory.csv", dtype={"param": str})
    m = m[m.key == key]
    fig, ax = ps.atom(58, 52, left=10.0, right=2.0, bottom=BOTTOM, top=TOP)
    for meth, ls in (("spindle", ":"), ("whole", ":")):
        x = m[m.method == meth].MB.iloc[0]
        ax.axvline(x, ls=ls, lw=0.75, color=SPINDLE if meth == "spindle" else WHOLE, zorder=1)
    r = m[m.method == "lite"].iloc[0]
    ax.errorbar(r.MB, 100 * r.NX5, yerr=100 * r.NX5_sd, fmt="D", ms=3.0, color=LITE, elinewidth=0.5, zorder=4)
    for meth, mk in (("pca_flat", "v"), ("pca256_pq", "P")):
        s = m[m.method == meth].assign(v=lambda d: d.param.astype(float)).sort_values("v")
        if s.empty:
            continue
        ax.errorbar(s.MB, 100 * s.NX5, yerr=100 * s.NX5_sd, fmt=f"-{mk}", ms=2.8, color=COMP[meth],
                    elinewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ps.plain_log(ax, "x", subs=(1.0,))
    ps.log_grid(ax, "x")
    ax.set_ylim(0, None)
    ax.set_xlabel("Stored MB (log scale)")
    ax.set_ylabel("NX at 5 % of N (%)")
    # parameter labels (text boxes) once the axis limits are fixed
    for meth in ("pca_flat", "pca256_pq"):
        s = m[m.method == meth].assign(v=lambda d: d.param.astype(float))
        for _, r in s.iterrows():
            x, y = to_mm(fig, ax, r.MB, 100 * r.NX5)
            note(f"memory_{key}", f"{'D' if meth == 'pca_flat' else 'm'} = {int(r.v)}", x + 1.0, y - 3.0)
    ps.save(fig, FIG, f"memory_{key}", source="results/fig3_whole_tile/memory.csv")


def legends():
    bar = [Patch(color=WHOLE, label="Whole-matrix scan"), Patch(color=SPINDLE, label="Spindle"),
           Patch(color=LITE, label="Spindle-lite"),
           Line2D([], [], ls="", marker="D", ms=2.4, mfc="white", mec=ps.INK, mew=0.6, label="Query's whole-matrix log"),
           Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=SPINDLE, mew=0.7, label="Query's block logs")]
    ps.legend_atom(FIG, "legend_timing", bar, 88, 8, ncol=3)
    mem = [Line2D([], [], ls="", marker="D", ms=3.0, color=LITE, label="Spindle-lite"),
           Line2D([], [], marker="v", ms=2.8, color=COMP["pca_flat"], label=COMP_LABEL["pca_flat"]),
           Line2D([], [], marker="P", ms=2.8, color=COMP["pca256_pq"], label=COMP_LABEL["pca256_pq"]),
           Line2D([], [], ls=":", color=SPINDLE, label="Spindle store"),
           Line2D([], [], ls=":", color=WHOLE, label="Whole-matrix log store")]
    ps.legend_atom(FIG, "legend_memory", mem, 116, 10, ncol=3)


if __name__ == "__main__":
    timing()
    dot_rows("nx", [("NX5", True)], "Spindle-lite NX at 5 % of N (%)", "nx.csv")
    band()
    for k in ("breast_cancer", "brain_cancer"):
        memory(k)
    legends()
    ANN.write_text(json.dumps(annotations, indent=1, ensure_ascii=False))
