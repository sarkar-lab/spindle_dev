"""Fig 2D/E atoms from results/fig2_cost/ladder.csv: storage, build time and peak memory against cells and genes.

Usage (login node): python experiments/fig2_cost/f_ladder.py
Per ladder (cells: tonsil subsampled, log x; genes: lymph node 5k, linear x), three atoms stacked in the pptx, sharing
the x axis (only the bottom one carries the x label):
  <ladder>_storage   whole covariances, Spindle, Spindle-lite, Spindle-partial (open), Spindle-lite-partial (MB, log)
  <ladder>_time      Spindle's full build (covariances, layout, block logs) and the seconds each other index adds (log)
  <ladder>_memory    peak resident memory of the whole build (GB, log)
Markers and colours as Fig 2B/C (legend_tiers).
"""

import math
import sys
from pathlib import Path

import pandas as pd
from matplotlib.ticker import FuncFormatter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import style as ps  # noqa: E402

FIG = "fig2_cost"
SRC = "results/fig2_cost/ladder.csv"
W, H_TOP, H_BOTTOM = 86.0, 32.0, 38.0  # mm; the bottom atom also holds the x label
LEFT, RIGHT, TOP, BOTTOM_PLAIN, BOTTOM_LABEL = 13.0, 2.0, 1.5, 4.0, 10.0


def line(ax, x, y, c, mk, filled=True, ls="-"):
    k = ps.TIER_COLORS[c]
    ax.plot(x, y, ls=ls, marker=mk, ms=2.6, color=k, mfc=k if filled else "white", mec=k, mew=0.7, zorder=3)


def axes(ladder, name, ylabel, bottom):
    fig, ax = ps.atom(W, H_BOTTOM if bottom else H_TOP, left=LEFT, right=RIGHT, top=TOP,
                      bottom=BOTTOM_LABEL if bottom else BOTTOM_PLAIN)
    ax.set_yscale("log")
    ps.plain_log(ax, "y", subs=(1.0,) if name in ("storage", "time") else (1.0, 2.0, 5.0))
    ps.log_grid(ax, "y")
    ax.set_ylabel(ylabel)
    if ladder == "cells":
        ax.set_xscale("log")
        ps.plain_log(ax, "x", subs=(1.0,))
        ax.set_xlim(7e2, 2e6)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"$10^{{{round(math.log10(v))}}}$"))
    else:
        ax.set_xlim(0, 1560)
        ax.set_xticks(range(0, 1501, 250))
    if bottom:
        ax.set_xlabel("Cells (log scale)" if ladder == "cells" else "Genes")
    else:
        ax.set_xticklabels([])
    return fig, ax


def plot(df, ladder):
    d = df[df.ladder == ladder]
    x = d.n_cells if ladder == "cells" else d.p

    fig, ax = axes(ladder, "storage", "Storage\n(MB, log scale)", False)
    for c, mk, filled in (("whole", "s", True), ("spindle", "o", True), ("lite", "D", True),
                          ("partial", "o", False), ("plite", "^", True)):
        line(ax, x, d[f"bytes_{c}"] / 1e6, "spindle" if c == "partial" else c, mk, filled)
    ps.save(fig, FIG, f"{ladder}_storage", source=SRC)

    fig, ax = axes(ladder, "time", "Build time\n(s, log scale)", False)
    line(ax, x, d.spindle_build_s, "spindle", "o")
    line(ax, x, d.lite_s, "lite", "D", ls="--")
    line(ax, x, d.partial_s, "spindle", "o", filled=False, ls="--")
    line(ax, x, d.plite_s, "plite", "^", ls="--")
    ps.save(fig, FIG, f"{ladder}_time", source=SRC)

    fig, ax = axes(ladder, "memory", "Peak memory\n(GB, log scale)", True)
    ax.plot(x, d.peak_gb_plite, marker="o", ms=2.6, color=ps.INK, zorder=3)
    ps.save(fig, FIG, f"{ladder}_memory", source=SRC)


def main():
    df = pd.read_csv(ps.ROOT / SRC)
    for ladder in ("cells", "genes"):
        if (df.ladder == ladder).any():
            plot(df, ladder)


if __name__ == "__main__":
    main()
