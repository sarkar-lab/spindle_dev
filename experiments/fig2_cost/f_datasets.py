"""Fig 2A-C atoms from the index cache (results/index_cache/builds.csv): datasets, storage, build cost.

Usage (login node): python experiments/fig2_cost/f_datasets.py
The three panels share one row per dataset (datasets.yaml order, top to bottom) and the same atom height and vertical
margins, so the rows line up side by side in the pptx.
  datasets     A: cells, genes, index tiles, niches, blocks of the seed-0 build; dot area proportional to the value
               within each column (the range across datasets is a pptx text box).
  storage      B: bytes, log scale, mean +- s.d. over seeds 0-4. Upper row: whole covariances, Spindle, Spindle-lite;
               lower row: Spindle-partial (open circle), Spindle-lite-partial.
  build_time   C: Spindle's full build (covariances, layout, block logs; file loading excluded), log scale.
  build_adds   C: seconds each other index adds to it (Spindle-lite, Spindle-partial, Spindle-lite-partial), log scale.
  peak_memory  C: peak resident memory of the whole build (all four indexes), log scale.
  legend_storage, legend_adds: legends as their own atoms.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import style as ps  # noqa: E402

FIG = "fig2_cost"
SRC = "results/index_cache/builds.csv"
H, TOP, BOTTOM = 54.0, 6.5, 11.0  # mm; shared by every row-aligned atom
ROW_OFF = 0.17  # the two storage rows of a dataset
STORAGE_XLIM = (0.7, 8000.0)  # MB; layout.py places the fold labels with it
# atom widths (mm), left to right: A, B, C (build time, added, peak memory); they fill the 180 mm row
W_A, W_B, W_C1, W_C2, W_C3 = 70.0, 42.0, 24.0, 24.0, 20.0
A_LEFT, B_LEFT = 19.0, 1.5


def load():
    df = pd.read_csv(ps.ROOT / SRC)
    df["spindle_build_s"] = df[["prepare_s", "layout_s", "spindle_s"]].sum(axis=1)
    df["key"] = pd.Categorical(df["key"], ps.DATASET_ORDER, ordered=True)
    return df.sort_values(["key", "seed"])


def rows(ax):
    n = len(ps.DATASET_ORDER)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(range(n))
    return n


def mean_sd(df, col):
    g = df.groupby("key", observed=True)[col]
    return g.mean().to_numpy(), g.std().to_numpy()


def datasets(df):
    s0 = df[df.seed == 0].set_index("key").loc[ps.DATASET_ORDER]
    cols = [("n_cells", "Cells"), ("p", "Genes"), ("n_train", "Index\ntiles"), ("n_niches", "Niches"),
            ("n_blocks", "Blocks")]
    fig, ax = ps.atom(W_A, H, left=A_LEFT, right=1, top=TOP, bottom=BOTTOM)
    n = rows(ax)
    ax.set_yticklabels([ps.DATASET_LABELS[k] for k in ps.DATASET_ORDER])
    max_area = 50.0  # pt^2 of the largest dot in a column
    for j, (col, _) in enumerate(cols):
        v = s0[col].to_numpy(float)
        ax.scatter(np.full(n, j), range(n), s=max_area * v / v.max(), c=[ps.DATASET_COLORS[k] for k in s0.index],
                   linewidths=0, zorder=3)
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c for _, c in cols])
    ax.xaxis.tick_top()
    ax.tick_params(axis="both", length=0)
    for side in ("left", "bottom", "top"):
        ax.spines[side].set_visible(False)
    ps.save(fig, FIG, "datasets", source=SRC)


def storage(df):
    fig, ax = ps.atom(W_B, H, left=B_LEFT, right=1.5, top=TOP, bottom=BOTTOM)
    n = rows(ax)
    y = np.arange(n)
    up, lo = y - ROW_OFF, y + ROW_OFF
    m = {c: mean_sd(df, f"bytes_{c}") for c in ("whole", "spindle", "lite", "partial", "plite")}
    mb = {c: (mu / 1e6, sd / 1e6) for c, (mu, sd) in m.items()}
    for yy, a, b in ((up, "whole", "lite"), (lo, "partial", "plite")):
        ax.hlines(yy, mb[b][0], mb[a][0], color=ps.LIGHT, lw=1.2, zorder=1)
    spec = [("whole", up, ps.TIER_COLORS["whole"], "s", True), ("spindle", up, ps.TIER_COLORS["spindle"], "o", True),
            ("lite", up, ps.TIER_COLORS["lite"], "D", True), ("partial", lo, ps.TIER_COLORS["spindle"], "o", False),
            ("plite", lo, ps.TIER_COLORS["plite"], "^", True)]
    for c, yy, col, mk, filled in spec:
        mu, sd = mb[c]
        ax.errorbar(mu, yy, xerr=sd, fmt=mk, ms=3.2, color=col, mfc=col if filled else "white", mec=col, mew=0.8,
                    elinewidth=0.6, zorder=3)
    ax.set_xscale("log")
    ps.plain_log(ax, "x", subs=(1.0,))
    ps.log_grid(ax, "x")
    ax.set_xlim(*STORAGE_XLIM)
    ax.set_xlabel("Storage (MB, log scale)")
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ps.save(fig, FIG, "storage", source=SRC)


def xlog_rows(w, xlabel, left=1.5):
    fig, ax = ps.atom(w, H, left=left, right=1.5, top=TOP, bottom=BOTTOM)
    rows(ax)
    ax.set_xscale("log")
    ps.log_grid(ax, "x")
    ax.set_xlabel(xlabel)
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return fig, ax


def build_cost(df):
    y = np.arange(len(ps.DATASET_ORDER))
    fig, ax = xlog_rows(W_C1, "Build time\n(s, log scale)")
    mu, sd = mean_sd(df, "spindle_build_s")
    ax.errorbar(mu, y, xerr=sd, fmt="o", ms=3.2, color=ps.TIER_COLORS["spindle"], elinewidth=0.6, zorder=3)
    ps.plain_log(ax, "x", subs=(1.0, 2.0, 5.0))
    ps.save(fig, FIG, "build_time", source=SRC)

    fig, ax = xlog_rows(W_C2, "Added\n(s, log scale)")
    for col, c, mk, filled, dy in (("lite_s", "lite", "D", True, -ROW_OFF), ("partial_s", "spindle", "o", False, 0.0),
                                   ("plite_s", "plite", "^", True, ROW_OFF)):
        mu, sd = mean_sd(df, col)
        k = ps.TIER_COLORS[c]
        ax.errorbar(mu, y + dy, xerr=sd, fmt=mk, ms=3.0, color=k, mfc=k if filled else "white", mec=k, mew=0.8,
                    elinewidth=0.6, zorder=3)
    ps.plain_log(ax, "x", subs=(1.0, 2.0, 5.0))
    ps.save(fig, FIG, "build_adds", source=SRC)

    fig, ax = xlog_rows(W_C3, "Peak memory\n(GB, log scale)")
    mu, sd = mean_sd(df, "peak_gb_plite")
    ax.errorbar(mu, y, xerr=sd, fmt="o", ms=3.2, color=ps.INK, elinewidth=0.6, zorder=3)
    ps.plain_log(ax, "x", subs=(1.0, 2.0, 5.0))
    ax.set_xlim(2, 80)
    ps.save(fig, FIG, "peak_memory", source=SRC)


def legends():
    def h(c, mk, label, filled=True):
        k = ps.TIER_COLORS[c]
        return Line2D([], [], ls="", marker=mk, ms=3.5, color=k, mfc=k if filled else "white", mec=k, mew=0.8,
                      label=label)

    handles = [h("whole", "s", "Whole covariances"), h("spindle", "o", "Spindle"), h("lite", "D", "Spindle-lite"),
               h("spindle", "o", "Spindle-partial", filled=False), h("plite", "^", "Spindle-lite-partial")]
    ps.legend_atom(FIG, "legend_tiers", handles, 178, 5, ncol=5, source=SRC)


def main():
    df = load()
    datasets(df)
    storage(df)
    build_cost(df)
    legends()


if __name__ == "__main__":
    main()
