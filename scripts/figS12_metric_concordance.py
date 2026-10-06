"""Fig. S12 -- Block vs whole-matrix distance: biology of the exact neighbours (validates the metric).

For every held-out query (seeds 0-4), the exact top 10 under the block distance d_B (Spindle-Exact) and
under the whole-matrix log-Euclidean distance d_W (same shrinkage, floor and float32 storage) are compared
with the query tile on readouts that neither distance uses directly:
A  pseudo-bulk expression r (query vs mean of the 10 tiles; index genes z-scored over training tiles), all 8
   datasets: whole-matrix (x) vs block (y), mean over queries, +/- s.d. over seeds; above y = x: block better
B  cell-type composition JSD (base 2; query vs mean composition of the 10 tiles), all 8 datasets (breast:
   obs['Cluster']; the others: 10x graph-based clusters from the Xenium analysis output); below y = x: block better
C  share of queries on which the block neighbours are better (higher r / lower JSD); paired Wilcoxon over the
   pooled queries in results/neighbour_biology/tests.csv (all p < 1e-4)

Input: results/neighbour_biology/{summary,tests}.csv (benchmarks/neighbour_biology.py).
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import figstyle as fs

ORDER = fs.DATASET_ORDER
K = 10
READOUTS = [("expr_r", "Expression r (k = 10)"), ("comp_jsd", "Composition JSD (k = 10)")]


def load():
    s = pd.read_csv(fs.RESULTS / "neighbour_biology" / "summary.csv")
    t = pd.read_csv(fs.RESULTS / "neighbour_biology" / "tests.csv")
    for d in (s, t):
        d["key"] = d["dataset"].map(fs.STEMS)
    return s[s["k"] == K], t[(t["k"] == K) & (t["a"] == "exact") & (t["b"] == "whole")]


def scatter(ax, s, readout, label):
    vals = []
    for k in ORDER:
        d = s[s["key"] == k].set_index("method")
        if "exact" not in d.index or pd.isna(d.loc["exact", f"{readout}_mean"]):
            continue
        x, y = d.loc["whole", f"{readout}_mean"], d.loc["exact", f"{readout}_mean"]
        ax.errorbar(x, y, xerr=d.loc["whole", f"{readout}_std"], yerr=d.loc["exact", f"{readout}_std"],
                    fmt=fs.DATASET_MARKERS[k], color=fs.DATASET_COLORS[k], ms=4, elinewidth=0.5, capsize=0,
                    mec="white", mew=0.3, zorder=3)
        vals += [x, y]
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    ax.plot([lo, hi], [lo, hi], color=fs.MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(f"Whole-matrix distance\n{label}")
    ax.set_ylabel(f"Block distance (Spindle-Exact)\n{label}")
    better = "above" if readout == "expr_r" else "below"
    ax.text(0.03 if better == "above" else 0.97, 0.97 if better == "above" else 0.03, f"block better {better} y = x",
            transform=ax.transAxes, ha="left" if better == "above" else "right",
            va="top" if better == "above" else "bottom", fontsize=fs.TICK_PT, color=fs.MUTED)


def share(ax, t):
    keys = [k for k in ORDER if k in set(t["key"])]
    w = 0.38
    for j, (readout, label) in enumerate(READOUTS):
        d = t[t["readout"] == readout].set_index("key")
        for i, k in enumerate(keys):
            if k not in d.index:
                continue
            ax.bar(i + (j - 0.5) * w, 100 * d.loc[k, "a_better_frac"], width=w, color=fs.DATASET_COLORS[k],
                   alpha=1.0 if j == 0 else 0.45, lw=0)
    ax.axhline(50, color=fs.MUTED, lw=0.6, ls="--")
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([fs.DATASET_SHORT[k] for k in keys], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Queries where block\nneighbours are better (%)")
    ax.legend(handles=[Line2D([], [], color=fs.INK, lw=5, label="expression r"),
                       Line2D([], [], color=fs.INK, lw=5, alpha=0.45, label="composition JSD")],
              loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=fs.TICK_PT)


def main():
    s, t = load()
    fig = fs.figure(fs.DOUBLE, 75)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1.25], wspace=0.6, left=0.08, right=0.99, top=0.92,
                  bottom=0.3)
    a, b, c = (fig.add_subplot(gs[i]) for i in range(3))
    scatter(a, s, *READOUTS[0])
    scatter(b, s, *READOUTS[1])
    share(c, t)
    fs.label_panel(fig, a, "A", -14, 2)
    fs.label_panel(fig, b, "B", -14, 2)
    fs.label_panel(fig, c, "C", -14, 2)
    top = max(t.get_position()[1] for t in fig._fs_letters.values())
    for t in fig._fs_letters.values():  # one baseline for all letters (A and B are square, C is not)
        t.set_y(top)
    fs.legend_below(fig, fs.dataset_handles(ORDER), ncol=8, y=0.07)
    fs.save(fig, "figS12_metric_concordance")


if __name__ == "__main__":
    main()
