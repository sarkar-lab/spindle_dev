"""Fig. S12 -- Block vs whole-matrix LE neighbours against biology (E14).

A  breast, seeds 0-4 (mean +- s.d.): expression r, composition JSD and majority-type match
   of each metric's exact top-k neighbours, vs k; random tiles as reference
B  all datasets, seed 73: expression r of the exact top-10, block vs whole (y = x)
C  all datasets, seed 73: per-query Spearman rho between distance and expression
   dissimilarity over all training tiles, block vs whole (y = x) -- the global-ordering caveat

Readouts enter neither distance. Input: results/metric_concordance/summary.csv
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import figstyle as fs

KIND_COLORS = {"block": "#1A1A1A", "whole": "#DDAA33", "random": "#9E9E9E"}
KIND_LABELS = {"block": "Block LE", "whole": "Whole-matrix LE", "random": "Random tiles"}
READOUTS = [("expr_r", "Expression r ↑"), ("comp_jsd", "Composition JSD ↓"), ("majority_match", "Majority-type match ↑")]
KS = ["1", "5", "10", "50"]


def scatter_block_whole(ax, s, readout, k, lo, hi, label):
    d = s[(s["seed"] == 73) & (s["readout"] == readout) & (s["k"] == k)]
    for r in d.itertuples():
        key = fs.stem_to_key(r.dataset)
        ax.scatter(r.whole, r.block, s=14, color=fs.DATASET_COLORS[key], marker=fs.DATASET_MARKERS[key], lw=0,
                   zorder=3)
        if key == "kidney_nondiseased":
            ax.annotate("Kidney (1 niche)", (r.whole, r.block), xytext=(4, -8), textcoords="offset points",
                        fontsize=fs.TICK_PT)
    fs.identity_line(ax, lo, hi)
    ax.set_xlabel(f"{label}, whole-matrix LE")
    ax.set_ylabel(f"{label}, block LE")


def main():
    s = pd.read_csv(fs.RESULTS / "metric_concordance" / "summary.csv")
    s["k"] = s["k"].astype(str)
    breast = s[(s["dataset"] == "xenium_human_breast_cancer") & s["seed"].between(0, 4)]

    fig = fs.figure(fs.DOUBLE, 110)
    outer = GridSpec(2, 1, figure=fig, hspace=0.5, left=0.07, right=0.99, top=0.94, bottom=0.08)
    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], width_ratios=[1, 1, 1, 0.8], wspace=0.45)
    bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1, 1, 0.9], wspace=0.5)

    axes_a = []
    x = np.arange(len(KS))
    for j, (readout, label) in enumerate(READOUTS):
        ax = fig.add_subplot(top[j])
        d = breast[breast["readout"] == readout]
        for kind, c in KIND_COLORS.items():
            g = d.groupby("k")[kind].agg(["mean", "std"]).reindex(KS)
            ax.errorbar(x, g["mean"], yerr=g["std"], color=c, marker="o", ms=2.8, mew=0, lw=0.9, elinewidth=0.6)
        ax.set_xticks(x, KS)
        ax.set_xlabel("Neighbours, k")
        ax.set_title(label, fontsize=fs.TEXT_PT)
        ax.set_ylim(0, None)
        axes_a.append(ax)
    axes_a[0].set_ylabel("Breast, mean over queries")
    leg = fig.add_subplot(top[3])
    leg.set_axis_off()
    leg.legend(handles=[Line2D([], [], color=c, marker="o", ms=2.8, lw=0.9, label=KIND_LABELS[k])
                        for k, c in KIND_COLORS.items()], loc="center left")

    ax_b = fig.add_subplot(bot[0])
    scatter_block_whole(ax_b, s, "expr_r", "10", 0.0, 0.75, "Top-10 expression r")
    ax_c = fig.add_subplot(bot[1])
    scatter_block_whole(ax_c, s, "rho_expr", "all", 0.0, 0.75, r"Global $\rho$")
    leg2 = fig.add_subplot(bot[2])
    leg2.set_axis_off()
    leg2.legend(handles=fs.dataset_handles(), loc="center left")

    fig.canvas.draw()
    fs.label_panel(fig, axes_a[0], "A", dx_mm=-11)
    fs.label_panel(fig, ax_b, "B", dx_mm=-11)
    fs.label_panel(fig, ax_c, "C", dx_mm=-11)
    fs.save(fig, "figS12_metric_concordance")


if __name__ == "__main__":
    main()
