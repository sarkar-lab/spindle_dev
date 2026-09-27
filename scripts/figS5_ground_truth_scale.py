"""Fig. S5 -- Whole-matrix ground truth and the scale of epsilon.

A  budget sweep scored against the whole-matrix LE ground truth (band anchored on the
   dataset-median epsilon): Recall@0.1 eps and Overlap@0.5 eps vs budget multiplier
B  within-niche block-LE distances between random pairs of training tiles, in units of the
   niche's own epsilon, with the 0.1, 0.5 and 1 eps bands marked (seed-73 builds)

Inputs: results/budget_sweep/sweep_summary_whole.csv,
results/figure_data/{within_niche_distances.csv, niche_epsilons.csv}
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import figstyle as fs

ORDER = fs.DATASET_ORDER
BANDS = [(0.1, "#DDAA33"), (0.5, "#AA3377"), (1.0, "#1A1A1A")]


def main():
    b = pd.read_csv(fs.RESULTS / "budget_sweep" / "sweep_summary_whole.csv")
    b["key"] = b["Dataset"].map(fs.stem_to_key)
    dist = pd.read_csv(fs.FIG_DATA / "within_niche_distances.csv")
    eps = pd.read_csv(fs.FIG_DATA / "niche_epsilons.csv")
    dist = dist.merge(eps[["dataset", "niche", "epsilon"]], on=["dataset", "niche"])
    dist["d_eps"] = dist["distance"] / dist["epsilon"]

    fig = fs.figure(fs.DOUBLE, 140)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.9], hspace=0.35, left=0.07, right=0.99,
                     top=0.95, bottom=0.08)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1, 1, 0.6], wspace=0.3)
    axes_a = []
    for j, (col, label) in enumerate((("recall_at_eps_0.1", r"Recall@0.1$\varepsilon$"),
                                      ("overlap_at_eps_0.5", r"Overlap@0.5$\varepsilon$"))):
        ax = fig.add_subplot(top[j])
        for key in ORDER:
            d = b[b["key"] == key].sort_values("budget_multiplier")
            ax.plot(d["budget_multiplier"], d[col], color=fs.DATASET_COLORS[key], lw=0.8)
        ax.axvline(1.0, color=fs.MUTED, lw=0.5, ls=":", zorder=0)
        ax.set_xscale("log")
        fs.plain_log(ax, "x", subs=(1.0,))
        ax.set_xlim(0.05, 16)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Budget multiplier")
        ax.set_ylabel(label + " (whole-matrix GT)")
        axes_a.append(ax)
    leg = fig.add_subplot(top[2])
    leg.set_axis_off()
    leg.legend(handles=fs.dataset_handles(marker=False) +
               [Line2D([], [], color=fs.MUTED, ls=":", lw=0.5, label="Production budget (A)")] +
               [Line2D([], [], color=c, ls="--", lw=0.8, label=f"{f:g}ε band (B)") for f, c in BANDS],
               loc="center left")

    grid = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[1], hspace=0.5, wspace=0.15)
    axes_b = []
    bins = np.logspace(np.log10(0.05), np.log10(100), 70)
    for n, key in enumerate(ORDER):
        ax = fig.add_subplot(grid[n // 4, n % 4])
        v = dist.loc[dist["dataset"] == key, "d_eps"].to_numpy()
        ax.hist(np.clip(v, 0.05, 100), bins=bins, color=fs.DATASET_COLORS[key], alpha=0.7, lw=0, density=True)
        for f, c in BANDS:
            ax.axvline(f, color=c, lw=0.8, ls="--")
        ax.set_title(f"{fs.DATASET_LABELS[key]} (median {np.median(v):.1f}ε)", fontsize=fs.TEXT_PT)
        ax.set_xscale("log")
        fs.plain_log(ax, "x", subs=(1.0,))
        ax.set_xlim(0.05, 100)
        ax.set_yticks([])
        if n % 4 == 0:
            ax.set_ylabel("Density")
        if n >= 4:
            ax.set_xlabel(r"Within-niche distance / niche $\varepsilon$")
        axes_b.append(ax)

    fig.canvas.draw()
    fs.label_panel(fig, axes_a[0], "A", dx_mm=-11)
    fs.label_panel(fig, axes_b[0], "B", dx_mm=-8, dy_mm=4)
    fs.save(fig, "figS5_ground_truth_scale")


if __name__ == "__main__":
    main()
