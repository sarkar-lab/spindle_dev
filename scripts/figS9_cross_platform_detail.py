"""Fig. S9 -- Cross-platform search: Spindle-DAG, tile size, seeds and the x2v correction.

A  Spindle-DAG vs Spindle-Exact (global correction): Overlap(5 %, c) against c / N, both directions
B  hit@1/5/10 of the co-located tile for Spindle-Exact and Spindle-DAG (global correction)
C  tile size: tissue-agreement r and hit@5 at <= 2000 vs <= 1000 Xenium cells per tile (global correction)
D  correction ablation on the co-located tile: hit@5 (median rank above the bars)
E  correction ablation: mean distance from the query tile to its top-10 tiles, relative to all index tiles
F  PC1/PC2 before and after correction for x2v (Visium index; Fig. 5B shows v2x)
Spindle-Exact unless stated; mean +- s.d. over seeds 0-4.

Inputs: results/cross_platform_tiers/{summary.csv, dag_curves.csv, bias_pca*.csv}.
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as fs
from fig5_cross_platform import ARMS, DIR_COLORS, DIR_LABELS, DIRS, RES, sel

HITS = [1, 5, 10]
SIZES = [2000, 1000]


def panel_dag_overlap(ax):
    cur = sel(pd.read_csv(RES / "dag_curves.csv"), max_pts=2000)
    for d in DIRS:
        c = cur[cur["direction"] == d]
        x = 100 * c["c_frac_mean"]
        ax.plot(x, c["overlap_0.05_mean"], color=DIR_COLORS[d], lw=1.0, label=DIR_LABELS[d], zorder=3)
        ax.fill_between(x, c["overlap_0.05_mean"] - c["overlap_0.05_std"], c["overlap_0.05_mean"]
                        + c["overlap_0.05_std"], color=DIR_COLORS[d], alpha=0.25, lw=0)
    ax.set_xscale("log")
    fs.plain_log(ax, "x")
    fs.log_ygrid(ax, "x")
    ax.axhline(0.9, color=fs.MUTED, lw=0.6, ls=":")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Tiles returned, c (% of N)")
    ax.set_ylabel("Overlap(5 %, c), DAG vs Exact")
    ax.legend(loc="lower right")


def panel_dag_hits(ax, summ):
    x = np.arange(len(HITS))
    w = 0.2
    for k, (d, tier) in enumerate([(d, t) for d in DIRS for t in ("exact", "dag")]):
        r = sel(summ, max_pts=2000, direction=d, arm="global", tier=tier).iloc[0]
        ax.bar(x + (k - 1.5) * w, [r[f"hit@{h}_mean"] for h in HITS], yerr=[r[f"hit@{h}_std"] for h in HITS],
               width=w * 0.92, color=DIR_COLORS[d], alpha=1.0 if tier == "exact" else 0.45, lw=0,
               error_kw=dict(elinewidth=0.6, capsize=0))
    ax.set_xticks(x, [f"Top {h}" for h in HITS])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Co-located tile found")
    ax.tick_params(axis="x", length=0)
    ax.legend(handles=[Patch(color=fs.INK, label="Spindle-Exact"), Patch(color=fs.INK, alpha=0.45,
                                                                         label="Spindle-DAG")], loc="upper left")


def panel_tile_size(fig, spec, summ):
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.7)
    axes = []
    for j, (col, ylabel) in enumerate((("tissue_r", "Tumour-fraction r"),
                                       ("hit@5", "Co-located tile in top 5"))):
        ax = fig.add_subplot(gs[j])
        for k, d in enumerate(DIRS):
            for i, m in enumerate(SIZES):
                r = sel(summ, max_pts=m, direction=d, arm="global", tier="exact")
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(i + (k - 0.5) * 0.25, r[f"{col}_mean"], yerr=r[f"{col}_std"], fmt="o", ms=3.2, mew=0,
                            color=DIR_COLORS[d], elinewidth=0.7)
        ax.set_xticks(range(len(SIZES)), [f"≤{m}" for m in SIZES])
        ax.set_xlim(-0.6, len(SIZES) - 0.4)
        ax.set_xlabel("Xenium cells per tile")
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(ylabel)
        axes.append(ax)
    return axes


def panel_arms(ax, summ, col, ylabel, ylim, ref=None, note=None):
    x = np.arange(len(ARMS))
    w = 0.36
    for k, d in enumerate(DIRS):
        rows = [sel(summ, max_pts=2000, direction=d, arm=a, tier="exact").iloc[0] for a, _ in ARMS]
        xs = x + (k - 0.5) * w
        ax.bar(xs, [r[f"{col}_mean"] for r in rows], yerr=[r[f"{col}_std"] for r in rows], width=w * 0.92,
               color=DIR_COLORS[d], lw=0, error_kw=dict(elinewidth=0.6, capsize=0))
        if note:
            for xi, r in zip(xs, rows):
                ax.text(xi, r[f"{col}_mean"] + r[f"{col}_std"] + 0.01, f"{r[note]:.0f}", ha="center", va="bottom",
                        fontsize=fs.TICK_PT - 1, color=fs.MUTED)
    if ref is not None:
        ax.axhline(ref, color=fs.MUTED, lw=0.6, ls=":")
    ax.set_xticks(x, [g for _, g in ARMS])
    ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Platform correction")
    ax.tick_params(axis="x", length=0)


def panel_pca_x2v(fig, spec):
    p = sel(pd.read_csv(RES / "bias_pca.csv"), direction="x2v")
    summ = pd.read_csv(RES / "bias_pca_summary.csv").set_index(["direction", "corrected"])
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.12)
    pad_x, pad_y = 0.05 * np.ptp(p["PC1"]), 0.05 * np.ptp(p["PC2"])
    axes = []
    for j, corrected in enumerate((False, True)):
        ax = fig.add_subplot(gs[j])
        d = pd.concat([p[p["role"] == "index"], p[(p["role"] == "query") & (p["corrected"] == corrected)]])
        for mod in ("Visium", "Xenium"):
            m = d[d["modality"] == mod]
            ax.scatter(m["PC1"], m["PC2"], s=4, color=fs.PLATFORM_COLORS[mod], lw=0, alpha=0.8)
        ax.set_xlim(p["PC1"].min() - pad_x, p["PC1"].max() + pad_x)
        ax.set_ylim(p["PC2"].min() - pad_y, p["PC2"].max() + pad_y)
        ax.set_xlabel("PC1")
        if j == 0:
            ax.set_ylabel("PC2")
        else:
            ax.set_yticklabels([])
        sil = summ.loc[("x2v", corrected), "modality_silhouette"]
        ax.set_title(("Corrected" if corrected else "Uncorrected") + f" (silhouette {sil:.2f})", fontsize=fs.TICK_PT)
        axes.append(ax)
    axes[1].legend(handles=[Line2D([], [], ls="none", marker="o", ms=3, color=fs.PLATFORM_COLORS[m], label=m)
                            for m in ("Xenium", "Visium")], loc="upper left")
    return axes


def main():
    summ = pd.read_csv(RES / "summary.csv")
    fig = fs.figure(fs.DOUBLE, 110)
    outer = GridSpec(2, 1, figure=fig, hspace=0.6, left=0.07, right=0.99, top=0.95, bottom=0.16)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.1, 1.0, 1.5], wspace=0.45)
    bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[0.8, 0.8, 1.5], wspace=0.45)
    ax_a = fig.add_subplot(top[0])
    panel_dag_overlap(ax_a)
    ax_b = fig.add_subplot(top[1])
    panel_dag_hits(ax_b, summ)
    ax_c = panel_tile_size(fig, top[2], summ)
    ax_d = fig.add_subplot(bot[0])
    rnd = sel(summ, max_pts=2000, direction="v2x", arm="global", tier="exact")["hit@5_random_mean"].iloc[0]
    panel_arms(ax_d, summ, "hit@5", "Co-located tile in top 5", (0, 0.4), ref=rnd, note="median_rank_mean")
    ax_e = fig.add_subplot(bot[1])
    panel_arms(ax_e, summ, "ratio_k10", "Distance to top-10 tiles\n(relative to random)", (0, 1.1), ref=1.0)
    ax_f = panel_pca_x2v(fig, bot[2])
    fs.legend_below(fig, [Line2D([], [], ls="none", marker="s", ms=4, color=DIR_COLORS[d], label=DIR_LABELS[d])
                          for d in DIRS], ncol=2, y=0.06)

    fig.canvas.draw()
    fs.label_panel(fig, ax_a, "A", dx_mm=-10)
    fs.label_panel(fig, ax_b, "B", dx_mm=-10)
    fs.label_panel(fig, ax_c[0], "C", dx_mm=-10, panel=(ax_c[1],))
    fs.label_panel(fig, ax_d, "D", dx_mm=-10)
    fs.label_panel(fig, ax_e, "E", dx_mm=-10)
    fs.label_panel(fig, ax_f[0], "F", dx_mm=-9)
    fs.save(fig, "figS9_cross_platform_detail")


if __name__ == "__main__":
    main()
