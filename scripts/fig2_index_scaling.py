"""Fig. 2 -- Index construction and scaling.

A  dataset overview dot matrix (cells, genes, tiles, niches, blocks) of the production (seed-73) builds
B  build wall time and peak memory vs cells (seeds 0-4)
C  compression: dense covariance storage vs index size (seeds 0-4)
D  synthetic sweep 1e3-1e6 cells: index-only build time and index size

The niche-map panel of the plan was dropped (user decision, 2026-09-25).
Inputs: results/index_stats/index_stats{,_summary}.csv, results/scalability_sweep/*.
"""

import json

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import figstyle as fs

ORDER = fs.DATASET_ORDER


def load_stats():
    s = pd.read_csv(fs.RESULTS / "index_stats" / "index_stats_summary.csv").set_index("dataset").loc[ORDER]
    return s


def load_production():
    """Seed-73 production builds: integer niche/block counts (they vary across seeds 0-4)."""
    p = pd.read_csv(fs.RESULTS / "index_stats" / "index_stats.csv")
    p = p[p["seed"] == 73].set_index("dataset").loc[ORDER]
    p["tiles"] = p["n_tiles_train"] + p["n_tiles_query"]
    return p


def panel_overview(ax, s):
    cols = [("cells", "Cells", lambda v: f"{v / 1e3:.0f}k"),
            ("genes", "Genes", lambda v: f"{v:.0f}"),
            ("tiles", "Tiles", lambda v: f"{v:,.0f}"),
            ("n_niches", "Niches", lambda v: f"{v:.0f}"),
            ("n_blocks", "Blocks", lambda v: f"{v:.0f}")]
    for j, (col, _, fmt) in enumerate(cols):
        vmax = s[col].max()
        vmin = s[col].min()
        for i, key in enumerate(ORDER):
            v = s.loc[key, col]
            ax.scatter(j, i, s=20 + 150 * v / vmax, color=fs.DATASET_COLORS[key], lw=0, alpha=0.45, zorder=2)
        ax.text(j, len(ORDER) - 0.1, f"({fmt(vmin)}–{fmt(vmax)})", ha="center", va="top",
                fontsize=fs.TICK_PT - 1, color=fs.MUTED, zorder=3)
    ax.set_xticks(range(len(cols)), [c[1] for c in cols])
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(len(ORDER)), [fs.DATASET_SHORT[k] for k in ORDER])
    ax.set_xlim(-0.55, len(cols) - 0.45)
    ax.set_ylim(len(ORDER) + 0.35, -0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def err(ax, x, y, xerr, yerr, key, **kw):
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=fs.DATASET_MARKERS[key], color=fs.DATASET_COLORS[key],
                ms=3.2, elinewidth=0.6, mew=0, zorder=3, **kw)


def panel_compression(ax, s):
    for key in ORDER:
        r = s.loc[key]
        err(ax, r["dense_train_cov_mb_mean"], r["index_size_mb_mean"], None, r["index_size_mb_sd"], key)
    # Axes span 200-30,000 MB x 10-400 MB; the dashed guides mark constant compression ratios.
    lo, hi = 100, 3e4
    for ratio in (10, 30, 100):
        ax.plot([lo, hi], [lo / ratio, hi / ratio], color=fs.LIGHT, lw=0.6, ls="--", zorder=1)
        y_lab = min(300, 0.8 * hi / ratio)
        ax.text(y_lab * ratio / 1.15, y_lab, f"{ratio}×", fontsize=fs.TICK_PT, color=fs.MUTED,
                ha="right", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(200, hi)
    ax.set_ylim(10, 400)
    fs.plain_log(ax, "x", subs=(1.0,))
    fs.plain_log(ax, "y", subs=(1.0, 3.0))
    ax.set_xlabel("Dense covariances (MB)")
    ax.set_ylabel("Index size (MB)")
    lo_r, hi_r = s["compression_ratio_mean"].min(), s["compression_ratio_mean"].max()
    ax.text(0.97, 0.04, f"{lo_r:.0f}–{hi_r:.0f}× smaller", transform=ax.transAxes, fontsize=fs.TICK_PT,
            ha="right", va="bottom")


def panel_vs_cells(ax, s, col, ylabel):
    for key in ORDER:
        r = s.loc[key]
        err(ax, r["cells_mean"], r[f"{col}_mean"], None, r[f"{col}_sd"], key)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fs.log_ygrid(ax)
    fs.plain_log(ax, "y")
    ax.set_xlabel("Cells")
    ax.set_ylabel(ylabel)
    ax.set_xlim(6e4, 1.2e6)


def panel_sweep(ax, col, ylabel):
    for key in ("breast_cancer", "lymph_node"):
        stem = f"xenium_human_{key}"
        d = pd.read_csv(fs.RESULTS / "scalability_sweep" / f"{stem}_synthetic_scaling.csv")
        fit = json.loads((fs.RESULTS / "scalability_sweep" / f"{stem}_scaling_exponents.json").read_text())[col]
        x = d["actual_cells"].to_numpy(float)
        c = fs.DATASET_COLORS[key]
        ax.plot(x, d[col], marker=fs.DATASET_MARKERS[key], color=c, ms=3, lw=0.8, mew=0, zorder=3)
        xx = np.array([8e2, 1.25e6])
        ax.plot(xx, 10 ** fit["intercept"] * xx ** fit["scaling_exponent"], color=c, lw=0.6, ls=":", zorder=2)
        base = d["real_base_cells"].iloc[0]
        ax.axvline(base, color=c, lw=0.5, ls="-", alpha=0.5, zorder=1)
    ax.axvspan(1e5, 5e5, color="#F2F2F2", lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fs.log_ygrid(ax)
    fs.plain_log(ax, "y")
    ax.set_xlabel("Cells (synthetic)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(8e2, 1.25e6)


def sweep_labels(ax, col):
    lines = []
    for key in ("breast_cancer", "lymph_node"):
        fit = json.loads((fs.RESULTS / "scalability_sweep" /
                          f"xenium_human_{key}_scaling_exponents.json").read_text())[col]
        lines.append((key, fit["scaling_exponent"], fit["r_squared"]))
    for i, (key, b, r2) in enumerate(lines):
        y = 0.17 - i * 0.11
        ax.text(0.76, y, f"slope {b:.2f}", transform=ax.transAxes, fontsize=fs.TICK_PT, color=fs.INK,
                ha="left", va="center")
        ax.plot([0.73], [y], transform=ax.transAxes, marker=fs.DATASET_MARKERS[key],
                color=fs.DATASET_COLORS[key], ms=2.8, mew=0)


def main():
    s = load_stats()
    prod = load_production()

    fig = fs.figure(fs.DOUBLE, 122)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.1, 1], hspace=0.42,
                     left=0.09, right=0.99, top=0.93, bottom=0.14)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.35, 0.85, 0.85], wspace=0.45)
    bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[0.9, 1.2, 1.2], wspace=0.4)

    ax_a = fig.add_subplot(top[0])
    panel_overview(ax_a, prod)
    ax_b1 = fig.add_subplot(top[1])
    panel_vs_cells(ax_b1, s, "build_wall_time_s", "Build time (s)")
    ax_b2 = fig.add_subplot(top[2])
    panel_vs_cells(ax_b2, s, "peak_rss_gb", "Peak memory (GB)")

    ax_c = fig.add_subplot(bot[0])
    panel_compression(ax_c, s)
    ax_d1 = fig.add_subplot(bot[1])
    panel_sweep(ax_d1, "build_time_s", "Index build time (s)")
    sweep_labels(ax_d1, "build_time_s")
    ax_d2 = fig.add_subplot(bot[2])
    panel_sweep(ax_d2, "index_size_mb", "Index size (MB)")
    sweep_labels(ax_d2, "index_size_mb")

    fig.canvas.draw()
    fs.label_panel(fig, ax_a, "A", dx_mm=-16.0, dy_mm=5.0)
    fs.label_panel(fig, ax_b1, "B", dx_mm=-11.0)
    fs.label_panel(fig, ax_c, "C", dx_mm=-11.0)
    fs.label_panel(fig, ax_d1, "D", dx_mm=-11.0)

    handles = fs.dataset_handles()
    handles += [Line2D([], [], color=fs.MUTED, lw=0.6, ls=":", label="Power-law fit (D)"),
                Rectangle((0, 0), 1, 1, fc="#F2F2F2", ec="none", label="Subsample → bootstrap (D)")]
    fs.legend_below(fig, handles, ncol=10, y=0.04)
    fs.save(fig, "fig2_index_scaling")


if __name__ == "__main__":
    main()
