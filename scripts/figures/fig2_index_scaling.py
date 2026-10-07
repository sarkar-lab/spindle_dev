"""Fig. 2 -- Index construction, storage and scaling.

A  the eight datasets: cells, genes, training tiles, niches, blocks (production builds, seed 73);
   marker area scaled within each column, the column's range printed below it
B  storage per dataset: whole covariances -> Spindle-Exact (block logs) -> Spindle-DAG, with folds
C  index build time (h5ad -> Spindle-Exact, and -> Spindle-DAG, which is built from Exact's block
   logs) and peak memory per dataset (seeds 0-4)
D  cell ladder: one Xenium section (human tonsil, 1.35M cells, 377 genes) subsampled; storage, build
   time and peak memory vs cells
E  gene ladder (lymph node 5k, all tiles, streaming builder): the same rows vs genes

Storage: float32 upper triangles, training tiles only. Retrieval overlap is not shown here
(supplementary S7).
Inputs: results/{index_stats,dag_size_table,tier_build_stats,dag_cell_ladder,dag_gene_ladder}/.
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import figstyle as fs

ORDER = fs.DATASET_ORDER
MB = 1024 ** 2
TIERS = fs.TIER_ORDER
SAME_TOL = 0.01  # Exact and DAG peaks within 1 % are drawn as one "same for both" series


def whole_mb(n_tiles, p):
    return n_tiles * p * (p + 1) / 2 * 4 / MB


def same_peak(a, b):
    return bool(np.all(np.abs(np.asarray(b) - np.asarray(a)) <= SAME_TOL * np.asarray(b)))


# ---------------------------------------------------------------- data
def load_datasets():
    size = pd.read_csv(fs.RESULTS / "dag_size_table" / "summary.csv")
    size["key"] = size.dataset.map(fs.STEMS)
    size = size.set_index("key").loc[ORDER]
    stats = pd.read_csv(fs.RESULTS / "index_stats" / "index_stats.csv")
    size["cells"] = stats[stats.seed == 73].set_index("dataset").loc[ORDER, "cells"]
    size["whole"], size["exact"], size["dag"] = size.whole_cov_upper_mb, size.block_logs_mb, size.dag_mb
    build = pd.read_csv(fs.RESULTS / "tier_build_stats" / "summary_by_dataset.csv")
    build["key"] = build.dataset.map(fs.STEMS)
    return size, build.set_index("key").loc[ORDER]


def load_cell_ladder():
    s = pd.read_csv(fs.RESULTS / "dag_cell_ladder" / "summary.csv")
    s = s[s.status == "ok"].sort_values("cells").copy()
    s["whole"], s["exact"], s["dag"] = whole_mb(s.n_tiles, s.n_genes), s.exact_mb, s.dag_mb
    return s


def load_gene_ladder():
    files = sorted((fs.RESULTS / "dag_gene_ladder").glob("lymph_node_5k_G*_summary.csv"))
    s = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    s = s[s.status == "ok"].sort_values("n_genes").copy()
    s["whole"], s["exact"], s["dag"] = whole_mb(s.n_tiles, s.n_genes), s.exact_mb, s.dag_mb
    return s


# ---------------------------------------------------------------- top row: one row per dataset
def _rows(ax):
    ax.set_ylim(len(ORDER) + 0.35, -0.5)
    ax.set_yticks(range(len(ORDER)))
    ax.tick_params(axis="y", length=0)


def panel_overview(ax, size):
    cols = [("cells", "Cells", lambda v: f"{v / 1e3:.0f}k"),
            ("n_genes", "Genes", lambda v: f"{v:.0f}"),
            ("n_tiles", "Tiles", lambda v: f"{v:,.0f}"),
            ("n_niches", "Niches", lambda v: f"{v:.0f}"),
            ("n_blocks", "Blocks", lambda v: f"{v:.0f}")]
    for j, (col, _, fmt) in enumerate(cols):
        vmax, vmin = size[col].max(), size[col].min()
        for i, key in enumerate(ORDER):
            v = size.loc[key, col]
            ax.scatter(j, i, s=20 + 150 * v / vmax, color=fs.DATASET_COLORS[key], lw=0, alpha=0.45, zorder=2)
        ax.text(j, len(ORDER) - 0.1, f"{fmt(vmin)}–{fmt(vmax)}", ha="center", va="top",
                fontsize=fs.TICK_PT - 1.5, color=fs.MUTED, zorder=3)
    ax.set_xticks(range(len(cols)), [c[1] for c in cols])
    ax.xaxis.set_ticks_position("top")
    _rows(ax)
    ax.set_yticklabels([fs.DATASET_SHORT[k] for k in ORDER])
    ax.set_xlim(-0.55, len(cols) - 0.45)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)


def _row_axis(ax, xlabel):
    ax.set_xscale("log")
    fs.log_ygrid(ax, "x")
    ax.set_xlabel(xlabel)
    _rows(ax)
    ax.set_yticklabels([])
    fs.despine(ax, left=True)


def panel_storage(ax, size):
    for i, key in enumerate(ORDER):
        vals = [size.loc[key, t] for t in TIERS]
        ax.plot(vals, [i] * 3, color=fs.LIGHT, lw=1.2, zorder=1)
        for t, v in zip(TIERS, vals):
            ax.plot(v, i, marker=fs.TIER_MARKERS[t], color=fs.TIER_COLORS[t], ms=3.6, mec="white", mew=0.4,
                    ls="none", zorder=3)
        for a, b in ((vals[0], vals[1]), (vals[1], vals[2])):
            ax.text(np.sqrt(a * b), i - 0.2, fs.fmt_x(a / b), ha="center", va="bottom", fontsize=fs.TICK_PT - 1,
                    color=fs.MUTED)
    _row_axis(ax, "Storage (MB, log scale)")
    fs.plain_log(ax, "x", subs=(1.0,))


def panel_time(ax, build):
    y = np.arange(len(ORDER))
    for t, col, dy in (("exact", "exact_build_s", -0.17), ("dag", "build_s", 0.17)):  # offset: they nearly coincide
        ax.errorbar(build[f"{col}_mean"], y + dy, xerr=build[f"{col}_sd"], fmt=fs.TIER_MARKERS[t], color=fs.TIER_COLORS[t],
                    ms=3.0, mec="white", mew=0.3, elinewidth=0.6, capsize=1.2, zorder=3)
    extra = build.build_s_mean - build.exact_build_s_mean
    for i, key in enumerate(ORDER):
        ax.text(1.03, i, f"+{extra[key]:.1f} s", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=fs.TICK_PT - 1, color=fs.TIER_COLORS["dag"])
    ax.text(1.03, -0.75, "DAG\nadds", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=fs.TICK_PT - 1, color=fs.MUTED)
    _row_axis(ax, "Build time (s, log scale)")
    fs.plain_log(ax, "x")


def panel_memory(ax, build):
    y = np.arange(len(ORDER))
    same = same_peak(build.peak_rss_exact_gb_mean, build.peak_rss_gb_mean)
    series = [("same", "peak_rss_gb")] if same else [("exact", "peak_rss_exact_gb"), ("dag", "peak_rss_gb")]
    for t, col in series:
        color = fs.INK if t == "same" else fs.TIER_COLORS[t]
        ax.errorbar(build[f"{col}_mean"], y, xerr=build[f"{col}_sd"], fmt="o" if t == "same" else fs.TIER_MARKERS[t],
                    color=color, ms=3.0, elinewidth=0.6, capsize=1.2, zorder=3)
    if same:
        ax.set_title("Same for Exact and DAG", fontsize=fs.TICK_PT, color=fs.MUTED, pad=2)
    _row_axis(ax, "Peak memory (GB, log scale)")
    fs.plain_log(ax, "x")
    return same


# ---------------------------------------------------------------- ladders: storage / time / memory rows
def _lines(ax, df, x, col, color, marker, ls="-"):
    ax.plot(x(df), df[col], color=color, marker=marker, ms=2.6, lw=1.0, ls=ls, mec="white", mew=0.3, zorder=3)


def ladder_rows(fig, spec, df, x, mem_cols, title, xlabel, xlog, cache_col=None):
    gs = GridSpecFromSubplotSpec(3, 1, spec, hspace=0.2)
    axes = [fig.add_subplot(gs[r, 0]) for r in range(3)]
    a_s, a_t, a_m = axes
    for t in TIERS:
        _lines(a_s, df, x, t, fs.TIER_COLORS[t], fs.TIER_MARKERS[t])
    _lines(a_t, df, x, "exact_build_s", fs.TIER_COLORS["exact"], fs.TIER_MARKERS["exact"])
    _lines(a_t, df, x, "build_s", fs.TIER_COLORS["dag"], fs.TIER_MARKERS["dag"], ls="--")
    extra = df.build_s - df.exact_build_s
    share = (extra / df.build_s).max()
    a_t.text(0.02, 0.95, f"DAG adds {extra.min():.1f}–{extra.max():.1f} s (≤ {100 * share:.1f} % of the build)",
             transform=a_t.transAxes, ha="left", va="top", fontsize=fs.TICK_PT - 1, color=fs.TIER_COLORS["dag"])
    exact_col, dag_col = mem_cols
    if same_peak(df[exact_col], df[dag_col]):
        _lines(a_m, df, x, dag_col, fs.INK, "o")
        a_m.text(0.02, 0.95, "Same for Exact and DAG", transform=a_m.transAxes, ha="left", va="top",
                 fontsize=fs.TICK_PT - 1, color=fs.MUTED)
    else:
        _lines(a_m, df, x, exact_col, fs.TIER_COLORS["exact"], fs.TIER_MARKERS["exact"])
        _lines(a_m, df, x, dag_col, fs.TIER_COLORS["dag"], fs.TIER_MARKERS["dag"], ls="--")
    if cache_col is not None:
        a_m.plot(x(df), df[cache_col], color=fs.MUTED, ls=":", lw=0.9, zorder=2)
        a_m.text(x(df).iloc[-1], df[cache_col].iloc[-1], " incl. disk\n cache pages", fontsize=fs.TICK_PT - 1,
                 color=fs.MUTED, va="center", ha="left")
    for ax, label in zip(axes, ["Storage (MB)", "Build time (s)", "Peak memory (GB)"]):
        ax.set_yscale("log")
        fs.plain_log(ax, "y", subs=(1.0,) if ax is a_s else (1.0, 2.0, 3.0, 5.0))
        fs.log_ygrid(ax)
        fs.despine(ax)
        ax.set_ylabel(f"{label}\nlog scale", linespacing=1.0)
        if xlog:
            ax.set_xscale("log")
        if ax is not a_m:
            ax.tick_params(axis="x", labelbottom=False)
    if xlog:
        fs.plain_log(a_m, "x", subs=(1.0,))
        for ax in axes:
            ax.set_xlim(7e2, 2e6)
    else:
        for ax in axes:
            ax.set_xlim(0, 4900)
    a_m.set_xlabel(xlabel)
    a_s.set_title(title, fontsize=fs.TEXT_PT, pad=3)
    return axes


# ---------------------------------------------------------------- figure
def main():
    size, build = load_datasets()
    cells = load_cell_ladder()
    genes = load_gene_ladder()

    fig = fs.figure(fs.DOUBLE, 160)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[0.78, 1.25], hspace=0.34,
                     left=0.085, right=0.95, top=0.94, bottom=0.10)
    top = GridSpecFromSubplotSpec(1, 4, outer[0], width_ratios=[1.75, 1.2, 0.75, 0.75], wspace=0.3)
    ax_a, ax_b, ax_t, ax_m = (fig.add_subplot(top[i]) for i in range(4))
    panel_overview(ax_a, size)
    panel_storage(ax_b, size)
    panel_time(ax_t, build)
    panel_memory(ax_m, build)

    bottom = GridSpecFromSubplotSpec(1, 2, outer[1], wspace=0.32)
    d_axes = ladder_rows(fig, bottom[0], cells, lambda d: d.cells, ("peak_rss_exact_gb", "peak_rss_gb"),
                         "Human tonsil (1.35M cells, 377 genes), subsampled", "Cells (log scale)", xlog=True)
    e_axes = ladder_rows(fig, bottom[1], genes, lambda d: d.n_genes, ("mem_anon_exact_gb", "mem_anon_gb"),
                         "Lymph node 5k, all 7,646 tiles", "Genes", xlog=False, cache_col="peak_rss_gb")

    fs.label_panel(fig, ax_a, "A", dx_mm=-15, dy_mm=4)
    fs.label_panel(fig, ax_b, "B", dx_mm=-3, dy_mm=4)
    fs.label_panel(fig, ax_t, "C", dx_mm=-3, dy_mm=4, panel=(ax_m,))
    fs.label_panel(fig, d_axes[0], "D", dx_mm=-14, dy_mm=4, panel=d_axes[1:])
    fs.label_panel(fig, e_axes[0], "E", dx_mm=-14, dy_mm=4, panel=e_axes[1:])

    handles = [Line2D([], [], color=fs.TIER_COLORS[t], marker=fs.TIER_MARKERS[t], ms=3.5, lw=1.0,
                      ls="--" if t == "dag" else "-", label=fs.TIER_LABELS[t]) for t in TIERS]
    handles += [Line2D([], [], color=fs.INK, marker="o", ms=3.0, lw=1.0, label="Same for Exact and DAG")]
    fs.legend_below(fig, handles, ncol=4, y=0.035)
    fs.save(fig, "fig2_index_scaling")


if __name__ == "__main__":
    main()
