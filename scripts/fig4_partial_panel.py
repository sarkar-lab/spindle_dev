"""Fig. 4 -- Partial-panel search (E11).

A  schematic of the interval index and a partial-panel query (drawn from the code:
   spindle_dev.interval_index + benchmarks/partial_panel_core.search_all_clusters_spindle)
B  Recall@0.05 eps and Recall@0.1 eps vs query length (genes)
C  contiguous vs non-contiguous gene subsets, Recall@0.1 eps per dataset
D  speedup vs query length

Thin lines: one dataset (mean over 5 seeds). Bold line: all datasets pooled,
mean +- s.d. over the 5 seeds. Overlap and the rank distribution were dropped
from the main figure (user decision, 2026-09-25); overlap is in Fig. S9.

Input: results/partial_panel_search/<stem>/seed_<n>/query_metrics.csv
"""

import sys

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle


import figstyle as fs

sys.path.insert(0, str(fs.PROJECT_ROOT / "src"))
from spindle_dev.interval_index import decompose_to_dyadic  # noqa: E402  (the schematic uses the real decomposition)

ORDER = fs.DATASET_ORDER
BINS = ["<=6", "7-12", "13-16", ">16"]
BIN_LABELS = ["≤6", "7–12", "13–16", ">16"]
POOLED = fs.INK


def load():
    frames = []
    for key in ORDER:
        for seed in range(5):
            p = fs.RESULTS / "partial_panel_search" / f"xenium_human_{key}" / f"seed_{seed}" / "query_metrics.csv"
            frames.append(pd.read_csv(p).assign(dataset=key, seed=seed))
    return pd.concat(frames, ignore_index=True)


def agg(g, col):
    """Mean of ``col`` over a group's queries; speedup is mean brute-force time / mean Spindle time,
    the convention of partial_panel_search.py's summary.csv."""
    if col == "speedup":
        return g["brute_force_time_ms"].mean() / g["spindle_time_ms"].mean()
    return g[col].mean()


def by_length(ax, d, col, ylabel, ylim=None):
    x = np.arange(len(BINS))
    per_ds_seed = d.groupby(["dataset", "seed", "Length_Bin"]).apply(agg, col, include_groups=False)
    for key in ORDER:
        m = per_ds_seed.loc[key].groupby("Length_Bin").mean().reindex(BINS)
        ax.plot(x, m, color=fs.DATASET_COLORS[key], lw=0.6, marker=fs.DATASET_MARKERS[key], ms=2.2, mew=0,
                alpha=0.9, zorder=2)
    # pooled: each seed's mean over all datasets' queries in the bin, then mean +- s.d. over seeds
    pooled = d.groupby(["seed", "Length_Bin"]).apply(agg, col, include_groups=False).unstack().reindex(columns=BINS)
    ax.errorbar(x, pooled.mean(), yerr=pooled.std(ddof=1), color=POOLED, lw=1.6, elinewidth=0.8, zorder=3)
    ax.set_xticks(x, BIN_LABELS)
    ax.set_xlim(-0.25, len(BINS) - 0.75)
    ax.set_xlabel("Query length (genes)")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def contiguity(ax, d, col="recall_at_eps_0.1"):
    """Per dataset, contiguous (filled) and non-contiguous (hollow) subsets side by side; bars: s.d. over seeds."""
    s = d.groupby(["dataset", "seed", "Case"])[col].mean().unstack()
    for i, key in enumerate(ORDER):
        c = fs.DATASET_COLORS[key]
        for off, case, fill in ((-0.17, "Contiguous Random", c), (0.17, "Non-Contiguous Random", "white")):
            v = s.loc[key, case]
            ax.errorbar(i + off, v.mean(), yerr=v.std(ddof=1), fmt=fs.DATASET_MARKERS[key], color=c, mfc=fill,
                        mec=c, mew=0.8, ms=3.4, elinewidth=0.6, zorder=3)
    ax.set_xticks(range(len(ORDER)), [fs.DATASET_SHORT[k] for k in ORDER], rotation=35, ha="right",
                  rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel(r"Recall@0.1$\varepsilon$")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", color=fs.MUTED, ms=3.4, label="Contiguous"),
                       Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.MUTED, mew=0.8, ms=3.4,
                              label="Non-contiguous")], loc="lower left", ncol=2)


# ------------------------------------------------------------------ schematic (panel A)
PIECE_COLORS = ["#CC6677", "#DDCC77", "#44AA99", "#88CCEE", "#AA4499"]
G = 2.3          # mm per gene
N_GENES = 16     # genes in the example block
CONTIG = (3, 11)                  # the decompose_to_dyadic example in interval_index.py
NONCONTIG = [(1, 3), (9, 14)]


def _box(ax, x, y, w, h, fc, ec=fs.MUTED, lw=0.4, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, **kw))


def _arrow(ax, x0, x1, y):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=fs.INK,
                                 shrinkA=0, shrinkB=0))


def _title(ax, x, y, n, text):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((x, y), 1.7, fc=fs.INK, ec="none", zorder=4))
    ax.text(x, y, str(n), fontsize=fs.TICK_PT, ha="center", va="center", color="white", zorder=5)
    ax.text(x + 2.8, y, text, fontsize=fs.TEXT_PT, ha="left", va="center")


def _strip(ax, x0, y, genes_on, pieces, h=2.6):
    """A row of N_GENES gene cells (query genes dark) with its dyadic pieces coloured underneath."""
    for g in range(N_GENES):
        _box(ax, x0 + g * G, y, G, h, "#4D4D4D" if g in genes_on else "white", lw=0.3)
    for k, (a, b) in enumerate(pieces):
        _box(ax, x0 + a * G, y - h - 0.9, (b - a) * G, h, PIECE_COLORS[k % 5], ec=fs.INK, lw=0.5)


def panel_schematic(ax, H):
    from matplotlib.patches import Circle
    ax.set_xlim(0, 180)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    top = H - 3.5
    contig_pieces = decompose_to_dyadic(*CONTIG)
    colour_of = {p: PIECE_COLORS[k] for k, p in enumerate(contig_pieces)}
    t = 6

    # ---- 1: index -------------------------------------------------------------
    _title(ax, 8, top, 1, "Index dyadic intervals")
    x0 = 12
    y_rows = []
    y = top - 8
    for L in [16, 8, 4, 2, 1]:
        for a in range(0, N_GENES, L):
            _box(ax, x0 + a * G, y, L * G, 2.6, colour_of.get((a, a + L), "#F2F2F2"), lw=0.4)
        ax.text(x0 - 0.8, y + 1.3, str(L), fontsize=t, ha="right", va="center")
        y_rows.append(y)
        y -= 3.6
    ax.text(x0 - 5.0, (y_rows[0] + y_rows[-1]) / 2 + 1.3, "length", fontsize=t, rotation=90, ha="center",
            va="center")
    ax.text(x0, y + 1.6, "genes of block ℓ, niche j", fontsize=t, ha="left", va="center", color=fs.MUTED)
    # zoom into cell [4, 8): epsilon-clusters of the tiles' log sub-matrices
    cy = 13.5
    y_cell = y_rows[2]
    for xa, xb in ((x0 + 4 * G, 13), (x0 + 8 * G, 31)):
        ax.plot([xa, xb], [y_cell, cy + 5.5], color=fs.MUTED, lw=0.4, ls=":")
    rng = np.random.default_rng(4)
    for (mx, my), r in (((16.5, cy + 1.5), 3.0), ((25.5, cy + 2.2), 2.5), ((21.5, cy - 3.2), 2.7)):
        pts = rng.normal(size=(6, 2)) * r * 0.33
        ax.scatter(mx + pts[:, 0], my + pts[:, 1], s=2.5, color=fs.MUTED, lw=0, zorder=3)
        ax.add_patch(Circle((mx, my), r, fill=False, ec=fs.INK, lw=0.5, ls=(0, (2, 1))))
        ax.plot(mx, my, marker="x", color=fs.INK, ms=3, mew=0.8, zorder=4)
    ax.text(31.5, cy - 0.5, "per cell: clusters of\nthe tiles' log sub-\nmatrices; × centroid,\nradius ≤ ε/2",
            fontsize=t, ha="left", va="center", linespacing=1.25)

    _arrow(ax, 51, 55, top - 16)

    # ---- 2: query -------------------------------------------------------------
    x1 = 59
    _title(ax, x1 - 1, top, 2, "Split the query into dyadic pieces")
    y = top - 10
    ax.text(x1, y + 3.4, "contiguous: genes 3–10 → 4 pieces", fontsize=t, ha="left", va="bottom")
    _strip(ax, x1, y, set(range(*CONTIG)), contig_pieces)
    y = top - 24
    on = {g for a, b in NONCONTIG for g in range(a, b)}
    nc_pieces = [p for a, b in NONCONTIG for p in decompose_to_dyadic(a, b)]
    ax.text(x1, y + 3.4, "non-contiguous: genes 1–2, 9–13 → 5 pieces", fontsize=t, ha="left", va="bottom")
    _strip(ax, x1, y, on, nc_pieces)
    ax.text(x1, 11, "Each run of consecutive genes (in the block's\norder) gives at most 2 log₂(run length)\n"
            "pieces. Repeated in every niche's gene order.", fontsize=t, ha="left", va="center",
            linespacing=1.25, color=fs.MUTED)

    _arrow(ax, 101, 105, top - 16)

    # ---- 3: score pieces -------------------------------------------------------
    x2 = 109
    _title(ax, x2 - 1, top, 3, "Score each piece")
    n = CONTIG[1] - CONTIG[0]
    m = 2.2
    my_top = top - 6.5
    for i in range(n):
        for j in range(n):
            _box(ax, x2 + j * m, my_top - (i + 1) * m, m, m, "#EDEDED", ec="white", lw=0.3)
    for (a, b), c in colour_of.items():
        a0, w = a - CONTIG[0], b - a
        _box(ax, x2 + a0 * m, my_top - (a0 + w) * m, w * m, w * m, c, ec=fs.INK, lw=0.5)
    ax.text(x2, my_top - n * m - 1.2, "query covariance on S", fontsize=t, ha="left", va="top")
    ax.text(x2 + n * m + 1.8, my_top - 1.0, "colour: piece blocks\n$Q_i$, compared with\nthe cell's centroids\n\n"
            "grey: used only in\nthe exact re-rank", fontsize=t, ha="left", va="top", linespacing=1.25)
    ax.text(x2, 16.5, r"$d_i(t) = ||\log Q_i - C_i(t)||_F\, /$ √$p_i$", fontsize=fs.TEXT_PT, ha="left",
            va="center")
    ax.text(x2, 11.5, r"$C_i(t)$: centroid of tile $t$'s cluster in the", fontsize=t, ha="left", va="center",
            color=fs.MUTED)
    ax.text(x2, 8.3, r"cell of piece $i$; $p_i$ genes", fontsize=t, ha="left", va="center", color=fs.MUTED)

    _arrow(ax, 147, 151, top - 16)

    # ---- 4: combine, pool, re-rank --------------------------------------------
    x3 = 154
    _title(ax, x3 + 1, top, 4, "Combine, re-rank")
    ax.text(x3, top - 7, r"$D(t)^2 = \Sigma_i\, p_i\, d_i(t)^2 / |S|$", fontsize=fs.TEXT_PT, ha="left",
            va="center")
    steps = [("all niches' tiles,\nranked by D", [8, 7.3, 6.6, 5.9]), ("keep top 50", [5.2, 4.7, 4.2]),
             ("exact LE on S×S\n→ best match", [3.6, 3.2, 2.8])]
    yy = top - 13
    for k, (txt, widths) in enumerate(steps):
        for r, w in enumerate(widths):
            _box(ax, x3, yy - r * 1.5, w, 1.0, fs.INK if (k == 2 and r == 0) else "#BDBDBD", ec="none")
        ax.text(x3 + 9.5, yy - 1.5, txt, fontsize=t, ha="left", va="center", linespacing=1.2)
        if k < 2:
            ax.add_patch(FancyArrowPatch((x3 + 2, yy - len(widths) * 1.5 - 0.3), (x3 + 2, yy - len(widths) * 1.5 - 3.2),
                                         arrowstyle="-|>", mutation_scale=5, lw=0.5, color=fs.MUTED))
        yy -= len(widths) * 1.5 + 5.5


def main():
    d = load()
    H_SCHEM = 58.0
    H = 125.0
    fig = fs.figure(fs.DOUBLE, H)
    ax_a = fig.add_axes([0, 1 - H_SCHEM / H, 1, H_SCHEM / H])
    panel_schematic(ax_a, H_SCHEM)

    gs = GridSpec(1, 4, figure=fig, width_ratios=[1, 1, 1.45, 1], wspace=0.45, left=0.065, right=0.985,
                  top=1 - (H_SCHEM + 8) / H, bottom=0.2)
    ax_b1 = fig.add_subplot(gs[0])
    by_length(ax_b1, d, "recall_at_eps_0.05", r"Recall@0.05$\varepsilon$", (0.7, 1.005))
    ax_b2 = fig.add_subplot(gs[1])
    by_length(ax_b2, d, "recall_at_eps_0.1", r"Recall@0.1$\varepsilon$", (0.7, 1.005))
    ax_c = fig.add_subplot(gs[2])
    contiguity(ax_c, d)
    ax_d = fig.add_subplot(gs[3])
    by_length(ax_d, d, "speedup", "Speedup vs brute force", (0, None))

    fig.canvas.draw()
    fs.label_at(fig, 1.0, 1.5, "A")
    for ax, letter in ((ax_b1, "B"), (ax_c, "C"), (ax_d, "D")):
        fs.label_panel(fig, ax, letter, dx_mm=-11)
    handles = fs.dataset_handles(ms=3.0) + [Line2D([], [], color=POOLED, lw=1.6, label="All datasets")]
    fs.legend_below(fig, handles, ncol=9, y=0.045)
    fs.save(fig, "fig4_partial_panel")


if __name__ == "__main__":
    main()
