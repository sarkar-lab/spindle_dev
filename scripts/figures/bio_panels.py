"""Shared panels for the biology figures (Figs 6, 7, S10).

CSV / npy inputs only:
  results/figure_data/tiles_seed73.csv          tile boxes, status and niche of the seed-73 tiling
  results/niche_concordance/<key>_*.csv         niche x cell type composition, ARI / NMI, cell-type names
  results/block_programs/<stem>/                blocks, enrichment, niche-mean correlations, rewiring
  results/signature_queries/<stem>/             signature statistics, top matches, per-cell scores

Panels:
  niche_map          tiles coloured by covariance niche on the tissue
  niche_cells        every cell coloured by its tile's niche (held-out / dropped tiles' cells light grey)
  composition        niche x cell type heatmap (fraction of the niche's cells)
  enrichment_heatmap niche x cell type log2 enrichment over the section, with the section's fractions
  block_heatmap      a niche-mean correlation in a given gene order with that niche's blocks outlined
  signature_map      per-cell signature score with returned tiles outlined

Niches and blocks are printed 1-based; the CSVs are 0-based.
Axes are placed in millimetres from the figure's top-left corner (ax_mm).
"""

import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

import figstyle as fs

BLOCKS = fs.RESULTS / "block_programs"
SIGS = fs.RESULTS / "signature_queries"
NICHE = fs.RESULTS / "niche_concordance"
SEED = 73
FIG_SEEDS = [0, 1, 2, 3, 4]
CORR_LIM = 0.6       # niche-mean correlations rarely exceed this; clipped
_tiles = None


# ---------------------------------------------------------------- layout / data
def ax_mm(fig, x, y, w, h, **kw):
    """Axes at x, y (mm from the left / top edges), w x h mm."""
    W, H = fig.get_size_inches() / fs.MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def stem(key):
    return f"xenium_human_{key}"


def all_tiles():
    global _tiles
    if _tiles is None:
        _tiles = pd.read_csv(fs.FIG_DATA / "tiles_seed73.csv")
    return _tiles


def tiles(key):
    return all_tiles().query("dataset == @key")


def tissue_aspect(t):
    return (t.y1.max() - t.y0.min()) / (t.x1.max() - t.x0.min())


def tissue_axes(ax, t):
    ax.set_aspect("equal")
    ax.set_xlim(t.x0.min(), t.x1.max())
    ax.set_ylim(t.y1.max(), t.y0.min())  # image orientation, as in Figs 1, 2 and 5
    ax.set_axis_off()


def rects(ax, rows, **kw):
    pc = PatchCollection([Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0) for r in rows.itertuples()], **kw)
    ax.add_collection(pc)
    return pc


def niche_label(j):
    return f"Niche {j + 1}"


def n_niches(t):
    return int(t.loc[t.status == "train", "niche"].max()) + 1


def blocks(key, seed=SEED):
    """seed<s>_blocks.csv with gene lists split; rows in (niche, block) order = each niche's gene order."""
    b = pd.read_csv(BLOCKS / stem(key) / f"seed{seed}_blocks.csv")
    b["gene_list"] = b.genes.str.split()
    return b


def niche_order(b, k):
    """The niche's gene order (data.perm_list[k]) and its block runs [(start, end)]."""
    rows = b[b.niche == k].sort_values("block")
    genes, runs, s = [], [], 0
    for g in rows.gene_list:
        genes += g
        runs.append((s, s + len(g)))
        s += len(g)
    return genes, runs


def corr(key, k, b, order=None, seed=SEED):
    """Niche k's mean correlation (shrunk tiles) as a DataFrame, rows / columns in ``order`` (default its own)."""
    own, _ = niche_order(b, k)
    R = pd.DataFrame(np.load(BLOCKS / stem(key) / f"seed{seed}_corr_niche{k}.npy"), index=own, columns=own)
    return R if order is None else R.loc[order, order]


# ---------------------------------------------------------------- niche map / composition
def niche_map(ax, t, only=None, lw=0.1):
    """Training tiles in their niche colour; held-out / dropped tiles light grey.

    only: a niche index to show in colour with every other tile light grey."""
    pal = fs.niche_palette(n_niches(t))
    grey = t[(t.status != "train") | ((t.niche != only) if only is not None else False)]
    rects(ax, grey, facecolor=fs.LIGHT, edgecolor="white", lw=lw)
    tr = t[(t.status == "train") & ((t.niche == only) if only is not None else True)]
    rects(ax, tr, facecolor=[pal[int(n)] for n in tr.niche], edgecolor="white", lw=lw)
    tissue_axes(ax, t)


def cell_niches(cells, t):
    """Niche of each cell's training tile (-1: the cell lies in a held-out or dropped tile)."""
    out = np.full(len(cells), -1, dtype=int)
    x, y = cells.x.to_numpy(), cells.y.to_numpy()
    for r in t[t.status == "train"].itertuples():
        out[(x >= r.x0) & (x < r.x1) & (y >= r.y0) & (y < r.y1)] = int(r.niche)
    return out


def niche_cells(ax, cells, t, niche_of, s=0.04, only=None):
    """Cells coloured by their tile's niche (``niche_of`` from cell_niches); others light grey.

    only: a niche index to colour, every other cell light grey."""
    pal = np.array(fs.niche_palette(n_niches(t)))
    show = niche_of >= 0 if only is None else niche_of == only
    ax.scatter(cells.x[~show], cells.y[~show], s=s, color=fs.LIGHT, lw=0, rasterized=True)
    ax.scatter(cells.x[show], cells.y[show], s=s, c=pal[niche_of[show]], lw=0, rasterized=True)
    tissue_axes(ax, t)


def niche_handles(n, niches=None):
    from matplotlib.patches import Patch
    pal = fs.niche_palette(n)
    niches = range(n) if niches is None else niches
    return [Patch(facecolor=pal[j], edgecolor="none", label=niche_label(j)) for j in niches]


def composition(fig, comp, x, y, pitch, col_w, labels, n, cbar_dy=8.0, vmax=None):
    """Niche x cell type heatmap at (x, y) mm; rows = labels' keys (cell types) in order."""
    types = list(labels)
    M = comp.pivot(index="cell_type", columns="niche", values="fraction").reindex(types).fillna(0)
    hw = col_w * M.shape[1]
    hax = ax_mm(fig, x, y, hw, pitch * len(types))
    vmax = vmax or M.to_numpy().max()
    im = hax.imshow(M.to_numpy(), cmap=fs.SEQUENTIAL, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    hax.set_yticks(range(len(types)))
    hax.set_yticklabels([labels[s] for s in types], fontsize=fs.TICK_PT)
    hax.set_xticks(range(M.shape[1]))
    hax.set_xticklabels([str(j + 1) for j in M.columns], fontsize=fs.TICK_PT)
    hax.xaxis.tick_top()
    hax.tick_params(axis="x", length=0, pad=5.5)
    hax.tick_params(axis="y", length=0, pad=1.5)
    for sp in hax.spines.values():
        sp.set_visible(False)
    pal = fs.niche_palette(n)
    for j in range(M.shape[1]):
        hax.add_patch(Rectangle((j - 0.5, -0.95), 1, 0.35, color=pal[j], clip_on=False, lw=0))
    hax.set_title("Niche", fontsize=fs.TICK_PT, pad=11)
    cax = ax_mm(fig, x + hw + 1.5, y + cbar_dy, 1.3, 16)
    cb = fig.colorbar(im, cax=cax)
    cb.locator = MaxNLocator(3)
    cb.update_ticks()
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1)
    cb.set_label("Fraction of niche's cells", fontsize=fs.TICK_PT, labelpad=2)
    return hax


def enrichment_heatmap(fig, comp, x, y, pitch, col_w, labels, n, lim=3.0, bar_w=9.0):
    """Niche x cell type log2(fraction in niche / fraction in section) at (x, y) mm, rows = labels' keys in
    order; to its right, each cell type's fraction of the section's cells (training tiles) as bars."""
    types = list(labels)
    F = comp.pivot(index="cell_type", columns="niche", values="fraction").reindex(types).fillna(0)
    N = comp.pivot(index="cell_type", columns="niche", values="n_cells").reindex(types).fillna(0)
    section = N.sum(1) / N.to_numpy().sum()
    E = np.log2((F.to_numpy() + 1e-3) / (section.to_numpy()[:, None] + 1e-3))
    hw = col_w * F.shape[1]
    hh = pitch * len(types)
    hax = ax_mm(fig, x, y, hw, hh)
    im = hax.imshow(np.clip(E, -lim, lim), cmap=fs.DIVERGING, vmin=-lim, vmax=lim, aspect="auto",
                    interpolation="nearest")
    hax.set_yticks(range(len(types)))
    hax.set_yticklabels([labels[s] for s in types], fontsize=fs.TICK_PT)
    hax.set_xticks(range(F.shape[1]))
    hax.set_xticklabels([str(j + 1) for j in F.columns], fontsize=fs.TICK_PT)
    hax.xaxis.tick_top()
    hax.tick_params(axis="x", length=0, pad=5.5)
    hax.tick_params(axis="y", length=0, pad=1.5)
    for sp in hax.spines.values():
        sp.set_visible(False)
    pal = fs.niche_palette(n)
    for j in range(F.shape[1]):
        hax.add_patch(Rectangle((j - 0.5, -0.95), 1, 0.35, color=pal[j], clip_on=False, lw=0))
    hax.set_title("Niche", fontsize=fs.TICK_PT, pad=11)

    bax = ax_mm(fig, x + hw + 1.5, y, bar_w, hh)
    bax.barh(range(len(types)), section.to_numpy(), color=fs.REFERENCE_COLOR, height=0.7, lw=0)
    bax.set_ylim(len(types) - 0.5, -0.5)
    bax.set_yticks([])
    bax.xaxis.tick_top()
    bax.xaxis.set_label_position("top")
    bax.set_xlabel("Section", fontsize=fs.TICK_PT, labelpad=2)
    bax.xaxis.set_major_locator(MaxNLocator(2))
    bax.tick_params(axis="x", length=1.5, pad=1, labelsize=fs.TICK_PT - 0.5)
    for side in ("right", "bottom"):
        bax.spines[side].set_visible(False)

    cax = ax_mm(fig, x, y + hh + 2.0, hw + 1.5 + bar_w, 1.4)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-lim, 0, lim])
    cb.ax.set_xticklabels([f"≤ −{lim:g}", "0", f"≥ {lim:g}"])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1, labelsize=fs.TICK_PT)
    cb.set_label("log$_2$ enrichment over section", fontsize=fs.TICK_PT, labelpad=1)
    return hax, bax, cax, E, section


# ---------------------------------------------------------------- block heatmap
def block_heatmap(ax, R, runs, lim=CORR_LIM, color=fs.INK, lw=0.5):
    """Correlation matrix (diagonal blank) with the blocks ``runs`` outlined along the diagonal."""
    A = np.asarray(R, dtype=float).copy()
    np.fill_diagonal(A, np.nan)
    cmap = fs.plt.get_cmap(fs.DIVERGING).copy()
    cmap.set_bad("white")
    im = ax.imshow(A, cmap=cmap, vmin=-lim, vmax=lim, interpolation="nearest", aspect="auto", rasterized=True)
    for s, e in runs:
        ax.add_patch(Rectangle((s - 0.5, s - 0.5), e - s, e - s, fill=False, ec=color, lw=lw))
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)
    n = A.shape[0]
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    return im


def hcolorbar(fig, im, cax, label, ticks=None):
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=ticks)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1, labelsize=fs.TICK_PT)
    if ticks is None:
        cb.locator = MaxNLocator(3)
        cb.update_ticks()
    cax.text(-0.04, 0.5, label, transform=cax.transAxes, ha="right", va="center", fontsize=fs.TICK_PT)
    return cb


def short_term(term, width=30):
    import re
    t = re.sub(r"\s*\(GO:\d+\)\s*$", "", str(term)).strip()
    return t if len(t) <= width else t[: width - 1].rstrip() + "…"


GENERIC_TERM = ("positive regulation of", "negative regulation of", "regulation of")


def block_term(row, enr):
    """A block's label term: its Hallmark term if enriched, else its GO BP term unless that is a generic
    'regulation of' term. -> (term, FDR, overlapping block genes) or None."""
    for lib, full in (("MSigDB", "MSigDB_Hallmark_2020"), ("GO", "GO_Biological_Process_2023")):
        if not getattr(row, f"{lib}_enriched"):
            continue
        term = getattr(row, f"{lib}_top_term")
        if lib == "GO" and short_term(term, 999).lower().startswith(GENERIC_TERM):
            continue
        e = enr[(enr.niche == row.niche) & (enr.block == row.block) & (enr.library == full) & (enr.term == term)]
        return short_term(term), float(getattr(row, f"{lib}_top_fdr")), e.genes.iloc[0].split()
    return None


def block_hubs(R, genes, n=3):
    """The n genes of a block with the largest summed |r| to the rest of the block."""
    sub = np.abs(R.loc[genes, genes].to_numpy())
    np.fill_diagonal(sub, 0)
    return [genes[i] for i in np.argsort(-sub.sum(1))[:n]]


# ---------------------------------------------------------------- signature map
def signature_map(ax, cells, score, t, boxes, clip=(1, 99), s=0.03):
    """Cells coloured by a signature score; ``boxes`` = [(rows, style dict)] outlined on top."""
    v = cells[score].to_numpy()
    lo, hi = np.nanpercentile(v, clip)
    sc = ax.scatter(cells.x, cells.y, c=v, s=s, cmap=fs.SEQUENTIAL, vmin=lo, vmax=hi, lw=0, rasterized=True)
    for rows, style in boxes:
        rects(ax, rows, facecolor="none", zorder=3, **style)
    tissue_axes(ax, t)
    return sc
