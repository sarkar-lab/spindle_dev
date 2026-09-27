"""Shared panel template for the biology figures (Figs 6, 7, S11).

CSV inputs only: results/biology/<dataset>/ (scripts/bio_modules.py) and
results/figure_data/tiles_seed73.csv. The four template panels:

  (i)   niche_map        tiles coloured by covariance niche on the tissue
  (ii)  module_heatmap   centroid correlation, genes ordered by module, module bar on top,
                         italic gene labels, RdBu_r on [-1, 1]
  (iii) enrichment       lollipop of -log10 FDR for the highlighted module's top terms
                         (panel background); the overlap k/n is printed, not size-encoded
  (iv)  module_map       each tile's module co-variation: mean within-module correlation of the
                         highlighted module's genes (dark = strong). The log-Euclidean distance to
                         the layer's centroid (d_module in tile_scores.csv) was tried first; on these
                         near-singular sub-blocks it ranks strongly co-varying tiles as far (skin),
                         so the map shows the heatmap's own statistic instead.

Niches, layers and modules are printed 1-based (as in Fig. 1); the CSVs are 0-based.
Axes are placed in millimetres from the figure's top-left corner (ax_mm).
"""

import re
import textwrap

import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle

import figstyle as fs

BIO = fs.RESULTS / "biology"
MODULE_GREYS = ["#BDBDBD", "#8C8C8C"]
HIGHLIGHT = fs.INK
MAP_CMAP = "magma_r"         # dark = strong module co-variation
FDR_LINE = 0.05
MIN_OVERLAP = 2              # terms hit by a single module gene are not shown
LIB_LABEL = {"MSigDB_Hallmark_2020": "MSigDB Hallmark", "GO_Biological_Process_2023": "GO BP"}
_GO_RE = re.compile(r"\s*\(GO:\d+\)\s*$")
_tiles = None


# ---------------------------------------------------------------- layout / data
def ax_mm(fig, x, y, w, h, **kw):
    """Axes at x, y (mm from the left / top edges), w x h mm."""
    W, H = fig.get_size_inches() / fs.MM
    return fig.add_axes([x / W, 1 - (y + h) / H, w / W, h / H], **kw)


def load(key):
    d = {name: pd.read_csv(BIO / key / f"{name}.csv")
         for name in ("selection", "modules", "enrichment", "tile_scores")}
    d["corr"] = pd.read_csv(BIO / key / "corr.csv", index_col=0)
    d["sel"] = d["selection"].iloc[0]
    d["tiles"] = all_tiles().query("dataset == @key")
    d["key"] = key
    return d


def all_tiles():
    global _tiles
    if _tiles is None:
        _tiles = pd.read_csv(fs.FIG_DATA / "tiles_seed73.csv")
    return _tiles


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


def n_niches(d):
    return int(d["sel"]["n_niches"])


def niche_label(j):
    return f"Niche {j + 1}"


def node_label(d):
    s = d["sel"]
    return f"niche {s.niche + 1}, layer {s.layer + 1}"


def module_order(d):
    m = d["modules"].query("kept").sort_values("plot_order")
    mods = list(dict.fromkeys(m.module))  # greedy-modularity order (largest first)
    return m, {mod: i + 1 for i, mod in enumerate(mods)}


def highlight_label(d):
    _, num = module_order(d)
    return f"M{num[int(d['sel']['highlight_module'])]}"


# ---------------------------------------------------------------- (i) niche map
def niche_map(ax, d, only=None, lw=0.1):
    """Training tiles in their niche colour; held-out / dropped tiles light grey.

    only: a niche index to show in colour with every other tile light grey."""
    t = d["tiles"]
    pal = fs.niche_palette(n_niches(d))
    grey = t[(t.status != "train") | ((t.niche != only) if only is not None else False)]
    rects(ax, grey, facecolor=fs.LIGHT, edgecolor="white", lw=lw)
    tr = t[(t.status == "train") & ((t.niche == only) if only is not None else True)]
    rects(ax, tr, facecolor=[pal[int(n)] for n in tr.niche], edgecolor="white", lw=lw)
    tissue_axes(ax, t)


def niche_handles(d, niches=None):
    from matplotlib.patches import Patch
    pal = fs.niche_palette(n_niches(d))
    niches = range(n_niches(d)) if niches is None else niches
    return [Patch(facecolor=pal[j], edgecolor="none", label=niche_label(j)) for j in niches]


# ---------------------------------------------------------------- (ii) module heatmap
LABEL_GAP_MM = 2.3   # 6 pt text needs ~2.1 mm per line


def _spread(ys, gap, lo, hi):
    """Label positions near ys (sorted), at least gap apart, kept inside [lo, hi] when possible."""
    pos = list(map(float, ys))
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap)
    over = pos[-1] - hi if pos else 0
    if over > 0:
        pos[-1] -= over
        for k in range(len(pos) - 2, -1, -1):
            pos[k] = min(pos[k], pos[k + 1] - gap)
    if pos and pos[0] < lo:  # does not fit: centre the column on the rows
        mid = (ys[0] + ys[-1]) / 2
        pos = [mid + (k - (len(pos) - 1) / 2) * gap for k in range(len(pos))]
    return pos


def module_heatmap(fig, d, x, y, side, label_pt=fs.TICK_PT, bar_h=1.6, cbar=True, cbar_w=None):
    """Heatmap at (x, y) mm, side x side mm, with the module bar above it.

    Gene labels (italic): every gene on the left if a row is >= 2.2 mm, alternating
    left / right if >= 1.1 mm, otherwise only the highlighted module's genes (alternating)."""
    m, num = module_order(d)
    genes = m.gene.tolist()
    R = d["corr"].loc[genes, genes].to_numpy().copy()
    np.fill_diagonal(R, np.nan)
    n = len(genes)
    pitch = side / n
    ax = ax_mm(fig, x, y, side, side)
    cmap = fs.plt.get_cmap(fs.DIVERGING).copy()
    cmap.set_bad(fs.LIGHT)
    im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)

    hi = int(d["sel"]["highlight_module"])
    bounds, start = [], 0
    for mod, grp in m.groupby("module", sort=False):
        bounds.append((mod, start, start + len(grp)))
        start += len(grp)
    for _, a, b in bounds[1:]:
        ax.axhline(a - 0.5, color=fs.INK, lw=0.5)
        ax.axvline(a - 0.5, color=fs.INK, lw=0.5)
    for mod, a, b in bounds:
        if mod == hi:
            ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec=HIGHLIGHT, lw=1.2, zorder=3))

    # module bar
    bar = ax_mm(fig, x, y - bar_h - 0.4, side, bar_h)
    for i, (mod, a, b) in enumerate(bounds):
        col = HIGHLIGHT if mod == hi else MODULE_GREYS[i % 2]
        bar.add_patch(Rectangle((a - 0.5, 0), b - a, 1, color=col, lw=0))
        if (b - a) * pitch >= 3.5:
            bar.text((a + b - 1) / 2, 1.25, f"M{num[mod]}", ha="center", va="bottom", fontsize=fs.TICK_PT,
                     fontweight="normal", color=fs.INK)
    bar.set_xlim(-0.5, n - 0.5)
    bar.set_ylim(0, 1)
    bar.set_axis_off()

    # gene labels
    if pitch >= 2.2:
        rows = [(i, "left") for i in range(n)]
    elif pitch >= 1.1:
        rows = [(i, "left" if i % 2 == 0 else "right") for i in range(n)]
    else:
        idx = [i for i, g in enumerate(genes) if m.iloc[i].module == hi]
        rows = [(i, "left" if k % 2 == 0 else "right") for k, i in enumerate(idx)]
    spread = pitch < 1.1
    gap = LABEL_GAP_MM / pitch             # minimum label spacing, in rows
    lead = 2.8 / pitch if spread else 0.0  # leader-line length, in rows
    for where in ("left", "right"):
        sel = [i for i, w in rows if w == where]
        ypos = _spread(sel, gap, -0.5, n - 0.5) if spread else sel
        for i, yl in zip(sel, ypos):
            style = dict(fontsize=label_pt, fontstyle="italic", va="center", clip_on=False,
                         color=fs.INK if m.iloc[i].module == hi else fs.MUTED)
            if where == "left":
                x_edge, x_text, ha = -0.5, -0.9 - lead, "right"
            else:
                x_edge, x_text, ha = n - 0.5, n - 0.1 + lead, "left"
            if spread:
                x_knee = x_edge + (x_text - x_edge) * 0.75
                ax.plot([x_edge, x_knee, x_text], [i, yl, yl], color=fs.MUTED, lw=0.3, clip_on=False)
            ax.text(x_text, yl, genes[i], ha=ha, **style)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)

    cax = None
    if cbar:
        cw = cbar_w or min(side * 0.45, 22)
        cax = ax_mm(fig, x + side - cw, y + side + 2.2, cw, 1.4)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-1, 0, 1])
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(length=1.5, pad=1)
        cax.text(-0.04, 0.5, "Correlation", transform=cax.transAxes, ha="right", va="center", fontsize=fs.TICK_PT)
    return ax, bar, cax


# ---------------------------------------------------------------- (iii) enrichment
def clean_term(term, width=34):
    return "\n".join(textwrap.wrap(_GO_RE.sub("", str(term)).strip(), width=width))


def top_terms(d, library=None, n=7, background="panel", module=None):
    s = d["sel"]
    library = library or s.library
    module = int(s.highlight_module) if module is None else module
    e = d["enrichment"]
    e = e[(e.module == module) & (e.library == library) & (e.background == background) & (e.overlap >= MIN_OVERLAP)]
    return e.sort_values(["p", "overlap"], ascending=[True, False]).head(n).reset_index(drop=True)


def enrichment(ax, d, library=None, n=7, width=34, color=fs.INK):
    """Lollipop of -log10 FDR (panel background, BH) for the highlighted module's top terms."""
    t = top_terms(d, library, n)
    y = np.arange(len(t))[::-1]
    v = -np.log10(t.fdr.to_numpy())
    ax.hlines(y, 0, v, color=color, lw=0.8)
    ax.plot(v, y, "o", color=color, ms=3.2, zorder=3)
    xmax = max(v.max(), -np.log10(FDR_LINE)) * 1.35
    ax.axvline(-np.log10(FDR_LINE), color=fs.MUTED, lw=0.6, ls="--", zorder=0)
    for yi, vi, r in zip(y, v, t.itertuples()):
        ax.text(vi + 0.03 * xmax, yi, f"{r.overlap}/{r.set_size}", va="center", ha="left", fontsize=fs.TICK_PT,
                color=fs.MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels([clean_term(x, width) for x in t.term], fontsize=fs.TICK_PT, linespacing=0.95)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.6, len(t) - 0.4)
    ax.set_xlabel(r"$-\log_{10}$ FDR")
    ax.spines["left"].set_visible(False)
    return t


# ---------------------------------------------------------------- (iv) module co-variation map
def module_map(ax, d, col="coherence", q=(2, 98)):
    """Training tiles coloured by module coherence (dark = strong); other tiles light grey."""
    t, s = d["tiles"], d["tile_scores"]
    rects(ax, t[t.status != "train"], facecolor=fs.LIGHT, edgecolor="none")
    v = s[col].to_numpy()
    lo, hi = np.nanpercentile(v, q)
    pc = rects(ax, s, cmap=MAP_CMAP, norm=Normalize(lo, hi), edgecolor="none")
    pc.set_array(v)
    tissue_axes(ax, t)
    return pc, (lo, hi)


def map_colorbar(fig, pc, cax, label="Mean within-module r", orientation="horizontal"):
    cb = fig.colorbar(pc, cax=cax, orientation=orientation)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1, labelsize=fs.TICK_PT)
    from matplotlib.ticker import MaxNLocator
    cb.locator = MaxNLocator(3)
    cb.update_ticks()
    if orientation == "horizontal":
        cb.set_label(label, fontsize=fs.TICK_PT, labelpad=1)
    else:
        cb.set_label(label, fontsize=fs.TICK_PT, labelpad=2)
    return cb


# ---------------------------------------------------------------- numbers for captions
def caption_numbers(d, library=None):
    s = d["sel"]
    library = library or s.library
    panel = top_terms(d, library, 7)
    e = d["enrichment"]
    g = e[(e.module == int(s.highlight_module)) & (e.library == library)
          & (e.background == "genome_enrichr")].set_index("term")
    rows = []
    for r in panel.itertuples():
        rows.append({"term": _GO_RE.sub("", r.term), "overlap": f"{r.overlap}/{r.set_size}", "p_panel": r.p,
                     "fdr_panel": r.fdr, "fdr_genome": g.fdr.get(r.term, np.nan),
                     "overlap_genome": (f"{int(g.overlap[r.term])}/{int(g.set_size[r.term])}"
                                        if r.term in g.index else "")})
    return pd.DataFrame(rows)
