"""Paper style for the figure notebooks (experiments/figures/F*.ipynb): fonts, colours, panels and pages in mm.

    import style as ps
    fig = ps.panel(60, 50); ax = ps.axes(fig, left=9, bottom=7, right=1, top=1)   # mm: panel size, then axes margins
    ...
    ps.save_panel(fig, "fig2_cost", "B")                  # figures/fig2_cost/panels/B.pdf + .png
    pg = ps.page(180, 200); panel_b(ps.place(pg, "B", x=120, y=0, w=60, h=50)); ps.save_figure(pg, "fig2_cost")

Rules:
- Saved at exactly its size (never bbox_inches="tight"). PDF vector, TrueType (fonttype 42); PNG 1200 dpi; artists
  made with ``rasterized=True`` (cell clouds) are rasterised at 1200 dpi. Resolution rule (user, 26 Oct): nothing may
  break when zoomed in, within 10 MB per file.
- Liberation Sans; axis labels 7 pt, ticks and legends 6 pt, panel letters 8 pt bold, lines 0.75 pt.
- Plotting only: compute nodes lack the font, so never import this module in a result notebook or sbatch job.
"""

import sys
from pathlib import Path

import matplotlib

if "ipykernel" not in sys.modules:  # notebooks keep their inline backend
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"
DPI = 1200
MM = 1 / 25.4
PAGE_W, PAGE_H = 180.0, 225.0  # mm

LABEL_PT, TICK_PT, LEGEND_PT = 7, 6, 6
LW = 0.75

# ---------------------------------------------------------------- font
for _f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSans-*.ttf"):
    font_manager.fontManager.addfont(str(_f))
FONT = "Liberation Sans"
if FONT not in {f.name for f in font_manager.fontManager.ttflist}:
    raise RuntimeError("Liberation Sans not found (plot on the login node, never in sbatch)")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": [FONT],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.sf": FONT,
    "mathtext.cal": FONT,
    "mathtext.tt": FONT,
    "font.size": LABEL_PT,
    "axes.labelsize": LABEL_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": LEGEND_PT,
    "legend.title_fontsize": LEGEND_PT,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.0,
    "legend.borderpad": 0.0,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.labelpad": 2,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.pad": 1.5,
    "ytick.major.pad": 1.5,
    "lines.linewidth": LW,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.5,
    "errorbar.capsize": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": DPI,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------- colours (poster/style.py)
DATASET_ORDER = ["skin_melanoma", "kidney_nondiseased", "breast_cancer", "lung_cancer",
                 "pancreatic_cancer", "lymph_node", "lymph_node_5k", "brain_cancer"]
DATASET_LABELS = {
    "skin_melanoma": "Skin", "kidney_nondiseased": "Kidney", "breast_cancer": "Breast", "lung_cancer": "Lung",
    "pancreatic_cancer": "Pancreas", "lymph_node": "Lymph node", "lymph_node_5k": "Lymph node 5k",
    "brain_cancer": "Brain",
}
# Okabe-Ito plus Tol's wine; markers back up the weakest colour-vision pair (skin / pancreas).
DATASET_COLORS = {
    "skin_melanoma": "#CC79A7", "kidney_nondiseased": "#E69F00", "breast_cancer": "#D55E00",
    "lung_cancer": "#56B4E9", "pancreatic_cancer": "#009E73", "lymph_node": "#0072B2",
    "lymph_node_5k": "#332288", "brain_cancer": "#882255",
}
DATASET_MARKERS = {
    "skin_melanoma": "o", "kidney_nondiseased": "s", "breast_cancer": "D", "lung_cancer": "^",
    "pancreatic_cancer": "v", "lymph_node": "P", "lymph_node_5k": "X", "brain_cancer": "h",
}

# What is stored / searched: whole covariances (grey), Spindle-naive (block logs), Spindle, Spindle on gene sets
# (Tol high-contrast blue / red / yellow). Spindle-naive on gene sets shares Spindle-naive's colour.
TIER_ORDER = ["whole", "naive", "spindle", "spindle_partial"]
TIER_LABELS = {"whole": "Whole covariances", "naive": "Spindle-naive", "spindle": "Spindle",
               "spindle_partial": "Spindle (gene sets)"}
TIER_COLORS = {"whole": "#9E9E9E", "naive": "#004488", "spindle": "#BB5566", "spindle_partial": "#DDAA33"}
TIER_MARKERS = {"whole": "s", "naive": "o", "spindle": "D", "spindle_partial": "^"}

LIGHT = "#D9D9D9"  # background tiles / all-tile distributions
INK = "#1A1A1A"
MUTED = "#6E6E6E"
QUERY_COLOR = "#EE3377"  # Tol vibrant magenta: the query tile and what links to it
DIVERGING = "RdBu_r"

# Categorical niche palette (Tol muted, then Tol light).
_NICHE = ["#332288", "#88CCEE", "#DDCC77", "#117733", "#CC6677", "#882255", "#44AA99", "#999933",
          "#AA4499", "#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#BBCC33", "#AAAA00",
          "#DDDDDD", "#555555"]


def niche_palette(n):
    """n distinct niche colours in a fixed order (niche i always gets colour i)."""
    if n > len(_NICHE):
        raise ValueError(f"niche_palette supports up to {len(_NICHE)} niches, got {n}")
    return _NICHE[:n]


def plain_log(ax, axis="y", subs=(1.0, 2.0, 5.0)):
    """Log axis labelled with plain numbers at 1-2-5 steps; no minor labels."""
    import numpy as np
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=subs, numticks=20))
    a.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:g}" if v > 0 else ""))
    a.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    a.set_minor_formatter(NullFormatter())


def log_grid(ax, axis="y"):
    """Faint major grid on a log axis (the only grid the style allows)."""
    ax.grid(True, axis=axis, which="major", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------- panels (figure notebooks)
# A panel is a function that draws into a Figure or SubFigure of its size in mm; one cell draws it alone and saves it
# (save_panel), the last cell places every panel on the page (page + place) and saves the figure (save_figure).
#     def panel_b(fig):
#         ax = ps.axes(fig, left=9, bottom=7, right=1, top=1)       # mm margins inside the panel
#         ...
#     fig = ps.panel(60, 50); panel_b(fig); ps.save_panel(fig, FIG, "B")
#     page = ps.page(180, 200); panel_b(ps.place(page, "B", x=120, y=0, w=60, h=50)); ps.save_figure(page, FIG)
LETTER_PT = 8


def panel(w, h):
    """A w x h mm figure for one panel."""
    if w > PAGE_W or h > PAGE_H:
        raise ValueError(f"panel {w} x {h} mm exceeds the {PAGE_W:g} x {PAGE_H:g} mm page")
    return plt.figure(figsize=(w * MM, h * MM))


def size_mm(fig):
    """(w, h) in mm of a Figure or SubFigure."""
    root = fig.figure if hasattr(fig, "figure") and fig.figure is not None else fig
    return fig.bbox.width / root.dpi / MM, fig.bbox.height / root.dpi / MM


def axes(fig, left=0.0, bottom=0.0, right=0.0, top=0.0, **kw):
    """One axes inset by the given margins (mm) in a panel (Figure or SubFigure)."""
    w, h = size_mm(fig)
    return fig.add_axes([left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h], **kw)


def page(w=PAGE_W, h=PAGE_H):
    return plt.figure(figsize=(w * MM, h * MM))


def place(pg, letter, x, y, w, h, dx=0.0, dy=0.0):
    """The w x h mm SubFigure at (x, y) mm from the page's top-left corner, with its panel letter (bold, 8 pt) at its
    top-left corner, offset by (dx, dy) mm. letter=None: no letter."""
    W, H = size_mm(pg)
    if x < 0 or y < 0 or x + w > W + 1e-6 or y + h > H + 1e-6:
        raise ValueError(f"panel {letter}: {w} x {h} mm at ({x}, {y}) leaves the {W:.0f} x {H:.0f} mm page")
    # subfigures follow only the gridspec's width/height ratios (not its left/right/...): a 3 x 3 grid, middle cell
    eps = 1e-6
    gs = pg.add_gridspec(3, 3, width_ratios=[max(x, eps), w, max(W - x - w, eps)],
                         height_ratios=[max(y, eps), h, max(H - y - h, eps)], wspace=0, hspace=0,
                         left=0, right=1, bottom=0, top=1)
    sf = pg.add_subfigure(gs[1, 1])
    if letter:
        pg.text((x + dx) / W, 1 - (y + dy) / H, letter, fontsize=LETTER_PT, fontweight="bold", ha="left", va="top")
    return sf


def _save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), dpi=DPI, facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=DPI, facecolor="white")
    sizes = [stem.with_suffix(e).stat().st_size / 1e6 for e in (".pdf", ".png")]
    w, h = size_mm(fig)
    print(f"{stem.relative_to(ROOT)}.pdf/.png: {w:.0f} x {h:.0f} mm, {sizes[0]:.1f} / {sizes[1]:.1f} MB")
    if max(sizes) > 10:
        print("  WARNING: over the 10 MB budget")


def save_panel(fig, fig_name, name):
    """figures/<fig>/panels/<name>.pdf + .png (1200 dpi), at the panel's own size (never bbox_inches="tight")."""
    _save(fig, FIGURES / fig_name / "panels" / name)


def save_figure(fig, fig_name):
    """figures/<fig>/<fig>.pdf + .png (1200 dpi)."""
    _save(fig, FIGURES / fig_name / fig_name)
