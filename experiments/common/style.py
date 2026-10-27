"""Paper style: every atom is one plot, drawn at its placed size in mm on the 180 mm page.

    import style as ps
    fig, ax = ps.atom(55, 40, left=9, bottom=7, right=1, top=1)       # mm: figure size, then axes margins
    ...
    ps.save(fig, "fig2_cost", "storage", source="results/fig2_cost/storage.csv")

Rules:
- No titles, captions, panel letters, annotations or legends inside a plot: axes, axis labels (with "log scale" when
  log), ticks and colourbar ticks only. A legend is its own atom (``legend_atom``), so is a colourbar
  (``colorbar_atom``).
- Saved at exactly its size (never bbox_inches="tight"), so the pptx never rescales it.
- PDF: vector, TrueType (fonttype 42), transparent background (render.py stamps it onto the page, so text boxes
  under it stay visible); artists made with ``rasterized=True`` (cell clouds) are rasterised at 1200 dpi.
  PNG: 1200 dpi, white (the pptx picture). Written to figures/<fig>/atoms/{pdf,png}/<atom>.*, one row per atom in
  figures/manifest.csv. Resolution rule (user, 26 Oct): nothing may break when zoomed in, within the size budget.
- Liberation Sans; axis labels 7 pt, ticks and legends 6 pt, lines 0.75 pt.
- Plotting only: compute nodes lack the font, so never import this module in an sbatch job.
"""

import csv
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"
MANIFEST = FIGURES / "manifest.csv"
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

# What is stored / searched: whole covariances (grey), Spindle (block logs), Spindle-lite, Spindle-lite-partial
# (Tol high-contrast blue / red / yellow). Spindle-partial shares Spindle's colour.
TIER_ORDER = ["whole", "spindle", "lite", "plite"]
TIER_LABELS = {"whole": "Whole covariances", "spindle": "Spindle", "lite": "Spindle-lite",
               "plite": "Spindle-lite-partial"}
TIER_COLORS = {"whole": "#9E9E9E", "spindle": "#004488", "lite": "#BB5566", "plite": "#DDAA33"}
TIER_MARKERS = {"whole": "s", "spindle": "o", "lite": "D", "plite": "^"}

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


# ---------------------------------------------------------------- atoms
def atom(w, h, left=0.0, bottom=0.0, right=0.0, top=0.0, **kw):
    """A w x h mm figure with one axes inset by the given margins (mm)."""
    if w > PAGE_W or h > PAGE_H:
        raise ValueError(f"atom {w} x {h} mm exceeds the {PAGE_W:g} x {PAGE_H:g} mm page")
    fig = plt.figure(figsize=(w * MM, h * MM))
    ax = fig.add_axes([left / w, bottom / h, 1 - (left + right) / w, 1 - (bottom + top) / h], **kw)
    return fig, ax


def legend_atom(fig_name, name, handles, w, h, ncol=1, source="", **kw):
    """A legend on its own, centred on a w x h mm canvas."""
    fig = plt.figure(figsize=(w * MM, h * MM))
    kw.setdefault("loc", "center")
    fig.legend(handles=handles, ncol=ncol, **kw)
    save(fig, fig_name, name, source=source)


def colorbar_atom(fig_name, name, cmap, vmin, vmax, w, h, ticks=None, label=None, orientation="vertical", source=""):
    """A colourbar on its own (bar 2.5 mm thick); ticks only (a label, if any, is an axis label)."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    if orientation == "vertical":
        fig, ax = atom(w, h, left=0.5, right=w - 3.0, bottom=1.5, top=1.5)
    else:
        fig, ax = atom(w, h, left=2.0, right=2.0, bottom=h - 3.0, top=0.5)
    cb = fig.colorbar(ScalarMappable(Normalize(vmin, vmax), cmap), cax=ax, orientation=orientation, ticks=ticks)
    cb.outline.set_linewidth(0.5)
    if label:
        cb.set_label(label)
    save(fig, fig_name, name, source=source)


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


def describe():
    return subprocess.run(["git", "describe", "--tags", "--dirty", "--always"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def atom_paths(fig_name, name):
    d = FIGURES / fig_name / "atoms"
    return d / "pdf" / f"{name}.pdf", d / "png" / f"{name}.png"


FIELDS = ["figure", "atom", "w_mm", "h_mm", "px", "source", "tag"]


def save(fig, fig_name, name, source=""):
    """figures/<fig>/atoms/{pdf,png}/<name>.* at the figure's own size; one manifest row per (figure, atom)."""
    pdf, png = atom_paths(fig_name, name)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    w, h = fig.get_size_inches() / MM
    fig.savefig(pdf, dpi=DPI, transparent=True)
    fig.savefig(png, dpi=DPI)
    plt.close(fig)
    rows = {}
    if MANIFEST.exists():
        with open(MANIFEST) as fh:
            rows = {(r["figure"], r["atom"]): r for r in csv.DictReader(fh)}
    rows[fig_name, name] = {"figure": fig_name, "atom": name, "w_mm": f"{w:.2f}", "h_mm": f"{h:.2f}",
                            "px": f"{round(w * MM * DPI)}x{round(h * MM * DPI)}", "source": source, "tag": describe()}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows[k] for k in sorted(rows))
    print(f"{fig_name}/{name}: {w:.1f} x {h:.1f} mm")


def manifest():
    """{(figure, atom): (w_mm, h_mm)}."""
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST) as fh:
        return {(r["figure"], r["atom"]): (float(r["w_mm"]), float(r["h_mm"])) for r in csv.DictReader(fh)}
