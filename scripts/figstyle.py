"""Shared figure style for every manuscript figure.

Import this before creating any figure:

    import figstyle as fs
    fig = fs.figure(fs.DOUBLE, 120)          # width, height in mm
    ...
    fs.label_panel(fig, ax, "A")
    fs.save(fig, "fig2_index_scaling")       # figures/{pdf,png}/<name>.* + figures/{pdf,png}/panels/<name>/<letter>.*

Rules the helpers enforce:
- 85 mm (SINGLE) or 180 mm (DOUBLE) wide; one figure + GridSpec, never pasted PNGs.
- Liberation Sans (metric-compatible with Arial); 7 pt text, 6 pt ticks/legends,
  8 pt bold panel letters. Bold is used for panel letters only.
- Fixed colours per dataset, method and platform, in every figure.
- No dual axes; no truncated bars (use dots); error bars are s.d. across seeds.
- Legends right of / below the data, never top left; despined axes, no grid
  except faint y-grid lines on log axes.
- PDFs keep text as TrueType (fonttype 42).

Panel export: every axes, figure-level text and patch is assigned to the nearest panel
letter above and left of it (``label_panel`` / ``label_at`` register the letters); pass
``panel=`` to either to override. Figures without letters export one file per axes, and
figure-level legends are exported as ``legend*``. Every file is written twice: a PDF under
figures/pdf/ and a PNG (PNG_DPI) under figures/png/, with the same relative path.
"""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
FIG_DATA = RESULTS / "figure_data"
FIG_DIR = PROJECT_ROOT / "figures"
PDF_DIR = FIG_DIR / "pdf"
PNG_DIR = FIG_DIR / "png"
PANEL_DIR = PDF_DIR / "panels"
PNG_PANEL_DIR = PNG_DIR / "panels"
PNG_DPI = 600

MM = 1 / 25.4
SINGLE = 85.0  # mm
DOUBLE = 180.0  # mm

TEXT_PT, TICK_PT, LETTER_PT = 7, 6, 8

# ---------------------------------------------------------------- font
_FONT_FILES = [
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf",
]
for _f in _FONT_FILES:
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
FONT = "Liberation Sans"
if FONT not in {f.name for f in font_manager.fontManager.ttflist}:
    raise RuntimeError("Liberation Sans not found: install it into ~/.fonts (see figstyle docstring)")

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
    "font.size": TEXT_PT,
    "axes.titlesize": TEXT_PT,
    "axes.titleweight": "normal",
    "axes.labelsize": TEXT_PT,
    "xtick.labelsize": TICK_PT,
    "ytick.labelsize": TICK_PT,
    "legend.fontsize": TICK_PT,
    "legend.title_fontsize": TICK_PT,
    "legend.frameon": False,
    "legend.handlelength": 1.2,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.2,
    "legend.labelspacing": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.grid": False,
    "axes.labelpad": 2,
    "axes.titlepad": 3,
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
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.5,
    "errorbar.capsize": 0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.dpi": 300,
    "figure.dpi": 150,
})

# ---------------------------------------------------------------- datasets
# Keys are the short names used by index_stats.csv; STEMS maps the
# xenium_human_* stems used by every benchmark CSV onto them.
DATASET_ORDER = ["skin_melanoma", "kidney_nondiseased", "breast_cancer", "lung_cancer",
                 "pancreatic_cancer", "lymph_node", "lymph_node_5k", "brain_cancer"]
DATASET_LABELS = {
    "skin_melanoma": "Skin",
    "kidney_nondiseased": "Kidney",
    "breast_cancer": "Breast",
    "lung_cancer": "Lung",
    "pancreatic_cancer": "Pancreas",
    "lymph_node": "Lymph node",
    "lymph_node_5k": "Lymph node (5k panel)",
    "brain_cancer": "Brain",
}
DATASET_SHORT = {**DATASET_LABELS, "lymph_node_5k": "Lymph node 5k"}
STEMS = {f"xenium_human_{k}": k for k in DATASET_ORDER}
# Okabe-Ito plus Tol's wine. Checked with Machado CVD simulation + OKLab:
# normal-vision min dE 15.6; the weakest CVD pair (skin/pancreas, deutan 7.6)
# is always backed by DATASET_MARKERS.
DATASET_COLORS = {
    "skin_melanoma": "#CC79A7",
    "kidney_nondiseased": "#E69F00",
    "breast_cancer": "#D55E00",
    "lung_cancer": "#56B4E9",
    "pancreatic_cancer": "#009E73",
    "lymph_node": "#0072B2",
    "lymph_node_5k": "#332288",
    "brain_cancer": "#882255",
}
DATASET_MARKERS = {
    "skin_melanoma": "o", "kidney_nondiseased": "s", "breast_cancer": "D", "lung_cancer": "^",
    "pancreatic_cancer": "v", "lymph_node": "P", "lymph_node_5k": "X", "brain_cancer": "h",
}

# ---------------------------------------------------------------- methods / platforms
METHOD_ORDER = ["spindle", "brute_force", "flat", "hnsw", "pca_hnsw", "phi_knn"]
METHOD_LABELS = {
    "spindle": "Spindle",
    "brute_force": "Brute force",
    "flat": "FAISS-Flat",
    "hnsw": "FAISS-HNSW",
    "pca_hnsw": "PCA-256 + HNSW",
    "phi_knn": r"$\varphi$-kNN",
}
METHOD_COLORS = {
    "spindle": "#1A1A1A",
    "brute_force": "#9E9E9E",
    "flat": "#DDAA33",
    "hnsw": "#AA3377",
    "pca_hnsw": "#4477AA",
    "phi_knn": "#228833",
}
METHOD_MARKERS = {"spindle": "o", "brute_force": "s", "flat": "D", "hnsw": "^", "pca_hnsw": "v", "phi_knn": "P"}

PLATFORM_COLORS = {"Xenium": "#0077BB", "Visium": "#EE7733"}

# Exact vs noisy / reference comparisons (e.g. the E7 exact-noisy ceiling).
REFERENCE_COLOR = "#9E9E9E"
LIGHT = "#D9D9D9"      # background tiles / individual queries
INK = "#1A1A1A"
MUTED = "#6E6E6E"

SEQUENTIAL = "viridis"
DIVERGING = "RdBu_r"

# Categorical niche palette (Tol muted, 9 hues) extended by Tol light for up to 18 niches.
# Order puts sand third so the first four niches span blue / light blue / yellow / green
# (OKLab min dE 17.8 normal vision, 16.0 tritan; the old indigo-cyan-teal-green start was 15.9 / 13.5).
_NICHE = ["#332288", "#88CCEE", "#DDCC77", "#117733", "#CC6677", "#882255", "#44AA99", "#999933",
          "#AA4499", "#77AADD", "#EE8866", "#EEDD88", "#FFAABB", "#99DDFF", "#BBCC33", "#AAAA00",
          "#DDDDDD", "#555555"]

# Query accent (Tol vibrant magenta): a query tile and what it retrieves, in schematic and
# example panels. dE >= 10.9 from the first four niche colours and grey under every CVD type.
QUERY_COLOR = "#EE3377"


def niche_palette(n):
    """n distinct niche colours in a fixed order (niche i always gets colour i)."""
    if n > len(_NICHE):
        raise ValueError(f"niche_palette supports up to {len(_NICHE)} niches, got {n}")
    return _NICHE[:n]


def niche_cmap(n):
    return ListedColormap(niche_palette(n))


# ---------------------------------------------------------------- figure helpers
def figure(width_mm, height_mm):
    if width_mm not in (SINGLE, DOUBLE):
        raise ValueError("figure width must be SINGLE (85 mm) or DOUBLE (180 mm)")
    return plt.figure(figsize=(width_mm * MM, height_mm * MM))


def _register_letter(fig, letter, text, members):
    if not hasattr(fig, "_fs_letters"):
        fig._fs_letters, fig._fs_members = {}, {}
    fig._fs_letters[letter] = text
    for a in members or ():
        fig._fs_members[a] = letter


def add_to_panel(fig, letter, *artists):
    """Force ``artists`` (axes, texts, legends) into panel ``letter`` for the per-panel export."""
    for a in artists:
        fig._fs_members[a] = letter


def label_panel(fig, ax, letter, dx_mm=-6.0, dy_mm=1.5, panel=()):
    """Bold panel letter at the top-left corner of ``ax``'s bounding box, in figure coordinates."""
    fig.canvas.draw_idle()
    bbox = ax.get_position()
    w, h = fig.get_size_inches()
    x = bbox.x0 + dx_mm * MM / w
    y = bbox.y1 + dy_mm * MM / h
    t = fig.text(x, y, letter, fontsize=LETTER_PT, fontweight="bold", ha="left", va="bottom")
    _register_letter(fig, letter, t, [ax, *panel])


def label_at(fig, x_mm, y_mm_from_top, letter, panel=()):
    """Panel letter at an absolute position (mm from the figure's left / top edges)."""
    w, h = fig.get_size_inches()
    t = fig.text(x_mm * MM / w, 1 - y_mm_from_top * MM / h, letter, fontsize=LETTER_PT,
                 fontweight="bold", ha="left", va="top")
    _register_letter(fig, letter, t, panel)


def log_ygrid(ax, axis="y"):
    """Faint grid lines on a log axis (the only grid the style allows)."""
    ax.grid(True, axis=axis, which="major", color="#E6E6E6", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def _plain(v, _pos=None):
    if v <= 0:
        return ""
    if v >= 1e4:
        e = np.log10(v)
        return f"$10^{{{e:.0f}}}$" if abs(e - round(e)) < 1e-9 else ""
    return f"{v:g}"


def plain_log(ax, axis="y", subs=(1.0, 2.0, 5.0)):
    """Log axis labelled with plain numbers at 1-2-5 steps (powers of ten above 10^4); no minor labels."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=subs, numticks=20))
    a.set_major_formatter(FuncFormatter(_plain))
    a.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
    a.set_minor_formatter(NullFormatter())


def despine(ax, left=False, bottom=False):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(not left)
    ax.spines["bottom"].set_visible(not bottom)
    if left:
        ax.tick_params(left=False)
    if bottom:
        ax.tick_params(bottom=False)


def blank(ax):
    ax.set_axis_off()


def identity_line(ax, lo, hi, **kw):
    """y = x reference; also makes the axes square with equal limits."""
    ax.plot([lo, hi], [lo, hi], color=kw.pop("color", MUTED), lw=kw.pop("lw", 0.6),
            ls=kw.pop("ls", "--"), zorder=kw.pop("zorder", 1), **kw)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def dataset_handles(keys=None, label=DATASET_SHORT, marker=True, **kw):
    from matplotlib.lines import Line2D
    keys = keys or DATASET_ORDER
    return [Line2D([], [], color=DATASET_COLORS[k], marker=DATASET_MARKERS[k] if marker else None,
                   ls=kw.get("ls", "none" if marker else "-"), ms=kw.get("ms", 3.5), lw=kw.get("lw", 1.0),
                   label=label[k]) for k in keys]


def method_handles(keys, **kw):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=METHOD_COLORS[k], marker=METHOD_MARKERS[k], ls="none",
                   ms=kw.get("ms", 3.5), label=METHOD_LABELS[k]) for k in keys]


def legend_right(ax, handles=None, **kw):
    kw.setdefault("loc", "center left")
    kw.setdefault("bbox_to_anchor", (1.02, 0.5))
    return ax.legend(handles=handles, **kw) if handles is not None else ax.legend(**kw)


def legend_below(obj, handles, ncol, y=-0.02, **kw):
    """Legend centred below an axes (``obj`` is an Axes) or the figure (``obj`` is a Figure)."""
    kw.setdefault("loc", "upper center")
    kw.setdefault("bbox_to_anchor", (0.5, y))
    kw.setdefault("columnspacing", 1.0)
    return obj.legend(handles=handles, ncol=ncol, **kw)


def fmt_x(v):
    """Speedup-style multiplier text, e.g. 25.9 -> '26×', 5.94 -> '5.9×'."""
    return f"{v:.0f}×" if v >= 10 else f"{v:.1f}×"


def _anchor(fig, a, r):
    """Top-left corner (display px) used to assign ``a`` to a panel."""
    if hasattr(a, "get_position") and hasattr(a, "get_subplotspec"):  # Axes: frame, not titles
        b = a.get_position().transformed(fig.transFigure)
    else:
        b = a.get_window_extent(r)
    return b.x0, b.y1


def _panel_groups(fig, r):
    """{panel name: [artists]} for every visible figure-level artist except figure legends."""
    letters = getattr(fig, "_fs_letters", {})
    forced = getattr(fig, "_fs_members", {})
    items = [a for a in [*fig.axes, *fig.texts, *fig.patches, *fig.lines, *fig.images]
             if a.get_visible() and a not in letters.values()]
    if letters:
        anchors = {k: (t.get_window_extent(r).x0, t.get_window_extent(r).y1) for k, t in letters.items()}
        groups = {k: [t] for k, t in letters.items()}
    else:  # no letters: one panel per top-level axes
        top = [ax for ax in fig.axes if ax.get_visible() and (ax.axison or ax.has_data())]
        anchors = {f"panel{i + 1:02d}": _anchor(fig, ax, r) for i, ax in enumerate(top)}
        groups = {k: [] for k in anchors}
    tol = 4 * MM * fig.dpi
    for a in items:
        if a in forced:
            groups[forced[a]].append(a)
            continue
        x, y = _anchor(fig, a, r)
        # rows first: a letter in the same row beats a closer one in the row above
        cands = [(3 * abs(ly - y) + (x - lx), k) for k, (lx, ly) in anchors.items()
                 if lx <= x + tol and ly >= y - tol]
        if not cands:  # nothing above-left: nearest anchor
            cands = [(abs(ly - y) + abs(x - lx), k) for k, (lx, ly) in anchors.items()]
        groups[min(cands)[1]].append(a)
    groups = {k: v for k, v in groups.items() if v}
    for i, leg in enumerate(fig.legends):
        groups["legend" if i == 0 else f"legend{i + 1}"] = [leg]
    return groups


def png_path(pdf):
    """figures/pdf/<...>.pdf -> figures/png/<...>.png (parent created)."""
    png = PNG_DIR / Path(pdf).relative_to(PDF_DIR).with_suffix(".png")
    png.parent.mkdir(parents=True, exist_ok=True)
    return png


def pdf_to_png(pdf):
    """Rasterise a one-page PDF (e.g. one assembled with pypdf) to its PNG twin via pdftoppm."""
    import subprocess
    png = png_path(pdf)
    subprocess.run(["pdftoppm", "-png", "-r", str(PNG_DPI), "-singlefile", str(pdf), str(png.with_suffix(""))],
                   check=True)
    return png


def _save_panels(fig, name):
    """Write figures/{pdf,png}/panels/<name>/<panel>.*: each panel alone, cropped to its extent."""
    from matplotlib.transforms import Bbox
    out, out_png = PANEL_DIR / name, PNG_PANEL_DIR / name
    for d, ext in ((out, "pdf"), (out_png, "png")):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob(f"*.{ext}"):
            old.unlink()
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    groups = _panel_groups(fig, r)
    everything = {a for v in groups.values() for a in v}
    pad = 1.0 * MM * fig.dpi
    W, H = fig.bbox.width, fig.bbox.height
    for key, members in groups.items():
        boxes = [a.get_tightbbox(r) if hasattr(a, "get_tightbbox") else a.get_window_extent(r) for a in members]
        b = Bbox.union([bb for bb in boxes if bb is not None and bb.width > 0 and bb.height > 0])
        b = Bbox.from_extents(max(b.x0 - pad, 0), max(b.y0 - pad, 0), min(b.x1 + pad, W), min(b.y1 + pad, H))
        hidden = [a for a in everything if a not in members and a.get_visible()]
        for a in hidden:
            a.set_visible(False)
        bbox = b.transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(out / f"{key}.pdf", dpi=600, bbox_inches=bbox)
        fig.savefig(out_png / f"{key}.png", dpi=PNG_DPI, bbox_inches=bbox)
        for a in hidden:
            a.set_visible(True)
    return len(groups)


def save(fig, name):
    """Write figures/pdf/<name>.pdf (vector, exact printed size), its PNG twin and one file per panel."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf = PDF_DIR / f"{name}.pdf"
    fig.savefig(pdf, dpi=600)  # 600 dpi applies to rasterized layers only
    fig.savefig(png_path(pdf), dpi=PNG_DPI)
    n = _save_panels(fig, name)
    plt.close(fig)
    print(f"saved {pdf.relative_to(PROJECT_ROOT)} (+ .png) + {n} panels in {(PANEL_DIR / name).relative_to(PROJECT_ROOT)}/ "
          f"and {(PNG_PANEL_DIR / name).relative_to(PROJECT_ROOT)}/")


def stem_to_key(s):
    """Benchmark 'Dataset'/'dataset' values (xenium_human_*[_seedN]) -> DATASET_ORDER key."""
    s = str(s)
    for stem, key in sorted(STEMS.items(), key=lambda kv: -len(kv[0])):
        if s == stem or s.startswith(stem + "_seed"):
            return key
    if s in DATASET_LABELS:
        return s
    raise KeyError(s)
