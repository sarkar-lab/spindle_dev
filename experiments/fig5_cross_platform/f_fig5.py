"""Fig 5 atoms from the small inputs of c_summary.py and c_extras.py (login node).

Usage: python experiments/fig5_cross_platform/f_fig5.py
Atoms (figures/fig5_cross_platform/atoms/):
  map_xenium, map_visium  the two sections, every Xenium cell / Visium spot (55 um) coloured by coarse class
                          (rasterised), the tile boxes in grey; the example query box (Visium, magenta) and its top 5
                          (Xenium, black) outlined
  legend_class            the coarse classes
  pca_raw, pca_corr       PCA of the shrunk whole-matrix logs, fit on the Xenium tiles (index of v2x); Xenium tiles
                          and Visium tiles before / after the global shift; same axes
  legend_platform         Xenium / Visium
  tissue                  seed 0, Spindle with the global shift: each query's tumour fraction against the mean of its
                          top 10 on the other platform, both directions; diagonal dashed
  ablation                tissue agreement r per correction (none, per niche, global) and direction, mean +/- s.d.
                          over seeds 0-4
  legend_direction        Xenium -> Visium / Visium -> Xenium (tissue, ablation)
  ex_query, ex_hit1..5    the example (rule in c_extras.py): the Visium query tile's spots and its top 5 Xenium
                          tiles' cells (only those in the box), each box outlined (query magenta, hits black, the
                          co-located hit dashed)
Point clouds are drawn without antialiasing (flat class colours), which keeps the 1200 dpi PNGs within the size budget.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import style as ps  # noqa: E402

FIG = "fig5_cross_platform"
RES = ps.ROOT / "results" / FIG
CLASS_COLORS = {"Invasive tumour": "#CC6677", "DCIS": "#882255", "Myoepithelial": "#DDCC77", "Stroma": "#44AA99",
                "Immune": "#332288", "Adipocytes": "#E69F00", "Mixed": "#AAAAAA"}
PLATFORM = {"Xenium": "#0077BB", "Visium": "#EE7733"}
DIRECTION = {"x2v": PLATFORM["Xenium"], "v2x": PLATFORM["Visium"]}  # coloured by the query platform
DIRECTION_LABEL = {"x2v": "Xenium → Visium", "v2x": "Visium → Xenium"}
SPOT_UM = 55.0
MAP_W = 42.0

cells = np.load(RES / "cells.npz")
classes = list(cells["classes"])
tiles = pd.read_csv(RES / "tiles.csv")
example = pd.read_csv(RES / "example.csv")
X0, Y0 = tiles[["x0", "y0"]].min()
X1, Y1 = tiles[["x1", "y1"]].max()
MAP_H = MAP_W * (Y1 - Y0) / (X1 - X0)


def points(platform, sel=None):
    xy, cl = cells[f"{platform}_xy"], cells[f"{platform}_class"]
    if sel is not None:
        xy, cl = xy[sel], cl[sel]
    return xy, np.array([CLASS_COLORS[classes[c]] for c in cl])


def spot_size(ax, fig, w_um):
    """Marker area (pt^2) of a SPOT_UM-wide spot on an axes w_um wide."""
    w_pt = ax.get_position().width * fig.get_size_inches()[0] * 72
    return (SPOT_UM / w_um * w_pt) ** 2


def box(ax, r, color, ls="-", lw=0.75):
    ax.add_patch(Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0, fill=False, ec=color, lw=lw, ls=ls, zorder=5))


def spatial(ax, x0, x1, y0, y1):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # image orientation
    ax.set_aspect("equal")
    ax.axis("off")


# ---------------------------------------------------------------- A
def maps():
    q = int(example["query"].iloc[0])
    vis = tiles[tiles.platform == "Visium"].set_index("tile")
    xen = tiles[tiles.platform == "Xenium"].set_index("tile")
    for platform in ("Xenium", "Visium"):
        fig, ax = ps.atom(MAP_W, MAP_H)
        xy, col = points(platform)
        s = 0.1 if platform == "Xenium" else spot_size(ax, fig, X1 - X0)
        ax.scatter(xy[:, 0], xy[:, 1], s=s, c=col, lw=0, rasterized=True, antialiased=False)
        for r in tiles[tiles.platform == platform].itertuples():
            box(ax, r, "#BDBDBD", lw=0.25)
        if platform == "Visium":
            box(ax, vis.loc[q], ps.QUERY_COLOR, lw=1.0)
        else:
            for t in example["index_tile"]:
                box(ax, xen.loc[t], ps.INK, lw=1.0)
        spatial(ax, X0, X1, Y0, Y1)
        ps.save(fig, FIG, f"map_{platform.lower()}", source=f"results/{FIG}/cells.npz, tiles.csv, example.csv")
    present = [c for c in classes if c in CLASS_COLORS]
    ps.legend_atom(FIG, "legend_class", [Patch(fc=CLASS_COLORS[c], ec="none", label=c) for c in present], 180, 5,
                   ncol=len(present), frameon=False, handlelength=1.0, columnspacing=1.2,
                   source=f"results/{FIG}/cells.npz")


# ---------------------------------------------------------------- B
def pca():
    p = pd.read_csv(RES / "bias_pca.csv")
    p = p[p.direction == "v2x"]
    idx, qs = p[p.role == "index"], {c: p[(p.role == "query") & (p.corrected == c)] for c in (False, True)}
    allp = pd.concat([idx, *qs.values()])
    pad = 0.05
    lim = [(allp[c].min() - pad * np.ptp(allp[c]), allp[c].max() + pad * np.ptp(allp[c])) for c in ("PC1", "PC2")]
    for corr, name in ((False, "pca_raw"), (True, "pca_corr")):
        fig, ax = ps.atom(42, 40, left=9.0, bottom=8.0, right=1.5, top=1.5)
        ax.scatter(idx.PC1, idx.PC2, s=3, c=PLATFORM["Xenium"], lw=0, alpha=0.8)
        ax.scatter(qs[corr].PC1, qs[corr].PC2, s=3, c=PLATFORM["Visium"], lw=0, alpha=0.8)
        ax.set_xlim(*lim[0])
        ax.set_ylim(*lim[1])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ps.save(fig, FIG, name, source=f"results/{FIG}/bias_pca.csv")
    ps.legend_atom(FIG, "legend_platform", [Line2D([], [], ls="", marker="o", ms=3, color=PLATFORM[m], label=m)
                                            for m in ("Xenium", "Visium")], 40, 5, ncol=2, frameon=False,
                   handletextpad=0.2, source="")


# ---------------------------------------------------------------- C, E
def tissue():
    t = pd.read_csv(RES / "tissue_seed0.csv")
    fig, ax = ps.atom(50, 46, left=10.0, bottom=8.0, right=1.5, top=1.5)
    ax.plot([0, 1], [0, 1], ls="--", lw=0.5, color=ps.MUTED, zorder=1)
    for d in ("x2v", "v2x"):
        g = t[t.direction == d]
        ax.scatter(g.query_tumour, g.hits_tumour, s=3, c=DIRECTION[d], lw=0, alpha=0.8, zorder=2)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Tumour fraction, query tile")
    ax.set_ylabel("Tumour fraction, top-10 tiles")
    ps.save(fig, FIG, "tissue", source=f"results/{FIG}/tissue_seed0.csv")


def ablation():
    t = pd.read_csv(RES / "tissue.csv").set_index(["direction", "method"])
    arms = [("spindle_none", "None"), ("spindle_per_niche", "Per niche"), ("spindle_global", "Global")]
    fig, ax = ps.atom(50, 46, left=10.0, bottom=8.0, right=1.5, top=1.5)
    w = 0.36
    for j, d in enumerate(("x2v", "v2x")):
        x = np.arange(len(arms)) + (j - 0.5) * w
        m = [t.loc[(d, a), "tissue_r_mean"] for a, _ in arms]
        s = [t.loc[(d, a), "tissue_r_std"] for a, _ in arms]
        ax.bar(x, m, w * 0.95, color=DIRECTION[d], lw=0, zorder=2)
        ax.errorbar(x, m, yerr=s, fmt="none", ecolor=ps.INK, elinewidth=0.5, capsize=1.0, capthick=0.5, zorder=3)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([l for _, l in arms])
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Platform correction")
    ax.set_ylabel("Tissue agreement r")
    ps.save(fig, FIG, "ablation", source=f"results/{FIG}/tissue.csv")
    ps.legend_atom(FIG, "legend_direction", [Line2D([], [], ls="", marker="s", ms=4, color=DIRECTION[d],
                                                    label=DIRECTION_LABEL[d]) for d in ("x2v", "v2x")],
                   70, 5, ncol=2, frameon=False, handletextpad=0.2, source="")


# ---------------------------------------------------------------- F
def zoom(name, platform, r, color, ls, w=24.0):
    fig, ax = ps.atom(w, w)
    side = max(r.x1 - r.x0, r.y1 - r.y0) * 1.08
    cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
    xy = cells[f"{platform}_xy"]
    sel = np.flatnonzero((xy[:, 0] >= r.x0) & (xy[:, 0] < r.x1) & (xy[:, 1] >= r.y0) & (xy[:, 1] < r.y1))
    xy, col = points(platform, sel)
    s = 0.8 if platform == "Xenium" else spot_size(ax, fig, side)
    ax.scatter(xy[:, 0], xy[:, 1], s=s, c=col, lw=0, rasterized=True, antialiased=False)
    box(ax, r, color, ls=ls, lw=1.0)
    spatial(ax, cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)
    ps.save(fig, FIG, name, source=f"results/{FIG}/cells.npz, example.csv")


def example_tiles():
    vis = tiles[tiles.platform == "Visium"].set_index("tile")
    xen = tiles[tiles.platform == "Xenium"].set_index("tile")
    zoom("ex_query", "Visium", vis.loc[int(example["query"].iloc[0])], ps.QUERY_COLOR, "-")
    for r in example.itertuples():
        zoom(f"ex_hit{r.rank}", "Xenium", xen.loc[r.index_tile], ps.INK, "--" if r.matched else "-")


if __name__ == "__main__":
    maps()
    pca()
    tissue()
    ablation()
    example_tiles()
