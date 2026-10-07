"""Fig. 5 -- Cross-platform search, Xenium <-> Visium serial breast sections.

A  both sections, cells / spots coloured by annotated cell type (coarse shared classes), tile boxes
B  PC1/PC2 of shrunk tile log-covariances before and after the global correction (v2x: Xenium index)
C  Spindle-Exact, global correction. Left: tissue agreement, the tumour fraction of each query tile (its own
   platform's annotation) vs the mean of its top-10 tiles on the other platform (seed 0; r = mean +- s.d.
   over seeds 0-4). Right: fraction of queries whose co-located tile is within rank r, with a
   tumour-fraction oracle and random
D  correction ablation (none, per niche, global): tissue-agreement r, mean +- s.d. over seeds
E  a DCIS-rich Visium query tile and its top-5 Xenium tiles (seed 0): both sections, then each tile enlarged
   with its share of the query's class
Sections are drawn rotated by ~2 degrees so the tissue is square to the axes (class Frame).

Inputs: results/cross_platform_tiers/{summary.csv, rank_cdf.csv, oracle_rank_cdf.csv,
tissue_per_query_seed0.csv, bias_pca*.csv, tile_boxes_tiles2000.csv,
cross_modal_cells.csv, example_v2x.csv}.
"""

import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
from scipy.spatial import cKDTree

import matplotlib.patheffects as pe

import figstyle as fs

RES = fs.RESULTS / "cross_platform_tiers"
MAX_PTS = 2000
DIRS = ["x2v", "v2x"]
DIR_LABELS = {"x2v": "Xenium → Visium", "v2x": "Visium → Xenium"}
# a direction is drawn in its query platform's colour
DIR_COLORS = {"x2v": fs.PLATFORM_COLORS["Xenium"], "v2x": fs.PLATFORM_COLORS["Visium"]}
ARMS = [("none", "None"), ("per_niche", "Per\nniche"), ("global", "Global")]
CLASSES = ["Invasive tumour", "DCIS", "Myoepithelial", "Stroma", "Immune", "Adipocytes", "Mixed"]
CLASS_COLORS = dict(zip(CLASSES, ["#CC6677", "#882255", "#DDCC77", "#44AA99", "#332288", "#E69F00", "#AAAAAA"]))
CLASS_COLORS["Unlabeled"] = "#E6E6E6"


class Frame:
    """Display rotation that squares the sections to the axes.

    The registered frame is rotated: the Visium spot grid (hexagonal, neighbours every 60 degrees) sits at
    about -2 degrees. Cells, spots and tile corners are all rotated by the opposite angle about the Visium
    centre for display only, and each overview map is cropped to the largest straight rectangle inside its
    section. Tiles were cut in the original frame, so their boxes keep the small tilt.
    """

    def __init__(self, cells):
        v = cells.loc[cells["modality"] == "Visium", ["x", "y"]].to_numpy()
        d, i = cKDTree(v).query(v, k=7)
        nb = (v[i[:, 1:]] - v[:, None, :]).reshape(-1, 2)
        nb = nb[np.hypot(*nb.T) < 1.2 * np.median(d[:, 1])]
        a = 6 * np.arctan2(nb[:, 1], nb[:, 0])  # 60-degree period -> full circle
        self.angle = -np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / 6
        self.centre = v.mean(0)
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.R = np.array([[c, -s], [s, c]])

        # crop per section: the largest axis-aligned rectangle (display frame) inside its rotated area (the
        # Xenium imaging area; the Visium spot area, half a spot pitch in from the outer spots)
        self.lims = {}
        for mod, inset in (("Xenium", 0.0), ("Visium", 0.5 * np.median(d[:, 1]))):
            xy = cells.loc[cells["modality"] == mod, ["x", "y"]].to_numpy()
            (x0, y0), (x1, y1) = xy.min(0) + inset, xy.max(0) - inset
            k = self([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])  # corners: tl, tr, br, bl (y down)
            self.lims[mod] = ((max(k[0, 0], k[3, 0]), min(k[1, 0], k[2, 0])),
                              (min(k[2, 1], k[3, 1]), max(k[0, 1], k[1, 1])))  # y inverted: bottom first

    def frame(self, ax, mod):
        ax.set_xlim(*self.lims[mod][0])
        ax.set_ylim(*self.lims[mod][1])

    def __call__(self, xy):
        return (np.asarray(xy, float) - self.centre) @ self.R.T + self.centre

    def cells(self, cells):
        out = cells.copy()
        out[["x", "y"]] = self(cells[["x", "y"]].to_numpy())
        return out

    def corners(self, r):
        return self([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])


FRAME = None  # set in main()


def boxes(ax, b, **kw):
    ax.add_collection(PatchCollection([Polygon(FRAME.corners(r), closed=True) for r in b.itertuples()], **kw))


OVERVIEW_XENIUM = 30000  # cells drawn on a whole-section map (all cells in the enlarged tiles)


def section(ax, cells, mod, title, s=0.4):
    c = cells[cells["modality"] == mod]
    if len(c) > OVERVIEW_XENIUM:
        c = c.sample(OVERVIEW_XENIUM, random_state=0)
    ax.scatter(c["x"], c["y"], s=s, c=c["coarse"].map(CLASS_COLORS), lw=0, rasterized=True, zorder=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(title, fontsize=fs.TEXT_PT, pad=2)


def panel_sections(fig, spec, cells, tile_boxes):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, width_ratios=[1, 1, 0.55], wspace=0.05)
    axes = []
    for j, mod in enumerate(("Xenium", "Visium")):
        ax = fig.add_subplot(gs[j])
        n_tiles = (tile_boxes["modality"] == mod).sum()
        section(ax, cells, mod, f"{mod} ({n_tiles} tiles)", s=0.25 if mod == "Xenium" else 1.6)
        boxes(ax, tile_boxes[tile_boxes["modality"] == mod], facecolor="none", edgecolor=fs.INK, lw=0.2, zorder=2)
        FRAME.frame(ax, mod)
        axes.append(ax)
    leg = fig.add_subplot(gs[2])
    leg.set_axis_off()
    leg.legend(handles=[Patch(color=CLASS_COLORS[k], label=k) for k in CLASSES + ["Unlabeled"]],
               loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=0.8, handleheight=0.8)
    return axes


def panel_pca(fig, spec):
    p = pd.read_csv(RES / "bias_pca.csv")
    p = p[p["direction"] == "v2x"]
    summ = pd.read_csv(RES / "bias_pca_summary.csv").set_index(["direction", "corrected"])
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.12)
    pad_x, pad_y = 0.05 * np.ptp(p["PC1"]), 0.05 * np.ptp(p["PC2"])
    axes = []
    for j, corrected in enumerate((False, True)):
        ax = fig.add_subplot(gs[j])
        # index tiles are never corrected, so they are stored once (corrected == False)
        d = pd.concat([p[p["role"] == "index"], p[(p["role"] == "query") & (p["corrected"] == corrected)]])
        for mod in ("Xenium", "Visium"):
            m = d[d["modality"] == mod]
            ax.scatter(m["PC1"], m["PC2"], s=4, color=fs.PLATFORM_COLORS[mod], lw=0, alpha=0.8)
        ax.set_xlim(p["PC1"].min() - pad_x, p["PC1"].max() + pad_x)
        ax.set_ylim(p["PC2"].min() - pad_y, p["PC2"].max() + pad_y)
        ax.set_xlabel("PC1")
        if j == 0:
            ax.set_ylabel("PC2")
        else:
            ax.set_yticklabels([])
        sil = summ.loc[("v2x", corrected), "modality_silhouette"]
        ax.set_title(("Corrected" if corrected else "Uncorrected") + f" (silhouette {sil:.2f})", fontsize=fs.TICK_PT)
        axes.append(ax)
    axes[1].legend(handles=[Line2D([], [], ls="none", marker="o", ms=3, color=fs.PLATFORM_COLORS[m], label=m)
                            for m in ("Xenium", "Visium")], loc="upper right")
    return axes


def sel(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df


def panel_tissue(ax, summ):
    """Query box's tumour fraction (own platform) vs the mean of its top-10 hits (other platform), seed 0."""
    t = sel(pd.read_csv(RES / "tissue_per_query_seed0.csv"), max_pts=MAX_PTS, arm="global")
    for j, d in enumerate(DIRS):
        v = t[t["direction"] == d]
        r = sel(summ, direction=d, arm="global", tier="exact").iloc[0]
        ax.scatter(v["query_tumour"], v["hits_tumour"], s=4, color=DIR_COLORS[d], lw=0, alpha=0.7)
        ax.text(0.98, 0.04 + 0.09 * (1 - j), f"r = {r['tissue_r_mean']:.2f} ± {r['tissue_r_std']:.2f}",
                color=DIR_COLORS[d], transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.TICK_PT)
    ax.plot([0, 1], [0, 1], color=fs.MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlabel("Tumour fraction, query tile")
    ax.set_ylabel("Tumour fraction, top-10 tiles")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", ms=3, color=DIR_COLORS[d], label=DIR_LABELS[d])
                       for d in DIRS], loc="upper left", handletextpad=0.1)


def panel_rank_cdf(ax, summ):
    cdf = sel(pd.read_csv(RES / "rank_cdf.csv"), max_pts=MAX_PTS, arm="global", tier="exact")
    oracle = sel(pd.read_csv(RES / "oracle_rank_cdf.csv"), max_pts=MAX_PTS)
    for d in DIRS:
        c = cdf[cdf["direction"] == d]
        ax.plot(c["r"], c["frac_mean"], color=DIR_COLORS[d], lw=1.0, zorder=3)
        ax.fill_between(c["r"], c["frac_mean"] - c["frac_std"], c["frac_mean"] + c["frac_std"],
                        color=DIR_COLORS[d], alpha=0.25, lw=0, zorder=2)
        o = oracle[oracle["direction"] == d]
        ax.plot(o["r"], o["frac"], color=DIR_COLORS[d], lw=0.7, ls="--", zorder=2)
        n = sel(summ, direction=d, arm="global", tier="exact")["n_index_mean"].iloc[0]
        ax.plot(c["r"], np.minimum(1, c["r"] / n), color=DIR_COLORS[d], lw=0.6, ls=":", zorder=1)
    ax.set_xlim(1, cdf["r"].max())
    ax.set_ylim(0, 0.6)
    ax.set_xlabel("Rank r of the co-located tile")
    ax.set_ylabel("Queries with rank ≤ r", labelpad=1)
    ax.legend(handles=[Line2D([], [], color=fs.INK, lw=1.0, label="Spindle-Exact"),
                       Line2D([], [], color=fs.INK, lw=0.7, ls="--", label="Tumour-fraction oracle"),
                       Line2D([], [], color=fs.INK, lw=0.6, ls=":", label="Random")], loc="upper left")


def panel_ablation(ax, summ):
    x = np.arange(len(ARMS))
    w = 0.36
    for k, d in enumerate(DIRS):
        rows = [sel(summ, direction=d, arm=a, tier="exact").iloc[0] for a, _ in ARMS]
        ax.bar(x + (k - 0.5) * w, [r["tissue_r_mean"] for r in rows], yerr=[r["tissue_r_std"] for r in rows],
               width=w * 0.92, color=DIR_COLORS[d], lw=0, error_kw=dict(elinewidth=0.6, capsize=0),
               label=DIR_LABELS[d])
    ax.set_xticks(x, [g for _, g in ARMS])
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Tumour-fraction r,\nquery vs top-10 tiles")
    ax.set_xlabel("Platform correction")
    ax.tick_params(axis="x", length=0)


def zoom(ax, cells, mod, corners, title, s, edge, ls="-"):
    """One tile at full size: its cells / spots, its box, a title."""
    c = cells[cells["modality"] == mod]
    lo, hi = corners.min(0), corners.max(0)
    pad = 0.04 * (hi - lo).max()
    inside = c[(c["x"] >= lo[0] - pad) & (c["x"] <= hi[0] + pad) & (c["y"] >= lo[1] - pad) & (c["y"] <= hi[1] + pad)]
    ax.scatter(inside["x"], inside["y"], s=s, c=inside["coarse"].map(CLASS_COLORS), lw=0, rasterized=True)
    ax.add_patch(Polygon(corners, closed=True, fill=False, ec=edge, lw=1.0, ls=ls))
    side = (hi - lo).max() / 2 + pad
    mid = (lo + hi) / 2
    ax.set_xlim(mid[0] - side, mid[0] + side)
    ax.set_ylim(mid[1] + side, mid[1] - side)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title(title, fontsize=fs.TICK_PT, pad=1.5)


def panel_example(fig, spec, cells, tile_boxes):
    ex = pd.read_csv(RES / "example_v2x.csv")
    q = int(ex["query"].iloc[0])
    hits = ex[ex["rank"] <= 5]
    vb = tile_boxes[tile_boxes["modality"] == "Visium"].set_index("id")
    xb = tile_boxes[tile_boxes["modality"] == "Xenium"].set_index("id")
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, width_ratios=[1.05, 1.0], wspace=0.08)
    over = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.04)
    ax_v = fig.add_subplot(over[0])
    section(ax_v, cells, "Visium", "Visium, query tile", s=1.6)
    boxes(ax_v, vb.loc[[q]], facecolor="none", edgecolor=fs.INK, lw=1.2, zorder=3)
    ax_x = fig.add_subplot(over[1])
    section(ax_x, cells, "Xenium", f"Xenium, top 5 (section: {100 * ex['section_frac'].iloc[0]:.0f}% "
                                   f"{ex['focal_class'].iloc[0]})", s=0.3)
    ax_x.add_patch(Polygon(FRAME.corners(vb.loc[q]), closed=True, fill=False, ls=(0, (2, 1.5)),
                           ec=fs.PLATFORM_COLORS["Visium"], lw=1.2, zorder=4))
    boxes(ax_x, xb.loc[hits["index_tile"]], facecolor="none", edgecolor=fs.INK, lw=1.0, zorder=3)
    for r in hits.itertuples():
        x, y = FRAME.corners(xb.loc[r.index_tile])[2]
        ax_x.text(x, y, str(r.rank), fontsize=fs.TICK_PT, ha="left", va="top", zorder=4,
                  path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])
    FRAME.frame(ax_v, "Visium")
    FRAME.frame(ax_x, "Xenium")
    ax_x.legend(handles=[Patch(fc="none", ec=fs.INK, lw=1.0, label="Top-5 tile (rank)"),
                         Patch(fc="none", ec=fs.PLATFORM_COLORS["Visium"], lw=1.0, ls=(0, (2, 1.5)),
                               label="Query location")],
                loc="upper center", bbox_to_anchor=(0.0, -0.05), ncol=2, handlelength=1.0)
    tiles = GridSpecFromSubplotSpec(2, 3, subplot_spec=gs[1], wspace=0.08, hspace=0.25)
    cls = ex["focal_class"].iloc[0]
    zoom(fig.add_subplot(tiles[0, 0]), cells, "Visium", FRAME.corners(vb.loc[q]),
         f"Query (Visium): {100 * ex['query_frac'].iloc[0]:.0f}% {cls}", 14, fs.PLATFORM_COLORS["Visium"])
    for k, r in enumerate(hits.itertuples(), start=1):
        b = xb.loc[r.index_tile]
        title = f"{r.rank}{' (co-located)' if r.matched else ''}: {100 * r.hit_frac:.0f}% {cls}"
        zoom(fig.add_subplot(tiles[k // 3, k % 3]), cells, "Xenium", FRAME.corners(b), title, 0.8, fs.INK,
             ls=(0, (2, 1.5)) if r.matched else "-")
    return ax_v, q


def main():
    global FRAME
    raw = pd.read_csv(RES / "cross_modal_cells.csv")
    FRAME = Frame(raw)
    cells = FRAME.cells(raw)
    tile_boxes = pd.read_csv(RES / f"tile_boxes_tiles{MAX_PTS}.csv")
    summ = sel(pd.read_csv(RES / "summary.csv"), max_pts=MAX_PTS)
    print(f"display rotation {np.degrees(FRAME.angle):+.2f} degrees")

    fig = fs.figure(fs.DOUBLE, 185)
    outer = GridSpec(3, 1, figure=fig, height_ratios=[1.0, 0.78, 1.05], hspace=0.42, left=0.06, right=0.99,
                     top=0.965, bottom=0.06)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.25, 1], wspace=0.12)
    mid = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1.0, 1.0, 0.9], wspace=0.45)

    ax_a = panel_sections(fig, top[0], cells, tile_boxes)
    ax_b = panel_pca(fig, top[1])
    ax_c = fig.add_subplot(mid[0])
    panel_tissue(ax_c, summ)
    ax_c2 = fig.add_subplot(mid[1])
    panel_rank_cdf(ax_c2, summ)
    ax_d = fig.add_subplot(mid[2])
    panel_ablation(ax_d, summ)
    ax_e, q = panel_example(fig, outer[2], cells, tile_boxes)
    print(f"Fig. 5E example: Visium query tile {q}")

    fig.canvas.draw()
    fs.label_panel(fig, ax_a[0], "A", dx_mm=-4)
    fs.label_panel(fig, ax_b[0], "B", dx_mm=-9)
    fs.label_panel(fig, ax_c, "C", dx_mm=-10, panel=(ax_c2,))
    fs.label_panel(fig, ax_d, "D", dx_mm=-9)
    fs.label_panel(fig, ax_e, "E", dx_mm=-4)
    fs.save(fig, "fig5_cross_platform")


if __name__ == "__main__":
    main()
