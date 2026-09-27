"""Fig. 5 -- Cross-platform search, Xenium <-> Visium serial breast sections (E12).

A  both sections, cells / spots coloured by annotated cell type (coarse shared classes), tile boxes
B  PC1/PC2 of tile log-covariances before and after the global correction (v2x: Xenium index)
C  Recall/Overlap@eps in both directions, mean +- s.d. over 5 seeds
D  correction ablation (seed 0): none, per-niche, global
E  one Visium query tile and its top-5 Spindle matches in the Xenium index (seed 0); the
   all-niche vs routed comparison was dropped from the figure (user decision, 2026-09-25)

Inputs: results/cross_modal_search/{summary.csv, seed_*/, correction_ablation_seed0/,
bias_pca.csv, tile_overlay_boxes.csv}, results/figure_data/{cross_modal_cells.csv,
cross_modal_v2x_seed0_topk.csv}.
"""

import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import figstyle as fs

CM = fs.RESULTS / "cross_modal_search"
DIRS = ["x2v", "v2x"]
DIR_LABELS = {"x2v": "Xenium → Visium", "v2x": "Visium → Xenium"}
# a direction is drawn in its query platform's colour
DIR_COLORS = {"x2v": fs.PLATFORM_COLORS["Xenium"], "v2x": fs.PLATFORM_COLORS["Visium"]}
CLASSES = ["Invasive tumour", "DCIS", "Myoepithelial", "Stroma", "Immune", "Adipocytes", "Mixed"]
CLASS_COLORS = dict(zip(CLASSES, ["#CC6677", "#882255", "#DDCC77", "#44AA99", "#332288", "#E69F00", "#AAAAAA"]))
CLASS_COLORS["Unlabeled"] = "#E6E6E6"
METRICS = [("recall_at_eps_0.1", r"Recall@0.1$\varepsilon$"), ("recall_at_eps_0.5", r"Recall@0.5$\varepsilon$"),
           ("overlap_at_eps_0.5", r"Overlap@0.5$\varepsilon$"), ("overlap_at_eps_1.0", r"Overlap@1$\varepsilon$")]


def boxes(ax, b, **kw):
    rects = [Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0) for r in b.itertuples()]
    ax.add_collection(PatchCollection(rects, **kw))


def section(ax, cells, mod, title, colour=True, s=0.4):
    c = cells[cells["modality"] == mod]
    col = c["coarse"].map(CLASS_COLORS) if colour else fs.LIGHT
    ax.scatter(c["x"], c["y"], s=s, c=col, lw=0, rasterized=True, zorder=1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(title, fontsize=fs.TEXT_PT, pad=2)


def panel_sections(fig, spec, cells, tile_boxes):
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, width_ratios=[1, 1, 0.55], wspace=0.05)
    axes = []
    for j, (mod, n_label) in enumerate((("Xenium", "cells"), ("Visium", "spots"))):
        ax = fig.add_subplot(gs[j])
        s = 0.25 if mod == "Xenium" else 1.6
        n_tiles = (tile_boxes["modality"] == mod).sum()
        section(ax, cells, mod, f"{mod} ({n_tiles} tiles)", s=s)
        boxes(ax, tile_boxes[tile_boxes["modality"] == mod], facecolor="none", edgecolor=fs.INK, lw=0.2, zorder=2)
        axes.append(ax)
    ylim = (max(a.get_ylim()[0] for a in axes), min(a.get_ylim()[1] for a in axes))
    for a in axes:
        a.set_ylim(*ylim)
    leg = fig.add_subplot(gs[2])
    leg.set_axis_off()
    leg.legend(handles=[Patch(color=CLASS_COLORS[k], label=k) for k in CLASSES + ["Unlabeled"]],
               loc="center left", bbox_to_anchor=(0.0, 0.5), handlelength=0.8, handleheight=0.8)
    return axes


def panel_pca(fig, spec):
    p = pd.read_csv(CM / "bias_pca.csv")
    p = p[p["direction"] == "v2x"]
    summ = pd.read_csv(CM / "bias_pca_summary.csv").set_index(["direction", "corrected"])
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.12)
    axes = []
    lim_x = (p["PC1"].min() - 1, p["PC1"].max() + 1)
    lim_y = (p["PC2"].min() - 1, p["PC2"].max() + 1)
    for j, corrected in enumerate((False, True)):
        ax = fig.add_subplot(gs[j])
        # index tiles are never corrected, so they are stored once (corrected == False)
        d = pd.concat([p[p["role"] == "index"], p[(p["role"] == "query") & (p["corrected"] == corrected)]])
        for mod in ("Xenium", "Visium"):
            m = d[d["modality"] == mod]
            ax.scatter(m["PC1"], m["PC2"], s=4, color=fs.PLATFORM_COLORS[mod], lw=0, alpha=0.8, label=mod)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xlabel("PC1")
        if j == 0:
            ax.set_ylabel("PC2")
        else:
            ax.set_yticklabels([])
        sil = summ.loc[("v2x", corrected), "modality_silhouette"]
        ax.set_title(("Corrected" if corrected else "Uncorrected") + f" (silhouette {sil:.2f})", fontsize=fs.TICK_PT)
        axes.append(ax)
    axes[1].legend(handles=[Line2D([], [], ls="none", marker="o", ms=3, color=fs.PLATFORM_COLORS[m], label=m)
                            for m in ("Xenium", "Visium")], loc="lower right")
    return axes


def panel_metrics(ax):
    summ = pd.read_csv(CM / "summary.csv").set_index("direction")
    per_seed = pd.concat([pd.read_csv(CM / f"seed_{s}" / f"{d}_summary.csv") for s in range(5) for d in DIRS])
    x = np.arange(len(METRICS))
    for k, d in enumerate(DIRS):
        off = (k - 0.5) * 0.3
        for i, (col, _) in enumerate(METRICS):
            v = per_seed.loc[per_seed["direction"] == d, col]
            ax.scatter(np.full(len(v), i + off) + np.linspace(-0.06, 0.06, len(v)), v, s=3, color=DIR_COLORS[d],
                       alpha=0.4, lw=0, zorder=2)
        ax.errorbar(x + off, [summ.loc[d, f"mean_{c}"] for c, _ in METRICS],
                    yerr=[summ.loc[d, f"sd_{c}"] for c, _ in METRICS], fmt="o", ms=3.2, mew=0,
                    color=DIR_COLORS[d], elinewidth=0.7, zorder=3, label=DIR_LABELS[d])
    ax.set_xticks(x, [m for _, m in METRICS], rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.85, 1.01)
    ax.set_ylabel("Score")
    ax.legend(loc="lower left", handletextpad=0.1)


def seed0_summary(direction, variant="all_niche", correction="global"):
    tag = "" if variant == "all_niche" else "_single_niche_baseline"
    if correction == "per_niche":
        f = CM / "correction_ablation_seed0" / f"{direction}{tag}_summary_corr-per_niche.csv"
    else:
        f = CM / "correction_ablation_seed0" / f"{direction}{tag}_corr-{correction}_summary.csv"
    return pd.read_csv(f).iloc[0]


def grouped_bars(ax, groups, values, ylabel):
    """values[direction] = list aligned with groups; bars start at 0 (full axis)."""
    x = np.arange(len(groups))
    w = 0.36
    for k, d in enumerate(DIRS):
        v = values[d]
        ax.bar(x + (k - 0.5) * w, v, width=w * 0.92, color=DIR_COLORS[d], lw=0, label=DIR_LABELS[d])
        for xi, vi in zip(x + (k - 0.5) * w, v):
            ax.text(xi, vi + 0.02, f"{vi:.2f}".replace("1.00", "1").replace("0.00", "0"), ha="center",
                    va="bottom", fontsize=fs.TICK_PT)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", length=0)


def panel_ablation(ax):
    groups = [("none", "None"), ("per_niche", "Per niche"), ("global", "Global")]
    vals = {d: [seed0_summary(d, correction=c)["recall_at_eps_0.1"] for c, _ in groups] for d in DIRS}
    grouped_bars(ax, [g for _, g in groups], vals, r"Recall@0.1$\varepsilon$")
    ax.set_xlabel("Bias correction")


def pick_example(topk, tile_boxes):
    """Query whose Spindle top-5 equals the exact top-5 and lies closest to the query's own location.

    Visium tiles reuse the Xenium quadtree boxes, so distances are between tile centres in the
    shared (co-registered) coordinate frame. Across the 50 seed-0 queries the co-located Xenium
    tile is in the top-5 for 10; this picks the clearest of those cases (state it in the caption).
    """
    vb = tile_boxes[tile_boxes["modality"] == "Visium"].set_index("id")
    xb = tile_boxes[tile_boxes["modality"] == "Xenium"].set_index("id")
    cx, cy = (xb["x0"] + xb["x1"]) / 2, (xb["y0"] + xb["y1"]) / 2
    best, best_d = None, np.inf
    for q, d in topk[topk["rank"] <= 5].groupby("query_tile_id"):
        hits = d.loc[d["method"] == "spindle", "index_tile_id"]
        if set(hits) != set(d.loc[d["method"] == "exact", "index_tile_id"]):
            continue
        qx, qy = (vb.loc[q, "x0"] + vb.loc[q, "x1"]) / 2, (vb.loc[q, "y0"] + vb.loc[q, "y1"]) / 2
        dist = np.hypot(cx[hits] - qx, cy[hits] - qy).mean()
        if dist < best_d:
            best, best_d = q, dist
    return best


def panel_example(fig, spec, cells, tile_boxes):
    topk = pd.read_csv(fs.FIG_DATA / "cross_modal_v2x_seed0_topk.csv")
    q = pick_example(topk, tile_boxes)
    hits = topk[(topk["query_tile_id"] == q) & (topk["method"] == "spindle") & (topk["rank"] <= 5)]
    vb = tile_boxes[tile_boxes["modality"] == "Visium"].set_index("id")
    xb = tile_boxes[tile_boxes["modality"] == "Xenium"].set_index("id")
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.05)
    ax_v = fig.add_subplot(gs[0])
    section(ax_v, cells, "Visium", "Visium query tile", s=1.2)
    boxes(ax_v, vb.loc[[q]], facecolor="none", edgecolor=fs.INK, lw=1.0, zorder=3)
    ax_x = fig.add_subplot(gs[1])
    section(ax_x, cells, "Xenium", "Top-5 Xenium matches", s=0.2)
    b = vb.loc[q]
    ax_x.add_patch(Rectangle((b.x0, b.y0), b.x1 - b.x0, b.y1 - b.y0, fill=False, ls=(0, (2, 1.5)),
                             ec=fs.PLATFORM_COLORS["Visium"], lw=1.0, zorder=4))
    boxes(ax_x, xb.loc[hits["index_tile_id"]], facecolor="none", edgecolor=fs.INK, lw=0.9, zorder=3)
    for r in hits.itertuples():
        b = xb.loc[r.index_tile_id]
        ax_x.text(b.x1, b.y0, str(r.rank), fontsize=fs.TICK_PT, ha="left", va="bottom", zorder=4)
    ylim = (max(ax_v.get_ylim()[0], ax_x.get_ylim()[0]), min(ax_v.get_ylim()[1], ax_x.get_ylim()[1]))
    for a in (ax_v, ax_x):
        a.set_ylim(*ylim)
    ax_x.legend(handles=[Patch(fc="none", ec=fs.INK, lw=0.9, label="Top-5 match (rank)"),
                         Patch(fc="none", ec=fs.PLATFORM_COLORS["Visium"], lw=1.0, ls=(0, (2, 1.5)),
                               label="Query location")],
                loc="upper center", bbox_to_anchor=(0.0, -0.02), ncol=2, handlelength=1.0)
    return ax_v, int(q)


def main():
    cells = pd.read_csv(fs.FIG_DATA / "cross_modal_cells.csv")
    tile_boxes = pd.read_csv(CM / "tile_overlay_boxes.csv")

    fig = fs.figure(fs.DOUBLE, 135)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.15], hspace=0.3, left=0.06, right=0.99,
                     top=0.95, bottom=0.1)
    top = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0], width_ratios=[1.25, 1], wspace=0.12)
    bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1.0, 1.0, 2.3], wspace=0.4)

    ax_a = panel_sections(fig, top[0], cells, tile_boxes)
    ax_b = panel_pca(fig, top[1])
    ax_c = fig.add_subplot(bot[0])
    panel_metrics(ax_c)
    ax_d = fig.add_subplot(bot[1])
    panel_ablation(ax_d)
    ax_e, q = panel_example(fig, bot[2], cells, tile_boxes)
    print(f"Fig. 5E example: Visium query tile {q}")

    fig.canvas.draw()
    fs.label_panel(fig, ax_a[0], "A", dx_mm=-4)
    fs.label_panel(fig, ax_b[0], "B", dx_mm=-9)
    fs.label_panel(fig, ax_c, "C", dx_mm=-10)
    fs.label_panel(fig, ax_d, "D", dx_mm=-9)
    fs.label_panel(fig, ax_e, "E", dx_mm=-3)
    fs.save(fig, "fig5_cross_platform")


if __name__ == "__main__":
    main()
