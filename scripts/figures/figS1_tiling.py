"""Fig. S1 -- Quadtree tiling and low-density tile removal.

A  breast (seed-73 tiling): training, held-out and dropped low-density tiles over the cells
B  zoom into the region with the most dropped tiles
C  tiles and cells removed by the low-density filter, per dataset

Inputs: results/figure_data/{tiles_seed73.csv, cells_sample.csv}, results/index_stats/index_stats.csv
"""

import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, Rectangle

import figstyle as fs

STATUS = {"train": ("none", fs.INK, 0.25, "Training tile"),
          "heldout": ("#88CCEE", fs.INK, 0.25, "Held-out (query) tile"),
          "dropped": ("#CC6677", "#CC6677", 0.4, "Dropped, low density")}


def draw_tiles(ax, t, cells, lw_scale=1.0):
    ax.scatter(cells["x"], cells["y"], s=0.15, color="#BDBDBD", lw=0, rasterized=True, zorder=1)
    for status, (fc, ec, lw, _) in STATUS.items():
        sub = t[t["status"] == status]
        rects = [Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0) for r in sub.itertuples()]
        ax.add_collection(PatchCollection(rects, facecolor=fc, edgecolor=ec, lw=lw * lw_scale,
                                          alpha=0.8 if fc != "none" else 1, zorder=2))
    ax.set_aspect("equal")
    ax.set_axis_off()


def main():
    tiles = pd.read_csv(fs.FIG_DATA / "tiles_seed73.csv")
    cells = pd.read_csv(fs.FIG_DATA / "cells_sample.csv")
    stats = pd.read_csv(fs.RESULTS / "index_stats" / "index_stats.csv")
    # the low-density drop depends only on the tiling (collect_index_stats.py checks it is seed-constant);
    # the dropped-cell count was logged for seeds 0-4 only, so it is read from seed 0
    spots = stats[stats["seed"] == 0].set_index("dataset")["n_spots_dropped_low_density"]
    stats = stats[stats["seed"] == 73].set_index("dataset").loc[fs.DATASET_ORDER]
    stats["n_spots_dropped_low_density"] = spots.loc[fs.DATASET_ORDER]

    key = "breast_cancer"
    t, c = tiles[tiles["dataset"] == key], cells[cells["dataset"] == key]

    fig = fs.figure(fs.DOUBLE, 80)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 0.9, 1.0], wspace=0.42, left=0.02, right=0.98,
                  top=0.9, bottom=0.2)
    ax_a = fig.add_subplot(gs[0])
    draw_tiles(ax_a, t, c)
    ax_a.set_xlim(t["x0"].min(), t["x1"].max())
    ax_a.set_ylim(t["y1"].max(), t["y0"].min())
    n = t["status"].value_counts()
    ax_a.set_title(f"Breast: {n['train']:,} training, {n['heldout']} held-out, {n['dropped']} dropped tiles",
                   fontsize=fs.TICK_PT)

    # zoom: the 1/4-width window containing the most dropped tiles
    d = t[t["status"] == "dropped"]
    w = (t["x1"].max() - t["x0"].min()) / 4
    cx = (d["x0"] + d["x1"]) / 2
    cy = (d["y0"] + d["y1"]) / 2
    counts = [((abs(cx - x) < w / 2) & (abs(cy - y) < w / 2)).sum() for x, y in zip(cx, cy)]
    i = int(np.argmax(counts))
    x0, y0 = cx.iloc[i] - w / 2, cy.iloc[i] - w / 2
    ax_a.add_patch(Rectangle((x0, y0), w, w, fill=False, ec=fs.INK, lw=0.8, ls="--", zorder=3))
    ax_b = fig.add_subplot(gs[1])
    draw_tiles(ax_b, t, c, lw_scale=2.0)
    ax_b.set_xlim(x0, x0 + w)
    ax_b.set_ylim(y0 + w, y0)
    ax_b.set_title("Zoom (dashed box in A)", fontsize=fs.TICK_PT)
    for sp in ("top", "right", "bottom", "left"):
        ax_b.spines[sp].set_visible(True)

    ax_c = fig.add_subplot(gs[2])
    y = np.arange(len(stats))
    total_tiles = stats["n_tiles_train"] + stats["n_tiles_query"] + stats["n_tiles_dropped_low_density"]
    pct_tiles = 100 * stats["n_tiles_dropped_low_density"] / total_tiles
    pct_cells = 100 * stats["n_spots_dropped_low_density"] / stats["cells"]
    ax_c.barh(y - 0.19, pct_tiles, height=0.36, color="#CC6677", lw=0, label="Tiles")
    ax_c.barh(y + 0.19, pct_cells, height=0.36, color="#BDBDBD", lw=0, label="Cells")
    for yi, v, n_t in zip(y, pct_tiles, stats["n_tiles_dropped_low_density"]):
        ax_c.text(v + 0.1, yi - 0.19, f"{int(n_t)}", va="center", fontsize=fs.TICK_PT)
    ax_c.set_yticks(y, [fs.DATASET_SHORT[k] for k in stats.index])
    ax_c.invert_yaxis()
    ax_c.set_xlabel("Removed by low-density filter (%)")
    ax_c.tick_params(axis="y", length=0)
    ax_c.legend(loc="lower right")

    fig.canvas.draw()
    fs.label_panel(fig, ax_a, "A", dx_mm=0, dy_mm=3)
    fs.label_panel(fig, ax_b, "B", dx_mm=-2, dy_mm=3)
    fs.label_panel(fig, ax_c, "C", dx_mm=-18, dy_mm=-2)
    fs.legend_below(fig, [Patch(fc=fc, ec=ec, lw=0.6, label=lab) for fc, ec, _, lab in STATUS.values()],
                    ncol=3, y=0.12)
    fs.save(fig, "figS1_tiling")


if __name__ == "__main__":
    main()
