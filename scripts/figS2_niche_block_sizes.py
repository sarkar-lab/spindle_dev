"""Fig. S2 -- Covariance-niche sizes and gene-block sizes per dataset.

A  training tiles per covariance niche (every niche of every seed-0-4 build)
B  genes per block (every block of every niche, seeds 0-4 pooled)
C  blocks per niche vs niche size

Inputs: results/index_stats/index_stats.csv, results/figure_data/block_sizes.csv
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import figstyle as fs

ORDER = fs.DATASET_ORDER


def main():
    stats = pd.read_csv(fs.RESULTS / "index_stats" / "index_stats.csv")
    stats = stats[stats["seed"].between(0, 4)]
    blocks = pd.read_csv(fs.FIG_DATA / "block_sizes.csv")
    blocks = blocks[blocks["seed"].between(0, 4)]
    rng = np.random.default_rng(0)

    fig = fs.figure(fs.DOUBLE, 62)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.8], wspace=0.35, left=0.06, right=0.99,
                  top=0.93, bottom=0.3)

    ax_a = fig.add_subplot(gs[0])
    niche = []
    for i, key in enumerate(ORDER):
        for row in stats[stats["dataset"] == key].itertuples():
            sizes = [int(v) for v in str(row.niche_sizes).split(",")]
            blocks_n = [int(v) for v in str(row.blocks_per_niche).split(",")]
            niche += [(key, s, b) for s, b in zip(sizes, blocks_n)]
            ax_a.scatter(i + rng.uniform(-0.25, 0.25, len(sizes)), sizes, s=4, color=fs.DATASET_COLORS[key],
                         marker=fs.DATASET_MARKERS[key], lw=0, alpha=0.8)
    ax_a.set_yscale("log")
    fs.plain_log(ax_a, "y")
    fs.log_ygrid(ax_a)
    ax_a.set_ylabel("Tiles per niche")

    ax_b = fig.add_subplot(gs[1])
    for i, key in enumerate(ORDER):
        v = blocks.loc[blocks["dataset"] == key, "size"].to_numpy()
        parts = ax_b.violinplot(v, positions=[i], widths=0.8, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(fs.DATASET_COLORS[key])
            pc.set_alpha(0.6)
            pc.set_linewidth(0)
        ax_b.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=fs.INK, lw=0.8)
    ax_b.set_ylabel("Genes per block")
    ax_b.set_ylim(0, None)

    for ax in (ax_a, ax_b):
        ax.set_xticks(range(len(ORDER)), [fs.DATASET_SHORT[k] for k in ORDER], rotation=40, ha="right",
                      rotation_mode="anchor")
        ax.set_xlim(-0.6, len(ORDER) - 0.4)
        ax.tick_params(axis="x", length=0)

    ax_c = fig.add_subplot(gs[2])
    niche = pd.DataFrame(niche, columns=["dataset", "tiles", "blocks"])
    for key in ORDER:
        d = niche[niche["dataset"] == key]
        ax_c.scatter(d["tiles"], d["blocks"], s=5, color=fs.DATASET_COLORS[key], marker=fs.DATASET_MARKERS[key],
                     lw=0, alpha=0.8)
    ax_c.set_xscale("log")
    fs.plain_log(ax_c, "x")
    ax_c.set_xlabel("Tiles per niche")
    ax_c.set_ylabel("Blocks per niche")

    fig.canvas.draw()
    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C")):
        fs.label_panel(fig, ax, letter, dx_mm=-10)
    fs.legend_below(fig, fs.dataset_handles(), ncol=8, y=0.06)
    fs.save(fig, "figS2_niche_block_sizes")


if __name__ == "__main__":
    main()
