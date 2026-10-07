"""Fig. S11 -- module analysis in the five other tissues, 180 mm.

One row per tissue (kidney, lung, pancreas, lymph node, brain), each with template panels
(i) niche map, (ii) module heatmap of the selected node, (iii) GO BP enrichment of the
highlighted module (panel background). Node selection: scripts/figure_data/bio_configs/<dataset>.yaml.
CSV inputs only: results/biology/<dataset>/, results/figure_data/tiles_seed73.csv.
"""

from matplotlib.patches import Patch

import bio_panels as bp
import figstyle as fs

KEYS = ["kidney_nondiseased", "lung_cancer", "pancreatic_cancer", "lymph_node", "brain_cancer"]
ROW_H = 44.0
TOP = 6.0
W, H = fs.DOUBLE, TOP + ROW_H * len(KEYS) + 4.0
MAP_W, MAP_H = 52.0, 30.0
HEAT = 36.0
N_TERMS = 5


def row(fig, d, y0, letter, last):
    s = d["sel"]
    fs.label_at(fig, 1.5, y0, letter)
    fig.text(4.5 / W, 1 - (y0 + 1.2) / H, f"{fs.DATASET_LABELS[d['key']]}: {bp.node_label(d)}, node score rank "
             f"{s['rank']:,} of {s.n_nodes:,}", fontsize=fs.TEXT_PT, va="top")

    # (i) niche map, fitted into MAP_W x MAP_H
    asp = bp.tissue_aspect(d["tiles"])
    w = min(MAP_W, MAP_H / asp)
    h = w * asp
    ax = bp.ax_mm(fig, 6 + (MAP_W - w) / 2, y0 + 7 + (MAP_H - h) / 2, w, h)
    bp.niche_map(ax, d, lw=0.05)
    pal = fs.niche_palette(bp.n_niches(d))
    ax.legend(handles=[Patch(facecolor=pal[int(s.niche)], edgecolor="none",
                             label=f"Source: {bp.niche_label(int(s.niche))} of {bp.n_niches(d)}")],
              loc="upper center", bbox_to_anchor=(0.5, -0.02), handlelength=0.9)

    # (ii) heatmap
    x = 88.0
    bp.module_heatmap(fig, d, x, y0 + 5.5, HEAT, cbar=last)

    # (iii) enrichment
    ax = bp.ax_mm(fig, 160.0, y0 + 7, 17.0, 29.0)
    bp.enrichment(ax, d, n=N_TERMS, width=24)
    ax.set_title(f"{bp.LIB_LABEL[s.library]}, {bp.highlight_label(d)}", fontsize=fs.TICK_PT, loc="right", pad=2)
    if not last:
        ax.set_xlabel("")


def main():
    fig = fs.figure(W, H)
    for k, key in enumerate(KEYS):
        d = bp.load(key)
        row(fig, d, TOP + k * ROW_H, "ABCDE"[k], k == len(KEYS) - 1)
        s = d["sel"]
        print(f"\n{key}: {bp.node_label(d)}, node {s.node}, rank {s['rank']}/{s.n_nodes}, {s.n_tiles_node} tile(s), "
              f"highlight {bp.highlight_label(d)} ({s.highlight_size} genes, {s.reference_in_module}/"
              f"{s.reference_size} of the original module)")
        print(bp.caption_numbers(d).head(N_TERMS).to_string())
    fs.save(fig, "figS11_other_tissues")


if __name__ == "__main__":
    main()
