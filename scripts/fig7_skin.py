"""Fig. 7 -- skin melanoma case study, 180 mm. Same template as Fig. 6:

A  niche map (template i)
B  module heatmap of the selected node (template ii)
C  GO BP enrichment of the highlighted (epidermal) module, panel background (template iii)
D  module co-variation map (template iv)

The cross-tissue E9 / E10 tables sit in a \\SubOnly slot of the caption, not in this figure.
CSV inputs only: results/biology/skin_melanoma/, results/figure_data/tiles_seed73.csv.
"""

import bio_panels as bp
import figstyle as fs

KEY = "skin_melanoma"
W, H = fs.DOUBLE, 140.0


def main():
    d = bp.load(KEY)
    s = d["sel"]
    fig = fs.figure(W, H)
    asp = bp.tissue_aspect(d["tiles"])
    mw = 76.0
    mh = mw * asp

    # A niche map
    ax = bp.ax_mm(fig, 4, 9, mw, mh)
    bp.niche_map(ax, d)
    ax.set_title("Covariance niches", fontsize=fs.TEXT_PT, pad=2)
    ax.legend(handles=bp.niche_handles(d), loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=4,
              handlelength=0.9, columnspacing=0.8)
    fs.label_at(fig, 1.5, 3.5, "A")

    # D module co-variation map
    yd = 9 + mh + 18
    ax = bp.ax_mm(fig, 4, yd, mw, mh)
    pc, _ = bp.module_map(ax, d)
    ax.set_title(f"Module {bp.highlight_label(d)} co-variation", fontsize=fs.TEXT_PT, pad=2)
    cax = bp.ax_mm(fig, 4 + mw - 24, yd + mh + 1.5, 24, 1.4)
    bp.map_colorbar(fig, pc, cax)
    fs.label_at(fig, 1.5, yd - 5.5, "D")

    # B heatmap
    m, _ = bp.module_order(d)
    side = 1.5 * len(m)
    x, y = 101.0, 13.0
    heat = bp.module_heatmap(fig, d, x, y, side)
    fig.text((x + side / 2) / W, 1 - 3.8 / H,
             f"Node centroid, {bp.node_label(d)} ({s.n_tiles_node} tile; score rank {s['rank']} of {s.n_nodes:,})",
             ha="center", va="top", fontsize=fs.TEXT_PT)
    fs.label_at(fig, 86.0, 3.5, "B", panel=[a for a in heat if a is not None])  # colorbar sits in D's row

    # C enrichment
    yc = y + side + 11
    ax = bp.ax_mm(fig, 150.0, yc + 4, 26.0, 36.0)
    bp.enrichment(ax, d, n=7, width=30)
    ax.set_title(f"{bp.LIB_LABEL[s.library]}, module {bp.highlight_label(d)}", fontsize=fs.TEXT_PT, loc="right",
                 pad=3)
    fs.label_at(fig, 86.0, yc, "C")
    fs.save(fig, "fig7_skin")

    print(f"node: {bp.node_label(d)}, node {s.node}, rank {s['rank']}/{s.n_nodes}, {s.n_tiles_node} tile(s), "
          f"highlight {bp.highlight_label(d)} ({s.highlight_size} genes, {s.reference_in_module}/"
          f"{s.reference_size} of the original module)")
    print("enrichment (panel vs genome):\n" + bp.caption_numbers(d).to_string())
    ts = d["tile_scores"]
    print("coherence median by niche:\n" + ts.groupby("niche").coherence.median().round(2).to_string())
    print("highlight genes:", ", ".join(d["modules"].query("highlight").gene))


if __name__ == "__main__":
    main()
