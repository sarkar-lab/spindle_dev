"""Fig. S3 -- the layered index (DAG) of the Fig. 6 breast niche as a Sankey diagram, 180 mm.

Every layer of niche 3 (0-based niche 2) of the breast production build (seed 73): one column per
layer (gene block), one bar per node (epsilon-cover cluster) with height = its tiles, bands between
consecutive layers with width = shared tiles. The semantics of plotting.visualize_block_dag_sankey,
drawn in matplotlib (vector, house font) with the node order and band geometry of Fig. 1E
(fig1_subpanels.ordered_layers / band). The Fig. 6C node is marked.

CSV inputs only: results/figure_data/fig1/dag_{nodes,edges}.csv, niche_blocks.csv,
results/biology/breast_cancer/selection.csv.
"""

import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch, Rectangle

import bio_panels as bp
import fig1_subpanels as f1
import figstyle as fs

W, H = fs.DOUBLE, 110.0


def main():
    sel = pd.read_csv(bp.BIO / "breast_cancer" / "selection.csv").iloc[0]
    niche, mark_layer, mark_node = int(sel.niche), int(sel.layer), int(sel.node)
    nodes = pd.read_csv(f1.D / "dag_nodes.csv").query("niche == @niche")
    edges = pd.read_csv(f1.D / "dag_edges.csv").query("niche == @niche")
    blocks = pd.read_csv(f1.D / "niche_blocks.csv").query("niche == @niche").set_index("block")
    layers = sorted(nodes.layer.unique())
    order, _ = f1.ordered_layers(nodes, edges, layers)
    size = nodes.set_index("node").n_tiles
    total = nodes.groupby("layer").n_tiles.sum().max()
    gap = 0.12 * total / max(len(v) for v in order.values())
    span = {}
    for l in layers:
        y = 0.0
        for v in order[l]:
            span[v] = (y, y + size[v])
            y += size[v] + gap
    ymax = max(s[1] for s in span.values())

    fig = fs.figure(W, H)
    ax = bp.ax_mm(fig, 14, 7, W - 18, H - 25)
    col = fs.niche_palette(int(sel.n_niches))[niche]
    wn = 0.12
    out_next = {v: span[v][0] for v in span}
    in_next = dict(out_next)
    centre = {v: sum(span[v]) / 2 for v in span}
    patches = []
    for l in layers[:-1]:
        e = edges[edges.layer == l].copy()
        e["cu"], e["cv"] = e.src.map(centre), e.dst.map(centre)
        slots = {}
        for r in e.sort_values(["src", "cv"]).itertuples():
            slots[(r.src, r.dst)] = [out_next[r.src], out_next[r.src] + r.n_tiles]
            out_next[r.src] += r.n_tiles
        for r in e.sort_values(["dst", "cu"]).itertuples():
            slots[(r.src, r.dst)] += [in_next[r.dst], in_next[r.dst] + r.n_tiles]
            in_next[r.dst] += r.n_tiles
        for (u, v), (a0, a1, b0, b1) in slots.items():
            patches.append(PathPatch(f1.band(l + wn / 2, a0, a1, l + 1 - wn / 2, b0, b1)))
    ax.add_collection(PatchCollection(patches, facecolor=col, edgecolor="none", alpha=0.5, zorder=1))
    bars = [Rectangle((l - wn / 2, span[v][0]), wn, size[v]) for l in layers for v in order[l]]
    ax.add_collection(PatchCollection(bars, facecolor=col, edgecolor="none", zorder=2))

    # the Fig. 6C node
    y0, y1 = span[mark_node]
    ax.add_patch(Rectangle((mark_layer - wn, y0 - 2), 2 * wn, (y1 - y0) + 4, fill=False, ec=fs.INK, lw=0.8, zorder=3))
    ax.annotate("Fig. 6C node", (mark_layer + wn, (y0 + y1) / 2), xytext=(mark_layer + 0.6, -0.035 * ymax),
                fontsize=fs.TICK_PT, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=fs.INK, shrinkA=0, shrinkB=1))

    for l in layers:
        b = blocks.loc[l]
        ax.text(l, ymax * 1.02, f"ℓ{l + 1}", ha="center", va="top", fontsize=fs.TICK_PT)
        ax.text(l, ymax * 1.065, f"{b.end - b.start}", ha="center", va="top", fontsize=fs.TICK_PT, color=fs.MUTED)
        ax.text(l, ymax * 1.11, f"{len(order[l])}", ha="center", va="top", fontsize=fs.TICK_PT, color=fs.MUTED)
    ax.text(-0.75, ymax * 1.02, "Layer", ha="right", va="top", fontsize=fs.TICK_PT)
    ax.text(-0.75, ymax * 1.065, "Genes", ha="right", va="top", fontsize=fs.TICK_PT, color=fs.MUTED)
    ax.text(-0.75, ymax * 1.11, "Nodes", ha="right", va="top", fontsize=fs.TICK_PT, color=fs.MUTED)
    ax.set_xlim(-0.4, layers[-1] + 0.4)
    ax.set_ylim(ymax, 0)
    ax.set_axis_off()
    fs.save(fig, "figS3_dag_sankey")

    n_tiles = int(nodes.query("layer == 0").n_tiles.sum())
    npl = nodes.groupby("layer").size()
    print(f"niche {niche + 1}: {n_tiles} tiles, {len(layers)} layers, {len(nodes)} nodes ({npl.min()}-{npl.max()} per "
          f"layer), {len(edges)} edges; block sizes {blocks.end.sub(blocks.start).tolist()}")


if __name__ == "__main__":
    main()
