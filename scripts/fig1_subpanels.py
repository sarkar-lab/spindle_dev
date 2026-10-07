"""Fig. 1: every sub-panel as its own borderless PDF, then the assembled figure.

Every part is drawn at its final printed size (mm below), so the assembler
places it at 100% and the 6-7 pt text stays 6-7 pt. Parts are written to
figures/pdf/panels/fig1_overview/ (PNG twins in figures/png/panels/fig1_overview/); the layout grid is BOXES / PLACE below.

Top row, index construction (breast production build, seed 73):
  a_tissue_tiles        cells + adaptive quadtree tiles, zoomed dense/sparse inset
  b_tile_covariance     one indexed tile's gene x gene correlation, ~40 genes
  c_umap_niches         UMAP of phi(Sigma_m), coloured by covariance niche
  c_spatial_niches      the same niche colours on the tissue
  d_raw_matrix          the example niche's mean correlation, original gene order
  d_blockdiag_matrix    the same after the consensus permutation, blocks outlined
  e_dag_sankey          first layers of that niche's DAG; node height = tiles
  e_dag_layered         the same layers, largest nodes only, uniform nodes
Bottom row, query:
  f_glyph_{tile,partial_panel,visium,gene_list}   the four query modes
  g_search_paths        every niche's DAG with the example query's best path
  g_rerank              Stage-2 exact re-rank of the Stage-1 pool
  h_query_hits          the example held-out query and its Spindle top 10
Shared keys: c_niche_legend, colorbar_corr.
figures/pdf/fig1_overview.pdf (+ figures/png/fig1_overview.png): the assembled figure, every part PDF placed at the
BOXES / PLACE coordinates below; fails loudly if a part leaves its box.

Example choices (rules, not hand picks; printed when run):
  * query: among the held-out queries whose Spindle top 10 equals the exact
    top 10, the one whose hits lie farthest from it (median distance);
  * niche for D/E: the niche of that query's top hit;
  * tile for B: the tile of that niche nearest its mean in phi-space.

Inputs: results/figure_data/fig1/ (scripts/extract_fig1_data.py) and
results/cross_platform_tiers/cross_modal_cells.csv (Visium spots for the glyph).
"""

import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyBboxPatch, Patch, PathPatch, Rectangle
from matplotlib.path import Path

import figstyle as fs

D = fs.FIG_DATA / "fig1"
OUT = fs.PANEL_DIR / "fig1_overview"
MM = fs.MM

N_LAYERS = 6          # DAG layers drawn in E
N_LAYERED = 6         # largest nodes per layer in e_dag_layered
G_LAYERS = 5          # DAG layers drawn in g_search_paths
G_COVER, G_MIN, G_MAX = 0.3, 2, 6  # per layer: largest nodes holding 30% of its tiles, 2-6 of them (+ path node)
N_B_GENES = 40        # genes shown in B and the tile glyphs
K = 10                # retrieved tiles shown in H
VLIM = 0.5            # correlation colour limits (B, D, glyphs)
SIGNATURE = ["ESR1", "PGR", "ERBB2", "FOXA1", "GATA3", "KRT8", "EPCAM", "CDH1"]  # E9-lite luminal core

CELL_GREY = "#C4C4C4"
TILE_EDGE = "#7A8A9A"   # grey-blue tile outlines
Q = fs.QUERY_COLOR

# Assembly grid (mm from the canvas's top-left corner).
CANVAS = (180.0, 122.0)
HEADER = {"index": ("#DCE5EE", "#F4F7FA"), "query": ("#F9DCE7", "#FDF4F7")}  # header bar, box fill
BOXES = [  # letter, title, x0, x1, y0, y1, kind
    ("A", "Input tissue and tiles", 0, 36, 0, 62, "index"),
    ("B", "Tile covariances", 39, 68, 0, 62, "index"),
    ("C", "Covariance niches", 71, 107, 0, 62, "index"),
    ("D", "Shared blocks", 110, 137, 0, 62, "index"),
    ("E", "Layered index", 140, 180, 0, 62, "index"),
    ("F", "Query types", 0, 38, 68, 122, "query"),
    ("G", "Search every niche, then re-rank", 41, 129, 68, 122, "query"),
    ("H", "Retrieved tiles", 132, 180, 68, 122, "query"),
]
PLACE = [  # part, x, y (top-left)
    ("a_tissue_tiles", 1, 6.5), ("b_tile_covariance", 39.5, 6.5), ("colorbar_corr", 44, 36),
    ("c_umap_niches", 72, 6.5), ("c_spatial_niches", 72, 33), ("d_raw_matrix", 111, 6.5),
    ("d_blockdiag_matrix", 111, 35.5), ("e_dag_layered", 141, 6.5), ("c_niche_legend", 71, 63.5),
    ("f_glyph_tile", 2, 75), ("f_glyph_partial_panel", 20, 75), ("f_glyph_visium", 2, 94),
    ("f_glyph_gene_list", 17, 94), ("g_search_paths", 42, 76), ("g_rerank", 95, 78), ("h_query_hits", 134.5, 80),
]
BOX_TEXT = [  # x, y (top), text, pt
    (1, 56, "≤ 200 cells per tile", 7), (41, 51, r"$\Sigma_m = \mathrm{cov}(X_m)$", 7),
    (72, 58.5, r"$z_m = \varphi(\Sigma_m)$ → Leiden", 7), (141, 57.5, "layer = gene block\nnode = ε-cover cluster", 7),
    (43, 116, r"budget $\eta_j$ in each niche's DAG", 7),
    (95, 110, "Stage 1: ≤ 400 tiles per niche\nby path distance\n" r"Stage 2: exact $d_{\mathrm{LE}}$ re-rank", 7),
    (125, 33.5, r"$P_\sigma \Sigma P_\sigma^{\mathrm{T}}$", 6),
]
CHEVRONS = [(36, 33), (68, 33), (107, 33), (137, 33), (38, 97), (92, 93), (129, 97)]  # left x, centre y


# ---------------------------------------------------------------- io helpers
def part(w_mm, h_mm):
    return plt.figure(figsize=(w_mm * MM, h_mm * MM))


def save(fig, name):
    """Transparent, borderless PDF (vector; scatter/heatmap layers raster at 600 dpi) and its PNG twin."""
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", dpi=600, transparent=True)
    fig.savefig(fs.png_path(OUT / f"{name}.pdf"), dpi=fs.PNG_DPI, transparent=True)
    plt.close(fig)
    print(f"saved {(OUT / name).relative_to(fs.PROJECT_ROOT)}.pdf (+ .png)")


def ax_mm(fig, x, y, w, h):
    """Axes at (x, y) mm from the figure's bottom-left corner."""
    fw, fh = fig.get_size_inches() / MM
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh])


def load():
    d = {n: pd.read_csv(D / f"{n}.csv") for n in
         ("tiles", "cells", "genes", "niche_perm", "niche_blocks", "example_tiles", "dag_nodes", "dag_edges",
          "query_summary", "query_hits", "query_stage1", "query_paths")}
    t = d["tiles"]
    t["cx"], t["cy"] = (t.x0 + t.x1) / 2, (t.y0 + t.y1) / 2
    t["side"] = np.maximum(t.x1 - t.x0, t.y1 - t.y0)
    d["niches"] = sorted(t.loc[t.status == "train", "niche"].unique())
    from scanpy.pl import palettes  # niche label j -> default_20[j], as in the spot-level cluster plots
    d["colors"] = {j: palettes.default_20[int(j)] for j in d["niches"]}
    return d


def niche_label(j):
    return f"Niche {j + 1}"


def corr(cov):
    s = np.sqrt(np.diag(cov))
    return cov / np.outer(s, s)


def tissue_axes(ax, t, pad=0.0):
    ax.set_aspect("equal")
    ax.set_xlim(t.x0.min() - pad, t.x1.max() + pad)
    ax.set_ylim(t.y1.max() + pad, t.y0.min() - pad)  # image orientation, as in Figs 2 and 5
    ax.set_axis_off()


def rects(ax, rows, **kw):
    ax.add_collection(PatchCollection([Rectangle((r.x0, r.y0), r.x1 - r.x0, r.y1 - r.y0)
                                       for r in rows.itertuples()], **kw))


def tissue_aspect(t):
    return (t.y1.max() - t.y0.min()) / (t.x1.max() - t.x0.min())


# ---------------------------------------------------------------- example choices
def pick_query(d):
    h, t = d["query_hits"], d["tiles"].set_index("tile_id")
    ok = []
    for q, g in h.groupby("query_idx"):
        sp = g[g.method == "spindle"].sort_values("rank")
        ex = g[g.method == "exact"].sort_values("rank")
        if set(sp.tile_id) == set(ex.tile_id):
            qt = int(sp.query_tile_id.iloc[0])
            dist = np.hypot(t.loc[sp.tile_id, "cx"] - t.at[qt, "cx"], t.loc[sp.tile_id, "cy"] - t.at[qt, "cy"])
            ok.append((float(np.median(dist)), q))
    spread, q = max(ok)
    n_q = h.query_idx.nunique()
    print(f"example query {q}: Spindle top {K} = exact top {K} for {len(ok)}/{n_q} queries; "
          f"median hit distance {spread:.0f} px")
    return q, len(ok), n_q


def pick_inset(t, frac=0.2, n_min=10):
    """Square window (frac of tissue width) with the largest max/min tile-side ratio among >= n_min tiles."""
    kept = t
    side = frac * (t.x1.max() - t.x0.min())
    best = None
    for cx in np.linspace(t.x0.min() + side / 2, t.x1.max() - side / 2, 30):
        for cy in np.linspace(t.y0.min() + side / 2, t.y1.max() - side / 2, 30):
            x0, y0 = cx - side / 2, cy - side / 2
            inside = kept[(kept.x0 >= x0) & (kept.x1 <= x0 + side) & (kept.y0 >= y0) & (kept.y1 <= y0 + side)]
            if len(inside) < n_min:
                continue
            score = (inside.side.max() / inside.side.min(), len(inside))
            if best is None or score > best[0]:
                best = (score, (x0, y0, side))
    (ratio, n), win = best
    print(f"inset: {n} tiles, largest/smallest tile side {ratio:.0f}x")
    return win


def b_genes(d, niche):
    """The leading whole blocks of the niche's permutation whose gene count is closest to N_B_GENES,
    shown in original (top-variance) order so the block structure is only revealed in D."""
    perm = d["niche_perm"].query("niche == @niche").sort_values("position").gene_idx.to_numpy()
    ends = d["niche_blocks"].query("niche == @niche").sort_values("block").end.to_numpy()
    end = ends[np.argmin(abs(ends - N_B_GENES))]
    return np.sort(perm[:end])


# ---------------------------------------------------------------- A
def panel_a(d, example_tile):
    t, cells = d["tiles"], d["cells"]
    W, inset = 34.0, 20.0
    hmap = W * tissue_aspect(t)
    H = hmap + 3.0 + inset
    fig = part(W, H)
    ax = ax_mm(fig, 0, H - hmap, W, hmap)
    ax.scatter(cells.x, cells.y, s=0.02, c=CELL_GREY, lw=0, rasterized=True, zorder=1)
    rects(ax, t, facecolor="none", edgecolor=TILE_EDGE, lw=0.12, zorder=2)
    tissue_axes(ax, t)
    x0, y0, s = pick_inset(t)
    ax.add_patch(Rectangle((x0, y0), s, s, fill=False, ec=fs.INK, lw=0.6, zorder=4))
    ex = t.set_index("tile_id").loc[example_tile]
    ax.add_patch(Rectangle((ex.x0, ex.y0), ex.x1 - ex.x0, ex.y1 - ex.y0, fill=False,
                           ec=d["colors"][int(ex.niche)], lw=0.9, zorder=5))

    ai = ax_mm(fig, (W - inset) / 2, 0, inset, inset)
    c = cells[(cells.x >= x0) & (cells.x <= x0 + s) & (cells.y >= y0) & (cells.y <= y0 + s)]
    ai.scatter(c.x, c.y, s=0.35, c=CELL_GREY, lw=0, rasterized=True, zorder=1)
    rects(ai, t, facecolor="none", edgecolor=TILE_EDGE, lw=0.45, zorder=2)
    ai.set_xlim(x0, x0 + s)
    ai.set_ylim(y0 + s, y0)
    ai.set_aspect("equal")
    ai.set_xticks([])
    ai.set_yticks([])
    for sp in ai.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.6)
    for xa, xb in ((x0, x0), (x0 + s, x0 + s)):
        fig.add_artist(ConnectionPatch((xa, y0 + s), (xb, y0), "data", "data", axesA=ax, axesB=ai,
                                       color=fs.MUTED, lw=0.4))
    save(fig, "a_tissue_tiles")
    return {"W": W, "H": H}


# ---------------------------------------------------------------- B and matrix helpers
def heat(ax, R):
    im = ax.imshow(R, cmap=fs.DIVERGING, vmin=-VLIM, vmax=VLIM, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
    return im


def panel_b(R40):
    W = H = 28.0
    m, off = 21.0, 1.4  # matrix side, stack offset (mm)
    fig = part(W, H)
    for k in (2, 1):  # Sigma_1 ... Sigma_M: two sheets behind the shown matrix
        a = ax_mm(fig, 3.0 + k * off, 3.0 + k * off, m, m)
        a.set_xticks([])
        a.set_yticks([])
        a.set_facecolor("white")
        for sp in a.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.5)
            sp.set_color(fs.MUTED)
    ax = ax_mm(fig, 3.0, 3.0, m, m)
    heat(ax, R40)
    # ax.set_xlabel(f"{len(R40)} genes", fontsize=fs.TICK_PT, labelpad=1.5)
    # ax.set_ylabel("Genes", fontsize=fs.TICK_PT, labelpad=1.5)
    save(fig, "b_tile_covariance")
    return {"W": W, "H": H}


def colorbar():
    fig = part(19, 9)
    ax = ax_mm(fig, 2.5, 4.0, 14, 1.4)
    cb = fig.colorbar(plt.cm.ScalarMappable(Normalize(-VLIM, VLIM), fs.DIVERGING), cax=ax, orientation="horizontal")
    cb.set_ticks([-VLIM, 0, VLIM])
    cb.set_ticklabels([f"−{VLIM:g}", "0", f"{VLIM:g}"])
    cb.outline.set_linewidth(0.4)
    ax.tick_params(length=1.5, width=0.4, pad=1)
    ax.set_title("Correlation", fontsize=fs.TICK_PT, pad=1.5)
    save(fig, "colorbar_corr")


# ---------------------------------------------------------------- C
def panel_c(d):
    t = d["tiles"]
    tr = t[t.status == "train"]
    col = tr.niche.map(d["colors"])
    W = 34.0
    fig = part(W, 26.0)
    ax = ax_mm(fig, 3.0, 3.0, W - 4.0, 22.0)
    ax.scatter(tr.umap1, tr.umap2, s=0.9, c=col, lw=0, rasterized=True)
    for j in d["niches"]:
        s = tr[tr.niche == j]
        ax.text(s.umap1.median(), s.umap2.median(), str(j + 1), fontsize=fs.TICK_PT, ha="center", va="center",
                bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec=fs.INK, lw=0.4))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP 1", fontsize=fs.TICK_PT, labelpad=1)
    ax.set_ylabel("UMAP 2", fontsize=fs.TICK_PT, labelpad=1)
    save(fig, "c_umap_niches")

    h = W * tissue_aspect(t)
    fig = part(W, h)
    ax = ax_mm(fig, 0, 0, W, h)
    cells = d["cells"]
    x, y = cells.x.to_numpy(), cells.y.to_numpy()
    cell_col = np.full(len(cells), None, dtype=object)  # each cell takes its tile's niche colour (held out: grey)
    for r in t.itertuples():
        inside = (cell_col == None) & (x >= r.x0) & (x <= r.x1) & (y >= r.y0) & (y <= r.y1)  # noqa: E711
        cell_col[inside] = d["colors"][r.niche] if r.status == "train" else fs.LIGHT
    keep = cell_col != None  # noqa: E711
    ax.scatter(x[keep], y[keep], s=0.02, c=list(cell_col[keep]), lw=0, alpha=0.8, rasterized=True, zorder=1)
    rects(ax, t, facecolor="none", edgecolor="black", lw=0.08, zorder=2)
    tissue_axes(ax, t)
    save(fig, "c_spatial_niches")

    fig = part(50.0, 4.0)
    fig.legend(handles=[Patch(color=d["colors"][j], label=niche_label(j)) for j in d["niches"]],
               loc="center", ncol=len(d["niches"]), handlelength=0.8, handleheight=0.8, columnspacing=0.8,
               handletextpad=0.3, borderaxespad=0)
    save(fig, "c_niche_legend")
    return {"W": W, "H_umap": 26.0, "H_spatial": h}


# ---------------------------------------------------------------- D
def panel_d(d, niche):
    R = pd.read_csv(D / f"niche{niche}_mean_corr.csv").to_numpy()
    perm = d["niche_perm"].query("niche == @niche").sort_values("position").gene_idx.to_numpy()
    blocks = d["niche_blocks"].query("niche == @niche").sort_values("block")
    m = 24.0
    for name, M, runs in (("d_raw_matrix", R, None), ("d_blockdiag_matrix", R[np.ix_(perm, perm)], blocks)):
        fig = part(m + 1.0, m + 1.0)
        ax = ax_mm(fig, 0.5, 0.5, m, m)
        heat(ax, M)
        if runs is not None:  # plotting.plot_blocks_from_fcluster's outline rule, restyled
            for r in runs.itertuples():
                ax.add_patch(Rectangle((r.start - 0.5, r.start - 0.5), r.end - r.start, r.end - r.start,
                                       fill=False, ec=fs.INK, lw=0.45))
        save(fig, name)
    return {"n_genes": len(R), "n_blocks": len(blocks)}


# ---------------------------------------------------------------- E: DAG layouts
def ordered_layers(nodes, edges, layers):
    """Node order per layer: layer 1 by order_id, then each layer by the tile-weighted mean parent position."""
    pos, order = {}, {}
    for l in layers:
        nl = nodes[nodes.layer == l]
        if l == layers[0]:
            ids = nl.sort_values("order_id").node.tolist()
        else:
            e = edges[(edges.layer == l - 1) & edges.dst.isin(nl.node)]
            bary = (e.assign(w=e.src.map(pos) * e.n_tiles).groupby("dst").w.sum()
                    / e.groupby("dst").n_tiles.sum())
            ids = sorted(nl.node, key=lambda v: (bary.get(v, np.inf), v))
        order[l] = ids
        pos.update({v: i for i, v in enumerate(ids)})
    return order, pos


def band(x0, y0t, y0b, x1, y1t, y1b):
    xm = (x0 + x1) / 2
    verts = [(x0, y0t), (xm, y0t), (xm, y1t), (x1, y1t), (x1, y1b), (xm, y1b), (xm, y0b), (x0, y0b), (x0, y0t)]
    codes = [Path.MOVETO] + [Path.CURVE4] * 3 + [Path.LINETO] + [Path.CURVE4] * 3 + [Path.CLOSEPOLY]
    return Path(verts, codes)


def layer_labels(ax, d, niche, layers, y):
    blocks = d["niche_blocks"].query("niche == @niche").set_index("block")
    """Layer number, and below it (muted) the genes in that layer's block."""
    for l in layers:
        b = blocks.loc[l]
        t = ax.text(l, y, f"ℓ{l + 1}", ha="center", va="top", fontsize=fs.TICK_PT)
        ax.annotate(f"{b.end - b.start}", (0.5, 0), xycoords=t, xytext=(0, -1), textcoords="offset points",
                    ha="center", va="top", fontsize=fs.TICK_PT, color=fs.MUTED)


def panel_e_sankey(d, niche, W=38.0, H=50.0):
    nodes = d["dag_nodes"].query("niche == @niche and layer < @N_LAYERS")
    edges = d["dag_edges"].query("niche == @niche and layer < @N_LAYERS - 1")
    layers = list(range(N_LAYERS))
    order, _ = ordered_layers(nodes, edges, layers)
    size = nodes.set_index("node").n_tiles
    total = nodes.groupby("layer").n_tiles.sum().max()
    gap = 0.15 * total / max(len(v) for v in order.values())
    span = {}
    for l in layers:
        y = 0.0
        for v in order[l]:
            span[v] = (y, y + size[v])
            y += size[v] + gap
    wn = 0.10  # node bar width, in layer units
    fig = part(W, H)
    ax = ax_mm(fig, 1.0, 7.0, W - 2.0, H - 7.5)
    col = d["colors"][niche]
    out_next = {v: span[v][0] for v in span}
    in_next = {v: span[v][0] for v in span}
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
            s = slots[(r.src, r.dst)]
            s += [in_next[r.dst], in_next[r.dst] + r.n_tiles]
            in_next[r.dst] += r.n_tiles
        for (u, v), (a0, a1, b0, b1) in slots.items():
            patches.append(PathPatch(band(l + wn / 2, a0, a1, l + 1 - wn / 2, b0, b1)))
    ax.add_collection(PatchCollection(patches, facecolor=col, edgecolor="none", alpha=0.45, zorder=1))
    bars = [Rectangle((l - wn / 2, span[v][0]), wn, size[v]) for l in layers for v in order[l]]
    ax.add_collection(PatchCollection(bars, facecolor=col, edgecolor="none", zorder=2))
    ymax = max(s[1] for s in span.values())
    ax.set_xlim(-0.3, layers[-1] + 0.3)
    ax.set_ylim(ymax, 0)
    ax.set_axis_off()
    ax.set_clip_on(False)
    layer_labels(ax, d, niche, layers, ymax + 0.02 * ymax)
    save(fig, "e_dag_sankey")
    n_per_layer = nodes.groupby("layer").size()
    return {"nodes_per_layer": (int(n_per_layer.min()), int(n_per_layer.max())), "tiles": int(total)}


def top_nodes(nodes, n, force=()):
    keep = set(nodes.sort_values(["layer", "n_tiles"], ascending=[True, False]).groupby("layer").head(n).node)
    return keep | set(force)


def panel_e_layered(d, niche, W=38.0, H=50.0):
    nodes = d["dag_nodes"].query("niche == @niche and layer < @N_LAYERS")
    edges = d["dag_edges"].query("niche == @niche and layer < @N_LAYERS - 1")
    layers = list(range(N_LAYERS))
    keep = top_nodes(nodes, N_LAYERED)
    nk, ek = nodes[nodes.node.isin(keep)], edges[edges.src.isin(keep) & edges.dst.isin(keep)]
    order, pos = ordered_layers(nk, ek, layers)
    fig = part(W, H)
    ax = ax_mm(fig, 1.0, 7.0, W - 2.0, H - 7.5)
    xy = {v: (l, pos[v]) for l in layers for v in order[l]}
    ax.add_collection(LineCollection([[xy[r.src], xy[r.dst]] for r in ek.itertuples()], colors=fs.MUTED,
                                     lw=0.35, zorder=1))
    pts = np.array(list(xy.values()))
    ax.scatter(pts[:, 0], pts[:, 1], s=16, c=d["colors"][niche], edgecolors="white", linewidths=0.5, zorder=2)
    for l in layers:
        hidden = int((nodes.layer == l).sum()) - len(order[l])
        ax.text(l, N_LAYERED - 0.35, f"+{hidden}", ha="center", va="top", fontsize=fs.TICK_PT, color=fs.MUTED)
    ax.set_xlim(-0.4, layers[-1] + 0.4)
    ax.set_ylim(N_LAYERED + 0.35, -0.6)
    ax.set_axis_off()
    layer_labels(ax, d, niche, layers, N_LAYERED + 0.45)
    save(fig, "e_dag_layered")


# ---------------------------------------------------------------- F: query-mode glyphs
def glyph_frame(fig, label, W):
    fig.text(0.5, 0.0, label, ha="center", va="bottom", fontsize=fs.TICK_PT)


def glyphs(R40, d):
    G, W, H = 13.0, 15.0, 17.0
    # full tile
    fig = part(W, H)
    heat(ax_mm(fig, (W - G) / 2, 3.2, G, G), R40)
    glyph_frame(fig, "Tile", W)
    save(fig, "f_glyph_tile")
    # partial gene panel: one contiguous sub-panel of the same matrix
    fig = part(W, H)
    ax = ax_mm(fig, (W - G) / 2, 3.2, G, G)
    heat(ax, R40).set_alpha(0.22)
    a, b = 9, 23
    sub = np.full_like(R40, np.nan)
    sub[a:b, a:b] = R40[a:b, a:b]
    ax.imshow(sub, cmap=fs.DIVERGING, vmin=-VLIM, vmax=VLIM, interpolation="nearest")
    ax.add_patch(Rectangle((a - 0.5, a - 0.5), b - a, b - a, fill=False, ec=Q, lw=0.9))
    glyph_frame(fig, "Gene panel", W)
    save(fig, "f_glyph_partial_panel")
    # Visium spots: a real window of the Visium section (Fig. 5)
    v = pd.read_csv(fs.RESULTS / "cross_platform_tiers" / "cross_modal_cells.csv").query("modality == 'Visium'")
    cx, cy = v.x.median(), v.y.median()
    pitch = np.median(np.sort(np.hypot(v.x.to_numpy()[:200, None] - v.x.to_numpy()[None], v.y.to_numpy()[:200, None]
                                       - v.y.to_numpy()[None]), axis=1)[:, 1])
    half = 3.5 * pitch
    w = v[(abs(v.x - cx) <= half) & (abs(v.y - cy) <= half)]
    fig = part(W, H)
    ax = ax_mm(fig, (W - G) / 2, 3.2, G, G)
    ax.scatter(w.x, w.y, s=9, c=Q, alpha=0.55, lw=0)
    ax.set_xlim(cx - half - pitch / 2, cx + half + pitch / 2)
    ax.set_ylim(cy + half + pitch / 2, cy - half - pitch / 2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.5)
    glyph_frame(fig, "Visium spots", W)
    save(fig, "f_glyph_visium")
    # gene list
    genes = set(d["genes"].gene)
    missing = [g for g in SIGNATURE if g not in genes]
    if missing:
        raise ValueError(f"signature genes not in the breast panel: {missing}")
    Wl = 21.0  # two columns of 6 pt gene names need ~19 mm
    fig = part(Wl, H)
    ax = ax_mm(fig, 1.0, 3.2, Wl - 2.0, G)
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="white", ec=Q, lw=0.7, transform=ax.transAxes))
    half_n = (len(SIGNATURE) + 1) // 2
    for i, g in enumerate(SIGNATURE):
        ax.text(0.12 + 0.44 * (i // half_n), 0.8 - 0.2 * (i % half_n), g, fontsize=fs.TICK_PT, va="center",
                transform=ax.transAxes)
    glyph_frame(fig, "Gene list", W)
    save(fig, "f_glyph_gene_list")


# ---------------------------------------------------------------- G: all-niche search
def snake(a, b):
    """S-shaped cubic Bezier from a to b, leaving and entering horizontally."""
    xm = (a[0] + b[0]) / 2
    return Path([a, (xm, a[1]), (xm, b[1]), b], [Path.MOVETO] + [Path.CURVE4] * 3)


def panel_g_paths(d, q, W=50.0, H=38.0):
    """Each niche's DAG, first G_LAYERS layers: per layer the largest nodes holding G_COVER of its tiles
    (G_MIN..G_MAX of them, plus the path node), spread and centred in the strip; edges as S-curves."""
    paths = d["query_paths"].query("query_idx == @q and kind == 'best'")
    nodes_all, edges_all = d["dag_nodes"], d["dag_edges"]
    layers = list(range(G_LAYERS))
    shown = {}
    for j in d["niches"]:
        p = paths[(paths.niche == j) & (paths.layer < G_LAYERS)].sort_values("layer")
        nodes = nodes_all[(nodes_all.niche == j) & (nodes_all.layer < G_LAYERS)]
        keep = set(p.node)
        for l in layers:
            nl = nodes[nodes.layer == l].sort_values("n_tiles", ascending=False)
            n = int(np.searchsorted(nl.n_tiles.cumsum() / nl.n_tiles.sum(), G_COVER) + 1)
            keep |= set(nl.node[:int(np.clip(n, G_MIN, G_MAX))])
        e = edges_all[(edges_all.niche == j) & edges_all.src.isin(keep) & edges_all.dst.isin(keep)]
        order, pos = ordered_layers(nodes[nodes.node.isin(keep)], e, layers)
        shown[j] = p, e, order, pos
    rows = max(len(o[l]) for _, _, o, _ in shown.values() for l in layers)
    fig = part(W, H)
    strip = (H - 1.0) / len(d["niches"])
    for i, j in enumerate(d["niches"]):
        p, e, order, pos = shown[j]
        ax = ax_mm(fig, 9.0, H - (i + 1) * strip, W - 10.0, strip - 1.2)
        xy = {}
        for l in layers:  # height set by node count, centred, alternate layers nudged so rows rarely align
            n = len(order[l])
            span = min(rows - 1.6, 1.5 * (n - 1))
            y0 = (rows - 1 - span) / 2 + (0.3 if l % 2 else -0.3)
            xy.update({v: (l, y0 + (span * pos[v] / (n - 1) if n > 1 else 0)) for v in order[l]})
        ax.add_collection(PatchCollection([PathPatch(snake(xy[r.src], xy[r.dst])) for r in e.itertuples()],
                                          facecolor="none", edgecolor=d["colors"][j], alpha=0.45, lw=0.3, zorder=1))
        pts = np.array(list(xy.values()))
        ax.scatter(pts[:, 0], pts[:, 1], s=3.5, c=d["colors"][j], lw=0, zorder=2)
        for a, b in zip(p.node[:-1], p.node[1:]):
            ax.add_patch(PathPatch(snake(xy[a], xy[b]), fc="none", ec=Q, lw=1.0, capstyle="round", zorder=3))
        pp = np.array([xy[v] for v in p.node])
        ax.scatter(pp[:, 0], pp[:, 1], s=5, c=Q, lw=0, zorder=4)
        ax.set_xlim(-0.3, G_LAYERS - 0.7)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_axis_off()
        yc = (H - (i + 0.5) * strip - 0.6) / H
        fig.add_artist(Line2D([0.3 / W, 1.6 / W], [yc] * 2, color=d["colors"][j], lw=2.2, solid_capstyle="butt"))
        fig.text(2.0 / W, yc, niche_label(j), fontsize=fs.TICK_PT, va="center", ha="left")
    save(fig, "g_search_paths")


def panel_g_rerank(d, q, W=34.0, H=30.0):
    s = d["query_stage1"].query("query_idx == @q").sort_values("stage2_rank")
    summ = d["query_summary"].set_index("query_idx").loc[q]
    fig = part(W, H)
    ax = ax_mm(fig, 9.0, 6.5, W - 10.5, H - 8.0)
    rest, top = s[s.stage2_rank > K], s[s.stage2_rank <= K]
    ax.scatter(rest.stage2_rank, rest.distance, s=1.2, c=fs.REFERENCE_COLOR, lw=0, rasterized=True)
    ax.scatter(top.stage2_rank, top.distance, s=5, c=Q, lw=0)
    ax.set_xscale("log")
    fs.plain_log(ax, "x", subs=(1.0,))
    ax.set_xlabel("Rank after exact re-rank", fontsize=fs.TICK_PT)
    ax.set_ylabel(r"$d_{\mathrm{LE}}$ to query", fontsize=fs.TICK_PT)
    ax.text(1.3, top.distance.max(), f"top {K}", color=Q, fontsize=fs.TICK_PT, va="bottom")
    save(fig, "g_rerank")
    return {"n_stage1": int(summ.n_stage1), "n_matched": int(summ.n_matched)}


# ---------------------------------------------------------------- H: retrieved tiles
def panel_h(d, q, W=43.0):
    t, cells = d["tiles"], d["cells"]
    ti = t.set_index("tile_id")
    hits = d["query_hits"].query("query_idx == @q and method == 'spindle' and rank <= @K")
    qt = int(hits.query_tile_id.iloc[0])
    h = W * tissue_aspect(t)
    fig = part(W, h)
    ax = ax_mm(fig, 0, 0, W, h)
    ax.scatter(cells.x, cells.y, s=0.02, c=CELL_GREY, lw=0, rasterized=True, zorder=1)
    rects(ax, t, facecolor="none", edgecolor=TILE_EDGE, lw=0.1, zorder=2)
    # hits: tile outline + open circle (small dense tiles would vanish otherwise); query: filled square
    rects(ax, ti.loc[hits.tile_id].reset_index(), facecolor="none", edgecolor=Q, lw=0.6, zorder=3)
    ax.scatter(ti.loc[hits.tile_id, "cx"], ti.loc[hits.tile_id, "cy"], s=18, facecolors="none", edgecolors=Q,
               lw=0.9, zorder=4)
    qrow = ti.loc[[qt]].reset_index()
    ax.scatter(qrow.cx, qrow.cy, s=26, marker="s", c=Q, edgecolors=fs.INK, lw=0.7, zorder=6)
    tissue_axes(ax, t)
    ax.annotate("Query", (qrow.cx.iloc[0], qrow.cy.iloc[0]), xytext=(5, 0), textcoords="offset points",
                va="center", fontsize=fs.TICK_PT, zorder=7)  # no path effects: they turn PDF text into outlines
    save(fig, "h_query_hits")
    return {"query_tile": qt, "hit_niches": hits.niche.value_counts().to_dict()}


def assemble():
    """Fig. 1 itself: the canvas (boxes, headers, chevrons, in-box text) drawn as a PDF, then every part
    PDF overlaid at its brief coordinates, so the result stays vector with real text. The PNG is built the
    same way from the canvas PNG and the part PNGs (never rasterised from the PDF)."""
    from matplotlib.patches import Polygon
    from PIL import Image
    from pypdf import PdfReader, PdfWriter, Transformation

    W, H = CANVAS
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_axis_off()
    for letter, title, x0, x1, y0, y1, kind in BOXES:
        head, body = HEADER[kind]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc=body, ec="none", zorder=0))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, 5, fc=head, ec="none", zorder=0))
        ax.text(x0 + 1.5, y0 + 2.6, letter, fontsize=fs.LETTER_PT, fontweight="bold", va="center")
        t = ax.text(x0 + 5, y0 + 2.6, title, fontsize=fs.TEXT_PT, va="center")
        fig.canvas.draw()
        if t.get_window_extent().transformed(ax.transData.inverted()).x1 > x1 - 0.5:
            raise ValueError(f"header of {letter} does not fit its box")
    for x, y in CHEVRONS:
        ax.add_patch(Polygon([(x, y - 3), (x + 1.6, y - 3), (x + 3, y), (x + 1.6, y + 3), (x, y + 3), (x + 1.4, y)],
                             fc="#C8C8C8", ec="none", zorder=1))
    for x, y, s, pt in BOX_TEXT:
        ax.text(x, y, s, fontsize=pt, va="top" if pt == 7 else "center", zorder=4)
    ax.annotate("", (123.5, 35.3), (123.5, 31.8), zorder=4,
                arrowprops=dict(arrowstyle="-|>", color=fs.MUTED, lw=0.6, mutation_scale=5))
    canvas_pdf = os.path.join(tempfile.mkdtemp(), "_canvas.pdf")
    fig.savefig(canvas_pdf)
    canvas = fs.render_png(fig)
    plt.close(fig)

    pt = 72 / 25.4
    page = PdfReader(canvas_pdf).pages[0]
    for name, x, y in PLACE:
        part_page = PdfReader(OUT / f"{name}.pdf").pages[0]
        w, h = float(part_page.mediabox.width) / pt, float(part_page.mediabox.height) / pt
        box = next(b for b in BOXES if b[2] <= x < b[3] and b[4] <= y < b[5]) if y < 62 or y > 68 else None
        if box and (x + w > box[3] + 0.01 or y + h > box[5] + 0.01):
            raise ValueError(f"{name} leaves box {box[0]}")
        page.merge_transformed_page(part_page, Transformation().translate(x * pt, (H - y - h) * pt))
        part_png = Image.open(fs.png_path(OUT / f"{name}.pdf")).convert("RGBA")
        px = fs.PNG_DPI / 25.4
        canvas.alpha_composite(part_png, (round(x * px), round(y * px)))
    out = PdfWriter()
    out.add_page(page)
    target = fs.PDF_DIR / "fig1_overview.pdf"
    with open(target, "wb") as fh:
        out.write(fh)
    os.remove(canvas_pdf)
    png = fs.png_path(target)
    canvas.convert("RGB").save(png, dpi=(fs.PNG_DPI, fs.PNG_DPI))
    print(f"saved {target.relative_to(fs.PROJECT_ROOT)} + {png.relative_to(fs.PROJECT_ROOT)}")


def main():
    d = load()
    q, n_ok, n_q = pick_query(d)
    top = d["query_hits"].query("query_idx == @q and method == 'spindle' and rank == 1").iloc[0]
    niche = int(top.niche)
    ex_tile = int(d["example_tiles"].set_index("niche").at[niche, "tile_id"])
    print(f"example niche {niche_label(niche)} (top hit tile {int(top.tile_id)}); B tile {ex_tile}")

    cov = pd.read_csv(D / f"tile_cov_{ex_tile}.csv").to_numpy()
    idx = b_genes(d, niche)
    R40 = corr(cov)[np.ix_(idx, idx)]

    a = panel_a(d, ex_tile)
    panel_b(R40)
    colorbar()
    c = panel_c(d)
    dd = panel_d(d, niche)
    e = panel_e_sankey(d, niche)
    panel_e_layered(d, niche)
    glyphs(R40, d)
    panel_g_paths(d, q)
    panel_g_rerank(d, q)
    h = panel_h(d, q)
    t = d["tiles"]
    print(f"numbers: {len(d['cells'])} cells, {len(t)} tiles ({(t.status == 'train').sum()} indexed, "
          f"{(t.status == 'heldout').sum()} held out), {len(d['niches'])} niches; B shows {len(idx)} genes; "
          f"D {dd['n_genes']} genes, {dd['n_blocks']} blocks; E {e['nodes_per_layer']} nodes per layer, "
          f"{e['tiles']} tiles; query tile {h['query_tile']}; hits by niche {h['hit_niches']}; A height {a['H']:.1f} mm, "
          f"C spatial height {c['H_spatial']:.1f} mm; example rule matched {n_ok}/{n_q} queries")
    assemble()


if __name__ == "__main__":
    main()
