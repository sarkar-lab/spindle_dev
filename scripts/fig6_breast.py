"""Fig. 6 -- breast cancer case study, 180 mm.

A  niche map + niche x cell-type composition, tile-level ARI / NMI (E10-lite, seed 73)
B  the DCIS-core niche next to the annotated DCIS_1 / DCIS_2 cells
C  module heatmap of the selected node (template ii)
D  MSigDB Hallmark enrichment of the highlighted module, panel background (template iii)
E  module co-variation map (template iv)
F  gene-signature hit maps: per-cell score, Spindle top-10 tiles outlined (E9-lite, seed 0)
G  signature score of the top-10 tiles vs all other training tiles, 5 signatures (E9-lite, seed 0)
H  composition JSD, Spindle vs exact vs random top-k (E13, seeds 0-4)

CSV inputs only: results/biology/breast_cancer/, results/niche_concordance/,
results/gene_signature_search/breast/, results/composition_concordance/,
results/figure_data/tiles_seed73.csv.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import bio_panels as bp
import figstyle as fs

KEY = "breast_cancer"
SEED_MAP = 73        # niche maps, composition, ARI / NMI
E9_SEED = 0          # E9-lite maps and per-tile scores (the per-cell scores exist for seed 0)
E9_DIR = fs.RESULTS / "gene_signature_search" / "breast"
SIGNATURES = [("Luminal_Tumor_Core", "Luminal"), ("Basal_Myoepithelial", "Myoepithelial"),
              ("Proliferation_Signature", "Proliferation"), ("Endothelial_Vascular", "Endothelial"),
              ("Macrophage_Myeloid", "Macrophage")]
MAP_SIGS = [("Luminal_Tumor_Core", "Luminal / tumour core"), ("Basal_Myoepithelial", "Myoepithelial / basal")]
DCIS_COLORS = {"DCIS_1": "#EE7733", "DCIS_2": "#0077BB"}
JSD_STYLE = {  # method: (label, colour, marker, filled, x offset)
    "spindle": ("Spindle", fs.METHOD_COLORS["spindle"], "o", True, -0.09),
    "exact": ("Exact top-k", fs.METHOD_COLORS["brute_force"], "s", True, 0.09),
    "random_same_niche": ("Random, same niche", fs.MUTED, "^", False, 0.0),
    "random": ("Random", fs.REFERENCE_COLOR, "D", False, 0.0),
}
W, H = fs.DOUBLE, 206.0


def pretty_type(s):
    return s.replace("_", " ").replace("Myoepi ", "Myoepi. ").replace("Prolif Invasive", "Prolif. invasive")


def load_extra():
    comp = pd.read_csv(fs.RESULTS / "niche_concordance" / "breast_niche_composition.csv")
    ari = pd.read_csv(fs.RESULTS / "niche_concordance" / "breast_ari_nmi.csv")
    cells = pd.read_csv(E9_DIR / "Luminal_Tumor_Core_spatial_cells.csv")
    stats = pd.read_csv(E9_DIR / "signature_stats.csv")
    sig_tiles = pd.read_csv(bp.BIO / KEY / "signature_tile_scores.csv")
    jsd = pd.read_csv(fs.RESULTS / "composition_concordance" / "summary.csv")
    return comp[comp.seed == SEED_MAP], ari.set_index("seed"), cells, stats, sig_tiles, jsd


# ---------------------------------------------------------------- A
def panel_a(fig, d, comp, ari):
    t = d["tiles"]
    w = 46.0
    h = w * bp.tissue_aspect(t)
    ax = bp.ax_mm(fig, 4, 7, w, h)
    bp.niche_map(ax, d)
    fs.label_at(fig, 1.5, 3.5, "A")
    ax.legend(handles=bp.niche_handles(d), loc="upper left", bbox_to_anchor=(0, -0.01), ncol=4, handlelength=0.9,
              columnspacing=0.8)
    a = ari.loc[SEED_MAP]
    fig.text(4 / W, 1 - (7 + h + 8.5) / H, f"Tile level vs majority cell type: ARI {a.ari_tile_majority:.2f}, "
             f"NMI {a.nmi_tile_majority:.2f}", fontsize=fs.TICK_PT, va="top")

    types = sorted(comp.cell_type.unique())
    M = comp.pivot(index="cell_type", columns="niche", values="fraction").loc[types].fillna(0)
    pitch = 2.15
    hx, hy, hw = 81.0, 12.0, 4.4 * M.shape[1]
    hax = bp.ax_mm(fig, hx, hy, hw, pitch * len(types))
    im = hax.imshow(M.to_numpy(), cmap=fs.SEQUENTIAL, vmin=0, vmax=M.to_numpy().max(), aspect="auto",
                    interpolation="nearest")
    hax.set_yticks(range(len(types)))
    hax.set_yticklabels([pretty_type(s) for s in types], fontsize=fs.TICK_PT)
    hax.set_xticks(range(M.shape[1]))
    hax.set_xticklabels([str(j + 1) for j in M.columns], fontsize=fs.TICK_PT)
    hax.xaxis.tick_top()
    hax.tick_params(axis="x", length=0, pad=5.5)
    hax.tick_params(axis="y", length=0, pad=1.5)
    for sp in hax.spines.values():
        sp.set_visible(False)
    pal = fs.niche_palette(bp.n_niches(d))
    for j in range(M.shape[1]):
        hax.add_patch(Rectangle((j - 0.5, -0.95), 1, 0.35, color=pal[j], clip_on=False, lw=0))
    hax.set_title("Niche", fontsize=fs.TICK_PT, pad=11)
    cax = bp.ax_mm(fig, hx + hw + 1.5, hy + 8, 1.3, 18)
    cb = fig.colorbar(im, cax=cax, ticks=[0, 0.3, 0.6])
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(length=1.5, pad=1)
    cb.set_label("Fraction of niche's cells", fontsize=fs.TICK_PT, labelpad=2)


# ---------------------------------------------------------------- B
def panel_b(fig, d, comp, cells):
    t = d["tiles"]
    dcis_niche = int(comp[comp.cell_type.str.startswith("DCIS")].groupby("niche").fraction.sum().idxmax())
    w = 32.0
    h = w * bp.tissue_aspect(t)
    x0, y0 = 113.0, 10.0
    ax1 = bp.ax_mm(fig, x0, y0, w, h)
    bp.niche_map(ax1, d, only=dcis_niche)
    ax1.set_title(f"{bp.niche_label(dcis_niche)} (DCIS core)", fontsize=fs.TICK_PT, pad=2)
    ax2 = bp.ax_mm(fig, x0 + w + 2.5, y0, w, h)
    other = cells[~cells.cell_type.isin(DCIS_COLORS)]
    ax2.scatter(other.x, other.y, s=0.02, color=fs.LIGHT, lw=0, rasterized=True)
    for ct, col in DCIS_COLORS.items():
        c = cells[cells.cell_type == ct]
        ax2.scatter(c.x, c.y, s=0.04, color=col, lw=0, rasterized=True)
    bp.tissue_axes(ax2, t)
    ax2.set_title("Annotated DCIS cells", fontsize=fs.TICK_PT, pad=2)
    fs.label_at(fig, x0 - 3.5, 3.5, "B")
    handles = [Line2D([], [], marker="o", ls="none", color=c, ms=3, label=k.replace("_", " "))
               for k, c in DCIS_COLORS.items()]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, handletextpad=0.2)
    frac = comp[(comp.niche == dcis_niche) & comp.cell_type.str.startswith("DCIS")].fraction.sum()
    held = (comp[(comp.niche == dcis_niche) & comp.cell_type.str.startswith("DCIS")].n_cells.sum()
            / comp[comp.cell_type.str.startswith("DCIS")].n_cells.sum())
    ax1.text(0.5, -0.04, f"{frac:.0%} of its cells are DCIS;\nit holds {held:.0%} of all DCIS cells",
             transform=ax1.transAxes, ha="center", va="top", fontsize=fs.TICK_PT)
    return dcis_niche, frac, held


# ---------------------------------------------------------------- C, D, E
def panel_c(fig, d):
    m, _ = bp.module_order(d)
    side = 1.5 * len(m)
    x, y = 19.0, 63.0
    bp.module_heatmap(fig, d, x, y, side)
    s = d["sel"]
    fs.label_at(fig, 4.0, 55.5, "C")
    fig.text((x + side / 2) / W, 1 - 56 / H,
             f"Node centroid, {bp.node_label(d)} ({s.n_tiles_node} tile; score rank {s['rank']} of {s.n_nodes:,})",
             ha="center", va="top", fontsize=fs.TEXT_PT)
    return side


def panel_d(fig, d):
    ax = bp.ax_mm(fig, 150.0, 62.0, 26.0, 30.0)
    t = bp.enrichment(ax, d, n=7, width=26)
    fs.label_at(fig, 113.5, 55.5, "D")
    ax.set_title(f"{bp.LIB_LABEL[d['sel'].library]}, module {bp.highlight_label(d)}", fontsize=fs.TEXT_PT,
                 loc="right", pad=3)
    return t


def panel_e(fig, d):
    t = d["tiles"]
    w = 48.0
    h = w * bp.tissue_aspect(t)
    x, y = 125.0, 108.0
    ax = bp.ax_mm(fig, x, y, w, h)
    pc, _ = bp.module_map(ax, d)
    fs.label_at(fig, 113.5, 104.0, "E")
    ax.set_title(f"Module {bp.highlight_label(d)} co-variation", fontsize=fs.TEXT_PT, pad=2)
    cax = bp.ax_mm(fig, x + w - 22, y + h + 1.5, 22, 1.4)
    bp.map_colorbar(fig, pc, cax)


# ---------------------------------------------------------------- F, G, H
def panel_f(fig, cells_by_sig, sig_tiles):
    t = sig_tiles
    xs = [4.0, 47.0]
    w = 40.0
    asp = (t.y1.max() - t.y0.min()) / (t.x1.max() - t.x0.min())
    h = w * asp
    y = 156.0
    for x, (sig, title) in zip(xs, MAP_SIGS):
        c = cells_by_sig[sig]
        ax = bp.ax_mm(fig, x, y, w, h)
        lo, hi = np.percentile(c.score, [1, 99])
        sc = ax.scatter(c.x, c.y, c=c.score, s=0.03, cmap=fs.SEQUENTIAL, vmin=lo, vmax=hi, lw=0, rasterized=True)
        top = t[t[f"{sig}_top10"]]
        bp.rects(ax, top, facecolor="none", edgecolor=fs.QUERY_COLOR, lw=0.9, zorder=3)
        ax.set_aspect("equal")
        ax.set_xlim(t.x0.min(), t.x1.max())
        ax.set_ylim(t.y1.max(), t.y0.min())
        ax.set_axis_off()
        ax.set_title(title, fontsize=fs.TEXT_PT, pad=2)
        cax = bp.ax_mm(fig, x + w - 16, y + h + 1.5, 16, 1.4)
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
        cb.outline.set_linewidth(0.4)
        cb.ax.tick_params(length=1.5, pad=1)
        from matplotlib.ticker import MaxNLocator
        cb.locator = MaxNLocator(3)
        cb.update_ticks()
        cax.text(-0.05, 0.5, "Cell score", transform=cax.transAxes, ha="right", va="center", fontsize=fs.TICK_PT)
    fs.label_at(fig, 1.5, 149.5, "F")
    fig.legend(handles=[Rectangle((0, 0), 1, 1, fc="none", ec=fs.QUERY_COLOR, lw=0.9, label="Spindle top-10 tiles")],
               loc="upper left", bbox_to_anchor=(4 / W, 1 - (y + h + 5) / H), frameon=False)


def panel_g(fig, stats, sig_tiles):
    ax = bp.ax_mm(fig, 101.0, 160.0, 38.0, 25.0)
    st = stats[(stats.seed == E9_SEED) & (stats.K == 10) & (stats.method == "spindle")
               & (stats.construction == "coexpression")].set_index("signature")
    rng = np.random.default_rng(0)
    ymax = 0.0
    for i, (sig, _) in enumerate(SIGNATURES):
        v = sig_tiles[sig].to_numpy()
        top = sig_tiles[f"{sig}_top10"].to_numpy()
        bg = v[~top]
        z = lambda a: (a - bg.mean()) / bg.std()  # noqa: E731
        parts = ax.violinplot(z(bg), positions=[i], widths=0.75, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(fs.LIGHT)
            pc.set_edgecolor("none")
            pc.set_alpha(1)
        ymax = max(ymax, z(v[top]).max())
        ax.scatter(i + rng.uniform(-0.12, 0.12, top.sum()), z(v[top]), s=5, color=fs.QUERY_COLOR, lw=0, zorder=3)
        r = st.loc[sig]
        ax.text(i, 1.0, f"δ {r.cliffs_delta:.2f}", transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=fs.TICK_PT)
    ax.set_xticks(range(len(SIGNATURES)))
    ax.set_xticklabels([lab for _, lab in SIGNATURES], rotation=35, ha="right", fontsize=fs.TICK_PT)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Tile score (z vs background)")
    ax.axhline(0, color=fs.MUTED, lw=0.4, ls=":", zorder=0)
    ax.set_xlim(-0.6, len(SIGNATURES) - 0.4)
    ax.set_ylim(-2.5, ymax + 0.8)
    fs.label_at(fig, 91.0, 149.5, "G")
    handles = [Rectangle((0, 0), 1, 1, fc=fs.LIGHT, ec="none", label="Other training tiles"),
               Line2D([], [], marker="o", ls="none", color=fs.QUERY_COLOR, ms=2.5, label="Spindle top 10")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.05, -0.36), ncol=2, columnspacing=0.8)
    return st


def panel_h(fig, jsd):
    ax = bp.ax_mm(fig, 157.0, 160.0, 21.0, 25.0)
    s = jsd[jsd.seed != SEED_MAP]
    agg = s.groupby(["method", "k"]).mean_jsd.agg(["mean", "std"]).reset_index()
    ks = sorted(agg.k.unique())
    for meth, (lab, col, mk, filled, dx) in JSD_STYLE.items():
        a = agg[agg.method == meth].set_index("k").loc[ks]
        x = np.arange(len(ks)) + dx
        ax.errorbar(x, a["mean"], yerr=a["std"], color=col, marker=mk, ms=3, lw=0.8, elinewidth=0.6,
                    mfc=col if filled else "white", mec=col, label=lab)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_xlabel("k (tiles returned)")
    ax.set_ylabel("Composition JSD")
    ax.set_ylim(0, None)
    ax.set_xlim(-0.4, len(ks) - 0.6)
    fs.label_at(fig, 144.0, 149.5, "H")
    ax.legend(loc="upper right", bbox_to_anchor=(1.12, -0.36), ncol=1, handlelength=1.4)
    return agg


def main():
    d = bp.load(KEY)
    comp, ari, cells, stats, sig_tiles, jsd = load_extra()
    cells_by_sig = {sig: pd.read_csv(E9_DIR / f"{sig}_spatial_cells.csv") for sig, _ in MAP_SIGS}
    fig = fs.figure(W, H)
    panel_a(fig, d, comp, ari)
    dcis_niche, frac, held = panel_b(fig, d, comp, cells)
    panel_c(fig, d)
    terms = panel_d(fig, d)
    panel_e(fig, d)
    panel_f(fig, cells_by_sig, sig_tiles)
    st = panel_g(fig, stats, sig_tiles)
    agg = panel_h(fig, jsd)
    fs.save(fig, "fig6_breast")

    s = d["sel"]
    print(f"node: {bp.node_label(d)}, node {s.node}, rank {s['rank']}/{s.n_nodes}, {s.n_tiles_node} tile(s), "
          f"highlight {bp.highlight_label(d)} ({s.highlight_size} genes)")
    print(f"DCIS niche {dcis_niche + 1}: {frac:.3f} DCIS, holds {held:.3f} of DCIS cells")
    print("enrichment (panel vs genome):\n" + bp.caption_numbers(d).to_string())
    print("E9 seed-0 top-10:\n" + st[["n_genes", "cliffs_delta", "mwu_fdr_bh"]].to_string())
    print("E13 seeds 0-4:\n" + agg.to_string())
    ts = d["tile_scores"]
    print("coherence median by niche:\n" + ts.groupby("niche").coherence.median().round(2).to_string())


if __name__ == "__main__":
    main()
