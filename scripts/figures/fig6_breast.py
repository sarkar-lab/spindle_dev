"""Fig. 6 -- breast cancer case study, 180 mm.

A  every cell coloured by its tile's covariance niche + niche x cell type log2 enrichment over the section, with
   the section's cell-type fractions (seed 73)
B  the DCIS-core niche's tiles outlined over the annotated DCIS_1 / DCIS_2 cells
C  block programs: the DCIS-core niche's mean correlation with its blocks, and the stroma niche's in the
   DCIS-core niche's gene order and blocks; co-varying blocks (within-block |r| above size-matched random sets)
   are labelled with their three hub genes and their term (bio_panels.block_term: Hallmark, else a non-generic
   GO BP term; FDR < 0.05, panel background) with the block genes in that term
D  rewiring: ACTA2 (a marker of both myoepithelial cells and fibroblasts) and its block partners per niche;
   right, every gene's partner Jaccard between two niches vs the same niche across builds (line: median)
E  one whole-tile query (seed 73): a duct-rim tile and its exact top 10 with every cell by type (marked if also in
   the DAG's set of 10); composition of the query, the exact top 10, the DAG's set and the section
F  signature queries (template covariance, Spindle-Exact on S, seed 73): per-cell score with the top 10
   tiles and the 10 highest-scoring tiles outlined
G  signature queries, seeds 0-4: Cliff's delta of the top 10 (template vs tissue-mean control) and their
   co-variation on S (Spindle vs highest-scoring tiles vs background)

CSV inputs only (see bio_panels): results/{niche_concordance, block_programs, signature_queries,
query_example}/ and results/figure_data/tiles_seed73.csv.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import bio_panels as bp
import figstyle as fs

KEY = "breast_cancer"
STEM = bp.stem(KEY)
DCIS_COLORS = {"DCIS_1": "#EE7733", "DCIS_2": "#0077BB"}
STROMA_TYPE = "Stromal"
REWIRE_GENE = "ACTA2"
N_PARTNERS = 6
SIGNATURES = [("Luminal_Tumor_Core", "Luminal"), ("Basal_Myoepithelial", "Myoepithelial"),
              ("Proliferation_Signature", "Proliferation"), ("Endothelial_Vascular", "Endothelial"),
              ("Macrophage_Myeloid", "Macrophage")]
MAP_SIGS = [("Luminal_Tumor_Core", "Luminal / tumour core"), ("Basal_Myoepithelial", "Myoepithelial / basal")]
SPINDLE_BOX = dict(edgecolor=fs.QUERY_COLOR, lw=0.9)
EXPR_BOX = dict(edgecolor=fs.INK, lw=0.8, linestyle=(0, (1.2, 0.8)))
CELL_GROUPS = [("Myoepithelial", ["Myoepi_ACTA2+", "Myoepi_KRT15+"], "#EE7733"),
               ("DCIS", ["DCIS_1", "DCIS_2"], "#0077BB"),
               ("Invasive tumour", ["Invasive_Tumor", "Prolif_Invasive_Tumor"], "#009988"),
               ("Stromal", ["Stromal"], "#CCBB44"),
               ("Immune", ["B_Cells", "CD4+_T_Cells", "CD8+_T_Cells", "Macrophages_1", "Macrophages_2",
                           "IRF7+_DCs", "LAMP3+_DCs", "Mast_Cells"], "#AA3377")]
OTHER_COLOR = "#BBBBBB"
W, H = fs.DOUBLE, 226.0
Y_C, Y_D, Y_E = 63.0, 127.0, 171.0


def pretty_type(s):
    return s.replace("_", " ").replace("Myoepi ", "Myoepi. ").replace("Prolif Invasive", "Prolif. invasive")


def load():
    nc = bp.NICHE
    comp = pd.read_csv(nc / f"{KEY}_niche_composition.csv")
    types = pd.read_csv(nc / f"{KEY}_cell_types.csv").set_index("code").name
    cells = pd.read_csv(bp.SIGS / STEM / f"cells_seed{bp.SEED}.csv")
    cells["type_name"] = types.reindex(cells.cell_type).to_numpy()
    return comp, cells


# ---------------------------------------------------------------- A, B
def panel_a(fig, t, comp, cells, niche_of):
    w = 46.0
    h = w * bp.tissue_aspect(t)
    ax = bp.ax_mm(fig, 4, 7, w, h)
    bp.niche_cells(ax, cells, t, niche_of, s=0.09)
    fs.label_at(fig, 1.5, 3.5, "A")
    n = bp.n_niches(t)
    ax.legend(handles=bp.niche_handles(n), loc="upper left", bbox_to_anchor=(0, -0.01), ncol=4, handlelength=0.9,
              columnspacing=0.8)
    c = comp[comp.seed == bp.SEED]
    types = sorted(c.cell_type.unique())
    _, bax, cax, E, section = bp.enrichment_heatmap(fig, c, 79.0, 12.0, 2.0, 4.4, {s: pretty_type(s) for s in types}, n)
    fs.add_to_panel(fig, "A", bax, cax)
    return pd.DataFrame(E, index=types, columns=[f"niche{j + 1}" for j in range(n)]).assign(section=section.to_numpy())


def panel_b(fig, t, comp, cells):
    c = comp[comp.seed == bp.SEED]
    dcis = c[c.cell_type.str.startswith("DCIS")]
    dcis_niche = int(dcis.groupby("niche").fraction.sum().idxmax())
    w = 58.0
    h = w * bp.tissue_aspect(t)
    x0, y0 = 118.0, 9.0
    ax = bp.ax_mm(fig, x0, y0, w, h)
    other = cells[~cells.type_name.isin(DCIS_COLORS)]
    ax.scatter(other.x, other.y, s=0.02, color=fs.LIGHT, lw=0, rasterized=True)
    for ct, col in DCIS_COLORS.items():
        cc = cells[cells.type_name == ct]
        ax.scatter(cc.x, cc.y, s=0.04, color=col, lw=0, rasterized=True)
    niche_col = fs.niche_palette(bp.n_niches(t))[dcis_niche]
    bp.rects(ax, t[(t.status == "train") & (t.niche == dcis_niche)], facecolor="none", edgecolor=niche_col,
             lw=0.5, zorder=3)
    bp.tissue_axes(ax, t)
    fs.label_at(fig, x0 - 4.0, 3.5, "B")
    handles = [Rectangle((0, 0), 1, 1, fc="none", ec=niche_col, lw=0.8,
                         label=f"{bp.niche_label(dcis_niche)} tiles (DCIS core)")]
    handles += [Line2D([], [], marker="o", ls="none", color=col, ms=3, label=k.replace("_", " "))
                for k, col in DCIS_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3, handletextpad=0.3,
              columnspacing=0.8)
    frac = dcis[dcis.niche == dcis_niche].fraction.sum()
    held = dcis[dcis.niche == dcis_niche].n_cells.sum() / dcis.n_cells.sum()
    fig.text((x0 + w / 2) / W, 1 - (y0 + h + 6.0) / H,
             f"{frac:.0%} of {bp.niche_label(dcis_niche).lower()}'s cells are DCIS; it holds {held:.0%} of all DCIS cells",
             ha="center", va="top", fontsize=fs.TICK_PT)
    stroma_niche = int(c[c.cell_type == STROMA_TYPE].set_index("niche").fraction.idxmax())
    return dcis_niche, stroma_niche, frac, held


# ---------------------------------------------------------------- C
def label_blocks(b, k, R, key=KEY, seed=bp.SEED):
    """Co-varying blocks of niche k (within-block |r| above the random null), with hubs and term (block_term)."""
    enr = pd.read_csv(bp.BLOCKS / bp.stem(key) / f"seed{seed}_enrichment.csv")
    rows = []
    for r in b[b.niche == k].sort_values("block").itertuples():
        if r.p_within_vs_null >= 0.05:
            continue
        t = bp.block_term(r, enr)
        rows.append({"block": r.block, "genes": bp.block_hubs(R, r.gene_list), "term": t[0] if t else "",
                     "fdr": t[1] if t else np.nan, "term_genes": t[2] if t else [], "within": r.within_abs_r})
    return rows


def block_key(fig, key_ax, lab, max_genes=6, gap=0.45):
    """Numbered key: hub genes, then (if any) the term with its FDR and the block genes in it.

    Spacing is in text lines: one line for a block without a term, three with one, ``gap`` between blocks."""
    key_ax.set_axis_off()
    key_ax.set_xlim(0, 1)
    y = 0.0
    for i, L in enumerate(lab):
        key_ax.text(0, y, f"{i + 1}", fontsize=fs.TICK_PT, fontweight="bold", va="top")
        key_ax.text(0.07, y, ", ".join(L["genes"]), fontsize=fs.TICK_PT, fontstyle="italic", va="top")
        y += 1.0
        if L["term"]:
            g = L["term_genes"]
            genes = ", ".join(g[:max_genes]) + (", …" if len(g) > max_genes else "")
            key_ax.text(0.07, y, f"{L['term']} (FDR {L['fdr']:.1g})", fontsize=fs.TICK_PT - 1, color=fs.MUTED,
                        va="top")
            key_ax.text(0.07, y + 0.9, genes, fontsize=fs.TICK_PT - 1, color=fs.MUTED, fontstyle="italic", va="top")
            y += 1.8
        y += gap
    key_ax.set_ylim(max(y, 1.0), -0.1)
    return y


def panel_c(fig, b, dcis_niche, stroma_niche):
    order, runs = bp.niche_order(b, dcis_niche)
    R_d = bp.corr(KEY, dcis_niche, b)
    R_s = bp.corr(KEY, stroma_niche, b, order)
    side, x, y = 48.0, 6.0, Y_C + 6.0
    ax1 = bp.ax_mm(fig, x, y, side, side)
    im = bp.block_heatmap(ax1, R_d, runs)
    ax2 = bp.ax_mm(fig, x + side + 3.0, y, side, side)
    bp.block_heatmap(ax2, R_s, runs, color=fs.MUTED, lw=0.35)
    ax1.set_title(f"{bp.niche_label(dcis_niche)} (DCIS core), its blocks", fontsize=fs.TICK_PT, pad=2)
    ax2.set_title(f"{bp.niche_label(stroma_niche)} (stroma), in {bp.niche_label(dcis_niche).lower()}'s gene order and blocks",
                  fontsize=fs.TICK_PT, pad=2)
    fs.label_at(fig, 1.5, Y_C, "C")
    cax = bp.ax_mm(fig, x + side - 20, y + side + 2.0, 20, 1.4)
    bp.hcolorbar(fig, im, cax, "Correlation (niche mean)", ticks=[-bp.CORR_LIM, 0, bp.CORR_LIM])

    lab = label_blocks(b, dcis_niche, R_d)
    pitch = side / len(order)
    for i, L in enumerate(lab):
        s, e = runs[L["block"]]
        ymid = (s + e) / 2
        ax1.text(-1.5, ymid, str(i + 1), fontsize=fs.TICK_PT - 0.5, ha="right", va="center", color=fs.INK,
                 clip_on=False)
    key_ax = bp.ax_mm(fig, x + 2 * side + 7.0, y, 60.0, side + 6.0)
    block_key(fig, key_ax, lab)
    key_ax.set_title("Co-varying blocks: hub genes; term (FDR), its block genes", fontsize=fs.TICK_PT, loc="left",
                     pad=2)
    fs.add_to_panel(fig, "C", key_ax)
    return lab, pitch


# ---------------------------------------------------------------- D
def panel_d(fig, b, n):
    x, y = 6.0, Y_D + 11.0
    gene = REWIRE_GENE
    Rs = {k: bp.corr(KEY, k, b) for k in range(n)}
    member = {k: next(r.gene_list for r in b[b.niche == k].itertuples() if gene in r.gene_list) for k in range(n)}
    rw = pd.read_csv(bp.BLOCKS / STEM / f"seed{bp.SEED}_rewiring.csv")
    pair = rw[rw.gene == gene].sort_values("jaccard").iloc[0]  # the two niches where its partners differ most
    rows = []
    for k in (int(pair.niche_a), int(pair.niche_b)):
        part = [g for g in member[k] if g != gene]
        rows += [g for g in Rs[k].loc[gene, part].sort_values(ascending=False).index[:N_PARTNERS] if g not in rows]
    M = np.array([[Rs[k].loc[gene, g] for k in range(n)] for g in rows])
    inb = np.array([[g in member[k] for k in range(n)] for g in rows])
    cw, rh = 5.0, 2.3
    ax = bp.ax_mm(fig, x + 13, y, cw * n, rh * len(rows))
    im = ax.imshow(M, cmap=fs.DIVERGING, vmin=-bp.CORR_LIM, vmax=bp.CORR_LIM, aspect="auto", interpolation="nearest")
    yy, xx = np.nonzero(inb)
    ax.scatter(xx, yy, s=4, facecolor="none", edgecolor=fs.INK, lw=0.6)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontstyle="italic", fontsize=fs.TICK_PT)
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(k + 1) for k in range(n)], fontsize=fs.TICK_PT)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=4.5)
    pal = fs.niche_palette(n)
    for k in range(n):
        ax.add_patch(Rectangle((k - 0.5, -0.5 - 0.42), 1, 0.3, color=pal[k], clip_on=False, lw=0))
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("Niche", fontsize=fs.TICK_PT, pad=9)
    fig.text((x + 13 + cw * n / 2) / W, 1 - (y - 9.5) / H, f"r with $\\it{{{gene}}}$", ha="center",
             fontsize=fs.TEXT_PT, va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", mfc="none", mec=fs.INK, mew=0.6, ms=2.4,
                              label=f"in $\\it{{{gene}}}$'s block")],
              loc="upper left", bbox_to_anchor=(-0.6, -0.01), handletextpad=0.2)
    fs.label_at(fig, 1.5, Y_D, "D")

    # every gene: partner Jaccard across niches vs the same niche across builds
    nul = pd.read_csv(bp.BLOCKS / STEM / "rewiring_null.csv")
    same = nul.groupby(["gene", "niche"]).jaccard.mean().to_numpy()
    cross = rw.jaccard.to_numpy()
    bx = bp.ax_mm(fig, x + 13 + cw * n + 16, y - 4, 22, 26)
    parts = bx.violinplot([same, cross], positions=[0, 1], widths=0.75, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(fs.LIGHT)
        pc.set_edgecolor("none")
        pc.set_alpha(1)
    for i, v in enumerate([same, cross]):
        bx.plot([i - 0.22, i + 0.22], [np.median(v)] * 2, color=fs.INK, lw=1.0)
    bx.set_xticks([0, 1])
    bx.set_xticklabels(["Same niche,\nother\nbuilds", "Two\nniches"], fontsize=fs.TICK_PT)
    bx.tick_params(axis="x", length=0)
    bx.set_ylabel("Block-partner Jaccard")
    bx.set_ylim(0, 1)
    bx.set_xlim(-0.6, 1.6)
    fs.despine(bx)
    fs.add_to_panel(fig, "D", ax, bx)
    return rows, float(np.median(same)), float(np.median(cross)), member, pair


# ---------------------------------------------------------------- E, F
def panel_e(fig, t, cells, top):
    w = 41.0
    h = w * bp.tissue_aspect(t)
    y = Y_E + 7.0
    for x, (sig, title) in zip([4.0, 48.0], MAP_SIGS):
        tm = top[(top.signature == sig) & (top.seed == bp.SEED) & (top.construction == "coexpression")]
        ax = bp.ax_mm(fig, x, y, w, h)
        sc = bp.signature_map(ax, cells, sig, t, [(tm[(tm.method == "expression") & (tm["rank"] <= 10)], EXPR_BOX),
                                                  (tm[(tm.method == "spindle_exact") & (tm["rank"] <= 10)], SPINDLE_BOX)])
        ax.set_title(title, fontsize=fs.TEXT_PT, pad=2)
        cax = bp.ax_mm(fig, x + w - 14, y + h + 1.5, 14, 1.4)
        bp.hcolorbar(fig, sc, cax, "Cell score")
        fs.add_to_panel(fig, "F", ax, cax)
    fs.label_at(fig, 1.5, Y_E, "F")
    handles = [Rectangle((0, 0), 1, 1, fc="none", ec=fs.QUERY_COLOR, lw=0.9, label="Spindle top 10"),
               Rectangle((0, 0), 1, 1, fc="none", ec=fs.INK, lw=0.8, ls=(0, (1.2, 0.8)), label="Highest-scoring 10")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(4 / W, 1 - (y + h + 5) / H), frameon=False,
                     ncol=2)
    fs.add_to_panel(fig, "F", leg)


def panel_f(fig, stats):
    s = stats[(stats.K == 10) & stats.seed.isin(bp.FIG_SEEDS)]
    sigs = [k for k, _ in SIGNATURES]
    x0, y = 104.0, Y_E + 9.0
    ax1 = bp.ax_mm(fig, x0, y, 30, 22)
    ax2 = bp.ax_mm(fig, x0 + 45, y, 30, 22)
    style = {  # (construction, method): (label, colour, marker, filled, dx)
        ("coexpression", "spindle_exact"): ("Spindle top 10", fs.QUERY_COLOR, "o", True, -0.17),
        ("tissue_mean", "spindle_exact"): ("Tissue-mean query", fs.MUTED, "s", False, 0.17),
        ("coexpression", "expression"): ("Highest-scoring 10", fs.INK, "D", False, 0.0),
    }
    out = {}
    for (con, meth), (lab, col, mk, filled, dx) in style.items():
        g = s[(s.construction == con) & (s.method == meth)].groupby("signature")
        for ax, val in ((ax1, "cliffs_delta"), (ax2, "mean_r_s")):
            if ax is ax1 and meth == "expression":
                continue
            if ax is ax2 and con == "tissue_mean":
                continue
            a = g[val].agg(["mean", "std"]).reindex(sigs)
            ddx = dx if ax is ax1 else {"spindle_exact": -0.17, "expression": 0.17}[meth]
            ax.errorbar(np.arange(len(sigs)) + ddx, a["mean"], yerr=a["std"], ls="none", marker=mk, ms=3,
                        color=col, mfc=col if filled else "white", mec=col, elinewidth=0.6, label=lab)
            out[(con, meth, val)] = a
    bg = s[(s.construction == "coexpression") & (s.method == "spindle_exact")].groupby("signature").mean_r_s_background.mean().reindex(sigs)
    ax2.scatter(np.arange(len(sigs)), bg, marker="_", s=30, color=fs.REFERENCE_COLOR, lw=1.0, label="Background", zorder=1)
    out["bg"] = bg
    ax1.axhline(0, color=fs.MUTED, lw=0.4, ls=":")
    ax1.set_ylim(-1, 1)
    ax1.set_ylabel("Cliff's δ, signature score")
    ax2.set_ylabel("Co-variation on S (mean r)")
    ax2.set_ylim(0, None)
    for ax in (ax1, ax2):
        ax.set_xticks(range(len(sigs)))
        ax.set_xticklabels([lab for _, lab in SIGNATURES], rotation=40, ha="right", fontsize=fs.TICK_PT)
        ax.set_xlim(-0.6, len(sigs) - 0.4)
        fs.despine(ax)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    hh = dict(zip(l1 + l2, h1 + h2))
    leg = fig.legend(hh.values(), hh.keys(), loc="upper left", bbox_to_anchor=((x0 - 10) / W, 1 - (y + 36) / H), ncol=2,
                     frameon=False, columnspacing=0.9, handletextpad=0.2)
    fs.label_at(fig, 92.0, Y_E, "G")
    fs.add_to_panel(fig, "G", ax1, ax2, leg)
    return out


# ---------------------------------------------------------------- G
def group_of(type_names):
    out = np.full(len(type_names), "Other", dtype=object)
    for name, types, _ in CELL_GROUPS:
        out[np.isin(type_names, types)] = name
    return out


def panel_g(fig, cells, ex, comp_ex):
    """Query + exact top 10 as enlarged tiles with every cell, and composition bars."""
    x0, y = 92.0, Y_D + 7.0
    crops = ex[ex.method.isin(["query", "exact"])].copy()
    dag_ids = set(ex.loc[ex.method == "dag", "tile_id"])
    n_shared = int(crops[crops.method == "exact"].tile_id.isin(dag_ids).sum())
    size, gap, per_row = 9.4, 1.4, 6
    asp = float(np.median((crops.y1 - crops.y0) / (crops.x1 - crops.x0)))
    ch = size * asp  # crop height: tiles share the quadtree's aspect
    grp = group_of(cells.type_name.to_numpy())
    colors = {name: col for name, _, col in CELL_GROUPS} | {"Other": OTHER_COLOR}
    frame = {"query": fs.QUERY_COLOR, "exact": fs.TIER_COLORS["exact"]}
    for i, r in enumerate(crops.itertuples()):
        cx = x0 + (i % per_row) * (size + gap)
        cy = y + (i // per_row) * (ch + gap + 2.6)
        ax = bp.ax_mm(fig, cx, cy, size, ch)
        m = (cells.x >= r.x0) & (cells.x < r.x1) & (cells.y >= r.y0) & (cells.y < r.y1)
        ax.scatter(cells.x[m], cells.y[m], s=1.6, c=[colors[g] for g in grp[m.to_numpy()]], lw=0)
        ax.set_xlim(r.x0, r.x1)
        ax.set_ylim(r.y1, r.y0)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_edgecolor(frame[r.method])
            sp.set_linewidth(1.0)
        lab = "Query" if r.method == "query" else (f"{int(r.exact_rank)}" + (" •" if r.tile_id in dag_ids else ""))
        ax.text(0.5, -0.04, lab, transform=ax.transAxes, ha="center", va="top", fontsize=fs.TICK_PT,
                color=frame[r.method] if r.method != "exact" else fs.INK)
        fs.add_to_panel(fig, "E", ax)
    fs.label_at(fig, x0 - 3.5, Y_D, "E")
    t1 = fig.text(x0 / W, 1 - (Y_D + 2.5) / H,
                  f"Held-out duct-rim query and its exact top 10 by rank; • = also in the DAG's set ({n_shared} of 10)",
                  fontsize=fs.TICK_PT, va="center")
    fs.add_to_panel(fig, "E", t1)

    # composition bars
    bx = bp.ax_mm(fig, x0 + per_row * (size + gap) + 10, y, 13, 2 * ch + gap + 2.6)
    cols = [("query", "Query"), ("exact_top10", "Exact 10"), ("dag_set", "DAG set"), ("section", "Section")]
    gcomp = comp_ex.assign(group=group_of(comp_ex.index.to_numpy())).groupby("group").sum()
    bottom = np.zeros(len(cols))
    for name in [g for g, _, _ in CELL_GROUPS] + ["Other"]:
        v = gcomp.loc[name, [c for c, _ in cols]].to_numpy() if name in gcomp.index else np.zeros(len(cols))
        bx.bar(range(len(cols)), v, bottom=bottom, color=colors[name], width=0.75, label=name, lw=0)
        bottom += v
    bx.set_xticks(range(len(cols)))
    bx.set_xticklabels([lab for _, lab in cols], rotation=55, ha="right", fontsize=fs.TICK_PT)
    bx.tick_params(axis="x", length=0)
    bx.set_ylim(0, 1)
    bx.set_ylabel("Fraction of cells")
    fs.despine(bx)
    leg = bx.legend(loc="upper right", bbox_to_anchor=(1.1, -0.62), ncol=3, frameon=False, handlelength=0.8,
                    handletextpad=0.3, columnspacing=0.8)
    fs.add_to_panel(fig, "E", bx, leg)


def main():
    t = bp.tiles(KEY)
    comp, cells = load()
    niche_of = bp.cell_niches(cells, t)
    b = bp.blocks(KEY)
    stats = pd.read_csv(bp.SIGS / STEM / "stats.csv")
    top = pd.read_csv(bp.SIGS / STEM / "top_matches.csv")
    fig = fs.figure(W, H)
    enr_a = panel_a(fig, t, comp, cells, niche_of)
    dcis_niche, stroma_niche, frac, held = panel_b(fig, t, comp, cells)
    lab, _ = panel_c(fig, b, dcis_niche, stroma_niche)
    rows, j_same, j_cross, member, pair = panel_d(fig, b, bp.n_niches(t))
    ex = pd.read_csv(fs.RESULTS / "query_example" / "breast_example.csv")
    comp_ex = pd.read_csv(fs.RESULTS / "query_example" / "breast_example_comp.csv", index_col=0)
    panel_g(fig, cells, ex, comp_ex)
    panel_e(fig, t, cells, top)
    f = panel_f(fig, stats)
    fs.save(fig, "fig6_breast")

    print("A: log2 enrichment over section (seed 73):\n" + enr_a.round(2).to_string())
    print(f"DCIS niche {dcis_niche + 1}: {frac:.3f} DCIS, holds {held:.3f}; stroma niche {stroma_niche + 1}")
    print("labelled blocks:\n" + pd.DataFrame(lab).to_string())
    print(f"{REWIRE_GENE} block per niche:\n" + "\n".join(f"  {k + 1}: {' '.join(v)}" for k, v in member.items()))
    print(f"{REWIRE_GENE} niche pair {int(pair.niche_a) + 1} vs {int(pair.niche_b) + 1}: J {pair.jaccard:.2f}")
    print(f"partner Jaccard median: same niche other builds {j_same:.2f}, two niches {j_cross:.2f}")
    for k, v in f.items():
        print(k, "\n", v.round(3).to_string())


if __name__ == "__main__":
    main()
