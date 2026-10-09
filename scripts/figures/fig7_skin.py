"""Fig. 7 -- skin melanoma worked example and all eight datasets, 180 mm.

A  every skin cell coloured by its tile's niche + niche x cell type log2 enrichment over the section, with the
   section's fractions (10x graph-based clusters, named by their top marker genes; seed 73)
B  block programs of the epidermis niche (the niche whose block holds the most Keratinocyte_Epidermis genes):
   its mean correlation with its blocks, and the melanoma niche's (most Melanocyte_Tumor genes in one block) in
   the epidermis niche's gene order and blocks; co-varying blocks labelled with hub genes and their term
   (bio_panels.block_term) with the block genes in that term
C  skin signature queries (template covariance, Spindle-Exact on S, seed 73): per-cell score, top 10 outlined
D  all datasets, seeds 0-4: fraction of blocks enriched (Hallmark, GO BP; FDR < 0.05, overlap >= 2) vs
   size-matched random panel gene sets
E  all datasets: block-partner Jaccard of each gene, same niche across builds vs two niches of one build (median)
F  all datasets, seeds 0-4: Cliff's delta of each signature's top 10 (template vs tissue-mean control)
G  all datasets, seeds 0-4: co-variation on S of Spindle's top 10 vs the 10 highest-scoring tiles

CSV inputs only (see bio_panels).
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import bio_panels as bp
import fig6_breast as f6
import figstyle as fs
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "benchmarks" / "common"))
from gene_signatures import TISSUE_MODULES  # noqa: E402

KEY = "skin_melanoma"
STEM = bp.stem(KEY)
ANCHOR_SIG = "Keratinocyte_Epidermis"
MAP_SIGS = [("Melanocyte_Tumor", "Melanocyte / tumour"), ("Keratinocyte_Epidermis", "Keratinocyte / epidermis"),
            ("Immune_Infiltrate", "Immune infiltrate")]
LIBS = [("MSigDB", "Hallmark", "o"), ("GO", "GO BP", "^")]
W, H = fs.DOUBLE, 224.0
Y_B, Y_C, Y_D = 68.0, 130.0, 174.0


def keys():
    return [k for k in fs.DATASET_ORDER if (bp.BLOCKS / bp.stem(k)).exists()]


# ---------------------------------------------------------------- A
def panel_a(fig, t, cells):
    comp = pd.read_csv(bp.NICHE / f"{KEY}_niche_composition.csv").query("seed == @bp.SEED")
    names = pd.read_csv(bp.NICHE / f"{KEY}_cell_types.csv")
    w = 80.0
    h = w * bp.tissue_aspect(t)
    ax = bp.ax_mm(fig, 4, 10, w, h)
    bp.niche_cells(ax, cells, t, bp.cell_niches(cells, t), s=0.12)
    n = bp.n_niches(t)
    ax.legend(handles=bp.niche_handles(n), loc="upper left", bbox_to_anchor=(0, -0.01), ncol=4, handlelength=0.9,
              columnspacing=0.8)
    fs.label_at(fig, 1.5, 3.5, "A")
    labels = {r.name: f"{r.code + 1}: {', '.join(r.markers.split()[:2])}" for r in names.itertuples()}
    order = [r.name for r in names.sort_values("code").itertuples()]
    _, bax, cax, E, section = bp.enrichment_heatmap(fig, comp, 128.0, 12.0, 1.75, 4.4, {k: labels[k] for k in order}, n)
    t1 = fig.text(108 / W, 1 - 7.5 / H, "Cluster: top markers", fontsize=fs.TICK_PT, ha="center", va="top")
    fs.add_to_panel(fig, "A", bax, cax, t1)
    return pd.DataFrame(E, index=[labels[k] for k in order],
                        columns=[f"niche{j + 1}" for j in range(n)]).assign(section=section.to_numpy())


# ---------------------------------------------------------------- B
def anchor_niche(b, sig=ANCHOR_SIG):
    ker = set(TISSUE_MODULES["skin"][sig]["genes"])
    b = b.assign(n_anchor=b.gene_list.apply(lambda g: len(ker & set(g))))
    r = b.sort_values(["n_anchor", "within_abs_r"], ascending=False).iloc[0]
    return int(r.niche)


def panel_b(fig, b):
    k = anchor_niche(b)
    order, runs = bp.niche_order(b, k)
    R = bp.corr(KEY, k, b)
    side, x, y = 48.0, 8.0, Y_B + 6.0
    ax = bp.ax_mm(fig, x, y, side, side)
    im = bp.block_heatmap(ax, R, runs)
    ax.set_title(f"{bp.niche_label(k)} (epidermis), its blocks", fontsize=fs.TICK_PT, pad=2)
    km = anchor_niche(b, "Melanocyte_Tumor")
    ax2 = bp.ax_mm(fig, x + side + 3.0, y, side, side)
    bp.block_heatmap(ax2, bp.corr(KEY, km, b, order), runs, color=fs.MUTED, lw=0.35)
    ax2.set_title(f"{bp.niche_label(km)} (melanoma), in {bp.niche_label(k).lower()}'s gene order and blocks",
                  fontsize=fs.TICK_PT, pad=2)
    fs.add_to_panel(fig, "B", ax2)
    fs.label_at(fig, 1.5, Y_B, "B")
    cax = bp.ax_mm(fig, x + side - 20, y + side + 2.0, 20, 1.4)
    bp.hcolorbar(fig, im, cax, "Correlation (niche mean)", ticks=[-bp.CORR_LIM, 0, bp.CORR_LIM])
    lab = f6.label_blocks(b, k, R, KEY)
    for i, L in enumerate(lab):
        s, e = runs[L["block"]]
        ax.text(-1.5, (s + e) / 2, str(i + 1), fontsize=fs.TICK_PT - 0.5, ha="right", va="center", clip_on=False)
    key_ax = bp.ax_mm(fig, x + 2 * side + 7.0, y, 62.0, side + 6.0)
    f6.block_key(fig, key_ax, lab)
    key_ax.set_title("Co-varying blocks: hub genes; term (FDR), its block genes", fontsize=fs.TICK_PT, loc="left",
                     pad=2)
    fs.add_to_panel(fig, "B", key_ax)
    return k, lab


# ---------------------------------------------------------------- C
def panel_c(fig, t, cells):
    top = pd.read_csv(bp.SIGS / STEM / "top_matches.csv")
    w, gap = 53.0, 6.0
    h = w * bp.tissue_aspect(t)
    y = Y_C + 6.0
    for i, (sig, title) in enumerate(MAP_SIGS):
        x = 4.0 + i * (w + gap)
        tm = top[(top.signature == sig) & (top.seed == bp.SEED) & (top.construction == "coexpression")]
        ax = bp.ax_mm(fig, x, y, w, h)
        sc = bp.signature_map(ax, cells, sig, t, [(tm[(tm.method == "expression") & (tm["rank"] <= 10)], f6.EXPR_BOX),
                                                  (tm[(tm.method == "spindle_exact") & (tm["rank"] <= 10)], f6.SPINDLE_BOX)],
                              s=0.05)
        ax.set_title(title, fontsize=fs.TEXT_PT, pad=2)
        cax = bp.ax_mm(fig, x + w - 14, y + h + 1.2, 14, 1.4)
        bp.hcolorbar(fig, sc, cax, "Cell score")
        fs.add_to_panel(fig, "C", ax, cax)
    fs.label_at(fig, 1.5, Y_C, "C")
    handles = [Rectangle((0, 0), 1, 1, fc="none", ec=fs.QUERY_COLOR, lw=0.9, label="Spindle top 10"),
               Rectangle((0, 0), 1, 1, fc="none", ec=fs.INK, lw=0.8, ls=(0, (1.2, 0.8)), label="Highest-scoring 10")]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(4 / W, 1 - (y + h + 6.5) / H), frameon=False,
                     ncol=2)
    fs.add_to_panel(fig, "C", leg)


# ---------------------------------------------------------------- D-G (all datasets)
def block_summary():
    rows = []
    for k in keys():
        d = bp.BLOCKS / bp.stem(k)
        for s in bp.FIG_SEEDS:
            b = pd.read_csv(d / f"seed{s}_blocks.csv")
            nl = pd.read_csv(d / f"seed{s}_null.csv")
            row = {"key": k, "seed": s, "n_blocks": len(b)}
            for short, _, _ in LIBS:
                lib = next(x for x in nl.library.unique() if x.startswith(short))
                exp = nl[nl.library == lib].set_index("size").enriched_frac
                row[short] = b[f"{short}_enriched"].mean()
                row[f"{short}_null"] = exp.reindex(b["size"]).mean()
            try:
                rw = pd.read_csv(d / f"seed{s}_rewiring.csv")
                row["cross"] = rw.jaccard.median()
            except pd.errors.EmptyDataError:  # one niche: nothing to rewire
                row["cross"] = np.nan
            rows.append(row)
        nul = pd.read_csv(d / "rewiring_null.csv")
        rows[-1]["same"] = nul.jaccard.median()
    return pd.DataFrame(rows)


def panel_d(fig, bs, x, y, w, h):
    ax = bp.ax_mm(fig, x, y, w, h)
    ks = keys()
    g = bs.groupby("key")
    for j, (short, lab, mk) in enumerate(LIBS):
        dx = -0.15 + 0.3 * j
        m, sd = g[short].mean().reindex(ks), g[short].std().reindex(ks)
        nm = g[f"{short}_null"].mean().reindex(ks)
        for i, k in enumerate(ks):
            ax.errorbar(i + dx, m[k], yerr=sd[k], marker=mk, ms=3, color=fs.DATASET_COLORS[k], elinewidth=0.6, ls="none")
            ax.plot(i + dx, nm[k], marker=mk, ms=3, mfc="white", mec=fs.REFERENCE_COLOR, mew=0.6, ls="none")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([fs.DATASET_SHORT[k] for k in ks], rotation=45, ha="right", fontsize=fs.TICK_PT)
    ax.set_ylabel("Blocks enriched (fraction)")
    ax.set_ylim(0, 0.5)
    ax.set_xlim(-0.6, len(ks) - 0.4)
    fs.despine(ax)
    handles = [Line2D([], [], marker=mk, ls="none", color=fs.INK, ms=3, label=lab) for _, lab, mk in LIBS]
    handles.append(Line2D([], [], marker="o", ls="none", mfc="white", mec=fs.REFERENCE_COLOR, ms=3,
                          label="Random sets, same sizes"))
    ax.legend(handles=handles, loc="upper right", ncol=1, handletextpad=0.2, borderaxespad=0.1)
    return ax


def panel_e(fig, bs, x, y, w, h):
    ax = bp.ax_mm(fig, x, y, w, h)
    ks = keys()
    same = bs.groupby("key").same.first().reindex(ks)
    cross = bs.groupby("key").cross.mean().reindex(ks)
    for i, k in enumerate(ks):
        c = fs.DATASET_COLORS[k]
        if np.isfinite(cross[k]):
            ax.plot([i, i], [cross[k], same[k]], color=fs.LIGHT, lw=1.2, zorder=1)
            ax.plot(i, cross[k], marker="o", ms=3.2, mfc="white", mec=c, mew=0.8, ls="none", zorder=2)
        ax.plot(i, same[k], marker="o", ms=3.2, color=c, ls="none", zorder=2)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([fs.DATASET_SHORT[k] for k in ks], rotation=45, ha="right", fontsize=fs.TICK_PT)
    ax.set_ylabel("Block-partner Jaccard (median)")
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xlim(-0.6, len(ks) - 0.4)
    fs.despine(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=fs.INK, ms=3, label="Same niche, other builds"),
                       Line2D([], [], marker="o", ls="none", mfc="white", mec=fs.INK, ms=3, label="Two niches")],
              loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    return same, cross


def sig_summary():
    rows = []
    for k in keys():
        f = bp.SIGS / bp.stem(k) / "stats.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f)
        s = s[(s.K == 10) & s.seed.isin(bp.FIG_SEEDS)]
        for sig, g in s.groupby("signature"):
            pick = lambda con, meth, col: g[(g.construction == con) & (g.method == meth)][col].mean()  # noqa: E731
            rows.append({"key": k, "signature": sig,
                         "delta": pick("coexpression", "spindle_exact", "cliffs_delta"),
                         "delta_control": pick("tissue_mean", "spindle_exact", "cliffs_delta"),
                         "r_spindle": pick("coexpression", "spindle_exact", "mean_r_s"),
                         "r_expression": pick("coexpression", "expression", "mean_r_s"),
                         "r_background": pick("coexpression", "spindle_exact", "mean_r_s_background")})
    return pd.DataFrame(rows)


def panel_f(fig, ss, x, y, w, h):
    ax = bp.ax_mm(fig, x, y, w, h)
    ks = [k for k in keys() if k in set(ss.key)]
    rng = np.random.default_rng(0)
    for i, k in enumerate(ks):
        g = ss[ss.key == k]
        jit = rng.uniform(-0.18, 0.18, len(g))
        ax.scatter(i + jit - 0.1, g.delta, s=7, color=fs.DATASET_COLORS[k], lw=0, zorder=2)
        ax.scatter(i + jit + 0.1, g.delta_control, s=7, facecolor="white", edgecolor=fs.REFERENCE_COLOR, lw=0.6, zorder=2)
    ax.axhline(0, color=fs.MUTED, lw=0.4, ls=":")
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([fs.DATASET_SHORT[k] for k in ks], rotation=45, ha="right", fontsize=fs.TICK_PT)
    ax.set_ylabel("Cliff's δ, signature score")
    ax.set_ylim(-1, 1.6)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlim(-0.6, len(ks) - 0.4)
    fs.despine(ax)
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", color=fs.INK, ms=3, label="Template query"),
                       Line2D([], [], marker="o", ls="none", mfc="white", mec=fs.REFERENCE_COLOR, ms=3,
                              label="Tissue-mean query")],
              loc="upper right", handletextpad=0.2, borderaxespad=0.1)


def panel_g(fig, ss, x, y, w):
    ax = bp.ax_mm(fig, x, y, w, w)
    for k, g in ss.groupby("key"):
        ax.scatter(g.r_expression, g.r_spindle, s=8, color=fs.DATASET_COLORS[k], marker=fs.DATASET_MARKERS[k], lw=0)
    hi = max(ss.r_expression.max(), ss.r_spindle.max()) * 1.08
    fs.identity_line(ax, min(0, ss[["r_expression", "r_spindle"]].min().min()), hi)
    ax.set_xlabel("Highest-scoring 10, r on S")
    ax.set_ylabel("Spindle top 10, r on S")
    fs.despine(ax)
    return ax


def main():
    t = bp.tiles(KEY)
    b = bp.blocks(KEY)
    fig = fs.figure(W, H)
    cells = pd.read_csv(bp.SIGS / STEM / f"cells_seed{bp.SEED}.csv")
    enr_a = panel_a(fig, t, cells)
    k, lab = panel_b(fig, b)
    panel_c(fig, t, cells)

    bs, ss = block_summary(), sig_summary()
    yy, hh = Y_D + 6.0, 24.0
    ax_d = panel_d(fig, bs, 14.0, yy, 34.0, hh)
    same, cross = panel_e(fig, bs, 61.0, yy, 34.0, hh)
    panel_f(fig, ss, 108.0, yy, 34.0, hh)
    panel_g(fig, ss, 151.0, yy, hh)
    for letter, xl in (("D", 1.5), ("E", 49.0), ("F", 96.0), ("G", 141.0)):
        fs.label_at(fig, xl, Y_D, letter)
    leg = fig.legend(handles=fs.dataset_handles(keys()), loc="upper center", bbox_to_anchor=(0.5, 1 - (yy + hh + 15) / H),
                     ncol=8, frameon=False, columnspacing=0.8, handletextpad=0.2)
    fs.save(fig, "fig7_skin")

    print("A: log2 enrichment over section (seed 73):\n" + enr_a.round(2).to_string())
    print(f"epidermis niche {k + 1}; labelled blocks:\n" + pd.DataFrame(lab).to_string())
    g = bs.groupby("key").agg(["mean", "std"])
    print("blocks (seeds 0-4):\n" + g[["MSigDB", "MSigDB_null", "GO", "GO_null"]].round(3).to_string())
    print("partner Jaccard same / cross:\n" + pd.DataFrame({"same": same, "cross": cross}).round(3).to_string())
    print("signatures:\n" + ss.round(3).to_string())
    print(f"Spindle r_S >= expression r_S: {(ss.r_spindle >= ss.r_expression - 1e-3).sum()} of {len(ss)}; "
          f"delta > 0.5: {(ss.delta > 0.5).sum()}; control delta < 0.2: {(ss.delta_control < 0.2).sum()}")


if __name__ == "__main__":
    main()
