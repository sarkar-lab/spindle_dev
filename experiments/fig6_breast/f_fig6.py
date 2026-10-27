"""Fig 6 atoms from results/fig6_breast/seed<s>/ (c_programs.py, c_queries.py; login node).

Usage: python experiments/fig6_breast/f_fig6.py
Niches are shown 1-based (niche k of the build = "Niche k + 1"). A-C (niches, ACTA2, programmes) come from the
whole-section build (results/fig6_breast/full/, user 27 Oct), the queries from seed 0.
Atoms (figures/fig6_breast/atoms/):
  map_niches, legend_niche  every cell coloured by its tile's niche (cells of dropped low-density tiles grey; rasterised)
  niche_enrich              log2 enrichment of the 19 annotated cell types in each niche over the section (clipped at
                            +/-3), and the section's fractions (bars, right)
  cbar_enrich               its colourbar
  corr_main, corr_contrast  niche-mean correlation of the chosen niche in its gene order, blocks outlined; the contrast
                            niche in the same order and blocks. Rule (approved 26 Oct): the niche with the most blocks
                            that both co-vary (null_p < 0.05) and are enriched (ties: more tiles); contrast = the
                            largest other niche. Block numbers and the key are pptx text (layout.py)
  cbar_corr                 their colourbar
  covariation, legend_cov   per niche, every block's mean |r| within the block, between the block and the niche's
                            other genes, and the mean within random gene sets of the same size
  acta2, cbar_acta2         r of ACTA2 with its block partners per niche: rows = the union over niches of the 6 block
                            partners with the largest r; circle = in ACTA2's block in that niche
  ex_query, ex_hit1..10     the example whole-tile query (rule in c_queries.py) and Spindle's top 10, cells by class
  legend_class              the 8 cell classes
  ex_comp                   class composition of the query, the top 10, a random 10 (mean of 200 draws), the section
  tv_base                   all held-out queries: composition difference (TV) to the top 10 and to a random 10
  sig_map_<list>, cbar_sig_<list>  the two signatures with the largest T delta (rule in c_queries.py): cells by
                            signature score, Spindle-partial's top 10 outlined
  sig_delta, sig_cov        all 5 signatures, template query, mean +/- s.d. over seeds 0-4: Cliff's delta of the top
                            10's score vs the other index tiles; co-variation on S of the top 10 (median) against the
                            typical index tile (median, grey bar)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import style as ps  # noqa: E402

FIG = "fig6_breast"
RES = ps.ROOT / "results" / FIG
S0 = RES / "seed0"  # queries (D, E and the spares): the seed-0 build
FULL = RES / "full"  # niches and programmes (A-C): every tile of the section indexed (user, 27 Oct)
SEEDS = [0, 1, 2, 3, 4]
CLASSES = [
    ("Invasive tumour", "#CC3311", ["Invasive_Tumor", "Prolif_Invasive_Tumor", "T_Cell_&_Tumor_Hybrid"]),
    ("DCIS", "#EE7733", ["DCIS_1", "DCIS_2"]),
    ("Myoepithelial", "#882255", ["Myoepi_ACTA2+", "Myoepi_KRT15+"]),
    ("Stromal", "#DDCC77", ["Stromal", "Stromal_&_T_Cell_Hybrid"]),
    ("Endothelial / perivascular", "#44AA99", ["Endothelial", "Perivascular-Like"]),
    ("Myeloid", "#0077BB", ["Macrophages_1", "Macrophages_2", "IRF7+_DCs", "LAMP3+_DCs", "Mast_Cells"]),
    ("T cells", "#33BBEE", ["CD4+_T_Cells", "CD8+_T_Cells"]),
    ("B cells", "#117733", ["B_Cells"]),
]
CLASS_COLORS = [c for _, c, _ in CLASSES]
SIG_LABELS = {"Luminal_Tumor_Core": "Luminal", "Macrophage_Myeloid": "Macrophage",
              "Basal_Myoepithelial": "Myoepithelial", "Endothelial_Vascular": "Endothelial",
              "Proliferation_Signature": "Proliferation"}
NO_TILE = "#E3E3E3"  # cells of the low-density tiles that tiling drops
SPINDLE = ps.TIER_COLORS["spindle"]
MAP_W = 60.0
CORR_LIM, ACTA2_LIM = 0.4, 0.6
plt_viridis = colormaps["viridis"]

blocks = pd.read_csv(FULL / "blocks.csv")
LAYOUT = json.loads((FULL / "layout.json").read_text())
GENES = np.array(LAYOUT["genes"])
NICHES = sorted(blocks.niche.unique())
NICHE_COLORS = ps.niche_palette(len(NICHES))


def chosen_niches():
    b = blocks.assign(both=blocks.covarying & blocks.enriched)
    g = b.groupby("niche").agg(both=("both", "sum"), tiles=("n_tiles", "first"))
    main = int(g.sort_values(["both", "tiles"], ascending=False).index[0])
    contrast = int(g.drop(main).sort_values("tiles", ascending=False).index[0])
    return main, contrast


def spatial(ax, x0, x1, y0, y1):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # image orientation
    ax.set_aspect("equal")
    ax.axis("off")


def section_extent(xy):
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    return x0, x1, y0, y1, MAP_W * (y1 - y0) / (x1 - x0)


# ---------------------------------------------------------------- A
def niche_map():
    z = np.load(FULL / "cells_niche.npz")
    xy, niche = z["xy"], z["niche"]
    x0, x1, y0, y1, h = section_extent(xy)
    fig, ax = ps.atom(MAP_W, h)
    col = np.array([NO_TILE] + NICHE_COLORS)[niche + 1]
    order = np.argsort(niche >= 0, kind="stable")  # cells without a tile underneath
    ax.scatter(xy[order, 0], xy[order, 1], s=0.05, c=col[order], lw=0, rasterized=True, antialiased=False)
    spatial(ax, x0, x1, y0, y1)
    ps.save(fig, FIG, "map_niches", source="results/fig6_breast/full/cells_niche.npz")
    handles = [Patch(fc=c, ec="none", label=f"Niche {k + 1}") for k, c in zip(NICHES, NICHE_COLORS)]
    ps.legend_atom(FIG, "legend_niche", handles, MAP_W, 4, ncol=len(handles), handlelength=0.8,
                   columnspacing=0.8, handletextpad=0.3, source="")


def niche_enrich():
    t = pd.read_csv(FULL / "niche_types.csv")
    types = [m for _, _, members in CLASSES for m in members]
    F = t.pivot(index="type", columns="niche", values="frac").loc[types, NICHES].to_numpy()
    sec = t.groupby("type")["section_frac"].first().loc[types].to_numpy()
    with np.errstate(divide="ignore"):
        E = np.clip(np.log2(F / sec[:, None]), -3, 3)
    w, h = 62.0, 56.0
    fig, ax = ps.atom(w, h, left=26.0, bottom=1.0, right=18.0, top=4.0)
    ax.imshow(E, cmap=ps.DIVERGING, vmin=-3, vmax=3, aspect="auto", interpolation="nearest")
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels([s.replace("_", " ").replace("Myoepi ", "Myoepithelial ") for s in types])
    ax.set_xticks(range(len(NICHES)))
    ax.set_xticklabels([str(k + 1) for k in NICHES])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for j, c in enumerate(NICHE_COLORS):
        ax.add_patch(Rectangle((j - 0.5, -0.5 - 0.45), 1, 0.4, fc=c, ec="none", clip_on=False))
    bx = fig.add_axes([(w - 16.0) / w, 1.0 / h, 13.0 / w, 1 - 5.0 / h], sharey=ax)
    bx.barh(range(len(types)), sec, height=0.7, color=ps.MUTED, lw=0)
    bx.set_xlim(0, 0.3)
    bx.set_xticks([0, 0.3])
    bx.xaxis.tick_top()
    bx.tick_params(axis="y", left=False, labelleft=False)
    bx.spines["bottom"].set_visible(False)
    bx.spines["top"].set_visible(True)
    ps.save(fig, FIG, "niche_enrich", source="results/fig6_breast/full/niche_types.csv")
    ps.colorbar_atom(FIG, "cbar_enrich", ps.DIVERGING, -3, 3, 30, 7, ticks=[-3, 0, 3], orientation="horizontal",
                     source="")


# ---------------------------------------------------------------- B
def corr_panels():
    main, contrast = chosen_niches()
    perm = np.array(LAYOUT[str(main)]["perm"])
    runs = LAYOUT[str(main)]["blocks"]
    R_main = np.load(FULL / f"corr_niche{main}.npy")
    R_c = np.load(FULL / f"corr_niche{contrast}.npy")
    pos_c = np.empty(len(perm), int)
    pos_c[np.array(LAYOUT[str(contrast)]["perm"])] = np.arange(len(perm))
    R_c = R_c[np.ix_(pos_c[perm], pos_c[perm])]  # contrast niche in the main niche's gene order
    cov = blocks[(blocks.niche == main)].set_index("block")
    for name, R in (("corr_main", R_main), ("corr_contrast", R_c)):
        fig, ax = ps.atom(48, 48)
        R = R.copy()
        np.fill_diagonal(R, np.nan)
        ax.imshow(R, cmap=ps.DIVERGING, vmin=-CORR_LIM, vmax=CORR_LIM, interpolation="nearest")
        for b, (s, e) in enumerate(runs):
            lw = 0.6 if cov.loc[b, "covarying"] else 0.3
            ax.add_patch(Rectangle((s - 0.5, s - 0.5), e - s, e - s, fill=False, ec=ps.INK, lw=lw))
        ax.set_xlim(-0.5, len(perm) - 0.5)
        ax.set_ylim(len(perm) - 0.5, -0.5)
        ax.axis("off")
        ps.save(fig, FIG, name, source=f"results/fig6_breast/full/corr_niche{main}.npy, corr_niche{contrast}.npy")
    ps.colorbar_atom(FIG, "cbar_corr", ps.DIVERGING, -CORR_LIM, CORR_LIM, 30, 7, ticks=[-CORR_LIM, 0, CORR_LIM],
                     orientation="horizontal", source="")


# ---------------------------------------------------------------- C
COV_KIND = [("within", "Within block", SPINDLE, "o"), ("between", "Block to rest of niche", ps.MUTED, "o"),
            ("null_mean", "Random gene sets", "#ABABAB", "s")]


def covariation():
    fig, ax = ps.atom(48, 42, left=10.0, bottom=6.0, right=1.0, top=1.5)
    rng = np.random.default_rng(0)
    w = 0.27
    for j, k in enumerate(NICHES):
        b = blocks[blocks.niche == k]
        for i, (col, _, c, m) in enumerate(COV_KIND):
            x = j + (i - 1) * w
            v = b[col].to_numpy()
            fc = c if i != 1 else "none"
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(v)), v, s=4, marker=m, fc=fc, ec=c, lw=0.5, zorder=2)
            ax.plot([x - 0.1, x + 0.1], [np.median(v)] * 2, color=ps.INK, lw=0.75, zorder=3)
    ax.set_xticks(range(len(NICHES)))
    ax.set_xticklabels([f"Niche {k + 1}" for k in NICHES])
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, None)
    ax.set_ylabel("Mean |r|")
    ps.save(fig, FIG, "covariation", source="results/fig6_breast/seed0/blocks.csv")
    handles = [Line2D([], [], ls="", marker=m, ms=3, mfc=c if i != 1 else "none", mec=c, mew=0.5, label=l)
               for i, (_, l, c, m) in enumerate(COV_KIND)]
    ps.legend_atom(FIG, "legend_cov", handles, 48, 7, ncol=2, handletextpad=0.2, columnspacing=0.8, source="")


# ---------------------------------------------------------------- D
def acta2():
    a = pd.read_csv(FULL / "acta2.csv")
    rows = []
    for k in NICHES:
        for g in a[(a.niche == k) & a.in_block].nlargest(6, "r").gene:
            if g not in rows:
                rows.append(g)
    r = a.pivot(index="gene", columns="niche", values="r").loc[rows, NICHES].to_numpy()
    inb = a.pivot(index="gene", columns="niche", values="in_block").loc[rows, NICHES].to_numpy().astype(bool)
    h = 3.0 + 3.5 * len(rows)
    fig, ax = ps.atom(38, h, left=13.0, bottom=0.5, right=1.0, top=2.5)
    ax.imshow(r, cmap=ps.DIVERGING, vmin=-ACTA2_LIM, vmax=ACTA2_LIM, aspect="auto", interpolation="nearest")
    yy, xx = np.nonzero(inb)
    ax.scatter(xx, yy, s=3, marker="o", fc="none", ec=ps.INK, lw=0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontstyle="italic")
    ax.set_xticks(range(len(NICHES)))
    ax.set_xticklabels([str(k + 1) for k in NICHES])
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ps.save(fig, FIG, "acta2", source="results/fig6_breast/full/acta2.csv")
    ps.colorbar_atom(FIG, "cbar_acta2", ps.DIVERGING, -ACTA2_LIM, ACTA2_LIM, 26, 7, ticks=[-ACTA2_LIM, 0, ACTA2_LIM],
                     orientation="horizontal", source="")


# ---------------------------------------------------------------- E
def example():
    z = np.load(S0 / "example_cells.npz")
    for k in range(len(z["tiles"])):
        x0, y0, x1, y1 = z[f"bbox{k}"]
        side = max(x1 - x0, y1 - y0) * 1.06
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        fig, ax = ps.atom(10, 10)
        ax.scatter(*z[f"xy{k}"].T, s=0.9, c=np.array(CLASS_COLORS)[z[f"cls{k}"]], lw=0, rasterized=True)
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, lw=0.9,
                               ec=ps.QUERY_COLOR if k == 0 else SPINDLE))
        spatial(ax, cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)
        ps.save(fig, FIG, "ex_query" if k == 0 else f"ex_hit{k}", source="results/fig6_breast/seed0/example_cells.npz")
    ps.legend_atom(FIG, "legend_class", [Patch(fc=c, ec="none", label=n) for n, c, _ in CLASSES], 66, 9, ncol=3,
                   handlelength=0.8, columnspacing=0.8, handletextpad=0.3, source="")
    ex = pd.read_csv(S0 / "example.csv").set_index("set")
    names = [n for n, _, _ in CLASSES]
    sets = [("query", "Query"), ("top10", "Top 10"), ("random10", "Random 10"), ("section", "Section")]
    fig, ax = ps.atom(26, 40, left=8.0, bottom=11.0, right=1.0, top=1.5)
    for i, (s, _) in enumerate(sets):
        bottom = 0.0
        for n, c in zip(names, CLASS_COLORS):
            v = ex.loc[s, n]
            ax.bar(i, v, 0.75, bottom=bottom, color=c, lw=0)
            bottom += v
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([l for _, l in sets], rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Fraction of cells")
    ps.save(fig, FIG, "ex_comp", source="results/fig6_breast/seed0/example.csv")

    q = pd.read_csv(S0 / "tile_queries.csv")
    fig, ax = ps.atom(24, 40, left=8.0, bottom=11.0, right=1.0, top=1.5)
    rng = np.random.default_rng(0)
    for i, (col, c) in enumerate((("tv_top", SPINDLE), ("tv_random", ps.MUTED))):
        v = q[col].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=1.5, c=c, lw=0, alpha=0.7, zorder=2)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=ps.INK, lw=0.75, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Top 10", "Random 10"], rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Composition difference")
    ps.save(fig, FIG, "tv_base", source="results/fig6_breast/seed0/tile_queries.csv")


# ---------------------------------------------------------------- F, G
def gene_lists():
    return pd.concat([pd.read_csv(RES / f"seed{s}" / "gene_lists.csv").assign(seed=s) for s in SEEDS])


def map_lists():
    g = pd.read_csv(S0 / "gene_lists.csv")
    return g[g["query"] == "T"].nlargest(2, "delta")["list"].tolist()


def signature_maps():
    z = np.load(S0 / "signature_cells.npz")
    xy = z["xy"]
    x0, x1, y0, y1, h = section_extent(xy)
    w = 44.0
    h = w * (y1 - y0) / (x1 - x0)
    cmap = ListedColormap(plt_viridis(np.linspace(0, 1, 10)))  # 10 steps keep the 1200 dpi PNG within budget
    for name in map_lists():
        s = z[f"score_{name}"].astype(np.float32)
        vmax = float(np.quantile(s, 0.995))
        fig, ax = ps.atom(w, h)
        o = np.argsort(s, kind="stable")
        ax.scatter(xy[o, 0], xy[o, 1], s=0.03, c=s[o], cmap=cmap, vmin=0, vmax=vmax, lw=0, rasterized=True,
                   antialiased=False)
        for bx0, by0, bx1, by1 in z[f"top_{name}"]:
            ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=ps.QUERY_COLOR, lw=0.8))
        spatial(ax, x0, x1, y0, y1)
        ps.save(fig, FIG, f"sig_map_{name}", source="results/fig6_breast/seed0/signature_cells.npz")
        ps.colorbar_atom(FIG, f"cbar_sig_{name}", cmap, 0, vmax, 24, 7, ticks=[0, np.floor(10 * vmax) / 10],
                         orientation="horizontal", source="")


def signature_summary():
    g = gene_lists()
    t = g[g["query"] == "T"]
    order = list(SIG_LABELS)
    m = t.groupby("list").mean(numeric_only=True).loc[order]
    s = t.groupby("list").std(numeric_only=True).loc[order]
    x = np.arange(len(order))
    labels = [SIG_LABELS[n] for n in order]

    fig, ax = ps.atom(29, 40, left=8.0, bottom=13.0, right=1.0, top=1.5)
    ax.axhline(0, color=ps.MUTED, lw=0.5, ls="--", zorder=1)
    ax.errorbar(x, m.delta, yerr=s.delta, fmt="o", ms=3, color=SPINDLE, elinewidth=0.6, capsize=0, zorder=2)
    ax.set_ylim(-1, 1)
    ax.set_yticks([-1, 0, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("Cliff's δ")
    ps.save(fig, FIG, "sig_delta", source="results/fig6_breast/seed<0-4>/gene_lists.csv")

    fig, ax = ps.atom(29, 40, left=8.0, bottom=13.0, right=1.0, top=1.5)
    for i in x:
        ax.plot([i - 0.25, i + 0.25], [m.cov_all.iloc[i]] * 2, color=ps.MUTED, lw=0.9, zorder=1)
    ax.errorbar(x, m.cov_top, yerr=s.cov_top, fmt="o", ms=3, color=SPINDLE, elinewidth=0.6, capsize=0, zorder=2)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylabel("Co-variation on S")
    ps.save(fig, FIG, "sig_cov", source="results/fig6_breast/seed<0-4>/gene_lists.csv")
    ps.legend_atom(FIG, "legend_sig", [Line2D([], [], ls="", marker="o", ms=3, color=SPINDLE, label="Top 10"),
                                       Line2D([], [], color=ps.MUTED, lw=0.9, label="Typical tile")],
                   40, 4, ncol=2, source="")


if __name__ == "__main__":
    niche_map()
    niche_enrich()
    corr_panels()
    covariation()
    acta2()
    example()
    signature_maps()
    signature_summary()
