"""Fig. S10 -- block programs in the other tissues and their robustness, 180 mm.

A-F  one niche per remaining dataset (the niche with the most co-varying, enriched blocks; ties: more tiles):
     its mean correlation with its blocks, and its top four co-varying enriched blocks (hub genes, top term)
G    co-variation, all datasets, seeds 0-4: mean |r| within blocks, between a block and the niche's other
     genes, and within size-matched random gene sets
H    recurrence: each seed-73 block's best Jaccard match among the blocks of each other build (seeds 0-4),
     averaged over builds; one point per block

CSV / npy inputs only: results/block_programs/.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import bio_panels as bp
import fig6_breast as f6
import figstyle as fs

PANEL_KEYS = ["kidney_nondiseased", "lung_cancer", "pancreatic_cancer", "lymph_node", "lymph_node_5k", "brain_cancer"]
N_KEY = 4
W, H = fs.DOUBLE, 184.0
SIDE, ROW_H, COL_W = 30.0, 41.0, 90.0


def pick_niche(b):
    b = b.assign(good=(b.p_within_vs_null < 0.05) & (b.MSigDB_enriched | b.GO_enriched))
    g = b.groupby("niche").agg(good=("good", "sum"), n=("size", "size"))
    return int(g.sort_values(["good", "n"], ascending=False).index[0])


def block_panel(fig, key, letter, x, y):
    b = bp.blocks(key)
    k = pick_niche(b)
    order, runs = bp.niche_order(b, k)
    R = bp.corr(key, k, b)
    ax = bp.ax_mm(fig, x + 4, y + 6, SIDE, SIDE)
    im = bp.block_heatmap(ax, R, runs, lw=0.35)
    lab = [L for L in f6.label_blocks(b, k, R, key) if L["term"]]
    lab = sorted(lab, key=lambda L: L["fdr"])[:N_KEY]
    for i, L in enumerate(lab):
        s, e = runs[L["block"]]
        ax.text(-1.5, (s + e) / 2, str(i + 1), fontsize=fs.TICK_PT - 0.5, ha="right", va="center", clip_on=False)
    n_nich = b.niche.nunique()
    fig.text((x + 4) / W, 1 - (y + 3.5) / H,
             f"{fs.DATASET_SHORT[key]}: {bp.niche_label(k)} of {n_nich}, {len(runs)} blocks, {len(order)} genes",
             fontsize=fs.TICK_PT, va="bottom")
    kx = bp.ax_mm(fig, x + SIDE + 7, y + 6, COL_W - SIDE - 9, SIDE)
    kx.set_axis_off()
    kx.set_xlim(0, 1)
    kx.set_ylim(N_KEY - 0.2, -0.3)
    for i, L in enumerate(lab):
        kx.text(0, i, f"{i + 1}", fontsize=fs.TICK_PT, fontweight="bold", va="top")
        kx.text(0.07, i, ", ".join(L["genes"]), fontsize=fs.TICK_PT, fontstyle="italic", va="top")
        kx.text(0.07, i + 0.36, f"{bp.short_term(L['term'], 34)} (FDR {L['fdr']:.1g})", fontsize=fs.TICK_PT - 0.5,
                color=fs.MUTED, va="top")
    fs.label_at(fig, x, y, letter, panel=(ax, kx))
    return im, k, lab


def panel_g(fig, x, y, w, h):
    rows = []
    for key in fs.DATASET_ORDER:
        for s in bp.FIG_SEEDS:
            b = pd.read_csv(bp.BLOCKS / bp.stem(key) / f"seed{s}_blocks.csv")
            rows.append({"key": key, "within": b.within_abs_r.mean(), "between": b.between_abs_r.mean(),
                         "random": b.null_within_abs_r.mean()})
    d = pd.DataFrame(rows).groupby("key").agg(["mean", "std"]).reindex(fs.DATASET_ORDER)
    ax = bp.ax_mm(fig, x, y, w, h)
    style = [("within", "Within block", "o", True), ("between", "Block vs rest of niche", "s", False),
             ("random", "Random sets, same sizes", "D", False)]
    for j, (col, lab, mk, filled) in enumerate(style):
        xs = np.arange(len(d)) + (j - 1) * 0.22
        for i, key in enumerate(d.index):
            c = fs.DATASET_COLORS[key] if filled else fs.REFERENCE_COLOR
            ax.errorbar(xs[i], d.loc[key, (col, "mean")], yerr=d.loc[key, (col, "std")], marker=mk, ms=3,
                        color=c, mfc=c if filled else "white", mec=c, elinewidth=0.6, ls="none")
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([fs.DATASET_SHORT[k] for k in d.index], rotation=45, ha="right", fontsize=fs.TICK_PT)
    ax.set_ylabel("Mean |r| (niche mean)")
    ax.set_ylim(0, None)
    ax.set_xlim(-0.6, len(d) - 0.4)
    fs.despine(ax)
    ax.legend(handles=[Line2D([], [], marker=mk, ls="none", color=fs.INK if f else fs.REFERENCE_COLOR,
                              mfc=fs.INK if f else "white", ms=3, label=lab) for _, lab, mk, f in style],
              loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    return d


def panel_h(fig, x, y, w, h):
    ax = bp.ax_mm(fig, x, y, w, h)
    rng = np.random.default_rng(0)
    med = {}
    for i, key in enumerate(fs.DATASET_ORDER):
        r = pd.read_csv(bp.BLOCKS / bp.stem(key) / "recurrence.csv").groupby(["niche", "block"]).best_jaccard.mean()
        ax.scatter(i + rng.uniform(-0.25, 0.25, len(r)), r, s=1.5, color=fs.DATASET_COLORS[key], lw=0, alpha=0.6,
                   rasterized=True)
        ax.plot([i - 0.3, i + 0.3], [r.median()] * 2, color=fs.INK, lw=1.0)
        med[key] = float(r.median())
    ax.set_xticks(range(len(fs.DATASET_ORDER)))
    ax.set_xticklabels([fs.DATASET_SHORT[k] for k in fs.DATASET_ORDER], rotation=45, ha="right", fontsize=fs.TICK_PT)
    ax.set_ylabel("Best match in another build (Jaccard)")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(fs.DATASET_ORDER) - 0.4)
    fs.despine(ax)
    return med


def main():
    fig = fs.figure(W, H)
    picks = {}
    im = None
    for i, key in enumerate(PANEL_KEYS):
        x = 2.0 + (i % 2) * COL_W
        y = 2.0 + (i // 2) * ROW_H
        im, k, lab = block_panel(fig, key, "ABCDEF"[i], x, y)
        picks[key] = (k, lab)
    yb = 2.0 + 3 * ROW_H
    cax = bp.ax_mm(fig, W - 32, yb - 2.5, 22, 1.4)
    bp.hcolorbar(fig, im, cax, "Correlation (niche mean)", ticks=[-bp.CORR_LIM, 0, bp.CORR_LIM])
    yy = yb + 10.0
    d = panel_g(fig, 14.0, yy, 66.0, 34.0)
    med = panel_h(fig, 104.0, yy, 66.0, 34.0)
    fs.label_at(fig, 2.0, yb + 3.0, "G")
    fs.label_at(fig, 92.0, yb + 3.0, "H")
    fs.save(fig, "figS10_block_programs")

    for key, (k, lab) in picks.items():
        print(f"{key}: niche {k + 1}")
        for L in lab:
            print(f"   {', '.join(L['genes'])}: {L['term']} ({L['fdr']:.2g})")
    print("mean |r| (seeds 0-4):\n" + d.round(3).to_string())
    ratio = d[("within", "mean")] / d[("between", "mean")]
    print(f"within / between: {ratio.min():.1f}-{ratio.max():.1f}")
    print("recurrence median best Jaccard:", {k: round(v, 2) for k, v in med.items()})


if __name__ == "__main__":
    main()
