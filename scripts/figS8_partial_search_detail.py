"""Fig. S8 -- Partial (gene-set) search in detail (companion of Fig 4).

Contiguous sets: m consecutive genes of one block in one niche's learned gene order (contiguous there only; other
niches order genes differently). Signatures: the tissue's curated gene lists. Random sets: m genes at random.

A  Overlap(5 %, c = 5 % of N) vs query size per dataset, contiguous sets: interval index, the same decomposition
   on the tiles' own logs (no index), imputation, padding (mean +/- s.d. over seeds 0-4).
B  the random-gene-set (filled) and curated-signature (hollow) families, per dataset (all sizes pooled).
C  c90 (the c at which the mean Overlap(5 %, c) reaches 0.9; % of N) per method, 16-gene contiguous sets; hollow at
   100 % = never reaches 0.9.
F  interval-index configurations: K_s = 32 s^0.5 node means per interval of s genes with intervals <= 16 genes
   (Fig 4), K_s = 32 s with <= 16, K_s = 32 s with every dyadic length; Overlap vs size (all datasets) and
   index size.
E  the error bound: per (query, niche), the largest |interval score - own-log score| over the niche's tiles,
   divided by sqrt(sum eps^2) (<= 1 always; first 20 queries per seed).
D  query structure: segments per gene in each niche's order (1 = every gene its own segment), weighted by
   niche size, for random sets and contiguous sets.

Inputs: results/partial_search_final/{by_size,storage,bound_summary,query_structure,summary}.csv, *_bound.csv
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import figstyle as fs
from fig4_partial_search import MAIN, RES, SIZES, main_cfg, pending, read

ORDER = fs.DATASET_ORDER
# (k_alpha, max_len, label, line style, marker)
CFGS = [(0.5, 16, "$K_s = 32\\sqrt{s}$, $s \\leq 16$ (Fig 4)", "-", "D"),
        (1.0, 16, "$K_s = 32s$, $s \\leq 16$", "--", "o"),
        (1.0, -1, "$K_s = 32s$, every $s$", ":", "s")]


def method_line(ax, d, m, x_of, **kw):
    r = d[d.method == m].set_index("size").reindex(SIZES)
    ok = r["overlap_0.05_mean"].notna()
    ax.errorbar(np.asarray(x_of)[ok.values], r["overlap_0.05_mean"][ok], yerr=r["overlap_0.05_std"][ok],
                color=kw.pop("color", None) or fs.PARTIAL_COLORS.get(m), ls=kw.pop("ls", None) or fs.PARTIAL_LS.get(m),
                marker=kw.pop("marker", None) or fs.PARTIAL_MARKERS.get(m), ms=2.2, lw=0.8, elinewidth=0.4, capsize=0, mew=0, **kw)


def size_axis(ax, label=True, ticks=SIZES):
    ax.set_xticks(np.log2(ticks), [str(s) for s in ticks])
    ax.set_ylim(0, 1.04)
    if label:
        ax.set_xlabel("Genes in the query")


def panel_per_dataset(fig, spec):
    d = read("by_size.csv")
    g = GridSpecFromSubplotSpec(1, 8, subplot_spec=spec, wspace=0.25)
    axes = [fig.add_subplot(g[i]) for i in range(8)]
    if d is None:
        return pending(axes[0], "per dataset"), axes
    d = main_cfg(d[(d.family == "program") & (d.c_target == 0.05)])
    for i, (ax, k) in enumerate(zip(axes, ORDER)):
        for m in MAIN:
            method_line(ax, d[d.dataset == k], m, np.log2(SIZES))
        size_axis(ax, label=(i == 0), ticks=[4, 16, 64])
        ax.set_title(fs.DATASET_SHORT[k], fontsize=fs.TICK_PT, pad=2)
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Overlap(5 %, 5 %)")
    return axes


def panel_families(ax):
    d = read("by_size.csv")
    if d is None:
        return pending(ax, "families")
    d = main_cfg(d[(d.dataset != "all") & (d.c_target == 0.05) & d.family.isin(["random", "signature"])])
    d = d.groupby(["dataset", "family", "method"])["overlap_0.05_mean"].mean().reset_index()
    keys = [k for k in ORDER if k in set(d.dataset)]
    for fam in ("random", "signature"):
        for m, o in zip(MAIN, np.linspace(-0.3, 0.3, len(MAIN))):
            r = d[(d.family == fam) & (d.method == m)].set_index("dataset").reindex(keys)
            ax.plot(np.arange(len(keys)) + o, r["overlap_0.05_mean"], ls="none", marker=fs.PARTIAL_MARKERS[m], ms=2.6,
                    color=fs.PARTIAL_COLORS[m], mfc=fs.PARTIAL_COLORS[m] if fam == "random" else "white", mew=0.6)
    ax.set_xticks(range(len(keys)), [fs.DATASET_SHORT[k] for k in keys], rotation=40, ha="right",
                  rotation_mode="anchor")
    ax.set_ylim(0, 1.04)
    ax.set_ylabel("Overlap(5 %, 5 %)")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", color=fs.MUTED, ms=2.6, label="filled: random sets"),
                       Line2D([], [], ls="none", marker="o", mfc="white", color=fs.MUTED, ms=2.6, mew=0.6,
                              label="hollow: signatures")],
              loc="center right", fontsize=fs.TICK_PT - 0.5)


def panel_configs(ax, ax_mb):
    d, st = read("by_size.csv"), read("storage.csv")
    if d is None or st is None:
        return pending(ax, "configs")
    d = d[(d.dataset == "all") & (d.family == "program") & (d.c_target == 0.05) & (d.method == "interval")]
    for a, ml, lab, ls, mk in CFGS:
        method_line(ax, d[(d.k_alpha == a) & (d.max_len == ml)], "interval", np.log2(SIZES), ls=ls, marker=mk, label=lab)
    size_axis(ax)
    ax.set_ylabel("Overlap(5 %, 5 %)")
    ax.legend(loc="lower left", fontsize=fs.TICK_PT - 0.5)
    ax.set_title("Interval index configurations", fontsize=fs.TICK_PT, pad=2, loc="left")
    keys = [k for k in ORDER if k in set(st.dataset)]
    for a, ml, lab, ls, mk in CFGS:
        r = st[(st.k_alpha == a) & (st.max_len == ml)].set_index("dataset").reindex(keys)
        ax_mb.plot(r["index_mb_mean"], range(len(keys)), ls="none", marker=mk, ms=2.6,
                   color=fs.PARTIAL_COLORS["interval"])
    ax_mb.set_xscale("log")
    fs.plain_log(ax_mb, "x")
    fs.log_ygrid(ax_mb, "x")
    ax_mb.set_yticks(range(len(keys)), [fs.DATASET_SHORT[k] for k in keys])
    ax_mb.invert_yaxis()
    ax_mb.set_xlabel("Index size (MB, log scale)")


def panel_bound(ax):
    files = sorted(RES.glob("*_seed[0-9]_bound.csv"))
    if not files:
        return pending(ax, "bound")
    b = pd.concat([pd.read_csv(f) for f in files])
    b["r"] = b["max_gap"] / b["bound"]
    keys = [k for k in ORDER if k in set(b.dataset)]
    parts = ax.violinplot([b.loc[b.dataset == k, "r"] for k in keys], positions=range(len(keys)), widths=0.7,
                          showextrema=False)
    for pc, k in zip(parts["bodies"], keys):
        pc.set_facecolor(fs.DATASET_COLORS[k])
        pc.set_alpha(0.6)
        pc.set_linewidth(0)
    ax.axhline(1.0, color=fs.INK, lw=0.6, ls="--")
    ax.set_xticks(range(len(keys)), [fs.DATASET_SHORT[k] for k in keys], rotation=40, ha="right",
                  rotation_mode="anchor")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("largest |score − own-log score|\n/ √Σε² (per query, niche)")


def panel_structure(ax):
    q = read("query_structure.csv")
    if q is None:
        return pending(ax, "segments")
    for k in ORDER:
        for fam, ls in (("program", "-"), ("random", ":")):
            r = q[(q.dataset == k) & (q.family == fam)].set_index("size").reindex(SIZES)
            ok = r["seg_frac_mean_mean"].notna()
            ax.plot(np.log2(SIZES)[ok.values], r["seg_frac_mean_mean"][ok], color=fs.DATASET_COLORS[k], ls=ls, lw=0.7)
    size_axis(ax)
    ax.set_ylabel("segments per gene\n(niche-size weighted)")
    ax.legend(handles=[Line2D([], [], color=fs.MUTED, ls="-", lw=0.7, label="contiguous sets"),
                       Line2D([], [], color=fs.MUTED, ls=":", lw=0.7, label="random sets")],
              loc="lower left", fontsize=fs.TICK_PT - 0.5)


def panel_c90(ax):
    s = read("summary.csv")
    if s is None:
        return pending(ax, "c90")
    s = main_cfg(s[(s.family == "program") & (s["size"] == 16)])
    keys = [k for k in ORDER if k in set(s.dataset)]
    for m, o in zip(MAIN, np.linspace(-0.27, 0.27, len(MAIN))):
        r = s[s.method == m].set_index("dataset").reindex(keys)
        x, y = np.arange(len(keys)) + o, r["c90_pct_mean"].values
        ok = ~np.isnan(y)
        ax.plot(x[ok], y[ok], ls="none", marker=fs.PARTIAL_MARKERS[m], ms=2.8, color=fs.PARTIAL_COLORS[m])
        # never reaches 0.9 within N: hollow, at 100 %
        ax.plot(x[~ok], np.full((~ok).sum(), 100.0), ls="none", marker=fs.PARTIAL_MARKERS[m], ms=2.8, mfc="white",
                color=fs.PARTIAL_COLORS[m], mew=0.6)
    ax.set_yscale("log")
    fs.plain_log(ax, "y")
    fs.log_ygrid(ax, "y")
    ax.set_xticks(range(len(keys)), [fs.DATASET_SHORT[k] for k in keys], rotation=40, ha="right",
                  rotation_mode="anchor")
    ax.set_ylabel("c90 (% of N, log scale)")
    ax.set_title("16-gene contiguous sets", fontsize=fs.TICK_PT, pad=2, loc="left")


def main():
    fig = fs.figure(fs.DOUBLE, 200)
    outer = GridSpec(3, 1, figure=fig, height_ratios=[0.8, 1, 1], hspace=0.75, left=0.08, right=0.985, top=0.965,
                     bottom=0.12)
    a = panel_per_dataset(fig, outer[0])
    r2 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], wspace=0.55, width_ratios=[1.1, 1.0, 1.1])
    b, c, d = (fig.add_subplot(r2[i]) for i in range(3))
    r3 = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2], wspace=0.6, width_ratios=[1.3, 1.1, 0.8])
    e, f1, f2 = (fig.add_subplot(r3[i]) for i in range(3))
    panel_families(b)
    panel_c90(c)
    panel_structure(d)
    panel_bound(e)
    panel_configs(f1, f2)
    fs.label_panel(fig, a[0], "A", -10, 3, panel=tuple(a[1:]))
    fs.label_panel(fig, b, "B", -10, 3)
    fs.label_panel(fig, c, "C", -12, 3)
    fs.label_panel(fig, d, "D", -12, 3)
    fs.label_panel(fig, e, "E", -12, 3)
    fs.label_panel(fig, f1, "F", -10, 3, panel=(f2,))
    hd = [Line2D([], [], color=fs.PARTIAL_COLORS[m], ls=fs.PARTIAL_LS[m], marker=fs.PARTIAL_MARKERS[m], ms=3, lw=1.0,
                 label=fs.PARTIAL_LABELS[m]) for m in MAIN]
    fs.legend_below(fig, hd, ncol=2, y=0.06)
    fs.legend_below(fig, fs.dataset_handles(ORDER, marker=False, ls="-"), ncol=8, y=0.025)
    fs.save(fig, "figS8_partial_search_detail")


if __name__ == "__main__":
    main()
