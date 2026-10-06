"""Fig. 4 -- Partial (gene-set) search: Spindle-Exact on S, the interval index, and padding / imputation.

A  schematic, drawn from a real query (breast, seed 73; results/partial_search_final/schematic_example.json,
   selection rule in partial_search_final.schematic): the first 12-gene contiguous set that is one segment in its
   source niche and 3-5 segments in another niche.
   A1 Spindle-Exact on S: S is cut by each niche's blocks; per block, the stored block covariance's
      S x S sub-block is eigendecomposed (its stored log is reused when S covers the block), and
      d_B^S = sqrt(sum_b ||log Q[S cap b] - log T[S cap b]||_F^2) / sqrt|S|.
   A2 interval index: per niche and block, the dyadic intervals of <= 16 genes, each with eps-bounded node
      means (K_s ~ 32 s^0.5; inset: the members of the query's longest piece, 2-D PCA, with the node means
      nearest the query and their eps balls -- a projection never leaves the ball); the query is split
      into segments and each segment into the fewest dyadic pieces, in every niche's gene order; a tile's
      score is the root of the summed squared distances from the query's piece logs to its node means,
      within sqrt(sum eps^2) of the same sum over its own logs.
B  Overlap(5 %, c = 5 % of N) vs query size, contiguous sets: interval index, the same decomposition on the
   tiles' own logs (no index), imputation and padding (each then searched with Spindle-Exact over all
   genes). Per seed the mean over queries and datasets; mean +/- s.d. over seeds 0-4. Spindle-Exact on S
   is the ground truth (1).
C  stored MB per dataset (log scale): whole covariances (reference), Spindle-Exact for gene sets (block logs +
   block covariances), imputation (Spindle-Exact's block logs + one mean covariance per niche), padding
   (block logs only), interval index. Printed: Spindle-Exact's store / the interval index's.
D  query time per dataset (median over contiguous-set queries, mean +/- s.d. over seeds; one BLAS thread).

Inputs: results/partial_search_final/{schematic_example.json, by_size.csv, storage.csv, timing_summary.csv}
"""

import json

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

import figstyle as fs

ORDER = fs.DATASET_ORDER
RES = fs.RESULTS / "partial_search_final"
MAIN = ["interval", "limit", "imputation", "padding"]
SIZES = [4, 8, 16, 32, 48, 64]
PIECE_COLORS = ["#CC6677", "#DDCC77", "#44AA99", "#88CCEE", "#AA4499", "#117733", "#882255", "#999933"]
H_A = 74.0  # mm, schematic height
H = 150.0   # mm, figure height


def read(name):
    p = RES / name
    return pd.read_csv(p) if p.exists() else None


def pending(ax, what):
    ax.text(0.5, 0.5, f"{what}\n(data pending)", transform=ax.transAxes, ha="center", va="center", color=fs.MUTED)
    ax.set_xticks([])
    ax.set_yticks([])


def main_cfg(d):
    """Rows of the paper's configuration: interval K = 32 s^0.5, L = 16; limit L = 16; the baselines."""
    return d[((d.method == "interval") & (d.k_alpha == 0.5) & (d.max_len == 16))
             | ((d.method == "limit") & (d.max_len == 16)) | d.method.isin(["padding", "imputation"])]


# ------------------------------------------------------------------ schematic helpers (mm coordinates)
def _box(ax, x, y, w, h, fc, ec=fs.MUTED, lw=0.35, **kw):
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=lw, **kw))


def _arrow(ax, x0, y0, x1, y1, color=fs.INK):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6, lw=0.6, color=color,
                                 shrinkA=0, shrinkB=0))


def _step(ax, x, y, n, text):
    ax.add_patch(Circle((x, y), 1.5, fc=fs.INK, ec="none", zorder=4))
    ax.text(x, y, str(n), fontsize=fs.TICK_PT - 0.5, ha="center", va="center", color="white", zorder=5)
    ax.text(x + 2.5, y, text, fontsize=fs.TEXT_PT, ha="left", va="center")


def _in_s(segments, block):
    """Positions of the query's genes inside one block (from the segments)."""
    return sorted(p for s in segments if s["block"] == block for p in range(s["a"], s["b"]))


def _block_square(ax, x, y, side, n, on, label):
    """A block covariance (n x n, drawn side mm) with the S x S cells of the query's genes filled."""
    c = side / n
    _box(ax, x, y, side, side, "#F2F2F2", lw=0.4)
    for i in on:
        for j in on:
            _box(ax, x + j * c, y + side - (i + 1) * c, c, c, fs.PARTIAL_COLORS["exact_partial"], ec="none", lw=0)
    ax.text(x + side + 2.0, y + side / 2, label, fontsize=fs.TICK_PT - 0.5, ha="left", va="center", color=fs.MUTED,
            linespacing=1.2)


def schematic_exact(ax, ex, x0, top):
    """A1: Spindle-Exact answers S exactly from stored block covariances."""
    _step(ax, x0 + 1.5, top, 1, "Spindle-Exact on a gene set S")
    k1, k2 = str(ex["source_niche"]), str(ex["other_niche"])
    y = top - 6
    ax.text(x0, y, f"Query: {len(ex['genes'])} genes (a contiguous set)", fontsize=fs.TICK_PT, va="center")
    for g in range(len(ex["genes"])):
        _box(ax, x0 + 34 + g * 2.0, y - 1.0, 2.0, 2.0, fs.INK, ec="white", lw=0.3)
    y -= 5
    for k, name in ((k1, "niche j"), (k2, "niche j′")):
        info = ex["niches"][k]
        blocks = info["exact_blocks"][:4]
        ax.text(x0, y, f"{name}: S meets {len(info['exact_blocks'])} of {info['n_blocks']} blocks",
                fontsize=fs.TICK_PT, va="center")
        y -= 3
        xx, side_max = x0 + 2, 15.0
        for b in blocks:
            n = b["block_len"]
            side = max(8.0, side_max * min(1.0, n / 30))
            _block_square(ax, xx, y - side, side, n, _in_s(info["segments"], b["block"]),
                          f"block {b['block'] + 1}: {b['n_in_S']} of its {n} genes\nare in S (filled: S × S)")
            xx += side + 3
        y -= side_max + 3
    ax.text(x0, y, "Stored per tile: block logs + block covariances.\n"
                   "Per block, the S × S part of the stored covariance\n"
                   "is eigendecomposed (stored log if S covers it).",
            fontsize=fs.TICK_PT, va="top", linespacing=1.3, color=fs.MUTED)
    ax.text(x0, y - 11, r"$d_B^S(q,t)=\sqrt{\sum_b \|\log Q_{S\cap b}-\log T_{S\cap b}\|_F^2}\,/\sqrt{|S|}$",
            fontsize=fs.TEXT_PT, va="top")
    ax.text(x0, y - 16.5, "every tile of every niche: exact ranking", fontsize=fs.TICK_PT, va="top",
            color=fs.PARTIAL_COLORS["exact_partial"])


def _dyadic_rows(ax, x0, y0, w0, n, start, pieces, cell):
    """Dyadic intervals (s = 16 ... 1) of block positions [start, start + w0) with the query's pieces filled."""
    col = {(p["s"], p["a"]): PIECE_COLORS[i % len(PIECE_COLORS)] for i, p in enumerate(pieces)}
    y = y0
    for s in (16, 8, 4, 2, 1):
        for a in range(start - start % s, start + w0, s):
            lo, hi = max(a, start), min(a + s, start + w0)
            if a + s > n or hi <= lo or a < start:
                continue
            _box(ax, x0 + (lo - start) * cell, y, (hi - lo) * cell, 2.2, col.get((s, a), "white"),
                 ec=fs.INK if (s, a) in col else fs.LIGHT, lw=0.5 if (s, a) in col else 0.3)
        ax.text(x0 - 0.8, y + 1.1, str(s), fontsize=fs.TICK_PT - 1, ha="right", va="center")
        y -= 3.0
    return y


def _block_strip(ax, x0, y, segs, cell, pidx):
    """One block: a window of its positions around the query's genes (dark) with every segment's pieces below."""
    lo = max(0, min(sg["a"] for sg in segs) - 1)
    hi = min(segs[0]["block_len"], max(sg["b"] for sg in segs) + 1)
    on = {p for sg in segs for p in range(sg["a"], sg["b"])}
    for p in range(lo, hi):
        _box(ax, x0 + (p - lo) * cell, y, cell, 1.8, fs.INK if p in on else "white", lw=0.25)
    for sg in segs:
        for pc in sg["pieces"]:
            _box(ax, x0 + (pc["a"] - lo) * cell, y - 2.4, pc["s"] * cell, 1.8, PIECE_COLORS[pidx % len(PIECE_COLORS)],
                 ec=fs.INK, lw=0.45)
            pidx += 1
    return pidx, (hi - lo) * cell


def schematic_interval(ax, ex, x0, top):
    """A2: the interval index and how a query is scored."""
    _step(ax, x0 + 1.5, top, 2, "Interval index (approximate)")
    k1, k2 = str(ex["source_niche"]), str(ex["other_niche"])
    seg0 = ex["niches"][k1]["segments"][0]
    n = seg0["block_len"]
    start = max(0, (seg0["a"] // 16) * 16)
    w0 = min(32, n - start)
    cell = 34.0 / 32
    y = top - 5.5
    ax.text(x0, y, f"(i) index: every block (here niche j, block {seg0['block'] + 1},\n     {n} genes) → dyadic intervals of ≤ 16 genes",
            fontsize=fs.TICK_PT, va="center", linespacing=1.2)
    y_end = _dyadic_rows(ax, x0 + 4, y - 7, w0, n, start, seg0["pieces"], cell)
    ax.text(x0 + 4, y_end + 1.3, f"genes {start + 1}–{start + w0} of the block; filled: the query's pieces",
            fontsize=fs.TICK_PT - 1, va="center", color=fs.MUTED)
    ax.text(x0, y_end - 3, "per interval of s genes: ~32·√s node means;\nevery tile within $\\varepsilon_s$ of its node mean\n"
                           "(stored: node means + one code per tile)",
            fontsize=fs.TICK_PT, va="top", linespacing=1.25, color=fs.MUTED)

    # (ii) query split, in two niches
    x1 = x0 + 46
    y = top - 5.5
    ax.text(x1, y, "(ii) query → segments (runs in a block)\n      → fewest dyadic pieces, per niche",
            fontsize=fs.TICK_PT, va="center", linespacing=1.2)
    y -= 6.5
    cell2 = 1.25
    for k, name in ((k1, "niche j"), (k2, "niche j′")):
        info = ex["niches"][k]
        segs = info["segments"]
        npieces = sum(len(s["pieces"]) for s in segs)
        ax.text(x1, y, f"{name}: {len(segs)} segment{'s' * (len(segs) > 1)}, {npieces} pieces",
                fontsize=fs.TICK_PT, va="center")
        y -= 3.6
        pidx = 0
        for b in sorted({sg["block"] for sg in segs}):
            pidx, w = _block_strip(ax, x1 + 2, y - 0.9, [sg for sg in segs if sg["block"] == b], cell2, pidx)
            ax.text(x1 + 2 + w + 1.0, y, f"block {b + 1}", fontsize=fs.TICK_PT - 1, va="center", color=fs.MUTED)
            y -= 5.5
        y -= 1.0

    # (iii) score and bound
    eb1, eb2 = ex["niches"][k1]["error_bound"], ex["niches"][k2]["error_bound"]
    ax.text(x1, y - 0.5, "(iii) score every tile t:", fontsize=fs.TICK_PT, va="top")
    ax.text(x1, y - 4.5, r"$\hat d(t)=\sqrt{\sum_{I}\|\log Q_I-\mu_{I}(t)\|_F^2}$", fontsize=fs.TEXT_PT, va="top")
    ax.text(x1, y - 10.0, r"$|\hat d(t)-d_{own}(t)|\leq\sqrt{\sum_I \varepsilon_{|I|}^2}$" +
            f"  (here: niche j {eb1:.2f}, j′ {eb2:.2f})", fontsize=fs.TICK_PT, va="top")
    ax.text(x1, y - 14.5, r"$\mu_I(t)$: t's node mean on piece I; $d_{own}$: same sum with t's own logs",
            fontsize=fs.TICK_PT - 1, va="top", color=fs.MUTED)


def inset_nodes(ax, ex):
    """Members of the query's longest piece (source niche), 2-D PCA; the 4 node means nearest the query."""
    ins = ex["inset"]
    X, M, q = np.asarray(ins["members_2d"]), np.asarray(ins["means_2d"]), np.asarray(ins["query_2d"])
    codes = np.asarray(ins["codes"])
    near = np.argsort(((M - q) ** 2).sum(1))[:3]
    ax.scatter(X[:, 0], X[:, 1], s=0.6, color=fs.LIGHT, lw=0, zorder=1, rasterized=True)
    for i, k in enumerate(near):
        m = codes == k
        col = PIECE_COLORS[(i + 2) % len(PIECE_COLORS)]
        ax.scatter(X[m, 0], X[m, 1], s=1.4, color=col, lw=0, zorder=2)
        ax.add_patch(Circle(M[k], ins["eps"], fill=False, ec=col, lw=0.5, ls=(0, (2, 1)), zorder=3))
        ax.plot(*M[k], marker="x", color=fs.INK, ms=2.5, mew=0.7, zorder=4)
    ax.plot(*q, marker="*", color=fs.INK, ms=4.5, mec="white", mew=0.3, zorder=5)
    r = ins["eps"] * 1.2
    ax.set_xlim(q[0] - r, q[0] + r)
    ax.set_ylim(q[1] - r, q[1] + r)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(True)
        sp.set_linewidth(0.4)
        sp.set_color(fs.MUTED)
    ax.set_title(f"tiles on one {ins['s']}-gene interval (PCA):\n× node mean, dashed $\\varepsilon$ ball, $\\star$ query",
                 fontsize=fs.TICK_PT - 0.5, pad=1.5, linespacing=1.15)


# ------------------------------------------------------------------ data panels
def panel_size(ax):
    d = read("by_size.csv")
    if d is None:
        return pending(ax, "Overlap vs size")
    d = main_cfg(d[(d.dataset == "all") & (d.family == "program") & (d.c_target == 0.05)])
    x = np.log2(SIZES)
    for m in MAIN:
        r = d[d.method == m].set_index("size").reindex(SIZES)
        ax.errorbar(x, r["overlap_0.05_mean"], yerr=r["overlap_0.05_std"], color=fs.PARTIAL_COLORS[m],
                    ls=fs.PARTIAL_LS[m], marker=fs.PARTIAL_MARKERS[m], ms=3, lw=1.0, elinewidth=0.5, capsize=0,
                    mec="white", mew=0.3)
    ax.axhline(1.0, color=fs.PARTIAL_COLORS["exact_partial"], lw=0.7, ls=":", zorder=1)
    per = read("by_size.csv")
    per = main_cfg(per[(per.dataset != "all") & (per.family == "program") & (per.c_target == 0.05)])
    n_ds = per[per.method == "interval"].groupby("size")["dataset"].nunique().reindex(SIZES).fillna(0).astype(int)
    ax.set_xticks(x, [f"{s}\n({n})" if n < len(ORDER) else str(s) for s, n in zip(SIZES, n_ds)])
    ax.set_xlabel("Genes in the query (contiguous sets;\nin brackets: datasets with blocks that long)")
    ax.set_ylabel("Overlap(5 %, c = 5 % of N)")
    ax.set_ylim(0, 1.04)


def panel_storage(ax):
    s = read("storage.csv")
    if s is None:
        return pending(ax, "Storage")
    s = s[(s.k_alpha == 0.5) & (s.max_len == 16)].set_index("dataset")
    keys = [k for k in ORDER if k in s.index]
    # padding stores Spindle-Exact's block logs; imputation also one mean covariance per niche
    cols = [("whole_cov_upper_mb_mean", "whole"), ("exact_partial_mb_mean", "exact_partial"),
            ("exact_plus_means_mb_mean", "imputation"), ("exact_block_logs_mb_mean", "padding"),
            ("index_mb_mean", "interval")]
    for i, k in enumerate(keys):
        r = s.loc[k]
        vals = [r[c] for c, _ in cols]
        ax.plot([min(vals), max(vals)], [i, i], color=fs.LIGHT, lw=0.8, zorder=1)
        for z, ((c, m), v) in enumerate(zip(cols, vals)):
            if m == "whole":
                ax.plot(v, i, marker="s", color=fs.TIER_COLORS["whole"], ms=3.2, ls="none", zorder=3)
            else:
                ax.plot(v, i, marker=fs.PARTIAL_MARKERS[m], color=fs.PARTIAL_COLORS[m], ms=3.2, ls="none",
                        zorder=3 + z, mec="white", mew=0.3)
        ax.text(r["index_mb_mean"] / 1.3, i, fs.fmt_x(r["exact_partial_mb_mean"] / r["index_mb_mean"]), ha="right",
                va="center", fontsize=fs.TICK_PT - 0.5, color=fs.PARTIAL_COLORS["interval"])
    ax.set_xscale("log")
    fs.plain_log(ax, "x", subs=(1.0,))
    fs.log_ygrid(ax, "x")
    ax.set_yticks(range(len(keys)), [fs.DATASET_SHORT[k] for k in keys])
    ax.set_xlim(ax.get_xlim()[0] / 3, ax.get_xlim()[1])
    ax.invert_yaxis()
    ax.set_xlabel("Stored MB (log scale)")


def panel_time(ax):
    t = read("timing_summary.csv")
    if t is None:
        return pending(ax, "Query time")
    keys = [k for k in ORDER if k in set(t.dataset)]
    meths = ["exact_partial", "interval", "imputation", "padding"]
    off = dict(zip(meths, np.linspace(-0.24, 0.24, len(meths))))
    for i, k in enumerate(keys):
        d = t[t.dataset == k].set_index("method")
        for m in meths:
            if m in d.index:
                ax.errorbar(d.loc[m, "mean"], i + off[m], xerr=d.loc[m, "std"], fmt=fs.PARTIAL_MARKERS[m],
                            color=fs.PARTIAL_COLORS[m], ms=3, elinewidth=0.5, capsize=0, mec="white", mew=0.3)
    ax.set_xscale("log")
    fs.plain_log(ax, "x")
    fs.log_ygrid(ax, "x")
    ax.set_yticks(range(len(keys)), [fs.DATASET_SHORT[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlabel("Query time (ms, log scale)")


def main():
    fig = fs.figure(fs.DOUBLE, H)
    ax_a = fig.add_axes([0, 1 - H_A / H, 1, H_A / H])
    ax_a.set_xlim(0, 180)
    ax_a.set_ylim(0, H_A)
    ax_a.set_axis_off()
    ins = None
    ex_path = RES / "schematic_example.json"
    if ex_path.exists():
        ex = json.loads(ex_path.read_text())
        schematic_exact(ax_a, ex, 5, H_A - 3)
        ax_a.plot([66, 66], [4, H_A - 2], color=fs.LIGHT, lw=0.6)
        schematic_interval(ax_a, ex, 70, H_A - 3)
        ins = fig.add_axes([71 / 180, 1 - (H_A - 4) / H, 26 / 180, 22 / H])
        inset_nodes(ins, ex)
    else:
        ax_a.text(90, H_A / 2, "schematic (data pending)", ha="center", color=fs.MUTED)

    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.05, 1.05], wspace=0.55, left=0.075, right=0.985,
                  top=1 - (H_A + 9) / H, bottom=0.17)
    b, c, d = (fig.add_subplot(gs[i]) for i in range(3))
    panel_size(b)
    panel_storage(c)
    panel_time(d)

    fs.label_at(fig, 1.0, 1.5, "A", panel=(ax_a,) + ((ins,) if ins is not None else ()))
    fs.label_panel(fig, b, "B", -12, 3)
    fs.label_panel(fig, c, "C", -16, 3)
    fs.label_panel(fig, d, "D", -16, 3)
    h = [Line2D([], [], color=fs.PARTIAL_COLORS["exact_partial"], ls=":", lw=0.8, marker=fs.PARTIAL_MARKERS["exact_partial"],
                ms=3, label=fs.PARTIAL_LABELS["exact_partial"])]
    h += [Line2D([], [], color=fs.PARTIAL_COLORS[m], ls=fs.PARTIAL_LS[m], marker=fs.PARTIAL_MARKERS[m], ms=3, lw=1.0,
                 label=fs.PARTIAL_LABELS[m]) for m in MAIN]
    h.append(Line2D([], [], color=fs.TIER_COLORS["whole"], marker="s", ls="none", ms=3, label="Whole covariances"))
    fs.legend_below(fig, h, ncol=3, y=0.075)
    fs.save(fig, "fig4_partial_search")


if __name__ == "__main__":
    main()
