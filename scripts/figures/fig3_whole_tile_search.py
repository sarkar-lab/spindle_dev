"""Fig. 3 -- Whole-tile search: Spindle-Exact, Spindle-DAG and indexes over whole covariances.

A  query time per dataset (seeds 0-4): whole-matrix exact (whole-matrix logs), Spindle-Exact (block logs),
   Spindle-DAG; the lighter part of each bar is the query's own logs. Only Exact vs whole is a ratio.
B  Spindle-DAG retrieval (K = 32): mean Overlap(5 %, c) vs c/N, one line per dataset (+/- s.d. over
   seeds); the dot marks c90, where the mean overlap reaches 0.9.
   Right: distance ratio at c = 1, 5 and 10 % of N. Per query it is the ratio of means: the mean exact d_B
   from the query to the c tiles the DAG returns, divided by the mean exact d_B to the exact top c (the best
   possible c tiles); 1 = as close as the exact answer, 1.05 = the returned tiles are 5 % farther on average.
   Averaged over queries, then mean +/- s.d. over seeds.
C  K sweep: DAG size (% of the block logs) vs c90 (% of N), K = 4, 8, 16, 32, 64, 128; K = 32 filled.
D  memory vs fidelity against indexes over whole covariances (breast; brain; all 8 datasets in S6). Each
   method is graded against its own exact distance: Spindle-DAG against Spindle-Exact (block distance d_B),
   the whole-matrix methods against the exact whole-matrix scan (d_W). y = c at which 90 % of that exact top 10
   is returned (% of N, log scale; lower = better), x = stored MB. Spindle-DAG over K (4 ... 128); on the
   whole-matrix log vectors: HNSW (M = 32; at these N it visits nearly every tile, so it reaches the exact
   floor, at the memory of the whole-matrix store), PCA to D = 16, 64, 256 dimensions + exact scan (labelled
   by D), PCA to 256 dimensions + product quantization with m = 8 or 32 sub-quantizers of 8 bits (labelled by
   m). Stored MB counts the codes and every model a query needs (projection, codebooks). The two exact scans
   are the ground truths and are not drawn as points; dotted lines mark their stores (block logs;
   whole-matrix logs). PCA + PQ is vertical because its memory is the PCA projection (256 x p(p+1)/2 floats,
   50-118 MB), not the codes (N x m bytes); more sub-quantizers only approach PCA-256 + flat, its ceiling.
   Any method trained on whole-matrix vectors stores such a model, so its memory floor grows with p^2. Which ground truth is biologically more relevant is Fig S12 (block).
Block vs whole-matrix neighbour biology (validates the metric) is Fig S12.

Inputs: results/{exact_vs_whole,dag_holdout,whole_cov_baselines}/ (aggregated).
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import figstyle as fs

ORDER = fs.DATASET_ORDER
TIERS = fs.TIER_ORDER
TIER_SHORT = {"whole": "Whole-matrix exact", "exact": "Spindle-Exact", "dag": "Spindle-DAG"}
C_FRACS = [0.01, 0.05, 0.10]
D_DATASETS = ["breast_cancer", "brain_cancer"]


def key(stem_col):
    return stem_col.map(fs.STEMS)


def read(folder, name):
    path = fs.RESULTS / folder / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["key"] = key(df["dataset"])
    return df


def pending(ax, what):
    ax.text(0.5, 0.5, f"{what}\n(data pending)", transform=ax.transAxes, ha="center", va="center",
            color=fs.MUTED)
    ax.set_xticks([])
    ax.set_yticks([])


def family(method):
    if method == "hnsw":
        return "hnsw"
    if method.startswith("pca") and method.endswith("_flat"):
        return "pca_flat"
    if "_pq" in method:
        return "pca_pq"
    return None


# ---------------------------------------------------------------- A
def panel_time(ax):
    s = read("exact_vs_whole", "summary.csv")
    if s is None:
        return pending(ax, "Query time")
    s = s.set_index("key").reindex([k for k in ORDER if k in set(s["key"])])
    h = 0.26
    for i, k in enumerate(s.index):
        r = s.loc[k]
        for j, t in enumerate(TIERS):
            y = i + (j - 1) * h
            tot, q = r[f"{t}_total_ms_mean"], r[f"{t}_query_ms_mean"]
            ax.barh(y, tot, height=h, color=fs.TIER_COLORS[t], left=0, zorder=2)
            ax.barh(y, q, height=h, color="white", alpha=0.55, left=0, zorder=3, lw=0)
            ax.errorbar(tot, y, xerr=r[f"{t}_total_ms_std"], fmt="none", ecolor=fs.INK, elinewidth=0.5,
                        capsize=0, zorder=4)
        ax.text(r["whole_total_ms_mean"] * 1.15, i, fs.fmt_x(r["whole_over_exact_total_mean"]),
                va="center", ha="left", fontsize=fs.TICK_PT, color=fs.TIER_COLORS["exact"])
    ax.set_xscale("log")
    ax.set_yticks(range(len(s)))
    ax.set_yticklabels([fs.DATASET_SHORT[k] for k in s.index])
    ax.invert_yaxis()
    ax.set_xlabel("Query time (ms, log scale)")
    lo = min(s[f"{t}_query_ms_mean"].min() for t in TIERS)
    ax.set_xlim(lo * 0.5, s["whole_total_ms_mean"].max() * 4)
    fs.log_ygrid(ax, "x")
    ax.text(1.0, 1.01, "Whole-matrix / Exact", transform=ax.transAxes, ha="right", va="bottom", fontsize=fs.TICK_PT,
            color=fs.TIER_COLORS["exact"])


# ---------------------------------------------------------------- B
def panel_overlap(ax):
    cur, summ = read("dag_holdout", "curves.csv"), read("dag_holdout", "summary.csv")
    if cur is None or summ is None:
        return pending(ax, "DAG overlap")
    for k in ORDER:
        c = cur[cur["key"] == k].sort_values("c")
        if c.empty:
            continue
        x, y, e = c["c_frac_mean"] * 100, c["overlap_0.05_mean"], c["overlap_0.05_std"].fillna(0)
        col = fs.DATASET_COLORS[k]
        ax.plot(x, y, color=col, lw=0.9, zorder=3)
        ax.fill_between(x, y - e, y + e, color=col, alpha=0.15, lw=0, zorder=2)
        r = summ[summ["key"] == k].iloc[0]
        ax.plot(r["c90_pct_mean"], 0.9, marker=fs.DATASET_MARKERS[k], color=col, ms=3.5, ls="none", zorder=4,
                mec="white", mew=0.4)
    ax.axhline(0.9, color=fs.LIGHT, lw=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100)
    fs.plain_log(ax, "x", subs=(1.0,))
    ax.set_xlabel("Returned tiles c (% of N, log scale)")
    ax.set_ylabel("Overlap(5 %, c)")
    ax.set_ylim(0, 1.02)
    fs.log_ygrid(ax, "x")


def panel_ratio(ax):
    cur = read("dag_holdout", "curves.csv")
    if cur is None:
        return pending(ax, "Distance ratio")
    keys = [k for k in ORDER if k in set(cur["key"])]
    off = np.linspace(-0.27, 0.27, len(keys))
    for i, f in enumerate(C_FRACS):
        for o, k in zip(off, keys):
            d = cur[cur["key"] == k]
            r = d.iloc[(d["c_frac_mean"] - f).abs().argmin()]
            ax.errorbar(i + o, r["dist_ratio_mean"], yerr=r["dist_ratio_std"], fmt=fs.DATASET_MARKERS[k],
                        color=fs.DATASET_COLORS[k], ms=3, elinewidth=0.5, capsize=0, mec="white", mew=0.3)
    ax.axhline(1.0, color=fs.MUTED, lw=0.6, ls="--", zorder=1)
    ax.set_xticks(range(len(C_FRACS)))
    ax.set_xticklabels([f"{100 * f:g} %" for f in C_FRACS])
    ax.set_xlim(-0.5, len(C_FRACS) - 0.5)
    ax.set_xlabel("Returned tiles c (% of N)")
    ax.set_ylabel("Distance ratio\n(DAG set / exact top c)")


# ---------------------------------------------------------------- C
def panel_ksweep(ax):
    ks = read("dag_holdout", "k_sweep.csv")
    if ks is None:
        return pending(ax, "K sweep")
    for k in ORDER:
        d = ks[ks["key"] == k].sort_values("K")
        if d.empty:
            continue
        col = fs.DATASET_COLORS[k]
        ax.plot(d["dag_pct_of_block_logs_mean"], d["c90_pct_mean"], color=col, lw=0.9, marker=fs.DATASET_MARKERS[k],
                ms=2.5, mfc="white", mec=col, mew=0.6)
        r = d[d["K"] == 32]
        ax.plot(r["dag_pct_of_block_logs_mean"], r["c90_pct_mean"], marker=fs.DATASET_MARKERS[k], color=col, ms=3.5,
                ls="none", mec="white", mew=0.4, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("DAG size (% of block logs, log scale)")
    ax.set_ylabel("c90 (% of N, log scale)")
    fs.plain_log(ax, "x")
    fs.plain_log(ax, "y")
    fs.log_ygrid(ax, "both")
    ax.text(0.98, 0.97, "K = 4 → 128\nfilled: K = 32", transform=ax.transAxes, ha="right", va="top",
            fontsize=fs.TICK_PT, color=fs.MUTED)


# ---------------------------------------------------------------- D
def panel_memory(ax, k, first):
    m, ks = read("whole_cov_baselines", "methods.csv"), read("dag_holdout", "k_sweep.csv")
    if m is None or ks is None or k not in set(m["key"]):
        return pending(ax, fs.DATASET_SHORT[k])
    m = m[(m["key"] == k) & (m["reference"] == "own")]
    n = m["n_tiles_mean"].iloc[0]
    d = ks[ks["key"] == k].sort_values("K")
    ax.plot(d["dag_mb_mean"], 100 * d["c90_top10_mean"] / n, color=fs.TIER_COLORS["dag"], marker=fs.TIER_MARKERS["dag"],
            ms=2.5, lw=0.9, mfc="white", mew=0.6)
    r = d[d["K"] == 32]
    ax.plot(r["dag_mb_mean"], 100 * r["c90_top10_mean"] / n, color=fs.TIER_COLORS["dag"], marker=fs.TIER_MARKERS["dag"],
            ms=3.5, ls="none")
    top = 100.0  # c = N: a method that never returns 90 % of the top 10 within its list is drawn here, hollow
    for fam in fs.COMPETITOR_ORDER:
        f = m[m["method"].map(family) == fam].sort_values("mb_mean")
        y = 100 * f["c90_top10_mean"] / n
        ok = y.notna()
        ax.plot(f["mb_mean"][ok], y[ok], color=fs.COMPETITOR_COLORS[fam], marker=fs.COMPETITOR_MARKERS[fam], ms=3,
                lw=0.8 if ok.sum() > 1 else 0, zorder=5)
        if (~ok).any():
            ax.plot(f["mb_mean"][~ok], [top] * int((~ok).sum()), color=fs.COMPETITOR_COLORS[fam],
                    marker=fs.COMPETITOR_MARKERS[fam], ms=3.5, ls="none", mfc="white", mew=0.7)
        for r in f.itertuples():
            tag = (f"D={r.method[3:].split('_')[0]}" if fam == "pca_flat"
                   else f"m={r.method.split('_pq')[1]}" if fam == "pca_pq" else None)
            if tag:
                # PCA dimensions under their points, PQ sub-quantizers to the right of theirs
                below = fam == "pca_flat"
                ax.annotate(tag, (r.mb_mean, 100 * r.c90_top10_mean / n), xytext=(0, -4) if below else (4, 0),
                            textcoords="offset points", fontsize=fs.TICK_PT - 1, color=fs.COMPETITOR_COLORS[fam],
                            va="top" if below else "center", ha="center" if below else "left")
    for t, ls in (("exact", ":"), ("whole", ":")):
        mb = m.loc[m["method"] == t, "mb_mean"].iloc[0]
        ax.axvline(mb, color=fs.TIER_COLORS[t], lw=0.7, ls=ls, zorder=1)
    if first:
        ax.text(0.03, 0.97, "lower = better", transform=ax.transAxes, fontsize=fs.TICK_PT, color=fs.MUTED, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    fs.plain_log(ax, "x", subs=(1.0,))
    fs.plain_log(ax, "y")
    fs.log_ygrid(ax, "both")
    ax.set_xlabel("Stored MB (log scale)")
    if first:
        ax.set_ylabel("c for 90 % of own exact top 10\n(% of N, log scale)")
    ax.set_title(fs.DATASET_SHORT[k], fontsize=fs.TEXT_PT, pad=2)


# ---------------------------------------------------------------- figure
def main():
    fig = fs.figure(fs.DOUBLE, 150)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.45, left=0.09, right=0.985, top=0.96,
                     bottom=0.17)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.3, 1.15, 0.75], wspace=0.5)
    mid = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1.3, 1.0, 1.0], wspace=0.4)
    a = fig.add_subplot(top[0])
    b1, b2 = fig.add_subplot(top[1]), fig.add_subplot(top[2])
    c = fig.add_subplot(mid[0])
    d1, d2 = fig.add_subplot(mid[1]), fig.add_subplot(mid[2])
    panel_time(a)
    panel_overlap(b1)
    panel_ratio(b2)
    panel_ksweep(c)
    panel_memory(d1, D_DATASETS[0], True)
    panel_memory(d2, D_DATASETS[1], False)
    if d1.get_ylim() and d2.get_ylim() and d1.get_yscale() == "log" and d2.get_yscale() == "log":
        lo = min(d1.get_ylim()[0], d2.get_ylim()[0])
        hi = max(d1.get_ylim()[1], d2.get_ylim()[1])
        d1.set_ylim(min(lo, 1.0), hi)
        d2.set_ylim(min(lo, 1.0), hi)

    fs.label_panel(fig, a, "A", -16, 3)
    fs.label_panel(fig, b1, "B", -10, 3, panel=(b2,))
    fs.label_panel(fig, c, "C", -11, 3)
    fs.label_panel(fig, d1, "D", -11, 3, panel=(d2,))

    tiers = [Patch(color=fs.TIER_COLORS[t], label=TIER_SHORT[t]) for t in TIERS]
    tiers.append(Patch(facecolor="white", edgecolor=fs.MUTED, lw=0.5, label="lighter part: query's own logs"))
    comp = [Line2D([], [], color=fs.TIER_COLORS["dag"], marker=fs.TIER_MARKERS["dag"], ms=3, lw=0.9,
                   label="Spindle-DAG (K = 4 → 128)")]
    comp += [Line2D([], [], color=fs.COMPETITOR_COLORS[f], marker=fs.COMPETITOR_MARKERS[f], ms=3, lw=0.8,
                    label=fs.COMPETITOR_LABELS[f]) for f in fs.COMPETITOR_ORDER]
    comp.append(Line2D([], [], color=fs.MUTED, ls=":", lw=0.7, label="store of block logs / whole-matrix logs"))
    fs.legend_below(fig, fs.dataset_handles(ORDER), ncol=8, y=0.075)
    fig.legend(handles=tiers + comp, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.045), columnspacing=1.0)
    fs.save(fig, "fig3_whole_tile_search")


if __name__ == "__main__":
    main()
