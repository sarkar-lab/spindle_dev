"""Fig. S11 -- the biology results on the approximate tiers, 180 mm.

A  signature queries (template covariance), all datasets, seeds 0-4: Jaccard of the interval index's top 10
   with Spindle-Exact's top 10 on S, one point per signature
B  Cliff's delta of the signature score of the top 10: interval index vs Spindle-Exact
C  whole-tile queries, breast seed 73 (100 held-out tiles): composition JSD of each query vs its exact top 10,
   the DAG's set of 10 and 10 random training tiles
D  the same queries: myoepithelial fraction of the query vs that of its exact top 10

CSV inputs only: results/signature_queries/, results/query_example/.
"""

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr, wilcoxon

import bio_panels as bp
import figstyle as fs

W, H = fs.DOUBLE, 64.0
K = 10


def sig_pairs():
    rows = []
    for key in fs.DATASET_ORDER:
        f = bp.SIGS / bp.stem(key) / "stats.csv"
        s = pd.read_csv(f)
        s = s[(s.K == K) & s.seed.isin(bp.FIG_SEEDS) & (s.construction == "coexpression")]
        g = s.groupby(["signature", "method"])[["jaccard_vs_exact", "cliffs_delta"]].mean().unstack("method")
        for sig, r in g.iterrows():
            rows.append({"key": key, "signature": sig, "jaccard": r[("jaccard_vs_exact", "interval")],
                         "delta_exact": r[("cliffs_delta", "spindle_exact")],
                         "delta_interval": r[("cliffs_delta", "interval")]})
    return pd.DataFrame(rows)


def main():
    sp = sig_pairs()
    q = pd.read_csv(fs.RESULTS / "query_example" / "breast_all_queries.csv")
    fig = fs.figure(W, H)
    y, h = 8.0, 34.0

    ax = bp.ax_mm(fig, 13, y, 34, h)
    rng = np.random.default_rng(0)
    keys = [k for k in fs.DATASET_ORDER if k in set(sp.key)]
    for i, k in enumerate(keys):
        g = sp[sp.key == k]
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(g)), g.jaccard, s=8, color=fs.DATASET_COLORS[k],
                   marker=fs.DATASET_MARKERS[k], lw=0)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([fs.DATASET_SHORT[k] for k in keys], rotation=45, ha="right", fontsize=fs.TICK_PT)
    ax.set_ylabel(f"Top-{K} Jaccard, interval vs exact")
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.6, len(keys) - 0.4)
    fs.despine(ax)
    fs.label_at(fig, 1.5, 3.0, "A", panel=(ax,))

    ax = bp.ax_mm(fig, 60, y, h, h)
    for k, g in sp.groupby("key"):
        ax.scatter(g.delta_exact, g.delta_interval, s=8, color=fs.DATASET_COLORS[k], marker=fs.DATASET_MARKERS[k], lw=0)
    fs.identity_line(ax, -1, 1)
    ax.set_xlabel("Cliff's δ, Spindle-Exact")
    ax.set_ylabel("Cliff's δ, interval index")
    fs.despine(ax)
    fs.label_at(fig, 50.0, 3.0, "B", panel=(ax,))

    ax = bp.ax_mm(fig, 113, y, 26, h)
    cols = [("jsd_exact", "Exact\ntop 10", fs.TIER_COLORS["exact"]), ("jsd_dag", "DAG\nset", fs.TIER_COLORS["dag"]),
            ("jsd_random", "Random\n10", fs.REFERENCE_COLOR)]
    for i, (c, lab, col) in enumerate(cols):
        v = q[c].to_numpy()
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=2.5, color=col, lw=0, alpha=0.7)
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color=fs.INK, lw=1.0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lab for _, lab, _ in cols], fontsize=fs.TICK_PT)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Composition JSD vs query")
    ax.set_ylim(0, None)
    fs.despine(ax)
    fs.label_at(fig, 101.0, 3.0, "C", panel=(ax,))

    ax = bp.ax_mm(fig, 152, y, 24, 24)
    ax.scatter(q.myoepi_frac, q.myoepi_frac_exact10, s=4, color=fs.TIER_COLORS["exact"], lw=0)
    fs.identity_line(ax, 0, 1)
    ax.set_xlabel("Query, myoepithelial")
    ax.set_ylabel("Exact top 10, myoepithelial")
    fs.despine(ax)
    fs.label_at(fig, 142.0, 3.0, "D", panel=(ax,))

    fig.legend(handles=fs.dataset_handles(keys), loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=8, frameon=False,
               columnspacing=0.8, handletextpad=0.2)
    fs.save(fig, "figS11_tier_checks")

    print(f"interval vs exact top-{K} Jaccard: median {sp.jaccard.median():.2f}, range {sp.jaccard.min():.2f}-"
          f"{sp.jaccard.max():.2f}")
    print(f"delta interval - exact: median {np.median(sp.delta_interval - sp.delta_exact):+.2f}; "
          f"interval delta > 0.5: {(sp.delta_interval > 0.5).sum()} of {len(sp)} (exact {(sp.delta_exact > 0.5).sum()})")
    print("JSD median: " + ", ".join(f"{c} {q[c].median():.3f} (mean {q[c].mean():.3f})" for c, _, _ in cols))
    print(f"DAG vs exact JSD Wilcoxon p {wilcoxon(q.jsd_dag, q.jsd_exact).pvalue:.2g}; "
          f"DAG vs random p {wilcoxon(q.jsd_dag, q.jsd_random).pvalue:.2g}")
    r = spearmanr(q.myoepi_frac, q.myoepi_frac_exact10)
    print(f"myoepithelial Spearman {r.statistic:.2f} (p {r.pvalue:.2g}); queries with any myoepithelial cells: "
          f"{(q.myoepi_frac > 0).sum()} of {len(q)}")


if __name__ == "__main__":
    main()
