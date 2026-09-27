"""Fig. S9 -- Partial-panel search per dataset (E11).

One panel per dataset: Recall@0.05 eps, Recall@0.1 eps and Overlap@0.1 eps vs query length,
mean +- s.d. over the 5 seeds (each seed: mean over its queries in the bin, 50 draws x
2 query patterns per bin).

Input: results/partial_panel_search/<stem>/seed_<n>/query_metrics.csv
"""

import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import figstyle as fs
from fig4_partial_panel import BIN_LABELS, BINS, load

METRICS = [("recall_at_eps_0.05", r"Recall@0.05$\varepsilon$", "#1A1A1A", "--"),
           ("recall_at_eps_0.1", r"Recall@0.1$\varepsilon$", "#1A1A1A", "-"),
           ("overlap_at_eps_0.1", r"Overlap@0.1$\varepsilon$", "#4477AA", "-")]


def main():
    d = load()
    fig = fs.figure(fs.DOUBLE, 85)
    gs = GridSpec(2, 4, figure=fig, hspace=0.5, wspace=0.12, left=0.06, right=0.99, top=0.94, bottom=0.2)
    x = np.arange(len(BINS))
    for n, key in enumerate(fs.DATASET_ORDER):
        ax = fig.add_subplot(gs[n // 4, n % 4])
        s = d[d["dataset"] == key].groupby(["seed", "Length_Bin"])
        for col, _, c, ls in METRICS:
            m = s[col].mean().unstack().reindex(columns=BINS)
            ax.errorbar(x, m.mean(), yerr=m.std(ddof=1), color=c, ls=ls, lw=0.8, elinewidth=0.6, marker="o",
                        ms=2, mew=0)
        ax.set_xticks(x, BIN_LABELS)
        ax.set_xlim(-0.3, len(BINS) - 0.7)
        ax.set_ylim(0, 1.03)
        ax.set_title(fs.DATASET_LABELS[key], fontsize=fs.TEXT_PT)
        if n % 4 == 0:
            ax.set_ylabel("Score")
        else:
            ax.set_yticklabels([])
        if n >= 4:
            ax.set_xlabel("Query length (genes)")
    fs.legend_below(fig, [Line2D([], [], color=c, ls=ls, lw=0.8, label=lab) for _, lab, c, ls in METRICS],
                    ncol=3, y=0.06)
    fs.save(fig, "figS9_partial_panel_per_dataset")


if __name__ == "__main__":
    main()
