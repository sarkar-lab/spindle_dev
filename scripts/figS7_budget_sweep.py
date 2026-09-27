"""Fig. S7 -- budget-multiplier sweep per dataset, block ground truth (E1).

The whole-matrix ground-truth sweep is Fig. S5A (scripts/figS5_ground_truth_scale.py).
One panel per dataset: Recall@0.1/0.5 eps and Overlap@0.5/1 eps vs the budget multiplier,
with the production budget (1.0) marked. Seed-73 production build, 100 held-out queries.

Inputs: results/budget_sweep/sweep_summary_{block,whole}.csv
"""

import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import figstyle as fs

METRICS = [("recall_at_eps_0.1", r"Recall@0.1$\varepsilon$", "#1A1A1A", "-"),
           ("recall_at_eps_0.5", r"Recall@0.5$\varepsilon$", "#1A1A1A", "--"),
           ("overlap_at_eps_0.5", r"Overlap@0.5$\varepsilon$", "#4477AA", "-"),
           ("overlap_at_eps_1.0", r"Overlap@1$\varepsilon$", "#4477AA", "--")]


def sweep_grid(fig, gs, kind):
    b = pd.read_csv(fs.RESULTS / "budget_sweep" / f"sweep_summary_{kind}.csv")
    b["key"] = b["Dataset"].map(fs.stem_to_key)
    axes = []
    for n, key in enumerate(fs.DATASET_ORDER):
        ax = fig.add_subplot(gs[n // 4, n % 4])
        d = b[b["key"] == key].sort_values("budget_multiplier")
        for col, _, c, ls in METRICS:
            ax.plot(d["budget_multiplier"], d[col], color=c, ls=ls, lw=0.8)
        ax.axvline(1.0, color=fs.MUTED, lw=0.5, ls=":", zorder=0)
        sp = d.loc[d["budget_multiplier"] == 1.0, "mean_speedup"].iloc[0]
        ax.text(0.97, 0.05, f"{fs.fmt_x(sp)} at 1.0", transform=ax.transAxes, ha="right", fontsize=fs.TICK_PT)
        ax.set_xscale("log")
        fs.plain_log(ax, "x", subs=(1.0,))
        ax.set_xlim(0.05, 16)
        ax.set_ylim(0, 1.03)
        ax.set_title(fs.DATASET_LABELS[key], fontsize=fs.TEXT_PT)
        if n % 4 == 0:
            ax.set_ylabel("Score")
        else:
            ax.set_yticklabels([])
        if n >= 4:
            ax.set_xlabel("Budget multiplier")
        axes.append(ax)
    return axes


def handles():
    return [Line2D([], [], color=c, ls=ls, lw=0.8, label=lab) for _, lab, c, ls in METRICS] + \
           [Line2D([], [], color=fs.MUTED, ls=":", lw=0.5, label="Production budget (1.0)")]


def main():
    fig = fs.figure(fs.DOUBLE, 85)
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.12, left=0.06, right=0.99, top=0.94, bottom=0.2)
    sweep_grid(fig, gs, "block")
    fs.legend_below(fig, handles(), ncol=5, y=0.06)
    fs.save(fig, "figS7_budget_sweep")


if __name__ == "__main__":
    main()
