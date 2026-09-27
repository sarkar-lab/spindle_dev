"""Fig. S4 -- Held-out retrieval per seed and dataset (E5).

All four accuracy metrics and the speedup, one dot per seed (5 seeds, 10% of tiles held out);
the vertical tick is the mean over seeds. Fig. 3A shows two of these metrics.

Input: results/holdout_search/<stem>/seed_<n>/query_metrics.csv
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

import figstyle as fs

ORDER = fs.DATASET_ORDER
PANELS = [("recall_at_eps_0.1", r"Recall@0.1$\varepsilon$"), ("recall_at_eps_0.5", r"Recall@0.5$\varepsilon$"),
          ("overlap_at_eps_0.5", r"Overlap@0.5$\varepsilon$"), ("overlap_at_eps_1.0", r"Overlap@1$\varepsilon$"),
          ("speedup", "Speedup vs brute force")]


def per_seed():
    rows = []
    for key in ORDER:
        for seed in range(5):
            q = pd.read_csv(fs.RESULTS / "holdout_search" / f"xenium_human_{key}" / f"seed_{seed}" / "query_metrics.csv")
            r = q.mean(numeric_only=True)
            # speedup as in holdout_search.py / summary.csv: mean brute-force time over mean Spindle time
            r["speedup"] = q["bf_time_ms"].mean() / q["spindle_time_ms"].mean()
            rows.append({"dataset": key, "seed": seed, "n": len(q), **r})
    return pd.DataFrame(rows)


def main():
    d = per_seed()
    fig = fs.figure(fs.DOUBLE, 62)
    gs = GridSpec(1, len(PANELS), figure=fig, wspace=0.15, left=0.155, right=0.99, top=0.88, bottom=0.12)
    axes = []
    offsets = np.linspace(-0.24, 0.24, 5)
    for j, (col, label) in enumerate(PANELS):
        ax = fig.add_subplot(gs[j])
        for i, key in enumerate(ORDER):
            v = d.loc[d["dataset"] == key, col].to_numpy()
            ax.scatter(v, i + offsets, s=5, color=fs.DATASET_COLORS[key], marker=fs.DATASET_MARKERS[key], lw=0)
            ax.plot([v.mean()] * 2, [i - 0.35, i + 0.35], color=fs.INK, lw=0.7)
        ax.set_ylim(len(ORDER) - 0.5, -0.5)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([f"{fs.DATASET_SHORT[k]} (n={int(d.loc[d['dataset'] == k, 'n'].iloc[0])})"
                            for k in ORDER] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        if col != "speedup":
            ax.set_xlim(0.94, 1.003)
            ax.set_xticks([0.95, 1.0], ["0.95", "1"])
        else:
            ax.set_xscale("log")
            fs.plain_log(ax, "x", subs=(1.0, 2.0, 5.0))
            ax.set_xlim(15, 110)
        ax.set_title(label, fontsize=fs.TEXT_PT)
        for i in range(len(ORDER)):
            ax.axhline(i, color="#F0F0F0", lw=0.4, zorder=0)
        axes.append(ax)
    fig.canvas.draw()
    fs.label_panel(fig, axes[0], "A", dx_mm=-26)
    fs.label_panel(fig, axes[4], "B", dx_mm=-3)
    fs.save(fig, "figS4_holdout_per_seed")


if __name__ == "__main__":
    main()
