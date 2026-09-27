"""Fig. S10 -- Cross-platform search per seed, index niches and routing (E12).

A  Recall@0.1 eps and Overlap@0.5 eps per seed, both directions (50 queries each)
B  covariance niches in the index per seed (x2v: Visium index; v2x: Xenium index)
C  speedup per seed
D  all-niche search vs kNN-routed single-niche search, seed 0 (moved from Fig. 5)
E  PC1/PC2 before and after correction for x2v (Visium index; Fig. 5B shows v2x)

Inputs: results/cross_modal_search/{seed_*/, correction_ablation_seed0/, bias_pca*.csv}
"""

import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import figstyle as fs
from fig5_cross_platform import CM, DIR_COLORS, DIR_LABELS, DIRS, grouped_bars, seed0_summary


def per_seed():
    return pd.concat([pd.read_csv(CM / f"seed_{s}" / f"{d}_summary.csv") for s in range(5) for d in DIRS])


def dots_by_seed(ax, s, col, ylabel, ylim=None):
    for k, d in enumerate(DIRS):
        v = s[s["direction"] == d].sort_values("seed")
        ax.plot(v["seed"] + (k - 0.5) * 0.2, v[col], "o", color=DIR_COLORS[d], ms=3.2, mew=0)
    ax.set_xticks(range(5))
    ax.set_xlim(-0.5, 4.5)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    s = per_seed()
    fig = fs.figure(fs.DOUBLE, 105)
    outer = GridSpec(2, 1, figure=fig, hspace=0.55, left=0.07, right=0.99, top=0.95, bottom=0.14)
    top = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.5)
    bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[1], width_ratios=[1, 1, 1], wspace=0.35)

    ax_a1 = fig.add_subplot(top[0])
    dots_by_seed(ax_a1, s, "recall_at_eps_0.1", r"Recall@0.1$\varepsilon$", (0.85, 1.01))
    ax_a2 = fig.add_subplot(top[1])
    dots_by_seed(ax_a2, s, "overlap_at_eps_0.5", r"Overlap@0.5$\varepsilon$", (0.85, 1.01))
    ax_b = fig.add_subplot(top[2])
    dots_by_seed(ax_b, s, "n_niches", "Niches in index", (0, 4))
    ax_b.set_yticks(range(5))
    ax_c = fig.add_subplot(top[3])
    dots_by_seed(ax_c, s, "mean_speedup", "Speedup vs brute force", (0, 130))

    ax_d = fig.add_subplot(bot[0])
    vals = {d: [seed0_summary(d)["recall_at_eps_0.1"], seed0_summary(d, "single_niche")["recall_at_eps_0.1"]]
            for d in DIRS}
    grouped_bars(ax_d, ["All niches", "Routed"], vals, r"Recall@0.1$\varepsilon$ (seed 0)")
    ax_d.set_xlabel("Niches searched")

    p = pd.read_csv(CM / "bias_pca.csv")
    p = p[p["direction"] == "x2v"]
    axes_e = []
    lim_x = (p["PC1"].min() - 1, p["PC1"].max() + 1)
    lim_y = (p["PC2"].min() - 1, p["PC2"].max() + 1)
    for j, corrected in enumerate((False, True)):
        ax = fig.add_subplot(bot[1 + j])
        d = pd.concat([p[p["role"] == "index"], p[(p["role"] == "query") & (p["corrected"] == corrected)]])
        for mod in ("Visium", "Xenium"):
            m = d[d["modality"] == mod]
            ax.scatter(m["PC1"], m["PC2"], s=4, color=fs.PLATFORM_COLORS[mod], lw=0, alpha=0.8)
        ax.set_xlim(*lim_x)
        ax.set_ylim(*lim_y)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("x2v, " + ("corrected" if corrected else "uncorrected"), fontsize=fs.TICK_PT)
        axes_e.append(ax)
    axes_e[1].legend(handles=[Line2D([], [], ls="none", marker="o", ms=3, color=fs.PLATFORM_COLORS[m], label=m)
                              for m in ("Xenium", "Visium")], loc="lower right")

    fig.canvas.draw()
    for ax, letter in ((ax_a1, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D"), (axes_e[0], "E")):
        fs.label_panel(fig, ax, letter, dx_mm=-11)
    fs.legend_below(fig, [Line2D([], [], ls="none", marker="o", ms=3.2, color=DIR_COLORS[d], label=DIR_LABELS[d])
                          for d in DIRS], ncol=2, y=0.045)
    fs.save(fig, "figS10_cross_platform_detail")


if __name__ == "__main__":
    main()
