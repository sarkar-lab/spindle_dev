"""Fig. 3 -- Accuracy and speed of retrieval.

A  Recall@0.1 eps and Overlap@0.5 eps per dataset (E5; mean +- s.d. over 5 seeds; the
   other two metrics are in Fig. S4)
B  query time, Spindle vs whole-matrix brute force, with speedup (E5)
C  Recall@0.1 eps vs spectral query noise sigma, Spindle vs exact search on the noisy query (E7)
D  recall-speedup trade-off over the budget multiplier (E1), production budget 1.0 hollow
E  Spindle vs generic ANN baselines (E4): Recall@0.1 eps vs speedup, and index size
   relative to Spindle's index

The plan's schematic panel was dropped (user decision, 2026-09-25); the epsilon
band is explained in Methods. The E2 block-diagonalization ablation is a
submission-only slot and is not drawn in the preprint figure.
"""

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

import figstyle as fs

ORDER = fs.DATASET_ORDER
R = fs.RESULTS


# ------------------------------------------------------------------ data
def per_seed_holdout():
    rows = []
    for key in ORDER:
        for seed in range(5):
            q = pd.read_csv(R / "holdout_search" / f"xenium_human_{key}" / f"seed_{seed}" / "query_metrics.csv")
            rows.append({"dataset": key, "seed": seed, **q.drop(columns=["Dataset"]).mean(numeric_only=True)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ A
# Recall@0.5 eps and Overlap@1 eps are in Fig. S4.
B_METRICS = [("recall_at_eps_0.1", r"Recall@0.1$\varepsilon$"), ("overlap_at_eps_0.5", r"Overlap@0.5$\varepsilon$")]


def panel_accuracy(fig, spec, seeds):
    gs = GridSpecFromSubplotSpec(1, len(B_METRICS), subplot_spec=spec, wspace=0.18)
    axes = []
    for j, (col, label) in enumerate(B_METRICS):
        ax = fig.add_subplot(gs[j])
        for i, key in enumerate(ORDER):
            v = seeds.loc[seeds["dataset"] == key, col].to_numpy()
            c = fs.DATASET_COLORS[key]
            ax.scatter(v, np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), s=3, color=c, alpha=0.35,
                       lw=0, zorder=2)
            ax.errorbar(v.mean(), i, xerr=v.std(ddof=1), fmt=fs.DATASET_MARKERS[key], color=c, ms=3.2,
                        elinewidth=0.7, mew=0, zorder=3)
        ax.set_xlim(0.9, 1.004)
        ax.set_xticks([0.9, 0.95, 1.0], ["0.9", "0.95", "1"])
        ax.set_ylim(len(ORDER) - 0.5, -0.5)
        ax.set_yticks(range(len(ORDER)))
        ax.set_yticklabels([fs.DATASET_SHORT[k] for k in ORDER] if j == 0 else [])
        ax.tick_params(axis="y", length=0)
        ax.set_title(label, fontsize=fs.TEXT_PT)
        for i in range(len(ORDER)):
            ax.axhline(i, color="#F0F0F0", lw=0.4, zorder=0)
        axes.append(ax)
    return axes


# ------------------------------------------------------------------ B
def panel_query_time(ax, seeds, summary):
    for i, key in enumerate(ORDER):
        s = seeds[seeds["dataset"] == key]
        sp, bf = s["spindle_time_ms"] / 1e3, s["bf_time_ms"] / 1e3
        ax.plot([sp.mean(), bf.mean()], [i, i], color=fs.LIGHT, lw=1.0, zorder=1)
        ax.errorbar(bf.mean(), i, xerr=bf.std(ddof=1), fmt=fs.METHOD_MARKERS["brute_force"],
                    color=fs.METHOD_COLORS["brute_force"], ms=3.2, mew=0, elinewidth=0.7, zorder=2)
        ax.errorbar(sp.mean(), i, xerr=sp.std(ddof=1), fmt=fs.METHOD_MARKERS["spindle"],
                    color=fs.METHOD_COLORS["spindle"], ms=3.2, mew=0, elinewidth=0.7, zorder=3)
        ax.text(bf.mean() * 1.6, i, fs.fmt_x(summary.loc[key, "mean_mean_speedup"]), va="center",
                fontsize=fs.TICK_PT)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 3000)
    fs.plain_log(ax, "x", subs=(1.0,))
    ax.set_ylim(len(ORDER) - 0.5, -0.5)
    ax.set_yticks(range(len(ORDER)), [fs.DATASET_SHORT[k] for k in ORDER])
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Query time (s)")
    ax.legend(handles=fs.method_handles(["spindle", "brute_force"]), loc="lower right", fontsize=fs.TICK_PT,
              bbox_to_anchor=(1.0, 1.0), ncol=2)


# ------------------------------------------------------------------ C
def panel_budget(ax):
    b = pd.read_csv(R / "budget_sweep" / "sweep_summary_block.csv")
    b["key"] = b["Dataset"].map(fs.stem_to_key)
    for key in ORDER:
        d = b[b["key"] == key].sort_values("budget_multiplier")
        c = fs.DATASET_COLORS[key]
        ax.plot(d["mean_speedup"], d["recall_at_eps_0.1"], color=c, lw=0.8, zorder=2)
        p = d[d["budget_multiplier"] == 1.0]
        ax.plot(p["mean_speedup"], p["recall_at_eps_0.1"], marker=fs.DATASET_MARKERS[key], mfc="white", mec=c,
                mew=0.8, ms=3.8, ls="none", zorder=3)
    ax.set_xscale("log")
    fs.plain_log(ax, "x", subs=(1.0, 3.0))
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Speedup vs brute force")
    ax.set_ylabel(r"Recall@0.1$\varepsilon$")
    ax.legend(handles=[Line2D([], [], ls="none", marker="o", mfc="white", mec=fs.MUTED, mew=0.8, ms=3.8,
                              label="Budget 1.0")], loc="upper right", fontsize=fs.TICK_PT)


# ------------------------------------------------------------------ D
def panel_baselines(ax, ax_leg, ax_size):
    f = pd.read_csv(R / "ann_baselines" / "summary.csv")
    methods = ["spindle", "flat", "hnsw", "pca_hnsw", "phi_knn"]
    for m in methods:
        d = f[f["method"] == m]
        ax.scatter(d["mean_speedup"], d["recall_at_eps_0.1"], s=12, color=fs.METHOD_COLORS[m],
                   marker=fs.METHOD_MARKERS[m], lw=0, zorder=3 if m == "spindle" else 2)
    ax.set_xscale("log")
    fs.plain_log(ax, "x", subs=(1.0,))
    ax.set_xlim(10, 3e4)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel("Speedup vs brute force")
    ax.set_ylabel(r"Recall@0.1$\varepsilon$")
    ax_leg.set_axis_off()
    ax_leg.legend(handles=fs.method_handles(methods), loc="center left", fontsize=fs.TICK_PT,
                  borderaxespad=0.0, bbox_to_anchor=(-0.25, 0.5))

    # index size relative to Spindle's own index on the same dataset
    sp = f[f["method"] == "spindle"].set_index("dataset")["index_mb"]
    rng = np.random.default_rng(0)
    for j, m in enumerate(methods[1:]):
        d = f[f["method"] == m].set_index("dataset")
        ratio = (d["index_mb"] / sp.loc[d.index]).to_numpy()
        ax_size.scatter(j + rng.uniform(-0.15, 0.15, len(ratio)), ratio, s=8, color=fs.METHOD_COLORS[m],
                        marker=fs.METHOD_MARKERS[m], lw=0, zorder=2)
    ax_size.axhline(1, color=fs.METHOD_COLORS["spindle"], lw=0.6, ls="--", zorder=1)
    ax_size.set_yscale("log")
    fs.plain_log(ax_size, "y", subs=(1.0, 3.0))
    fs.log_ygrid(ax_size)
    # method identity comes from marker colour/shape (legend to the right)
    ax_size.set_xticks([])
    ax_size.set_xlabel("Baseline method")
    ax_size.set_xlim(-0.5, len(methods) - 1.5)
    ax_size.set_ylabel("Index size / Spindle")


# ------------------------------------------------------------------ E
def panel_noise(ax):
    n = pd.read_csv(R / "noise_robustness" / "summary.csv")
    n = n[n["model"] == "spectral"].copy()
    n["key"] = n["Dataset"].map(fs.stem_to_key)
    for key in ORDER:
        d = n[n["key"] == key].sort_values("level")
        c = fs.DATASET_COLORS[key]
        ax.plot(d["level"], d["mean_exact_noisy_recall_at_eps_0.1"], color=c, lw=0.6, ls=":", zorder=1)
        ax.errorbar(d["level"], d["mean_recall_at_eps_0.1"], yerr=d["sd_recall_at_eps_0.1"], color=c, lw=0.8,
                    marker=fs.DATASET_MARKERS[key], ms=2.6, mew=0, elinewidth=0.5, zorder=2)
    ax.set_xlim(-0.02, 0.52)
    ax.set_ylim(0.75, 1.01)
    ax.set_xlabel(r"Query noise $\sigma$")
    ax.set_ylabel(r"Recall@0.1$\varepsilon$")
    ax.legend(handles=[Line2D([], [], color=fs.MUTED, lw=0.8, label="Spindle"),
                       Line2D([], [], color=fs.MUTED, lw=0.6, ls=":", label="Exact, noisy query")],
              loc="lower left", fontsize=fs.TICK_PT)


def main():
    seeds = per_seed_holdout()
    summary = pd.read_csv(R / "holdout_search" / "summary.csv")
    summary["key"] = summary["dataset"].map(fs.stem_to_key)
    summary = summary.set_index("key")

    fig = fs.figure(fs.DOUBLE, 118)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1, 1], hspace=0.45, left=0.1, right=0.985,
                     top=0.93, bottom=0.14)
    top = GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], width_ratios=[1.0, 1.05, 0.9], wspace=0.45)
    bot = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], width_ratios=[1.2, 1.2, 0.85, 0.5],
                                  wspace=0.42)

    ax_a = panel_accuracy(fig, top[0], seeds)
    ax_b = fig.add_subplot(top[1])
    panel_query_time(ax_b, seeds, summary)
    ax_c = fig.add_subplot(top[2])
    panel_noise(ax_c)
    ax_d = fig.add_subplot(bot[0])
    panel_budget(ax_d)
    ax_e = fig.add_subplot(bot[1])
    ax_e2 = fig.add_subplot(bot[2])
    panel_baselines(ax_e, fig.add_subplot(bot[3]), ax_e2)

    fig.canvas.draw()
    fs.label_panel(fig, ax_a[0], "A", dx_mm=-17, dy_mm=0.5)
    fs.label_panel(fig, ax_b, "B", dx_mm=-17)
    fs.label_panel(fig, ax_c, "C", dx_mm=-11)
    fs.label_panel(fig, ax_d, "D", dx_mm=-11)
    fs.label_panel(fig, ax_e, "E", dx_mm=-11)
    fs.legend_below(fig, fs.dataset_handles(), ncol=8, y=0.035)
    fs.save(fig, "fig3_accuracy_speed")


if __name__ == "__main__":
    main()
