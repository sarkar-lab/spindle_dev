"""Fig. S6 -- Query perturbation: calibration and cell subsampling (E7).

A  spectral noise sigma vs the realised block-LE shift of the query, in units of its niche epsilon
B  cell-subsampling fraction f vs the same shift
C  Recall@1 eps under cell subsampling, Spindle (solid) vs exact search on the perturbed query (dotted)
D  Recall@0.1 eps under cell subsampling, same encoding

Seed-73 builds, 100 held-out queries, 5 noise seeds per level (error bars: s.d. over noise seeds).
Input: results/noise_robustness/summary.csv
"""

import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import figstyle as fs

ORDER = fs.DATASET_ORDER


def lines(ax, n, model, y, sd=None, ls="-", xreverse=False, ref=None):
    for key in ORDER:
        d = n[(n["key"] == key) & (n["model"] == model)].copy()
        if model == "cell_subsample":
            d.loc[d["level"] == 0, "level"] = 1.0  # level 0 = unperturbed = all cells
        d = d.sort_values("level")
        c = fs.DATASET_COLORS[key]
        if ref:
            ax.plot(d["level"], d[ref], color=c, lw=0.6, ls=":", zorder=1)
        ax.errorbar(d["level"], d[y], yerr=d[sd] if sd else None, color=c, lw=0.8, ls=ls,
                    marker=fs.DATASET_MARKERS[key], ms=2.4, mew=0, elinewidth=0.5, zorder=2)
    if xreverse:
        ax.invert_xaxis()


def main():
    n = pd.read_csv(fs.RESULTS / "noise_robustness" / "summary.csv")
    n["key"] = n["Dataset"].map(fs.stem_to_key)
    fig = fs.figure(fs.DOUBLE, 62)
    gs = GridSpec(1, 4, figure=fig, wspace=0.42, left=0.06, right=0.99, top=0.93, bottom=0.3)

    ax_a = fig.add_subplot(gs[0])
    lines(ax_a, n, "spectral", "mean_d_block_over_eps", "sd_d_block_over_eps")
    ax_a.axhline(1, color=fs.MUTED, lw=0.5, ls="--", zorder=0)
    ax_a.set_xlabel(r"Spectral noise $\sigma$")
    ax_a.set_ylabel(r"Query shift ($\varepsilon$ units)")

    ax_b = fig.add_subplot(gs[1])
    lines(ax_b, n, "cell_subsample", "mean_d_block_over_eps", "sd_d_block_over_eps", xreverse=True)
    ax_b.axhline(1, color=fs.MUTED, lw=0.5, ls="--", zorder=0)
    ax_b.set_xlabel("Fraction of cells kept, f")
    ax_b.set_ylabel(r"Query shift ($\varepsilon$ units)")

    ax_c = fig.add_subplot(gs[2])
    lines(ax_c, n, "cell_subsample", "mean_recall_at_eps_1.0", "sd_recall_at_eps_1.0", xreverse=True,
          ref="mean_exact_noisy_recall_at_eps_1.0")
    ax_c.set_ylim(0.4, 1.02)
    ax_c.set_xlabel("Fraction of cells kept, f")
    ax_c.set_ylabel(r"Recall@1$\varepsilon$")

    ax_d = fig.add_subplot(gs[3])
    lines(ax_d, n, "cell_subsample", "mean_recall_at_eps_0.1", "sd_recall_at_eps_0.1", xreverse=True,
          ref="mean_exact_noisy_recall_at_eps_0.1")
    ax_d.set_ylim(0, 1.02)
    ax_d.set_xlabel("Fraction of cells kept, f")
    ax_d.set_ylabel(r"Recall@0.1$\varepsilon$")

    for ax in (ax_b, ax_c, ax_d):
        ax.set_xticks([1.0, 0.9, 0.75, 0.5])
    fig.canvas.draw()
    for ax, letter in ((ax_a, "A"), (ax_b, "B"), (ax_c, "C"), (ax_d, "D")):
        fs.label_panel(fig, ax, letter, dx_mm=-10)
    handles = fs.dataset_handles(ms=3.0) + [Line2D([], [], color=fs.MUTED, lw=0.8, label="Spindle"),
                                            Line2D([], [], color=fs.MUTED, lw=0.6, ls=":", label="Exact, noisy query")]
    fs.legend_below(fig, handles, ncol=10, y=0.08)
    fs.save(fig, "figS6_noise_calibration")


if __name__ == "__main__":
    main()
