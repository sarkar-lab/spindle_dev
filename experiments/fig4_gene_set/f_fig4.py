"""Fig 4 atoms from the small inputs of c_summary.py (login node). No schematic (user, 26 Oct).

Usage: python experiments/fig4_gene_set/f_fig4.py
Atoms (figures/fig4_gene_set/atoms/); row plots list datasets top to bottom in style.DATASET_ORDER, so they line up:
  band         the typical |S| = 16 breast query (rule in c_nx.py): exact d_B^S of every index tile (grey) and of
                the tiles Spindle returned (colour), same bins
  nx_size       NX at 5 % of N (%) against |S| (log2 axis): Spindle, padding, imputation; median over
                datasets, mean +/- s.d. over seeds
  overlap_size  Overlap(5 %, 5 % of N) against |S|, the same methods and aggregation: share of the exact near set
                (d <= 1.05 x the nearest d_B^S) in the returned top 5 % of N
  nx          Spindle NX at 5 % of N (%) per dataset, all sizes pooled; mean +/- s.d. over seeds
  timing        time per query (ms, log scale): search only, Spindle-naive (its scan without the query logs it
                computes inside, c_timing.py) and Spindle (bars, mean over seeds of the per-seed median;
                whiskers s.d.); each index's query logs as open markers in its colour
  storage       stored MB per dataset (log scale): whole covariances, Spindle-naive, Spindle,
                padding (Spindle-naive), imputation (Spindle-naive + niche means)
  legend_*      one legend atom per panel
Numbers next to plots (speed-ups) are text boxes in the pptx: positions go to figures/fig4_gene_set/annotations.json.
The speed-up is search only, as the bars (user, 26 Oct: Spindle-naive timed without its query logs, like for like).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import patheffects
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import style as ps  # noqa: E402

FIG = "fig4_gene_set"
RES = ps.ROOT / "results" / FIG
ANN = ps.FIGURES / FIG / "annotations.json"
KEYS = ps.DATASET_ORDER
ROW_H = 62.0
BOTTOM, TOP = 9.0, 1.5
SPINDLE, NAIVE, WHOLE = ps.TIER_COLORS["spindle_partial"], ps.TIER_COLORS["naive"], ps.TIER_COLORS["whole"]
BASE = {"padding": "#44AA99", "imputation": "#882255"}
METHOD_LABEL = {"spindle_partial": "Spindle", "padding": "Padding + Spindle-naive",
                "imputation": "Imputation + Spindle-naive"}
MARK = {"spindle_partial": "^", "padding": "s", "imputation": "o"}
COLOR = {"spindle_partial": SPINDLE, **BASE}
annotations = {}


def to_mm(fig, ax, x, y):
    fx, fy = fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    w, h = fig.get_size_inches() / ps.MM
    return float(fx * w), float((1 - fy) * h)


def note(atom, text, x_mm, y_mm):
    annotations.setdefault(atom, []).append([text, round(x_mm, 2), round(y_mm, 2)])


def rows_axis(ax, labels):
    ax.set_ylim(len(KEYS) - 0.5, -0.5)
    ax.set_yticks(range(len(KEYS)))
    ax.set_yticklabels([ps.DATASET_LABELS[k] for k in KEYS] if labels else [])
    ax.tick_params(axis="y", length=0)


# ---------------------------------------------------------------- A, B, C
def band():
    z = np.load(RES / "band" / "breast_cancer.npz")
    d, found = z["d"], z["found"]
    fig, ax = ps.atom(56, 46, left=10.0, right=2.0, bottom=BOTTOM, top=TOP)
    bins = np.linspace(d.min(), np.quantile(d, 0.995), 50)
    ax.hist(d, bins=bins, color=ps.LIGHT, lw=0, zorder=2)
    ax.hist(d[found], bins=bins, color=SPINDLE, lw=0, zorder=3)
    ax.set_xlabel(r"Gene-set distance $d^{S}_{\mathcal{B}}$ to the query")
    ax.set_ylabel("Index tiles")
    ax.set_xlim(bins[0], bins[-1])
    ps.save(fig, FIG, "band", source=f"results/{FIG}/band/breast_cancer.npz")


def by_size(name, col, scale, ylabel, ylim):
    t = pd.read_csv(RES / "nx_size.csv")
    t = t[t["size"] > 0]
    fig, ax = ps.atom(56, 46, left=10.0, right=2.0, bottom=BOTTOM, top=TOP)
    for m in ("padding", "imputation", "spindle_partial"):
        s = t[t.method == m].sort_values("size")
        ax.errorbar(s["size"], scale * s[col], yerr=scale * s[f"{col}_sd"], fmt=f"-{MARK[m]}", ms=2.8,
                    color=COLOR[m], elinewidth=0.5, zorder=3 if m == "spindle_partial" else 2)
    ax.set_xscale("log", base=2)
    ax.set_xticks([4, 8, 16, 32], ["4", "8", "16", "32"])
    ax.minorticks_off()
    ax.set_ylim(*ylim)
    ax.set_xlabel("Genes in the query, |S| (log scale)")
    ax.set_ylabel(ylabel)
    ps.save(fig, FIG, name, source=f"results/{FIG}/nx_size.csv")


def nx_size():
    by_size("nx_size", "NX5", 100, "NX at 5 % of N (%)", (0, None))


def overlap_size():
    by_size("overlap_size", "overlap5", 1, "Overlap(5 %, 5 % of N)", (0, 1.0))


def nx_rows():
    t = pd.read_csv(RES / "nx.csv")
    t = t[(t.method == "spindle_partial") & (t["size"] == 0)].set_index("key")
    fig, ax = ps.atom(44, ROW_H, left=2.0, right=3.0, bottom=BOTTOM, top=TOP)
    for i, k in enumerate(KEYS):
        if k in t.index:
            r = t.loc[k]
            ax.errorbar(100 * r.NX5, i, xerr=100 * r.NX5_sd, fmt="^", ms=3.0, color=SPINDLE, elinewidth=0.5, zorder=3)
    rows_axis(ax, False)
    ax.set_xlim(0, None)
    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel("NX at 5 % of N (%)")
    ps.save(fig, FIG, "nx", source=f"results/{FIG}/nx.csv")


# ---------------------------------------------------------------- D, E, F
HALO = [patheffects.withStroke(linewidth=1.6, foreground="white")]  # query-log markers stay visible on a bar


def timing():
    t = pd.read_csv(RES / "timing.csv").set_index("key")
    fig, ax = ps.atom(68, ROW_H, left=17.0, right=12.0, bottom=BOTTOM, top=TOP)
    bh = 0.34
    for i, k in enumerate(KEYS):
        if k not in t.index:
            continue
        r = t.loc[k]
        for off, col, qcol, c in ((-bh / 2, "naive_partial_search_ms", "qlog_naive_partial_ms", NAIVE),
                                  (bh / 2, "spindle_partial_ms", "qlog_spindle_partial_ms", SPINDLE)):
            ax.barh(i + off, r[col], height=bh * 0.92, color=c, lw=0, zorder=2)
            ax.errorbar(r[col], i + off, xerr=r[f"{col}_sd"], fmt="none", ecolor=ps.INK, elinewidth=0.5, zorder=3)
            ax.plot(r[qcol], i + off, "o", ms=2.6, mfc="white", mec=c, mew=0.7, zorder=4, path_effects=HALO)
    ax.set_xscale("log")
    ps.plain_log(ax, "x", subs=(1.0,))
    ps.log_grid(ax, "x")
    lo = min(t.spindle_partial_ms.min(), t.qlog_spindle_partial_ms.min(), t.qlog_naive_partial_ms.min()) / 2
    hi = max(t.naive_partial_search_ms.max(), t.qlog_naive_partial_ms.max(),
             t.qlog_spindle_partial_ms.max()) * 2
    ax.set_xlim(lo, hi)
    rows_axis(ax, True)
    ax.set_xlabel("Time per query (ms, log scale)")
    for i, k in enumerate(KEYS):
        if k in t.index:
            x, y = to_mm(fig, ax, hi, i)
            note("timing", f"{t.loc[k].speedup_spindle_partial:.0f}×", x + 1.5, y)
    ps.save(fig, FIG, "timing", source=f"results/{FIG}/timing.csv")


STORE = [("whole", WHOLE, "s"), ("naive_partial", NAIVE, "o"), ("imputation", BASE["imputation"], "o"),
         ("padding", BASE["padding"], "s"), ("spindle_partial", SPINDLE, "^")]


def storage():
    t = pd.read_csv(RES / "storage.csv").set_index("key")
    fig, ax = ps.atom(50, ROW_H, left=2.0, right=3.0, bottom=BOTTOM, top=TOP)
    for i, k in enumerate(KEYS):
        if k not in t.index:
            continue
        r = t.loc[k]
        ax.plot([r.spindle_partial, r.whole], [i, i], color=ps.LIGHT, lw=0.75, zorder=1)
        for col, c, mk in STORE:
            ax.plot(r[col], i, mk, ms=3.0 if mk != "^" else 3.3, color=c, mew=0,
                    zorder=3 if col == "spindle_partial" else 2)
    ax.set_xscale("log")
    ps.plain_log(ax, "x", subs=(1.0,))
    ps.log_grid(ax, "x")
    rows_axis(ax, False)
    ax.set_xlabel("Stored MB (log scale)")
    ps.save(fig, FIG, "storage", source=f"results/{FIG}/storage.csv")


def legends():
    nxh = [Line2D([], [], marker=MARK[m], ms=2.8, color=COLOR[m], label=METHOD_LABEL[m])
           for m in ("spindle_partial", "padding", "imputation")]
    ps.legend_atom(FIG, "legend_nx", nxh, 120, 6, ncol=3)
    tm = [Patch(color=NAIVE, label="Spindle-naive"), Patch(color=SPINDLE, label="Spindle"),
          Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=NAIVE, mew=0.7, label="Query's block logs"),
          Line2D([], [], ls="", marker="o", ms=2.6, mfc="white", mec=SPINDLE, mew=0.7, label="Query's piece logs")]
    ps.legend_atom(FIG, "legend_timing", tm, 70, 8, ncol=2)
    lab = {"whole": "Whole covariances", "naive_partial": "Spindle-naive",
           "imputation": "Imputation (Spindle-naive + niche means)", "padding": "Padding (Spindle-naive)",
           "spindle_partial": "Spindle"}
    st = [Line2D([], [], ls="", marker=mk, ms=3.0, color=c, label=lab[col]) for col, c, mk in STORE]
    ps.legend_atom(FIG, "legend_storage", st, 90, 8, ncol=3)


if __name__ == "__main__":
    for f, src in ((band, "band/breast_cancer.npz"), (nx_size, "nx_size.csv"), (overlap_size, "nx_size.csv"),
                   (nx_rows, "nx.csv"), (timing, "timing.csv"), (storage, "storage.csv")):
        if (RES / src).exists():
            f()
    legends()
    ANN.write_text(json.dumps(annotations, indent=1, ensure_ascii=False))
