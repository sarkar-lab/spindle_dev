"""Plots of the gene-set constructor pilot (c_pilot.py), login node.

Usage: python experiments/gene_set_pilot/f_pilot.py
Reads results/gene_set_pilot/<key>/lists.csv; writes results/gene_set_pilot/{summary.csv, pilot.png,
pilot_paired.png}. One point per (dataset, list) and constructor, coloured by dataset; bar = median.
summary.csv: per constructor the median of every readout and the share of lists with lift_c > 0 and delta > 0.
pilot_paired.png: per list, E minus T (and T minus D) on lift_c, delta and Jaccard with D.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]
import paths  # noqa: E402

RES = ROOT / "results" / "gene_set_pilot"
ORDER = ["T", "T75", "T50", "E", "D", "X"]
OKABE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
PANELS = [("lift_c", "co-variation lift, cells (median c top 10 − all)"),
          ("lift_rs", "co-variation lift, shrunk cov (r_S)"),
          ("delta", "tile score Cliff's δ (top 10 vs rest)"),
          ("spec", "composition: TV(top 10) − TV(random 10)"),
          ("var_level", "variance percentile of the hits"),
          ("jacc_D", "Jaccard with D's top 10")]


def main():
    keys = paths.indexed_keys()
    df = pd.concat([pd.read_csv(RES / k / "lists.csv") for k in keys if (RES / k / "lists.csv").exists()])
    keys = [k for k in keys if k in set(df.dataset)]
    col = dict(zip(keys, OKABE))
    summ = df.groupby("constructor").agg(
        n=("list", "size"), lift_c=("lift_c", "median"), frac_lift_pos=("lift_c", lambda x: (x > 0).mean()),
        lift_rs=("lift_rs", "median"), delta=("delta", "median"), frac_delta_pos=("delta", lambda x: (x > 0).mean()),
        spec=("spec", "median"), var_level=("var_level", "median"), jacc_D=("jacc_D", "median"),
        jacc_X=("jacc_X", "median")).reindex(ORDER)
    summ.to_csv(RES / "summary.csv")
    print(summ.round(3).to_string())

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, (m, lab) in zip(axes.flat, PANELS):
        for i, c in enumerate(ORDER):
            sub = df[df.constructor == c]
            x = i + rng.uniform(-0.25, 0.25, len(sub))
            ax.scatter(x, sub[m], s=9, c=[col[d] for d in sub.dataset], lw=0, alpha=0.8)
            if len(sub):
                ax.hlines(sub[m].median(), i - 0.35, i + 0.35, color="k", lw=1.5)
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.set_xticks(range(len(ORDER)), ORDER)
        ax.set_title(lab, fontsize=9)
    handles = [plt.Line2D([], [], marker="o", ls="", color=col[k], label=k) for k in keys]
    fig.legend(handles=handles, loc="lower center", ncol=len(keys), fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(RES / "pilot.png", dpi=150)

    w = df.pivot_table(index=["dataset", "list"], columns="constructor", values=["lift_c", "delta", "jacc_D"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (m, lab) in zip(axes, [("lift_c", "co-variation lift"), ("delta", "Cliff's δ"), ("jacc_D", "Jaccard with D")]):
        for a, b, mk in (("T", "E", "o"), ("D", "T", "^")):
            if (m, a) in w and (m, b) in w:
                ok = w[(m, a)].notna() & w[(m, b)].notna()
                ds = w.index.get_level_values("dataset")[ok]
                ax.scatter(w[(m, a)][ok], w[(m, b)][ok], c=[col[d] for d in ds], marker=mk, s=16, lw=0,
                           label=f"{b} vs {a}")
        lo, hi = ax.get_xlim()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.5)
        ax.set_xlabel("x: T (circles) / D (triangles)")
        ax.set_ylabel("y: E (circles) / T (triangles)")
        ax.set_title(lab, fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "pilot_paired.png", dpi=150)


if __name__ == "__main__":
    main()
