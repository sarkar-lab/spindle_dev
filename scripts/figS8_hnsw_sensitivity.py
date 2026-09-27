"""Fig. S8 -- FAISS-HNSW sensitivity to M and efSearch (E4).

Breast and lung, seed-73 build, 100 held-out queries; 400 candidates re-ranked exactly as
for Spindle. efSearch below the 400 candidates requested has no effect in FAISS, so the
grid starts at 400. Spindle's Recall@0.1 eps on the same queries is the dashed line.

Inputs: results/ann_baselines/{hnsw_sensitivity.csv, summary.csv}
"""

import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import figstyle as fs

M_COLORS = {16: "#88CCEE", 32: "#4477AA", 64: "#332288"}


def main():
    h = pd.read_csv(fs.RESULTS / "ann_baselines" / "hnsw_sensitivity.csv")
    f = pd.read_csv(fs.RESULTS / "ann_baselines" / "summary.csv")
    keys = ["breast_cancer", "lung_cancer"]
    fig = fs.figure(fs.SINGLE, 55)
    gs = GridSpec(1, 2, figure=fig, wspace=0.15, left=0.13, right=0.98, top=0.88, bottom=0.36)
    axes = []
    for j, key in enumerate(keys):
        ax = fig.add_subplot(gs[j])
        d = h[h["Dataset"] == f"xenium_human_{key}"]
        for m, c in M_COLORS.items():
            g = d[d["hnsw_M"] == m].sort_values("efSearch")
            ax.plot(g["efSearch"], g["recall_at_eps_0.1"], marker="o", ms=2.8, mew=0, color=c, lw=0.8)
        sp = f[(f["method"] == "spindle") & (f["dataset"] == f"xenium_human_{key}")]["recall_at_eps_0.1"].iloc[0]
        ax.axhline(sp, color=fs.METHOD_COLORS["spindle"], ls="--", lw=0.7)
        ax.set_xscale("log")
        ax.set_xticks([400, 800, 1600], ["400", "800", "1600"])
        ax.minorticks_off()
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("efSearch")
        ax.set_title(fs.DATASET_LABELS[key], fontsize=fs.TEXT_PT)
        if j == 0:
            ax.set_ylabel(r"Recall@0.1$\varepsilon$")
        else:
            ax.set_yticklabels([])
        axes.append(ax)
    handles = [Line2D([], [], color=c, marker="o", ms=2.8, lw=0.8, label=f"HNSW, M = {m}") for m, c in M_COLORS.items()]
    handles.append(Line2D([], [], color=fs.METHOD_COLORS["spindle"], ls="--", lw=0.7, label="Spindle"))
    fs.legend_below(fig, handles, ncol=2, y=0.14)
    fs.save(fig, "figS8_hnsw_sensitivity")


if __name__ == "__main__":
    main()
