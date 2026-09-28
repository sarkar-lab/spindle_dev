"""Stage 1 vs Stage 1 + Stage 2 vs Stage 2 only: plots (results/stage_benchmark/plots/{pdf,png}/).

Exploratory, not a manuscript figure. Block-LE ground truth throughout.
1. quality: Recall@0.1/0.5 eps and Overlap@0.5/1 eps against the number of Stage-1 candidates c,
   Stage 1 alone (dashed) and Stage 1 + 2 (solid), one colour per dataset. At equal c both
   return the same tiles, so their Overlap is identical; only the order (Recall) differs.
2. time: time per query against c for Stage 1 and Stage 1 + 2, with Stage 2 only (no stored
   logs) as a flat reference, one panel per dataset.
3. storage: Stage 1 (DAG), Stage 1 + 2 (DAG + block logs), Stage 2 only (raw diagonal blocks).

Inputs: results/stage_benchmark/<dataset>/{summary,timing,storage}.csv
Usage:  python scripts/stage_benchmark_plots.py [--datasets <stem> ...]
"""

import argparse

import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

import figstyle as fs

IN_DIR = fs.RESULTS / "stage_benchmark"
OUT_DIR = IN_DIR / "plots"
QUALITY = [("recall_at_eps_0.1", r"Recall@0.1$\varepsilon$"), ("recall_at_eps_0.5", r"Recall@0.5$\varepsilon$"),
           ("overlap_at_eps_0.5", r"Overlap@0.5$\varepsilon$"), ("overlap_at_eps_1.0", r"Overlap@1$\varepsilon$")]
MODE_LS = {"stage1": "--", "stage1_2": "-"}
MODE_LABEL = {"stage1": "Stage 1 (DAG path score)", "stage1_2": "Stage 1 + 2 (exact re-rank)"}
T_COLORS = {"stage1": fs.QUERY_COLOR, "stage1_2": "#4477AA", "stage2_only": "#555555"}
T_LABELS = {"stage1": "Stage 1", "stage1_2": "Stage 1 + 2", "stage2_only": "Stage 2 only (logs per query)"}


def save(fig, name):
    for ext in ("pdf", "png"):
        d = OUT_DIR / ext
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / f"{name}.{ext}", dpi=fs.PNG_DPI if ext == "png" else 600)
    fs.plt.close(fig)
    print(f"saved {(OUT_DIR / 'pdf' / name).relative_to(fs.PROJECT_ROOT)}.pdf (+ .png)")


def load(datasets):
    names = ("summary", "timing", "storage")
    parts = {n: [] for n in names}
    dirs = [IN_DIR / d for d in datasets] if datasets else sorted(
        p for p in IN_DIR.iterdir() if p.is_dir() and p.name != "plots")
    for d in dirs:
        if all((d / f"{n}.csv").exists() for n in names):
            for n in names:
                parts[n].append(pd.read_csv(d / f"{n}.csv"))
        else:
            print(f"skip {d.name}: incomplete")
    if not parts["storage"]:
        raise SystemExit(f"no results under {IN_DIR}")
    res = {n: pd.concat(v, ignore_index=True) for n, v in parts.items()}
    for df in res.values():
        df["key"] = df["Dataset"].map(fs.stem_to_key)
    res["keys"] = [k for k in fs.DATASET_ORDER if k in set(res["storage"]["key"])]
    return res


def c_axis(ax, cs):
    ax.set_xscale("log")
    ax.set_xticks(cs)
    ax.set_xticklabels([str(c) for c in cs])
    ax.tick_params(axis="x", which="minor", length=0)
    ax.set_xlim(cs[0] * 0.8, cs[-1] * 1.25)
    ax.set_xlabel("Stage-1 candidates c")


def quality_figure(res):
    s = res["summary"]
    cs = sorted(s["top_c"].unique())
    fig = fs.figure(fs.DOUBLE, 62)
    gs = GridSpec(1, len(QUALITY), figure=fig, wspace=0.35, left=0.06, right=0.99, top=0.92, bottom=0.36)
    for j, (col, lab) in enumerate(QUALITY):
        ax = fig.add_subplot(gs[0, j])
        for key in res["keys"]:
            for mode, ls in MODE_LS.items():
                d = s[(s["key"] == key) & (s["mode"] == mode)].sort_values("top_c")
                ax.plot(d["top_c"], d[col], color=fs.DATASET_COLORS[key], ls=ls, lw=0.9,
                        marker=fs.DATASET_MARKERS[key] if mode == "stage1_2" else None, ms=2.5)
        c_axis(ax, cs)
        ax.set_ylim(0, 1.03)
        ax.set_ylabel(lab)
        fs.label_panel(fig, ax, "ABCD"[j])
    handles = fs.dataset_handles(res["keys"]) + [Line2D([], [], color=fs.MUTED, ls=ls, lw=0.9, label=MODE_LABEL[m])
                                                 for m, ls in MODE_LS.items()]
    fs.legend_below(fig, handles, ncol=min(5, len(handles)), y=0.1)
    save(fig, "quality_vs_c")


def time_figure(res):
    t = res["timing"]
    cs = sorted(int(c.split("_c")[1].replace("_ms", "")) for c in t.columns if c.startswith("stage1_c"))
    keys = res["keys"]
    ncol = min(4, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig = fs.figure(fs.DOUBLE, 45 * nrow + 20)
    gs = GridSpec(nrow, ncol, figure=fig, hspace=0.7, wspace=0.35, left=0.07, right=0.99, top=0.93,
                  bottom=0.3 if nrow == 1 else 0.16)
    for n, key in enumerate(keys):
        ax = fig.add_subplot(gs[n // ncol, n % ncol])
        tk = t[t["key"] == key].mean(numeric_only=True)
        s1 = [tk[f"stage1_c{c}_ms"] for c in cs]
        ax.plot(cs, s1, color=T_COLORS["stage1"], marker="o", ms=2.5, lw=0.9)
        ax.plot(cs, [a + tk[f"rerank_c{c}_ms"] for a, c in zip(s1, cs)], color=T_COLORS["stage1_2"], marker="s",
                ms=2.5, lw=0.9)
        ax.axhline(tk["stage2_only_ms"], color=T_COLORS["stage2_only"], ls=":", lw=1.0)
        c_axis(ax, cs)
        ax.set_yscale("log")
        fs.plain_log(ax, subs=(1.0, 2.0, 5.0))
        fs.log_ygrid(ax)
        ax.set_title(fs.DATASET_LABELS[key])
        if n % ncol == 0:
            ax.set_ylabel("Time per query (ms)")
    handles = [Line2D([], [], color=T_COLORS[m], ls=":" if m == "stage2_only" else "-",
                      marker=None if m == "stage2_only" else ("o" if m == "stage1" else "s"), ms=3, lw=1.0,
                      label=T_LABELS[m]) for m in T_COLORS]
    fs.legend_below(fig, handles, ncol=3, y=0.08 if nrow == 1 else 0.04)
    save(fig, "time_vs_c")


def storage_figure(res):
    st = res["storage"].set_index("key")
    keys = res["keys"]
    x = np.arange(len(keys))
    fig = fs.figure(fs.SINGLE, 62)
    ax = fig.add_axes([0.16, 0.26, 0.8, 0.7])
    for j, (col, m) in enumerate([("stage1_mb", "stage1"), ("stage1_2_mb", "stage1_2"), ("stage2_only_mb", "stage2_only")]):
        ax.plot(x + (j - 1) * 0.22, [st.loc[k, col] for k in keys], color=T_COLORS[m], marker="osv"[j], ms=3.5,
                ls="none", label={"stage1": "Stage 1 (DAG)", "stage1_2": "Stage 1 + 2 (DAG + block logs)",
                                  "stage2_only": "Stage 2 only (raw diagonal blocks)"}[m])
    ax.set_xticks(x)
    ax.set_xticklabels([fs.DATASET_SHORT[k] for k in keys], rotation=40, ha="right")
    ax.set_xlim(-0.6, len(keys) - 0.4)
    ax.set_yscale("log")
    fs.plain_log(ax, subs=(1.0, 2.0, 5.0))
    fs.log_ygrid(ax)
    ax.set_ylabel("Storage (MB, float32)")
    ax.legend(loc="upper left", fontsize=fs.TICK_PT)
    save(fig, "storage")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--datasets", nargs="*", default=None, help="stems under results/stage_benchmark/ (default: all)")
    args = ap.parse_args()
    res = load(args.datasets)
    quality_figure(res)
    time_figure(res)
    storage_figure(res)


if __name__ == "__main__":
    main()
