"""Every number Fig 7's caption and text quote -> results/fig7_skin/numbers.json.

Usage (login node or anywhere): python experiments/fig7_skin/write_numbers.py
Inputs: results/fig7_skin/skin_melanoma/seed0/ (A, B: the worked example) and the summaries of c_summary.py (C-E:
seeds 0-4). Across splits: mean over seeds 0-4 of the per-seed value; ranges are over datasets or signatures. The
controls (D = variances only, X = top signature score, the expression top 10) are reported, never claimed against.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402

RES = data.RESULTS / "fig7_skin"
S0 = RES / "skin_melanoma" / f"seed{data.EXAMPLE_SEED}"
OUT = RES / "numbers.json"
SKIN = "skin_melanoma"


def r3(x):
    return round(float(x), 3)


def rng(s, f=r3):
    return [f(s.min()), f(s.max())]


def named(s):
    return {f"{k}/{n}": r3(v) for (k, n), v in s.items()}


def main():
    ex = pd.read_csv(S0 / "example.csv").set_index("set")
    classes = pd.read_csv(S0 / "skin_classes.csv", dtype={"type": str})
    tq = pd.read_csv(S0 / "tile_queries.csv")
    names = classes["class"].unique().tolist()
    example = {"query_tile": int(ex.query_tile.iloc[0]), "candidates": int(ex.n_candidates.iloc[0]),
               "queries": int(ex.n_queries.iloc[0]), "base_rate": r3(ex.base_rate.iloc[0]),
               "n_below": int((tq.tv_top < tq.tv_random).sum()),
               "tv_top_median": r3(tq.tv_top.median()), "tv_random_median": r3(tq.tv_random.median()),
               "composition": {s: {n: r3(ex.loc[s, n]) for n in ["Melanocyte", "Keratinocyte", "Immune", "Other"]}
                               for s in ex.index},
               "clusters_per_class": classes.groupby("class").size().to_dict(),
               "cells_per_class": classes.groupby("class").n_cells.sum().to_dict()}

    L = pd.read_csv(RES / "lists.csv")
    q = {k: L[L["query"] == k].set_index(["key", "list"]) for k in ("T", "TL", "D", "X")}
    t, tl, d, x = q["T"], q["TL"].loc[q["T"].index], q["D"].loc[q["T"].index], q["X"].loc[q["T"].index]
    lists = {"n": len(t), "per_dataset": t.groupby(level=0).size().to_dict(),
             "dropped_genes": {f"{k}/{n}": m for (k, n), m in t.missing.items() if isinstance(m, str) and m},
             "T_delta": rng(t.delta), "T_delta_gt0": int((t.delta > 0).sum()),
             "T_delta_gt05": int((t.delta > 0.5).sum()), "T_delta_le05": named(t.delta[t.delta <= 0.5]),
             "T_lift_gt0": int((t.lift > 0).sum()), "T_lift_le0": named(t.lift[t.lift <= 0]), "T_lift": rng(t.lift),
             "TL_delta": rng(tl.delta), "TL_delta_gt0": int((tl.delta > 0).sum()),
             "TL_delta_gt05": int((tl.delta > 0.5).sum()), "TL_lift_gt0": int((tl.lift > 0).sum()),
             "TL_delta_minus_T": rng(tl.delta - t.delta), "TL_delta_minus_T_median": r3((tl.delta - t.delta).median()),
             "TL_lift_minus_T_median": r3((tl.lift - t.lift).median()),
             "TL_returned_lt10": named(tl.n_returned[tl.n_returned < 10]),
             "TL_jacc_T_median": r3(tl.jacc_T.median()),
             "D_lift_gt0": int((d.lift > 0).sum()), "X_cov_above_T": int((x.cov_top > t.cov_top).sum()),
             "jacc_TD": rng(t.jacc_D), "jacc_TD_ge05": int((t.jacc_D >= 0.5).sum()),
             "skin": {n: {"genes": int(t.loc[(SKIN, n), "n_genes"]), "delta": r3(t.loc[(SKIN, n), "delta"]),
                          "cov_top": r3(t.loc[(SKIN, n), "cov_top"]), "cov_all": r3(t.loc[(SKIN, n), "cov_all"]),
                          "TL_delta": r3(tl.loc[(SKIN, n), "delta"]), "TL_returned": r3(tl.loc[(SKIN, n), "n_returned"]),
                          "D_delta": r3(d.loc[(SKIN, n), "delta"])}
                      for n in t.loc[SKIN].index}}

    G = pd.read_csv(RES / "tiles.csv")
    gm = G.groupby("key", sort=False)[["tv_top", "tv_lite", "tv_random", "tv_expr", "lite_in_top", "below_random",
                                       "lite_below_random"]].mean()
    tiles = {"per_dataset": {k: {c: r3(v) for c, v in row.items()} for k, row in gm.iterrows()},
             "tv_top": rng(gm.tv_top), "tv_lite": rng(gm.tv_lite), "tv_random": rng(gm.tv_random),
             "tv_expr": rng(gm.tv_expr),
             "below_random_pct_per_seed": rng(100 * G.below_random, lambda v: round(float(v), 1)),
             "lite_below_random_pct_per_seed": rng(100 * G.lite_below_random, lambda v: round(float(v), 1)),
             "queries_per_seed": rng(G.n_queries, int), "lite_in_top_median": rng(G.lite_in_top, int),
             "breast_check": {"tv_top": r3(gm.loc["breast_cancer", "tv_top"]),
                              "tv_random": r3(gm.loc["breast_cancer", "tv_random"])}}
    out = {"stamp": data.stamp(), "example": example, "lists": lists, "tiles": tiles}
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
