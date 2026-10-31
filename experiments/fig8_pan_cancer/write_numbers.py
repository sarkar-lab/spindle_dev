"""Every number quoted in the Fig 8 caption and Results text -> results/fig8_pan_cancer/numbers.json.

Usage: python experiments/fig8_pan_cancer/write_numbers.py   (after c_summary.py)
Pairs = section x programme (5 x 4 = 20); every value is the mean over seeds 0-4 (programs.csv). A gain counts as
positive above GAIN_EPS (gains of programmes that co-vary in no tile are 0 to the fourth decimal).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common8 as c8  # noqa: E402
from common8 import data  # noqa: E402

GAIN_EPS = 1e-3


def r(v, n=3):
    return round(float(v), n)


def main():
    T = pd.read_csv(c8.OUT / "programs.csv")
    E = pd.read_csv(c8.OUT / "exemplar.csv")
    m = {k: T[T.method == k].set_index(["key", "program"]) for k in ("naive_partial", "spindle_partial", "D", "X")}
    NV, SP, D, X = m["naive_partial"], m["spindle_partial"], m["D"], m["X"]
    pos = NV.gain > GAIN_EPS
    pairs = lambda s: [f"{k}/{p}" for k, p in s[s].index]  # noqa: E731
    out = {
        "stamp": data.stamp(),
        "pairs": len(NV),
        "hits_c": {k: int(NV.xs(k).c.iloc[0]) for k in c8.KEYS},
        "index_tiles": {k: r(NV.xs(k).n_index.iloc[0], 1) for k in c8.KEYS},
        "exemplars": {e.program: {"tile": int(e.tile), "cells": int(e.n_cells), "c": r(e.c), "mean_r": r(e.mean_r),
                                  "n_robust": int(e.n_robust), "genes": int(e.n_genes)} for e in E.itertuples()},
        "absent": pairs(NV.absent.astype(bool)),
        "naive_partial": {"gain_pos": int(pos.sum()), "gain_pos_range": [r(NV.gain[pos].min()), r(NV.gain[pos].max())],
                          "gain_zero": pairs(~pos), "delta_pos": int((NV.delta > 0).sum()),
                          "lift_pos": int((NV.lift > 0).sum())},
        "spindle_partial": {"gain_pos": int((SP.gain > GAIN_EPS).sum()), "delta_pos": int((SP.delta > 0).sum()),
                            "short_of_c": int((SP.n_returned < SP.c - 1e-9).sum()),
                            "returned_min": r(SP.n_returned.min(), 1),
                            "returned_min_pair": pairs(SP.n_returned == SP.n_returned.min()),
                            "jaccard_with_naive_partial_median": r(SP.jacc_naive_partial.median()),
                            "gain_within_0.01_of_naive_partial": int(((SP.gain - NV.gain).abs() <= 0.01).sum())},
        "controls": {"variances_only_gain_below_naive_partial": int((D.gain < NV.gain - GAIN_EPS).sum()),
                     "variances_only_gain_pos": int((D.gain > GAIN_EPS).sum()),
                     "jaccard_naive_partial_variances_median": r(NV.jacc_D.median()),
                     "jaccard_naive_partial_variances_range": [r(NV.jacc_D.min()), r(NV.jacc_D.max())],
                     "expression_gain_above_naive_partial": int((X.gain > NV.gain + GAIN_EPS).sum()),
                     "jaccard_naive_partial_expression_median": r(NV.jacc_X.median())},
        "gain_eps": GAIN_EPS,
    }
    ok = (pos & (NV.delta > 0)).groupby(level="program").all()
    out["shown_rule"] = "gain > gain_eps and delta > 0 (Spindle-naive) in all five sections"
    out["shown"] = {}
    for prog in [p for p in c8.PROGRAMS if ok[p]]:
        nv, sp, dd = NV.xs(prog, level="program"), SP.xs(prog, level="program"), D.xs(prog, level="program")
        out["shown"][prog] = {"gain_range": [r(nv.gain.min()), r(nv.gain.max())],
                              "delta_range": [r(nv.delta.min(), 2), r(nv.delta.max(), 2)],
                              "spindle_partial_gain_range": [r(sp.gain.min()), r(sp.gain.max())],
                              "spindle_partial_returned_range": [r(sp.n_returned.min(), 1), r(sp.n_returned.max(), 1)],
                              "variances_only_below": int((dd.gain < nv.gain - GAIN_EPS).sum()),
                              "lift_range": [r(nv.lift.min()), r(nv.lift.max())],
                              "spindle_partial_lift_range": [r(sp.lift.min()), r(sp.lift.max())],
                              "variances_only_lift_below": int((dd.lift < nv.lift).sum()),
                              "variances_only_delta_above": int((dd.delta > nv.delta).sum())}
    out["not_shown"] = [p for p in c8.PROGRAMS if not ok[p]]
    sh = list(out["shown"])
    NVs, SPs, Xs = (t[t.index.get_level_values("program").isin(sh)] for t in (NV, SP, X))
    out["shown_all"] = {"spindle_partial_short_of_c": int((SPs.n_returned < SPs.c - 1e-9).sum()), "pairs": len(NVs),
                        "spindle_partial_returned_range": [r(SPs.n_returned.min(), 1), r(SPs.n_returned.max(), 1)],
                        "expression_gain_above": int((Xs.gain > NVs.gain + GAIN_EPS).sum()),
                        "expression_lift_above": int((Xs.lift > NVs.lift).sum())}
    if len(sh) == 2:  # how differently the two shown queries rank the tiles
        from scipy.stats import spearmanr
        rho, jac = {}, {}
        for k in c8.KEYS:
            z = np.load(c8.OUT / k / f"seed{data.EXAMPLE_SEED}" / "gradient.npz", allow_pickle=True)
            rho[k] = r(spearmanr(z[f"d_{sh[0]}"], z[f"d_{sh[1]}"])[0], 2)
            js = []
            for s in data.SEEDS:
                h = pd.read_csv(c8.OUT / k / f"seed{s}" / "hits.csv")
                a, b = (set(h[(h.method == "naive_partial") & (h.program == p)].tile) for p in sh)
                js.append(len(a & b) / len(a | b))
            jac[k] = r(np.mean(js))
        out["shown_pair"] = {"spearman_seed0": rho, "hits_jaccard": jac}
    for prog in out["not_shown"]:
        nv = NV.xs(prog, level="program")
        out.setdefault("not_shown_detail", {})[prog] = {
            "gain_pos": [k for k in c8.KEYS if nv.gain[k] > GAIN_EPS], "gain_range_pos": [
                r(nv.gain[nv.gain > GAIN_EPS].min()), r(nv.gain[nv.gain > GAIN_EPS].max())],
            "delta_le0": {k: r(nv.delta[k], 2) for k in c8.KEYS if nv.delta[k] <= 0}}
    with open(c8.OUT / "numbers.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
