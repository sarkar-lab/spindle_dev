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
    m = {k: T[T.method == k].set_index(["key", "program"]) for k in ("P", "PL", "D", "X")}
    P, PL, D, X = m["P"], m["PL"], m["D"], m["X"]
    pos = P.gain > GAIN_EPS
    pairs = lambda s: [f"{k}/{p}" for k, p in s[s].index]  # noqa: E731
    out = {
        "stamp": data.stamp(),
        "pairs": len(P),
        "hits_c": {k: int(P.xs(k).c.iloc[0]) for k in c8.KEYS},
        "index_tiles": {k: r(P.xs(k).n_index.iloc[0], 1) for k in c8.KEYS},
        "exemplars": {e.program: {"tile": int(e.tile), "cells": int(e.n_cells), "c": r(e.c), "mean_r": r(e.mean_r),
                                  "n_robust": int(e.n_robust), "genes": int(e.n_genes)} for e in E.itertuples()},
        "absent": pairs(P.absent.astype(bool)),
        "spindle": {"gain_pos": int(pos.sum()), "gain_pos_range": [r(P.gain[pos].min()), r(P.gain[pos].max())],
                    "gain_zero": pairs(~pos), "delta_pos": int((P.delta > 0).sum()),
                    "lift_pos": int((P.lift > 0).sum())},
        "lite": {"gain_pos": int((PL.gain > GAIN_EPS).sum()), "delta_pos": int((PL.delta > 0).sum()),
                 "short_of_c": int((PL.n_returned < PL.c - 1e-9).sum()),
                 "returned_min": r(PL.n_returned.min(), 1), "returned_min_pair": pairs(PL.n_returned ==
                                                                                        PL.n_returned.min()),
                 "jaccard_with_spindle_median": r(PL.jacc_P.median()),
                 "gain_within_0.01_of_spindle": int(((PL.gain - P.gain).abs() <= 0.01).sum())},
        "controls": {"variances_only_gain_below_spindle": int((D.gain < P.gain - GAIN_EPS).sum()),
                     "variances_only_gain_pos": int((D.gain > GAIN_EPS).sum()),
                     "jaccard_spindle_variances_median": r(P.jacc_D.median()),
                     "jaccard_spindle_variances_range": [r(P.jacc_D.min()), r(P.jacc_D.max())],
                     "expression_gain_above_spindle": int((X.gain > P.gain + GAIN_EPS).sum()),
                     "jaccard_spindle_expression_median": r(P.jacc_X.median())},
        "gain_eps": GAIN_EPS,
    }
    ok = (pos & (P.delta > 0)).groupby(level="program").all()
    out["shown_rule"] = "gain > gain_eps and delta > 0 (Spindle-partial) in all five sections"
    out["shown"] = {}
    for prog in [p for p in c8.PROGRAMS if ok[p]]:
        p, pl, dd = P.xs(prog, level="program"), PL.xs(prog, level="program"), D.xs(prog, level="program")
        out["shown"][prog] = {"gain_range": [r(p.gain.min()), r(p.gain.max())],
                              "delta_range": [r(p.delta.min(), 2), r(p.delta.max(), 2)],
                              "lite_gain_range": [r(pl.gain.min()), r(pl.gain.max())],
                              "lite_returned_range": [r(pl.n_returned.min(), 1), r(pl.n_returned.max(), 1)],
                              "variances_only_below": int((dd.gain < p.gain - GAIN_EPS).sum()),
                              "lift_range": [r(p.lift.min()), r(p.lift.max())],
                              "lite_lift_range": [r(pl.lift.min()), r(pl.lift.max())],
                              "variances_only_lift_below": int((dd.lift < p.lift).sum()),
                              "variances_only_delta_above": int((dd.delta > p.delta).sum())}
    out["not_shown"] = [p for p in c8.PROGRAMS if not ok[p]]
    sh = list(out["shown"])
    Ps, PLs, Xs = (t[t.index.get_level_values("program").isin(sh)] for t in (P, PL, X))
    out["shown_all"] = {"lite_short_of_c": int((PLs.n_returned < PLs.c - 1e-9).sum()), "pairs": len(Ps),
                        "lite_returned_range": [r(PLs.n_returned.min(), 1), r(PLs.n_returned.max(), 1)],
                        "expression_gain_above": int((Xs.gain > Ps.gain + GAIN_EPS).sum()),
                        "expression_lift_above": int((Xs.lift > Ps.lift).sum())}
    if len(sh) == 2:  # how differently the two shown queries rank the tiles
        from scipy.stats import spearmanr
        rho, jac = {}, {}
        for k in c8.KEYS:
            z = np.load(c8.OUT / k / f"seed{data.EXAMPLE_SEED}" / "gradient.npz", allow_pickle=True)
            rho[k] = r(spearmanr(z[f"d_{sh[0]}"], z[f"d_{sh[1]}"])[0], 2)
            js = []
            for s in data.SEEDS:
                h = pd.read_csv(c8.OUT / k / f"seed{s}" / "hits.csv")
                a, b = (set(h[(h.method == "P") & (h.program == p)].tile) for p in sh)
                js.append(len(a & b) / len(a | b))
            jac[k] = r(np.mean(js))
        out["shown_pair"] = {"spearman_seed0": rho, "hits_jaccard": jac}
    for prog in out["not_shown"]:
        p = P.xs(prog, level="program")
        out.setdefault("not_shown_detail", {})[prog] = {
            "gain_pos": [k for k in c8.KEYS if p.gain[k] > GAIN_EPS], "gain_range_pos": [
                r(p.gain[p.gain > GAIN_EPS].min()), r(p.gain[p.gain > GAIN_EPS].max())],
            "delta_le0": {k: r(p.delta[k], 2) for k in c8.KEYS if p.delta[k] <= 0}}
    with open(c8.OUT / "numbers.json", "w") as fh:
        json.dump(out, fh, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
