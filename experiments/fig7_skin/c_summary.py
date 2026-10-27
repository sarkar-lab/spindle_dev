"""Fig 7 summaries from results/fig7_skin/<key>/seed<s>/ (c_queries.py; light, login node).

Usage: python experiments/fig7_skin/c_summary.py
Writes results/fig7_skin/:
  lists.csv      per dataset, list and query (T = Spindle-partial, TL = Spindle-lite-partial, D, X): mean and s.d. over
                 seeds of delta, cov_top, cov_all, lift, jacc_T, jacc_D, jacc_X, n_returned
  tiles.csv      per dataset and seed: median TV to Spindle's top 10, Spindle-lite's first 10, a random 10 and the
                 expression top 10; share of queries below random (Spindle, Spindle-lite); median Spindle-lite overlap
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402

OUT = data.RESULTS / "fig7_skin"
KEYS = data.indexed_keys()
SEEDS = data.SEEDS


def seed_dir(key, s):
    return OUT / key / f"seed{s}"


def lists():
    g = pd.concat([pd.read_csv(seed_dir(k, s) / "gene_lists.csv") for k in KEYS for s in SEEDS])
    num = ["delta", "cov_top", "cov_all", "lift", "jacc_T", "jacc_D", "jacc_X", "n_genes", "n_returned"]
    m = g.groupby(["key", "list", "query"], sort=False)[num].mean()
    sd = g.groupby(["key", "list", "query"], sort=False)[num].std()
    out = m.join(sd, rsuffix="_sd").reset_index()
    out["missing"] = g.groupby(["key", "list", "query"], sort=False)["missing"].first().fillna("").to_numpy()
    out["tag"] = g.tag.iloc[0]
    return out


def tiles():
    rows = []
    for key in KEYS:
        for s in SEEDS:
            t = pd.read_csv(seed_dir(key, s) / "tile_queries.csv")
            rows.append({"key": key, "seed": s, "n_queries": len(t), "n_labels": int(t.n_labels.iloc[0]),
                         "tv_top": t.tv_top.median(), "tv_lite": t.tv_lite.median(),
                         "tv_random": t.tv_random.median(), "tv_expr": t.tv_expr.median(),
                         "below_random": (t.tv_top < t.tv_random).mean(),
                         "lite_below_random": (t.tv_lite < t.tv_random).mean(),
                         "lite_in_top": t.lite_in_top.median(), "tag": t.tag.iloc[0]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for name, df in (("lists", lists()), ("tiles", tiles())):
        df.to_csv(OUT / f"{name}.csv", index=False)
        print(df.round(3).to_string(index=False), flush=True)
