"""Every number Fig 4's caption and text quote -> results/fig4_gene_set/numbers.json.

Usage (login node or anywhere): python experiments/fig4_gene_set/write_numbers.py
Inputs: the small tables of c_summary.py (nx.csv, nx_size.csv, timing.csv, storage.csv, band/typical.csv,
band/breast_cancer.npz). Spindle at g = 1 (as built, beta = 0.10). Values per dataset are means over
seeds 0-4 (all sizes pooled unless a size is named); "range" is over the eight datasets, "median" the median over
them; per size, the median over datasets of each seed, then mean and s.d. over seeds (nx_size.csv). Timings: mean over
seeds of the per-seed median ms per query. Spindle-naive computes its query logs inside its scan; c_timing.py times
them again on their own (qlog_naive_partial), so "naive_partial_search" is its scan without them, like for like with
Spindle's search (user, 26 Oct). Speed-ups: "search" = naive_partial_search / spindle_partial (the bars of panel D,
its labels); "e2e" = each side with its own query logs (naive_partial / (qlog_spindle_partial + spindle_partial)).
Keys: spindle_partial = Spindle (SpindlePartial), naive_partial = Spindle-naive (SpindleNaivePartial).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402

RES = data.RESULTS / "fig4_gene_set"
OUT = RES / "numbers.json"
METHODS = {"spindle_partial": "spindle_partial", "padding": "padding", "imputation": "imputation"}


def rng(s, nd=1):
    return [round(float(s.min()), nd), round(float(s.max()), nd)]


def pct(x, nd=1):
    return round(100 * float(x), nd)


def main():
    keys = data.indexed_keys()
    n = pd.read_csv(RES / "nx.csv")
    sz = pd.read_csv(RES / "nx_size.csv")
    t = pd.read_csv(RES / "timing.csv").set_index("key").loc[keys]
    st = pd.read_csv(RES / "storage.csv").set_index("key").loc[keys]
    typ = pd.read_csv(RES / "band" / "typical.csv").set_index("key")
    tags = set(n.tag) | set(sz.tag) | set(t.tag) | set(st.tag) | set(typ.tag)
    if tags != {data.TAG}:
        raise RuntimeError(f"inputs carry tags {tags}, expected {data.TAG}")

    pooled = {m: n[(n.method == m) & (n["size"] == 0)].set_index("key").loc[keys] for m in METHODS}
    spindle_partial = pooled["spindle_partial"]

    per = {}
    for k in keys:
        r, s, q = t.loc[k], st.loc[k], spindle_partial.loc[k]
        per[k] = {
            "NX5_pct": {m: [pct(pooled[m].loc[k, "NX5"]), pct(pooled[m].loc[k, "NX5_sd"])] for m in METHODS},
            "overlap5_pct": {m: pct(pooled[m].loc[k, "overlap5"]) for m in METHODS},
            "spindle_partial_short10_pct": pct(q.short10), "spindle_partial_short5_pct": pct(q.short5),
            "spindle_partial_found_median": round(float(q.n_found), 0),
            "spindle_partial_found_pct_of_N": pct(q.n_found / q.N),
            "imputation_home_niche_pct": pct(pooled["imputation"].loc[k, "k_hat_home"]),
            "N": int(q.N), "queries_per_seed": round(float(q.n_queries), 0),
            "ms": {c: round(float(r[f"{c}_ms"]), 3)
                   for c in ("naive_partial", "naive_partial_search", "qlog_naive_partial", "spindle_partial",
                             "qlog_spindle_partial", "padding", "imputation")},
            "speedup_search": round(float(r.speedup_spindle_partial), 0),
            "speedup_e2e": round(float(r.speedup_e2e), 1),
            "MB": {c: round(float(s[c]), 2)
                   for c in ("whole", "naive_partial", "spindle_partial", "padding", "imputation")},
            "spindle_partial_pct_naive_partial": round(float(s.spindle_partial_pct_naive_partial), 2),
        }

    size = {}
    for m in METHODS:
        x = sz[sz.method == m].set_index("size")
        size[m] = {int(L): {"NX5_pct": [pct(x.loc[L, "NX5"]), pct(x.loc[L, "NX5_sd"])],
                            "overlap5_pct": [pct(x.loc[L, "overlap5"]), pct(x.loc[L, "overlap5_sd"])]}
                   for L in (4, 8, 16, 32)}
        if m == "spindle_partial":
            for L in (4, 8, 16, 32):
                size[m][int(L)]["short10_pct"] = pct(x.loc[L, "short10"])

    # the typical breast query of panel A
    z = np.load(RES / "band" / "breast_cancer.npz")
    d, found = z["d"], z["found"]
    c5 = max(1, round(0.05 * len(d)))
    tb = typ.loc["breast_cancer"]
    band = {
        "seed": data.EXAMPLE_SEED, "query": int(tb.query), "size": 16, "N": int(len(d)), "returned": int(len(found)),
        "NX5_pct": pct(tb.NX5), "c5": int(c5),
        "exact_top_c5_returned": int(np.isin(np.argsort(d, kind="stable")[:c5], found).sum()),
        "exact_top10_returned": int(np.isin(np.argsort(d, kind="stable")[:10], found).sum()),
        "queries": int(tb.n_queries), "share_short10_pct": pct(tb.short10),
    }

    nq = sum(per[k]["queries_per_seed"] for k in keys)
    out = {
        "stamp": data.stamp(), "tags": sorted(tags), "seeds": data.SEEDS, "g": 1, "sizes": [4, 8, 16, 32],
        "timed_tiles_per_seed": 200, "queries_per_seed_all_datasets": round(nq, 0),
        "datasets": per,
        "size": size,
        "range": {
            **{f"NX5_pct_{m}": rng(100 * pooled[m].NX5) for m in METHODS},
            **{f"overlap5_pct_{m}": rng(100 * pooled[m].overlap5) for m in METHODS},
            "spindle_partial_short10_pct": rng(100 * spindle_partial.short10),
            "spindle_partial_found_pct_of_N": rng(100 * spindle_partial.n_found / spindle_partial.N),
            "imputation_home_niche_pct": rng(100 * pooled["imputation"].k_hat_home, 0),
            "naive_partial_ms": rng(t.naive_partial_ms, 1),
            "naive_partial_search_ms": rng(t.naive_partial_search_ms, 1),
            "qlog_naive_partial_ms": rng(t.qlog_naive_partial_ms, 2),
            "spindle_partial_ms": rng(t.spindle_partial_ms, 2),
            "qlog_spindle_partial_ms": rng(t.qlog_spindle_partial_ms, 1),
            "qlog_spindle_partial_over_spindle_partial": rng(t.qlog_spindle_partial_ms / t.spindle_partial_ms, 1),
            "padding_ms": rng(t.padding_ms, 1), "imputation_ms": rng(t.imputation_ms, 1),
            "speedup_search": rng(t.speedup_spindle_partial, 0), "speedup_e2e": rng(t.speedup_e2e, 1),
            "spindle_partial_MB": rng(st.spindle_partial, 1), "naive_partial_MB": rng(st.naive_partial, 0),
            "whole_MB": rng(st.whole, 0),
            "spindle_partial_pct_naive_partial": rng(st.spindle_partial_pct_naive_partial, 1),
            "naive_partial_over_spindle_partial": rng(st.naive_partial / st.spindle_partial, 1),
            "padding_over_spindle_partial": rng(st.padding / st.spindle_partial, 1),
            "imputation_over_spindle_partial": rng(st.imputation / st.spindle_partial, 1),
            "whole_over_spindle_partial": rng(st.whole / st.spindle_partial, 0),
        },
        "median": {
            **{f"NX5_pct_{m}": pct(pooled[m].NX5.median()) for m in METHODS},
            **{f"overlap5_pct_{m}": pct(pooled[m].overlap5.median()) for m in METHODS},
            "speedup_search": round(float(t.speedup_spindle_partial.median()), 0),
            "speedup_e2e": round(float(t.speedup_e2e.median()), 1),
        },
        "band_breast": band,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("range", "median", "size", "band_breast")}, indent=1))
    print(f"wrote {OUT.relative_to(data.ROOT)}")


if __name__ == "__main__":
    main()
