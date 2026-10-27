"""Every number Fig 3's caption and text quote -> results/fig3_whole_tile/numbers.json.

Usage (login node or anywhere): python experiments/fig3_whole_tile/write_numbers.py
Inputs: the small tables of c_summary.py (timing.csv, nx.csv, memory.csv, band/typical.csv, band/breast_cancer.npz).
Spindle-lite at g = 0.25 (as built, beta = 0.10). Values per dataset are means over seeds 0-4; "range" is over the
eight datasets, "median" the median over them. Timings: mean over seeds of the per-seed median ms per query.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402

RES = data.RESULTS / "fig3_whole_tile"
OUT = RES / "numbers.json"
G = 0.25


def rng(s, nd=1):
    return [round(float(s.min()), nd), round(float(s.max()), nd)]


def main():
    keys = data.indexed_keys()
    t = pd.read_csv(RES / "timing.csv").set_index("key").loc[keys]
    n = pd.read_csv(RES / "nx.csv")
    n = n[n.g == G].set_index("key").loc[keys]
    m = pd.read_csv(RES / "memory.csv", dtype={"param": str})
    typ = pd.read_csv(RES / "band" / "typical.csv").set_index("key")
    tags = set(t.tag) | set(n.tag) | set(m.tag) | set(typ.tag)
    if tags != {data.TAG}:
        raise RuntimeError(f"inputs carry tags {tags}, expected {data.TAG}")

    t["e2e_whole_ms"] = t.qlog_whole_ms + t.whole_ms
    t["e2e_lite_ms"] = t.qlog_block_ms + t.lite_ms
    t["e2e_spindle_ms"] = t.qlog_block_ms + t.spindle_ms
    n["found_pct"] = 100 * n.n_found / n.N
    per = {}
    for k in keys:
        r, q = t.loc[k], n.loc[k]
        per[k] = {
            "ms": {c: round(float(r[f"{c}_ms"]), 2) for c in ("whole", "spindle", "lite", "qlog_whole", "qlog_block")},
            "speedup_spindle_vs_whole": round(float(r.speedup_spindle_vs_whole), 1),
            "speedup_lite_vs_spindle": round(float(r.speedup_lite_vs_spindle), 1),
            "speedup_lite_vs_whole": round(float(r.speedup_lite_vs_whole), 0),
            "e2e_ms": {c: round(float(r[f"e2e_{c}_ms"]), 1) for c in ("whole", "spindle", "lite")},
            "NX5_pct": [round(100 * q.NX5, 1), round(100 * q.NX5_sd, 1)],
            "overlap5_pct": round(100 * q.overlap5, 1),
            "short5_pct": round(100 * q.short5, 1), "short10_pct": round(100 * q.short10, 1),
            "found_median": round(float(q.n_found), 0), "found_pct_of_N": round(float(q.found_pct), 1),
            "N": int(q.N), "queries_per_seed": int(q.n_queries),
            "work_frac_pct": round(100 * q.work_frac, 1),
        }

    # the typical breast query of panel C
    z = np.load(RES / "band" / "breast_cancer.npz")
    d, found = z["d"], z["found"]
    c5 = max(1, round(0.05 * len(d)))
    tb = typ.loc["breast_cancer"]
    band = {
        "seed": int(tb.seed), "query": int(tb.query), "test_tile": int(tb.test_tile), "N": int(len(d)),
        "returned": int(len(found)), "NX5_pct": round(100 * float(tb.NX5), 1), "c5": int(c5),
        "exact_top_c5_returned": int(np.isin(np.argsort(d, kind="stable")[:c5], found).sum()),
        "exact_top10_returned": int(np.isin(np.argsort(d, kind="stable")[:10], found).sum()),
        "queries": int(tb.queries), "queries_ge10": int(tb.queries_ge10),
        "share_short10_pct": round(100 * float(tb.share_short10), 1),
    }

    mem = {}
    for k in ("breast_cancer", "brain_cancer"):
        s = m[m.key == k]
        rows = {}
        for _, r in s.iterrows():
            name = {"lite": "lite", "spindle": "spindle", "whole": "whole"}.get(r.method)
            if name is None:
                name = (f"pca{int(float(r.param))}" if r.method == "pca_flat" else f"pca256_pq{int(float(r.param))}")
            rows[name] = {"MB": round(float(r.MB), 1)}
            if not np.isnan(r.NX5):
                rows[name].update(NX5_pct=[round(100 * r.NX5, 1), round(100 * r.NX5_sd, 1)],
                                  overlap5_pct=round(100 * r.overlap5, 1), short5_pct=round(100 * r.short5, 1))
        mem[k] = rows

    out = {
        "stamp": data.stamp(), "tags": sorted(tags), "seeds": data.SEEDS, "g": G,
        "timed_queries_per_seed": 200,
        "datasets": per,
        "range": {
            "speedup_spindle_vs_whole": rng(t.speedup_spindle_vs_whole, 0),
            "speedup_lite_vs_spindle": rng(t.speedup_lite_vs_spindle),
            "speedup_lite_vs_whole": rng(t.speedup_lite_vs_whole, 0),
            "spindle_ms": rng(t.spindle_ms, 2), "lite_ms": rng(t.lite_ms, 2), "whole_ms": rng(t.whole_ms, 1),
            "qlog_block_ms": rng(t.qlog_block_ms, 1), "qlog_whole_ms": rng(t.qlog_whole_ms, 1),
            "qlog_block_over_spindle": rng(t.qlog_block_ms / t.spindle_ms, 1),
            "qlog_block_over_lite": rng(t.qlog_block_ms / t.lite_ms, 0),
            "qlog_block_lt_qlog_whole": [k for k in keys if t.loc[k, "qlog_block_ms"] < t.loc[k, "qlog_whole_ms"]],
            "e2e_whole_over_lite": rng(t.e2e_whole_ms / t.e2e_lite_ms, 1),
            "e2e_whole_over_spindle": rng(t.e2e_whole_ms / t.e2e_spindle_ms, 1),
            "NX5_pct": rng(100 * n.NX5),
            "overlap5_pct": rng(100 * n.overlap5), "short5_pct": rng(100 * n.short5),
            "short10_pct": rng(100 * n.short10), "found_pct_of_N": rng(n.found_pct),
            "work_frac_pct": rng(100 * n.work_frac),
        },
        "median": {
            "NX5_pct": round(float(100 * n.NX5.median()), 1),
            "overlap5_pct": round(float(100 * n.overlap5.median()), 1),
            "speedup_lite_vs_spindle": round(float(t.speedup_lite_vs_spindle.median()), 1),
            "speedup_spindle_vs_whole": round(float(t.speedup_spindle_vs_whole.median()), 0),
        },
        "band_breast": band,
        "memory": mem,
    }
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("range", "median", "band_breast")}, indent=1))
    print(f"wrote {OUT.relative_to(data.ROOT)}")


if __name__ == "__main__":
    main()
