"""Query time of the whole-matrix exact baseline, Spindle-Exact and Spindle-DAG (Fig 3A).

Per dataset and seed (0-4), three tiers are built on the saved index's training tiles, all on shrunk
covariances with stored logs (spindle_dev.tiers):
  whole  float32 whole-matrix log upper triangles, exact scan of d_W (the whole-matrix exact search)
  exact  float32 block-log upper triangles, exact scan of d_B (Spindle-Exact)
  dag    K = 32 node means + codes, approximate d_B (Spindle-DAG)
Each held-out query (at most --max-queries) is searched with ``tier.search(q, c=100)``; the time is split
by ``tier.last_timing`` into the query's own logs (query_s) and the scan (scan_s). One warm-up query per
tier is not recorded. Speedups are only like for like (exact vs whole, stored logs on both sides); the DAG
is reported in absolute ms. Run one at a time (singleton job name spindle_timing) with one BLAS thread.

Also recorded per query: the overlap of the whole and exact top 10 (they rank by different metrics).

Outputs (results/exact_vs_whole/): <stem>_seed<s>.csv (query x tier);
  --aggregate: summary.csv (per dataset: mean ms per tier and part over seeds, +/- s.d. of the seed means,
  exact/whole ratios of the total and of the scan, storage MB)

  bash slurm_jobs/whole_tile/submit_exact_vs_whole.sh
"""

import argparse
import platform
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import dag_eval_common as de
import experiment_common as ec
from spindle_dev import tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "exact_vs_whole"
TIERS = ["whole", "exact", "dag"]
C = 100


def run(stem, seed, max_queries):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = ec.load_index(stem, seed)["data"]
    covs = ec.load_raw_covs(stem, seed)
    train = covs["train_tile_covs"]
    queries = [ec.raw_cov(c) for c in covs["test_tile_covs"]][:max_queries]
    built = {}
    t0 = time.perf_counter()
    built["whole"] = tiers.build_tier("whole", None, train)
    built["exact"] = tiers.build_tier("exact", data, train)
    built["dag"] = tiers.build_tier("dag", data, exact=built["exact"])
    del covs, train
    print(f"{stem} seed {seed}: N = {built['exact'].n}, {len(queries)} queries, tiers built in "
          f"{time.perf_counter() - t0:.0f} s", flush=True)

    rows, tops = [], {}
    for name in TIERS:
        tier = built[name]
        tier.search(queries[0], c=C)  # warm-up
        tops[name] = []
        for qi, q in enumerate(queries):
            r, _, sec = tier.search(q, c=C)
            tops[name].append(r[:10])
            rows.append({"dataset": stem, "seed": seed, "tier": name, "query": qi, **tier.last_timing,
                         "total_s": sec, "nbytes_mb": tier.nbytes() / de.MB})
    df = pd.DataFrame(rows)
    ov = [len(np.intersect1d(a, b)) / 10 for a, b in zip(tops["whole"], tops["exact"])]
    df["top10_whole_vs_exact"] = df["query"].map(dict(enumerate(ov)))
    df["n_tiles"] = built["exact"].n
    df["hostname"] = platform.node()
    df.to_csv(OUT_DIR / f"{stem}_seed{seed}.csv", index=False)
    m = df.groupby("tier")[["query_s", "scan_s", "total_s"]].mean() * 1e3
    print(m.round(2).to_string(), flush=True)
    print(f"exact / whole: total {m.loc['whole', 'total_s'] / m.loc['exact', 'total_s']:.1f}x, "
          f"scan {m.loc['whole', 'scan_s'] / m.loc['exact', 'scan_s']:.1f}x; top-10 overlap {np.mean(ov):.2f}",
          flush=True)


def aggregate():
    df = pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob("*_seed[0-9].csv"))])
    df = df[df["seed"].isin(ec.MULTISEED)]
    per_seed = df.groupby(["dataset", "seed", "tier"])[["query_s", "scan_s", "total_s", "nbytes_mb"]].mean() * \
        np.array([1e3, 1e3, 1e3, 1.0])
    per_seed = per_seed.rename(columns={"query_s": "query_ms", "scan_s": "scan_ms", "total_s": "total_ms"})
    wide = per_seed.unstack("tier")
    wide.columns = [f"{t}_{c}" for c, t in wide.columns]
    for part in ("total", "scan"):
        wide[f"whole_over_exact_{part}"] = wide[f"whole_{part}_ms"] / wide[f"exact_{part}_ms"]
    wide["top10_whole_vs_exact"] = df.groupby(["dataset", "seed"])["top10_whole_vs_exact"].mean()
    wide["n_queries"] = df[df["tier"] == "exact"].groupby(["dataset", "seed"]).size()
    wide["hosts"] = df.groupby(["dataset", "seed"])["hostname"].first()
    wide = wide.reset_index()
    num = wide.select_dtypes("number").columns.drop("seed")
    g = wide.groupby("dataset")
    summ = g[list(num)].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["n_seeds"] = g["seed"].nunique()
    summ["hosts"] = g["hosts"].agg(lambda s: ",".join(sorted(set(s))))
    order = {ec.resolve_stem(k): i for i, k in enumerate(ec.DATASETS)}
    summ = summ.reset_index().sort_values("dataset", key=lambda s: s.map(order))
    wide.to_csv(OUT_DIR / "per_seed.csv", index=False)
    summ.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summ[["dataset", "n_seeds", "whole_total_ms_mean", "exact_total_ms_mean", "dag_total_ms_mean",
                "whole_over_exact_total_mean", "whole_over_exact_scan_mean", "top10_whole_vs_exact_mean"]]
          .round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=200)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run(ec.resolve_stem(args.dataset), args.seed, args.max_queries)


if __name__ == "__main__":
    main()
