"""Spindle-DAG retrieval on held-out tiles (Fig 3B, C; supplementary S4, S5).

Per dataset and seed: the exact tier (shrunk block logs, d_B) and the DAG tier (K node means per
niche and block) are built from the saved index's layout and training covariances; every held-out
tile is a query. The DAG's returned order is scored against the exact d_B with
dag_eval_common.retrieval_rows on c = dag_eval_common.c_grid(N):
  Overlap(delta, c) for delta in 1/2.5/5/10 %, hit, recall of the exact top 10, distance ratio
  (returned c vs exact top c) and tolerant precision (within 5 % of the exact c-th distance);
plus per query the c needed to hold 90 % of the exact 5 %-near set / top 10 (c_needed).
c90 = the c where the mean Overlap(5 %, c) curve reaches 0.9 (dag_eval_common.curve_c).

--k-sweep  also builds K in {4, 8, 16, 32, 64, 128} (k_min = K, alpha = 0) and records size vs c90.
--bound    (S5) for a sample of queries, each tile's exact d_B, the DAG's approximate distance and the
           triangle-inequality bound |d_B - d_DAG| <= ||t - m(t)|| (the tile's distance to its node means).

Outputs (results/dag_holdout/):
  <stem>_seed<s>_per_query.csv   query x c readouts (K = 32)
  <stem>_seed<s>_queries.csv     per-query c90 and near-set size
  <stem>_seed<s>_ksweep.csv      per K: sizes, c90, Overlap(5 %, 5 %), distance ratio (--k-sweep)
  <stem>_seed<s>_bound.csv       (--bound)
  --aggregate: curves.csv (mean over queries, then mean +/- s.d. over seeds, per c), summary.csv,
               k_sweep.csv

  sbatch slurm_jobs/whole_tile/run_dag_holdout.sbatch <dataset> <seed> [--k-sweep] [--bound] | --aggregate
"""

import argparse
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

OUT_DIR = ec.PROJECT_ROOT / "results" / "dag_holdout"
K_SWEEP = [4, 8, 16, 32, 64, 128]
K_MAIN = de.DAG_CFG["k_target"]
C_REPORT = [10, 50, 100]
BOUND_QUERIES = 20


def build_dag(data, exact, k):
    return tiers.build_tier("dag", data, exact=exact, k_target=k, alpha=0.0, k_min=k)


def dag_order(dag, q):
    ids, approx = dag.adc.scores(dag._query(q, None))
    return ids[np.argsort(approx, kind="stable")], ids, approx


def score(exact, dag, queries, cs):
    """-> (query x c rows, per-query rows) for one DAG."""
    rows, qrows = [], []
    for qi, q in enumerate(queries):
        d = exact.distances(exact._query(q, None))
        order, _, _ = dag_order(dag, q)
        rows += [{"query": qi, **r} for r in de.retrieval_rows(d, order, cs)]
        qrows.append({"query": qi, "d_star": float(d.min()), **de.c_needed(d, order)})
    return pd.DataFrame(rows), pd.DataFrame(qrows)


def curve_summary(per_q, n):
    """Readouts of one (dataset, seed, K) from its query x c rows."""
    m = per_q.groupby("c").mean(numeric_only=True)
    rec = {"c90": de.curve_c(m.index, m["overlap_0.05"]),
           "c90_top10": de.curve_c(m.index, m[f"recall_top{de.K_TOP}"])}
    rec["c90_pct"] = 100 * rec["c90"] / n
    for c in C_REPORT:
        r = m.loc[min(c, n)]
        rec[f"dist_ratio@c{c}"] = r["dist_ratio"]
        rec[f"overlap5@c{c}"] = r["overlap_0.05"]
        rec[f"recall_top{de.K_TOP}@c{c}"] = r[f"recall_top{de.K_TOP}"]
        rec[f"tol_prec5@c{c}"] = r["tol_prec_0.05"]
    for f in (0.05, 0.10):
        c = int(min(n, max(1, round(f * n))))
        rec[f"overlap5@{100 * f:g}pct"] = m.loc[c, "overlap_0.05"]
    return rec


def bound_rows(exact, dag, queries):
    """S5: exact d_B, the DAG's approximate distance and each tile's residual ||t - m(t)|| (the bound)."""
    resid = np.empty(exact.n)
    for k, (idx, vec) in exact.layouts.items():
        c = dag.compact[k]
        r2 = np.zeros(len(idx))
        for b in range(len(vec.runs)):
            T = exact.X[k][:, vec.offsets[b]:vec.offsets[b + 1]].astype(np.float64)
            r2 += ((T - c["centroids"][b][c["codes"][:, b].astype(np.int64)]) ** 2).sum(1)
        resid[idx] = np.sqrt(r2)
    out = []
    for qi, q in enumerate(queries[:BOUND_QUERIES]):
        d = exact.distances(exact._query(q, None))
        _, ids, approx = dag_order(dag, q)
        da = np.empty(exact.n)
        da[ids] = np.sqrt(np.maximum(approx, 0.0))
        out.append(pd.DataFrame({"query": qi, "tile": np.arange(exact.n), "d_exact": d, "d_dag": da,
                                 "bound": resid, "exact_rank": np.argsort(np.argsort(d, kind="stable"))}))
    return pd.concat(out, ignore_index=True)


def run(stem, seed, k_sweep, bound, max_queries):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{stem}_seed{seed}"
    data = ec.load_index(stem, seed)["data"]
    covs = ec.load_raw_covs(stem, seed)
    queries = [ec.raw_cov(c) for c in covs["test_tile_covs"]][:max_queries]
    t0 = time.perf_counter()
    exact = tiers.build_tier("exact", data, covs["train_tile_covs"])
    del covs
    n, cs = exact.n, de.c_grid(exact.n)
    print(f"{tag}: N = {n}, {len(queries)} queries, exact tier {time.perf_counter() - t0:.0f} s", flush=True)
    base = {"dataset": stem, "seed": seed, "n_tiles": n, "n_queries": len(queries)}

    ks = K_SWEEP if k_sweep else [K_MAIN]
    sweep = []
    for k in ks:
        dag = build_dag(data, exact, k)
        per_q, qrows = score(exact, dag, queries, cs)
        info = de.dag_info(dag.compact, data)
        rec = {**base, "K": k, **info, "dag_pct_of_block_logs": 100 * info["dag_mb"] / info["block_logs_mb"],
               "dag_build_s": dag.build_seconds, **curve_summary(per_q, n),
               "c90_query_median": qrows["c90_near"].median(), "c90_top10_query_median": qrows["c90_top10"].median(),
               "near5_median": qrows["near_size_0.05"].median()}
        sweep.append(rec)
        print(f"  K = {k:3d}: DAG {info['dag_mb']:.2f} MB ({rec['dag_pct_of_block_logs']:.1f} % of block logs), "
              f"c90 = {rec['c90']:.1f} ({rec['c90_pct']:.2f} % of N), Overlap(5 %, 5 %) = {rec['overlap5@5pct']:.3f}, "
              f"distance ratio @10 = {rec['dist_ratio@c10']:.3f}", flush=True)
        if k == K_MAIN:
            per_q.insert(0, "seed", seed)
            per_q.insert(0, "dataset", stem)
            per_q.to_csv(OUT_DIR / f"{tag}_per_query.csv", index=False)
            qrows.insert(0, "seed", seed)
            qrows.insert(0, "dataset", stem)
            qrows.to_csv(OUT_DIR / f"{tag}_queries.csv", index=False)
            if bound:
                bound_rows(exact, dag, queries).assign(dataset=stem, seed=seed).to_csv(
                    OUT_DIR / f"{tag}_bound.csv", index=False)
    pd.DataFrame(sweep).to_csv(OUT_DIR / f"{tag}_{'ksweep' if k_sweep else 'k32'}.csv", index=False)


def aggregate():
    per = [pd.read_csv(f) for f in sorted(OUT_DIR.glob("*_per_query.csv"))]
    per = [p for p in per if p["seed"].iloc[0] in ec.MULTISEED]
    if not per:
        raise SystemExit("no per-query files for seeds 0-4")
    # mean over queries per (dataset, seed, c), then mean / s.d. over seeds
    m = pd.concat([p.groupby(["dataset", "seed", "c"]).mean(numeric_only=True).drop(columns="query")
                   for p in per]).reset_index()
    curves = m.groupby(["dataset", "c"]).agg(["mean", "std"])
    curves.columns = [f"{a}_{b}" for a, b in curves.columns]
    curves = curves.drop(columns=["seed_mean", "seed_std"]).reset_index()
    curves["n_seeds"] = m.groupby(["dataset", "c"])["seed"].nunique().values
    curves.to_csv(OUT_DIR / "curves.csv", index=False)

    rows = []
    for f in sorted(OUT_DIR.glob("*_seed[0-9]_k32.csv")) + sorted(OUT_DIR.glob("*_seed[0-9]_ksweep.csv")):
        rows.append(pd.read_csv(f))
    allk = pd.concat(rows).drop_duplicates(["dataset", "seed", "K"])
    num = allk.select_dtypes("number").columns.drop(["seed", "K"])
    g = allk.groupby(["dataset", "K"])
    ks = g[list(num)].agg(["mean", "std"])
    ks.columns = [f"{a}_{b}" for a, b in ks.columns]
    ks["n_seeds"] = g["seed"].nunique()
    ks = ks.reset_index()
    ks.to_csv(OUT_DIR / "k_sweep.csv", index=False)
    summ = ks[ks["K"] == K_MAIN].drop(columns="K")
    order = {ec.resolve_stem(k): i for i, k in enumerate(ec.DATASETS)}
    summ = summ.sort_values("dataset", key=lambda s: s.map(order))
    summ.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summ[["dataset", "n_seeds", "c90_mean", "c90_pct_mean", "overlap5@5pct_mean", "dist_ratio@c10_mean",
                "dag_mb_mean"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset")
    ap.add_argument("--seed", type=int, default=ec.PRODUCTION_SEED)
    ap.add_argument("--k-sweep", action="store_true")
    ap.add_argument("--bound", action="store_true")
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run(ec.resolve_stem(args.dataset), args.seed, args.k_sweep, args.bound, args.max_queries)


if __name__ == "__main__":
    main()
