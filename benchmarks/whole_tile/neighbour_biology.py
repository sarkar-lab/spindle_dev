"""Metric-free biology of the tiles each method returns (Fig 3E).

For every held-out query and every method saved by whole_cov_baselines.py (exact block top k, exact whole
top k, the Spindle-DAG set, HNSW / PCA / PQ on whole covariances), plus random tiles and random tiles from
the niche of the exact top 1, the k returned tiles are compared with the query tile:
  expr_r    Pearson r between the query's pseudo-bulk profile and the mean profile of the k tiles (index genes,
            z-scored per gene over the training tiles); all datasets
  comp_jsd  Jensen-Shannon divergence (base 2) between the query's cell-type composition and the mean
            composition of the k tiles (as metric_check.py); all 8 datasets (breast obs['Cluster']; 10x graphclust from the Xenium
            analysis output for the others -- metric_check.cell_type_codes)
k in {10, 50}. Random baselines average 100 draws.

Outputs (results/neighbour_biology/): <stem>_seed<s>.csv (query x method x k);
  --aggregate: summary.csv (per dataset, method, k: mean over queries then mean +/- s.d. over seeds 0-4) and
  tests.csv (paired Wilcoxon over pooled queries: exact vs whole, dag vs exact)

  sbatch slurm_jobs/whole_tile/run_neighbour_biology.sbatch <dataset> <seed> | --aggregate
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import dag_eval_common as de
import experiment_common as ec
from metric_check import cell_type_codes, composition, jsd, rowcorr  # type: ignore

OUT_DIR = ec.PROJECT_ROOT / "results" / "neighbour_biology"
IN_DIR = ec.PROJECT_ROOT / "results" / "whole_cov_baselines"
KS = [10, 50]
N_RANDOM = 100
TESTS = [("exact", "whole"), ("dag", "exact")]


def run(stem, seed):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    returned = dict(np.load(IN_DIR / f"{stem}_seed{seed}_returned.npz"))
    data = ec.load_index(stem, seed)["data"]
    covs = ec.load_raw_covs(stem, seed)
    train_idx, test_idx = np.asarray(covs["train_idx"]), np.asarray(covs["test_idx"])
    del covs
    labels = np.asarray(data.labels).astype(int)
    adata = ec.load_adata(stem)
    tiles = ec.rebuild_tiles(adata)
    ec.check_tiles_match(tiles, data, train_idx)
    gene_pos = {g: i for i, g in enumerate(adata.var_names)}
    X = adata.X[:, [gene_pos[g] for g in data.metadata["genes"]]]
    train_tiles, query_tiles = data.metadata["tiles"], [tiles[int(i)] for i in test_idx]
    nq = returned["exact"].shape[0]
    query_tiles = query_tiles[:nq]
    E_train, E_query = de.zscore_pair(de.pseudobulk(X, train_tiles), de.pseudobulk(X, query_tiles))
    codes = cell_type_codes(stem, adata)
    C_train = composition(train_tiles, codes) if codes is not None else None
    C_query = composition(query_tiles, codes) if codes is not None else None
    del adata, X

    def readouts(qi, sel):
        r = {"expr_r": float(rowcorr(E_train[sel].mean(0, keepdims=True), E_query[qi])[0])}
        if C_train is not None:
            r["comp_jsd"] = jsd(C_train[sel].mean(0), C_query[qi])
        return r

    rng = np.random.default_rng(seed)
    n = len(labels)
    rows = []
    for qi in range(nq):
        for k in KS:
            for method, top in returned.items():
                sel = top[qi][top[qi] >= 0][:k]
                rows.append({"query": qi, "k": k, "method": method, **readouts(qi, sel)})
            pools = (("random", np.arange(n)),
                     ("random_same_niche", np.flatnonzero(labels == labels[returned["exact"][qi, 0]])))
            for method, pool in pools:
                draws = [readouts(qi, rng.choice(pool, min(k, len(pool)), replace=False)) for _ in range(N_RANDOM)]
                rows.append({"query": qi, "k": k, "method": method,
                             **{key: float(np.mean([d[key] for d in draws])) for key in draws[0]}})
    df = pd.DataFrame(rows)
    df.insert(0, "seed", seed)
    df.insert(0, "dataset", stem)
    df.to_csv(OUT_DIR / f"{stem}_seed{seed}.csv", index=False)
    print(df.groupby(["k", "method"]).mean(numeric_only=True).drop(columns=["seed", "query"]).round(3).to_string(),
          flush=True)


def aggregate():
    df = pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob("*_seed[0-9].csv"))])
    df = df[df["seed"].isin(ec.MULTISEED)]
    readouts = [c for c in ("expr_r", "comp_jsd") if c in df.columns]
    per_seed = df.groupby(["dataset", "seed", "method", "k"])[readouts].mean().reset_index()
    g = per_seed.groupby(["dataset", "method", "k"])
    summ = g[readouts].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ["n_seeds"] = g["seed"].nunique()
    summ.reset_index().to_csv(OUT_DIR / "summary.csv", index=False)
    tests = []
    for (ds, k), sub in df.groupby(["dataset", "k"]):
        wide = {r: sub.pivot_table(index=["seed", "query"], columns="method", values=r) for r in readouts}
        for a, b in TESTS:
            for r, w in wide.items():
                if a not in w or b not in w:
                    continue
                pair = w[[a, b]].dropna()
                if len(pair) < 10:
                    continue
                diff = pair[a] - pair[b]
                tests.append({"dataset": ds, "k": k, "readout": r, "a": a, "b": b, "n": len(pair),
                              "mean_a": pair[a].mean(), "mean_b": pair[b].mean(), "mean_diff": diff.mean(),
                              "a_better_frac": float(((diff > 0) if r == "expr_r" else (diff < 0)).mean()),
                              "wilcoxon_p": wilcoxon(pair[a], pair[b]).pvalue if diff.abs().sum() > 0 else 1.0})
    pd.DataFrame(tests).to_csv(OUT_DIR / "tests.csv", index=False)
    print(summ.reset_index().query("k == 10").round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run(ec.resolve_stem(args.dataset), args.seed)


if __name__ == "__main__":
    main()
