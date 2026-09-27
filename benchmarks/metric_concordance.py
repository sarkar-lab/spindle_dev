"""E14: which distance agrees better with biology -- block-diagonal or whole-matrix LE?

Motivation: accuracy is scored against the block-diagonalized log-Euclidean
distance (compute_ground_truth), not the whole-matrix one
(compute_ground_truth_whole_matrix). This tests whether that choice is also
the biologically better one, using readouts that never enter either distance.

For every held-out query, the exact top-k under each metric (both rankings are
the cached ground truths) and k random training tiles (100 draws) are compared
with the query on

* ``expr_r``   (all datasets, label-free): Pearson r between the query's
  pseudo-bulk profile and the mean profile of the k neighbours. Profile = mean
  expression of the index genes over the tile's cells, each gene z-scored across
  training tiles (so abundant genes do not dominate). Higher = closer.
* ``comp_jsd`` (breast): base-2 Jensen-Shannon divergence between cell-type
  compositions (obs['Cluster']), as in E13. Lower = closer.
* ``majority_match`` (breast): fraction of the k neighbours whose majority cell
  type equals the query's.

and, beyond the top-k, per query:
* ``rho_expr`` / ``rho_comp``: Spearman correlation, over ALL training tiles,
  between the metric's distance to the query and the tile's biological
  dissimilarity (1 - expr r, or composition JSD). Higher = the metric tracks biology.

Paired Wilcoxon signed-rank tests (block vs whole, over queries) per
(dataset, seed, k / readout).

Seeds: seed-73 production build for every dataset (both ground truths cached);
breast also seeds 0-4 (whole-matrix ground truth computed and cached on first run).

Outputs (results/metric_concordance/):
  <stem>[_seed<n>]_per_query.csv   one row per (query, k, method) + per-query rho
  summary.csv  (--aggregate)       means and Wilcoxon P per (dataset, seed, k, readout)

  sbatch slurm_jobs/run_metric_concordance.sbatch <dataset> [--seed N]
"""

import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wilcoxon

import experiment_common as ec
import holdout_core as hv  # type: ignore

OUT_DIR = ec.PROJECT_ROOT / "results" / "metric_concordance"
KS = [1, 5, 10, 50]
N_RANDOM = 100


def tile_means(X, tile_list):
    import scipy.sparse as sp

    out = np.empty((len(tile_list), X.shape[1]))
    for i, t in enumerate(tile_list):
        m = X[t.idx].mean(axis=0)
        out[i] = np.asarray(m).ravel() if sp.issparse(X) else m
    return out


def rowcorr(A, b):
    """Pearson r of every row of A with vector b."""
    A = A - A.mean(axis=1, keepdims=True)
    b = b - b.mean()
    return (A @ b) / (np.linalg.norm(A, axis=1) * np.linalg.norm(b) + 1e-12)


def jsd_rows(P, q):
    return np.array([jensenshannon(p, q, base=2) ** 2 for p in P])


def run(stem, seed, rng):
    bundle = ec.load_index(stem, seed)
    data = bundle["data"]
    covs = ec.load_raw_covs(stem, seed)
    train_idx, test_idx = np.asarray(covs["train_idx"]), np.asarray(covs["test_idx"])
    tag = ec.index_tag(stem, seed)
    kw = dict(cache_dir=ec.GT_CACHE_DIR, seed=seed)
    gt_b = hv.load_or_compute_ground_truth("block", tag, covs["test_tile_covs"], covs["train_tile_covs"],
                                           train_idx, test_idx, data=data, **kw)["per_query"]
    gt_w = hv.load_or_compute_ground_truth("whole", tag, covs["test_tile_covs"], covs["train_tile_covs"],
                                           train_idx, test_idx, **kw)["per_query"]
    del covs

    adata = ec.load_adata(stem)
    tiles = ec.rebuild_tiles(adata)
    ec.check_tiles_match(tiles, data, train_idx)
    gene_pos = {g: i for i, g in enumerate(adata.var_names)}
    X = adata.X[:, [gene_pos[g] for g in data.metadata["genes"]]]
    train_tiles = data.metadata["tiles"]
    query_tiles = [tiles[int(i)] for i in test_idx]
    E_train, E_query = tile_means(X, train_tiles), tile_means(X, query_tiles)
    mu, sd = E_train.mean(0), E_train.std(0) + 1e-9
    E_train, E_query = (E_train - mu) / sd, (E_query - mu) / sd

    has_labels = "Cluster" in adata.obs.columns
    if has_labels:
        cat = adata.obs["Cluster"].astype("category").cat.remove_unused_categories()
        codes, n_types = cat.cat.codes.to_numpy(), len(cat.cat.categories)

        def comp(tl):
            C = np.stack([np.bincount(codes[t.idx], minlength=n_types) for t in tl]).astype(float)
            return C / C.sum(1, keepdims=True)

        C_train, C_query = comp(train_tiles), comp(query_tiles)
        maj_train, maj_query = C_train.argmax(1), C_query.argmax(1)

    n = len(train_tiles)
    rows = []
    for qi in range(len(test_idx)):
        order = {"block": np.asarray(gt_b[qi]["true_order"]), "whole": np.asarray(gt_w[qi]["true_order"])}
        dist = {m: np.array([gt[qi]["dist_dict"][j] for j in range(n)]) for m, gt in (("block", gt_b), ("whole", gt_w))}
        expr_dis = 1 - rowcorr(E_train, E_query[qi])
        comp_dis = jsd_rows(C_train, C_query[qi]) if has_labels else None
        rho = {}
        for m in ("block", "whole"):
            rho[m] = {"rho_expr": spearmanr(dist[m], expr_dis).statistic}
            if has_labels:
                rho[m]["rho_comp"] = spearmanr(dist[m], comp_dis).statistic

        def readouts(sel):
            r = {"expr_r": float(rowcorr(E_train[sel].mean(0, keepdims=True), E_query[qi])[0])}
            if has_labels:
                r["comp_jsd"] = float(jensenshannon(C_train[sel].mean(0), C_query[qi], base=2) ** 2)
                r["majority_match"] = float(np.mean(maj_train[sel] == maj_query[qi]))
            return r

        for k in KS:
            overlap = len(set(order["block"][:k]) & set(order["whole"][:k])) / k
            for m in ("block", "whole"):
                rows.append({"query": qi, "k": k, "method": m, **readouts(order[m][:k]), **rho[m],
                             "topk_overlap_block_whole": overlap})
            draws = [readouts(rng.choice(n, k, replace=False)) for _ in range(N_RANDOM)]
            rows.append({"query": qi, "k": k, "method": "random",
                         **{key: float(np.mean([d[key] for d in draws])) for key in draws[0]}})
    df = pd.DataFrame(rows)
    df.insert(0, "seed", seed)
    df.insert(0, "dataset", stem)
    return df, tag


def aggregate():
    frames = [pd.read_csv(f) for f in sorted(OUT_DIR.glob("*_per_query.csv"))]
    df = pd.concat(frames, ignore_index=True)
    readouts = [c for c in ("expr_r", "comp_jsd", "majority_match") if c in df.columns]
    out = []
    for (ds, seed, k), g in df.groupby(["dataset", "seed", "k"]):
        wide = {m: g[g.method == m].set_index("query") for m in ("block", "whole", "random")}
        for col in readouts + ["rho_expr", "rho_comp"]:
            if col not in g or wide["block"][col].isna().all():
                continue
            if col.startswith("rho") and k != KS[0]:
                continue  # per-query, independent of k
            b, w = wide["block"][col], wide["whole"][col]
            rec = {"dataset": ds, "seed": seed, "k": k if not col.startswith("rho") else "all",
                   "readout": col, "block": b.mean(), "whole": w.mean(),
                   "random": wide["random"][col].mean() if col in readouts else np.nan,
                   "block_better_frac": float(np.mean((b < w) if col == "comp_jsd" else (b > w))),
                   "wilcoxon_p": wilcoxon(b, w).pvalue if np.any(b != w) else 1.0, "n_queries": len(b)}
            out.append(rec)
    s = pd.DataFrame(out)
    s.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset")
    parser.add_argument("--seed", type=int, default=ec.PRODUCTION_SEED)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    stem = ec.resolve_stem(args.dataset)
    df, tag = run(stem, args.seed, np.random.default_rng(args.seed))
    df.to_csv(OUT_DIR / f"{tag}_per_query.csv", index=False)
    print(df.groupby(["k", "method"]).mean(numeric_only=True).drop(columns=["seed", "query"]).round(4).to_string())


if __name__ == "__main__":
    main()
