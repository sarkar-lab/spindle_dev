"""E13: does retrieval preserve cell-type composition? (breast)

For each held-out breast query tile, compares its cell-type composition
(fractions of ``obs['Cluster']`` among its cells) with the mean composition of
the k tiles returned by

* ``spindle``  -- the production search (every niche, budget_multiplier=1.0,
  top_c=400 per niche, Stage-2 exact re-rank), as in holdout_core.py;
* ``exact``    -- the exact top-k under the block-diagonalized ground truth;
* ``random``   -- k training tiles drawn uniformly (100 draws, JSD averaged);
* ``random_same_niche`` -- k tiles drawn from the niche of the exact top-1
  (100 draws), i.e. what niche membership alone would give.

JSD is the Jensen-Shannon divergence with base-2 logs (0 = identical, 1 =
disjoint), i.e. ``scipy.spatial.distance.jensenshannon(p, q, base=2) ** 2``.
Wilcoxon signed-rank tests (two-sided, paired over queries) compare spindle
with each other method, per seed and k.

Seeds: the seed-73 production build (100 queries) and seeds 0-4 (10% holdout).
Block ground truth is loaded from results/ground_truth_cache/ or computed and
cached there (seeds 0-4 have no cache yet).

Outputs (results/composition_concordance/):
  breast_jsd.csv     one row per (seed, query, k, method)
  summary.csv        per (seed, k, method): mean/median JSD and Wilcoxon vs spindle
"""

import argparse

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wilcoxon

import experiment_common as ec
import holdout_core as hv  # type: ignore

STEM = ec.DATASETS["breast_cancer"]
OUT_DIR = ec.PROJECT_ROOT / "results" / "composition_concordance"
KS = [1, 5, 10]
N_RANDOM = 100
TOP_C = 400
BUDGET_MULT = 1.0


def jsd(p, q):
    return float(jensenshannon(p, q, base=2) ** 2)


def composition(tile_list, codes, n_types):
    comp = np.zeros((len(tile_list), n_types))
    for i, t in enumerate(tile_list):
        comp[i] = np.bincount(codes[t.idx], minlength=n_types)
    return comp / comp.sum(axis=1, keepdims=True)


def spindle_ranked(data, dag_dict, config, test_covs, gt_block, train_idx):
    """Per query: Spindle's Stage-2 re-ranked local training indices."""
    queries = hv.extract_query_matrices(test_covs)
    matched, _ = hv.perform_search(queries, data, dag_dict, config, budget_multiplier=BUDGET_MULT)
    g2l = {int(g): l for l, g in enumerate(train_idx)}
    tile_niche = gt_block["tile_niche"]
    ranked = []
    for i, ids in enumerate(matched):
        by_niche = {}
        for g in ids:
            loc = g2l.get(int(g))
            if loc is not None:
                by_niche.setdefault(tile_niche[loc], []).append(loc)
        pool = [loc for cands in by_niche.values() for loc in cands[:TOP_C]]
        q_logs = gt_block["per_query"][i]["query_blocks_log_by_niche"]
        ranked.append([loc for _, loc in ec.rerank_block(pool, q_logs, gt_block, data)])
    return ranked


def run_seed(seed, codes, n_types, tiles, rng):
    bundle = ec.load_index(STEM, seed)
    data, dag_dict, config = bundle["data"], bundle["dag_dict"], bundle["config"]
    covs = ec.load_raw_covs(STEM, seed)
    train_idx, test_idx = np.asarray(covs["train_idx"]), np.asarray(covs["test_idx"])
    ec.check_tiles_match(tiles, data, train_idx)

    gt_block = hv.load_or_compute_ground_truth(
        "block", ec.index_tag(STEM, seed), covs["test_tile_covs"], covs["train_tile_covs"],
        train_idx, test_idx, data=data, cache_dir=ec.GT_CACHE_DIR, seed=seed,
    )
    ranked = spindle_ranked(data, dag_dict, config, covs["test_tile_covs"], gt_block, train_idx)

    train_comp = composition(data.metadata["tiles"], codes, n_types)
    query_comp = composition([tiles[int(i)] for i in test_idx], codes, n_types)
    labels = np.asarray(data.labels).astype(int)
    niche_members = {c: np.flatnonzero(labels == c) for c in np.unique(labels)}
    n_train = len(train_comp)

    rows = []
    for qi in range(len(test_idx)):
        exact = gt_block["per_query"][qi]["true_order"]
        top_niche = labels[exact[0]]
        for k in KS:
            picks = {"spindle": ranked[qi][:k], "exact": exact[:k]}
            for method, sel in picks.items():
                rows.append({"seed": seed, "query": qi, "k": k, "method": method,
                             "jsd": jsd(query_comp[qi], train_comp[sel].mean(axis=0)),
                             "n_returned": len(sel)})
            for method, pool in (("random", np.arange(n_train)), ("random_same_niche", niche_members[top_niche])):
                kk = min(k, len(pool))
                draws = [jsd(query_comp[qi], train_comp[rng.choice(pool, kk, replace=False)].mean(axis=0))
                         for _ in range(N_RANDOM)]
                rows.append({"seed": seed, "query": qi, "k": k, "method": method,
                             "jsd": float(np.mean(draws)), "n_returned": kk})
    return pd.DataFrame(rows)


def summarize(df):
    out = []
    for (seed, k), g in df.groupby(["seed", "k"]):
        wide = g.pivot(index="query", columns="method", values="jsd")
        for method in wide.columns:
            rec = {"seed": seed, "k": k, "method": method, "n_queries": int(wide[method].notna().sum()),
                   "mean_jsd": wide[method].mean(), "median_jsd": wide[method].median()}
            if method != "spindle":
                diff = wide["spindle"] - wide[method]
                rec["mean_diff_spindle_minus_method"] = diff.mean()
                rec["wilcoxon_p_vs_spindle"] = (wilcoxon(wide["spindle"], wide[method]).pvalue
                                                 if np.any(diff != 0) else 1.0)
            out.append(rec)
    return pd.DataFrame(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=[ec.PRODUCTION_SEED] + ec.MULTISEED)
    parser.add_argument("--rng-seed", type=int, default=0, help="Seed for the random baselines.")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.rng_seed)

    adata = ec.load_adata(STEM)
    cat = adata.obs["Cluster"].astype("category").cat.remove_unused_categories()
    codes = cat.cat.codes.to_numpy()
    n_types = len(cat.cat.categories)
    tiles = ec.rebuild_tiles(adata)
    del adata

    frames = []
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        frames.append(run_seed(seed, codes, n_types, tiles, rng))
        # Write after every seed so a late failure keeps the finished seeds.
        df = pd.concat(frames, ignore_index=True)
        df.to_csv(OUT_DIR / "breast_jsd.csv", index=False)
        summary = summarize(df)
        summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
