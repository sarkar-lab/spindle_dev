"""E7: robustness of retrieval to perturbed queries.

For one dataset's seed-73 production build, perturbs every held-out query
and runs the unchanged production search (every niche, budget_multiplier=1.0,
top_c=400, Stage-2 exact re-rank) against the UNPERTURBED index. Scoring is
against the ground truth of the CLEAN query: a hit means the noisy query still
retrieves a tile within eps of the clean query's true nearest neighbour.

Noise models (``--model``):
  spectral        metrics.add_spd_noise: Gaussian noise with s.d. = level on the
                  log-eigenvalues (eigenvectors kept); levels 0.05-0.5.
  cell_subsample  recompute the query covariance from a random fraction
                  (level) of the tile's cells, as if the tile had been measured
                  with fewer cells; levels 0.9, 0.75, 0.5.
Level 0 (clean) is always run once and must reproduce E1 at budget 1.0.

Per condition and query it also records
  * d_le_whole: whole-matrix log-Euclidean distance clean -> noisy (calibration);
  * d_block_over_eps: block distance clean -> noisy in the clean query's true
    best niche, divided by that niche's epsilon (the unit of the eps bands);
  * exact_noisy_*: the same recall metric for an exact scan with the noisy
    query, i.e. the best any index could do with that query. Spindle's drop is
    compared against this, not against 1.

Outputs (results/noise_robustness/):
  <stem>_<model>[_<levels>]_noise_sweep.csv   per (level, noise_seed, query)
  summary.csv  (--aggregate)                  per (dataset, model, level)

  sbatch slurm_jobs/run_noise_robustness.sbatch <dataset> <model> [levels...]
"""

import argparse

import numpy as np
import pandas as pd

import experiment_common as ec
import holdout_core as hv  # type: ignore
import spindle_dev.metrics as metrics
from spindle_dev.preprocessing import _cov_ml

OUT_DIR = ec.PROJECT_ROOT / "results" / "noise_robustness"
LEVELS = {"spectral": [0.05, 0.1, 0.2, 0.3, 0.5], "cell_subsample": [0.9, 0.75, 0.5]}
NOISE_SEEDS = [0, 1, 2, 3, 4]
BUDGET_MULT = 1.0
TOP_C = 400
FRACS = hv.EPSILON_TOLERANCE_FRACTIONS


class ExactScanner:
    """Block-diagonalized distance from one query to every training tile, vectorized per (niche, block)."""

    def __init__(self, gt_block, data):
        self.data = data
        self.parts = []  # (niche, block_runs, local indices, [stacked block logs])
        for niche, (indices, cached) in gt_block["niche_train_cache"].items():
            runs = data.block_dict[niche]
            stacks = [np.stack([cached[t][b] for t in indices]) for b in range(len(runs))]
            self.parts.append((niche, runs, np.asarray(indices), stacks))
        self.n = sum(len(p[2]) for p in self.parts)

    def distances(self, q_logs_by_niche):
        d = np.empty(self.n)
        for niche, runs, indices, stacks in self.parts:
            q = q_logs_by_niche[niche]
            acc = np.zeros(len(indices))
            for b, (s, e) in enumerate(runs):
                acc += np.linalg.norm((stacks[b] - q[b]).reshape(len(indices), -1), axis=1) / np.sqrt(e - s)
            d[indices] = acc
        return d


def block_logs(q, data):
    out = {}
    for niche in sorted(data.block_dict):
        perm = data.perm_list[niche]
        qp = q[np.ix_(perm, perm)]
        out[niche] = [hv.log_spd(qp[s:e, s:e]) for s, e in data.block_dict[niche]]
    return out


def whole_log(q):
    return metrics.log_spd(q)


class CellSubsampler:
    """Rebuild a query tile's covariance from a random subset of its cells."""

    def __init__(self, stem, data, test_idx, queries):
        adata = ec.load_adata(stem)
        tiles = ec.rebuild_tiles(adata)
        gene_pos = {g: i for i, g in enumerate(adata.var_names)}
        gene_idx = [gene_pos[g] for g in data.metadata["genes"]]
        X = adata.X[:, gene_idx]
        self.cells = [tiles[int(i)].idx for i in test_idx]
        self.X = X
        # The clean covariance must reproduce the stored query exactly.
        for qi in (0, len(queries) - 1):
            rebuilt = _cov_ml(X[self.cells[qi], :], eps=1e-6, cast32=True)
            if not np.allclose(rebuilt, queries[qi], atol=1e-5):
                raise ValueError(f"rebuilt covariance of query {qi} does not match the stored one")

    def __call__(self, qi, frac, rng):
        idx = self.cells[qi]
        keep = rng.choice(idx, size=max(2, int(round(frac * idx.size))), replace=False)
        return _cov_ml(self.X[np.sort(keep), :], eps=1e-6, cast32=True)


def run_condition(model, level, noise_seed, queries, perturb, ctx):
    data, gt_block = ctx["data"], ctx["gt_block"]
    rng = np.random.default_rng([noise_seed, int(round(level * 1000))])
    if level == 0:
        noisy = queries
    elif model == "spectral":
        noisy = [metrics.add_spd_noise(q, noise_level=level, seed=int(rng.integers(2**31))) for q in queries]
    else:
        noisy = [perturb(qi, level, rng) for qi in range(len(queries))]

    matched, times = hv.perform_search(noisy, data, ctx["dag_dict"], ctx["config"], budget_multiplier=BUDGET_MULT)
    noisy_logs = [block_logs(q, data) for q in noisy]
    rerank_gt = {**gt_block, "per_query": [{"query_blocks_log_by_niche": L} for L in noisy_logs]}
    df, _ = hv.evaluate_against_ground_truth(
        gt_block, rerank_gt, ctx["train_idx"], matched, times, data, ctx["name"], ctx["config"],
        ground_truth_kind="block", top_c_candidates=TOP_C,
    )
    df = df.drop(columns=["bf_time_ms", "speedup"])

    extra = []
    for qi, (q, qn) in enumerate(zip(queries, noisy)):
        gt = gt_block["per_query"][qi]
        niche = gt["true_best_niche"]
        eps = float(ctx["config"].epsilon_dict[niche])
        d_whole = np.linalg.norm(ctx["clean_whole_logs"][qi] - whole_log(qn), ord="fro") / np.sqrt(q.shape[0])
        d_block = ec.block_distance(gt["query_blocks_log_by_niche"][niche], noisy_logs[qi][niche],
                                    data.block_dict[niche])
        # Exact scan with the noisy query, scored against the clean query's ground truth.
        best = int(np.argmin(ctx["scanner"].distances(noisy_logs[qi])))
        gap = gt["dist_dict"][best] - gt["closest_dist"]
        rec = {"d_le_whole": d_whole, "d_block_over_eps": d_block / eps, "exact_noisy_best_rank": gt["true_order"].index(best) + 1}
        for frac in FRACS:
            rec[f"exact_noisy_recall_at_eps_{frac}"] = int(gap <= frac * eps)
        extra.append(rec)
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    df.insert(0, "noise_seed", noise_seed)
    df.insert(0, "level", level)
    df.insert(0, "model", model)
    return df


def aggregate():
    files = sorted(OUT_DIR.glob("*_noise_sweep.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    metric_cols = [c for c in df.columns if c.startswith(("recall_at_eps", "overlap_at_eps", "exact_noisy_recall"))]
    # Mean over queries within each noise seed, then mean/s.d. across noise seeds.
    per_seed = df.groupby(["Dataset", "model", "level", "noise_seed"])[
        metric_cols + ["d_le_whole", "d_block_over_eps", "spindle_time_ms"]].mean()
    grouped = per_seed.groupby(["Dataset", "model", "level"])
    summary = grouped.mean().add_prefix("mean_").join(grouped.std().add_prefix("sd_"))
    summary.insert(0, "n_noise_seeds", grouped.size())
    summary = summary.reset_index()
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    print(f"Wrote {OUT_DIR / 'summary.csv'} ({len(summary)} rows from {len(files)} files)")
    # First level at which mean recall@0.1eps falls below 0.9, per dataset and model.
    for (ds, model), g in summary.groupby(["Dataset", "model"]):
        g = g.sort_values("level", ascending=(model == "spectral"))
        below = g[g["mean_recall_at_eps_0.1"] < 0.9]
        print(f"{ds} {model}: recall@0.1eps < 0.9 first at level {below['level'].iloc[0] if len(below) else 'never'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset")
    parser.add_argument("--model", choices=list(LEVELS))
    parser.add_argument("--levels", type=float, nargs="*", default=None,
                        help="subset of the model's levels (0 = clean); default all plus 0")
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=NOISE_SEEDS)
    parser.add_argument("--seed", type=int, default=ec.PRODUCTION_SEED, help="index build (holdout split)")
    parser.add_argument("--max-queries", type=int, default=None, help="debug: first N queries only")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return

    stem = ec.resolve_stem(args.dataset)
    tag = ec.index_tag(stem, args.seed)
    levels = args.levels if args.levels else [0.0] + LEVELS[args.model]
    suffix = "" if args.levels is None else "_" + "_".join(f"{lv:g}" for lv in levels)
    out_csv = OUT_DIR / f"{tag}_{args.model}{suffix}_noise_sweep.csv"

    bundle = ec.load_index(stem, args.seed)
    data = bundle["data"]
    covs = ec.load_raw_covs(stem, args.seed)
    train_idx, test_idx = np.asarray(covs["train_idx"]), np.asarray(covs["test_idx"])
    test_covs = covs["test_tile_covs"]
    gt_block = hv.load_or_compute_ground_truth("block", tag, test_covs, covs["train_tile_covs"], train_idx,
                                               test_idx, data=data, cache_dir=ec.GT_CACHE_DIR, seed=args.seed)
    del covs
    if args.max_queries:
        test_covs, test_idx = test_covs[: args.max_queries], test_idx[: args.max_queries]
        gt_block = {**gt_block, "per_query": gt_block["per_query"][: args.max_queries]}
    queries = [ec.raw_cov(c) for c in test_covs]  # as stored (float32), so level 0 reproduces E1
    ctx = dict(data=data, dag_dict=bundle["dag_dict"], config=bundle["config"], name=tag, train_idx=train_idx,
               gt_block=gt_block, scanner=ExactScanner(gt_block, data),
               clean_whole_logs=[whole_log(q) for q in queries])
    perturb = CellSubsampler(stem, data, test_idx, queries) if args.model == "cell_subsample" else None

    frames = []
    for level in levels:
        for noise_seed in (args.noise_seeds if level != 0 else [0]):
            print(f"\n=== {args.model} level={level} noise_seed={noise_seed} ===", flush=True)
            df = run_condition(args.model, level, noise_seed, queries, perturb, ctx)
            print(df[["recall_at_eps_0.1", "exact_noisy_recall_at_eps_0.1", "d_le_whole",
                      "d_block_over_eps"]].mean().to_string(), flush=True)
            frames.append(df)
            pd.concat(frames, ignore_index=True).to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
