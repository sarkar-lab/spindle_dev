"""Sanity check of src/spindle_dev/tiers.py on one production index (seed 73).

1. Shrinkage off (SHRINK_ALPHA = 0): ExactTier reproduces the old float64 ExactScorer; the DAG tier's
   codes equal build_dag_from_block_logs on float64 block logs; ExactPartialTier with S = all genes
   reproduces ExactTier.
2. Shrinkage on (the configured SHRINK_ALPHA): ExactTier equals ExactScorer on shrunk covariances; the
   partial tiers give the same answer for a p x p and an |S| x |S| (cut from the shrunk p x p) query;
   DAG Overlap(5%, 5%) and the interval index's Overlap(5%, 5%) on gene programs are printed.
   Partial-query baselines (Fig 4): ExactTier.from_exact_partial equals ExactTier; padding and conditional
   imputation with S = all genes reproduce the Exact d_B; the imputed query is PSD for a 16-gene set.
3. WholeTier (the whole-matrix baseline) equals metrics.log_euclidean_distance_for_SPD on shrunk covariances
   (eigenvalue floor 1e-6) on 20 query-tile pairs; its top 10 is compared with ExactTier's.

  sbatch slurm_jobs/checks/run_check_tiers.sbatch <dataset>
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import dag_eval_common as de
import experiment_common as ec
from spindle_dev import metrics, partial_search, tiers


def max_rel(a, b):
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-12)))


def overlap5(d, rows, n, c):
    near = np.flatnonzero(d <= 1.05 * d.min())
    return float(np.isin(near, rows[:c]).mean())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", default="skin_melanoma")
    ap.add_argument("--n-queries", type=int, default=30)
    args = ap.parse_args()
    stem = ec.resolve_stem(args.dataset)
    data = ec.load_index(stem, ec.PRODUCTION_SEED)["data"]
    covs = ec.load_raw_covs(stem, ec.PRODUCTION_SEED)
    train, tests = covs["train_tile_covs"], [ec.raw_cov(c) for c in covs["test_tile_covs"]][:args.n_queries]
    n = len(train)
    c5 = int(round(0.05 * n))
    alpha = metrics.SHRINK_ALPHA
    print(f"{stem}: {n} tiles, {len(tests)} queries, SHRINK_ALPHA = {alpha}", flush=True)

    # 1. shrinkage off
    metrics.SHRINK_ALPHA = 0.0
    old = tiers.ExactScorer(data, train)
    ex = tiers.build_tier("exact", data, train)
    err = max(max_rel(ex.search(q)[1][np.argsort(ex.search(q)[0])], old.distances(old.query_vectors(np.asarray(q, float))))
              for q in tests[:5])
    print(f"[off] exact tier vs float64 ExactScorer: max rel diff {err:.2e}", flush=True)
    dag = tiers.build_tier("dag", data, exact=ex)
    ref = tiers.build_dag_from_block_logs(
        {k: [old.X[k][:, vec.offsets[b]:vec.offsets[b + 1]] * np.sqrt(vec.p) for b in range(len(vec.runs))]
         for k, (_, vec) in tiers.niche_layouts(data).items()}, data, **de.DAG_CFG)
    same = all(np.array_equal(dag.compact[k]["codes"], ref[k]["codes"]) for k in ref)
    print(f"[off] DAG codes equal to the float64 build: {same}", flush=True)
    ep = tiers.build_tier("exact_partial", data, train)
    allg = np.arange(ep.p)
    err = max(max_rel(ep.distances(np.asarray(q, float), allg), old.distances(old.query_vectors(np.asarray(q, float))))
              for q in tests[:3])
    print(f"[off] exact_partial with S = all genes vs ExactScorer: max rel diff {err:.2e}", flush=True)
    del ex, dag, ep, old

    # 2. shrinkage on
    metrics.SHRINK_ALPHA = alpha
    old = tiers.ExactScorer(data, [metrics.shrink_cov(np.asarray(ec.raw_cov(c), float)) for c in train])
    ex = tiers.build_tier("exact", data, train)
    dag = tiers.build_tier("dag", data, exact=ex)
    ov, err = [], 0.0
    for q in tests:
        rows, d, _ = ex.search(q)
        dfull = np.empty(n)
        dfull[rows] = d
        err = max(err, max_rel(dfull, old.distances(old.query_vectors(metrics.shrink_cov(np.asarray(q, float))))))
        ov.append(overlap5(dfull, dag.search(q, c=c5)[0], n, c5))
    print(f"[on]  exact tier vs ExactScorer on shrunk covariances: max rel diff {err:.2e}", flush=True)
    print(f"[on]  DAG Overlap(5%, 5%) = {np.mean(ov):.3f}; exact {ex.nbytes() / 2**20:.1f} MB, "
          f"DAG {dag.nbytes() / 2**20:.2f} MB; last query {ex.last_timing} / {dag.last_timing}", flush=True)

    ep = tiers.build_tier("exact_partial", data, train, keep_niche_means=True)
    ex2 = tiers.ExactTier.from_exact_partial(ep, data)
    means = ep.niche_means()
    err_fp = err_pad = err_imp = 0.0
    min_eig = np.inf
    for q in tests[:5]:
        cov = tiers.prepare_cov(q)
        d_ref = ex.distances(ex.query_vectors(cov))
        err_fp = max(err_fp, max_rel(ex2.distances(ex2.query_vectors(cov)), d_ref))
        allg = np.arange(ep.p)
        err_pad = max(err_pad, max_rel(ex.distances(ex.query_vectors(
            partial_search.pad_query_scaled_identity(cov, allg, ep.p))), d_ref))
        k_hat, _ = partial_search.predict_niche(cov, allg, means, ep.blocks)
        err_imp = max(err_imp, max_rel(ex.distances(ex.query_vectors(
            partial_search.impute_query_conditional(cov, allg, means[k_hat]))), d_ref))
        g16 = np.sort(np.random.default_rng(1).choice(ep.p, 16, replace=False))
        k_hat, _ = partial_search.predict_niche(cov[np.ix_(g16, g16)], g16, means, ep.blocks)
        imp = partial_search.impute_query_conditional(cov[np.ix_(g16, g16)], g16, means[k_hat])
        min_eig = min(min_eig, float(np.linalg.eigvalsh(imp).min()))
    print(f"[on]  ExactTier.from_exact_partial vs ExactTier: max rel diff {err_fp:.2e}", flush=True)
    print(f"[on]  padding / imputation with S = all genes vs exact d_B: max rel diff {err_pad:.2e} / {err_imp:.2e}; "
          f"smallest eigenvalue of an imputed 16-gene query {min_eig:.3e}", flush=True)
    del ex2, means
    iv = tiers.build_tier("interval", data, exact_partial=ep)
    k0 = sorted(data.block_dict)[0]
    s, e = max(data.block_dict[k0], key=lambda r: r[1] - r[0])
    ov_iv, err_s = [], 0.0
    for q in tests:
        genes = np.sort(np.asarray(data.perm_list[k0])[s:e][:16])
        r_full, d_full, _ = ep.search(q, genes)
        cut = tiers.prepare_cov(q)[np.ix_(genes, genes)]
        dd = np.empty(n)
        dd[r_full] = d_full
        err_s = max(err_s, max_rel(ep.distances(cut, genes), dd))
        ov_iv.append(overlap5(dd, iv.search(q, genes, c=c5)[0], n, c5))
    print(f"[on]  exact_partial: p x p query vs pre-cut |S| x |S|: max rel diff {err_s:.2e}", flush=True)
    print(f"[on]  interval Overlap(5%, 5%) on a 16-gene program = {np.mean(ov_iv):.3f}; "
          f"exact_partial {ep.nbytes() / 2**20:.1f} MB, interval {iv.nbytes() / 2**20:.1f} MB", flush=True)
    del ep, iv

    # 3. whole-matrix baseline
    wh = tiers.build_tier("whole", None, train)
    rng = np.random.default_rng(0)
    err, top = 0.0, []
    for qi in range(min(5, len(tests))):
        q = tests[qi]
        rows, d, _ = wh.search(q)
        dfull = np.empty(n)
        dfull[rows] = d
        for t in rng.choice(n, 4, replace=False):
            ref = metrics.log_euclidean_distance_for_SPD(tiers.prepare_cov(q), tiers.prepare_cov(train[t]),
                                                         eps=tiers.LOG_FLOOR)
            err = max(err, abs(dfull[t] - ref) / ref)
    for q in tests:
        top.append(len(np.intersect1d(wh.search(q, c=10)[0], ex.search(q, c=10)[0])) / 10)
    print(f"[on]  whole tier vs log_euclidean_distance_for_SPD: max rel diff {err:.2e}; top-10 overlap with "
          f"exact {np.mean(top):.2f}; whole {wh.nbytes() / 2**20:.1f} MB; last query {wh.last_timing}", flush=True)
    try:
        ex.search(tests[0], genes)
    except ValueError as exc:
        print(f"[on]  routing guard ok: {exc}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
