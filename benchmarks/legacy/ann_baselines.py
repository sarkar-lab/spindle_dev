"""E4: generic approximate-nearest-neighbour baselines vs Spindle.

For one dataset and index build (default: the seed-73 production build with
100 held-out queries), runs the SAME held-out queries through

* ``spindle``   -- the production search (every niche, budget_multiplier=1.0),
                   rerun here so both sides are timed on the same node;
* ``flat``      -- FAISS IndexFlatL2 over the vectorized whole-matrix
                   log-covariance (exact L2 control);
* ``hnsw``      -- FAISS IndexHNSWFlat on the same vectors;
* ``pca_hnsw``  -- HNSW over a PCA projection (d=256) of those vectors;
* ``phi_knn``   -- exact kNN over Spindle's own 30-d PCA embedding of the
                   ultrametric features (``ProcessedData.reduce_dim``), the
                   shortcut a reviewer would try first.

Vectorization: v = upper triangle of log(Sigma) with off-diagonal entries
scaled by sqrt(2), divided by sqrt(p), so ||v_a - v_b||_2 equals the
whole-matrix log-Euclidean distance of metrics.log_euclidean_distance_for_SPD
(the "whole" ground truth). Training-tile logs are taken from the cached
whole-matrix ground truth (checked against a fresh log_spd on a sample);
the time to compute them is measured on that sample and extrapolated.

Every method returns top_c=400 candidates, which go through the identical
Stage-2 exact re-rank and scoring as Spindle
(``holdout_core.evaluate_against_ground_truth``: block-diagonalized
ground truth, recall/overlap@eps, speedup vs whole-matrix brute force). Query
time includes computing the query's own representation (log/projection/
embedding), since Spindle's search also computes its query logs internally.
Searches run single-threaded; index builds use all allocated CPUs.

Outputs (results/ann_baselines/):
  <stem>_query_metrics.csv     per (method, query)
  <stem>_summary.csv           per method: accuracy, time, build time, memory
  <stem>_hnsw_sensitivity.csv  (--sensitivity) M x efSearch sweep
  summary.csv / hnsw_sensitivity.csv   (--aggregate) all datasets combined

  sbatch slurm_jobs/legacy/run_ann_baselines.sbatch <dataset> [--sensitivity]
"""

import argparse
import os
import pickle
import time
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import experiment_common as ec
import holdout_core as hv  # type: ignore
import spindle_dev.metrics as metrics
from spindle_dev.metrics import build_ultrametrics
from spindle_dev.index import spd_tree_feature_matrix

OUT_DIR = ec.PROJECT_ROOT / "results" / "ann_baselines"
TOP_C = 400
BUDGET_MULT = 1.0
# FAISS HNSW searches with max(efSearch, k), so efSearch below TOP_C has no effect.
HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH = 32, 200, 400
PCA_DIM = 256
SENS_M = [16, 32, 64]
SENS_EF = [400, 800, 1600]
METHODS = ["spindle", "flat", "hnsw", "pca_hnsw", "phi_knn"]
MB = 1024 ** 2


class Vectorizer:
    """Whole-matrix log-covariance -> vector whose L2 is the log-Euclidean distance."""

    def __init__(self, p):
        self.iu = np.triu_indices(p)
        self.w = (np.where(self.iu[0] == self.iu[1], 1.0, np.sqrt(2.0)) / np.sqrt(p)).astype(np.float32)

    def from_log(self, L):
        return (L[self.iu] * self.w).astype(np.float32)

    def from_cov(self, C):
        return self.from_log(metrics.log_spd(C))


def index_mb(index):
    return faiss.serialize_index(index).nbytes / MB


def timed_search(index, query_fn, queries, k):
    """Per-query wall time (s) including query_fn, plus candidate rows."""
    faiss.omp_set_num_threads(1)
    times, rows = [], []
    for q in queries:
        t0 = time.perf_counter()
        x = query_fn(q)
        _, I = index.search(x[None, :], k)
        times.append(time.perf_counter() - t0)
        rows.append(I[0][I[0] >= 0])
    return rows, times


def evaluate(method, rows_local, times, ctx):
    train_idx = ctx["train_idx"]
    matched = [[int(train_idx[j]) for j in row] for row in rows_local]
    df, summ = hv.evaluate_against_ground_truth(
        ctx["gt_block"], ctx["gt_block"], train_idx, matched, times, ctx["data"], ctx["name"],
        ctx["config"], ground_truth_kind="block", top_c_candidates=TOP_C,
        ground_truth_whole=ctx["gt_whole"],
    )
    df.insert(0, "method", method)
    summ = {"method": method, **summ, "mean_candidates": float(np.mean([len(r) for r in rows_local]))}
    return df, summ


def build_hnsw(X, M, n_threads):
    faiss.omp_set_num_threads(n_threads)
    idx = faiss.IndexHNSWFlat(X.shape[1], M)
    idx.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    t0 = time.perf_counter()
    idx.add(X)
    return idx, time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", help="short key, stem or .h5ad path")
    parser.add_argument("--seed", type=int, default=ec.PRODUCTION_SEED)
    parser.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    parser.add_argument("--sensitivity", action="store_true", help="also run the HNSW M x efSearch sweep")
    parser.add_argument("--max-queries", type=int, default=None, help="debug: first N queries only")
    parser.add_argument("--aggregate", action="store_true", help="combine per-dataset CSVs and exit")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.aggregate:
        for pattern, out in (("*_summary.csv", "summary.csv"), ("*_hnsw_sensitivity.csv", "hnsw_sensitivity.csv")):
            files = sorted(f for f in OUT_DIR.glob(pattern) if f.name != out)
            if files:
                pd.concat([pd.read_csv(f) for f in files]).to_csv(OUT_DIR / out, index=False)
                print(f"Wrote {OUT_DIR / out} from {len(files)} files")
        return

    stem = ec.resolve_stem(args.dataset)
    tag = ec.index_tag(stem, args.seed)
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    rng = np.random.default_rng(args.seed)

    bundle = ec.load_index(stem, args.seed)
    data, dag_dict, config = bundle["data"], bundle["dag_dict"], bundle["config"]
    covs = ec.load_raw_covs(stem, args.seed)
    train_idx, test_idx = np.asarray(covs["train_idx"]), np.asarray(covs["test_idx"])
    test_covs, train_covs = covs["test_tile_covs"], covs["train_tile_covs"]
    del covs
    # Ground truth always on the FULL split (the cache is keyed by it); truncate afterwards.
    gt_kw = dict(cache_dir=ec.GT_CACHE_DIR, seed=args.seed)
    gt_block = hv.load_or_compute_ground_truth("block", tag, test_covs, train_covs, train_idx, test_idx,
                                               data=data, **gt_kw)
    gt_whole = hv.load_or_compute_ground_truth("whole", tag, test_covs, train_covs, train_idx, test_idx, **gt_kw)
    if args.max_queries:
        test_covs = test_covs[: args.max_queries]
        gt_block = {**gt_block, "per_query": gt_block["per_query"][: args.max_queries]}
        gt_whole = {**gt_whole, "per_query": gt_whole["per_query"][: args.max_queries]}
    queries = [ec.raw_cov(c) for c in test_covs]
    ctx = dict(data=data, config=config, name=tag, train_idx=train_idx, gt_block=gt_block, gt_whole=gt_whole)

    p = queries[0].shape[0]
    vec = Vectorizer(p)
    n_train = len(train_covs)

    # Training vectors from the cached whole-matrix logs; time + verify on a sample.
    sample = rng.choice(n_train, size=min(50, n_train), replace=False)
    t0 = time.perf_counter()
    fresh = [metrics.log_spd(ec.raw_cov(train_covs[j])) for j in sample]
    log_time_per_tile = (time.perf_counter() - t0) / len(sample)
    max_err = max(np.abs(f - gt_whole["train_logs"][j]).max() for f, j in zip(fresh, sample))
    assert max_err < 1e-4, f"cached training logs differ from log_spd (max abs err {max_err})"
    X = np.stack([vec.from_log(L) for L in gt_whole["train_logs"]])
    gt_whole.pop("train_logs")
    del train_covs, fresh
    vector_prep_s = log_time_per_tile * n_train
    print(f"{n_train} training vectors of dim {X.shape[1]} ({X.nbytes / MB:.1f} MB); "
          f"log prep {vector_prep_s:.1f} s (extrapolated)", flush=True)

    frames, summaries = [], []

    def record(method, rows, times, build_s, mem_mb, **extra):
        df, summ = evaluate(method, rows, times, ctx)
        summ.update(build_time_s=build_s, index_mb=mem_mb, **extra)
        frames.append(df)
        summaries.append(summ)
        print(f"[{method}] {summ}", flush=True)

    if "spindle" in args.methods:
        matched, times = hv.perform_search(queries, data, dag_dict, config, budget_multiplier=BUDGET_MULT)
        g2l = {int(g): l for l, g in enumerate(train_idx)}
        rows = [[g2l[int(g)] for g in m if int(g) in g2l] for m in matched]
        record("spindle", rows, times, float(bundle.get("build_time_s", np.nan)),
               float(bundle.get("index_size_mb", np.nan)))

    if "flat" in args.methods:
        flat = faiss.IndexFlatL2(X.shape[1])
        t0 = time.perf_counter()
        flat.add(X)
        build_s = vector_prep_s + time.perf_counter() - t0
        rows, times = timed_search(flat, vec.from_cov, queries, TOP_C)
        # Exact-L2 control: its unreranked ranking must equal the whole-matrix ground truth.
        top1 = np.mean([r[0] == gt_whole["per_query"][i]["true_order"][0] for i, r in enumerate(rows)])
        top10 = np.mean([len(set(r[:10]) & set(gt_whole["per_query"][i]["true_order"][:10])) / 10
                         for i, r in enumerate(rows)])
        record("flat", rows, times, build_s, index_mb(flat),
               top1_agrees_whole_gt=top1, top10_overlap_whole_gt=top10)
        del flat

    if "hnsw" in args.methods or args.sensitivity:
        Ms = sorted(set(([HNSW_M] if "hnsw" in args.methods else []) + (SENS_M if args.sensitivity else [])))
        sens_rows = []
        for M in Ms:
            hnsw, add_s = build_hnsw(X, M, n_threads)
            mem = index_mb(hnsw)
            efs = sorted(set(([HNSW_EF_SEARCH] if M == HNSW_M and "hnsw" in args.methods else [])
                             + (SENS_EF if args.sensitivity else [])))
            for ef in efs:
                hnsw.hnsw.efSearch = ef
                rows, times = timed_search(hnsw, vec.from_cov, queries, TOP_C)
                if M == HNSW_M and ef == HNSW_EF_SEARCH and "hnsw" in args.methods:
                    record("hnsw", rows, times, vector_prep_s + add_s, mem, hnsw_M=M, efSearch=ef)
                if args.sensitivity and ef in SENS_EF:
                    _, summ = evaluate(f"hnsw_M{M}_ef{ef}", rows, times, ctx)
                    sens_rows.append({**summ, "hnsw_M": M, "efSearch": ef, "build_time_s": vector_prep_s + add_s,
                                      "index_mb": mem})
            del hnsw
        if sens_rows:
            pd.DataFrame(sens_rows).to_csv(OUT_DIR / f"{tag}_hnsw_sensitivity.csv", index=False)

    if "pca_hnsw" in args.methods:
        d = min(PCA_DIM, n_train - 1)
        t0 = time.perf_counter()
        mean = X.mean(axis=0)
        # Top-d right singular vectors of the centred training matrix.
        _, _, Vt = randomized_svd(X - mean, n_components=d, random_state=args.seed)
        W = np.ascontiguousarray(Vt[:d].T, dtype=np.float32)
        Z = np.ascontiguousarray((X - mean) @ W, dtype=np.float32)
        pca_s = time.perf_counter() - t0
        hnsw, add_s = build_hnsw(Z, HNSW_M, n_threads)
        hnsw.hnsw.efSearch = HNSW_EF_SEARCH
        rows, times = timed_search(hnsw, lambda q: (vec.from_cov(q) - mean) @ W, queries, TOP_C)
        record("pca_hnsw", rows, times, vector_prep_s + pca_s + add_s,
               index_mb(hnsw) + (W.nbytes + mean.nbytes) / MB, pca_dim=d)
        del hnsw, Z, W

    if "phi_knn" in args.methods:
        Zphi = np.ascontiguousarray(data.latent["pca"], dtype=np.float32)
        assert Zphi.shape[0] == n_train
        flat = faiss.IndexFlatL2(Zphi.shape[1])
        flat.add(Zphi)

        def phi(q):
            U, _ = build_ultrametrics([q])
            return data.pca_model.transform(spd_tree_feature_matrix(U)).astype(np.float32)[0]

        rows, times = timed_search(flat, phi, queries, TOP_C)
        # Embedding is part of Spindle's own build; only the kNN structure is extra.
        record("phi_knn", rows, times, 0.0,
               index_mb(flat) + len(pickle.dumps(data.pca_model)) / MB, phi_dim=Zphi.shape[1])

    pd.concat(frames).to_csv(OUT_DIR / f"{tag}_query_metrics.csv", index=False)
    summary = pd.DataFrame(summaries)
    summary.insert(1, "dataset", stem)
    summary.insert(2, "seed", args.seed)
    summary["n_train"] = n_train
    summary["vector_dim"] = X.shape[1]
    summary["vector_prep_s"] = vector_prep_s
    summary["n_threads_build"] = n_threads
    summary.to_csv(OUT_DIR / f"{tag}_summary.csv", index=False)
    cols = ["method", "recall_at_eps_0.1", "recall_at_eps_0.5", "overlap_at_eps_0.5", "overlap_at_eps_1.0",
            "mean_spindle_time_ms", "mean_speedup", "build_time_s", "index_mb"]
    print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
