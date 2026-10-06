"""Index build cost and DAG overlap against the number of genes on lymph node 5K, on ALL tiles, with a
streaming builder that never holds every tile's genes x genes matrices at once.

The standard builder (build_indexes.run_index) keeps, for all ~7,600 tiles, the p x p covariance,
a full float64 ultrametric and its flattened features: several TB at p = 4,624. Here:

1. Ultrametric features, cached on disk: per tile (parallel), covariance from raw counts ->
   metrics.spd_to_ultrametric (the same code) -> float32 condensed ultrametric, written to one
   memmap under CACHE_ROOT on node-local disk (deleted at the end of the job).
2. Embedding: exact 30-component PCA via the tiles x tiles Gram matrix, accumulated over column
   blocks of the memmap (one pass), centred, eigendecomposed (no UMAP).
3. Niches: Leiden on the PCA scores (metrics.leiden_clustering_latent, resolution 0.2, raised by 0.1
   until no niche exceeds the cap, at most 15 times, then index.split_oversized_niches; as
   ProcessedData.cluster_spds).
4. Per niche: mean condensed ultrametric (streamed rows) -> average linkage -> leaf order; mean
   correlation (streamed over re-computed covariances); blocks from ProcessedData.get_adaptive_runs
   on a shell holding only labels, Z_list, perm_list and R_mean_list.
   Niches and blocks are learned from raw covariances, as the production indexes.
5. The exact tier: each tile's covariance is re-computed, shrunk (tiers.prepare_cov) and only its
   diagonal blocks are logged (tiers.BlockVectorizer, float32) -> tiers.ExactTier.from_block_vectors.
6. The DAG tier from the exact tier's block logs (tiers.DagTier, K = 32).

build_s (stages 1-6, from reading the h5ad) and the memory peaks are the Fig 2E numbers:
mem_anon_gb is the main process's allocated memory (MemSampler; joblib workers excluded),
peak_rss_gb also counts the pages of the on-disk feature cache (cache_gb on disk). The same three
up to the exact tier only (stages 1-5): exact_build_s, mem_anon_exact_gb, peak_rss_exact_gb.
Then (supplementary) Overlap(delta, c) of the DAG against the exact tier for N_QUERIES held-out
real tiles (seed 73), queries' covariances computed on the fly.

Outputs (results/dag_gene_ladder/):
  lymph_node_5k_G<g>_{per_query,summary}.csv, lymph_node_5k_G<g>_layout.pkl (niche labels,
  gene orders and blocks per niche); summary.csv + missing.txt (--aggregate)

  sbatch --mem=... slurm_jobs/run_dag_gene_ladder.sbatch <G> [--max-queries N]
  all points: bash slurm_jobs/submit_dag_gene_ladder.sh
"""

import argparse
import gc
import os
import pickle
import shutil
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from scipy.spatial.distance import squareform

import dag_eval_common as de
import experiment_common as ec
from spindle_dev import index, metrics, preprocessing, tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "dag_gene_ladder"
# node-local disk: the feature cache is I/O-bound, and the network drive is shared (timings would be noisy)
CACHE_ROOT = Path(os.environ.get("SPINDLE_LOCAL_CACHE", f"/tmp/spindle_cache_{os.environ.get('USER', 'spindle')}"))
STEM = "xenium_human_lymph_node_5k"
GENE_GRID = [250, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4624]
NICHE_CAP = 1000
N_QUERIES = 100
SEED = ec.PRODUCTION_SEED
N_PCA = 30
GRAM_COLS = 250_000
CHUNK = 64


def tile_cov(Xs):
    return preprocessing._cov_ml(Xs, eps=1e-6, cast32=True)


def leaf_order(Z):
    """dendrogram(...)['leaves'] (as consensus_tree_from_ultrametrics); leaves_list (same order) if it recurses too deep."""
    try:
        return np.asarray(dendrogram(Z, no_plot=True)["leaves"])
    except RecursionError:
        return leaves_list(Z)


# ---------------------------------------------------------------- per-tile workers (run in joblib processes)
def _features_chunk(path, shape, rows, slices):
    Y = np.memmap(path, dtype=np.float32, mode="r+", shape=shape)
    for r, Xs in zip(rows, slices):
        U, _ = metrics.spd_to_ultrametric(tile_cov(Xs))
        Y[r] = squareform(U, checks=False).astype(np.float32)
    Y.flush()
    del Y
    return len(rows)


def _corr_sum_chunk(slices):
    acc = None
    for Xs in slices:
        R = metrics.spd_to_correlation(tile_cov(Xs)).astype(np.float64)
        acc = R if acc is None else acc + R
    return acc


def _block_vecs_chunk(slices, vec):
    """The exact tier's float32 block-log vectors (tiers.ExactTier) of each tile's shrunk covariance."""
    return np.stack([vec.vector(tiers.prepare_cov(tile_cov(Xs)), np.float32) for Xs in slices])


def chunks(seq, size):
    return [seq[i:i + size] for i in range(0, len(seq), size)]


# ---------------------------------------------------------------- data
def load_ladder_data(g, max_queries):
    adata = ec.load_adata(STEM)
    tiles = ec.rebuild_tiles(adata)
    q_idx = np.sort(np.random.default_rng(SEED).choice(len(tiles), N_QUERIES, replace=False))
    if max_queries:
        q_idx = q_idx[:max_queries]
    held = set(q_idx.tolist())
    train = [t for i, t in enumerate(tiles) if i not in held]
    queries = [tiles[i] for i in q_idx]
    g_eff = min(g, adata.n_vars)
    genes, gene_idx = preprocessing.topvar_genes(adata, G=g_eff)
    Xg = adata.X[:, gene_idx].tocsr() if hasattr(adata.X, "tocsr") else adata.X[:, gene_idx]
    n_obs = adata.n_obs
    del adata
    gc.collect()
    return train, queries, list(genes), Xg, n_obs


def niches_from_scores(scores, cap):
    """ProcessedData.cluster_spds' adaptive Leiden on given latent scores."""
    res = 0.2
    for _ in range(15):
        labels, _, _ = metrics.leiden_clustering_latent(scores, k_neighbors=10, resolution=res, random_state=0)
        if cap is None or np.bincount(labels).max() <= cap:
            break
        res += 0.1
    else:
        labels = index.split_oversized_niches(scores, labels, cap, random_state=0)
    return np.asarray(labels)


# ---------------------------------------------------------------- streaming run
def build_streaming(g, cap, max_queries, n_jobs, mem):
    """Stages 1-6 -> (exact tier, DAG tier, layout, query tiles, Xg, info); ``mem`` is the running MemSampler."""
    T = {}
    t_start = t0 = time.perf_counter()
    train, queries, genes, Xg, n_obs = load_ladder_data(g, max_queries)
    p, M = len(genes), len(train)
    D = p * (p - 1) // 2
    T["load_s"] = time.perf_counter() - t0
    cache = CACHE_ROOT / f"gene_ladder_G{g}_{os.environ.get('SLURM_JOB_ID', 'local')}"
    cache.mkdir(parents=True, exist_ok=True)
    feat_path = str(cache / "ultrametric_features.f32")
    try:
        # 1. cached ultrametric features
        t0 = time.perf_counter()
        np.memmap(feat_path, dtype=np.float32, mode="w+", shape=(M, D)).flush()
        rows = list(range(M))
        Parallel(n_jobs=n_jobs)(delayed(_features_chunk)(feat_path, (M, D), r, [Xg[train[i].idx] for i in r])
                                for r in chunks(rows, CHUNK))
        T["features_s"] = time.perf_counter() - t0
        cache_gb = os.path.getsize(feat_path) / 1024 ** 3

        # 2. exact PCA via the Gram matrix (one pass over column blocks)
        t0 = time.perf_counter()
        Y = np.memmap(feat_path, dtype=np.float32, mode="r", shape=(M, D))
        G = np.zeros((M, M))
        for a in range(0, D, GRAM_COLS):
            blk = np.asarray(Y[:, a:a + GRAM_COLS], dtype=np.float64)
            G += blk @ blk.T
        H = np.eye(M) - 1.0 / M
        lam, vec = np.linalg.eigh(H @ G @ H)
        top = np.argsort(lam)[::-1][:min(N_PCA, M - 1)]
        scores = vec[:, top] * np.sqrt(np.maximum(lam[top], 0.0))
        del G, H, lam, vec
        T["pca_s"] = time.perf_counter() - t0

        # 3. niches
        t0 = time.perf_counter()
        labels = niches_from_scores(scores, cap)
        T["niches_s"] = time.perf_counter() - t0

        # 4. per-niche consensus order, mean correlation, blocks
        t0 = time.perf_counter()
        shell = index.ProcessedData.__new__(index.ProcessedData)
        shell.labels, shell.Z_list, shell.perm_list, shell.R_mean_list = labels, {}, {}, {}
        shell.block_initial_dict, shell.block_dict = {}, {}
        for k in sorted(set(labels.tolist())):
            members = np.flatnonzero(labels == k)
            acc = np.zeros(D)
            for r in chunks(members.tolist(), 32):
                acc += np.asarray(Y[r], dtype=np.float64).sum(0)
            Z = linkage(acc / len(members), method="average")
            shell.Z_list[k], shell.perm_list[k] = Z, leaf_order(Z)
            parts = Parallel(n_jobs=n_jobs)(delayed(_corr_sum_chunk)([Xg[train[i].idx] for i in r])
                                            for r in chunks(members.tolist(), CHUNK))
            shell.R_mean_list[k] = sum(parts) / len(members)
        del Y
        shell.get_adaptive_runs(find_blocks=True, with_size_guard=True, min_final_size=15, max_final_size=100)
        T["blocks_s"] = time.perf_counter() - t0
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    T["cache_removed"] = not cache.exists()

    # 5. the exact tier (shrunk block logs)
    t0 = time.perf_counter()
    data = tiers.LayoutShell(labels, shell.perm_list, shell.block_dict)
    X = {}
    for k, (idx, vec) in tiers.niche_layouts(data).items():
        parts = Parallel(n_jobs=n_jobs)(delayed(_block_vecs_chunk)([Xg[train[i].idx] for i in r], vec)
                                        for r in chunks(idx.tolist(), CHUNK))
        X[k] = np.vstack(parts)
    exact = tiers.ExactTier.from_block_vectors(data, X)
    T["exact_s"] = time.perf_counter() - t0
    mem.sample()
    T.update(exact_build_s=time.perf_counter() - t_start, mem_anon_exact_gb=mem.anon_gb,
             peak_rss_exact_gb=de.peak_rss_gb())

    # 6. the DAG tier
    t0 = time.perf_counter()
    dag = tiers.build_tier("dag", data, exact=exact)
    T["dag_s"] = time.perf_counter() - t0
    T["build_s"] = time.perf_counter() - t_start
    info = {"n_genes": p, "n_tiles": M, "cells": n_obs, "niche_max": int(np.bincount(labels).max()),
            "cache_gb": cache_gb, "exact_mb": exact.nbytes() / de.MB, **de.dag_info(dag.compact, data), **T}
    layout = {"genes": genes, "labels": labels, "perm_list": shell.perm_list, "block_dict": shell.block_dict}
    return exact, dag, layout, queries, Xg, info


def aggregate():
    s = pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob("lymph_node_5k_G*_summary.csv"))], ignore_index=True)
    s = s.sort_values("genes_requested")
    s.to_csv(OUT_DIR / "summary.csv", index=False)
    done = set(s[s.status == "ok"].genes_requested)
    missing = [f"G{g}" for g in GENE_GRID if g not in done]
    failed = s[s.status != "ok"]
    (OUT_DIR / "missing.txt").write_text("".join(f"{m}\n" for m in missing)
                                         + "".join(f"G{r.genes_requested} FAILED: {r.status}\n" for r in failed.itertuples()))
    pd.set_option("display.width", 250)
    cols = ["genes_requested", "n_genes", "n_tiles", "n_niches", "niche_max", "n_blocks", "exact_mb", "dag_mb",
            "exact_build_s", "build_s", "mem_anon_exact_gb", "mem_anon_gb", "peak_rss_gb", "cache_gb", "overlap5@5pct",
            "overlap5@c100", "status"]
    print(s[[c for c in cols if c in s]].round(3).to_string(index=False))
    print(f"missing: {len(missing)}, failed: {len(failed)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--genes", type=int)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    n_jobs = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    name = f"lymph_node_5k_G{args.genes}"
    out_dir = OUT_DIR / "smoke" if args.max_queries else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    base = {"dataset": STEM, "genes_requested": args.genes, "niche_cap": NICHE_CAP, **tiers.DAG_DEFAULTS}
    try:
        # per-tile workers get one BLAS thread each; the main process keeps all CPUs for the Gram step
        with parallel_config(backend="loky", inner_max_num_threads=1), de.MemSampler() as mem:
            exact, dag, layout, queries, Xg, info = build_streaming(args.genes, NICHE_CAP, args.max_queries, n_jobs, mem)
        info.update({"mem_anon_gb": mem.anon_gb, "peak_rss_gb": de.peak_rss_gb()})
        with open(out_dir / f"{name}_layout.pkl", "wb") as fh:
            pickle.dump(layout, fh, protocol=pickle.HIGHEST_PROTOCOL)
        t0 = time.perf_counter()
        rows, _ = de.score_tiers(exact, dag, (tile_cov(Xg[t.idx]) for t in queries))
        info["score_s"] = time.perf_counter() - t0
        per_q = pd.DataFrame(rows)
        per_q.insert(0, "genes_requested", args.genes)
        per_q.to_csv(out_dir / f"{name}_per_query.csv", index=False)
        summ = de.summarize(per_q, {**base, **info, "status": "ok"})
    except Exception as exc:
        traceback.print_exc()
        summ = pd.DataFrame([{**base, "status": f"{type(exc).__name__}: {exc}"[:300], "peak_rss_gb": de.peak_rss_gb()}])
    summ.to_csv(out_dir / f"{name}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).T.to_string())


if __name__ == "__main__":
    main()
