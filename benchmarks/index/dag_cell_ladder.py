"""Index build cost and DAG overlap against the number of cells: one large Xenium section subsampled.

Dataset (``--dataset``, default tonsil_reactive): the 10x human tonsil section (1.35M cells, 377
genes; benchmarks/index/prepare_xenium_h5ad.py), the largest public single Xenium section we found, or
any of the 8 indexed datasets. Each unit: tile the real data, hold out N_QUERIES real tiles
(seed 73, fixed for every unit) and remove their cells, subsample the remaining cells WITHOUT
replacement (seed SEED = 0; ``full`` = all of them).

Each unit runs in three stages:

1. prepare (this process): write the training cells and the query cells to a scratch folder.
2. build (a fresh process, so its peak memory is the build's alone): read the training cells,
   then the production path (build_indexes.prepare_to_index: tiles, top min(800, n) genes,
   per-tile covariances; build_indexes.run_index: niches + blocks, cap 1000, min_final_size 15,
   random_state = SEED), then spindle_dev.tiers: the exact tier (shrunk block logs) and the DAG
   (K = 32). Readouts (Fig 2D): exact_build_s / peak_rss_exact_gb (h5ad -> exact tier) and
   build_s / peak_rss_gb (h5ad -> both tiers); stage times are diagnostic.
3. score (this process): Overlap(delta, c) of the DAG's result sets against the exact tier
   (supplementary S7).

Outputs (results/dag_cell_ladder/):
  <unit>_{per_query,summary}.csv; summary.csv + missing.txt (--aggregate)

  sbatch slurm_jobs/index/run_dag_cell_ladder.sbatch <dataset> <cells|full>
  all units: bash slurm_jobs/index/submit_dag_cell_ladder.sh
"""

import argparse
import gc
import os
import pickle
import platform
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import dag_eval_common as de
import experiment_common as ec
from build_indexes import prepare_to_index, run_index  # type: ignore
from spindle_dev import preprocessing, tiers
from spindle_dev.preprocessing import QuadTile

OUT_DIR = ec.PROJECT_ROOT / "results" / "dag_cell_ladder"
SCRATCH = Path(os.environ.get("SPINDLE_LOCAL_CACHE", f"/tmp/spindle_cache_{os.environ.get('USER', 'spindle')}")) / "cell_ladder"
DEFAULT_DATASET = "tonsil_reactive"
N_QUERIES = 100
QUERY_SEED = ec.PRODUCTION_SEED
SEED = 0  # subsample + clustering (random_state 0, as the production builds)
NICHE_CAP = 1000
COV_JOBS = 8  # as build_indexes.prepare_to_index


def unit_name(dataset, cells):
    return f"{dataset}_cells{cells}"


def held_out(tiles, n_q):
    """Indices of the N_QUERIES held-out real tiles (seed 73; the tiling is spatial, so fixed per dataset)."""
    return np.sort(np.random.default_rng(QUERY_SEED).choice(len(tiles), N_QUERIES, replace=False))[:n_q]


# ---------------------------------------------------------------- 1. prepare
def prepare(dataset, cells, n_q, work):
    adata = ec.load_adata(paths.dataset_path(dataset))
    real = ec.rebuild_tiles(adata)
    q_idx = held_out(real, n_q)
    q_tile = np.full(adata.n_obs, -1)
    for i, t in enumerate(q_idx):
        q_tile[real[t].idx] = i
    queries = adata[q_tile >= 0].copy()
    queries.obs["query_tile"] = q_tile[q_tile >= 0]
    queries.write_h5ad(work / "query.h5ad")
    pool = adata[q_tile < 0].copy()
    del adata, queries
    if cells != "full":
        pool = ec.subsample_adata(pool, int(cells), SEED)
    pool.write_h5ad(work / "train.h5ad")
    print(f"{dataset}: {pool.n_obs} training cells, {pool.n_vars} genes, {len(q_idx)} query tiles", flush=True)
    del pool
    gc.collect()


# ---------------------------------------------------------------- 2. build (own process)
def build(work):
    import scanpy as sc

    T = {}
    t_start = t0 = time.perf_counter()
    adata = sc.read_h5ad(work / "train.h5ad")
    tiles, covs, genes = prepare_to_index(adata)
    n_obs = adata.n_obs
    del adata
    T["tiles_covs_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    data, _ = run_index(tiles, covs, genes, SimpleNamespace(n_obs=n_obs), resolution=0.2, min_final_size=15,
                        max_niche_size=NICHE_CAP, random_state=SEED)
    T["niches_blocks_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    exact = tiers.build_tier("exact", data, covs)
    T["exact_s"] = time.perf_counter() - t0
    exact_build_s, peak_exact = time.perf_counter() - t_start, de.peak_rss_gb()
    t0 = time.perf_counter()
    dag = tiers.build_tier("dag", data, exact=exact)
    T["dag_s"] = time.perf_counter() - t0
    info = {"exact_build_s": exact_build_s, "build_s": time.perf_counter() - t_start,
            "peak_rss_exact_gb": peak_exact, "peak_rss_gb": de.peak_rss_gb(), **T,
            "cells": n_obs, "n_genes": len(data.metadata["genes"]), "n_tiles": len(tiles),
            "exact_mb": exact.nbytes() / de.MB, **de.dag_info(dag.compact, data), **de.niche_stats(data, covs),
            "hostname": platform.node()}
    with open(work / "built.pkl", "wb") as fh:
        pickle.dump({"exact": exact, "dag": dag, "genes": list(data.metadata["genes"]), "info": info}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)


# ---------------------------------------------------------------- 3. score
def query_covs(genes, work, n_jobs):
    import scanpy as sc

    q = sc.read_h5ad(work / "query.h5ad")
    pos = {g: i for i, g in enumerate(map(str, q.var_names))}
    gene_idx = np.array([pos[g] for g in genes])
    tile_of = q.obs["query_tile"].to_numpy()
    q_tiles = [QuadTile(i, (0.0, 0.0, 0.0, 0.0), np.flatnonzero(tile_of == i)) for i in np.unique(tile_of)]
    return preprocessing.build_tile_covs_full(q, q_tiles, gene_idx, n_jobs=n_jobs, eps=1e-6)


def run(dataset, cells, max_queries, n_jobs):
    n_q = min(N_QUERIES, max_queries or N_QUERIES)
    name = unit_name(dataset, cells)
    work = SCRATCH / name
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        prepare(dataset, cells, n_q, work)
        gc.collect()
        subprocess.run([sys.executable, __file__, "--build", str(work)], check=True)
        with open(work / "built.pkl", "rb") as fh:
            built = pickle.load(fh)
        qcovs = query_covs(built["genes"], work, n_jobs)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    rows, _ = de.score_tiers(built["exact"], built["dag"], qcovs)
    base = {"unit": name, "dataset": dataset, "target_cells": cells, "niche_cap": NICHE_CAP,
            **tiers.DAG_DEFAULTS, **built["info"]}
    per_q = pd.DataFrame(rows)
    per_q.insert(0, "unit", name)
    return per_q, de.summarize(per_q, {**base, "status": "ok"})


def aggregate():
    s = pd.concat([pd.read_csv(f) for f in sorted(f for f in OUT_DIR.glob("*_summary.csv") if f.name != "summary.csv")], ignore_index=True)
    s.to_csv(OUT_DIR / "summary.csv", index=False)
    failed = s[s.status != "ok"]
    (OUT_DIR / "missing.txt").write_text("".join(f"{r.unit} FAILED: {r.status}\n" for r in failed.itertuples()))
    pd.set_option("display.width", 250)
    cols = ["unit", "cells", "n_genes", "n_tiles", "n_niches", "niche_max", "n_blocks", "exact_mb", "dag_mb",
            "exact_build_s", "build_s", "peak_rss_exact_gb", "peak_rss_gb", "overlap5@5pct", "overlap5@c100", "status"]
    print(s[[c for c in cols if c in s]].round(3).to_string(index=False))
    print(f"failed: {len(failed)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--cells", default="full", help="target training cells, or 'full'")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--build", help=argparse.SUPPRESS)  # internal: the build stage's scratch folder
    args = parser.parse_args()
    if args.build:
        build(Path(args.build))
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    out_dir = OUT_DIR / "smoke" if args.max_queries else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    name = unit_name(args.dataset, args.cells)
    try:
        per_q, summ = run(args.dataset, args.cells, args.max_queries,
                          int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
        per_q.to_csv(out_dir / f"{name}_per_query.csv", index=False)
    except Exception as exc:
        traceback.print_exc()
        summ = pd.DataFrame([{"unit": name, "dataset": args.dataset, "target_cells": args.cells,
                              "status": f"{type(exc).__name__}: {exc}"[:300]}])
    summ.to_csv(out_dir / f"{name}_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).T.to_string())


if __name__ == "__main__":
    main()
