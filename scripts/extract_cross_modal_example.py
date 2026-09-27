"""Ranked tile ids for the Fig. 5F example query.

results/cross_modal_search/seed_0/v2x_query_metrics.csv keeps each query's
metrics but not which Xenium tiles were returned. This reruns the seed-0 v2x
production search (the same calls as benchmarks/cross_modal_search.py:
global correction, all-niche search, budget 1.0, top_c 400) and writes, for
every query, Spindle's Stage-2 top-10 and the exact block-LE top-10:

  results/figure_data/cross_modal_v2x_seed0_topk.csv
      query_tile_id, method (spindle/exact), rank, index_tile_id, distance

Stage-2 re-ranks candidates by the exact block LE distance, so Spindle's
ranked list is its (per-niche top_c capped) candidate pool sorted by the ground
truth's dist_dict. A check against seed_0/v2x_query_metrics.csv
(spindle_best_dist) guards that the rebuilt index is the production one.

  sbatch slurm_jobs/run_extract_figure_data.sbatch   (runs both extractors)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import cross_modal_search as cms  # noqa: E402
import holdout_core as hv  # noqa: E402
import build_indexes  # noqa: E402
from spindle_dev.preprocessing import build_tile_covs_full  # noqa: E402

OUT = PROJECT_ROOT / "results" / "figure_data" / "cross_modal_v2x_seed0_topk.csv"
SEED, N_QUERIES, TOP_C, K = 0, 50, 400, 10


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    adata_vi = sc.read_h5ad(cms.DEFAULT_DATA_DIR / "visium_rotated.h5ad")
    adata_xe = sc.read_h5ad(cms.DEFAULT_DATA_DIR / "xenium_rotated.h5ad")
    adata_vi.var_names_make_unique()
    adata_xe.var_names_make_unique()
    genes = sorted(set(adata_vi.var_names).intersection(adata_xe.var_names))
    adata_xe, adata_vi = adata_xe[:, genes].copy(), adata_vi[:, genes].copy()
    tiles_xe, tiles_vi = cms.build_tiles(adata_xe, adata_vi)
    covs_xe = build_tile_covs_full(adata_xe, tiles_xe, gene_idx=None, n_jobs=8, eps=1e-6)
    covs_vi = build_tile_covs_full(adata_vi, tiles_vi, gene_idx=None, n_jobs=8, eps=1e-6)

    # v2x: Visium queries against a Xenium index (run_direction's own selection).
    rng = np.random.default_rng([SEED, 1])
    query_sel = np.sort(rng.choice(len(covs_vi), size=min(N_QUERIES, len(covs_vi)), replace=False))
    query_covs = [covs_vi[j]["cov"] for j in query_sel]
    query_ids = [covs_vi[j]["tile_id"] for j in query_sel]

    data, _ = build_indexes.run_index(tiles_xe, covs_xe, genes, adata_xe, resolution=0.2,
                                       min_final_size=15, max_niche_size=1000, random_state=SEED)
    dag_dict, config = build_indexes.configure_and_build_dag(data)
    train_covs = [t["cov"] for t in covs_xe]
    corrected = cms.blocks_log_per_niche(cms.compute_global_corrected_covs(query_covs, train_covs), data)
    gt = hv.compute_ground_truth(query_covs, train_covs, data, query_blocks_log_override=corrected)
    matched, _ = cms.perform_search_corrected(corrected, data, dag_dict, config, budget_multiplier=1.0)

    tile_niche = gt["tile_niche"]
    ref = pd.read_csv(PROJECT_ROOT / "results" / "cross_modal_search" / "seed_0" / "v2x_query_metrics.csv")
    rows, mismatches = [], 0
    for i, (qid, cands, g) in enumerate(zip(query_ids, matched, gt["per_query"])):
        pool, per_niche = [], {}
        for c in cands:
            n = tile_niche[c]
            per_niche[n] = per_niche.get(n, 0) + 1
            if per_niche[n] <= TOP_C:
                pool.append(c)
        ranked = sorted(pool, key=lambda c: g["dist_dict"][c])[:K]
        for method, order in (("spindle", ranked), ("exact", g["true_order"][:K])):
            for r, t in enumerate(order, 1):
                rows.append({"query_tile_id": qid, "method": method, "rank": r,
                             "index_tile_id": int(t), "distance": g["dist_dict"][t]})
        want = ref.loc[ref["query_tile_id"] == qid, "spindle_best_dist"].iloc[0]
        if ranked and not np.isclose(round(g["dist_dict"][ranked[0]], 4), want, atol=1e-3):
            mismatches += 1
    if mismatches:
        raise SystemExit(f"{mismatches} queries disagree with seed_0/v2x_query_metrics.csv; not writing")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(query_ids)} queries); top-1 distances match the seed-0 run")


if __name__ == "__main__":
    main()
