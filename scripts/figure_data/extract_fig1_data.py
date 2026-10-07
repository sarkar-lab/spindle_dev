"""Extract the breast inputs of the Fig. 1 sub-panels.

scripts/figures/fig1_subpanels.py reads CSVs only; everything it needs from the breast
production build (seed 73, results/indexes/) is written
here, once, to results/figure_data/fig1/:

  tiles.csv              every tile of the production tiling: bbox, n_cells,
                         status (train / heldout), niche and UMAP of phi (train),
                         query_idx (heldout)
  cells.csv              all cell coordinates (after the 'Unlabeled' filter)
  genes.csv              the 313 indexed genes in their original (top-variance) order
  niche_perm.csv         consensus permutation sigma_j per niche (position -> gene)
  niche_blocks.csv       block intervals [start, end) in permuted order per niche
  niche<j>_mean_corr.csv mean tile correlation of niche j, original gene order
  example_tiles.csv      per niche, the tile nearest the niche mean in phi-space
  tile_cov_<id>.csv      covariance of each example tile, original gene order
  dag_nodes.csv          every DAG node: niche, layer, order_id, n_tiles, radius
  dag_edges.csv          consecutive-layer edges with the number of shared tiles
  query_summary.csv      per held-out query: Stage-1 pool size, best distances
  query_hits.csv         Spindle Stage-2 top-10 and exact block-LE top-10
  query_stage1.csv       every tile in each query's Stage-1 pool, with its Stage-2 rank
                         and exact block-LE distance
  query_paths.csv        per query and niche, the lowest-distance DAG path, and the
                         lowest-distance path whose leaf contains Spindle's top hit

The search repeats holdout_core.perform_search (all niches, budget 1.0)
but keeps the DAG paths, which perform_search discards; Stage 1/2 follow
evaluate_against_ground_truth (top_c 400 per niche, exact block-LE re-rank).
Spindle's top-1 distances are checked against the Spindle rows of
results/ann_baselines/xenium_human_breast_cancer_query_metrics.csv.

  sbatch slurm_jobs/figures/run_extract_fig1_data.sbatch
"""

import argparse
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import paths  # noqa: E402,F401  (puts src/ and every benchmarks/ folder on sys.path)
import experiment_common as ec  # noqa: E402
import holdout_core as hv  # noqa: E402
import spindle_dev.search as search  # noqa: E402
from spindle_dev.preprocessing import build_quadtree_tiles  # noqa: E402

STEM, SEED, TOP_C, K, BUDGET_MULT = "xenium_human_breast_cancer", 73, 400, 10, 1.0
OUT_DIR = PROJECT_ROOT / "results" / "figure_data" / "fig1"
REF_METRICS = PROJECT_ROOT / "results" / "ann_baselines" / f"{STEM}_query_metrics.csv"


def search_with_paths(q_spd, data, dag_dict, config):
    """holdout_core.perform_search for one query, also returning each niche's paths."""
    matched, seen, paths_by_niche = [], set(), {}
    for c in sorted(set(int(v) for v in data.labels)):
        handle = dag_dict[c]
        n_niche = int(np.sum(data.labels == c))
        f = hv._niche_scale_factor(n_niche)
        cfg = search.SearchConfig(max_results=None, debug=False, max_failed_starts=max(1, round(100 * f)),
                                  max_failed_paths=max(1, round(200 * f)),
                                  total_paths_limit=max(1, round(3000 * f)))
        budget = float(config.epsilon_dict[c]) * len(handle.sorted_blocks) * BUDGET_MULT * f
        perm = data.perm_list[c]
        res = search.search_index(handle, q_spd[np.ix_(perm, perm)], [], data.block_dict[c], budget, config=cfg)
        paths = getattr(res, "paths", []) or []
        paths_by_niche[c] = []
        for p in paths:
            sets = [{int(s) for s, _ in handle.nodes[n].metadata.members} for n in p.node_path]
            leaf = set.intersection(*sets) if sets else set()
            paths_by_niche[c].append((p.total_distance, list(p.node_path), leaf))
            for s in sorted(leaf):
                if s not in seen:
                    seen.add(s)
                    matched.append(s)
    return matched, paths_by_niche


def dag_tables(data, dag_dict):
    node_rows, edge_rows, missing = [], [], 0
    for c, handle in dag_dict.items():
        by_layer = defaultdict(dict)  # layer -> tile -> node (the sankey helper's member rule)
        for n in handle.nodes:
            node_rows.append({"niche": int(c), "node": n.global_node_id, "layer": n.block_index,
                              "order_id": n.order_id, "n_tiles": len(n.metadata.members),
                              "radius": float(getattr(n.metadata, "radius", np.nan))})
            for s, blk in n.metadata.members:
                if blk == n.block_index:
                    by_layer[n.block_index][int(s)] = n.global_node_id
        children = {n.global_node_id: set(n.children) for n in handle.nodes}
        layers = sorted(by_layer)
        for l0, l1 in zip(layers[:-1], layers[1:]):
            counts = defaultdict(int)
            for s in by_layer[l0].keys() & by_layer[l1].keys():
                counts[(by_layer[l0][s], by_layer[l1][s])] += 1
            for (u, v), k in counts.items():
                missing += v not in children[u]
                edge_rows.append({"niche": int(c), "layer": l0, "src": u, "dst": v, "n_tiles": k})
    if missing:
        raise ValueError(f"{missing} shared-tile edges are not DAG children edges")
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    bundle = ec.load_index(STEM, SEED)
    data, dag_dict, config = bundle["data"], bundle["dag_dict"], bundle["config"]
    raw = ec.load_raw_covs(STEM, SEED)
    train_idx, test_idx = np.asarray(raw["train_idx"]), np.asarray(raw["test_idx"])
    if not np.array_equal(np.asarray(data.spd_ids), train_idx):
        raise ValueError("data.spd_ids is not train_idx")
    genes = np.asarray(data.metadata["genes"]).astype(str)
    labels = np.asarray(data.labels).astype(int)
    niches = sorted(set(labels.tolist()))

    # ---- tiles + cells (same tiling call as experiment_common.rebuild_tiles)
    a = ad.read_h5ad(ec.paths.dataset_path(STEM), backed="r")
    coords = np.asarray(a.obsm["spatial"])[:, :2]
    if "Cluster" in a.obs.columns:
        coords = coords[(a.obs["Cluster"] != "Unlabeled").to_numpy()]
    tiles = build_quadtree_tiles(coords, max_pts=200, min_side=0.0, max_depth=40)
    ec.check_tiles_match(tiles, data, train_idx)
    local_of = {int(g): k for k, g in enumerate(train_idx)}
    query_of = {int(g): q for q, g in enumerate(test_idx)}
    umap = data.latent["umap"]
    rows = []
    for t in tiles:
        k, q = local_of.get(t.id), query_of.get(t.id)
        rows.append({"tile_id": t.id, "x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2], "y1": t.bbox[3],
                     "n_cells": t.idx.size, "status": "train" if k is not None else "heldout",
                     "niche": labels[k] if k is not None else -1,
                     "umap1": umap[k, 0] if k is not None else np.nan,
                     "umap2": umap[k, 1] if k is not None else np.nan,
                     "query_idx": q if q is not None else -1})
    pd.DataFrame(rows).to_csv(out / "tiles.csv", index=False)
    pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]}).to_csv(out / "cells.csv", index=False, float_format="%.2f")
    pd.DataFrame({"gene_idx": np.arange(genes.size), "gene": genes}).to_csv(out / "genes.csv", index=False)
    print(f"tiles: {len(tiles)} ({len(train_idx)} train, {len(test_idx)} held out); cells: {len(coords)}", flush=True)

    # ---- niche structure: permutation, blocks, mean correlation, example tiles
    perm_rows, block_rows, ex_rows = [], [], []
    pca = data.latent["pca"]
    for c in niches:
        perm = np.asarray(data.perm_list[c])
        perm_rows += [{"niche": c, "position": p, "gene_idx": int(g), "gene": genes[g]} for p, g in enumerate(perm)]
        block_rows += [{"niche": c, "block": b, "start": s, "end": e} for b, (s, e) in enumerate(data.block_dict[c])]
        pd.DataFrame(data.R_mean_list[c], columns=genes).to_csv(out / f"niche{c}_mean_corr.csv", index=False,
                                                              float_format="%.4f")
        members = np.where(labels == c)[0]
        k = members[np.argmin(np.linalg.norm(pca[members] - pca[members].mean(0), axis=1))]
        tid = int(train_idx[k])
        ex_rows.append({"niche": c, "tile_id": tid, "n_tiles_in_niche": members.size})
        cov = np.asarray(ec.raw_cov(raw["train_tile_covs"][k]))
        pd.DataFrame(cov, columns=genes).to_csv(out / f"tile_cov_{tid}.csv", index=False, float_format="%.6g")
    pd.DataFrame(perm_rows).to_csv(out / "niche_perm.csv", index=False)
    pd.DataFrame(block_rows).to_csv(out / "niche_blocks.csv", index=False)
    pd.DataFrame(ex_rows).to_csv(out / "example_tiles.csv", index=False)

    nodes, edges = dag_tables(data, dag_dict)
    nodes.to_csv(out / "dag_nodes.csv", index=False)
    edges.to_csv(out / "dag_edges.csv", index=False)
    print(f"DAG: {len(nodes)} nodes, {len(edges)} edges over {len(niches)} niches", flush=True)

    # ---- queries: all-niche search with paths, Stage 1 pool, Stage 2 exact re-rank
    with open(ec.GT_CACHE_DIR / f"{STEM}_ground_truth_block.pkl", "rb") as fh:
        cache = pickle.load(fh)
    if not (np.array_equal(cache["train_idx"], train_idx) and np.array_equal(cache["test_idx"], test_idx)):
        raise ValueError("ground-truth cache split differs from the index split")
    gt = cache["ground_truth"]
    tile_niche, ntc = gt["tile_niche"], gt["niche_train_cache"]
    ref = pd.read_csv(REF_METRICS)
    ref = ref[ref["method"] == "spindle"].set_index("query_idx")["spindle_best_dist"]

    summ, hits, stage1, path_rows, mismatches = [], [], [], [], 0
    t0 = time.perf_counter()
    for q, entry in enumerate(raw["test_tile_covs"]):
        matched, paths_by_niche = search_with_paths(ec.raw_cov(entry), data, dag_dict, config)
        per_niche, pool = defaultdict(list), []
        for g in matched:
            loc = local_of.get(int(g))
            if loc is not None:
                per_niche[tile_niche[loc]].append(loc)
        for cands in per_niche.values():
            pool += cands[:TOP_C]
        g_q = gt["per_query"][q]
        ranked = ec.rerank_block(pool, g_q["query_blocks_log_by_niche"], gt, data)
        qid = int(test_idx[q])
        for method, order in (("spindle", [loc for _, loc in ranked[:K]]), ("exact", g_q["true_order"][:K])):
            for r, loc in enumerate(order, 1):
                hits.append({"query_idx": q, "query_tile_id": qid, "method": method, "rank": r,
                             "tile_id": int(train_idx[loc]), "niche": tile_niche[loc],
                             "distance": g_q["dist_dict"][loc]})
        stage1 += [{"query_idx": q, "stage2_rank": r, "tile_id": int(train_idx[loc]), "niche": tile_niche[loc],
                    "distance": d} for r, (d, loc) in enumerate(ranked, 1)]
        top_tile = int(train_idx[ranked[0][1]]) if ranked else -1
        for c, plist in paths_by_niche.items():
            if not plist:
                continue
            best = min(plist, key=lambda p: p[0])
            with_hit = [p for p in plist if top_tile in p[2]]
            for kind, p in (("best", best), ("hit", min(with_hit, key=lambda p: p[0]) if with_hit else None)):
                if p is not None:
                    path_rows += [{"query_idx": q, "niche": c, "kind": kind, "layer": l, "node": n,
                                   "total_distance": p[0], "leaf_size": len(p[2])} for l, n in enumerate(p[1])]
        sp_best = ranked[0][0] if ranked else np.nan
        summ.append({"query_idx": q, "query_tile_id": qid, "n_stage1": len(pool), "n_matched": len(matched),
                     "exact_best_dist": g_q["closest_dist"], "spindle_best_dist": sp_best,
                     "true_best_niche": g_q["true_best_niche"],
                     "epsilon_true_niche": float(config.epsilon_dict[g_q["true_best_niche"]])})
        if not np.isclose(round(sp_best, 4), ref.loc[q], atol=1e-3):
            mismatches += 1
    print(f"searched {len(summ)} queries in {time.perf_counter() - t0:.1f}s", flush=True)
    if mismatches:
        raise SystemExit(f"{mismatches} queries disagree with {REF_METRICS.name}; not writing query tables")
    pd.DataFrame(summ).to_csv(out / "query_summary.csv", index=False)
    pd.DataFrame(hits).to_csv(out / "query_hits.csv", index=False)
    pd.DataFrame(stage1).to_csv(out / "query_stage1.csv", index=False)
    pd.DataFrame(path_rows).to_csv(out / "query_paths.csv", index=False)
    print(f"Wrote Fig. 1 inputs to {out}; Spindle top-1 distances match {REF_METRICS.name}")


if __name__ == "__main__":
    main()
