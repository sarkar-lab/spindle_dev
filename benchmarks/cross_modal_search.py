"""Cross-modal (Xenium <-> Visium) search benchmark, searching EVERY niche.

Queries from one platform are searched against a Spindle index built on the
other platform (matched serial breast-cancer sections, shared gene subset).
Mirrors benchmarks/holdout_validation.py's methodology:

- every niche's DAG is searched per query (no single-niche routing; see
  ``holdout_validation.perform_search``'s docstring for why routing was removed),
- ground truth is the exact block-diagonalized log-Euclidean distance over
  every niche (``holdout_validation.compute_ground_truth``),
- metrics are ``recall_at_eps_*`` / ``overlap_at_eps_*`` plus speedup against
  the whole-matrix brute-force cost (``holdout_validation.evaluate_against_ground_truth``).

The one cross-modal-specific step is a tangent-space modality bias
correction (``--correction``, default ``global``): each query's whole-matrix
log is standardized against the mean/std of ALL queries and rescaled onto the
mean/std of ALL index tiles, once, before any niche's permutation/block layout
is applied. The corrected query then goes through exactly the holdout
pipeline, and the same corrected matrix defines the ground truth.

``per_niche`` (correct separately in every niche x block layout) and ``none``
are kept only as diagnostics. ``per_niche`` re-centres every query onto each
niche's own mean, which makes all niches look equally close and collapses the
true nearest neighbour onto whichever niche is tightest. On seed 0 that
starved Stage-1 on v2x (recall@eps0.1 = 0.24 vs 0.98 for ``global``); see
paper/results_tracking.md, E12.

``--seed`` controls the random query subsample and the PCA/UMAP/Leiden
random_state of the index build.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from tqdm.auto import tqdm

_this_dir = Path(__file__).resolve().parent
project_root = _this_dir.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(_this_dir))

import spindle_dev.search as search  # noqa: E402
from spindle_dev.preprocessing import build_quadtree_tiles, QuadTile, build_tile_covs_full  # noqa: E402
from spindle_dev.utils import log_spd, exp_spd  # noqa: E402
import holdout_validation as hv  # type: ignore  # noqa: E402
import index_datasets  # type: ignore  # noqa: E402
from run_logging import RunLogger  # type: ignore  # noqa: E402

DEFAULT_DATA_DIR = project_root / "dataset" / "cross_modal"
RESULTS_DIR = project_root / "results" / "cross_modal_search"
DATASET_TAG = "cross_modal_brca"


# =====================================================================
# Modality bias correction
# =====================================================================

def compute_corrected_query_logs(query_covs, train_covs, data):
    """Return ``corrected[i][niche] = [corrected log block, ...]`` for every query and niche.

    For niche ``c`` and block ``b``: ``L_corr = (L_q - mean_q) * min(1, std_t/std_q) + mean_t``,
    where ``mean_t/std_t`` come from niche ``c``'s training tiles and
    ``mean_q/std_q`` from ALL queries projected into niche ``c``'s layout.
    The scale is capped at 1.0 so the correction only ever shrinks query
    spread, never amplifies it (unchanged from the original routed version).
    """
    labels = np.asarray(data.labels).astype(int)
    unique_niches = sorted(set(labels.tolist()))
    corrected = [dict() for _ in query_covs]

    for c in unique_niches:
        perm = data.perm_list[c]
        block_runs = data.block_dict[c]
        train_members = np.where(labels == c)[0]

        q_perm_all = [q[np.ix_(perm, perm)] for q in query_covs]
        t_perm_all = [train_covs[t][np.ix_(perm, perm)] for t in train_members]

        for i in range(len(query_covs)):
            corrected[i][c] = []

        for s, e in block_runs:
            t_logs = np.stack([log_spd(t[s:e, s:e]) for t in t_perm_all])
            q_logs = np.stack([log_spd(q[s:e, s:e]) for q in q_perm_all])
            mean_t, std_t = t_logs.mean(axis=0), float(np.std(t_logs))
            mean_q, std_q = q_logs.mean(axis=0), float(np.std(q_logs))
            if std_q <= 1e-6:
                std_q = 1.0
            scale = min(1.0, std_t / std_q)
            for i in range(len(query_covs)):
                corrected[i][c].append((q_logs[i] - mean_q) * scale + mean_t)

    return corrected


def compute_global_corrected_covs(query_covs, train_covs):
    """Whole-matrix (niche-free) variant: one tangent-space shift for every query.

    ``L_corr = (log Q - mean_q) * min(1, std_t/std_q) + mean_t`` on the full
    gene x gene log matrix, with means/stds over ALL query / ALL index tiles.
    Applied once, before any niche's permutation/block layout is used, so it
    does not re-centre queries onto each niche separately.
    """
    q_logs = np.stack([log_spd(q) for q in query_covs])
    t_logs = np.stack([log_spd(t) for t in train_covs])
    mean_q, std_q = q_logs.mean(axis=0), float(np.std(q_logs))
    mean_t, std_t = t_logs.mean(axis=0), float(np.std(t_logs))
    scale = min(1.0, std_t / (std_q if std_q > 1e-6 else 1.0))
    return [exp_spd((L - mean_q) * scale + mean_t) for L in q_logs]


def blocks_log_per_niche(query_covs, data):
    """``out[i][niche] = [log block, ...]`` of each query in each niche's own layout (no correction)."""
    labels = np.asarray(data.labels).astype(int)
    out = [dict() for _ in query_covs]
    for c in sorted(set(labels.tolist())):
        perm, block_runs = data.perm_list[c], data.block_dict[c]
        for i, q in enumerate(query_covs):
            q_perm = q[np.ix_(perm, perm)]
            out[i][c] = [log_spd(q_perm[s:e, s:e]) for s, e in block_runs]
    return out


def _corrected_query_matrix(blocks_log, block_runs, p):
    """Assemble a permuted-space query matrix from corrected per-block logs.

    Only the diagonal blocks are ever read by ``search_index`` (it slices
    ``query_spd[s:e, s:e]`` per block), so off-block entries are left at zero.
    """
    M = np.zeros((p, p), dtype=np.float64)
    for L, (s, e) in zip(blocks_log, block_runs):
        M[s:e, s:e] = exp_spd(L)
    return M


# =====================================================================
# Stage 1: all-niche DAG search on corrected queries
# =====================================================================

def perform_search_corrected(corrected, data, dag_dict, config, budget_multiplier=1.0,
                             restrict_niches=None):
    """Mirror of ``holdout_validation.perform_search`` for per-niche corrected queries.

    Same per-niche SearchConfig caps, budget formula and cross-niche candidate
    merge; the only difference is that the query handed to niche ``c``'s DAG
    is that query's niche-``c`` bias-corrected version. ``restrict_niches``
    (optional list, one niche id per query) limits each query to a single
    niche -- used only by ``--single-niche-baseline``.
    """
    labels = np.asarray(data.labels).astype(int)
    unique_niches = sorted(set(labels.tolist()))
    niche_sizes = {c: int(np.sum(labels == c)) for c in unique_niches}
    niche_cfgs = {}
    for c, n in niche_sizes.items():
        f = hv._niche_scale_factor(n)
        niche_cfgs[c] = search.SearchConfig(
            max_results=None, debug=False,
            max_failed_starts=max(1, round(100 * f)),
            max_failed_paths=max(1, round(200 * f)),
            total_paths_limit=max(1, round(3000 * f)),
        )
    p = data.num_genes

    all_matched, times = [], []
    hits_per_niche = {c: 0 for c in unique_niches}
    for i, q_corr in enumerate(tqdm(corrected, desc="Querying index")):
        niches = unique_niches if restrict_niches is None else [int(restrict_niches[i])]
        matched, seen, q_time = [], set(), 0.0
        for c in niches:
            index_handle = dag_dict[c]
            f = hv._niche_scale_factor(niche_sizes[c])
            budget = (float(config.epsilon_dict[c]) * float(len(index_handle.sorted_blocks))
                      * float(budget_multiplier) * f)
            block_runs = data.block_dict[c]
            q_mat = _corrected_query_matrix(q_corr[c], block_runs, p)

            t0 = time.perf_counter()
            results = search.search_index(index_handle, q_mat, [], block_runs, budget, config=niche_cfgs[c])
            q_time += time.perf_counter() - t0

            # search_index returns a list ([SearchResults(paths=[])]) on its early exits.
            paths = [] if isinstance(results, list) else (results.paths or [])
            for path in paths:
                member_sets = []
                for node_id in path.node_path:
                    node = index_handle.nodes[node_id]
                    members = getattr(getattr(node, "metadata", None), "members", [])
                    member_sets.append({int(spd_id) for spd_id, _ in members})
                intersect_ids = set.intersection(*member_sets) if member_sets else set()
                for spd_id in sorted(intersect_ids):
                    if spd_id not in seen:
                        seen.add(spd_id)
                        matched.append(spd_id)
                        hits_per_niche[c] += 1
        all_matched.append(matched)
        times.append(q_time)

    print(f"Stage-1 candidates per niche (summed over queries): {hits_per_niche}")
    return all_matched, times


# =====================================================================
# One direction
# =====================================================================

def run_direction(direction, args, tiles, covs, adatas, common_genes, out_dir):
    index_mod, query_mod = ("vi", "xe") if direction == "x2v" else ("xe", "vi")
    tiles_index, covs_index, adata_index = tiles[index_mod], covs[index_mod], adatas[index_mod]
    covs_query = covs[query_mod]
    dataset_name = f"{DATASET_TAG}_{direction}_seed{args.seed}"

    rng = np.random.default_rng([args.seed, 0 if direction == "x2v" else 1])
    n_q = len(covs_query) if args.n_queries is None else min(args.n_queries, len(covs_query))
    query_sel = np.sort(rng.choice(len(covs_query), size=n_q, replace=False))
    query_covs = [covs_query[j]["cov"] for j in query_sel]
    query_tile_ids = [covs_query[j]["tile_id"] for j in query_sel]
    print(f"\n[{direction}] {n_q}/{len(covs_query)} query tiles, {len(covs_index)} index tiles")

    with RunLogger(dataset_name=dataset_name, stage="cross_modal_search",
                   out_dir=project_root / "results" / "run_logs", seed=args.seed,
                   direction=direction, n_queries=n_q, budget_mult=args.budget_mult,
                   n_tiles_index=len(covs_index), n_tiles_query=len(covs_query),
                   n_common_genes=len(common_genes)):
        t0 = time.perf_counter()
        data, _ = index_datasets.run_index(tiles_index, covs_index, common_genes, adata_index,
                                           resolution=0.2, min_final_size=15, max_niche_size=1000,
                                           random_state=args.seed)
        dag_dict, config = index_datasets.configure_and_build_dag(data)
        build_time_s = time.perf_counter() - t0
        n_niches = len(set(int(x) for x in data.labels))
        print(f"[{direction}] index built in {build_time_s:.1f}s, {n_niches} niches")

        train_covs = [t["cov"] for t in covs_index]
        if args.correction == "per_niche":
            corrected = compute_corrected_query_logs(query_covs, train_covs, data)
        elif args.correction == "global":
            corrected = blocks_log_per_niche(compute_global_corrected_covs(query_covs, train_covs), data)
        else:
            corrected = blocks_log_per_niche(query_covs, data)

        gt_block = hv.compute_ground_truth(query_covs, train_covs, data,
                                           query_blocks_log_override=corrected)
        gt_whole = hv.compute_ground_truth_whole_matrix(query_covs, train_covs)

        variants = [("all_niche", None)]
        if args.single_niche_baseline:
            assigned = search.assign_clusters_to_new_spds(query_covs, data, strategy="knn_majority",
                                                          n_neighbors=5)
            variants.append(("single_niche_baseline", [int(a) for a in assigned]))

        for variant, restrict in variants:
            matched, times = perform_search_corrected(corrected, data, dag_dict, config,
                                                      budget_multiplier=args.budget_mult,
                                                      restrict_niches=restrict)
            df, summary = hv.evaluate_against_ground_truth(
                gt_block, gt_block, data.spd_ids, matched, times, data, dataset_name, config,
                ground_truth_kind="block", top_c_candidates=args.top_c, ground_truth_whole=gt_whole,
            )
            extra = {"direction": direction, "seed": args.seed, "variant": variant,
                     "correction": args.correction,
                     "n_niches": n_niches, "n_tiles_index": len(covs_index),
                     "n_tiles_query": len(covs_query), "build_time_s": round(build_time_s, 2)}
            df.insert(1, "query_tile_id", query_tile_ids)
            if restrict is not None:
                df["searched_niche"] = restrict
            df["true_best_niche"] = [q["true_best_niche"] for q in gt_block["per_query"]]
            for k, v in extra.items():
                df[k] = v
            summary = {**extra, **summary}

            suffix = "" if variant == "all_niche" else f"_{variant}"
            if args.correction != "global":
                suffix += f"_corr-{args.correction}"
            df.to_csv(out_dir / f"{direction}{suffix}_query_metrics.csv", index=False)
            pd.DataFrame([summary]).to_csv(out_dir / f"{direction}{suffix}_summary.csv", index=False)
            print(f"[{direction}/{variant}] wrote {out_dir / f'{direction}{suffix}_query_metrics.csv'}")


# =====================================================================
# Main
# =====================================================================

def build_tiles(adata_xe, adata_vi):
    """Quadtree-tile Xenium, then overlay the same bounding boxes on Visium."""
    coords_xe = adata_xe.obsm["spatial"]
    coords_vi = adata_vi.obsm["spatial"]
    tiles_xe = build_quadtree_tiles(coords_xe, max_pts=2000, min_side=0.0, max_depth=40)

    tiles_vi = []
    for t in tiles_xe:
        x0, y0, x1, y1 = t.bbox
        mask = ((coords_vi[:, 0] >= x0) & (coords_vi[:, 0] < x1)
                & (coords_vi[:, 1] >= y0) & (coords_vi[:, 1] < y1))
        child_idx = np.where(mask)[0]
        if len(child_idx) >= 10:
            tiles_vi.append(QuadTile(t.id, t.bbox, child_idx))

    # Tile ids are positional so that spd_ids == position in the covariance list.
    for i, t in enumerate(tiles_xe):
        t.id = i
    for i, t in enumerate(tiles_vi):
        t.id = i
    return tiles_xe, tiles_vi


def write_overlay_csvs(tiles_xe, tiles_vi, coords_xe, coords_vi):
    """Figure-panel inputs (scripts/organize_panel_data.py panel G) -- seed-independent."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    boxes = [{'modality': mod, 'id': t.id, 'x0': t.bbox[0], 'y0': t.bbox[1], 'x1': t.bbox[2], 'y1': t.bbox[3]}
             for mod, ts in (("Xenium", tiles_xe), ("Visium", tiles_vi)) for t in ts]
    pd.DataFrame(boxes).to_csv(RESULTS_DIR / "tile_overlay_boxes.csv", index=False)

    rng = np.random.default_rng(42)
    records = []
    for mod, coords in (("Xenium", coords_xe), ("Visium", coords_vi)):
        pts = coords[rng.choice(len(coords), size=min(5000, len(coords)), replace=False)]
        records.extend({'modality': mod, 'x': x, 'y': y} for x, y in pts[:, :2])
    pd.DataFrame(records).to_csv(RESULTS_DIR / "spatial_coords_sample.csv", index=False)
    print(f"Wrote tile_overlay_boxes.csv / spatial_coords_sample.csv to {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Cross-modal (Xenium<->Visium) all-niche Spindle search")
    parser.add_argument("--direction", choices=["x2v", "v2x", "both"], default="both")
    parser.add_argument("--xenium-path", type=Path, default=DEFAULT_DATA_DIR / "xenium_rotated.h5ad")
    parser.add_argument("--visium-path", type=Path, default=DEFAULT_DATA_DIR / "visium_rotated.h5ad")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds the query subsample and the index build's PCA/UMAP/Leiden random_state.")
    parser.add_argument("--n-queries", type=int, default=50,
                        help="Queries per direction, drawn at random (without replacement) from the query "
                             "platform's tiles. Capped at the number of available tiles.")
    parser.add_argument("--budget-mult", type=float, default=1.0,
                        help="Search budget multiplier (production value 1.0, as in holdout_validation.py).")
    parser.add_argument("--top-c", type=int, default=400,
                        help="Per-niche Stage-1 candidate cap before Stage-2 exact re-ranking.")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Default: results/cross_modal_search/seed_<seed>/")
    parser.add_argument("--correction", choices=["global", "per_niche", "none"], default="global",
                        help="Modality bias correction: global (one whole-matrix tangent-space shift; production), "
                             "per_niche (per niche x block, stats over all queries; diagnostic), or none.")
    parser.add_argument("--single-niche-baseline", action="store_true",
                        help="Diagnostic: also run the old kNN-routed single-niche search, scored against "
                             "the same all-niche ground truth (writes *_single_niche_baseline_* files).")
    parser.add_argument("--write-overlay", action="store_true",
                        help="Also (re)write the seed-independent tile overlay / coordinate-sample CSVs.")
    args = parser.parse_args()

    out_dir = args.out_dir or (RESULTS_DIR / f"seed_{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    adata_vi = sc.read_h5ad(args.visium_path)
    adata_xe = sc.read_h5ad(args.xenium_path)
    adata_vi.var_names_make_unique()
    adata_xe.var_names_make_unique()
    common_genes = sorted(set(adata_vi.var_names).intersection(adata_xe.var_names))
    print(f"Found {len(common_genes)} common genes.")
    if not common_genes:
        raise SystemExit("No common genes found!")
    adata_xe = adata_xe[:, common_genes].copy()
    adata_vi = adata_vi[:, common_genes].copy()

    tiles_xe, tiles_vi = build_tiles(adata_xe, adata_vi)
    print(f"Built {len(tiles_xe)} Xenium tiles and {len(tiles_vi)} Visium tiles.")
    if args.write_overlay:
        write_overlay_csvs(tiles_xe, tiles_vi, adata_xe.obsm["spatial"], adata_vi.obsm["spatial"])

    print("Computing tile covariance matrices...")
    covs = {"xe": build_tile_covs_full(adata_xe, tiles_xe, gene_idx=None, n_jobs=8, eps=1e-6),
            "vi": build_tile_covs_full(adata_vi, tiles_vi, gene_idx=None, n_jobs=8, eps=1e-6)}
    tiles = {"xe": tiles_xe, "vi": tiles_vi}
    adatas = {"xe": adata_xe, "vi": adata_vi}

    directions = ["x2v", "v2x"] if args.direction == "both" else [args.direction]
    for direction in directions:
        run_direction(direction, args, tiles, covs, adatas, common_genes, out_dir)


if __name__ == "__main__":
    main()
