"""Cross-platform search, Xenium <-> Visium serial breast sections, on the search tiers (Fig 5, Fig S9).

Data: the two registered serial sections of datasets.yaml (cross_modal_xenium / cross_modal_visium; raw
counts, the 307 shared genes). The Xenium section is quadtree-tiled (--max-pts cells per tile; 2000 in the
main figure, because a Visium spot is ~55 um and a tile needs >= 10 of them) and the same boxes are laid
over the Visium spots (kept if they hold >= 10 spots). Every Visium tile therefore has a co-located Xenium
tile; 9 of the 165 Xenium boxes (at 2000) have no Visium tile.

Per seed (0-4; it seeds only the index build: PCA/UMAP/Leiden) and direction (x2v: Xenium queries against a
Visium index; v2x: the reverse), the index platform's tiles go through build_indexes.run_index (niches and
blocks) and tiers.ExactTier / DagTier (K = 32). Every tile of the query platform is a query. Query and index
covariances are shrunk (tiers.prepare_cov, alpha = metrics.SHRINK_ALPHA) and then a platform correction is
applied to the shrunk query logs (statistics over all query tiles and all index tiles; no labels):
  none       no correction;
  global     one shift of the whole p x p log: L' = (L - mean_q) * min(1, sd_q->sd_t) + mean_t, then exp;
  per_niche  the same shift per niche x block in each niche's own layout (statistics of that niche's tiles).
Distances are d_B (L2 over blocks, / sqrt p) of the corrected query to every index tile.

Readouts per query (the co-located box is the platform-independent truth):
  matched_rank        rank of the co-located index tile among all N (1 = nearest; queries with a match only)
  hit@k               matched_rank <= k, k = 1, 5, 10 (random: k / N)
  d_hit_k             mean physical distance (um) from the query box centre to the top-k hits' centres,
  d_rand              ... to all index tiles' centres (the random-k expectation); ratio_k = d_hit_k / d_rand
  tissue_r / _mae     tissue agreement (exact; --aggregate): per query, the tumour fraction (invasive + DCIS,
                      from each platform's own annotation) of the query box vs the mean of its top-10 hits;
                      Pearson r over queries and mean absolute difference (random: all index tiles)
  oracle              the co-located tile's rank when index tiles are ranked by tumour fraction alone
Spindle-DAG (global arm only, Fig S9): the same readouts on its order, and Overlap(delta, c) etc. against
the exact global d_B (dag_eval_common.retrieval_rows, c = c_grid(N)).

Outputs (results/cross_platform_tiers/):
  tiles<m>_seed<s>_per_query.csv   max_pts, seed, direction, arm, tier, query, readouts
  tiles<m>_seed<s>_top10.csv       the top 10 index tiles per query, direction and arm (exact)
  tiles<m>_seed<s>_dag_curves.csv  DAG vs exact (global) retrieval rows per query and c
  tiles<m>_seed<s>_index.csv       per direction: N, niches, blocks, DAG nodes, MB
  --overlay   tile_boxes_tiles<m>.csv (both platforms, with the co-located Xenium tile), cross_modal_cells.csv
              (all cells / spots with coarse classes), registration_tiles<m>.csv (coarse composition of the
              co-located boxes on both platforms)
  --bias-pca  bias_pca.csv, bias_pca_summary.csv (PCA of shrunk whole logs, fit on the index tiles)
  --example   example_v2x.csv (Fig 5E; seed 0; a DCIS- or immune-rich Visium query)
  --check     sanity checks (no files)
  --aggregate summary.csv, rank_cdf.csv, oracle_rank_cdf.csv, dag_curves.csv, index_summary.csv (mean over
              queries per seed, then mean +/- s.d. over seeds 0-4), tissue_per_query_seed0.csv

  sbatch slurm_jobs/run_cross_platform_tiers.sbatch <seed> <max_pts> | --overlay | --bias-pca | --example |
         --check | --aggregate
"""

import argparse
import pickle
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

import dag_eval_common as de
import experiment_common as ec
import build_indexes
import paths
from spindle_dev import metrics, tiers
from spindle_dev.preprocessing import QuadTile, build_quadtree_tiles, build_tile_covs_full

OUT_DIR = ec.PROJECT_ROOT / "results" / "cross_platform_tiers"
CACHE_DIR = OUT_DIR / "cache"
MAX_PTS_MAIN = 2000
MIN_SPOTS = 10
DIRECTIONS = {"x2v": ("Visium", "Xenium"), "v2x": ("Xenium", "Visium")}  # direction -> (index, query)
ARMS = ["none", "per_niche", "global"]
K_HIT = [1, 5, 10]
K_DIST = [5, 10]
RANK_CDF_MAX = 30
TISSUE_K = 10  # hits whose tumour fraction is compared with the query box's
N_PC = 5

# Shared coarse classes of the Xenium cell types and Visium region labels (Fig 5A, registration check).
# Labels not listed (the Xenium immune types and the hybrids) are "Immune".
COARSE = {
    "invasive": "Invasive tumour", "mixed/invasive": "Invasive tumour",
    "DCIS #1": "DCIS", "DCIS #2": "DCIS",
    "stromal": "Stroma", "stromal/endothelial": "Stroma",
    "immune": "Immune", "stromal/endothelial/immune": "Immune",
    "myoepithelial/stromal/immune": "Myoepithelial",
    "adipocytes": "Adipocytes", "mixed": "Mixed",
    "Invasive_Tumor": "Invasive tumour", "Prolif_Invasive_Tumor": "Invasive tumour",
    "DCIS_1": "DCIS", "DCIS_2": "DCIS",
    "Stromal": "Stroma", "Endothelial": "Stroma", "Perivascular-Like": "Stroma",
    "Myoepi_ACTA2+": "Myoepithelial", "Myoepi_KRT15+": "Myoepithelial",
    "Unlabeled": "Unlabeled",
}
CLASSES = ["Invasive tumour", "DCIS", "Myoepithelial", "Stroma", "Immune", "Adipocytes", "Mixed", "Unlabeled"]
TUMOUR = ["Invasive tumour", "DCIS"]
FOCAL_CLASSES = ["DCIS", "Immune"]  # Fig 5E: classes in separate, recognisable foci on both platforms


# ---------------------------------------------------------------------------------------------- data
def load_platforms():
    """Both sections on their shared genes (sorted), raw counts."""
    import scanpy as sc
    ads = {}
    for mod, key in (("Xenium", "cross_modal_xenium"), ("Visium", "cross_modal_visium")):
        a = sc.read_h5ad(paths.dataset_path(key))
        a.var_names_make_unique()
        ads[mod] = a
    genes = sorted(set(ads["Xenium"].var_names) & set(ads["Visium"].var_names))
    return {m: a[:, genes].copy() for m, a in ads.items()}, genes


def build_tiles(coords_xe, coords_vi, max_pts):
    """Quadtree-tile Xenium, lay the same boxes over Visium (>= MIN_SPOTS spots).

    -> (Xenium tiles, Visium tiles, matched): tile ids are positional per platform and matched[j] is the
    Xenium tile that shares Visium tile j's box.
    """
    tiles_xe = build_quadtree_tiles(coords_xe, max_pts=max_pts, min_side=0.0, max_depth=40)
    tiles_vi, matched = [], []
    for i, t in enumerate(tiles_xe):
        x0, y0, x1, y1 = t.bbox
        inside = np.flatnonzero((coords_vi[:, 0] >= x0) & (coords_vi[:, 0] < x1)
                                & (coords_vi[:, 1] >= y0) & (coords_vi[:, 1] < y1))
        if len(inside) >= MIN_SPOTS:
            tiles_vi.append(QuadTile(len(tiles_vi), t.bbox, inside))
            matched.append(i)
    for i, t in enumerate(tiles_xe):
        t.id = i
    return tiles_xe, tiles_vi, np.asarray(matched)


def setup(max_pts):
    """Tiles, raw covariances and box pairing for one tile size (cached; seed-independent)."""
    f = CACHE_DIR / f"tiles{max_pts}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    ads, genes = load_platforms()
    tiles_xe, tiles_vi, matched = build_tiles(ads["Xenium"].obsm["spatial"], ads["Visium"].obsm["spatial"], max_pts)
    S = {"genes": genes, "max_pts": max_pts, "matched_vi_to_xe": matched,
         "tiles": {"Xenium": tiles_xe, "Visium": tiles_vi},
         "n_obs": {m: a.n_obs for m, a in ads.items()},
         "covs": {m: [c["cov"] for c in build_tile_covs_full(ads[m], ts, n_jobs=8, eps=1e-6)]
                  for m, ts in (("Xenium", tiles_xe), ("Visium", tiles_vi))}}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(f, "wb") as fh:
        pickle.dump(S, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"tiles{max_pts}: {len(genes)} genes, {len(tiles_xe)} Xenium / {len(tiles_vi)} Visium tiles", flush=True)
    return S


def centres(tiles):
    return np.stack([t.centroid for t in tiles])


def match_of(S, direction):
    """matched[q] = co-located index tile of query q (-1 if none)."""
    m = S["matched_vi_to_xe"]
    if direction == "v2x":
        return m.copy()
    out = np.full(len(S["tiles"]["Xenium"]), -1)
    out[m] = np.arange(len(m))
    return out


# ---------------------------------------------------------------------------------------------- correction
def log_sym(C):
    return metrics.log_spd(C, eps=tiers.LOG_FLOOR)


def exp_sym(L):
    w, V = np.linalg.eigh(0.5 * (L + L.T))
    return (V * np.exp(w)) @ V.T


def shift(q_logs, t_logs):
    """(L - mean_q) * min(1, sd_t / sd_q) + mean_t for a stack of query logs (scalar s.d. over all entries)."""
    mean_q, sd_q = q_logs.mean(0), float(q_logs.std())
    mean_t, sd_t = t_logs.mean(0), float(t_logs.std())
    scale = min(1.0, sd_t / (sd_q if sd_q > 1e-6 else 1.0))
    return (q_logs - mean_q) * scale + mean_t


def global_corrected(Qs, Ts):
    """Prepared (shrunk) query covariances after the whole-matrix log shift onto the prepared index tiles."""
    L = shift(np.stack([log_sym(q) for q in Qs]), np.stack([log_sym(t) for t in Ts]))
    return [exp_sym(x) for x in L]


def per_niche_distances(exact, Qs, Ts):
    """(Q, N) d_B with each query's block logs shifted onto each niche's own block-log statistics."""
    D2 = np.zeros((len(Qs), exact.n))
    for k, (idx, vec) in exact.layouts.items():
        q_logs = [vec.block_logs(q) for q in Qs]
        t_logs = [vec.block_logs(Ts[t]) for t in idx]
        corrected = [[None] * len(vec.runs) for _ in Qs]
        for b in range(len(vec.runs)):
            sh = shift(np.stack([ql[b] for ql in q_logs]), np.stack([tl[b] for tl in t_logs]))
            for i in range(len(Qs)):
                corrected[i][b] = sh[i]
        for i in range(len(Qs)):
            qv = np.concatenate(vec.pieces(corrected[i]))
            D2[i, idx] = ((exact.X[k] - qv) ** 2).sum(1, dtype=np.float64)
    return np.sqrt(np.maximum(D2, 0.0))


def exact_distances(exact, Qs):
    return np.stack([exact.distances(exact.query_vectors(q)) for q in Qs])


# ---------------------------------------------------------------------------------------------- readouts
def readouts(order, d, matched, q_xy, idx_xy):
    """Co-located box readouts of one query from its order over the index (best first)."""
    n = len(order)
    rec = {"has_match": matched >= 0}
    if matched >= 0:
        if d is not None:  # exact: tie-aware rank from distances
            rank = 1 + int((d < d[matched]).sum()) + 0.5 * int((d == d[matched]).sum() - 1)
        else:
            rank = 1 + int(np.flatnonzero(order == matched)[0])
        rec["matched_rank"] = rank
        for k in K_HIT:
            rec[f"hit@{k}"] = float(rank <= k)
    dist = np.hypot(*(idx_xy - q_xy).T)
    rec["d_rand"] = float(dist.mean())
    for k in K_DIST:
        rec[f"d_hit_k{k}"] = float(dist[order[:min(k, n)]].mean())
        rec[f"ratio_k{k}"] = rec[f"d_hit_k{k}"] / rec["d_rand"]
    return rec


def build_direction(S, direction, seed):
    index_mod, query_mod = DIRECTIONS[direction]
    tiles, covs = S["tiles"][index_mod], S["covs"][index_mod]
    t0 = time.perf_counter()
    data, _ = build_indexes.run_index(tiles, [{"cov": c, "tile_id": i} for i, c in enumerate(covs)], S["genes"],
                                      SimpleNamespace(n_obs=S["n_obs"][index_mod]), resolution=0.2,
                                      min_final_size=15, max_niche_size=1000, random_state=seed)
    exact = tiers.build_tier("exact", data, covs)
    print(f"[{direction}] index of {len(tiles)} {index_mod} tiles: {len(exact.layouts)} niches, "
          f"{time.perf_counter() - t0:.0f} s", flush=True)
    return data, exact


def run(seed, max_pts):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    S = setup(max_pts)
    tag = f"tiles{max_pts}_seed{seed}"
    per_q, top, curves, info = [], [], [], []
    for direction, (index_mod, query_mod) in DIRECTIONS.items():
        data, exact = build_direction(S, direction, seed)
        dag = tiers.build_tier("dag", data, exact=exact, k_target=de.DAG_CFG["k_target"], alpha=0.0,
                               k_min=de.DAG_CFG["k_target"])
        Ts = [tiers.prepare_cov(c) for c in S["covs"][index_mod]]
        Qs = [tiers.prepare_cov(c) for c in S["covs"][query_mod]]
        Qg = global_corrected(Qs, Ts)
        D = {"none": exact_distances(exact, Qs), "per_niche": per_niche_distances(exact, Qs, Ts),
             "global": exact_distances(exact, Qg)}
        matched = match_of(S, direction)
        q_xy, idx_xy = centres(S["tiles"][query_mod]), centres(S["tiles"][index_mod])
        n, cs = exact.n, de.c_grid(exact.n)
        base = {"max_pts": max_pts, "seed": seed, "direction": direction}
        for arm in ARMS:
            for qi, d in enumerate(D[arm]):
                order = np.argsort(d, kind="stable")
                per_q.append({**base, "arm": arm, "tier": "exact", "query": qi, "matched_tile": matched[qi],
                              **readouts(order, d, matched[qi], q_xy[qi], idx_xy)})
                top += [{**base, "arm": arm, "query": qi, "rank": r + 1, "index_tile": int(t), "d": d[t]}
                        for r, t in enumerate(order[:10])]
        for qi, q in enumerate(Qg):
            ids, approx = dag.adc.scores({k: vec.pieces(vec.block_logs(q)) for k, (_, vec) in dag.layouts.items()})
            order = ids[np.argsort(approx, kind="stable")]
            per_q.append({**base, "arm": "global", "tier": "dag", "query": qi, "matched_tile": matched[qi],
                          **readouts(order, None, matched[qi], q_xy[qi], idx_xy)})
            curves += [{**base, "query": qi, **r} for r in de.retrieval_rows(D["global"][qi], order, cs)]
        info.append({**base, "index_platform": index_mod, "n_index": n, "n_query": len(Qs),
                     "n_query_matched": int((matched >= 0).sum()), "n_niches": len(exact.layouts),
                     "n_blocks": sum(len(v.runs) for _, v in exact.layouts.values()),
                     "dag_nodes": tiers.n_nodes(dag.compact), "exact_mb": exact.nbytes() / de.MB,
                     "dag_mb": dag.nbytes() / de.MB})
        pq = pd.DataFrame([r for r in per_q if r["direction"] == direction and r["tier"] == "exact"])
        print(pq[pq["has_match"]].groupby("arm")[["hit@1", "hit@5", "hit@10", "matched_rank"]].mean()
              .assign(ratio_k10=pq.groupby("arm")["ratio_k10"].mean()).round(3).to_string(), flush=True)
    pd.DataFrame(per_q).to_csv(OUT_DIR / f"{tag}_per_query.csv", index=False)
    pd.DataFrame(top).to_csv(OUT_DIR / f"{tag}_top10.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT_DIR / f"{tag}_dag_curves.csv", index=False)
    pd.DataFrame(info).to_csv(OUT_DIR / f"{tag}_index.csv", index=False)


# ---------------------------------------------------------------------------------------------- extras
def cross_modal_cells(n_xenium=None):
    """Xenium cells and Visium spots with their Cluster and coarse class (Fig 5A, E; Fig 1 glyph); all of them
    unless n_xenium caps the Xenium cells (random sample)."""
    import anndata as ad
    frames = []
    for mod, key, n in (("Xenium", "cross_modal_xenium", n_xenium), ("Visium", "cross_modal_visium", None)):
        a = ad.read_h5ad(paths.dataset_path(key), backed="r")
        xy = np.asarray(a.obsm["spatial"])[:, :2]
        cl = a.obs["Cluster"].astype(str).to_numpy()
        sel = np.arange(len(xy)) if n is None else np.random.default_rng(0).choice(len(xy), n, replace=False)
        frames.append(pd.DataFrame({"modality": mod, "x": xy[sel, 0], "y": xy[sel, 1], "cluster": cl[sel],
                                    "coarse": [COARSE.get(c, "Immune") for c in cl[sel]]}))
    return pd.concat(frames, ignore_index=True)


def overlay(max_pts):
    """Tile boxes with each tile's coarse composition, cells, and the registration check."""
    import anndata as ad
    S = setup(max_pts)
    m = S["matched_vi_to_xe"]
    comp = {}
    for mod, key in (("Xenium", "cross_modal_xenium"), ("Visium", "cross_modal_visium")):
        a = ad.read_h5ad(paths.dataset_path(key), backed="r")
        cl = np.array([COARSE.get(c, "Immune") for c in a.obs["Cluster"].astype(str)])
        comp[mod] = pd.DataFrame([{c: float(np.mean(cl[t.idx] == c)) for c in CLASSES} for t in S["tiles"][mod]])
    rows = []
    for mod in ("Xenium", "Visium"):
        for t in S["tiles"][mod]:
            rows.append({"modality": mod, "id": t.id, "x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2],
                         "y1": t.bbox[3], "n_obs": len(t.idx),
                         "matched_xenium": int(m[t.id]) if mod == "Visium" else t.id,
                         "tumour": float(comp[mod].loc[t.id, TUMOUR].sum()), **comp[mod].loc[t.id].to_dict()})
    pd.DataFrame(rows).to_csv(OUT_DIR / f"tile_boxes_tiles{max_pts}.csv", index=False)
    X, V = comp["Xenium"].iloc[m].reset_index(drop=True), comp["Visium"]
    reg = pd.concat([X.add_prefix("xenium_"), V.add_prefix("visium_")], axis=1)
    reg.insert(0, "xenium_tile", m)
    reg.insert(0, "visium_tile", np.arange(len(m)))
    for mod, F in (("xenium", X), ("visium", V)):
        reg[f"{mod}_tumour"] = F[TUMOUR].sum(1)
    reg.to_csv(OUT_DIR / f"registration_tiles{max_pts}.csv", index=False)
    print(f"tiles{max_pts}: tumour fraction r between co-located boxes = "
          f"{np.corrcoef(reg['xenium_tumour'], reg['visium_tumour'])[0, 1]:.3f} ({len(m)} boxes)")
    if max_pts == MAX_PTS_MAIN:
        cross_modal_cells().to_csv(OUT_DIR / "cross_modal_cells.csv", index=False)


def vectorize(logs):
    p = logs[0].shape[0]
    iu = np.triu_indices(p)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)) / np.sqrt(p)
    return np.stack([L[iu] * w for L in logs])


def bias_pca(max_pts=MAX_PTS_MAIN):
    """PCA of shrunk whole-matrix logs, fit on the index tiles; queries before and after the global shift."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    S = setup(max_pts)
    P = {m: [tiers.prepare_cov(c) for c in cs] for m, cs in S["covs"].items()}
    logs = {m: [log_sym(c) for c in cs] for m, cs in P.items()}
    rows, summary = [], []
    for direction, (index_mod, query_mod) in DIRECTIONS.items():
        V_index = vectorize(logs[index_mod])
        corr = shift(np.stack(logs[query_mod]), np.stack(logs[index_mod]))
        states = {False: vectorize(logs[query_mod]), True: vectorize(list(corr))}
        pca = PCA(n_components=N_PC, random_state=0).fit(V_index)
        for k, z in enumerate(pca.transform(V_index)):
            rows.append({"direction": direction, "modality": index_mod, "role": "index", "corrected": False,
                         "tile_id": k, **{f"PC{j + 1}": z[j] for j in range(N_PC)}})
        for c, V_q in states.items():
            for k, z in enumerate(pca.transform(V_q)):
                rows.append({"direction": direction, "modality": query_mod, "role": "query", "corrected": c,
                             "tile_id": k, **{f"PC{j + 1}": z[j] for j in range(N_PC)}})
            nn = np.array([np.min(np.linalg.norm(V_index - v, axis=1)) for v in V_q])
            lab = np.r_[np.zeros(len(V_index)), np.ones(len(V_q))]
            summary.append({"direction": direction, "corrected": c,
                            "centroid_gap": float(np.linalg.norm(V_index.mean(0) - V_q.mean(0))),
                            "median_nn_dist_to_index": float(np.median(nn)),
                            "modality_silhouette": float(silhouette_score(np.vstack([V_index, V_q]), lab)),
                            "pc1_pc2_explained_var": float(pca.explained_variance_ratio_[:2].sum()),
                            "n_index": len(V_index), "n_query": len(V_q)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "bias_pca.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT_DIR / "bias_pca_summary.csv", index=False)
    print(summ.to_string(index=False))


def example(seed=0, max_pts=MAX_PTS_MAIN):
    """Fig 5E: a v2x query of one recognisable, focal tissue type. Among queries whose co-located Xenium tile
    is in the exact (global) top 5, the one with the largest share of a single FOCAL_CLASSES class (its own
    Visium annotation); the hits' share of that class is reported against the Xenium section average."""
    tag = f"tiles{max_pts}_seed{seed}"
    pq = pd.read_csv(OUT_DIR / f"{tag}_per_query.csv")
    top = pd.read_csv(OUT_DIR / f"{tag}_top10.csv")
    tb = pd.read_csv(OUT_DIR / f"tile_boxes_tiles{max_pts}.csv")
    vis, xen = (tb[tb["modality"] == m].set_index("id") for m in ("Visium", "Xenium"))
    sel = pq[(pq["direction"] == "v2x") & (pq["arm"] == "global") & (pq["tier"] == "exact") & (pq["hit@5"] == 1)]
    share = vis.loc[sel["query"], FOCAL_CLASSES]
    q, cls = share.stack().idxmax()
    q = int(q)
    out = top[(top["direction"] == "v2x") & (top["arm"] == "global") & (top["query"] == q)].copy()
    out["matched"] = out["index_tile"] == int(vis.loc[q, "matched_xenium"])
    out["focal_class"] = cls
    out["query_frac"] = vis.loc[q, cls]
    out["hit_frac"] = xen.loc[out["index_tile"], cls].to_numpy()
    out["section_frac"] = xen[cls].mean()
    out.to_csv(OUT_DIR / "example_v2x.csv", index=False)
    h = out[out["rank"] <= 5]
    print(f"example: Visium query {q} ({len(sel)} candidates), {cls} {vis.loc[q, cls]:.2f}; top-5 Xenium {cls} "
          f"{h['hit_frac'].min():.2f}-{h['hit_frac'].max():.2f} (mean {h['hit_frac'].mean():.2f}, section "
          f"{xen[cls].mean():.2f}); co-located tile at rank {int(out.loc[out['matched'], 'rank'].iloc[0])}")


def check(seed=0, max_pts=MAX_PTS_MAIN):
    """Sanity checks: a platform searched with its own tiles finds each tile at rank 1 at distance 0; the
    query path equals ExactTier.search; the global shift of a platform onto itself is the identity (2e-10 on
    tiles2000, seed 0)."""
    S = setup(max_pts)
    data, exact = build_direction(S, "v2x", seed)  # Xenium index
    covs = S["covs"]["Xenium"]
    Ts = [tiers.prepare_cov(c) for c in covs]
    D = exact_distances(exact, Ts)
    ranks = np.array([1 + (d < d[i]).sum() for i, d in enumerate(D)])
    print(f"self-search: rank 1 for {np.mean(ranks == 1):.3f} of tiles, max self distance {np.diag(D).max():.2e}")
    _, s, _ = exact.search(covs[3])
    print(f"query path vs ExactTier.search: max rel diff {np.max(np.abs(np.sort(D[3]) - s) / np.maximum(s, 1e-9)):.1e}")
    G = exact_distances(exact, global_corrected(Ts, Ts))
    print(f"global shift onto itself: max |d - d_none| = {np.abs(G - D).max():.2e}")
    P = per_niche_distances(exact, Ts, Ts)
    # per_niche re-centres ALL queries onto each niche's own mean, so it is not the identity even here
    print(f"per-niche shift onto itself (not the identity by design): max |d - d_none| over own-niche tiles = "
          f"{max(np.abs(P[np.ix_(idx, idx)] - D[np.ix_(idx, idx)]).max() for idx, _ in exact.layouts.values()):.2e}")


# ---------------------------------------------------------------------------------------------- aggregate
def seed_stats(df, keys):
    """Mean +/- s.d. over seeds 0-4 of per-seed values; columns {col}_mean / {col}_std, n_seeds."""
    df = df[df["seed"].isin(ec.MULTISEED)]
    num = [c for c in df.columns if c not in keys + ["seed"]]
    g = df.groupby(keys)
    out = g[num].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


def aggregate():
    load = lambda suffix: pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob(f"tiles*_seed[0-9]_{suffix}.csv"))])
    keys = ["max_pts", "direction", "arm", "tier"]
    pq = load("per_query")
    info = load("index")
    m = pq[pq["has_match"]].groupby(keys + ["seed"])
    per_seed = m[[f"hit@{k}" for k in K_HIT]].mean()
    per_seed["median_rank"] = m["matched_rank"].median()
    per_seed["mean_rank"] = m["matched_rank"].mean()
    per_seed["n_matched"] = m.size()
    dist = pq.groupby(keys + ["seed"])[[f"ratio_k{k}" for k in K_DIST] + [f"d_hit_k{k}" for k in K_DIST] + ["d_rand"]]
    per_seed = per_seed.join(dist.mean()).reset_index()
    n_index = info.set_index(["max_pts", "direction", "seed"])["n_index"]
    per_seed["n_index"] = [n_index[(r.max_pts, r.direction, r.seed)] for r in per_seed.itertuples()]
    for k in K_HIT:
        per_seed[f"hit@{k}_random"] = np.minimum(1.0, k / per_seed["n_index"])
    seed_stats(per_seed, keys).to_csv(OUT_DIR / "summary.csv", index=False)

    rows = []
    for key, g in pq[pq["has_match"]].groupby(keys + ["seed"]):
        r = g["matched_rank"].to_numpy()
        rows += [dict(zip(keys + ["seed"], key), r=k, frac=float(np.mean(r <= k))) for k in range(1, RANK_CDF_MAX + 1)]
    seed_stats(pd.DataFrame(rows), keys + ["r"]).to_csv(OUT_DIR / "rank_cdf.csv", index=False)

    tissue_rows, oracle_rows, tissue_q = [], [], []
    for f in sorted(OUT_DIR.glob("tiles*_seed[0-9]_top10.csv")):
        top = pd.read_csv(f)
        mp = int(top["max_pts"].iloc[0])
        tb = pd.read_csv(OUT_DIR / f"tile_boxes_tiles{mp}.csv")
        tum = {mod: g.set_index("id")["tumour"].sort_index().to_numpy() for mod, g in tb.groupby("modality")}
        for (d, arm), g in top[top["rank"] <= TISSUE_K].groupby(["direction", "arm"]):
            index_mod, query_mod = DIRECTIONS[d]
            h = g.groupby("query")["index_tile"].apply(lambda s: tum[index_mod][s.to_numpy()].mean())
            q = tum[query_mod][h.index.to_numpy()]
            if int(top["seed"].iloc[0]) == ec.MULTISEED[0]:
                tissue_q.append(pd.DataFrame({"max_pts": mp, "direction": d, "arm": arm, "seed": ec.MULTISEED[0],
                                              "query": h.index, "query_tumour": q, "hits_tumour": h.to_numpy()}))
            tissue_rows.append({"max_pts": mp, "direction": d, "arm": arm, "tier": "exact",
                                "seed": int(top["seed"].iloc[0]), "tissue_r": float(np.corrcoef(q, h)[0, 1]),
                                "tissue_mae": float(np.abs(q - h.to_numpy()).mean()),
                                "tissue_mae_random": float(np.abs(q[:, None] - tum[index_mod][None, :]).mean())})
        if int(top["seed"].iloc[0]) == ec.MULTISEED[0]:
            # label oracle: rank of the co-located tile when index tiles are ranked by |tumour fraction - query's|
            for d, (index_mod, query_mod) in DIRECTIONS.items():
                matched = match_of({"matched_vi_to_xe": tb.loc[tb["modality"] == "Visium", "matched_xenium"]
                                    .to_numpy(), "tiles": {"Xenium": tum["Xenium"]}}, d)
                r = np.array([1 + int((np.abs(tum[index_mod] - tum[query_mod][q]) <
                                       abs(tum[index_mod][t] - tum[query_mod][q])).sum())
                              for q, t in enumerate(matched) if t >= 0])
                oracle_rows += [{"max_pts": mp, "direction": d, "r": k, "frac": float(np.mean(r <= k)),
                                 "median_rank": float(np.median(r))} for k in range(1, RANK_CDF_MAX + 1)]
    tissue = seed_stats(pd.DataFrame(tissue_rows), keys)
    summ = pd.read_csv(OUT_DIR / "summary.csv").merge(tissue, on=keys, how="left", suffixes=("", "_tissue"))
    summ = summ.drop(columns=[c for c in summ.columns if c.endswith("_tissue")])
    summ.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(OUT_DIR / "oracle_rank_cdf.csv", index=False)
    pd.concat(tissue_q).to_csv(OUT_DIR / "tissue_per_query_seed0.csv", index=False)

    cur = load("dag_curves")
    cur = cur.groupby(["max_pts", "direction", "seed", "c"]).mean(numeric_only=True).drop(columns="query").reset_index()
    seed_stats(cur, ["max_pts", "direction", "c"]).to_csv(OUT_DIR / "dag_curves.csv", index=False)
    seed_stats(info.drop(columns=["index_platform"]), ["max_pts", "direction"]).to_csv(OUT_DIR / "index_summary.csv",
                                                                                        index=False)
    s = pd.read_csv(OUT_DIR / "summary.csv")
    print(s[keys + ["n_seeds", "hit@1_mean", "hit@5_mean", "hit@10_mean", "median_rank_mean", "ratio_k10_mean",
                    "hit@5_random_mean", "tissue_r_mean", "tissue_mae_mean"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-pts", type=int, default=MAX_PTS_MAIN)
    mode = ap.add_mutually_exclusive_group()
    for flag in ("overlay", "bias-pca", "example", "check", "aggregate"):
        mode.add_argument(f"--{flag}", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
    elif args.overlay:
        overlay(args.max_pts)
    elif args.bias_pca:
        bias_pca(args.max_pts)
    elif args.example:
        example(args.seed, args.max_pts)
    elif args.check:
        check(args.seed, args.max_pts)
    else:
        run(args.seed, args.max_pts)


if __name__ == "__main__":
    main()
