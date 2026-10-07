"""Gene modules, enrichment and module-distance maps for the biology figures.

Standalone reimplementation of the ISMB notebooks' module analysis (never imports or
runs them), on the seed-73 production builds in results/indexes/.
For each dataset with a selection file in scripts/figure_data/bio_configs/<dataset>.yaml:

  1. score every DAG node of every niche (index.score_nodes_from_a_cluster: centroid
     correlation -> top-30 |r| gene graph -> greedy-modularity modules; node score
     E * Q * (1 - S)) and rank all nodes by score;
  2. check the configured (niche, layer, node) against the selection rule: among the
     modules whose Jaccard overlap with the original figure's module is >= 0.8 x the best
     overlap, take the node with the best global rank (fails loudly if they disagree);
  3. keep the node's modules as the heatmap does (>= 2 genes, mean |r| > 0.05);
     the highlighted module is the kept module closest to the reference genes;
  4. enrichment of every kept module with >= 3 genes, MSigDB Hallmark 2020 and
     GO BP 2023: hypergeometric test against the dataset's gene panel (offline, gseapy.enrich,
     BH-FDR; GMTs in results/biology/genesets/) and Enrichr's genome-wide background
     (gseapy.enrichr, online), side by side;
  5. per training tile, on the highlighted module's genes: log-Euclidean distance between
     the tile's correlation matrix and the module centroid (the layer's mean log-centroid),
     the module's coherence, and the notebook's score (see tile_scores).

Outputs, results/biology/<dataset>/:
  selection.csv    the chosen node, its rank / score and the rule's check
  node_scores.csv  every DAG node (niche, layer, node, score, E, Q, S, n_tiles, rank)
  modules.csv      the node's genes: module, kept, highlight, plot order
  corr.csv         the node's centroid correlation matrix (block gene order)
  enrichment.csv   tidy enrichment table, both backgrounds, both libraries
  tile_scores.csv  per training tile: bbox, niche, d_module, coherence, d_layer_mean
Breast only (Fig. 6G), from the E9-lite seed-0 per-cell scores:
  signature_tile_scores.csv  mean signature score per seed-0 training tile + top-10 flag

Nothing here is random (greedy modularity and the tests are deterministic); the numpy
seed is pinned anyway. Run through SLURM: sbatch slurm_jobs/figures/run_bio_modules.sbatch
"""

import argparse
import functools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

import paths  # noqa: E402,F401  (puts src/ and every benchmarks/ folder on sys.path)
import experiment_common as ec  # noqa: E402
import tqdm  # noqa: E402
from spindle_dev import index, metrics  # noqa: E402

index.tqdm.tqdm = functools.partial(tqdm.tqdm, disable=True)

CONFIG_DIR = PROJECT_ROOT / "scripts" / "figure_data" / "bio_configs"
OUT_ROOT = PROJECT_ROOT / "results" / "biology"
GENESET_DIR = OUT_ROOT / "genesets"
LIBRARIES = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2023"]
SEED = 0
MIN_MODULE = 2          # heatmap filter (plot_corr_heatmap_with_modules, filter_zero_correlation)
MIN_MEAN_ABS_R = 0.05
MIN_ENRICH = 3          # go_score.enrich_modules_with_gseapy skips smaller modules
JACCARD_FRAC = 0.8
E9_DIR = PROJECT_ROOT / "results" / "gene_signature_search" / "breast"


def load_config(key):
    with open(CONFIG_DIR / f"{key}.yaml") as fh:
        return yaml.safe_load(fh)


def mean_abs_offdiag(R, idx):
    if len(idx) < 2:
        return 0.0
    sub = np.abs(R[np.ix_(idx, idx)])
    return float(sub[~np.eye(len(idx), dtype=bool)].mean())


def score_all_nodes(data, dag_dict):
    """Every node's score record (score_nodes_from_a_cluster) plus a global rank."""
    recs = {}
    for c in sorted(dag_dict):
        for r in index.score_nodes_from_a_cluster(data, c, dag_dict, take_representative_mean=True):
            recs[(int(c), int(r["node_id"]))] = r
    rows = [{"niche": c, "layer": int(r["block_id"]), "node": n, "score": r["node_score"], "E": r["E"],
             "Q": r["Q"], "S": r["S"], "n_tiles": r["num_spds"], "block_size": r["block_size"],
             "n_modules": len(r["modules"])} for (c, n), r in recs.items()]
    df = pd.DataFrame(rows)
    df["rank"] = df["score"].rank(ascending=False, method="min").astype(int)
    return df.sort_values("rank").reset_index(drop=True), recs


def check_selection(cfg, nodes, recs):
    """Apply the selection rule over every module of every node; compare with the config."""
    ref = set(cfg["reference"]["genes"])
    rank = nodes.set_index(["niche", "node"])["rank"]
    rows = []
    for (c, n), r in recs.items():
        for m in r["modules"]:
            ov = len(ref & set(m))
            rows.append({"niche": c, "node": n, "layer": int(r["block_id"]), "overlap": ov,
                         "jaccard": ov / len(ref | set(m)), "rank": int(rank[(c, n)])})
    cand = pd.DataFrame(rows)
    best = cand["jaccard"].max()
    ok = cand[cand["jaccard"] >= JACCARD_FRAC * best].sort_values(["rank", "jaccard"], ascending=[True, False])
    pick = ok.iloc[0]
    chosen = (cfg["niche"], cfg["layer"], cfg["node"])
    rule = (int(pick.niche), int(pick.layer), int(pick.node))
    if chosen != rule:
        raise ValueError(f"{cfg['dataset']}: config picks niche/layer/node {chosen}, the rule picks {rule} "
                         f"(rank {pick['rank']}, Jaccard {pick.jaccard:.2f}); fix the YAML")
    return {"best_jaccard_any_node": best, "n_candidate_modules": len(ok)}


def node_modules(cfg, rec):
    """The node's modules as the heatmap keeps them, with the highlighted one marked."""
    R = rec["R"].copy()
    genes = list(rec["G"].nodes())  # block (permuted) order; R is aligned with it
    if len(genes) != R.shape[0]:
        raise ValueError("gene graph and correlation matrix disagree")
    g2i = {g: i for i, g in enumerate(genes)}
    ref = set(cfg["reference"]["genes"])
    mods = []
    for mi, m in enumerate(rec["modules"]):
        idx = [g2i[g] for g in m]
        mar = mean_abs_offdiag(R, idx)
        mods.append({"module": mi, "genes": sorted(m), "size": len(m), "mean_abs_r": mar,
                     "kept": len(m) >= MIN_MODULE and mar > MIN_MEAN_ABS_R,
                     "jaccard_ref": len(ref & set(m)) / len(ref | set(m))})
    kept = [m for m in mods if m["kept"]]
    hi = max(kept, key=lambda m: m["jaccard_ref"])["module"]
    rows, order = [], 0
    for m in mods:
        for g in m["genes"]:
            rows.append({"gene": g, "module": m["module"], "module_size": m["size"],
                         "module_mean_abs_r": m["mean_abs_r"], "kept": m["kept"], "highlight": m["module"] == hi,
                         "jaccard_ref": m["jaccard_ref"], "in_reference": g in ref,
                         "plot_order": order if m["kept"] else -1})
            order += int(m["kept"])
    np.fill_diagonal(R, 1.0)
    corr = pd.DataFrame(R, index=genes, columns=genes)
    return pd.DataFrame(rows), corr, mods, hi


def enrich(mods, panel, dataset):
    import gseapy as gp

    rows = []
    for m in mods:
        if not m["kept"] or m["size"] < MIN_ENRICH:
            continue
        for lib in LIBRARIES:
            res = gp.enrich(gene_list=m["genes"], gene_sets=str(GENESET_DIR / f"{lib}.gmt"), background=panel,
                            outdir=None, cutoff=1.0, no_plot=True).results
            res["background"] = "panel"
            frames = [res]
            for attempt in range(4):
                try:
                    web = gp.enrichr(gene_list=m["genes"], gene_sets=[lib], organism="human", outdir=None,
                                     cutoff=1.0, no_plot=True).results
                    web["background"] = "genome_enrichr"
                    frames.append(web)
                    break
                except Exception as e:  # network hiccup: retry, then fail loudly
                    if attempt == 3:
                        raise RuntimeError(f"Enrichr failed for {dataset} module {m['module']} / {lib}") from e
                    time.sleep(5 * (attempt + 1))
            for df in frames:
                for _, r in df.iterrows():
                    k, n = (int(v) for v in str(r["Overlap"]).split("/"))
                    rows.append({"dataset": dataset, "module": m["module"], "module_size": m["size"],
                                 "library": lib, "background": r["background"],
                                 "term": r["Term"], "overlap": k, "set_size": n, "p": r["P-value"],
                                 "fdr": r["Adjusted P-value"], "genes": r["Genes"]})
    out = pd.DataFrame(rows)
    out["rank_in_test"] = out.groupby(["module", "library", "background"])["p"].rank(method="first").astype(int)
    return out.sort_values(["module", "library", "background", "rank_in_test"])


def log_spd(C):
    return metrics.log_spd(0.5 * (C + C.T))


def cov_to_corr(C):
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def tile_scores(cfg, data, dag, covs, hi_genes):
    """Per training tile, on the highlighted module's k genes:

    d_module      LE distance between the tile's correlation matrix and the module centroid's,
                  over every tile. The centroid is the layer's mean log-centroid (mean of every
                  node's representative mean in the layer, index.get_node_log_ref -- the notebook's
                  reference), mapped back to a covariance. Correlations, not covariances, so that
                  expression level does not dominate (d on covariances tracks mean log-variance).
    coherence     mean off-diagonal correlation of the module genes in the tile.
    d_layer_mean  the notebook's score (tile_module_scores_from_reference): LE distance of the
                  covariance sub-block to the sub-block of the layer mean log-centroid, niche tiles
                  only; kept to check the reproduction of the original map.
    """
    genes = np.array(data.metadata["genes"])
    perm, (b0, b1) = dag.perm, dag.block_runs[cfg["layer"]]
    block_genes = list(genes[perm][b0:b1])
    node = next(n for n in dag.nodes if n.global_node_id == cfg["node"])
    if node.block_index != cfg["layer"]:
        raise ValueError("node is not in the configured layer")
    loc = [block_genes.index(g) for g in hi_genes]                 # within the block
    gidx = np.array([int(perm[b0 + i]) for i in loc])              # global gene index
    k = len(loc)
    off = ~np.eye(k, dtype=bool)
    L_layer_full = index.get_node_log_ref(dag, cfg["layer"], use_representative_mean=True)
    L_layer = L_layer_full[np.ix_(loc, loc)]
    L_centroid = log_spd(cov_to_corr(index.expm_sym(L_layer_full)[np.ix_(loc, loc)]))

    # Coordinate check: a single-tile node's centroid is the log of its member tile's block.
    members = {int(m[0]) for m in node.metadata.members}
    if len(members) == 1:
        k_m = [int(s) for s in data.spd_ids].index(next(iter(members)))
        pb = perm[b0:b1]
        d0 = np.linalg.norm(log_spd(np.asarray(covs[k_m], dtype=np.float64)[np.ix_(pb, pb)])
                            - node.metadata.representative_mean)
        if d0 > 1e-3:
            raise ValueError(f"member tile of a single-tile node is {d0:.4g} from it; coordinate mix-up")

    labels = np.asarray(data.labels)
    rows = []
    for k_t, (sid, t) in enumerate(zip(data.spd_ids, data.metadata["tiles"])):
        C = np.asarray(covs[k_t], dtype=np.float64)[np.ix_(gidx, gidx)]
        R = cov_to_corr(C)
        in_niche = labels[k_t] == cfg["niche"]
        rows.append({"tile_id": int(sid), "x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2], "y1": t.bbox[3],
                     "n_cells": int(t.idx.size), "niche": int(labels[k_t]),
                     "d_module": np.linalg.norm(log_spd(R) - L_centroid) / np.sqrt(k),
                     "coherence": float(R[off].mean()),
                     "d_layer_mean": np.linalg.norm(log_spd(C) - L_layer) / np.sqrt(k) if in_niche else np.nan})
    df = pd.DataFrame(rows)
    df["node_member"] = df["tile_id"].isin(members)
    return df


def breast_signature_tiles():
    """Seed-0 training tiles: mean per-cell signature score (E9-lite) and the Spindle top-10 flag."""
    bundle = ec.load_index(ec.DATASETS["breast_cancer"], 0)
    tiles = bundle["data"].metadata["tiles"]
    sids = bundle["data"].spd_ids
    sigs = sorted(p.name.replace("_spatial_cells.csv", "") for p in E9_DIR.glob("*_spatial_cells.csv"))
    out = pd.DataFrame({"tile_id": [int(s) for s in sids],
                        "x0": [t.bbox[0] for t in tiles], "y0": [t.bbox[1] for t in tiles],
                        "x1": [t.bbox[2] for t in tiles], "y1": [t.bbox[3] for t in tiles]})
    for sig in sigs:
        score = pd.read_csv(E9_DIR / f"{sig}_spatial_cells.csv", usecols=["score"])["score"].to_numpy()
        out[sig] = [score[t.idx].mean() for t in tiles]
        m = pd.read_csv(E9_DIR / f"{sig}_top_matches.csv")
        m = m[(m.seed == 0) & (m.construction == "coexpression") & (m["rank"] <= 10)]
        chk = out.set_index("tile_id").loc[m.tile_id, sig].to_numpy()
        if not np.allclose(chk, m.tile_score.to_numpy()):
            raise ValueError(f"{sig}: recomputed tile scores differ from E9-lite's top_matches")
        out[f"{sig}_top10"] = out.tile_id.isin(m.tile_id)
    return out


def run(key):
    np.random.seed(SEED)
    cfg = load_config(key)
    out = OUT_ROOT / key
    out.mkdir(parents=True, exist_ok=True)
    stem = ec.DATASETS[key]
    t0 = time.time()
    bundle = ec.load_index(stem, ec.PRODUCTION_SEED)
    data, dag_dict = bundle["data"], bundle["dag_dict"]
    raw = ec.load_raw_covs(stem, ec.PRODUCTION_SEED)
    covs = [ec.raw_cov(c) for c in raw["train_tile_covs"]]
    if [int(c["tile_id"]) for c in raw["train_tile_covs"]] != [int(s) for s in data.spd_ids]:
        raise ValueError("raw covariances are not aligned with data.spd_ids")
    del raw
    panel = list(data.metadata["genes"])

    nodes, recs = score_all_nodes(data, dag_dict)
    nodes.to_csv(out / "node_scores.csv", index=False)
    chk = check_selection(cfg, nodes, recs)
    rec = recs[(cfg["niche"], cfg["node"])]
    if int(rec["block_id"]) != cfg["layer"]:
        raise ValueError("configured node is not in the configured layer")
    modules, corr, mods, hi = node_modules(cfg, rec)
    modules.to_csv(out / "modules.csv", index=False)
    corr.to_csv(out / "corr.csv")
    hi_genes = [m for m in mods if m["module"] == hi][0]["genes"]

    row = nodes[(nodes.niche == cfg["niche"]) & (nodes.node == cfg["node"])].iloc[0]
    sel = {"dataset": key, "niche": cfg["niche"], "layer": cfg["layer"], "node": cfg["node"],
           "library": cfg["library"], "highlight_module": hi, "highlight_size": len(hi_genes),
           "jaccard_ref": [m for m in mods if m["module"] == hi][0]["jaccard_ref"],
           "reference_in_module": len(set(hi_genes) & set(cfg["reference"]["genes"])),
           "reference_size": len(cfg["reference"]["genes"]),
           "rank": int(row["rank"]), "n_nodes": len(nodes), "score": row["score"], "E": row["E"], "Q": row["Q"],
           "S": row["S"], "n_tiles_node": int(row["n_tiles"]), "block_size": int(row["block_size"]),
           "n_panel_genes": len(panel), "n_niches": len(dag_dict), **chk}
    pd.DataFrame([sel]).to_csv(out / "selection.csv", index=False)
    print(f"[{key}] niche {cfg['niche']} layer {cfg['layer']} node {cfg['node']}: rank {sel['rank']}/{len(nodes)}, "
          f"highlight module {hi} ({len(hi_genes)} genes, {sel['reference_in_module']}/{sel['reference_size']} "
          f"reference genes)", flush=True)

    ts = tile_scores(cfg, data, dag_dict[cfg["niche"]], covs, hi_genes)
    ts.to_csv(out / "tile_scores.csv", index=False)

    enr = enrich(mods, panel, key)
    enr.to_csv(out / "enrichment.csv", index=False)
    top = enr[(enr.module == hi) & (enr.library == cfg["library"])].sort_values("p")
    for bg in ("panel", "genome_enrichr"):
        t = top[top.background == bg].head(3)
        print(f"   {bg}: " + " | ".join(f"{r.term} ({r.overlap}/{r.set_size}, q={r.fdr:.2g})" for r in t.itertuples()))

    if key == "breast_cancer":
        breast_signature_tiles().to_csv(out / "signature_tile_scores.csv", index=False)
    print(f"[{key}] done in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="*", help="keys with a scripts/figure_data/bio_configs/<key>.yaml (default: all)")
    args = ap.parse_args()
    keys = args.datasets or sorted(p.stem for p in CONFIG_DIR.glob("*.yaml"))
    for key in keys:
        run(key)


if __name__ == "__main__":
    main()
