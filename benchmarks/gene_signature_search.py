"""E9-lite: gene-signature queries on breast, searching every niche.

Replaces benchmarks/gene_signature_search.py for the preprint. That script
rebuilt its own index (different tiling/genes from every other benchmark),
built each query from the single best-overlapping (niche, block) -- i.e. from
one niche's own training matrices -- hard-coded np.random.seed(42), and
appended two disagreeing runs into one CSV.

Here the existing breast index builds (seeds 0-4, with their dyadic interval
indices) are reused, and the query over the signature genes S is built
niche-agnostically in two ways:

* ``coexpression`` (primary): a template covariance asking "where do these
  genes co-vary strongly?" -- Sigma_S = D^1/2 [(1-rho) I + rho 11^T] D^1/2 with
  rho = 0.5 and D_g = the 90th percentile of gene g's variance across training
  tiles (the variance of g where it is expressed).
* ``tissue_mean`` (control, as first planned): the log-Euclidean mean of every
  training tile's S x S sub-matrix -- the "typical tissue" for these genes.

Each query goes through the partial-panel search of every niche
(partial_panel_core.search_all_clusters_spindle on the interval index, top_c=400
candidates), then an exact re-rank on the S x S sub-matrix; the exact
brute-force top-K over all training tiles is computed for comparison.

Statistics per (seed, construction, signature, K in {5, 10, 50}):
  * per-cell scanpy score_genes (use_raw=False, random_state=seed), averaged
    over each training tile's cells;
  * one-sided Mann-Whitney U of retrieved-tile scores vs all other training
    tiles, Cliff's delta, BH-FDR across the 5 signatures;
  * fraction of cells in retrieved tiles that belong to the signature's target
    cell types (obs['Cluster']) vs the fraction among all training cells;
  * Jaccard overlap of Spindle's top-K with the exact top-K.

Outputs (results/gene_signature_search/breast/; never appended):
  signature_stats.csv            one row per (seed, construction, signature, K, method)
  <signature>_top_matches.csv    top-50 tiles per (seed, construction) with bbox and score
  <signature>_spatial_cells.csv  per-cell score (seed 0) for maps

  sbatch slurm_jobs/run_gene_signature_search.sbatch
"""

import argparse
import time

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu

import experiment_common as ec
import partial_panel_core  # type: ignore
from gene_signatures import TISSUE_MODULES  # type: ignore
from spindle_dev.utils import exp_spd, log_spd

STEM = ec.DATASETS["breast_cancer"]
OUT_DIR = ec.PROJECT_ROOT / "results" / "gene_signature_search" / "breast"
SIGNATURES = TISSUE_MODULES["breast"]
KS = [5, 10, 50]
TOP_C = 400
RHO = 0.5
VAR_QUANTILE = 0.9


def sub_log(C, S):
    return log_spd(np.asarray(C, dtype=np.float64)[np.ix_(S, S)])


def build_queries(train_covs, S, p):
    """Both query constructions, embedded in a p x p identity (only S x S is read)."""
    subs = np.stack([np.asarray(C, dtype=np.float64)[np.ix_(S, S)] for C in train_covs])
    D = np.quantile(subs[:, np.arange(len(S)), np.arange(len(S))], VAR_QUANTILE, axis=0)
    R = (1 - RHO) * np.eye(len(S)) + RHO * np.ones((len(S), len(S)))
    templates = {
        "coexpression": np.sqrt(D)[:, None] * R * np.sqrt(D)[None, :],
        "tissue_mean": exp_spd(np.mean([log_spd(m) for m in subs], axis=0)),
    }
    out = {}
    for name, sub in templates.items():
        q = np.eye(p)
        q[np.ix_(S, S)] = 0.5 * (sub + sub.T)
        out[name] = q
    return out


def test_scores(sel, tile_scores):
    mask = np.zeros(tile_scores.size, bool)
    mask[sel] = True
    x, y = tile_scores[mask], tile_scores[~mask]
    return {
        "mean_score_retrieved": x.mean(), "mean_score_background": y.mean(),
        "mwu_p": mannwhitneyu(x, y, alternative="greater").pvalue,
        "cliffs_delta": ec.cliffs_delta(x, y),
    }


def run_seed(seed, adata, cell_types, tiles_all):
    bundle = ec.load_index(STEM, seed)
    data = bundle["data"]
    covs = ec.load_raw_covs(STEM, seed)
    train_idx = np.asarray(covs["train_idx"])
    ec.check_tiles_match(tiles_all, data, train_idx)
    train_covs = [ec.raw_cov(c) for c in covs["train_tile_covs"]]
    del covs
    ivl_idx = ec.load_interval_index(STEM, seed)
    genes = list(data.metadata["genes"])
    p = len(genes)
    sid_to_loc = {int(s): k for k, s in enumerate(data.spd_ids)}
    train_tiles = data.metadata["tiles"]
    cells_train = np.concatenate([t.idx for t in train_tiles])
    cluster = adata.obs["Cluster"].astype(str).to_numpy()

    stat_rows, match_rows, cell_frames = [], {}, {}
    for sig_name, info in SIGNATURES.items():
        S = [genes.index(g) for g in info["genes"] if g in genes]
        missing = [g for g in info["genes"] if g not in genes]
        sc.tl.score_genes(adata, gene_list=[genes[i] for i in S], score_name="sig", use_raw=False,
                          random_state=seed)
        cell_score = adata.obs["sig"].to_numpy()
        tile_scores = np.array([cell_score[t.idx].mean() for t in train_tiles])
        target = np.isin(cluster, info["target_clusters"])
        target_bg = target[cells_train].mean()
        if seed == 0:
            xy = adata.obsm["spatial"]
            cell_frames[sig_name] = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "score": cell_score,
                                                  "cell_type": cluster})

        # Exact brute force on the S x S sub-matrix over all training tiles.
        for construction, q in build_queries(train_covs, S, p).items():
            q_log = sub_log(q, S)
            scale = np.sqrt(len(S))
            t0 = time.perf_counter()
            exact = np.array([np.linalg.norm(sub_log(C, S) - q_log) / scale for C in train_covs])
            bf_ms = (time.perf_counter() - t0) * 1000
            exact_order = np.argsort(exact, kind="stable")

            t0 = time.perf_counter()
            hits = partial_panel_core.search_all_clusters_spindle(ivl_idx, data, S, q, top_k=TOP_C)
            cand = [sid_to_loc[int(t)] for _, ids in hits for t in ids if int(t) in sid_to_loc]
            rer = sorted(((np.linalg.norm(sub_log(train_covs[c], S) - q_log) / scale, c) for c in cand))
            spindle_ms = (time.perf_counter() - t0) * 1000
            spindle_order = [c for _, c in rer]

            for K in KS:
                for method, order in (("spindle", spindle_order[:K]), ("exact", list(exact_order[:K]))):
                    cells_sel = np.concatenate([train_tiles[c].idx for c in order])
                    stat_rows.append({
                        "seed": seed, "construction": construction, "signature": sig_name, "K": K, "method": method,
                        "n_genes": len(S), "genes_missing": ",".join(missing), "n_retrieved": len(order),
                        **test_scores(order, tile_scores),
                        "target_cell_fraction_retrieved": target[cells_sel].mean(),
                        "target_cell_fraction_background": target_bg,
                        "jaccard_vs_exact": len(set(spindle_order[:K]) & set(exact_order[:K])) / len(
                            set(spindle_order[:K]) | set(exact_order[:K])),
                        "spindle_time_ms": spindle_ms, "bf_time_ms": bf_ms, "n_candidates": len(cand),
                    })
            for rank, c in enumerate(spindle_order[:max(KS)], start=1):
                x0, y0, x1, y1 = train_tiles[c].bbox
                match_rows.setdefault(sig_name, []).append({
                    "seed": seed, "construction": construction, "rank": rank, "tile_id": int(data.spd_ids[c]),
                    "dist": rer[rank - 1][0], "exact_rank": int(np.where(exact_order == c)[0][0]) + 1,
                    "x0": x0, "y0": y0, "x1": x1, "y1": y1, "tile_score": tile_scores[c],
                })
    return stat_rows, match_rows, cell_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=ec.MULTISEED)
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ec.load_adata(STEM)
    tiles_all = ec.rebuild_tiles(adata)
    cell_types = sorted(adata.obs["Cluster"].astype(str).unique())
    for sig, info in SIGNATURES.items():
        unknown = set(info["target_clusters"]) - set(cell_types)
        if unknown:
            raise ValueError(f"{sig}: target cell types not in obs['Cluster']: {sorted(unknown)}")

    stats, matches = [], {}
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        s_rows, m_rows, cells = run_seed(seed, adata, cell_types, tiles_all)
        stats.extend(s_rows)
        for k, v in m_rows.items():
            matches.setdefault(k, []).extend(v)
        for sig, df in cells.items():
            df.to_csv(OUT_DIR / f"{sig}_spatial_cells.csv", index=False)

    df = pd.DataFrame(stats)
    df["mwu_fdr_bh"] = df.groupby(["seed", "construction", "K", "method"])["mwu_p"].transform(ec.bh_fdr)
    df.to_csv(OUT_DIR / "signature_stats.csv", index=False)
    for sig, rows in matches.items():
        pd.DataFrame(rows).to_csv(OUT_DIR / f"{sig}_top_matches.csv", index=False)

    view = df[(df["method"] == "spindle") & (df["K"] == 10)]
    agg = view.groupby(["construction", "signature"]).agg(
        delta=("cliffs_delta", "mean"), delta_sd=("cliffs_delta", "std"),
        fdr_max=("mwu_fdr_bh", "max"), target_frac=("target_cell_fraction_retrieved", "mean"),
        target_bg=("target_cell_fraction_background", "mean"), jaccard=("jaccard_vs_exact", "mean"))
    print("\nSpindle top-10, mean over seeds:\n" + agg.to_string())


if __name__ == "__main__":
    main()
