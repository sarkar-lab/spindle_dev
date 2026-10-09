"""Gene-signature queries on the partial tiers (Figs 6F-G, 7C, F-G; S11).

A signature is a gene list S (gene_signatures.TISSUE_MODULES, genes missing from the panel dropped, >= 2
kept). The query is a template over S, built on the same scale as the index's prepared (shrunk) tiles and
not shrunk again:

* ``coexpression`` (primary): Sigma_S = D^1/2 [(1 - rho) I + rho 11^T] D^1/2, rho = 0.5, D_g = the 90th
  percentile of gene g's prepared variance across training tiles ("where do these genes co-vary strongly?");
* ``tissue_mean`` (control): the log-Euclidean mean of every training tile's prepared S x S sub-matrix.

Search: Spindle-Exact on S (ExactPartialTier.distances = d_B^S to every training tile); with --interval also
the interval index (IntervalTier, same prepared query). Per (seed, construction, signature, K):

* tile score: scanpy score_genes per cell (random_state = seed), averaged over the tile's cells; one-sided
  Mann-Whitney of the top-K vs the other training tiles, Cliff's delta, BH over signatures;
* co-variation r_S: the mean off-diagonal correlation of the tile's prepared S x S covariance;
* expression control: the K tiles with the highest tile score (method ``expression``) scored the same way,
  and the Jaccard of its top-K with Spindle-Exact's;
* target-cell fraction (breast obs['Cluster'] only; the signature's target_clusters);
* interval: Jaccard of its top-K with the exact top-K.

Outputs (results/signature_queries/<stem>/):
  stats.csv          one row per (seed, construction, signature, K, method)
  top_matches.csv    top-50 tiles per (seed, construction, signature, method): rank, row, tile id, bbox,
                     d_B^S, tile score, r_S
  cells_seed<s>.csv  (--maps, seed 73) per cell: x, y, cell type code, one score column per signature

  sbatch slurm_jobs/biology/run_signature_queries.sbatch --datasets breast_cancer [--interval] [--maps]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import experiment_common as ec
from gene_signatures import TISSUE_MODULES  # type: ignore
from metric_check import cell_type_codes  # type: ignore
from spindle_dev import interval_dp as ivd
from spindle_dev import tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "signature_queries"
TISSUE = {"skin_melanoma": "skin", "kidney_nondiseased": "kidney", "breast_cancer": "breast",
          "lung_cancer": "lung", "pancreatic_cancer": "pancrea", "lymph_node": "lymph_node",
          "lymph_node_5k": "lymph_node", "brain_cancer": "brain"}
KS = [5, 10, 50]
N_MATCHES = 50
RHO = 0.5
VAR_QUANTILE = 0.9


def templates(subs):
    """Both query constructions from the training tiles' prepared S x S sub-matrices (n x s x s)."""
    s = subs.shape[1]
    D = np.quantile(subs[:, np.arange(s), np.arange(s)], VAR_QUANTILE, axis=0)
    R = (1 - RHO) * np.eye(s) + RHO * np.ones((s, s))
    logs = ivd.batched_log_spd(subs, tiers.LOG_FLOOR)
    w, V = np.linalg.eigh(logs.mean(0))
    return {"coexpression": np.sqrt(D)[:, None] * R * np.sqrt(D)[None, :],
            "tissue_mean": (V * np.exp(w)) @ V.T}


def mean_offdiag_r(subs):
    d = np.sqrt(np.einsum("nii->ni", subs))
    r = subs / (d[:, :, None] * d[:, None, :])
    s = subs.shape[1]
    return (r.sum((1, 2)) - s) / (s * (s - 1))


def score_rows(sel, tile_score, r_s):
    mask = np.zeros(tile_score.size, bool)
    mask[sel] = True
    x, y = tile_score[mask], tile_score[~mask]
    return {"mean_score": x.mean(), "mean_score_background": y.mean(),
            "mwu_p": mannwhitneyu(x, y, alternative="greater").pvalue, "cliffs_delta": ec.cliffs_delta(x, y),
            "mean_r_s": r_s[mask].mean(), "mean_r_s_background": r_s[~mask].mean()}


def run_seed(key, stem, seed, adata, codes, tiles_all, use_interval, maps):
    data = ec.load_index(stem, seed)["data"]
    raw = ec.load_raw_covs(stem, seed)
    train_idx = np.asarray(raw["train_idx"])
    ec.check_tiles_match(tiles_all, data, train_idx)
    train_covs = raw["train_tile_covs"]
    ep = tiers.build_tier("exact_partial", data, train_covs)
    ivl = tiers.build_tier("interval", data, exact_partial=ep) if use_interval else None
    genes = list(np.asarray(data.metadata["genes"]).astype(str))
    train_tiles = data.metadata["tiles"]
    cells_train = np.concatenate([t.idx for t in train_tiles])
    cluster = adata.obs["Cluster"].astype(str).to_numpy() if "Cluster" in adata.obs else None

    def subs_of(S):
        """Prepared S x S sub-matrices of every training tile (the full covariance, cross-block terms included)."""
        return np.stack([tiers.prepare_cov(c)[np.ix_(S, S)] for c in train_covs])

    stat_rows, match_rows, cell_cols = [], [], {}
    for sig, info in TISSUE_MODULES[TISSUE[key]].items():
        S = np.array([genes.index(g) for g in info["genes"] if g in genes])
        missing = [g for g in info["genes"] if g not in genes]
        if len(S) < 2:
            print(f"  {sig}: {len(S)} panel genes, skipped", flush=True)
            continue
        sc.tl.score_genes(adata, gene_list=[genes[i] for i in S], score_name="sig", use_raw=False,
                          random_state=seed)
        cell_score = adata.obs["sig"].to_numpy()
        if maps:
            cell_cols[sig] = cell_score.astype(np.float32)
        tile_score = np.array([cell_score[t.idx].mean() for t in train_tiles])
        subs = subs_of(S)
        r_s = mean_offdiag_r(subs)
        target = np.isin(cluster, info["target_clusters"]) if cluster is not None and info["target_clusters"] else None
        expr_order = np.argsort(-tile_score, kind="stable")
        for construction, q in templates(subs).items():
            d = ep.distances(q, S)
            exact_order = np.argsort(d, kind="stable")
            orders = {"spindle_exact": exact_order, "expression": expr_order}
            if ivl is not None:
                rows, _ = ivl._scan(q, S, N_MATCHES)  # q is already prepared: no second shrink
                orders["interval"] = np.asarray(rows)
            for K in KS:
                for method, order in orders.items():
                    sel = order[:K]
                    row = {"seed": seed, "construction": construction, "signature": sig, "K": K, "method": method,
                           "n_genes": len(S), "genes_missing": ",".join(missing),
                           **score_rows(sel, tile_score, r_s),
                           "jaccard_vs_exact": len(set(sel) & set(exact_order[:K])) / len(set(sel) | set(exact_order[:K])),
                           "mean_dist": d[sel].mean()}
                    if target is not None:
                        cells_sel = np.concatenate([train_tiles[c].idx for c in sel])
                        row["target_frac"] = target[cells_sel].mean()
                        row["target_frac_background"] = target[cells_train].mean()
                    stat_rows.append(row)
            for method, order in orders.items():
                for rank, c in enumerate(order[:N_MATCHES], start=1):
                    x0, y0, x1, y1 = train_tiles[c].bbox
                    match_rows.append({"seed": seed, "construction": construction, "signature": sig,
                                       "method": method, "rank": rank, "row": int(c),
                                       "tile_id": int(train_idx[c]), "dist": d[c], "tile_score": tile_score[c],
                                       "r_s": r_s[c], "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        v = [r for r in stat_rows if r["signature"] == sig and r["K"] == 10 and r["construction"] == "coexpression"]
        print(f"  {sig} ({len(S)} genes): " + "; ".join(
            f"{r['method']} delta {r['cliffs_delta']:.2f} r_S {r['mean_r_s']:.3f}" for r in v)
            + f" (background r_S {v[0]['mean_r_s_background']:.3f})", flush=True)
    if maps:
        xy = adata.obsm["spatial"]
        cells = pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1], "cell_type": codes, **cell_cols})
        cells.to_csv(OUT_DIR / stem / f"cells_seed{seed}.csv", index=False, float_format="%.4g")
    return stat_rows, match_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(TISSUE))
    ap.add_argument("--seeds", nargs="+", type=int, default=[ec.PRODUCTION_SEED] + ec.MULTISEED)
    ap.add_argument("--interval", action="store_true", help="also search the interval index")
    ap.add_argument("--maps", action="store_true", help="write per-cell scores for the seed-73 maps")
    args = ap.parse_args()
    for key in args.datasets:
        stem = ec.resolve_stem(key)
        (OUT_DIR / stem).mkdir(parents=True, exist_ok=True)
        adata = ec.load_adata(stem)
        codes = cell_type_codes(stem, adata)
        tiles_all = ec.rebuild_tiles(adata)
        stats, matches = [], []
        for seed in args.seeds:
            print(f"=== {stem} seed {seed} ===", flush=True)
            s, m = run_seed(key, stem, seed, adata, codes, tiles_all, args.interval,
                            args.maps and seed == ec.PRODUCTION_SEED)
            stats += s
            matches += m
        df = pd.DataFrame(stats)
        df["mwu_fdr_bh"] = df.groupby(["seed", "construction", "K", "method"])["mwu_p"].transform(ec.bh_fdr)
        df.to_csv(OUT_DIR / stem / "stats.csv", index=False)
        pd.DataFrame(matches).to_csv(OUT_DIR / stem / "top_matches.csv", index=False)


if __name__ == "__main__":
    main()
