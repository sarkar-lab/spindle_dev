"""Part 1.1 metric check: does flooring the ridge make d_B a co-variation distance without losing biology?

Every tile covariance carries a 1e-6 ridge, so a gene with no counts in a tile has log-variance
log(1e-6) = -13.8, against about -1 for an expressed gene. Those diagonal jumps dominate d_B. This
compares four covariance variants under the same L2 block distance d_B:

* ``raw``          the stored covariance (the current metric);
* ``shrink0.05``   (1 - a) S + a mean(diag S) I with a = 0.05, applied to the whole tile covariance;
* ``shrink0.1``    the same with a = 0.1;
* ``genefloor0.1`` S + diag(f), f_g = 0.1 x gene g's mean variance over training tiles (ridge removed).

The variant is applied to queries and training tiles alike, before any block is cut. The niche/block
layout is the seed-73 production one (learned from raw covariances) for every variant, so only the
distance changes; rebuilding the layout from the chosen variant is the follow-up.

Per query (100 held-out tiles), per variant:

* composition of d_B^2, summed over all training tiles and over the query's exact top-50:
  ``share_diag_unexpr`` (diagonal terms of genes with no counts in the query or the tile),
  ``share_diag_expr`` (the other diagonal terms) and ``share_offdiag`` (co-variation);
* ``rho_var`` / ``rho_cov``: Spearman over all training tiles between d_B and a scan that keeps only
  the diagonal / only the off-diagonal terms; ``top10_varonly`` / ``top10_covonly``: overlap of those
  scans' top-10 with the d_B top-10 (the all-tile Spearman is dominated by far tiles);
* ``rho_dense``: Spearman between d_B and the whole-matrix LE distance of the same variant;
  ``rho_ncells``: Spearman between d_B and the training tiles' cell counts (small-tile pull);
* ``top10_vs_raw``: overlap of the exact top-10 with the raw metric's top-10;
* DAG (K = 32 node means, built from the variant's block logs): Overlap(5%, c) at c = 5%, 10% and 100 tiles;
* biology of the exact top-k (k = 1, 5, 10, 50): ``expr_r`` (Pearson r of z-scored pseudo-bulk) and
  ``comp_jsd`` (cell-type composition JSD, base 2; breast obs['Cluster'], otherwise 10x graphclust);
  baselines ``random`` and ``random_same_niche`` (niche of the raw top-1), 100 draws each.

Outputs (results/metric_check/):
  <stem>_query.csv   one row per (variant, query): shares, rhos, overlaps
  <stem>_bio.csv     one row per (method, query, k): expr_r, comp_jsd
  <stem>_dag.csv     one row per variant: DAG size
  summary.csv        (--aggregate) one row per (dataset, variant); Wilcoxon vs raw at k = 10

  sbatch slurm_jobs/checks/run_metric_check.sbatch <dataset> | --aggregate
"""

import argparse
import io
import tarfile
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import dag_eval_common as de
import experiment_common as ec
import holdout_core as hv  # type: ignore

OUT_DIR = ec.PROJECT_ROOT / "results" / "metric_check"
VARIANTS = {"raw": ("raw", 0.0), "shrink0.05": ("shrink", 0.05), "shrink0.1": ("shrink", 0.1),
            "genefloor0.1": ("genefloor", 0.1)}
RIDGE = 1e-6
UNEXPR_TOL = 2e-6  # stored diagonal <= ridge (+ float32 slack): the gene has no counts in the tile
KS = [1, 5, 10, 50]
TOP_SHARE = 50
N_RANDOM = 100
GRAPHCLUST = {
    "xenium_human_skin_melanoma": "xenium_hskin/output-XETG00000__slide_id__hskin/analysis.tar.gz",
    "xenium_human_kidney_nondiseased": "xenium_hkidney/output-XETG0000__slide_id__hkidney/analysis.tar.gz",
    "xenium_human_lung_cancer": "xenium_hlungcancer/output-XETG00000__slide_id__hlungcancer/analysis.tar.gz",
    "xenium_human_lymph_node_5k": "xenium_hlymphnode_5k/output-XETG00000__slide_id__hlymphnode5k/analysis.tar.gz",
    "xenium_human_pancreatic_cancer": "xenium_hpancreas/output-XETG00000__slide_id__hpancreas/analysis.tar.gz",
    "xenium_human_lymph_node": "xenium_hlymphnode/output-XETG00000__slide_id__hlymphnode/analysis.tar.gz",
    "xenium_human_brain_cancer": "xenium_hbraincancer/output-XETG00000__slide_id__hbraincancer/analysis.tar.gz",
}


def transform(cov, kind, a, gene_floor):
    if kind == "raw":
        return cov
    if kind == "shrink":
        return (1 - a) * cov + a * np.mean(np.diag(cov)) * np.eye(len(cov))
    return cov + np.diag(gene_floor)


def cell_type_codes(stem, adata):
    """Integer cell-type code per cell (-1 = unlabelled): obs['Cluster'] if present, else 10x graphclust."""
    if "Cluster" in adata.obs.columns:
        return adata.obs["Cluster"].astype("category").cat.remove_unused_categories().cat.codes.to_numpy()
    if stem not in GRAPHCLUST:
        return None
    with tarfile.open(ec.DATASET_DIR / GRAPHCLUST[stem]) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("gene_expression_graphclust/clusters.csv"))
        clusters = pd.read_csv(io.BytesIO(tf.extractfile(member).read())).set_index("Barcode")["Cluster"]
    lab = clusters.reindex(adata.obs_names)
    print(f"{int(lab.isna().sum())} of {len(lab)} cells have no graphclust label", flush=True)
    return pd.Categorical(lab).codes


def composition(tile_list, codes):
    n_types = codes.max() + 1
    C = np.stack([np.bincount(codes[t.idx][codes[t.idx] >= 0], minlength=n_types) for t in tile_list]).astype(float)
    return C / np.maximum(C.sum(1, keepdims=True), 1)


def jsd(p, q):
    return float(jensenshannon(p, q, base=2) ** 2)


def rowcorr(A, b):
    A = A - A.mean(axis=1, keepdims=True)
    b = b - b.mean()
    return (A @ b) / (np.linalg.norm(A, axis=1) * np.linalg.norm(b) + 1e-12)


def dense_vec(cov):
    """Whole-matrix log as a float32 vector whose L2 norm is ||log S||_F / sqrt(p)."""
    p = len(cov)
    iu = np.triu_indices(p)
    return (hv.log_spd(cov)[iu] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)) / np.sqrt(p)).astype(np.float32)


class Layout:
    """One niche's diagonal positions in the block vector and the gene each one belongs to."""

    def __init__(self, vec):
        self.vec = vec
        diag, gene = [], []
        for (s, _), (i, j) in zip(vec.runs, vec.iu):
            diag.append(i == j)
            gene.append(np.where(i == j, vec.perm[s + i], -1))
        self.diag = np.concatenate(diag)
        self.diag_gene = np.concatenate(gene)[self.diag]  # original gene index of each diagonal entry
        self.offsets = np.cumsum([0] + vec.dims)


def run(stem):
    seed = ec.PRODUCTION_SEED
    data = ec.load_index(stem, seed)["data"]
    covs = ec.load_raw_covs(stem, seed)
    train_idx, test_idx = np.asarray(covs["train_idx"]), np.asarray(covs["test_idx"])
    train = [np.asarray(ec.raw_cov(c), dtype=np.float64) for c in covs["train_tile_covs"]]
    test = [np.asarray(ec.raw_cov(c), dtype=np.float64) for c in covs["test_tile_covs"]]
    del covs
    n, nq, p = len(train), len(test), len(train[0])
    labels = np.asarray(data.labels).astype(int)
    layouts = {k: Layout(vec) for k, (_, vec) in de.niche_layouts(data).items()}
    niche_idx = {k: idx for k, (idx, _) in de.niche_layouts(data).items()}
    n_cells = np.array([len(t.idx) for t in data.metadata["tiles"]])

    unexpr_train = np.stack([np.diag(c) <= UNEXPR_TOL for c in train])
    unexpr_test = np.stack([np.diag(c) <= UNEXPR_TOL for c in test])
    gene_floor = np.maximum(np.mean([np.diag(c) for c in train], axis=0) - RIDGE, RIDGE)
    print(f"{stem}: {n} train / {nq} queries, p = {p}, {len(layouts)} niches; "
          f"unexpressed genes per tile: median {np.median(unexpr_train.sum(1)):.0f} of {p}", flush=True)

    # biology readouts
    adata = ec.load_adata(stem)
    tiles = ec.rebuild_tiles(adata)
    ec.check_tiles_match(tiles, data, train_idx)
    gene_pos = {g: i for i, g in enumerate(adata.var_names)}
    X = adata.X[:, [gene_pos[g] for g in data.metadata["genes"]]]
    train_tiles, query_tiles = data.metadata["tiles"], [tiles[int(i)] for i in test_idx]
    E_train, E_query = de.zscore_pair(de.pseudobulk(X, train_tiles), de.pseudobulk(X, query_tiles))
    codes = cell_type_codes(stem, adata)
    C_train = composition(train_tiles, codes) if codes is not None else None
    C_query = composition(query_tiles, codes) if codes is not None else None
    del adata, X

    def readouts(qi, sel):
        r = {"expr_r": float(rowcorr(E_train[sel].mean(0, keepdims=True), E_query[qi])[0])}
        if C_train is not None:
            r["comp_jsd"] = jsd(C_train[sel].mean(0), C_query[qi])
        return r

    q_rows, bio_rows, dag_rows = [], [], []
    raw_top10, raw_top1 = None, None
    for vname, (kind, a) in VARIANTS.items():
        t0 = time.perf_counter()
        T = [transform(c, kind, a, gene_floor) for c in train]
        Q = [transform(c, kind, a, gene_floor) for c in test]
        Xn = {k: np.stack([L.vec.vector(T[t]) for t in niche_idx[k]]) for k, L in layouts.items()}
        D_train = np.stack([dense_vec(c) for c in T])
        del T

        block_vecs = {k: [Xn[k][:, L.offsets[b]:L.offsets[b + 1]] * np.sqrt(L.vec.p) for b in range(len(L.vec.runs))]
                      for k, L in layouts.items()}
        compact = de.build_dag_from_block_logs(block_vecs, data, **de.DAG_CFG)
        del block_vecs
        adc = de.ADC(compact)
        dag_rows.append({"variant": vname, **de.dag_info(compact, data)})

        top10 = np.empty((nq, 10), dtype=int)
        top1 = np.empty(nq, dtype=int)
        for qi, qc in enumerate(Q):
            d2 = np.empty(n)
            parts = np.zeros((n, 3))  # diag unexpressed, diag expressed, off-diagonal
            pieces = {}
            for k, L in layouts.items():
                qv = L.vec.vector(qc)
                pieces[k] = [qv[L.offsets[b]:L.offsets[b + 1]].astype(np.float32) for b in range(len(L.vec.runs))]
                sq = (Xn[k] - qv) ** 2
                idx = niche_idx[k]
                d2[idx] = sq.sum(1)
                sd = sq[:, L.diag]
                un = unexpr_train[np.ix_(idx, L.diag_gene)] | unexpr_test[qi][L.diag_gene][None, :]
                parts[idx, 0] = (sd * un).sum(1)
                parts[idx, 1] = (sd * ~un).sum(1)
                parts[idx, 2] = d2[idx] - sd.sum(1)
            d = np.sqrt(np.maximum(d2, 0))
            order = np.argsort(d, kind="stable")
            top10[qi], top1[qi] = order[:10], order[0]
            dense = np.sqrt(np.maximum(((D_train - dense_vec(qc)) ** 2).sum(1), 0))
            tot, near = parts.sum(0), parts[order[:TOP_SHARE]].sum(0)
            rec = {"variant": vname, "query": qi,
                   "share_diag_unexpr": tot[0] / tot.sum(), "share_diag_expr": tot[1] / tot.sum(),
                   "share_offdiag": tot[2] / tot.sum(),
                   "share_diag_unexpr_top50": near[0] / near.sum(), "share_diag_expr_top50": near[1] / near.sum(),
                   "share_offdiag_top50": near[2] / near.sum(),
                   "rho_var": spearmanr(d, parts[:, 0] + parts[:, 1]).statistic,
                   "rho_cov": spearmanr(d, parts[:, 2]).statistic,
                   "top10_varonly": len(set(order[:10]) & set(np.argsort(parts[:, 0] + parts[:, 1])[:10])) / 10,
                   "top10_covonly": len(set(order[:10]) & set(np.argsort(parts[:, 2])[:10])) / 10,
                   "rho_dense": spearmanr(d, dense).statistic,
                   "denseNN_in_top10": float(np.argmin(dense) in set(order[:10])),
                   "rho_ncells": spearmanr(d, n_cells).statistic,
                   "top1_ncells": int(n_cells[order[0]]),
                   "top10_vs_raw": np.nan if raw_top10 is None else len(set(order[:10]) & set(raw_top10[qi])) / 10}
            ids, approx = adc.scores(pieces)
            for o in de.overlap_rows(d, ids, approx, n):
                if (o["c_kind"], o["c_spec"]) in (("frac", 0.05), ("frac", 0.10), ("abs", 100)):
                    tag = f"{100 * o['c_spec']:g}pct" if o["c_kind"] == "frac" else f"c{o['c_spec']}"
                    rec[f"overlap5@{tag}"] = o["overlap_0.05"]
            rec["near5_size"] = int(np.sum(d <= 1.05 * d[order[0]]))
            q_rows.append(rec)
            for k in KS:
                bio_rows.append({"method": vname, "query": qi, "k": k, **readouts(qi, order[:k])})
        if raw_top10 is None:
            raw_top10, raw_top1 = top10, top1
        del Xn, D_train
        print(f"  {vname:13s} {time.perf_counter() - t0:6.0f}s  "
              + pd.DataFrame([r for r in q_rows if r["variant"] == vname]).mean(numeric_only=True)
              .drop("query").round(3).to_string().replace("\n", "  "), flush=True)

    rng = np.random.default_rng(0)
    for qi in range(nq):
        for k in KS:
            for method, pool in (("random", np.arange(n)), ("random_same_niche", np.flatnonzero(labels == labels[raw_top1[qi]]))):
                draws = [readouts(qi, rng.choice(pool, min(k, len(pool)), replace=False)) for _ in range(N_RANDOM)]
                bio_rows.append({"method": method, "query": qi, "k": k,
                                 **{key: float(np.mean([r[key] for r in draws])) for key in draws[0]}})

    for df, name in ((pd.DataFrame(q_rows), "query"), (pd.DataFrame(bio_rows), "bio"), (pd.DataFrame(dag_rows), "dag")):
        df.insert(0, "dataset", stem)
        df.to_csv(OUT_DIR / f"{stem}_{name}.csv", index=False)


def aggregate():
    out = []
    for f in sorted(OUT_DIR.glob("*_query.csv")):
        stem = f.name[: -len("_query.csv")]
        q = pd.read_csv(f)
        bio = pd.read_csv(OUT_DIR / f"{stem}_bio.csv")
        dag = pd.read_csv(OUT_DIR / f"{stem}_dag.csv").set_index("variant")
        b10 = bio[bio.k == 10].pivot(index="query", columns="method")
        for v, g in q.groupby("variant", sort=False):
            rec = {"dataset": stem, "variant": v, **g.drop(columns=["dataset", "variant", "query"]).mean().to_dict(),
                   "dag_mb": dag.loc[v, "dag_mb"]}
            for col in ("expr_r", "comp_jsd"):
                if col not in b10:
                    continue
                rec[f"{col}@10"] = b10[col][v].mean()
                rec[f"{col}@10_random_same_niche"] = b10[col]["random_same_niche"].mean()
                if v != "raw":
                    x, ref = b10[col][v], b10[col]["raw"]
                    rec[f"{col}@10_better_than_raw"] = float(np.mean((x < ref) if col == "comp_jsd" else (x > ref)))
                    rec[f"{col}@10_wilcoxon_p"] = wilcoxon(x, ref).pvalue if np.any(x != ref) else 1.0
            out.append(rec)
    s = pd.DataFrame(out)
    s.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(3).T.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    run(ec.resolve_stem(args.dataset))


if __name__ == "__main__":
    main()
