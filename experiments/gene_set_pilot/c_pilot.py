"""Session 1 pilot: which constructor turns a gene list into a gene-set query (paper/RESULTS_PLAN.md §4 Step 1).

Usage (sbatch array, one task per indexed dataset, datasets.yaml order):
  sbatch experiments/gene_set_pilot/run.sbatch
Writes results/gene_set_pilot/<key>/{lists.csv, hits.csv, comp.csv, meta.json}; plots: f_pilot.py (login node).

Setup: seed 73, 100 held-out tiles; the index = the other tiles (build_layout + SpindlePartial, exact d_B^S, top K = 10).
Gene lists: every list in experiments/common/signatures.py that applies to the dataset (TISSUE_MODULES of its tissue,
GENERIC_MODULES, PAN_CANCER_MODULES), restricted to the index genes, duplicates removed; a list with < 3 index genes is
skipped (reported in meta.json). A list identical to an earlier one after restriction is run once.

Constructors (Omega_q = diagonal of the q-th percentile, over index tiles, of each gene's shrunk variance; nothing is
shrunk again):
  T         Methods Eq. (signature): Omega_0.9^1/2 [(1 - rho) I + rho 11'] Omega_0.9^1/2, rho = 0.5.
  T75, T50  the same with q = 0.75, 0.5 (only to inform the option "T with another q").
  E         exemplar: the index tile with the largest robust co-variation c on S (rule of poster/scripts/c_bio_xtumour.py:
            c = mean off-diagonal correlation of S over the tile's cells, raw counts, zero-variance genes dropped;
            robust: >= 80 cells, S counts per cell >= 0.5, no cell holds > 15 % of the S counts, dropping the top 3
            cells keeps >= 70 % of c); its shrunk covariance cut to S. No robust tile -> no E for that list.
  D         diagonal-only control: Omega_0.9 (template variances, no correlation).
  X         expression control (recorded, never a candidate): the 10 tiles with the highest tile score (R2).
When E exists, the exemplar tile is removed from every constructor's ranking, so all are compared on the same tiles.

Readouts per (list, constructor), on the top 10 (declared before the run):
  R1 co-variation: median c (as above, no robust filter) of the top 10 minus the median c over all index tiles
     ("lift_c"); the same with r_S = mean off-diagonal correlation of the tile's shrunk S x S covariance ("lift_rs").
  R2 tile score = mean over S of the z-scored (across index tiles) tile-mean log-normalised expression; Cliff's delta
     of the top 10 vs the other index tiles.
  R3 composition (cells): TV between the pooled cell-type shares of the top 10's cells and the index tiles' cells,
     minus the mean of the same TV for 200 random 10-tile draws ("spec"; > 0 = a more specific cell make-up than
     random). Cell types: breast annotation ('Cluster'); otherwise Leiden clusters of the cells as
     presentation/scripts/s0_layouts.py (normalise_total, log1p, 30 PCs, 15-NN, resolution 0.5, seed 0).
     Breast lists with target clusters: also the target cells' share in the top 10 vs the index tiles.
  R4 variance-driven: Jaccard of the top 10 with D's top 10 (high = ranks by variances, not co-variation: the
     "generic stroma" failure); also Jaccard with X and between T and E.
  var_level: mean over S and the top 10 of each gene's variance percentile among index tiles.
How to read it (the user decides; no threshold here): a constructor is useful if its R1 lift is positive in most lists
and larger than D's, its R2 is not worse than D's and its R4 Jaccard with D is low. T and E are compared on these;
T75 / T50 show whether a lower q helps T.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT), str(ROOT / "experiments" / "common")]
import paths  # noqa: E402
import spindle_dev as sd  # noqa: E402
from signatures import GENERIC_MODULES, PAN_CANCER_MODULES, TISSUE, TISSUE_MODULES  # noqa: E402
from spindle_dev.metrics import as_cov  # noqa: E402

OUT = ROOT / "results" / "gene_set_pilot"
SEED, N_HOLDOUT, TOP, N_RANDOM, MIN_GENES = 73, 100, 10, 200, 3
RHO = 0.5
QS = {"T": 0.9, "T75": 0.75, "T50": 0.5}


def mean_offdiag_corr(X):
    Xc = X - X.mean(0)
    s = np.sqrt((Xc ** 2).sum(0))
    keep = s > 0
    if keep.sum() < 2:
        return np.nan
    Z = Xc[:, keep] / s[keep]
    m = Z.shape[1]
    return ((Z.T @ Z).sum() - m) / (m * (m - 1))


def robust_c(X):
    c = mean_offdiag_corr(X)
    tot = X.sum(1)
    if len(X) < 80 or tot.sum() / len(X) < 0.5 or tot.max() > 0.15 * tot.sum() or not np.isfinite(c) or c <= 0:
        return np.nan
    c3 = mean_offdiag_corr(X[np.argsort(tot)[:-3]])
    return c if np.isfinite(c3) and c3 >= 0.7 * c else np.nan


def offdiag_r(subs):
    d = np.sqrt(np.einsum("nii->ni", subs))
    r = subs / (d[:, :, None] * d[:, None, :])
    s = subs.shape[1]
    return (r.sum((1, 2)) - s) / (s * (s - 1))


def cliffs_delta(x, y):
    r = rankdata(np.concatenate([x, y]))
    u = r[:len(x)].sum() - len(x) * (len(x) + 1) / 2
    return 2 * u / (len(x) * len(y)) - 1


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / len(a | b)


def cell_labels(adata):
    if "Cluster" in adata.obs.columns:
        return np.asarray(adata.obs["Cluster"].astype(str)), "annotation"
    import scanpy as sc

    a = adata.copy()
    sc.pp.normalize_total(a)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=30, random_state=0)
    sc.pp.neighbors(a, n_neighbors=15, random_state=0)
    sc.tl.leiden(a, resolution=0.5, random_state=0, flavor="igraph", n_iterations=2, directed=False)
    sc.tl.rank_genes_groups(a, "leiden", method="t-test", n_genes=2)
    names = a.uns["rank_genes_groups"]["names"]
    lab = {g: f"{g}:{'/'.join(names[i][g] for i in range(2))}" for g in a.obs["leiden"].cat.categories}
    return np.asarray([lab[g] for g in a.obs["leiden"]]), "leiden"


def gene_lists(key, pos):
    """(family, name, S, missing, target clusters) per list; skipped and aliased lists separately."""
    fams = [("tissue", TISSUE_MODULES[TISSUE[key]]), ("generic", GENERIC_MODULES), ("pan_cancer", PAN_CANCER_MODULES)]
    out, skipped, alias, seen = [], [], {}, {}
    for fam, mods in fams:
        for name, info in mods.items():
            S, missing = [], []
            for g in info["genes"]:
                if g not in pos:
                    missing.append(g)
                elif pos[g] not in S:
                    S.append(pos[g])
            if len(S) < MIN_GENES:
                skipped.append(dict(family=fam, list=name, n_index_genes=len(S), missing=missing))
                continue
            k = tuple(sorted(S))
            if k in seen:
                alias[name] = seen[k]
                continue
            seen[k] = name
            out.append((fam, name, np.array(S), missing, info.get("target_clusters", [])))
    return out, skipped, alias


def main(key):
    t0 = time.time()
    OUT_K = OUT / key
    OUT_K.mkdir(parents=True, exist_ok=True)
    tag = subprocess.run(["git", "-C", str(ROOT), "describe", "--tags", "--always", "--dirty"],
                         capture_output=True, text=True).stdout.strip()
    adata = sd.load_adata(paths.dataset_path(key))
    tiles, covs, genes = sd.prepare_tiles(adata, max_genes=paths.dataset_max_genes(key))
    names = [str(g) for g in genes]
    train, _ = sd.split_tiles(len(tiles), seed=SEED, n_holdout=N_HOLDOUT)
    tr_covs = [covs[i] for i in train]
    data = sd.build_layout([tiles[i] for i in train], tr_covs, genes, adata.n_obs)
    part = sd.SpindlePartial(data, tr_covs)
    n = len(train)
    print(f"{key}: {n} index tiles, {len(set(np.asarray(data.labels).tolist()))} niches, p = {len(names)}; "
          f"{time.time() - t0:.0f}s", flush=True)

    cells = [tiles[i].idx for i in train]
    var = np.stack([np.diag(as_cov(c)) for c in tr_covs])
    var_pct = rankdata(var, axis=0) / n
    gi = adata.var_names.get_indexer(names)
    tot = np.asarray(adata.X.sum(1)).ravel()
    X = adata.X[:, gi]
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    L = np.log1p(X / np.maximum(tot, 1)[:, None] * np.median(tot))
    mu = np.stack([L[c].mean(0) for c in cells])
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    del L

    labels, source = cell_labels(adata)
    classes, cls = np.unique(labels, return_inverse=True)
    counts = np.stack([np.bincount(cls[c], minlength=len(classes)) for c in cells]).astype(float)
    section = counts.sum(0) / counts.sum()
    rng = np.random.default_rng(SEED)

    def pooled(rows):
        c = counts[rows].sum(0)
        return c / c.sum()

    tv_random = float(np.mean([tv(pooled(rng.choice(n, TOP, replace=False)), section) for _ in range(N_RANDOM)]))
    print(f"cell labels from {source}: {len(classes)} classes; TV random {tv_random:.3f}; {time.time() - t0:.0f}s",
          flush=True)

    pos = {g: i for i, g in enumerate(names)}
    lists, skipped, alias = gene_lists(key, pos)
    R = lambda s: (1 - RHO) * np.eye(s) + RHO * np.ones((s, s))  # noqa: E731
    rows, hits, comps = [], [], []
    for fam, name, S, missing, targets in lists:
        s = len(S)
        c_all = np.array([mean_offdiag_corr(X[c][:, S]) for c in cells])
        c_rob = np.array([robust_c(X[c][:, S]) for c in cells])
        ex = int(np.nanargmax(c_rob)) if np.isfinite(c_rob).any() else None
        subs = np.stack([as_cov(c)[np.ix_(S, S)] for c in tr_covs])
        r_all = offdiag_r(subs)
        score = z[:, S].mean(1)
        V = var[:, S]
        queries = {}
        for qn, q in QS.items():
            sd_ = np.sqrt(np.quantile(V, q, axis=0))
            queries[qn] = sd_[:, None] * R(s) * sd_[None, :]
        queries["D"] = np.diag(np.quantile(V, QS["T"], axis=0))
        if ex is not None:
            queries["E"] = subs[ex]
        valid = np.ones(n, bool)
        if ex is not None:
            valid[ex] = False
        tops = {}
        for qn, Q in queries.items():
            np.linalg.cholesky(Q)  # SPD
            d = part.distances(Q, S)
            d[~valid] = np.inf
            order = np.argsort(d, kind="stable")[:TOP]
            tops[qn] = order
            for rank, r in enumerate(order, 1):
                hits.append(dict(list=name, constructor=qn, rank=rank, row=int(r), tile=int(train[r]), dist=float(d[r])))
        sc_ = np.where(valid, score, -np.inf)
        tops["X"] = np.argsort(-sc_, kind="stable")[:TOP]
        for rank, r in enumerate(tops["X"], 1):
            hits.append(dict(list=name, constructor="X", rank=rank, row=int(r), tile=int(train[r]), dist=np.nan))
        tmask = np.isin(labels, targets) if targets and source == "annotation" else None
        cells_idx = np.concatenate(cells)
        for qn, top in tops.items():
            other = valid.copy()
            other[top] = False
            pc = pooled(top)
            row = dict(dataset=key, family=fam, list=name, n_genes=s, n_missing=len(missing),
                       missing=",".join(missing), constructor=qn,
                       exemplar_row=ex, exemplar_tile=None if ex is None else int(train[ex]),
                       exemplar_c=None if ex is None else float(c_rob[ex]),
                       exemplar_rs=None if ex is None else float(r_all[ex]),
                       n_robust=int(np.isfinite(c_rob).sum()),
                       c_top=float(np.nanmedian(c_all[top])), c_all=float(np.nanmedian(c_all[valid])),
                       rs_top=float(np.median(r_all[top])), rs_all=float(np.median(r_all[valid])),
                       delta=float(cliffs_delta(score[top], score[other])),
                       tv_hits=float(tv(pc, section)), tv_random=tv_random,
                       var_level=float(var_pct[np.ix_(top, S)].mean()),
                       **{f"jacc_{o}": jaccard(top, tops[o]) for o in ("D", "X", "T", "E") if o in tops})
            row["lift_c"] = row["c_top"] - row["c_all"]
            row["lift_rs"] = row["rs_top"] - row["rs_all"]
            row["spec"] = row["tv_hits"] - tv_random
            if tmask is not None:
                row["target_top"] = float(tmask[np.concatenate([cells[r] for r in top])].mean())
                row["target_all"] = float(tmask[cells_idx].mean())
            rows.append(row)
            for j in np.flatnonzero(pc):
                comps.append(dict(list=name, constructor=qn, label=classes[j], share=float(pc[j]),
                                  section=float(section[j])))
        print(f"  {name} ({s} genes, exemplar {ex}): " + ", ".join(
            f"{r['constructor']} lift {r['lift_c']:+.3f} d {r['delta']:+.2f} J_D {r.get('jacc_D', np.nan):.2f}"
            for r in rows[-len(tops):]), flush=True)

    pd.DataFrame(rows).to_csv(OUT_K / "lists.csv", index=False)
    pd.DataFrame(hits).to_csv(OUT_K / "hits.csv", index=False)
    pd.DataFrame(comps).to_csv(OUT_K / "comp.csv", index=False)
    meta = dict(dataset=key, tag=tag, seed=SEED, n_index=n, p=len(names), label_source=source,
                n_classes=len(classes), tv_random=tv_random, skipped=skipped, alias=alias,
                seconds=round(time.time() - t0))
    with open(OUT_K / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=1, default=int)
    print(json.dumps(meta, default=int), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
