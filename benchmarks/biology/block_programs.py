"""Niche x block gene programs (Figs 6C-D, 7B, D-E; S10).

Every niche's blocks partition the panel into gene programs (data.perm_list[k] cut at data.block_dict[k]).
Per dataset and seed, from the saved index layout and the training covariances:

* niche-mean correlation: the mean of the niche's prepared (shrunk, tiers.prepare_cov) tile covariances,
  as a correlation matrix (the pooled within-tile co-variation of the niche);
* co-variation: mean |r| within each block vs between the block and the niche's other genes, with the
  within-block |r| of size-matched random gene sets as the null (empirical p per block);
* enrichment: hypergeometric test of each block against Hallmark and GO BP (GMTs in
  results/biology/genesets/), panel genes as background; terms with >= MIN_TERM panel genes, BH over terms,
  a block is enriched if a term has FDR < 0.05 and overlap >= 2. The same test on N_NULL size-matched random
  panel gene sets gives the expected enriched fraction and an empirical p per block (min p vs the null's);
* rewiring: per gene and niche pair, the Jaccard of its block partners;
* recurrence (when several seeds run): each seed-73 block's best Jaccard match among the blocks of each
  other seed's index (any niche); and the rewiring null: each seed-73 niche is matched to the other seed's
  niche sharing most tiles (tile ids of the common tiling), and every gene's partner Jaccard between the two
  is the same-niche baseline that cross-niche rewiring must fall below.

Outputs (results/block_programs/<stem>/):
  seed<s>_blocks.csv       niche, block, size, genes, within/between |r|, null p, top term per library
  seed<s>_enrichment.csv   per (block, library): every term at FDR < 0.05 and the block's top term (overlap >= 1)
  seed<s>_null.csv         per (library, size): expected enriched fraction of random sets
  seed<s>_rewiring.csv     gene, niche_a, niche_b, partner Jaccard, block sizes
  seed<s>_corr_niche<k>.npy  niche-mean correlation in the niche's gene order (perm_list[k])
  seed<s>_genes.csv          panel genes (global index order)
  recurrence.csv            (--seeds with 73 and others)
  rewiring_null.csv         gene, niche (seed 73), seed, matched niche, tile Jaccard, partner Jaccard

  sbatch slurm_jobs/biology/run_block_programs.sbatch --datasets breast_cancer skin_melanoma --seeds 73
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import experiment_common as ec
from spindle_dev import metrics, tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "block_programs"
GENESET_DIR = ec.PROJECT_ROOT / "results" / "biology" / "genesets"
LIBRARIES = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2023"]
MIN_TERM = 3
FDR = 0.05
N_NULL = 1000


def read_gmt(path, panel):
    """-> (term names, terms x genes 0/1 matrix over the panel), terms with >= MIN_TERM panel genes."""
    pos = {g.upper(): i for i, g in enumerate(panel)}
    names, rows = [], []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        idx = sorted({pos[g.upper()] for g in parts[2:] if g.upper() in pos})
        if len(idx) >= MIN_TERM:
            row = np.zeros(len(panel), dtype=np.int32)
            row[idx] = 1
            names.append(parts[0])
            rows.append(row)
    return names, np.array(rows)


def enrich_sets(sets_mask, T):
    """Hypergeometric enrichment of gene sets (rows of a 0/1 mask) against terms T.

    -> (overlap, p, fdr), each sets x terms; BH per set over all terms.
    """
    M, n = T.shape[1], T.sum(1)
    k = sets_mask @ T.T
    N = sets_mask.sum(1)[:, None]
    p = hypergeom.sf(k - 1, M, n[None, :], N)
    p = np.where(k > 0, p, 1.0)
    fdr = np.vstack([ec.bh_fdr(row) for row in p])
    return k, p, fdr


def enriched(k, fdr):
    return ((fdr < FDR) & (k >= 2)).any(1)


def niche_mean_corr(rows, covs):
    total = None
    for r in rows:
        c = tiers.prepare_cov(covs[r])
        total = c if total is None else total + c
    return metrics.spd_to_correlation(total / len(rows))


def mean_abs_offdiag(R, idx):
    sub = np.abs(R[np.ix_(idx, idx)])
    n = len(idx)
    return (sub.sum() - n) / (n * (n - 1))


def run_one(stem, seed, n_null, rng):
    out = OUT_DIR / stem
    out.mkdir(parents=True, exist_ok=True)
    data = ec.load_index(stem, seed)["data"]
    raw = ec.load_raw_covs(stem, seed)
    covs, train_idx = raw["train_tile_covs"], np.asarray(raw["train_idx"])
    panel = np.asarray(data.metadata["genes"]).astype(str)
    p = len(panel)
    labels = np.asarray(data.labels).astype(int)
    pd.DataFrame({"gene": panel}).to_csv(out / f"seed{seed}_genes.csv", index_label="gene_index")
    libs = {lib: read_gmt(GENESET_DIR / f"{lib}.gmt", panel) for lib in LIBRARIES}

    blocks, corr = {}, {}
    for k in sorted(set(labels.tolist())):
        perm = np.asarray(data.perm_list[k])
        blocks[k] = [perm[s:e] for s, e in data.block_dict[k]]
        corr[k] = niche_mean_corr(np.flatnonzero(labels == k), covs)
        np.save(out / f"seed{seed}_corr_niche{k}.npy", corr[k][np.ix_(perm, perm)].astype(np.float32))
        print(f"{stem} seed {seed} niche {k}: {np.sum(labels == k)} tiles, {len(blocks[k])} blocks", flush=True)

    sizes = sorted({len(g) for bl in blocks.values() for g in bl})
    null_sets = {s: np.array([rng.choice(p, s, replace=False) for _ in range(n_null)]) for s in sizes}

    def mask(sets):
        m = np.zeros((len(sets), p), dtype=np.int32)
        for i, g in enumerate(sets):
            m[i, g] = 1
        return m

    # null: enriched fraction and min p per size, per library
    null_rows, null_minp = [], {}
    for lib, (_, T) in libs.items():
        for s, sets in null_sets.items():
            kk, pp, ff = enrich_sets(mask(sets), T)
            null_minp[lib, s] = pp.min(1)
            null_rows.append({"library": lib, "size": s, "n_sets": len(sets),
                              "enriched_frac": float(enriched(kk, ff).mean())})
    pd.DataFrame(null_rows).to_csv(out / f"seed{seed}_null.csv", index=False)

    block_rows, enr_rows = [], []
    for k, bl in blocks.items():
        R = corr[k]
        all_genes = np.concatenate(bl)
        for b, g in enumerate(bl):
            rest = np.setdiff1d(all_genes, g)
            within = mean_abs_offdiag(R, g)
            between = float(np.abs(R[np.ix_(g, rest)]).mean())
            null_within = np.array([mean_abs_offdiag(R, s) for s in null_sets[len(g)][:200]])
            row = {"niche": k, "block": b, "size": len(g), "genes": " ".join(panel[g]),
                   "within_abs_r": within, "between_abs_r": between,
                   "null_within_abs_r": float(null_within.mean()),
                   "p_within_vs_null": float((1 + np.sum(null_within >= within)) / (1 + len(null_within)))}
            for lib, (names, T) in libs.items():
                kk, pp, ff = enrich_sets(mask([g]), T)
                kk, pp, ff = kk[0], pp[0], ff[0]
                best = int(np.argmin(pp))
                short = lib.split("_")[0]
                row[f"{short}_enriched"] = bool(enriched(kk[None], ff[None])[0])
                row[f"{short}_top_term"] = names[best]
                row[f"{short}_top_fdr"] = float(ff[best])
                row[f"{short}_top_overlap"] = int(kk[best])
                row[f"{short}_p_vs_null"] = float((1 + np.sum(null_minp[lib, len(g)] <= pp[best]))
                                                  / (1 + n_null))
                for t in np.flatnonzero((kk > 0) & ((ff < FDR) | (np.arange(len(kk)) == best))):
                    enr_rows.append({"niche": k, "block": b, "library": lib, "term": names[t],
                                     "overlap": int(kk[t]), "term_size": int(T[t].sum()), "p": float(pp[t]),
                                     "fdr": float(ff[t]),
                                     "genes": " ".join(panel[np.intersect1d(g, np.flatnonzero(T[t]))])})
            block_rows.append(row)
    bdf = pd.DataFrame(block_rows)
    bdf.to_csv(out / f"seed{seed}_blocks.csv", index=False)
    pd.DataFrame(enr_rows).sort_values(["niche", "block", "library", "p"]).to_csv(
        out / f"seed{seed}_enrichment.csv", index=False)

    # rewiring: block partners of each gene in each niche pair
    member = {k: np.empty(p, dtype=int) for k in blocks}
    for k, bl in blocks.items():
        for b, g in enumerate(bl):
            member[k][g] = b
    rw = []
    for a, b in combinations(sorted(blocks), 2):
        for gi in range(p):
            pa = set(blocks[a][member[a][gi]].tolist()) - {gi}
            pb = set(blocks[b][member[b][gi]].tolist()) - {gi}
            rw.append({"gene": panel[gi], "niche_a": a, "niche_b": b,
                       "jaccard": len(pa & pb) / max(1, len(pa | pb)),
                       "size_a": len(pa) + 1, "size_b": len(pb) + 1})
    pd.DataFrame(rw).to_csv(out / f"seed{seed}_rewiring.csv", index=False)

    # pilot summary
    for lib in LIBRARIES:
        short = lib.split("_")[0]
        exp = np.mean([next(r["enriched_frac"] for r in null_rows if r["library"] == lib and r["size"] == s)
                       for s in bdf["size"]])
        print(f"  {short}: enriched blocks {bdf[f'{short}_enriched'].mean():.2f} "
              f"(size-matched random {exp:.2f}); blocks with p_vs_null < 0.05: "
              f"{(bdf[f'{short}_p_vs_null'] < 0.05).mean():.2f}", flush=True)
    print(f"  within |r| {bdf.within_abs_r.mean():.3f} vs between {bdf.between_abs_r.mean():.3f} "
          f"vs random sets {bdf.null_within_abs_r.mean():.3f}; blocks above null (p<0.05): "
          f"{(bdf.p_within_vs_null < 0.05).mean():.2f}", flush=True)
    tiles = {k: set(train_idx[labels == k].tolist()) for k in blocks}
    return {k: [panel[g] for g in bl] for k, bl in blocks.items()}, tiles


def partners(blocks):
    """{gene: set of its block partners} for one niche's list of gene-name arrays."""
    return {g: set(bl.tolist()) - {g} for bl in blocks for g in bl.tolist()}


def recurrence(stem, ref, others, ref_tiles, other_tiles):
    null = []
    for k, bl in ref.items():
        pa = partners(bl)
        for seed, oth in others.items():
            tj = {m: len(ref_tiles[k] & t) / len(ref_tiles[k] | t) for m, t in other_tiles[seed].items()}
            m = max(tj, key=tj.get)
            pb = partners(oth[m])
            for g, s in pa.items():
                null.append({"gene": g, "niche": k, "seed": seed, "matched_niche": m, "tile_jaccard": tj[m],
                             "jaccard": len(s & pb[g]) / max(1, len(s | pb[g]))})
    pd.DataFrame(null).to_csv(OUT_DIR / stem / "rewiring_null.csv", index=False)
    rows = []
    for k, bl in ref.items():
        for b, g in enumerate(bl):
            g = set(g)
            for seed, oth in others.items():
                best = max(len(g & set(h)) / len(g | set(h)) for obl in oth.values() for h in obl)
                rows.append({"niche": k, "block": b, "size": len(g), "seed": seed, "best_jaccard": best})
    pd.DataFrame(rows).to_csv(OUT_DIR / stem / "recurrence.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=list(ec.DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[ec.PRODUCTION_SEED])
    ap.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args()
    for name in args.datasets:
        stem = ec.resolve_stem(name)
        res = {s: run_one(stem, s, args.n_null, np.random.default_rng(s)) for s in args.seeds}
        ref = ec.PRODUCTION_SEED
        if ref in res and len(res) > 1:
            recurrence(stem, res[ref][0], {s: v[0] for s, v in res.items() if s != ref},
                       res[ref][1], {s: v[1] for s, v in res.items() if s != ref})


if __name__ == "__main__":
    main()
