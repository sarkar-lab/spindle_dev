"""Fig 6A-D inputs: niches, block programmes, co-variation within blocks and the rewiring of ACTA2 (one build per task).

Usage (sbatch array, task = seed 0-4): sbatch --array=0-4 experiments/common/run.sbatch experiments/fig6_breast/c_programs.py
       the whole section (Fig 6A-C; user, 27 Oct): sbatch experiments/common/run.sbatch experiments/fig6_breast/c_programs.py full
Writes results/fig6_breast/seed<s>/ (or full/) {blocks.csv, enrich_null.csv, niche_types.csv, acta2.csv}; full and seed 0 also
corr_niche<k>.npy (niche-mean correlation in niche k's gene order), layout.json (gene order and blocks per niche) and
cells_niche.npz (every cell: x, y, niche of its tile; -1 = held-out or dropped tile).

Per niche k and block b of the build (layout.perm_list[k] cut at layout.block_dict[k]), on the niche-mean correlation
R_k (the mean of the niche's shrunk index covariances, as a correlation):
- within = mean |r| over the block's gene pairs; between = mean |r| between the block's genes and the niche's other
  genes; null = within of N_NULL random panel gene sets of the same size on R_k; null_p = (1 + #null >= within) /
  (1 + N_NULL); a block co-varies if null_p < 0.05.
- hubs: the 3 genes with the largest summed |r| to the rest of the block.
- enrichment: hypergeometric test against MSigDB Hallmark 2020 and GO Biological Process 2023 (Enrichr GMTs), the
  panel genes as background, terms with >= MIN_TERM panel genes, GO terms containing "regulation of" removed before
  testing (generic terms; user rule 8 Oct); BH per block and library over the terms. A block is enriched if a term has
  FDR < 0.05 and >= 2 block genes; its key term is the best Hallmark term, else the best GO term (lowest FDR, then
  largest overlap). The same test on the N_NULL random sets of each block size gives the expected enriched fraction.
- niche cell types: the 19 annotated types of the cells of the niche's index tiles; section = all cells.
- ACTA2 (chosen before the run as a marker that myoepithelial cells and fibroblasts share; no gene search): r with
  every gene in every niche's R_k and whether that gene is in ACTA2's block there.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common6 as c6  # noqa: E402

N_NULL = 1000
MIN_TERM = 3
FDR = 0.05
MARKER = "ACTA2"


def read_gmt(path, panel, drop_regulation=False):
    """(term names, terms x panel 0/1 matrix), terms with >= MIN_TERM panel genes."""
    pos = {g.upper(): i for i, g in enumerate(panel)}
    names, rows = [], []
    for line in open(path):
        parts = line.rstrip("\n").split("\t")
        if drop_regulation and "regulation of" in parts[0].lower():
            continue
        idx = sorted({pos[g.upper()] for g in parts[2:] if g.upper() in pos})
        if len(idx) >= MIN_TERM:
            row = np.zeros(len(panel), dtype=np.int32)
            row[idx] = 1
            names.append(parts[0])
            rows.append(row)
    return names, np.array(rows)


def enrich(masks, T):
    """(overlap, fdr) of gene sets (rows of a 0/1 mask) against terms T, BH per set over the terms."""
    M, n = T.shape[1], T.sum(1)
    k = masks @ T.T
    N = masks.sum(1)[:, None]
    p = np.where(k > 0, hypergeom.sf(k - 1, M, n[None, :], N), 1.0)
    return k, np.vstack([c6.bh(row) for row in p])


def enriched_any(masks, libs):
    hit = np.zeros(len(masks), bool)
    for names, T in libs.values():
        k, q = enrich(masks, T)
        hit |= ((q < FDR) & (k >= 2)).any(1)
    return hit


def mean_abs_within(R, sets):
    """Mean |r| over the pairs of each gene set (rows of an index array)."""
    A = np.abs(R)
    s = sets.shape[1]
    sub = A[sets[:, :, None], sets[:, None, :]]
    return (sub.sum((1, 2)) - s) / (s * (s - 1))


def main(seed):
    S = c6.load(seed, parts=("layout",))
    c, genes, st = S["c"], np.array(S["genes"]), S["stamp"]
    lay = c.layout
    p = len(genes)
    labels = np.asarray(lay.labels).astype(int)
    niches = sorted(set(labels.tolist()))
    out = c6.OUT / ("full" if seed == "full" else f"seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0 if seed == "full" else seed)
    libs = {"Hallmark": read_gmt(c6.GMT["Hallmark"], genes),
            "GO_BP": read_gmt(c6.GMT["GO_BP"], genes, drop_regulation=True)}
    print(f"seed {seed}: {len(labels)} index tiles, {len(niches)} niches, p = {p}; terms "
          + ", ".join(f"{k} {len(v[0])}" for k, v in libs.items()), flush=True)

    R = {k: c6.niche_corr([lay.spd_matrices[i] for i in np.flatnonzero(labels == k)]) for k in niches}
    sizes = sorted({e - s for k in niches for s, e in lay.block_dict[k]})
    null_sets = {s: np.stack([rng.choice(p, s, replace=False) for _ in range(N_NULL)]) for s in sizes}
    null_rows = []
    null_enriched = {}
    for s, sets in null_sets.items():
        m = np.zeros((N_NULL, p), dtype=np.int32)
        m[np.arange(N_NULL)[:, None], sets] = 1
        null_enriched[s] = float(enriched_any(m, libs).mean())
        null_rows.append({"size": s, "enriched_fraction": null_enriched[s], "n_sets": N_NULL, "tag": st["tag"]})
    pd.DataFrame(null_rows).to_csv(out / "enrich_null.csv", index=False)

    rows = []
    for k in niches:
        perm = np.asarray(lay.perm_list[k])
        A = np.abs(R[k])
        for b, (s0, e0) in enumerate(lay.block_dict[k]):
            g = perm[s0:e0]
            size = len(g)
            sub = A[np.ix_(g, g)]
            within = (sub.sum() - size) / (size * (size - 1))
            rest = np.setdiff1d(np.arange(p), g)
            between = float(A[np.ix_(g, rest)].mean())
            null = mean_abs_within(R[k], null_sets[size])
            hubs = g[np.argsort(-(sub.sum(1) - 1), kind="stable")[:3]]
            row = {"niche": k, "block": b, "start": s0, "end": e0, "size": size,
                   "n_tiles": int((labels == k).sum()), "genes": " ".join(genes[g]), "hubs": " ".join(genes[hubs]),
                   "within": float(within), "between": between, "null_mean": float(null.mean()),
                   "null_p": float((1 + (null >= within).sum()) / (1 + N_NULL)),
                   "null_enriched": null_enriched[size]}
            mask = np.zeros((1, p), dtype=np.int32)
            mask[0, g] = 1
            best = None
            for lib, (names, T) in libs.items():
                kk, q = enrich(mask, T)
                ok = np.flatnonzero((q[0] < FDR) & (kk[0] >= 2))
                if len(ok):
                    j = ok[np.lexsort((-kk[0, ok], q[0, ok]))[0]]
                    row[f"{lib}_term"], row[f"{lib}_fdr"] = names[j], float(q[0, j])
                    row[f"{lib}_genes"] = " ".join(genes[g][T[j, g] > 0])
                    best = best or lib
            row["enriched"] = best is not None
            row["key_lib"] = best
            if best:
                row["key_term"], row["key_fdr"] = row[f"{best}_term"], row[f"{best}_fdr"]
                row["key_genes"] = row[f"{best}_genes"]
            row["covarying"] = row["null_p"] < 0.05
            row["tag"] = st["tag"]
            rows.append(row)
    B = pd.DataFrame(rows)
    B.to_csv(out / "blocks.csv", index=False)
    print(B.groupby("niche")[["covarying", "enriched"]].sum().to_string(), flush=True)
    print(f"within / between median {B.within.median():.3f} / {B.between.median():.3f}; null within "
          f"{B.null_mean.median():.3f}; enriched {B.enriched.mean():.2f} vs null "
          f"{B.null_enriched.mean():.2f}", flush=True)

    types = S["types"]
    section = pd.Series(types).value_counts(normalize=True)
    trows = []
    for k in niches:
        cells = np.concatenate([S["tiles"][c.train[i]].idx for i in np.flatnonzero(labels == k)])
        frac = pd.Series(types[cells]).value_counts(normalize=True)
        n_type = pd.Series(types[cells]).value_counts()
        for t in section.index:
            trows.append({"niche": k, "type": t, "n_cells": int(n_type.get(t, 0)), "frac": float(frac.get(t, 0.0)),
                          "section_frac": float(section[t]), "niche_cells": len(cells),
                          "share_of_type": float(n_type.get(t, 0) / (types == t).sum()), "tag": st["tag"]})
    pd.DataFrame(trows).to_csv(out / "niche_types.csv", index=False)

    m = int(np.flatnonzero(genes == MARKER)[0])
    arows = []
    for k in niches:
        perm = np.asarray(lay.perm_list[k])
        blk = next(perm[s0:e0] for s0, e0 in lay.block_dict[k] if m in perm[s0:e0])
        for j in range(p):
            if j != m:
                arows.append({"niche": k, "gene": genes[j], "r": float(R[k][m, j]), "in_block": bool(j in blk),
                              "block_size": len(blk), "tag": st["tag"]})
    pd.DataFrame(arows).to_csv(out / "acta2.csv", index=False)

    if seed in ("full", c6.data.EXAMPLE_SEED):
        for k in niches:
            perm = np.asarray(lay.perm_list[k])
            np.save(out / f"corr_niche{k}.npy", R[k][np.ix_(perm, perm)].astype(np.float32))
        (out / "layout.json").write_text(json.dumps(
            {str(k): {"perm": np.asarray(lay.perm_list[k]).tolist(),
                      "blocks": [[int(s0), int(e0)] for s0, e0 in lay.block_dict[k]]} for k in niches}
            | {"genes": genes.tolist(), "stamp": st}))
        niche_of = np.full(len(S["xy"]), -1, dtype=np.int8)
        for i, t in enumerate(c.train):
            niche_of[S["tiles"][t].idx] = labels[i]
        np.savez_compressed(out / "cells_niche.npz", xy=S["xy"], niche=niche_of)
    print(f"done seed {seed}", flush=True)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    main(arg if arg == "full" else int(arg))
