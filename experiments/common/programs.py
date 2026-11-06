"""Breast block programmes (Supplementary Fig S2; the whole-section build "full": every tile indexed) and breast
whole-tile queries (Fig 3C; seed 0). The section loader and the statistics are sections.py. Every function returns
tables (and arrays); R6_breast_programs.ipynb writes them.

    import sections as sx, programs as pg
    S = sx.load("breast_cancer", "full", parts=pg.PROGRAM_PARTS)
    out = pg.programs(S, "full")                         # blocks, enrich_null, niche_types, acta2 (+ maps)
    S = sx.load("breast_cancer", 0, parts=pg.QUERY_PARTS)
    out = pg.tile_queries(S, 0)

Programmes, per niche k and block b of the build (layout.perm_list[k] cut at layout.block_dict[k]), on the niche-mean
correlation R_k (the mean of the niche's shrunk index covariances, as a correlation):
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

Whole-tile queries = every held-out tile of the seed. Spindle's top 10 (top 5) = the first 10 (5) tiles of
search_vectors (default g = 0.25; fewer if it returns fewer, NaN if none; user, 2 Nov: every search result
from Spindle). Readout (cells, not tiles): TV = 0.5 sum |a - b| between the query's 8-class composition
(sections.CLASSES) and the pooled composition of the top 10's (top 5's) cells; null = the mean TV of N_RANDOM random
10-tile (5-tile) draws of index tiles. Recorded, never claimed: Spindle-naive's top 10 / top 5 (the smallest exact d_B,
the cache's D) and their TV; how many of Spindle-naive's 10 Spindle returns among its first 10.
EXAMPLE RULE (approved 26 Oct, fixed before the run): seed 0; among the held-out tiles with >= 100 cells, the one with
the largest share of myoepithelial cells (ties: lower tile id). Reported with it: Spindle's top 10's and a random 10's
pooled composition, the section's, and the base rate = the share of all held-out queries whose Spindle top-10 TV is
below the random-10 TV.
(Until 2 Nov the top 10 / top 5 were Spindle-naive's: columns tv_top / tv_top5, now tv_naive / tv_naive5.)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

import data
import sections as sx

KEY = sx.BREAST
PROGRAM_PARTS = ("layout",)
QUERY_PARTS = ("layout", "spindle", "queries", "D")
GMT = {"Hallmark": Path.home() / ".cache" / "gseapy" / "Enrichr.MSigDB_Hallmark_2020.gmt",
       "GO_BP": Path.home() / ".cache" / "gseapy" / "Enrichr.GO_Biological_Process_2023.gmt"}
N_NULL = 1000
MIN_TERM = 3
FDR = 0.05
MARKER = "ACTA2"
MIN_CELLS = 100
EXAMPLE_CLASS = "Myoepithelial"
TOP5 = 5  # the top 5 of Fig 4C, with its own random 5-tile null


# ---------------------------------------------------------------- enrichment
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
    return k, np.vstack([sx.bh(row) for row in p])


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


# ---------------------------------------------------------------- programmes
def programs(S, seed):
    """{"blocks", "enrich_null", "niche_types", "acta2": DataFrames}; for "full" and the example seed also
    "corr" ({niche: niche-mean correlation in the niche's gene order, float32}), "layout" (gene order and blocks per
    niche, JSON-ready) and "cells_niche" (every cell's x, y and its tile's niche; -1 = held-out or dropped tile)."""
    c, genes, st = S["c"], np.array(S["genes"]), S["stamp"]
    lay = c.layout
    p = len(genes)
    labels = np.asarray(lay.labels).astype(int)
    niches = sorted(set(labels.tolist()))
    rng = np.random.default_rng(0 if seed == "full" else seed)
    libs = {"Hallmark": read_gmt(GMT["Hallmark"], genes),
            "GO_BP": read_gmt(GMT["GO_BP"], genes, drop_regulation=True)}
    print(f"seed {seed}: {len(labels)} index tiles, {len(niches)} niches, p = {p}; terms "
          + ", ".join(f"{k} {len(v[0])}" for k, v in libs.items()), flush=True)

    R = {k: sx.niche_corr([lay.spd_matrices[i] for i in np.flatnonzero(labels == k)]) for k in niches}
    sizes = sorted({e - s for k in niches for s, e in lay.block_dict[k]})
    null_sets = {s: np.stack([rng.choice(p, s, replace=False) for _ in range(N_NULL)]) for s in sizes}
    null_rows = []
    null_enriched = {}
    for s, sets in null_sets.items():
        m = np.zeros((N_NULL, p), dtype=np.int32)
        m[np.arange(N_NULL)[:, None], sets] = 1
        null_enriched[s] = float(enriched_any(m, libs).mean())
        null_rows.append({"size": s, "enriched_fraction": null_enriched[s], "n_sets": N_NULL, "tag": st["tag"]})

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
    out = {"blocks": pd.DataFrame(rows), "enrich_null": pd.DataFrame(null_rows)}

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
    out["niche_types"] = pd.DataFrame(trows)

    m = int(np.flatnonzero(genes == MARKER)[0])
    arows = []
    for k in niches:
        perm = np.asarray(lay.perm_list[k])
        blk = next(perm[s0:e0] for s0, e0 in lay.block_dict[k] if m in perm[s0:e0])
        for j in range(p):
            if j != m:
                arows.append({"niche": k, "gene": genes[j], "r": float(R[k][m, j]), "in_block": bool(j in blk),
                              "block_size": len(blk), "tag": st["tag"]})
    out["acta2"] = pd.DataFrame(arows)

    if seed in ("full", data.EXAMPLE_SEED):
        out["corr"] = {k: R[k][np.ix_(np.asarray(lay.perm_list[k]), np.asarray(lay.perm_list[k]))].astype(np.float32)
                       for k in niches}
        out["layout"] = ({str(k): {"perm": np.asarray(lay.perm_list[k]).tolist(),
                                   "blocks": [[int(s0), int(e0)] for s0, e0 in lay.block_dict[k]]} for k in niches}
                         | {"genes": genes.tolist(), "stamp": st})
        niche_of = np.full(len(S["xy"]), -1, dtype=np.int8)
        for i, t in enumerate(c.train):
            niche_of[S["tiles"][t].idx] = labels[i]
        out["cells_niche"] = {"xy": S["xy"], "niche": niche_of}
    return out


# ---------------------------------------------------------------- queries
def tile_queries(S, seed):
    """{"tile_queries"}; for the example seed also "example" (DataFrame) and "example_cells" (arrays)."""
    c, st = S["c"], S["stamp"]
    n = len(c.train)
    cnt_idx = sx.class_counts(S, c.train)
    cnt_q = sx.class_counts(S, c.test)
    rng = np.random.default_rng(seed)
    rng5 = np.random.default_rng([seed, TOP5])  # own stream: the top-10 columns stay as before
    rows, tops = [], {}

    def tv_of(q, r):
        return sx.tv(q, sx.class_shares(cnt_idx[r])) if len(r) else np.nan

    for j, t in enumerate(c.test):
        q = cnt_q[j] / cnt_q[j].sum()
        naive = np.argsort(c.D[j], kind="stable")[:sx.TOP]
        spindle_rows, _, _ = c.spindle.search_vectors(c.qvs[j])
        top = np.asarray(spindle_rows[:sx.TOP], dtype=int)
        rand = [sx.tv(q, sx.class_shares(cnt_idx[rng.choice(n, sx.TOP, replace=False)])) for _ in range(sx.N_RANDOM)]
        rand5 = [sx.tv(q, sx.class_shares(cnt_idx[rng5.choice(n, TOP5, replace=False)])) for _ in range(sx.N_RANDOM)]
        tops[j] = top
        rows.append({"query": j, "tile": int(t), "n_cells": int(cnt_q[j].sum()),
                     **{f"q_{k}": float(v) for k, v in zip(sx.CLASS_NAMES, q)},
                     "tv_spindle": tv_of(q, top), "tv_random": float(np.mean(rand)),
                     "tv_spindle5": tv_of(q, top[:TOP5]), "tv_random5": float(np.mean(rand5)),
                     "tv_naive": tv_of(q, naive), "tv_naive5": tv_of(q, naive[:TOP5]),
                     "spindle_in_naive": int(len(set(naive.tolist()) & set(top.tolist()))),
                     "spindle_found": len(spindle_rows), "tag": st["tag"]})
    T = pd.DataFrame(rows)
    out = {"tile_queries": T}
    if seed != data.EXAMPLE_SEED:
        return out
    col = f"q_{EXAMPLE_CLASS}"
    cand = T[T.n_cells >= MIN_CELLS].sort_values([col, "tile"], ascending=[False, True])
    j = int(cand["query"].iloc[0])
    top = tops[j]
    rand = np.mean([sx.class_shares(cnt_idx[rng.choice(n, sx.TOP, replace=False)]) for _ in range(sx.N_RANDOM)], 0)
    section = np.bincount(S["cls"], minlength=len(sx.CLASSES)) / len(S["cls"])
    rand5 = np.mean([sx.class_shares(cnt_idx[rng5.choice(n, TOP5, replace=False)]) for _ in range(sx.N_RANDOM)], 0)
    comp = {"query": cnt_q[j] / cnt_q[j].sum(), "top10": sx.class_shares(cnt_idx[top]), "random10": rand,
            "section": section, "top5": sx.class_shares(cnt_idx[top[:TOP5]]), "random5": rand5}
    ex = pd.DataFrame([{"set": k, **{cn: float(v) for cn, v in zip(sx.CLASS_NAMES, vec)}} for k, vec in comp.items()])
    ex["query_tile"], ex["dist_top"] = int(c.test[j]), " ".join(f"{c.D[j][r]:.4f}" for r in top)
    ex["top_tiles"] = " ".join(str(int(c.train[r])) for r in top)
    ex["n_candidates"], ex["tag"] = len(cand), st["tag"]
    ex["base_rate"] = float((T.tv_spindle < T.tv_random).mean())
    ids = [int(c.test[j])] + [int(c.train[r]) for r in top]
    arrays = {}
    for k, t in enumerate(ids):
        tile = S["tiles"][t]
        arrays[f"xy{k}"], arrays[f"cls{k}"] = S["xy"][tile.idx], S["cls"][tile.idx].astype(np.int8)
        arrays[f"bbox{k}"] = np.asarray(tile.bbox, dtype=np.float64)
    out["example"], out["example_cells"] = ex, {"tiles": np.array(ids), **arrays}
    print(f"example: held-out tile {ids[0]} ({EXAMPLE_CLASS} {comp['query'][2]:.2f}, {len(cand)} candidates); top 10 "
          f"{comp['top10'][2]:.2f}, random 10 {rand[2]:.2f}, section {section[2]:.3f}", flush=True)
    return out

