"""Fig 7 inputs: whole-tile and gene-list queries on every indexed section, with every index (one build per task).

Usage (sbatch array, task = 5 * dataset + seed): sbatch --array=0-39 experiments/common/run.sbatch experiments/fig7_skin/c_queries.py
Writes results/fig7_skin/<key>/seed<s>/{tile_queries.csv, gene_lists.csv}; skin seed 0 also example.csv,
example_cells.npz (B), skin_classes.csv and signature_cells.npz (A).

As Fig 6 (experiments/fig6_breast/c_queries.py), on every section:
- Whole-tile queries = every held-out tile; Spindle's top 10 = the 10 smallest exact d_B (the cache's D); Spindle-lite's
  = the first 10 tiles of search_vectors (best first; default g; fewer if it returns fewer). TV between the query's
  composition and the pooled composition of the 10 tiles' cells, against the mean TV of N_RANDOM random 10-tile draws.
  Composition labels (common7): breast, the 8 classes of Fig 6; the others, 10x graph-based clusters (cells without a
  cluster left out). Recorded, never claimed: the expression top 10 (Euclidean on the z-scored tile-mean
  log-normalised expression).
- Gene-list queries: the tissue's curated signatures (signatures.TISSUE_MODULES[TISSUE[key]]) with >= 2 index genes
  (missing genes dropped by sd.gene_indices and recorded), each the template query T (sd.template_query on the build's
  omega) searched with Spindle-partial (exact d_B^S, top 10; query "T") and with Spindle-lite-partial (first 10 of
  search_vectors, default g; query "TL"). Controls computed and reported, never claimed against: D = diag(omega_S)
  with Spindle-partial, X = the 10 index tiles with the highest tile score. Readouts per (list, query): Cliff's delta
  of the tile score of the returned tiles against the other index tiles; co-variation = mean off-diagonal correlation
  of S over a tile's cells (raw counts), median over the returned tiles against the median over all index tiles;
  Jaccard with D and X.
SKIN EXAMPLE RULES (fixed before the run; seed 0):
- B: among the held-out tiles with >= MIN_CELLS cells, the one with the largest share of keratinocyte-class cells
  (common7.skin_classes; ties: lower tile id). Reported with it: the class composition of the query, of Spindle's top
  10, of a random 10 (mean of N_RANDOM draws) and of the section, and the base rate = the share of all held-out
  queries whose top-10 TV is below the random-10 TV.
- A: all three skin signatures; a cell's score = mean over S of its log-normalised expression; Spindle-partial's top
  10 outlined.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common7 as c7  # noqa: E402
from common7 import c6, sd  # noqa: E402

MIN_CELLS = 100
EXAMPLE_CLASS = "Keratinocyte"


def tile_queries(S, seed, out):
    c, st = S["c"], S["stamp"]
    n = len(c.train)
    cnt_idx, cnt_q = c7.comp_counts(S, c.train), c7.comp_counts(S, c.test)
    mu_idx, mu_q = c6.tile_means(S, c.train), c6.tile_means(S, c.test)
    m, s = mu_idx.mean(0), mu_idx.std(0) + 1e-9
    z_idx, z_q = (mu_idx - m) / s, (mu_q - m) / s
    rng = np.random.default_rng(seed)
    rows, tops = [], {}
    for j, t in enumerate(c.test):
        if cnt_q[j].sum() == 0:
            continue
        q = c7.shares(cnt_q[j])
        top = np.argsort(c.D[j], kind="stable")[:c7.TOP]
        expr = np.argsort(((z_idx - z_q[j]) ** 2).sum(1), kind="stable")[:c7.TOP]
        lite_rows, _, _ = c.lite.search_vectors(c.qvs[j])
        lite = np.asarray(lite_rows[:c7.TOP], dtype=int)
        rand = [c6.tv(q, c7.shares(cnt_idx[rng.choice(n, c7.TOP, replace=False)])) for _ in range(c7.N_RANDOM)]
        tops[j] = top
        rows.append({"key": S["key"], "seed": seed, "query": j, "tile": int(t), "n_cells": int(cnt_q[j].sum()),
                     "tv_top": c6.tv(q, c7.shares(cnt_idx[top])), "tv_random": float(np.mean(rand)),
                     "tv_lite": c6.tv(q, c7.shares(cnt_idx[lite])), "tv_expr": c6.tv(q, c7.shares(cnt_idx[expr])),
                     "lite_in_top": int(len(set(top.tolist()) & set(lite.tolist()))), "lite_found": len(lite_rows),
                     "n_labels": len(S["comp_names"]), "tag": st["tag"]})
    T = pd.DataFrame(rows)
    T.to_csv(out / "tile_queries.csv", index=False)
    print(f"tile queries: {len(T)} of {len(c.test)}; TV top {T.tv_top.median():.3f}, lite {T.tv_lite.median():.3f}, "
          f"random {T.tv_random.median():.3f}, expression {T.tv_expr.median():.3f}; below random "
          f"{(T.tv_top < T.tv_random).mean():.3f} (lite {(T.tv_lite < T.tv_random).mean():.3f})", flush=True)
    if S["key"] == c7.EXAMPLE_KEY and seed == c6.data.EXAMPLE_SEED:
        skin_example(S, T, tops, out)


def skin_example(S, T, tops, out):
    c, st = S["c"], S["stamp"]
    cls, table = c7.skin_classes(S)
    table.assign(tag=st["tag"]).to_csv(out / "skin_classes.csv", index=False)
    names = [n for n, _ in c7.SKIN_CLASSES]
    k_ex = names.index(EXAMPLE_CLASS)

    def counts(t):
        v = cls[S["tiles"][t].idx]
        return np.bincount(v[v >= 0], minlength=len(names)).astype(float)

    Ci = np.stack([counts(t) for t in c.train])
    T = T.assign(share=[counts(int(t))[k_ex] / max(counts(int(t)).sum(), 1) for t in T.tile])
    cand = T[T.n_cells >= MIN_CELLS].sort_values(["share", "tile"], ascending=[False, True])
    j = int(cand["query"].iloc[0])
    top = tops[j]
    rng = np.random.default_rng(0)
    rand = np.mean([c7.shares(Ci[rng.choice(len(c.train), c7.TOP, replace=False)]) for _ in range(c7.N_RANDOM)], 0)
    section = c7.shares(np.bincount(cls[cls >= 0], minlength=len(names)).astype(float))
    comp = {"query": c7.shares(counts(int(c.test[j]))), "top10": c7.shares(Ci[top]), "random10": rand,
            "section": section}
    ex = pd.DataFrame([{"set": k, **{cn: float(v) for cn, v in zip(names, vec)}} for k, vec in comp.items()])
    ex["query_tile"], ex["dist_top"] = int(c.test[j]), " ".join(f"{c.D[j][r]:.4f}" for r in top)
    ex["top_tiles"] = " ".join(str(int(c.train[r])) for r in top)
    ex["n_candidates"], ex["tag"] = len(cand), st["tag"]
    ex["base_rate"] = float((T.tv_top < T.tv_random).mean())
    ex["n_queries"] = len(T)
    ex.to_csv(out / "example.csv", index=False)
    ids = [int(c.test[j])] + [int(c.train[r]) for r in top]
    arrays = {}
    for k, t in enumerate(ids):
        tile = S["tiles"][t]
        arrays[f"xy{k}"], arrays[f"cls{k}"] = S["xy"][tile.idx], cls[tile.idx].astype(np.int8)
        arrays[f"bbox{k}"] = np.asarray(tile.bbox, dtype=np.float64)
    np.savez_compressed(out / "example_cells.npz", tiles=np.array(ids), xy=S["xy"], cls=cls.astype(np.int8),
                        classes=np.array(names), **arrays)
    print(f"example: held-out tile {ids[0]} ({EXAMPLE_CLASS} {comp['query'][k_ex]:.2f}, {len(cand)} candidates); "
          f"top 10 {comp['top10'][k_ex]:.2f}, random 10 {rand[k_ex]:.2f}, section {section[k_ex]:.3f}", flush=True)
    print(table.round(2).to_string(index=False), flush=True)


def gene_lists(S, seed, out):
    c, st = S["c"], S["stamp"]
    n = len(c.train)
    mu = c6.tile_means(S, c.train)
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    cells = [S["tiles"][t].idx for t in c.train]
    rows, cell_scores, top_tiles = [], {}, {}
    for name, idx, found, missing in c7.gene_lists(S["key"], S["genes"]):
        cov_t, _ = sd.template_query(c.omega, idx)
        tops = {"T": cov_t, "D": np.diag(c.omega[idx])}
        tops = {k: np.argsort(c.partial.distances(Q, idx), kind="stable")[:c7.TOP] for k, Q in tops.items()}
        lite_rows, _, _ = c.plite.search_vectors(c.plite.query_vectors(cov_t, idx))
        tops["TL"] = np.asarray(lite_rows[:c7.TOP], dtype=int)
        score = z[:, idx].mean(1)
        tops["X"] = np.argsort(-score, kind="stable")[:c7.TOP]
        cov_all = np.array([c6.mean_offdiag_corr(S["X"][cl][:, idx]) for cl in cells])
        for qn, top in tops.items():
            other = np.setdiff1d(np.arange(n), top)
            rows.append({"key": S["key"], "seed": seed, "list": name, "n_genes": len(idx), "genes": " ".join(found),
                         "missing": " ".join(missing), "query": qn, "n_returned": len(top),
                         "delta": float(c6.cliffs_delta(score[top], score[other])),
                         "cov_top": float(np.nanmedian(cov_all[top])), "cov_all": float(np.nanmedian(cov_all)),
                         "jacc_T": c6.jaccard(top, tops["T"]), "jacc_D": c6.jaccard(top, tops["D"]),
                         "jacc_X": c6.jaccard(top, tops["X"]), "tag": st["tag"]})
        cell_scores[name] = S["L"][:, idx].mean(1).astype(np.float16)
        top_tiles[name] = np.array([S["tiles"][c.train[i]].bbox for i in tops["T"]], dtype=np.float64)
    G = pd.DataFrame(rows)
    G["lift"] = G.cov_top - G.cov_all
    G.to_csv(out / "gene_lists.csv", index=False)
    print(G[["list", "n_genes", "query", "n_returned", "delta", "cov_top", "cov_all", "jacc_T"]].round(3)
          .to_string(index=False), flush=True)
    if S["key"] == c7.EXAMPLE_KEY and seed == c6.data.EXAMPLE_SEED:
        np.savez_compressed(out / "signature_cells.npz", xy=S["xy"], lists=np.array(list(cell_scores)),
                            **{f"score_{k}": v for k, v in cell_scores.items()},
                            **{f"top_{k}": v for k, v in top_tiles.items()})


def main(key, seed):
    S = c7.load(key, seed, parts=("layout", "lite", "partial", "plite", "queries", "D"))
    out = c7.out_dir(key, seed)
    tile_queries(S, seed, out)
    gene_lists(S, seed, out)
    print(f"done {key} seed {seed}", flush=True)


if __name__ == "__main__":
    main(*c7.task(int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))))
