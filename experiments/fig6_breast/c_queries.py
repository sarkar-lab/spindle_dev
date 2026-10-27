"""Fig 6E-G inputs: whole-tile queries and gene-list queries on breast (one build per task).

Usage (sbatch array, task = seed 0-4): sbatch --array=0-4 experiments/common/run.sbatch experiments/fig6_breast/c_queries.py
Writes results/fig6_breast/seed<s>/{tile_queries.csv, gene_lists.csv, gene_list_hits.csv}; seed 0 also
example.csv + example_cells.npz (Fig 6E) and signature_cells.npz (Fig 6F: every cell's score per signature).

Whole-tile queries = every held-out tile of the seed. Spindle's top 10 = the 10 smallest exact d_B (the cache's D).
Readout (cells, not tiles): TV = 0.5 sum |a - b| between the query's 8-class composition (common6.CLASSES) and the
pooled composition of the top 10's cells; null = the mean TV of N_RANDOM random 10-tile draws of index tiles.
Recorded, never claimed: the expression top 10 (Euclidean on the tile-mean log-normalised expression, z-scored over
the index tiles) and its TV; how many of Spindle's 10 Spindle-lite returns among its first 10 (search_vectors, default
g = 0.25 as in Fig 3).
EXAMPLE RULE (approved 26 Oct, fixed before the run): seed 0; among the held-out tiles with >= 100 cells, the one with
the largest share of myoepithelial cells (ties: lower tile id). Reported with it: its top 10's and a random 10's
pooled composition, the section's, and the base rate = the share of all held-out queries whose top-10 TV is below the
random-10 TV.

Gene-list queries: the 5 breast signatures (signatures.TISSUE_MODULES["breast"]; every gene is an index gene), each
the template query T (sd.template_query on the build's omega: Methods Eq. signature, rho = 0.5, 90th-percentile
variances) searched with Spindle-partial (exact d_B^S), top 10. Controls computed and reported, never claimed against:
D = diag(omega_S) (variances only), and X = the 10 index tiles with the highest tile score. Readouts per (list, query):
- delta = Cliff's delta of the tile score (mean over S of the z-scored tile-mean log-normalised expression) of the
  top 10 against the other index tiles;
- co-variation = mean off-diagonal correlation of S over a tile's cells (raw counts, zero-variance genes dropped);
  median over the top 10 against the median over all index tiles (the typical tile);
- jacc_D, jacc_X: Jaccard of the top 10 with D's and X's.
Map rule (Fig 6F, approved 26 Oct): the two signatures with the largest T delta at seed 0; a cell's score = mean over S
of its log-normalised expression.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common6 as c6  # noqa: E402
from common6 import sd  # noqa: E402
from signatures import TISSUE_MODULES  # noqa: E402

MIN_CELLS = 100
EXAMPLE_CLASS = "Myoepithelial"


def tile_queries(S, seed, out):
    c, st = S["c"], S["stamp"]
    n = len(c.train)
    cnt_idx = c6.class_counts(S, c.train)
    cnt_q = c6.class_counts(S, c.test)
    mu_idx, mu_q = c6.tile_means(S, c.train), c6.tile_means(S, c.test)
    m, s = mu_idx.mean(0), mu_idx.std(0) + 1e-9
    z_idx, z_q = (mu_idx - m) / s, (mu_q - m) / s
    rng = np.random.default_rng(seed)
    rows, tops = [], {}
    for j, t in enumerate(c.test):
        q = cnt_q[j] / cnt_q[j].sum()
        top = np.argsort(c.D[j], kind="stable")[:c6.TOP]
        expr = np.argsort(((z_idx - z_q[j]) ** 2).sum(1), kind="stable")[:c6.TOP]
        lite_rows, _, _ = c.lite.search_vectors(c.qvs[j])
        rand = [c6.tv(q, c6.shares(cnt_idx[rng.choice(n, c6.TOP, replace=False)])) for _ in range(c6.N_RANDOM)]
        tops[j] = top
        rows.append({"query": j, "tile": int(t), "n_cells": int(cnt_q[j].sum()),
                     **{f"q_{k}": float(v) for k, v in zip(c6.CLASS_NAMES, q)},
                     "tv_top": c6.tv(q, c6.shares(cnt_idx[top])), "tv_random": float(np.mean(rand)),
                     "tv_expr": c6.tv(q, c6.shares(cnt_idx[expr])),
                     "lite_in_top": int(len(set(top.tolist()) & set(np.asarray(lite_rows[:c6.TOP]).tolist()))),
                     "lite_found": len(lite_rows), "tag": st["tag"]})
    T = pd.DataFrame(rows)
    T.to_csv(out / "tile_queries.csv", index=False)
    print(f"tile queries: {len(T)}; TV top {T.tv_top.median():.3f}, random {T.tv_random.median():.3f}, expression "
          f"{T.tv_expr.median():.3f}; below random {(T.tv_top < T.tv_random).mean():.2f}; lite in top 10 median "
          f"{T.lite_in_top.median():.0f}", flush=True)
    if seed != c6.data.EXAMPLE_SEED:
        return
    col = f"q_{EXAMPLE_CLASS}"
    cand = T[T.n_cells >= MIN_CELLS].sort_values([col, "tile"], ascending=[False, True])
    j = int(cand["query"].iloc[0])
    top = tops[j]
    rand = np.mean([c6.shares(cnt_idx[rng.choice(n, c6.TOP, replace=False)]) for _ in range(c6.N_RANDOM)], 0)
    section = np.bincount(S["cls"], minlength=len(c6.CLASSES)) / len(S["cls"])
    comp = {"query": cnt_q[j] / cnt_q[j].sum(), "top10": c6.shares(cnt_idx[top]), "random10": rand,
            "section": section}
    ex = pd.DataFrame([{"set": k, **{cn: float(v) for cn, v in zip(c6.CLASS_NAMES, vec)}} for k, vec in comp.items()])
    ex["query_tile"], ex["dist_top"] = int(c.test[j]), " ".join(f"{c.D[j][r]:.4f}" for r in top)
    ex["top_tiles"] = " ".join(str(int(c.train[r])) for r in top)
    ex["n_candidates"], ex["tag"] = len(cand), st["tag"]
    ex["base_rate"] = float((T.tv_top < T.tv_random).mean())
    ex.to_csv(out / "example.csv", index=False)
    ids = [int(c.test[j])] + [int(c.train[r]) for r in top]
    arrays = {}
    for k, t in enumerate(ids):
        tile = S["tiles"][t]
        arrays[f"xy{k}"], arrays[f"cls{k}"] = S["xy"][tile.idx], S["cls"][tile.idx].astype(np.int8)
        arrays[f"bbox{k}"] = np.asarray(tile.bbox, dtype=np.float64)
    np.savez_compressed(out / "example_cells.npz", tiles=np.array(ids), **arrays)
    print(f"example: held-out tile {ids[0]} ({EXAMPLE_CLASS} {comp['query'][2]:.2f}, {len(cand)} candidates); top 10 "
          f"{comp['top10'][2]:.2f}, random 10 {rand[2]:.2f}, section {section[2]:.3f}", flush=True)


def gene_lists(S, seed, out):
    c, st = S["c"], S["stamp"]
    n = len(c.train)
    mu = c6.tile_means(S, c.train)
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    cells = [S["tiles"][t].idx for t in c.train]
    rows, hits, cell_scores, top_tiles = [], [], {}, {}
    for name, info in TISSUE_MODULES["breast"].items():
        idx, found, missing = sd.gene_indices(S["genes"], info["genes"])
        if missing:
            raise RuntimeError(f"{name}: {missing} not index genes")
        cov_t, _ = sd.template_query(c.omega, idx)
        tops = {"T": cov_t, "D": np.diag(c.omega[idx])}
        tops = {k: np.argsort(c.partial.distances(Q, idx), kind="stable")[:c6.TOP] for k, Q in tops.items()}
        score = z[:, idx].mean(1)
        tops["X"] = np.argsort(-score, kind="stable")[:c6.TOP]
        cov_all = np.array([c6.mean_offdiag_corr(S["X"][cl][:, idx]) for cl in cells])
        for qn, top in tops.items():
            other = np.setdiff1d(np.arange(n), top)
            rows.append({"list": name, "n_genes": len(idx), "genes": " ".join(found), "query": qn,
                         "delta": float(c6.cliffs_delta(score[top], score[other])),
                         "cov_top": float(np.nanmedian(cov_all[top])), "cov_all": float(np.nanmedian(cov_all)),
                         "cov_top_mean": float(np.nanmean(cov_all[top])),
                         "jacc_D": c6.jaccard(top, tops["D"]), "jacc_X": c6.jaccard(top, tops["X"]),
                         "tag": st["tag"]})
            hits += [{"list": name, "query": qn, "rank": r + 1, "row": int(i), "tile": int(c.train[i]),
                      "cov": float(cov_all[i]), "score": float(score[i])} for r, i in enumerate(top)]
        cell_scores[name] = S["L"][:, idx].mean(1).astype(np.float16)
        top_tiles[name] = np.array([S["tiles"][c.train[i]].bbox for i in tops["T"]], dtype=np.float64)
    G = pd.DataFrame(rows)
    G["lift"] = G.cov_top - G.cov_all
    G.to_csv(out / "gene_lists.csv", index=False)
    pd.DataFrame(hits).to_csv(out / "gene_list_hits.csv", index=False)
    print(G[["list", "query", "delta", "cov_top", "cov_all", "jacc_D", "jacc_X"]].round(3).to_string(index=False),
          flush=True)
    if seed == c6.data.EXAMPLE_SEED:
        np.savez_compressed(out / "signature_cells.npz", xy=S["xy"], lists=np.array(list(cell_scores)),
                            **{f"score_{k}": v for k, v in cell_scores.items()},
                            **{f"top_{k}": v for k, v in top_tiles.items()})


def main(seed):
    S = c6.load(seed, parts=("layout", "lite", "partial", "queries", "D"))
    out = c6.OUT / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    tile_queries(S, seed, out)
    gene_lists(S, seed, out)
    print(f"done seed {seed}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
