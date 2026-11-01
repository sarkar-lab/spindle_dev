"""Whole-tile and gene-list queries on every indexed section, with every index (Fig 7; R7_queries.ipynb). Ported from
experiments/fig7_skin/c_queries.py (in git history); section loader and statistics in sections.py.

As Fig 6, on every section (one (key, seed) at a time):
- Whole-tile queries = every held-out tile; Spindle-naive's top 10 = the 10 smallest exact d_B (the cache's D);
  Spindle's = the first 10 tiles of search_vectors (best first; default g; fewer if it returns fewer). TV between the
  query's composition and the pooled composition of the 10 tiles' cells, against the mean TV of N_RANDOM random 10-tile
  draws. Composition labels: breast, the 8 classes of Fig 6; the others, 10x graph-based clusters (cells without a
  cluster left out). Recorded, never claimed: the expression top 10 (Euclidean on the z-scored tile-mean
  log-normalised expression).
- Gene-list queries: the tissue's curated signatures (signatures.TISSUE_MODULES[TISSUE[key]]) with >= 2 index genes
  (missing genes dropped by sd.gene_indices and recorded), each the template query T (sd.template_query on the build's
  omega) searched with Spindle-naive (exact d_B^S, top 10; method "naive_partial") and with Spindle (first 10 of
  search_vectors, default g; method "spindle_partial"). Controls computed and reported, never claimed against:
  "D" = diag(omega_S) with Spindle-naive, "X" = the 10 index tiles with the highest tile score. Readouts per (list,
  method): Cliff's delta of the tile score of the returned tiles against the other index tiles; co-variation = mean
  off-diagonal correlation of S over a tile's cells (raw counts), median over the returned tiles against the median
  over all index tiles; Jaccard with naive_partial, D and X.
SKIN EXAMPLE RULES (fixed before the run; seed 0):
- B: among the held-out tiles with >= MIN_CELLS cells, the one with the largest share of keratinocyte-class cells
  (sections.skin_classes; ties: lower tile id). Reported with it: the class composition of the query, of
  Spindle-naive's top 10, of a random 10 (mean of N_RANDOM draws) and of the section, and the base rate = the share of
  all held-out queries whose top-10 TV is below the random-10 TV.
- A: all three skin signatures; a cell's score = mean over S of its log-normalised expression; Spindle-naive's top
  10 outlined.
Column names (old -> new): tv_top -> tv_naive, spindle_in_top -> spindle_in_naive; gene lists query T / TS / D / X ->
method naive_partial / spindle_partial / D / X, jacc_T -> jacc_naive_partial.
"""

import numpy as np
import pandas as pd

import sections as sx
import spindle_dev as sd

EXAMPLE_KEY = "skin_melanoma"
PARTS = ("layout", "spindle", "naive_partial", "spindle_partial", "queries", "D")
MIN_CELLS = 100
EXAMPLE_CLASS = "Keratinocyte"


def tile_queries(S, seed):
    """(table of every held-out query, {query position: Spindle-naive's top 10 rows})."""
    c, st = S["c"], S["stamp"]
    n = len(c.train)
    cnt_idx, cnt_q = sx.comp_counts(S, c.train), sx.comp_counts(S, c.test)
    mu_idx, mu_q = sx.tile_means(S, c.train), sx.tile_means(S, c.test)
    m, s = mu_idx.mean(0), mu_idx.std(0) + 1e-9
    z_idx, z_q = (mu_idx - m) / s, (mu_q - m) / s
    rng = np.random.default_rng(seed)
    rows, tops = [], {}
    for j, t in enumerate(c.test):
        if cnt_q[j].sum() == 0:
            continue
        q = sx.shares(cnt_q[j])
        top = np.argsort(c.D[j], kind="stable")[:sx.TOP]
        expr = np.argsort(((z_idx - z_q[j]) ** 2).sum(1), kind="stable")[:sx.TOP]
        spindle_rows, _, _ = c.spindle.search_vectors(c.qvs[j])
        spindle = np.asarray(spindle_rows[:sx.TOP], dtype=int)
        rand = [sx.tv(q, sx.shares(cnt_idx[rng.choice(n, sx.TOP, replace=False)])) for _ in range(sx.N_RANDOM)]
        tops[j] = top
        rows.append({"key": S["key"], "seed": seed, "query": j, "tile": int(t), "n_cells": int(cnt_q[j].sum()),
                     "tv_naive": sx.tv(q, sx.shares(cnt_idx[top])), "tv_random": float(np.mean(rand)),
                     "tv_spindle": sx.tv(q, sx.shares(cnt_idx[spindle])),
                     "tv_expr": sx.tv(q, sx.shares(cnt_idx[expr])),
                     "spindle_in_naive": int(len(set(top.tolist()) & set(spindle.tolist()))),
                     "spindle_found": len(spindle_rows),
                     "n_labels": len(S["comp_names"]), "tag": st["tag"]})
    return pd.DataFrame(rows), tops


def skin_example(S, T, tops):
    """The skin worked example (rule above): (example table, skin class table, arrays of example_cells.npz)."""
    c, st = S["c"], S["stamp"]
    cls, table = sx.skin_classes(S)
    table = table.assign(tag=st["tag"])
    names = [n for n, _ in sx.SKIN_CLASSES]
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
    rand = np.mean([sx.shares(Ci[rng.choice(len(c.train), sx.TOP, replace=False)]) for _ in range(sx.N_RANDOM)], 0)
    section = sx.shares(np.bincount(cls[cls >= 0], minlength=len(names)).astype(float))
    comp = {"query": sx.shares(counts(int(c.test[j]))), "top10": sx.shares(Ci[top]), "random10": rand,
            "section": section}
    ex = pd.DataFrame([{"set": k, **{cn: float(v) for cn, v in zip(names, vec)}} for k, vec in comp.items()])
    ex["query_tile"], ex["dist_top"] = int(c.test[j]), " ".join(f"{c.D[j][r]:.4f}" for r in top)
    ex["top_tiles"] = " ".join(str(int(c.train[r])) for r in top)
    ex["n_candidates"], ex["tag"] = len(cand), st["tag"]
    ex["base_rate"] = float((T.tv_naive < T.tv_random).mean())
    ex["n_queries"] = len(T)
    ids = [int(c.test[j])] + [int(c.train[r]) for r in top]
    arrays = {"tiles": np.array(ids), "xy": S["xy"], "cls": cls.astype(np.int8), "classes": np.array(names)}
    for k, t in enumerate(ids):
        tile = S["tiles"][t]
        arrays[f"xy{k}"], arrays[f"cls{k}"] = S["xy"][tile.idx], cls[tile.idx].astype(np.int8)
        arrays[f"bbox{k}"] = np.asarray(tile.bbox, dtype=np.float64)
    print(f"example: held-out tile {ids[0]} ({EXAMPLE_CLASS} {comp['query'][k_ex]:.2f}, {len(cand)} candidates); "
          f"top 10 {comp['top10'][k_ex]:.2f}, random 10 {rand[k_ex]:.2f}, section {section[k_ex]:.3f}", flush=True)
    return ex, table, arrays


def gene_lists(S, seed, keep_cells=False):
    """(table of every list x method, arrays of signature_cells.npz if keep_cells else None)."""
    c, st = S["c"], S["stamp"]
    n = len(c.train)
    mu = sx.tile_means(S, c.train)
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    cells = [S["tiles"][t].idx for t in c.train]
    rows, cell_scores, top_tiles = [], {}, {}
    for name, idx, found, missing in sx.gene_lists(S["key"], S["genes"]):
        cov_t, _ = sd.template_query(c.omega, idx)
        tops = {"naive_partial": cov_t, "D": np.diag(c.omega[idx])}
        tops = {k: np.argsort(c.naive_partial.distances(Q, idx), kind="stable")[:sx.TOP] for k, Q in tops.items()}
        spindle_rows, _, _ = c.spindle_partial.search_vectors(c.spindle_partial.query_vectors(cov_t, idx))
        tops["spindle_partial"] = np.asarray(spindle_rows[:sx.TOP], dtype=int)
        score = z[:, idx].mean(1)
        tops["X"] = np.argsort(-score, kind="stable")[:sx.TOP]
        cov_all = np.array([sx.mean_offdiag_corr(S["X"][cl][:, idx]) for cl in cells])
        for method in ("naive_partial", "D", "spindle_partial", "X"):  # the old order T, D, TS, X
            top = tops[method]
            other = np.setdiff1d(np.arange(n), top)
            rows.append({"key": S["key"], "seed": seed, "list": name, "n_genes": len(idx), "genes": " ".join(found),
                         "missing": " ".join(missing), "method": method, "n_returned": len(top),
                         "delta": float(sx.cliffs_delta(score[top], score[other])),
                         "cov_top": float(np.nanmedian(cov_all[top])), "cov_all": float(np.nanmedian(cov_all)),
                         "jacc_naive_partial": sx.jaccard(top, tops["naive_partial"]),
                         "jacc_D": sx.jaccard(top, tops["D"]), "jacc_X": sx.jaccard(top, tops["X"]),
                         "tag": st["tag"]})
        if keep_cells:
            cell_scores[name] = S["L"][:, idx].mean(1).astype(np.float16)
            top_tiles[name] = np.array([S["tiles"][c.train[i]].bbox for i in tops["naive_partial"]],
                                       dtype=np.float64)
    G = pd.DataFrame(rows)
    G["lift"] = G.cov_top - G.cov_all
    arrays = None
    if keep_cells:
        arrays = {"xy": S["xy"], "lists": np.array(list(cell_scores)),
                  **{f"score_{k}": v for k, v in cell_scores.items()},
                  **{f"top_{k}": v for k, v in top_tiles.items()}}
    return G, arrays


# ---------------------------------------------------------------- summaries (c_summary.py)
def summary_lists(g):
    """Per dataset, list and method: mean and s.d. over seeds."""
    num = ["delta", "cov_top", "cov_all", "lift", "jacc_naive_partial", "jacc_D", "jacc_X", "n_genes", "n_returned"]
    by = ["key", "list", "method"]
    out = g.groupby(by, sort=False)[num].mean().join(g.groupby(by, sort=False)[num].std(), rsuffix="_sd")
    out = out.reset_index()
    out["missing"] = g.groupby(by, sort=False)["missing"].first().fillna("").to_numpy()
    out["tag"] = g.tag.iloc[0]
    return out


def summary_tiles(t_all):
    """Per dataset and seed: medians of the composition differences, shares below random."""
    rows = []
    for (key, s), t in t_all.groupby(["key", "seed"], sort=False):
        rows.append({"key": key, "seed": s, "n_queries": len(t), "n_labels": int(t.n_labels.iloc[0]),
                     "tv_naive": t.tv_naive.median(), "tv_spindle": t.tv_spindle.median(),
                     "tv_random": t.tv_random.median(), "tv_expr": t.tv_expr.median(),
                     "below_random": (t.tv_naive < t.tv_random).mean(),
                     "spindle_below_random": (t.tv_spindle < t.tv_random).mean(),
                     "spindle_in_naive": t.spindle_in_naive.median(), "tag": t.tag.iloc[0]})
    return pd.DataFrame(rows)
