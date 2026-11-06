"""Gene-list queries on every indexed section (Fig 4C, Supplementary Table of the signatures; R7_queries.ipynb).
Section loader and statistics in sections.py.

On every section (one (key, seed) at a time): the tissue's curated signatures (signatures.TISSUE_MODULES[TISSUE[key]])
with >= 2 index genes (missing genes dropped by sd.gene_indices and recorded), each the template query T
(sd.template_query on the build's omega) searched with Spindle-naive (exact d_B^S, top 10; method "naive_partial") and
with Spindle (first 10 of search_vectors, default g; method "spindle_partial"). Readouts per (list, method): Cliff's
delta of the tile score (mean over S of the z-scored tile-mean log-normalised expression) of the returned tiles against
the other index tiles; co-variation = mean off-diagonal correlation of S over a tile's cells (raw counts), median over
the returned tiles against the median over all index tiles; Jaccard of Spindle's tiles with Spindle-naive's.
Column names (old -> new): gene-list query T / TS -> method naive_partial / spindle_partial, jacc_T ->
jacc_naive_partial.
"""

import numpy as np
import pandas as pd

import sections as sx
import spindle_dev as sd

PARTS = ("layout", "naive_partial", "spindle_partial")


def gene_lists(S, seed):
    """Table of every list x method."""
    c, st = S["c"], S["stamp"]
    n = len(c.train)
    mu = sx.tile_means(S, c.train)
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    cells = [S["tiles"][t].idx for t in c.train]
    rows = []
    for name, idx, found, missing in sx.gene_lists(S["key"], S["genes"]):
        cov_t, _ = sd.template_query(c.omega, idx)
        tops = {"naive_partial": np.argsort(c.naive_partial.distances(cov_t, idx), kind="stable")[:sx.TOP]}
        spindle_rows, _, _ = c.spindle_partial.search_vectors(c.spindle_partial.query_vectors(cov_t, idx))
        tops["spindle_partial"] = np.asarray(spindle_rows[:sx.TOP], dtype=int)
        score = z[:, idx].mean(1)
        cov_all = np.array([sx.mean_offdiag_corr(S["X"][cl][:, idx]) for cl in cells])
        for method in ("naive_partial", "spindle_partial"):
            top = tops[method]
            other = np.setdiff1d(np.arange(n), top)
            rows.append({"key": S["key"], "seed": seed, "list": name, "n_genes": len(idx), "genes": " ".join(found),
                         "missing": " ".join(missing), "method": method, "n_returned": len(top),
                         "delta": float(sx.cliffs_delta(score[top], score[other])),
                         "cov_top": float(np.nanmedian(cov_all[top])), "cov_all": float(np.nanmedian(cov_all)),
                         "jacc_naive_partial": sx.jaccard(top, tops["naive_partial"]), "tag": st["tag"]})
    G = pd.DataFrame(rows)
    G["lift"] = G.cov_top - G.cov_all
    return G


def summary_lists(g):
    """Per dataset, list and method: mean and s.d. over seeds."""
    num = ["delta", "cov_top", "cov_all", "lift", "jacc_naive_partial", "n_genes", "n_returned"]
    by = ["key", "list", "method"]
    out = g.groupby(by, sort=False)[num].mean().join(g.groupby(by, sort=False)[num].std(), rsuffix="_sd")
    out = out.reset_index()
    out["missing"] = g.groupby(by, sort=False)["missing"].first().fillna("").to_numpy()
    out["tag"] = g.tag.iloc[0]
    return out
