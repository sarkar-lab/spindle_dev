"""Search distance against expression on skin melanoma, and gene-set queries on skin (Fig 4; R9_skin_maps.ipynb).
Section loader and statistics in sections.py; programme queries as pan_cancer.py.

RULES (fixed before the run):
- MATCH, every section of pan_cancer.KEYS and seed 0-4, every programme of results/pan_cancer/exemplar.csv: the query is
  Fig 8's (diag(sigma) R_S diag(sigma), sigma quantile-matched to the section's index tiles). Closeness of an index tile
  = -exact d_B^S (Spindle-naive); tile score = mean over S of the z-scored tile-mean log-normalised expression (R8's
  readout). Readout: Spearman's rho between closeness and tile score over the index tiles (the breast exemplar tile left
  out, as in R8). "absent" is carried over from results/pan_cancer/programs.csv (Spindle-naive mean delta <= 0).
- SKIN CELLS (seed 0): a cell's programme score = mean over S of its log-normalised expression, each gene z-scored over
  the section's cells (so no one abundant gene dominates, as the tile score); also each index tile's tile score, each
  cell's index tile (-1 = held-out or dropped tile), each programme's query Q (the covariance searched for) and the
  index tiles Spindle returns for Q (SpindlePartial.search_vectors, default g; index-tile positions, best first; added
  2 Nov, user: the maps keep the exact d_B^S of every tile and mark the tiles Spindle returns).
- SKIN GENE-SET QUERIES (seed 0): gene_set_queries below; Fig 4C uses |S| = 16 (gene_set.BAND_SIZE, the size of the
  accuracy benchmark's typical query); queries with no tile returned are reported and left out of the figure.
- EVERY SECTION (seed 0; Supplementary Fig S7, added 1 Nov): the skin maps' arrays (SKIN CELLS, with each index tile's
  exact d_B^S to each programme query, centre and side, as pan_cancer.gradient) on every indexed section whose index
  genes hold every programme gene (map_keys: all but lymph_node_5k, which keeps its 400 most variable genes); the skin
  gene-set queries on every indexed section.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import gene_set as gs
import pan_cancer as pc
import sections as sx
import spindle_dev as sd

KEY = "skin_melanoma"


def match(S, seed, E, Z):
    """Spearman's rho of closeness and tile score per programme in one (section, seed)."""
    c, covs = S["c"], S["covs"]
    mu = sx.tile_means(S, c.train)
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    rows = []
    for _, e in E.iterrows():
        prog = e.program
        idx, _ = pc.program_idx(S["genes"], prog)
        R, q = Z[f"R_{prog}"], Z[f"q_{prog}"]
        sig = pc.query_scale(pc.sub_covs(covs, c.train, idx), q)
        d = c.naive_partial.distances(sig[:, None] * R * sig[None, :], idx)
        valid = np.ones(len(c.train), bool)
        if S["key"] == pc.SOURCE:
            valid[c.train == int(e.tile)] = False
        rho = spearmanr(-d[valid], z[valid][:, idx].mean(1)).statistic
        rows.append({"key": S["key"], "seed": seed, "program": prog, "n_genes": len(idx), "n_index": int(valid.sum()),
                     "rho": float(rho), "tag": S["stamp"]["tag"]})
    return pd.DataFrame(rows)


def summary(g, programs):
    """Mean and s.d. of rho over seeds per (section, programme), with R8's absent flag."""
    out = g.groupby(["key", "program"], sort=False).rho.agg(rho="mean", rho_sd="std").reset_index()
    a = programs[programs.method == "naive_partial"].set_index(["key", "program"]).absent
    out["absent"] = [bool(a.loc[(k, p)]) for k, p in zip(out.key, out.program)]
    out["tag"] = g.tag.iloc[0]
    return out


def skin_cells(S, E, Z):
    """Arrays of skin_cells.npz: cells, each programme's cell score, the index tiles' tile scores (index-tile order, as
    seed0_gradient.npz), each cell's index-tile position (-1 = none), each programme's query Q and genes, and the
    index-tile positions Spindle returns for Q (spindle_<programme>; needs the build's spindle_partial); the index
    tiles' boxes (bbox: x0, y0, x1, y1) to outline them."""
    c = S["c"]
    cell_tile = np.full(len(S["xy"]), -1)
    for pos, t in enumerate(c.train):
        cell_tile[S["tiles"][t].idx] = pos
    mu = sx.tile_means(S, c.train)
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    out = {"xy": S["xy"], "programs": E.program.to_numpy(), "tiles": c.train, "cell_tile": cell_tile,
           "bbox": np.array([S["tiles"][t].bbox for t in c.train], dtype=np.float64)}
    for prog in E.program:
        idx, found = pc.program_idx(S["genes"], prog)
        sig = pc.query_scale(pc.sub_covs(S["covs"], c.train, idx), Z[f"q_{prog}"])
        out[f"Q_{prog}"] = sig[:, None] * Z[f"R_{prog}"] * sig[None, :]
        rows, _, _ = c.spindle_partial.search_vectors(c.spindle_partial.query_vectors(out[f"Q_{prog}"], idx))
        out[f"spindle_{prog}"] = np.asarray(rows, dtype=np.int64)
        print(f"{S['key']} {prog}: Spindle returned {len(rows)} of {len(c.train)} index tiles", flush=True)
        out[f"genes_{prog}"] = np.array(found)
        L = S["L"][:, idx]
        out[f"score_{prog}"] = ((L - L.mean(0)) / (L.std(0) + 1e-9)).mean(1).astype(np.float32)
        out[f"tile_score_{prog}"] = z[:, idx].mean(1)
    return out



def gene_set_queries(S, seed):
    """Table of every skin gene-set query (the accuracy workload, gene_set.workload: per held-out tile and size, a run of
    consecutive genes in one block of its home niche; the query is the tile's shrunk covariance cut to S): composition
    difference (TV over the 10x clusters, as queries.tile_queries) of the query tile to Spindle's first 10 tiles (fewer
    if it returns fewer) and to a random 10 (mean of N_RANDOM draws)."""
    c = S["c"]
    n = len(c.train)
    cnt_idx, cnt_q = sx.comp_counts(S, c.train), sx.comp_counts(S, c.test)
    rng = np.random.default_rng(seed)
    rows = []
    for j, home, L, b, a, genes in gs.workload(c.layout, c.D, seed):
        if cnt_q[j].sum() == 0:
            continue
        q = sx.shares(cnt_q[j])
        cov_s, idx = sd.tile_query(S["covs"][c.test[j]]["cov"], genes)
        found, _, _ = c.spindle_partial.search_vectors(c.spindle_partial.query_vectors(cov_s, idx))
        top = np.asarray(found[:sx.TOP], dtype=int)
        rand = [sx.tv(q, sx.shares(cnt_idx[rng.choice(n, sx.TOP, replace=False)])) for _ in range(sx.N_RANDOM)]
        rows.append({"key": S["key"], "seed": seed, "query": j, "tile": int(c.test[j]), "size": L,
                     "n_cells": int(cnt_q[j].sum()), "n_found": len(found),
                     "tv_spindle": sx.tv(q, sx.shares(cnt_idx[top])) if len(top) else np.nan,
                     "tv_random": float(np.mean(rand)), "tag": S["stamp"]["tag"]})
    return pd.DataFrame(rows)


def map_keys(keys):
    """The sections of `keys` whose index genes hold every programme gene (EVERY SECTION rule)."""
    import data

    out = []
    for key in keys:
        genes = [str(g) for g in data.load_cache(key, data.EXAMPLE_SEED, parts=()).genes]
        if all(set(pc.PAN_CANCER_MODULES[p]["genes"]) <= set(genes) for p in pc.PROGRAMS):
            out.append(key)
    return out


def section_maps(S, E, Z):
    """Arrays of per_dataset/<key>/seed0_maps.npz: skin_cells' arrays and pan_cancer.gradient's (exact d_B^S of every
    index tile to each programme query, tile centres and sides) for one section."""
    return {**skin_cells(S, E, Z), **pc.gradient(S, E, Z)}
