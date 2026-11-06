"""One indexed section with its cached build and per-cell inputs, and the statistics of the query results (Figs 3-4),
from the staged inputs of R0 (no h5ad, no covariance recomputed). Ported from experiments/fig6_breast/common6.py and
experiments/fig7_skin/common7.py (in git history), themselves from the 17 Oct block-programme code, the poster v2 biology
and the gene-set pilot.

    import sections as sx
    S = sx.load("breast_cancer", 0, parts=("layout", "spindle", "naive_partial", "queries", "D"))
    S = sx.load("breast_cancer", "full")       # breast: the layout of every tile (no tile held out), built here

Cell labels: breast, the section's annotation (``Cluster``; "Unlabeled" cells are dropped by sd.load_adata), 19 types,
used as they are in the niche enrichment (Fig 6) and grouped into the 8 classes of CLASSES for tile compositions (cells,
not tiles); every other section, the 10x graph-based clusters (cells without a cluster are left out of compositions).
Expression: raw counts on the index genes,
log-normalised per cell, log1p(x / total * median total).
"""

from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata

import data
import spindle_dev as sd
from signatures import TISSUE, TISSUE_MODULES
from spindle_dev.metrics import as_cov

BREAST = "breast_cancer"
TOP = 10
N_RANDOM = 200  # random 10-tile draws for the composition null
MIN_LIST = 2  # a gene list is searched if at least this many of its genes are index genes

CLASSES = [
    ("Invasive tumour", "#CC3311", ["Invasive_Tumor", "Prolif_Invasive_Tumor", "T_Cell_&_Tumor_Hybrid"]),
    ("DCIS", "#EE7733", ["DCIS_1", "DCIS_2"]),
    ("Myoepithelial", "#882255", ["Myoepi_ACTA2+", "Myoepi_KRT15+"]),
    ("Stromal", "#DDCC77", ["Stromal", "Stromal_&_T_Cell_Hybrid"]),
    ("Endothelial / perivascular", "#44AA99", ["Endothelial", "Perivascular-Like"]),
    ("Myeloid", "#0077BB", ["Macrophages_1", "Macrophages_2", "IRF7+_DCs", "LAMP3+_DCs", "Mast_Cells"]),
    ("T cells", "#33BBEE", ["CD4+_T_Cells", "CD8+_T_Cells"]),
    ("B cells", "#117733", ["B_Cells"]),
]
CLASS_NAMES = [c[0] for c in CLASSES]


def load(key, seed, parts=("layout",)):
    """Section, build and per-cell inputs of one (key, seed) as a dict: key, tiles, covs (memory-mapped), c (the cached
    build, data.load_cache), genes, X, L, types (cell labels, "" = none), comp / comp_names (composition label per cell,
    -1 = none), cls (= comp; breast only), xy, n_obs, stamp. seed = "full" (breast): the layout of every tile."""
    tiles, covs = data.staged_tiles(key), data.staged_covs(key)
    cells = data.staged_cells(key)
    if seed == "full":
        genes = data.load_cache(key, data.EXAMPLE_SEED, parts=()).genes
        layout = sd.build_layout(tiles, [covs[i] for i in range(len(covs))], [str(g) for g in genes], cells["n_obs"])
        c = SimpleNamespace(key=key, seed=seed, train=np.arange(len(tiles)), test=np.array([], dtype=int),
                            genes=genes, layout=layout)
    else:
        c = data.load_cache(key, seed, parts=parts, covs=covs)
    X, tot = cells["X"], cells["tot"]
    L = np.log1p(X / np.maximum(tot, 1)[:, None] * np.median(tot)).astype(np.float32)
    types = cells["labels"]
    S = {"key": key, "tiles": tiles, "covs": covs, "c": c, "genes": [str(g) for g in c.genes], "X": X, "L": L,
         "types": types, "xy": cells["xy"], "n_obs": cells["n_obs"], "stamp": data.stamp()}
    if key == BREAST:
        to_class = {t: i for i, (_, _, members) in enumerate(CLASSES) for t in members}
        comp = np.array([to_class.get(t, -1) for t in types])
        if (comp < 0).any():
            raise RuntimeError(f"unmapped cell types: {sorted(set(types[comp < 0]))}")
        S.update(comp=comp, comp_names=list(CLASS_NAMES), cls=comp)
    else:
        clusters = sorted({str(t) for t in types if t}, key=int)
        pos = {t: i for i, t in enumerate(clusters)}
        S.update(comp=np.array([pos.get(t, -1) for t in types]), comp_names=clusters)
    return S


# ---------------------------------------------------------------- compositions and expression (common6, common7)
def class_counts(S, tile_ids):
    """Cells of each breast class in each tile, (len(tile_ids), 8)."""
    return np.stack([np.bincount(S["cls"][S["tiles"][t].idx], minlength=len(CLASSES)) for t in tile_ids]
                    ).astype(float)


def comp_counts(S, tile_ids):
    """Labelled cells of each composition label in each tile, (len(tile_ids), n_labels)."""
    n = len(S["comp_names"])
    out = []
    for t in tile_ids:
        v = S["comp"][S["tiles"][t].idx]
        out.append(np.bincount(v[v >= 0], minlength=n))
    return np.stack(out).astype(float)


def class_shares(counts):
    """Pooled class shares of the cells of a set of tiles (rows of class_counts) (common6.shares)."""
    c = np.asarray(counts).reshape(-1, len(CLASSES)).sum(0)
    return c / c.sum()


def shares(counts):
    """Pooled label shares of a set of tiles (rows of comp_counts) or of one count vector (common7.shares)."""
    c = np.asarray(counts).sum(0) if np.ndim(counts) > 1 else np.asarray(counts)
    return c / max(c.sum(), 1)


def tile_means(S, tile_ids):
    """Mean log-normalised expression of every index gene per tile, (len(tile_ids), p)."""
    return np.stack([S["L"][S["tiles"][t].idx].mean(0) for t in tile_ids])


def niche_corr(covs):
    """Correlation of the mean of a niche's (shrunk) tile covariances, global gene order."""
    M = np.mean([as_cov(c) for c in covs], axis=0)
    d = np.sqrt(np.diag(M))
    return M / np.outer(d, d)


def mean_offdiag_corr(X):
    """Mean off-diagonal correlation of the columns of X over its rows (cells); zero-variance columns dropped."""
    Xc = X - X.mean(0)
    s = np.sqrt((Xc ** 2).sum(0))
    keep = s > 0
    if keep.sum() < 2:
        return np.nan
    Z = Xc[:, keep] / s[keep]
    m = Z.shape[1]
    return ((Z.T @ Z).sum() - m) / (m * (m - 1))


def cliffs_delta(x, y):
    r = rankdata(np.concatenate([x, y]))
    u = r[:len(x)].sum() - len(x) * (len(x) + 1) / 2
    return 2 * u / (len(x) * len(y)) - 1


def tv(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def jaccard(a, b):
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / len(a | b) if a | b else np.nan


def bh(p):
    p = np.asarray(p, dtype=float)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


# ---------------------------------------------------------------- gene lists (common7)
def gene_lists(key, genes):
    """[(name, idx, found, missing)] of the tissue's curated signatures with >= MIN_LIST index genes."""
    out = []
    for name, info in TISSUE_MODULES[TISSUE[key]].items():
        idx, found, missing = sd.gene_indices(genes, info["genes"])
        if len(idx) >= MIN_LIST:
            out.append((name, np.asarray(idx), list(found), list(missing)))
    return out

