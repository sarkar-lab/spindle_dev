"""Shared by the Fig 6 compute scripts (no plotting; safe in sbatch): the breast section, its cached builds, cell
types, expression and the statistics of the biology panels. Ported from the 17 Oct block-programme code
(benchmarks/biology/block_programs.py, in git history), the poster v2 biology (poster/scripts/c_bio_*.py)
and the gene-set pilot (removed 27 Oct; in git history) onto the method-v1 public API.

Cell types: the section's annotation (``Cluster``; "Unlabeled" cells are dropped by sd.load_adata), 19 types, used as
they are in the niche enrichment (Fig 6A) and grouped into the 8 classes of CLASSES (as poster/scripts/f_layout.py)
for tile compositions (cells, not tiles). Expression: raw counts on the index genes, log-normalised per cell,
log1p(x / total * median total).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402  (puts src/ on sys.path)
import spindle_dev as sd  # noqa: E402,F401
from spindle_dev.metrics import as_cov  # noqa: E402

KEY = "breast_cancer"
OUT = data.RESULTS / "fig6_breast"
TOP = 10
N_RANDOM = 200  # random 10-tile draws for the composition null
GMT = {"Hallmark": Path.home() / ".cache" / "gseapy" / "Enrichr.MSigDB_Hallmark_2020.gmt",
       "GO_BP": Path.home() / ".cache" / "gseapy" / "Enrichr.GO_Biological_Process_2023.gmt"}

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


def load(seed, parts=("layout", "spindle", "lite", "partial", "queries", "D")):
    """Section, build and per-cell inputs of one seed, as a dict. seed = "full": the layout of every tile of the
    section (no tile held out; built here with sd.build_layout, as c_index_cache.py does for a seed's index tiles)."""
    adata, tiles, covs, genes = data.prepare(KEY)
    if seed == "full":
        layout = sd.build_layout(tiles, covs, genes, adata.n_obs)
        c = SimpleNamespace(key=KEY, seed=seed, train=np.arange(len(tiles)), test=np.array([], dtype=int),
                            genes=np.asarray(genes).astype(str), layout=layout)
    else:
        c = data.load_cache(KEY, seed, parts=parts, covs=covs)
    names = [str(g) for g in c.genes]
    gi = adata.var_names.get_indexer(names)
    if (gi < 0).any():
        raise RuntimeError("index genes missing from the section")
    X = adata.X[:, gi]
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    tot = np.asarray(adata.X.sum(1)).ravel()
    L = np.log1p(X / np.maximum(tot, 1)[:, None] * np.median(tot)).astype(np.float32)
    types = np.asarray(adata.obs["Cluster"].astype(str))
    to_class = {t: i for i, (_, _, members) in enumerate(CLASSES) for t in members}
    cls = np.array([to_class.get(t, -1) for t in types])
    if (cls < 0).any():
        raise RuntimeError(f"unmapped cell types: {sorted(set(types[cls < 0]))}")
    return {"adata": adata, "tiles": tiles, "covs": covs, "c": c, "genes": names, "X": X, "L": L,
            "types": types, "cls": cls, "xy": np.asarray(adata.obsm["spatial"], dtype=np.float32),
            "stamp": data.stamp()}


def class_counts(S, tile_ids):
    """Cells of each class in each tile, (len(tile_ids), 8)."""
    return np.stack([np.bincount(S["cls"][S["tiles"][t].idx], minlength=len(CLASSES)) for t in tile_ids]
                    ).astype(float)


def shares(counts):
    """Pooled class shares of the cells of a set of tiles (rows of class_counts)."""
    c = np.asarray(counts).reshape(-1, len(CLASSES)).sum(0)
    return c / c.sum()


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
