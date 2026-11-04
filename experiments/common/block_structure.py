"""How small the entries are that the block-diagonal layout leaves out (supplementary Fig S1), from the staged shrunk
covariances and the seed-0 layout of each dataset (R10). Fig S1B is the per-tile version (``tile_means``, after
HS_notebooks/dag_vs_true_le_distance.ipynb, Part G, 28 Sep; user, 3 Nov): each index tile's mean |r| over its
within-block and its between-block gene pairs, one value per tile.

Every index tile's shrunk covariance is turned into a correlation matrix and put in the gene order of its niche; its
off-diagonal gene pairs are split into within-block pairs (both genes in one block of the niche) and between-block
pairs (the entries that d_B ignores). Per entry only (user, 3 Nov): no share of the squared correlation. A gene with no
counts in a tile has no covariance with any other gene (exactly 0; shrinkage only adds to the diagonal), so its
correlations are undefined: pairs with such a gene are left out (they were 26 % of the breast pairs, all at r = 0;
H. Sarkar, 3 Nov).

    import block_structure as bs
    hist, summ = bs.dataset("breast_cancer")       # |r| histograms (EDGES), per-dataset summary row (from the cache)
    hist, summ = bs.tile_hist(layout, covs, key)   # the same from any layout and its index covariances
    means = bs.tile_means(layout, covs, key)       # per index tile: mean |r| within and between blocks (Fig S1B)
    ex = bs.niche_example(layout, covs)            # the largest niche's mean correlation matrix, niche order
"""

import numpy as np
import pandas as pd

import data
from spindle_dev.metrics import as_cov, spd_to_correlation

EDGES = np.linspace(0.0, 1.0, 1001)  # |r| bins of width 0.001
QUANTILES = (0.5, 0.9, 0.99)
THRESHOLDS = (0.1, 0.3)


def _masks(perm_len, runs):
    within = np.zeros((perm_len, perm_len), bool)
    for s, e in runs:
        within[s:e, s:e] = True
    upper = np.triu(np.ones((perm_len, perm_len), bool), 1)
    return within & upper, ~within & upper


def _quantiles(counts):
    """Quantiles of |r| from a histogram on EDGES (upper edge of the bin that reaches q)."""
    cdf = np.cumsum(counts) / counts.sum()
    return [EDGES[1:][np.searchsorted(cdf, q)] for q in QUANTILES]


def tile_hist(layout, train_covs, key, seed=data.EXAMPLE_SEED):
    """(histogram rows: key, part, bin_lo, bin_hi, count; summary row per part) over the index tiles of a layout
    (``train_covs[r]``: the shrunk covariance of the layout's tile r), pairs with a gene without counts in the tile left
    out. Shared by R10 (from the cache) and S1_block_from_h5ad (from the h5ad)."""
    L = layout
    labels = np.asarray(L.labels).astype(int)
    masks = {k: _masks(len(L.perm_list[k]), L.block_dict[k]) for k in sorted(set(labels.tolist()))}
    counts = {"within": np.zeros(len(EDGES) - 1, np.int64), "between": np.zeros(len(EDGES) - 1, np.int64)}
    above = {part: {t: 0 for t in THRESHOLDS} for part in counts}
    silent_pairs = {part: 0 for part in counts}
    for r in range(len(labels)):
        k = labels[r]
        perm = np.asarray(L.perm_list[k])
        cov = as_cov(train_covs[r])[np.ix_(perm, perm)]
        expressed = np.any((cov - np.diag(np.diag(cov))) != 0, axis=1)  # a gene with counts covaries with something
        defined = expressed[:, None] & expressed[None, :]
        a = np.abs(spd_to_correlation(cov))
        for part, m in zip(("within", "between"), masks[k]):
            silent_pairs[part] += int((m & ~defined).sum())
            v = np.minimum(a[m & defined], 1.0)
            counts[part] += np.histogram(v, EDGES)[0]
            for th in THRESHOLDS:
                above[part][th] += int((v > th).sum())
    hist, summ = [], []
    for part, h in counts.items():
        hist.append(pd.DataFrame({"key": key, "part": part, "bin_lo": EDGES[:-1], "bin_hi": EDGES[1:], "count": h}))
        n = int(h.sum())
        row = {"key": key, "seed": seed, "part": part, "n_tiles": len(labels), "n_niches": len(masks), "pairs": n,
               "pairs_per_tile": n / len(labels), "pairs_left_out": silent_pairs[part],
               "frac_left_out": silent_pairs[part] / (n + silent_pairs[part])}
        row.update({f"q{int(100 * q)}": v for q, v in zip(QUANTILES, _quantiles(h))})
        row.update({f"frac_above_{th}": above[part][th] / n for th in THRESHOLDS})
        summ.append(row)
    return pd.concat(hist, ignore_index=True), pd.DataFrame(summ)


def tile_means(layout, train_covs, key):
    """One row per index tile of a layout: its niche, and the mean |r| over its within-block and its between-block gene
    pairs (in its niche's gene order), pairs with a gene without counts in the tile left out, with the pair counts."""
    L = layout
    labels = np.asarray(L.labels).astype(int)
    masks = {k: _masks(len(L.perm_list[k]), L.block_dict[k]) for k in sorted(set(labels.tolist()))}
    rows = []
    for r in range(len(labels)):
        k = labels[r]
        perm = np.asarray(L.perm_list[k])
        cov = as_cov(train_covs[r])[np.ix_(perm, perm)]
        expressed = np.any((cov - np.diag(np.diag(cov))) != 0, axis=1)
        defined = expressed[:, None] & expressed[None, :]
        a = np.abs(spd_to_correlation(cov))
        row = {"key": key, "row": r, "niche": int(k), "genes_expressed": int(expressed.sum())}
        for part, m in zip(("within", "between"), masks[k]):
            v = a[m & defined]
            row[f"mean_{part}"] = float(v.mean()) if len(v) else np.nan
            row[f"pairs_{part}"] = len(v)
        rows.append(row)
    return pd.DataFrame(rows)


def dataset_means(key, seed=data.EXAMPLE_SEED):
    """tile_means of the cached seed-``seed`` layout of ``key`` with the staged covariances (R10)."""
    covs = data.staged_covs(key)
    c = data.load_cache(key, seed, parts=("layout",))
    return tile_means(c.layout, [covs[t] for t in c.train], key).assign(seed=seed, tile=lambda d: c.train[d.row])


def niche_example(layout, train_covs):
    """The largest niche's mean correlation matrix (correlation of the mean shrunk covariance of its index tiles, as
    index_cache.stage's niche_means.npz) in its gene order, with its block runs."""
    L = layout
    labels = np.asarray(L.labels).astype(int)
    sizes = np.bincount(labels)
    k = int(sizes.argmax())
    M = np.mean([as_cov(train_covs[r]) for r in np.flatnonzero(labels == k)], axis=0)
    perm = np.asarray(L.perm_list[k])
    C = spd_to_correlation(M)[np.ix_(perm, perm)]
    return {"corr": C.astype(np.float32), "runs": np.asarray(L.block_dict[k], np.int64), "niche": k,
            "n_tiles": int(sizes[k])}


def dataset(key, seed=data.EXAMPLE_SEED):
    """tile_hist of the cached seed-``seed`` layout of ``key`` with the staged covariances (R10)."""
    covs = data.staged_covs(key)
    c = data.load_cache(key, seed, parts=("layout",))
    return tile_hist(c.layout, [covs[t] for t in c.train], key, seed)


def example(key="breast_cancer", seed=data.EXAMPLE_SEED):
    """The index tile whose |r| between blocks is is the median of the largest niche: its
    correlation matrix in niche order, the block runs and its tile id."""
    covs = data.staged_covs(key)
    c = data.load_cache(key, seed, parts=("layout",))
    L = c.layout
    labels = np.asarray(L.labels).astype(int)
    k = int(np.bincount(labels).argmax())
    perm = np.asarray(L.perm_list[k])
    _, between = _masks(len(perm), L.block_dict[k])
    rows = np.flatnonzero(labels == k)
    med = np.array([np.median(np.abs(spd_to_correlation(as_cov(covs[c.train[r]]))[np.ix_(perm, perm)][between]))
                    for r in rows])
    r = rows[np.argmin(np.abs(med - np.median(med)))]
    C = spd_to_correlation(as_cov(covs[c.train[r]]))[np.ix_(perm, perm)]
    return {"corr": C.astype(np.float32), "runs": np.asarray(L.block_dict[k], np.int64), "niche": k,
            "tile": int(c.train[r]), "genes": np.asarray(c.genes)[perm]}


def niche_means(key, seed=data.EXAMPLE_SEED):
    """|r| histograms over each niche's mean correlation (the correlation of its mean shrunk covariance, staged as
    niche_means.npz: the matrices the gene order and blocks are chosen from), full matrix and inside blocks, pooled
    over the niches; and the largest niche's matrix in its gene order with its block runs (the example)."""
    z = np.load(data.cache_dir(key, seed) / "niche_means.npz")
    L = data.load_cache(key, seed, parts=("layout",)).layout
    sizes = np.bincount(np.asarray(L.labels).astype(int))
    counts = {"full": np.zeros(len(EDGES) - 1, np.int64), "blocks": np.zeros(len(EDGES) - 1, np.int64)}
    ex = None
    for k, M in zip(z["niches"], z["M"]):
        k = int(k)
        perm = np.asarray(L.perm_list[k])
        C = spd_to_correlation(np.asarray(M, dtype=np.float64))[np.ix_(perm, perm)]
        within, between = _masks(len(perm), L.block_dict[k])
        a = np.minimum(np.abs(C), 1.0)
        counts["full"] += np.histogram(a[within | between], EDGES)[0]
        counts["blocks"] += np.histogram(a[within], EDGES)[0]
        if k == int(sizes.argmax()):
            ex = {"corr": C.astype(np.float32), "runs": np.asarray(L.block_dict[k], np.int64), "niche": k,
                  "n_tiles": int(sizes[k])}
    hist = pd.concat([pd.DataFrame({"key": key, "part": part, "bin_lo": EDGES[:-1], "bin_hi": EDGES[1:], "count": h})
                      for part, h in counts.items()], ignore_index=True)
    return hist, ex


# ---------------------------------------------------------------------------------------------- Figs S6, S7 (4 Nov)
def consensus_example(key="breast_cancer", seed=data.EXAMPLE_SEED, n_tiles=3, rng_seed=0):
    """Fig S6 (user, 4 Nov: dendrograms, no knee): the consensus tree of the largest niche of the seed-0 layout. Every
    index tile's gene dendrogram (average linkage on 1 - |r|, ``layout.spd_to_ultrametric``) is recomputed from the
    staged covariances; their mean cophenetic matrix agrees with the layout's cached one to < 1e-3 (asserted; the
    staged covariances are rounded, which can swap near-tied merges), and the consensus tree is the layout's own
    (cached mean cophenetic matrix -> average linkage; its leaf order == the gene order, asserted). Returns
    the linkages and leaf orders of ``n_tiles`` random tiles of the niche, the consensus linkage, the gene order, the
    blocks, whether each block is a subtree (clade) of the consensus tree, two marker genes per block (largest mean |r|
    to the rest of the block), and the niche's mean correlation in its gene order."""
    from scipy.cluster.hierarchy import dendrogram, linkage, to_tree
    from scipy.spatial.distance import squareform
    from spindle_dev import layout as lay

    covs = data.staged_covs(key)
    c = data.load_cache(key, seed, parts=("layout",))
    L = c.layout
    labels = np.asarray(L.labels).astype(int)
    k = int(np.bincount(labels).argmax())
    rows = np.flatnonzero(labels == k)
    U_sum, tiles = None, {}
    pick = set(np.random.default_rng(rng_seed).choice(rows, n_tiles, replace=False).tolist())
    for r in rows:
        U, Zm = lay.spd_to_ultrametric(as_cov(covs[c.train[r]]))
        U_sum = U.copy() if U_sum is None else U_sum + U
        if r in pick:
            tiles[int(r)] = Zm
    assert np.abs(U_sum / len(rows) - np.asarray(L.U_mean_list[k])).max() < 1e-3
    Z = linkage(squareform(np.asarray(L.U_mean_list[k]), checks=False), method="average")
    perm = np.asarray(dendrogram(Z, no_plot=True)["leaves"])
    assert np.array_equal(perm, np.asarray(L.perm_list[k])), "consensus order differs from the layout"
    clades, stack = set(), [to_tree(Z)]
    while stack:
        nd = stack.pop()
        clades.add(frozenset(nd.pre_order()))
        if not nd.is_leaf():
            stack += [nd.left, nd.right]
    blocks = np.asarray(L.block_dict[k], np.int64)
    R = np.asarray(L.R_mean_list[k])[np.ix_(perm, perm)]
    names = np.asarray(L.metadata["genes"])[perm].astype(str)
    is_clade, markers = [], []
    for a, e in blocks:
        is_clade.append(frozenset(perm[a:e].tolist()) in clades)
        B = np.abs(R[a:e, a:e]); np.fill_diagonal(B, 0.0)
        markers.append(" ".join(names[a:e][np.argsort(-B.mean(1))[:2]]))
    order = sorted(tiles)
    return {"tile_rows": np.array(order), "tile_Z": np.stack([tiles[r] for r in order]),
            "tile_leaves": np.stack([np.asarray(dendrogram(tiles[r], no_plot=True)["leaves"]) for r in order]),
            "Z": Z, "perm": perm, "blocks": blocks, "is_clade": np.array(is_clade), "markers": np.array(markers),
            "names": names, "corr": R.astype(np.float32), "niche": k, "n_tiles": len(rows)}


DYADIC_POSITIONS = (*range(2, 9), 11, 12, *range(16, 20))  # an illustrative gene set inside one block (Fig S7)


def dyadic_example(key="breast_cancer", seed=data.EXAMPLE_SEED, block=None, positions=DYADIC_POSITIONS):
    """Fig S7: the dyadic decomposition of an illustrative gene set inside block ``block`` of the largest niche (default:
    the block of 20-32 genes with the largest mean |r| between its genes in the niche's mean correlation matrix)
    (seed-0 Spindle-Partial index). Every aligned dyadic interval of the block with its node count K_I, the gene set's
    pieces (SpindlePartial niche ``pieces``), the niche's mean correlation on the block, and every index tile's node in
    every piece (``paths``: tiles x pieces, the node index)."""
    c = data.load_cache(key, seed, parts=("layout", "spindle_partial"))
    L, SP = c.layout, c.spindle_partial
    labels = np.asarray(L.labels).astype(int)
    k = int(np.bincount(labels).argmax())
    N = SP.niches[k]
    perm = np.asarray(L.perm_list[k])
    if block is None:
        Rk = np.asarray(L.R_mean_list[k])[np.ix_(perm, perm)]
        score = [np.abs(Rk[a:e, a:e][~np.eye(e - a, dtype=bool)]).mean() if 20 <= e - a <= 32 else -1.0
                 for a, e in L.block_dict[k]]
        block = int(np.argmax(score))
    a, e = L.block_dict[k][block]
    block_genes = perm[a:e]
    assert np.all(N.gene_block[block_genes] == block) and np.array_equal(N.gene_pos[block_genes], np.arange(e - a))
    genes = block_genes[list(positions)]
    pieces = N.pieces(genes)
    intervals = np.array([(s, j, len(N.centres[i])) for i, (b, s, j) in enumerate(N.keys) if b == block], np.int64)
    paths = np.stack([np.asarray(N.node_index[N.pos[p]], np.uint8) for p in pieces], axis=1)
    R = np.asarray(L.R_mean_list[k])[np.ix_(block_genes, block_genes)]
    return {"names": np.asarray(L.metadata["genes"])[block_genes].astype(str), "positions": np.asarray(positions),
            "pieces": np.array([(s, j) for _, s, j in pieces], np.int64),
            "piece_K": np.array([len(N.centres[N.pos[p]]) for p in pieces], np.int64),
            "intervals": intervals, "paths": paths, "corr": R.astype(np.float32), "niche": k, "block": block,
            "n_tiles": int((labels == k).sum())}
