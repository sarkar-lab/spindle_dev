"""The block distance d_B against the whole-matrix log-Euclidean distance d_W, niche by niche (supplementary Fig S9).

How much does leaving out the between-block entries change the distance? For random pairs of index tiles in the same
niche (so both tiles share the niche's gene order and blocks), seed-0 layout, shrunk covariances:

* d_W(i, j) = ||log C_i - log C_j||_F / sqrt(p), the logarithm of the whole p x p matrix;
* d_B(i, j) = sqrt(sum_b ||log C_i[b] - log C_j[b]||_F^2) / sqrt(p), over the niche's blocks (Spindle-naive's scan).

Both come from vectors already in the cache, on the same scale (upper triangle, off-diagonals x sqrt(2), / sqrt(p)):
the staged whole-matrix vectors (``whole.npy``) and Spindle-naive's stored block vectors (``naive.pkl``). ``check``
recomputes a few pairs from the covariances.

    import block_distance as bd
    pairs, niches = bd.pairs("breast_cancer")    # one row per tile pair (d_whole, d_block); one row per niche
    bd.check("breast_cancer", pairs)             # a few pairs recomputed from the staged covariances
    pairs, info = bd.dataset_pairs("lung_cancer")  # the same over a whole dataset: pairs within any of its niches
"""

import numpy as np
import pandas as pd

import data
from spindle_dev.metrics import log_spd

N_NICHES = 3      # the largest niches of the seed-0 layout
N_PAIRS = 1000    # distinct unordered pairs per niche
RNG_SEED = 0      # one generator, drawn niche after niche (largest first)


def pairs(key, seed=data.EXAMPLE_SEED, n_niches=N_NICHES, n_pairs=N_PAIRS, rng_seed=RNG_SEED):
    """(pairs, niches): ``n_pairs`` distinct random pairs of index tiles in each of the ``n_niches`` largest niches,
    with d_W (``d_whole``) and d_B (``d_block``); and per niche its tiles and blocks."""
    c = data.load_cache(key, seed, parts=("naive",))
    W = data.staged_whole(key)  # (n_tiles, p(p+1)/2) float32, prepare() tile order (memory-mapped)
    naive = c.naive
    sizes = {k: len(rows) for k, (rows, _) in naive.layouts.items()}
    largest = sorted(sizes, key=lambda k: -sizes[k])[:n_niches]
    rng = np.random.default_rng(rng_seed)
    out, info = [], []
    for k in largest:
        rows, vec = naive.layouts[k]
        n = len(rows)
        iu = np.triu_indices(n, 1)
        pick = rng.choice(len(iu[0]), n_pairs, replace=False)
        out.append(_distances(c, W, k, iu[0][pick], iu[1][pick]))
        info.append({"niche": k, "tiles": n, "blocks": len(vec.runs), "p": len(c.genes)})
    return pd.concat(out, ignore_index=True), pd.DataFrame(info)


def dataset_pairs(key, seed=data.EXAMPLE_SEED, n_pairs=N_PAIRS, rng_seed=RNG_SEED):
    """(pairs, info): ``n_pairs`` distinct random pairs of index tiles of a dataset, drawn uniformly from every pair
    of tiles in the same niche (d_B exists only within a niche: both tiles need its gene order and blocks), so niches
    contribute in proportion to their pairs; and the dataset's index tiles, niches and genes."""
    c = data.load_cache(key, seed, parts=("naive",))
    W = data.staged_whole(key)
    naive = c.naive
    niches = sorted(naive.layouts)
    sizes = np.array([len(naive.layouts[k][0]) for k in niches])
    counts = sizes * (sizes - 1) // 2                  # same-niche pairs per niche
    starts = np.r_[0, np.cumsum(counts)]
    rng = np.random.default_rng(rng_seed)
    pick = np.sort(rng.choice(starts[-1], n_pairs, replace=False))
    which = np.searchsorted(starts, pick, side="right") - 1
    out = []
    for i, k in enumerate(niches):
        local = pick[which == i] - starts[i]
        if len(local) == 0:
            continue
        iu = np.triu_indices(sizes[i], 1)
        out.append(_distances(c, W, k, iu[0][local], iu[1][local]))
    info = {"tiles": int(sizes.sum()), "niches": len(niches), "p": len(c.genes)}
    return pd.concat(out, ignore_index=True), info


def _distances(c, W, k, a, b):
    """d_W and d_B of the pairs (a, b) of rows of niche k (rows into the niche; naive.X[k] has the same order)."""
    rows, _ = c.naive.layouts[k]
    X = c.naive.X[k].astype(np.float64)
    tile = c.train[rows]                               # prepare() tile index of each niche row
    d_block = np.linalg.norm(X[a] - X[b], axis=1)
    d_whole = np.linalg.norm(np.asarray(W[tile[a]], np.float64) - np.asarray(W[tile[b]], np.float64), axis=1)
    return pd.DataFrame({"niche": k, "tile_a": tile[a], "tile_b": tile[b], "d_whole": d_whole, "d_block": d_block})


def check(key, pairs, seed=data.EXAMPLE_SEED, per_niche=2, rtol=1e-3):
    """Recompute the first ``per_niche`` pairs of each niche from the staged covariances (whole log and block logs
    directly); raises if either distance is off by more than ``rtol``."""
    c = data.load_cache(key, seed, parts=("naive",))
    covs = data.staged_covs(key)
    p = len(c.genes)
    for k, g in pairs.groupby("niche", sort=False):
        perm = np.asarray(c.naive.layouts[k][1].perm)
        runs = c.naive.layouts[k][1].runs
        for r in g.head(per_niche).itertuples():
            Ca = np.asarray(covs[int(r.tile_a)]["cov"], float)
            Cb = np.asarray(covs[int(r.tile_b)]["cov"], float)
            dw = np.linalg.norm(log_spd(Ca) - log_spd(Cb)) / np.sqrt(p)
            Pa, Pb = Ca[np.ix_(perm, perm)], Cb[np.ix_(perm, perm)]
            db = np.sqrt(sum(np.linalg.norm(log_spd(Pa[s:e, s:e]) - log_spd(Pb[s:e, s:e])) ** 2
                             for s, e in runs)) / np.sqrt(p)
            print(f"niche {k}: d_W {dw:.5f} vs cached {r.d_whole:.5f} | d_B {db:.5f} vs cached {r.d_block:.5f}")
            assert abs(dw - r.d_whole) < rtol * dw and abs(db - r.d_block) < rtol * db
