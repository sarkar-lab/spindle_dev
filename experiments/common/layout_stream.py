"""The production layout and the d_B vs d_W tile pairs at any number of genes, without p x p matrices of every tile in
memory (lymph node 5k at 2,000 genes: the production build would hold ~110 GB of covariances, ~220 GB of
ultrametrics and a ~110 GB feature matrix).

Same steps as ``prepare_tiles`` + ``build_layout`` (seed-0 split), streamed:

* tiles: ``build_quadtree_tiles`` as in ``prepare_tiles``; genes: ``topvar_genes(adata, n_genes)``;
* a tile's covariance is recomputed from the counts whenever it is needed (``tile_cov``: the arithmetic of
  ``preprocessing._tile_cov``, shrunk once);
* ultrametric features (``spd_to_ultrametric``) are written tile by tile to a float32 file on disk;
* PCA (30 components): sklearn's randomized PCA (what ``PCA(30, random_state=0)`` runs on this wide matrix), streamed
  over column chunks of that file. At 400 genes it gives the cached niches exactly; exact PCA does not (ARI 0.50:
  the production randomized PCA is not converged, and Leiden follows its scores);
* Leiden niches as ``ProcessedData.cluster_spds`` (resolution 0.2, + 0.1 until no niche exceeds 1,000 tiles,
  k-means split otherwise); consensus gene order from the niche's mean ultrametric (streamed);
* niche mean correlations (streamed) and blocks by ``ProcessedData.get_adaptive_runs`` (min 15, max 100 genes).

Pairs (``pairs``): 1,000 random distinct pairs of index tiles drawn uniformly from all same-niche pairs
(``default_rng(0)``, as ``block_distance.dataset_pairs``); per pair d_W, d_B and the split of d_W^2 into the entries
inside and between blocks (d_on, d_off), and per tile the mean |r| within and between blocks (pairs with a gene
without counts in the tile left out, as R10).

    import layout_stream as ls
    sec = ls.section("lymph_node_5k", 2000)       # counts on the top-2,000 genes, tiles, split
    lay = ls.layout(sec, out_dir, n_jobs=16)      # {labels, perm, runs, ...}; out_dir keeps the feature file
    pr = ls.pairs(sec, lay, n_jobs=16)            # one row per tile pair
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from threadpoolctl import threadpool_limits

import data
import spindle_dev as sd
from spindle_dev.layout import (ProcessedData, leiden_clustering_latent, spd_to_ultrametric,
                                split_oversized_niches)
from spindle_dev.metrics import log_spd, shrink_cov, spd_to_correlation
from spindle_dev.preprocessing import build_quadtree_tiles, topvar_genes

# build_layout's defaults (index_cache.build calls it with them)
N_PCA, RESOLUTION, RES_STEP, RES_TRIES, MAX_NICHE, K_NEIGHBORS = 30, 0.2, 0.1, 15, 1000, 10
MIN_BLOCK, MAX_BLOCK = 15, 100
N_PAIRS, RNG_SEED = 1000, 0


def _log(msg, t0):
    print(f"[{time.perf_counter() - t0:7.0f} s] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- inputs
def section(key, n_genes, seed=data.EXAMPLE_SEED):
    """Counts on the top-``n_genes`` variable genes (CSR, adata's dtype), the tiles (as prepare_tiles), the genes and
    the seed's index tiles."""
    adata = sd.load_adata(data.paths.dataset_path(key))
    tiles = build_quadtree_tiles(adata.obsm["spatial"], max_pts=200, min_side=0.0, max_depth=40)
    genes, gidx = topvar_genes(adata, G=min(n_genes, adata.n_vars))
    X = adata.X[:, gidx].tocsr()
    train, test = data.split(len(tiles), seed)
    return {"key": key, "X": X, "idx": [t.idx for t in tiles], "genes": np.asarray(genes).astype(str),
            "train": train, "test": test, "p": len(genes)}


def tile_cov(Xs):
    """A tile's shrunk covariance from its rows ``Xs``: preprocessing._tile_cov's arithmetic (float32 out)."""
    n = Xs.shape[0]
    mu = np.asarray(Xs.mean(0)).ravel()
    Sig = (Xs.T @ Xs).toarray() / max(n, 1) - np.outer(mu, mu)
    Sig.flat[::Sig.shape[0] + 1] += 1e-6
    return shrink_cov(Sig).astype(np.float32)


def _chunks(n, k):
    return [c for c in np.array_split(np.arange(n), k) if len(c)]


# ---------------------------------------------------------------------------------------------- layout
def _features(rows, Xs_list, path):
    F = np.load(path, mmap_mode="r+")
    for r, Xs in zip(rows, Xs_list):
        U, _ = spd_to_ultrametric(tile_cov(Xs))
        F[r] = squareform(U, checks=False)
    F.flush()


def _col_mean(F, chunk):
    K, d = F.shape
    return np.concatenate([np.asarray(F[:, c:c + chunk], dtype=np.float64).mean(0) for c in range(0, d, chunk)])


def _xt_q(F, mean, Q, chunk):
    """(X - mean).T @ Q, d x m, over column chunks."""
    out = np.empty((F.shape[1], Q.shape[1]))
    for c in range(0, F.shape[1], chunk):
        out[c:c + chunk] = (np.asarray(F[:, c:c + chunk], dtype=np.float64) - mean[c:c + chunk]).T @ Q
    return out


def _x_p(F, mean, P, chunk):
    """(X - mean) @ P, K x m, over column chunks."""
    out = np.zeros((F.shape[0], P.shape[1]))
    for c in range(0, F.shape[1], chunk):
        out += (np.asarray(F[:, c:c + chunk], dtype=np.float64) - mean[c:c + chunk]) @ P[c:c + chunk]
    return out


def pca_randomized(F, n_components=N_PCA, random_state=0, chunk=50_000, threads=16):
    """``PCA(n_components, random_state).fit_transform(F)`` (sklearn's 'auto' solver for a wide F: randomized SVD,
    10 oversamples, 'auto' iterations, LU power-iteration normaliser, transposed since K < d, svd_flip on the
    components), streamed over column chunks of F (memory-mapped)."""
    from scipy import linalg
    from sklearn.utils.extmath import svd_flip

    K, d = F.shape
    assert K < d and 1 <= n_components < 0.8 * K, "only sklearn's randomized branch for a wide matrix is streamed"
    n_iter = 7 if n_components < 0.1 * min(K, d) else 4
    lu = lambda a: linalg.lu(a, permute_l=True, check_finite=False)  # noqa: E731
    with threadpool_limits(threads):
        mean = _col_mean(F, chunk)
        Q = np.random.RandomState(random_state).normal(size=(K, n_components + 10))  # M = X_c.T: M.shape[1] = K
        for _ in range(n_iter):
            Q, _ = lu(_xt_q(F, mean, Q, chunk))      # M @ Q
            Q, _ = lu(_x_p(F, mean, Q, chunk))       # M.T @ Q
        Q, _ = linalg.qr(_xt_q(F, mean, Q, chunk), mode="economic", check_finite=False)
        B = _x_p(F, mean, Q, chunk).T                # Q.T @ M
        Uhat, s, Vt = linalg.svd(B, full_matrices=False, lapack_driver="gesdd")
        U = Q @ Uhat
    U_, s_, Vt_ = Vt[:n_components].T, s[:n_components], U[:, :n_components].T
    U_, Vt_ = svd_flip(U_, Vt_, u_based_decision=False)
    return U_ * s_


def _niche_corr_sums(idxs, Xs_list, labels):
    out = {}
    for i, Xs in zip(idxs, Xs_list):
        k = labels[i]
        c = spd_to_correlation(tile_cov(Xs))
        out[k] = c if k not in out else out[k] + c
    return out


def features(sec, path, n_jobs=16):
    """The index tiles' ultrametric features (float32, K x p(p-1)/2) written to ``path``; returns it memory-mapped."""
    X, train, p = sec["X"], sec["train"], sec["p"]
    K, d = len(train), p * (p - 1) // 2
    done = Path(str(path) + ".done")
    if not done.exists():
        F = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(K, d)); del F
        Parallel(n_jobs=n_jobs)(delayed(_features)(c, [X[sec["idx"][train[r]]] for r in c], path)
                                for c in _chunks(K, 8 * n_jobs))
        done.touch()
    return np.load(path, mmap_mode="r")


def niches(pca, random_state=0):
    """(labels, resolution): Leiden niches as ProcessedData.cluster_spds."""
    res = RESOLUTION
    for _ in range(RES_TRIES):
        labels = leiden_clustering_latent(pca, k_neighbors=K_NEIGHBORS, resolution=res, random_state=random_state)
        if np.bincount(labels).max() <= MAX_NICHE:
            break
        res += RES_STEP
    else:
        labels = split_oversized_niches(pca, labels, MAX_NICHE, random_state)
    return labels, res


def _leaves(Z):
    """dendrogram(Z, no_plot=True)['leaves'] (as consensus_tree_from_ultrametrics); scipy recurses once per tree
    level, past Python's default limit for ~2,000 genes."""
    import sys
    old = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old, 20 * len(Z) + 1000))
    try:
        return np.array(dendrogram(Z, no_plot=True)["leaves"])
    finally:
        sys.setrecursionlimit(old)


def layout(sec, out_dir, n_jobs=16, keep_features=False):
    """The production layout of the seed's index tiles at the section's genes; cached in ``out_dir/layout.pkl``
    (features.npy, pca.npy are checkpoints: a rerun resumes after them)."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "layout.pkl").exists():
        return pickle.loads((out_dir / "layout.pkl").read_bytes())
    t0 = time.perf_counter()
    X, train, p = sec["X"], sec["train"], sec["p"]
    K, d = len(train), p * (p - 1) // 2
    rows_of = lambda r: X[sec["idx"][train[r]]]  # noqa: E731

    # 1. ultrametric features -> disk (float32)
    fpath = out_dir / "features.npy"
    F = features(sec, fpath, n_jobs)
    _log(f"features: {K} x {d} ({fpath.stat().st_size / 1e9:.1f} GB)", t0)

    # 2. PCA scores (sklearn's randomized PCA, streamed)
    if not (out_dir / "pca.npy").exists():
        if K * d * 8 <= 8e9:  # fits in memory: production's own call
            from sklearn.decomposition import PCA
            with threadpool_limits(n_jobs):
                np.save(out_dir / "pca.npy", PCA(n_components=min(N_PCA, K - 1), random_state=0)
                        .fit_transform(np.asarray(F, dtype=np.float64)))
        else:
            np.save(out_dir / "pca.npy", pca_randomized(F, min(N_PCA, K - 1), threads=n_jobs))
    pca = np.load(out_dir / "pca.npy")
    _log(f"PCA: {pca.shape[1]} components", t0)

    # 3. Leiden niches (ProcessedData.cluster_spds)
    labels, res = niches(pca)
    niche_ids = sorted(set(labels.tolist()))
    _log(f"Leiden resolution {res:.2f}: {len(niche_ids)} niches, sizes "
         f"{sorted(np.bincount(labels).tolist(), reverse=True)}", t0)

    # 4. consensus gene order: mean ultrametric per niche (streamed rows)
    U_sum = {k: np.zeros(d) for k in niche_ids}
    for c in _chunks(K, max(1, K // 200)):
        A = np.asarray(F[c], dtype=np.float64)
        for k in niche_ids:
            m = labels[c] == k
            if m.any():
                U_sum[k] += A[m].sum(0)
    Z_list, perm_list = {}, {}
    for k in niche_ids:
        U_mean = U_sum.pop(k) / np.sum(labels == k)
        Z = linkage(U_mean, method="average")
        Z_list[k], perm_list[k] = Z, _leaves(Z)
    del F
    _log("consensus gene orders", t0)

    # 5. niche mean correlations (streamed) and blocks (ProcessedData.get_adaptive_runs)
    parts = Parallel(n_jobs=n_jobs)(delayed(_niche_corr_sums)(c, [rows_of(r) for r in c], labels)
                                    for c in _chunks(K, n_jobs))
    R_sum = {}
    for part in parts:
        for k, s in part.items():
            R_sum[k] = s if k not in R_sum else R_sum[k] + s
    pd_ = ProcessedData.__new__(ProcessedData)
    pd_.labels, pd_.Z_list, pd_.perm_list = labels, Z_list, perm_list
    pd_.R_mean_list = {k: R_sum[k] / np.sum(labels == k) for k in niche_ids}
    pd_.block_initial_dict, pd_.block_dict = {}, {}
    pd_.get_adaptive_runs(min_final_size=MIN_BLOCK, max_final_size=MAX_BLOCK)
    _log(f"blocks: {sum(len(b) for b in pd_.block_dict.values())} over {len(niche_ids)} niches", t0)

    lay = {"key": sec["key"], "p": p, "genes": sec["genes"], "train": train, "labels": labels, "pca": pca,
           "resolution": res, "perm": perm_list, "runs": pd_.block_dict, "Z": Z_list}
    (out_dir / "layout.pkl").write_bytes(pickle.dumps(lay))
    if not keep_features:
        fpath.unlink(); Path(str(fpath) + ".done").unlink()
    return lay


# ---------------------------------------------------------------------------------------------- pairs
def sample_pairs(labels, n_pairs=N_PAIRS, rng_seed=RNG_SEED):
    """(niche, row_a, row_b) of distinct random pairs of index tiles drawn uniformly from all same-niche pairs; rows
    index the niche's tiles in index order (as block_distance.dataset_pairs)."""
    niches = sorted(set(np.asarray(labels).tolist()))
    sizes = np.array([np.sum(labels == k) for k in niches])
    starts = np.r_[0, np.cumsum(sizes * (sizes - 1) // 2)]
    pick = np.sort(np.random.default_rng(rng_seed).choice(starts[-1], n_pairs, replace=False))
    which = np.searchsorted(starts, pick, side="right") - 1
    out = []
    for i, k in enumerate(niches):
        local = pick[which == i] - starts[i]
        if len(local):
            iu = np.triu_indices(sizes[i], 1)
            out += [(k, a, b) for a, b in zip(iu[0][local], iu[1][local])]
    return out


def _mean_abs_r(C, Xs, within):
    R = np.abs(spd_to_correlation(C).astype(np.float64))
    has = np.asarray((Xs != 0).sum(0)).ravel() > 0           # genes with counts in the tile
    ok = np.triu(np.outer(has, has), 1)
    return R[ok & within].mean(), R[ok & ~within].mean()


def _pair_rows(jobs, p):
    out = []
    for (k, ta, tb, perm, runs, Xa, Xb) in jobs:
        blk = np.empty(p, int)
        for b, (s, e) in enumerate(runs):
            blk[perm[s:e]] = b
        within = blk[:, None] == blk[None, :]
        Ca, Cb = tile_cov(Xa), tile_cov(Xb)
        Dw = log_spd(Ca.astype(np.float64)) - log_spd(Cb.astype(np.float64))
        Pa, Pb = Ca[np.ix_(perm, perm)].astype(np.float64), Cb[np.ix_(perm, perm)].astype(np.float64)
        db2 = sum(np.linalg.norm(log_spd(Pa[s:e, s:e]) - log_spd(Pb[s:e, s:e])) ** 2 for s, e in runs)
        ra_w, ra_b = _mean_abs_r(Ca, Xa, within)
        out.append({"niche": k, "tile_a": ta, "tile_b": tb,
                    "d_whole": np.linalg.norm(Dw) / np.sqrt(p), "d_block": np.sqrt(db2) / np.sqrt(p),
                    "d_on": np.linalg.norm(Dw[within]) / np.sqrt(p), "d_off": np.linalg.norm(Dw[~within]) / np.sqrt(p),
                    "abs_r_within_a": ra_w, "abs_r_between_a": ra_b})
    return out


def pairs(sec, lay, n_jobs=16):
    """One row per sampled tile pair: niche, tiles (prepare() order), d_whole, d_block, d_on, d_off, and tile a's
    mean |r| within and between blocks."""
    X, train, labels, p = sec["X"], lay["train"], lay["labels"], lay["p"]
    members = {k: np.flatnonzero(labels == k) for k in lay["perm"]}
    jobs = []
    for k, a, b in sample_pairs(labels):
        ta, tb = train[members[k][a]], train[members[k][b]]
        jobs.append((k, int(ta), int(tb), np.asarray(lay["perm"][k]), lay["runs"][k], X[sec["idx"][ta]], X[sec["idx"][tb]]))
    rows = Parallel(n_jobs=n_jobs)(delayed(_pair_rows)([jobs[i] for i in c], p) for c in _chunks(len(jobs), 4 * n_jobs))
    return pd.DataFrame([r for part in rows for r in part])


# ---------------------------------------------------------------------------------------------- example matrices
def _niche_corr_sum(Xs_list):
    s = None
    for Xs in Xs_list:
        c = spd_to_correlation(tile_cov(Xs)).astype(np.float64)
        s = c if s is None else s + c
    return s


def example(sec, lay, tile_a, tile_b, n_jobs=16):
    """Matrices of tile_a's niche, in the niche's gene order: the niche's mean correlation, tile_a's correlation and
    |log C_a - log C_b| (both tiles index tiles of that niche, prepare() order), with the niche's blocks."""
    X, train, labels = sec["X"], lay["train"], lay["labels"]
    row = {int(t): i for i, t in enumerate(train)}
    k = int(labels[row[tile_a]])
    assert labels[row[tile_b]] == k, "both tiles must be in one niche"
    perm, runs = np.asarray(lay["perm"][k]), [tuple(map(int, r)) for r in lay["runs"][k]]
    members = np.flatnonzero(labels == k)
    parts = Parallel(n_jobs=n_jobs)(delayed(_niche_corr_sum)([X[sec["idx"][train[m]]] for m in c])
                                    for c in _chunks(len(members), n_jobs))
    R_mean = sum(parts) / len(members)
    Ca, Cb = tile_cov(X[sec["idx"][tile_a]]), tile_cov(X[sec["idx"][tile_b]])
    Ra = spd_to_correlation(Ca)
    D = np.abs(log_spd(Ca.astype(np.float64)) - log_spd(Cb.astype(np.float64)))
    ix = np.ix_(perm, perm)
    return {"niche": k, "niche_tiles": len(members), "perm": perm, "runs": np.array(runs), "genes": sec["genes"][perm],
            "mean_corr": R_mean[ix].astype(np.float32), "tile_corr": Ra[ix].astype(np.float32),
            "abs_log_diff": D[ix].astype(np.float32)}
