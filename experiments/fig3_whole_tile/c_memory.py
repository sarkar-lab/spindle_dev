"""Memory against fidelity (Fig 3D: breast and brain; no supplement, user 26 Oct): Spindle-lite on block logs
against compressed indexes over whole-matrix log vectors, each graded against its own exact distance.

Usage: sbatch --array=10-14,35-39 experiments/common/run.sbatch experiments/fig3_whole_tile/c_memory.py   (breast, brain)
       python experiments/fig3_whole_tile/c_memory.py [task id 0-39 | key seed]   (no argument: $SLURM_ARRAY_TASK_ID)
Needs c_whole.py first (whole.npy, DW.npy). Task t = dataset indexed_keys()[t // 5], seed SEEDS[t % 5].

Methods (the agreed competitors, memory no-new-competitors; ported from the old whole_cov_baselines.py). The
competitors index what a user without Spindle would index: the whole-matrix log vector of every index tile
(common3), whose exact search is the whole-matrix scan (d_W, DW.npy). HNSW was removed (user, 26 Oct): it stores
every whole vector plus a graph, so it compresses nothing, and at these N its search is nearly exhaustive.
  pca<D>_flat   PCA to D in {16, 64, 256} (randomized SVD of the centred index vectors, random_state = seed), exact
                scan of the codes; MB = codes N x D + projection + mean.
  pca256_pq<m>  PCA-256, then product quantization (m in {8, 32} sub-quantizers, 8 bits); MB = serialized PQ (codes +
                codebooks) + projection + mean.
  lite          Spindle-lite as built (the cached index, beta = 0.10), g = 0.25; MB = SpindleLite.nbytes(). Graded
                against d_B (D.npy). One point (user, 26 Oct): a beta curve at the fixed g is not monotone (above 0.10
                the finer nodes return fewer tiles: short of 5 % N 13 % -> 96 %, NX up), so it was dropped.
Reference stores (dotted lines, not graded: NX = 0 by definition): Spindle = Spindle.nbytes(), whole = float32 upper
triangles of the whole-matrix logs of the index tiles.
Readout per method: NX5 (c = 5 % of N; c_eff = min(c, returned)), Overlap(5 %, 5 % N), share of queries returning
fewer than c; mean over every held-out query. The PQ build uses the job's cores (build time is not a readout).

Writes results/fig3_whole_tile/memory/<key>_seed<k>.csv: one row per method.
"""

import os
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common3 as c3  # noqa: E402
from common3 import data  # noqa: E402

PCA_DIMS = [16, 64, 256]
PQ_M = [8, 32]


class Pca:
    def __init__(self, X, D, seed):
        self.mean = X.mean(axis=0)
        _, _, Vt = randomized_svd(X - self.mean, n_components=D, random_state=seed)
        self.W = np.ascontiguousarray(Vt.T, dtype=np.float32)

    def project(self, X, D=None):
        W = self.W if D is None else self.W[:, :D]
        return np.ascontiguousarray((np.atleast_2d(X) - self.mean) @ W, dtype=np.float32)


def grade(name, param, mb, orders, Dref, c):
    """Readout rows of one method: orders[j] = the rows it returns for query j, best first. NX is NaN for a query
    that returned nothing (left out of the mean, as in c_nx.py; their share is ``empty``)."""
    nxs = [c3.nx(Dref[j], o, c) for j, o in enumerate(orders)]
    return {"method": name, "param": param, "MB": mb / 1e6, "NX5": np.nanmean(nxs), "NX5_median": np.nanmedian(nxs),
            "overlap5": np.mean([c3.overlap(Dref[j], o, c) for j, o in enumerate(orders)]),
            "short5": np.mean([len(o) < c for o in orders]), "empty": np.mean([len(o) == 0 for o in orders]),
            "n_queries": len(orders)}


def run(key, seed):
    st = data.stamp()
    threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    cc = data.load_cache(key, seed, parts=("spindle", "lite", "queries", "D"))
    DW = np.load(data.cache_dir(key, seed) / "DW.npy")
    W = np.load(c3.whole_path(key), mmap_mode="r")
    X = np.ascontiguousarray(W[cc.train])
    Q = np.ascontiguousarray(W[cc.test])
    n = len(X)
    c = c3.c5(n)
    rows = []

    def log(r):
        r.update(key=key, seed=seed, N=n, c5=c, tag=st["tag"], describe=st["describe"])
        rows.append(r)
        print(f"{key} seed {seed}: {r['method']} {r['param']}: {r['MB']:.2f} MB, NX5 {100 * r.get('NX5', 0):.2f} %, "
              f"overlap {r.get('overlap5', 0):.3f}, short5 {r.get('short5', 0):.3f}, empty {r.get('empty', 0):.3f}",
              flush=True)

    # Spindle-lite as built, graded against d_B
    orders = [cc.lite.search_vectors(qv)[0] for qv in cc.qvs]
    log({**grade("lite", cc.lite.beta, cc.lite.nbytes(), orders, cc.D, c), "nodes": cc.lite.n_nodes()})
    log({"method": "spindle", "param": "", "MB": cc.spindle.nbytes() / 1e6})
    log({"method": "whole", "param": "", "MB": X.nbytes / 1e6})

    # whole-matrix competitors, graded against d_W
    faiss.omp_set_num_threads(threads)
    pca = Pca(X, min(max(PCA_DIMS), n - 1), seed)
    Z, Zq = pca.project(X), pca.project(Q)
    for D in PCA_DIMS:
        if D > Z.shape[1]:
            continue
        orders = [np.argsort(((Z[:, :D] - z[:D]) ** 2).sum(1), kind="stable")[:c] for z in Zq]
        log(grade("pca_flat", D, Z[:, :D].nbytes + pca.W[:, :D].nbytes + pca.mean.nbytes, orders, DW, c))
    for m in PQ_M:
        if Z.shape[1] % m or n < 256:
            continue
        pq = faiss.IndexPQ(Z.shape[1], m, 8)
        pq.train(Z)
        pq.add(Z)
        _, I = pq.search(Zq, c)
        log(grade("pca256_pq", m, faiss.serialize_index(pq).nbytes + pca.W.nbytes + pca.mean.nbytes,
                  [i[i >= 0] for i in I], DW, c))

    dst = c3.OUT / "memory"
    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dst / f"{key}_seed{seed}.csv", index=False)


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        t = int(sys.argv[1] if len(sys.argv) == 2 else os.environ["SLURM_ARRAY_TASK_ID"])
        run(data.indexed_keys()[t // len(data.SEEDS)], data.SEEDS[t % len(data.SEEDS)])
    else:
        run(sys.argv[1], int(sys.argv[2]))
