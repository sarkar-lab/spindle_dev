"""The index cache: one timed build of the layout and all four indexes per (dataset, seed), and the staged inputs of
every other notebook (R0_index_cache.ipynb runs both; data.py lists the files).

  python experiments/common/index_cache.py build <key> <seed>   one build (its own process, so its peak memory is its own)
  python experiments/common/index_cache.py build <task 0-39>    task t = dataset indexed_keys()[t // 5], seed SEEDS[t % 5]
  python experiments/common/index_cache.py stage <key>          the staged inputs of a dataset (after its 5 builds)
  python experiments/common/index_cache.py stage <task 0-7>

build: stages, each timed (perf_counter) with the peak resident memory after it (ru_maxrss, monotone over the process):
  load       load_adata
  prepare    prepare_tiles: tiles, top-variance genes, every tile's shrunk covariance (joblib threads, n_jobs = 8)
  layout     build_layout on the index tiles (split_tiles(seed, test_ratio = 0.1))
  naive            SpindleNaive(layout, covs)                     the exhaustive whole-tile index (block logs)
  spindle          Spindle.from_naive                             reuses SpindleNaive's block logs
  naive_partial    SpindleNaivePartial(layout, covs)              the exhaustive gene-set index (block covariances + logs)
  spindle_partial  SpindlePartial.from_naive_partial              reuses SpindleNaivePartial's block covariances
Then, untimed: omega = variance_quantiles(index covs); the held-out tiles' SpindleNaive.query_vectors and their exact
d_B to every index tile (SpindleNaive.distances). Bytes: whole covariances as float32 upper triangles of the index
tiles; SpindleNaive.nbytes, Spindle.nbytes, SpindleNaivePartial.nbytes, SpindlePartial.nbytes. The build times are
Fig 2C, so builds run one at a time on one node (timing.sbatch); a local run is sequential anyway.

stage (untimed; needs the dataset's builds, checks the covariances against them; covariances and whole-matrix vectors are
written tile by tile, so brain stages in < 6 GB):
  covs.npy / covs.json, tiles.npz, cells.npz   prepare()'s covariances and tiles, the per-cell inputs (data.py)
  whole.npy / whole.json   whole-matrix vector of every tile: the float32 upper triangle of its p x p log (eigenvalue
                           floor 1e-6), off-diagonals x sqrt 2, / sqrt p, i.e. Spindle-naive's block vector with one
                           block of all genes in panel order (the whole-matrix scan of the timings, on d_B's scale)
  seed<k>/niche_means.npz  mean shrunk covariance (float64) of each niche's index tiles
Cell labels: the section's annotation (obs "Cluster") where it has one (breast); otherwise the 10x graph-based clusters
of its Xenium output (analysis.tar.gz, gene_expression_graphclust/clusters.csv, matched by barcode; "" = no cluster).
"""

import io
import json
import os
import resource
import socket
import sys
import tarfile
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import spindle_dev as sd  # noqa: E402
from spindle_dev.metrics import as_cov  # noqa: E402
from spindle_dev.naive import BlockVectorizer  # noqa: E402
from spindle_dev.preprocessing import build_quadtree_tiles, build_tile_covs_full, topvar_genes  # noqa: E402

BUILD_FILES = ("build.json", "split.npz", "layout.pkl", "spindle.pkl", "spindle_partial.pkl", "naive.pkl",
               "naive_partial.pkl", "queries.pkl", "D.npy")
STAGE_FILES = ("covs.npy", "covs.json", "tiles.npz", "cells.npz", "whole.npy", "whole.json")
STAGE_SEED_FILES = ("niche_means.npz",)
GRAPHCLUST = {
    "skin_melanoma": "xenium_hskin/output-XETG00000__slide_id__hskin/analysis.tar.gz",
    "kidney_nondiseased": "xenium_hkidney/output-XETG0000__slide_id__hkidney/analysis.tar.gz",
    "lung_cancer": "xenium_hlungcancer/output-XETG00000__slide_id__hlungcancer/analysis.tar.gz",
    "pancreatic_cancer": "xenium_hpancreas/output-XETG00000__slide_id__hpancreas/analysis.tar.gz",
    "lymph_node": "xenium_hlymphnode/output-XETG00000__slide_id__hlymphnode/analysis.tar.gz",
    "lymph_node_5k": "xenium_hlymphnode_5k/output-XETG00000__slide_id__hlymphnode5k/analysis.tar.gz",
    "brain_cancer": "xenium_hbraincancer/output-XETG00000__slide_id__hbraincancer/analysis.tar.gz",
}


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # kB on Linux


class MemoryMonitor:
    """Peak anonymous resident memory of this process (RssAnon), sampled every ``interval`` s."""

    def __init__(self, interval=0.05):
        self.peak = 0
        self._stop = threading.Event()
        threading.Thread(target=self._run, args=(interval,), daemon=True).start()

    @staticmethod
    def now():
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
        return 0

    def _run(self, interval):
        while not self._stop.is_set():
            self.peak = max(self.peak, self.now())
            self._stop.wait(interval)

    def peak_gb(self):
        self.peak = max(self.peak, self.now())
        return self.peak / 2**30



# ---------------------------------------------------------------- status (what R0 skips)
def build_done(key, seed):
    d = data.cache_dir(key, seed)
    if not all((d / f).exists() for f in BUILD_FILES):
        return False
    return json.loads((d / "build.json").read_text()).get("tag") == data.TAG


def stage_done(key):
    d = data.CACHE / key
    if not all((d / f).exists() for f in STAGE_FILES):
        return False
    if json.loads((d / "covs.json").read_text()).get("tag") != data.TAG:
        return False
    return all((data.cache_dir(key, s) / f).exists() for s in data.SEEDS for f in STAGE_SEED_FILES)


def status():
    """One row per (dataset, seed): build done, dataset staged."""
    return pd.DataFrame([{"key": k, "seed": s, "built": build_done(k, s), "staged": stage_done(k)}
                         for k in data.indexed_keys() for s in data.SEEDS])


# ---------------------------------------------------------------- build
def build(key, seed):
    rec = {"key": key, "seed": seed, **data.stamp(), "host": socket.gethostname(),
           "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "cpus": os.environ.get("SLURM_CPUS_PER_TASK", "")}
    out = data.cache_dir(key, seed)
    out.mkdir(parents=True, exist_ok=True)

    def stage(name, fn):
        t = time.perf_counter()
        res = fn()
        rec[f"{name}_s"] = time.perf_counter() - t
        rec[f"peak_gb_{name}"] = peak_gb()
        print(f"{key} seed {seed}: {name} {rec[f'{name}_s']:.1f} s, peak {rec[f'peak_gb_{name}']:.1f} GB", flush=True)
        return res

    adata = stage("load", lambda: sd.load_adata(data.paths.dataset_path(key)))
    tiles, covs, genes = stage("prepare", lambda: sd.prepare_tiles(adata, max_genes=data.paths.dataset_max_genes(key)))
    train, test = data.split(len(tiles), seed)
    tr_covs = [covs[i] for i in train]
    layout = stage("layout", lambda: sd.build_layout([tiles[i] for i in train], tr_covs, genes, adata.n_obs))
    naive = stage("naive", lambda: sd.SpindleNaive(layout, tr_covs))
    spindle = stage("spindle", lambda: sd.Spindle.from_naive(naive, layout))
    naive_partial = stage("naive_partial", lambda: sd.SpindleNaivePartial(layout, tr_covs))
    spindle_partial = stage("spindle_partial", lambda: sd.SpindlePartial.from_naive_partial(naive_partial, layout))
    finish_build(rec, out, n_cells=int(adata.n_obs), n_tiles=len(tiles), cov_checksum=data.cov_checksum(covs),
                 genes=genes, train=train, test=test, tr_covs=tr_covs, test_covs=[covs[i] for i in test],
                 layout=layout, naive=naive, spindle=spindle, naive_partial=naive_partial,
                 spindle_partial=spindle_partial)


def finish_build(rec, out, *, n_cells, n_tiles, cov_checksum, genes, train, test, tr_covs, test_covs, layout, naive,
                 spindle, naive_partial, spindle_partial, omega=None, mon=None):
    """The untimed tail of a build (this module's and lowmem_build's): sizes and bytes into ``rec``, omega (computed
    here unless given), the held-out query vectors and D, then every BUILD_FILES file in ``out`` (build.json last).
    ``mon`` (a MemoryMonitor) adds the build's peak anonymous memory, peak_anon_gb."""
    key, seed = rec["key"], rec["seed"]
    p, n = len(genes), len(train)
    labels = np.asarray(layout.labels).astype(int)
    rec.update(n_cells=n_cells, p=p, n_tiles=n_tiles, n_train=n, n_test=len(test),
               n_niches=len(layout.block_dict), n_blocks=int(sum(len(b) for b in layout.block_dict.values())),
               niche_sizes=np.bincount(labels).tolist(),
               cov_checksum=cov_checksum,
               bytes_whole=4 * n * p * (p + 1) // 2, bytes_naive=naive.nbytes(), bytes_spindle=spindle.nbytes(),
               bytes_naive_partial=naive_partial.nbytes(), bytes_spindle_partial=spindle_partial.nbytes(),
               spindle_partial_naive_bytes=spindle_partial.naive_bytes,
               spindle_nodes=spindle.n_nodes(), spindle_partial_nodes=spindle_partial.n_nodes())
    rec["spindle_pct"] = 100 * rec["bytes_spindle"] / rec["bytes_naive"]
    rec["spindle_partial_pct"] = 100 * rec["bytes_spindle_partial"] / spindle_partial.naive_bytes

    if omega is None:
        omega = sd.variance_quantiles(tr_covs)
    qvs = [naive.query_vectors(c) for c in test_covs]
    D = np.stack([naive.distances(qv) for qv in qvs])

    data.save_layout(layout, out / "layout.pkl")
    for name, obj in (("spindle", spindle), ("spindle_partial", spindle_partial), ("naive", naive),
                      ("naive_partial", naive_partial), ("queries", qvs)):
        data._dump(obj, out / f"{name}.pkl")
    np.savez(out / "split.npz", train=train, test=test, genes=np.asarray(genes).astype(str), omega=omega)
    np.save(out / "D.npy", D)
    rec["peak_gb"] = peak_gb()
    if mon is not None:
        rec["peak_anon_gb"] = mon.peak_gb()
    (out / "build.json").write_text(json.dumps(rec, indent=1))  # last: build_done() keys on it
    print(f"{key} seed {seed}: {n} index / {len(test)} held-out tiles, {rec['n_niches']} niches, "
          f"Spindle {rec['spindle_pct']:.2f} %, Spindle partial {rec['spindle_partial_pct']:.2f} % of Spindle-naive; "
          f"done", flush=True)


# ---------------------------------------------------------------- stage
def cell_labels(key, adata):
    if "Cluster" in adata.obs.columns:
        return np.asarray(adata.obs["Cluster"].astype(str)).astype(str)
    with tarfile.open(data.paths.data_dir("xenium") / GRAPHCLUST[key]) as tf:
        m = next(m for m in tf if m.name.endswith("gene_expression_graphclust/clusters.csv"))
        cl = pd.read_csv(io.BytesIO(tf.extractfile(m).read()))
    lab = cl.set_index(cl.Barcode.astype(str))["Cluster"].reindex(adata.obs_names)
    print(f"{key}: {int(lab.isna().sum())} of {len(lab)} cells without a graph-based cluster", flush=True)
    return np.array(["" if pd.isna(v) else str(int(v)) for v in lab])


def whole_vectorizer(p):
    return BlockVectorizer(np.arange(p), [(0, p)])


# ---------------------------------------------------------------- covariances one at a time (stage, lowmem_build)
COV_BATCH = 512  # tiles per build_tile_covs_full call


def tiles_and_genes(adata, max_genes):
    """prepare_tiles' tiles, genes and gene columns, without the covariances."""
    tiles = build_quadtree_tiles(adata.obsm["spatial"], max_pts=200, min_side=0.0, max_depth=40)
    genes, gene_idx = topvar_genes(adata, G=min(max_genes, adata.n_vars))
    return tiles, genes, gene_idx


def iter_tile_covs(adata, tiles, gene_idx, batch=COV_BATCH):
    """prepare_tiles' covariances ({'tile_id', 'cov', 'n'}) in tile order, ``batch`` tiles at a time: every tile's
    covariance is computed alone, so these are the values of prepare_tiles(adata, max_genes)[1]."""
    X = SimpleNamespace(X=adata.X[:, gene_idx])  # what build_tile_covs_full(adata, tiles, gene_idx) slices
    for a in range(0, len(tiles), batch):
        yield from build_tile_covs_full(X, tiles[a:a + batch], None, n_jobs=8, eps=1e-6)


def stage(key):
    """Covariances and whole-matrix vectors are written tile by tile (memory: the section, one niche's covariances in
    float64 for its mean, and the cell matrix; brain < 6 GB), with the values of data.prepare(key)."""
    st = data.stamp()
    t0 = time.perf_counter()
    mon = MemoryMonitor()
    d = data.CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    adata = sd.load_adata(data.paths.dataset_path(key))
    tiles, genes, gene_idx = tiles_and_genes(adata, data.paths.dataset_max_genes(key))
    p = len(genes)

    # covariances and tiles, exactly as prepare() returns them
    C = np.lib.format.open_memmap(d / "covs.npy", mode="w+", dtype=np.float32, shape=(len(tiles), p, p))
    chk = 0  # = data.cov_checksum(covs): the same sum in the same order
    for i, c in enumerate(iter_tile_covs(adata, tiles, gene_idx)):
        C[i] = c["cov"]
        chk += np.trace(np.asarray(c["cov"], dtype=np.float64))
    C.flush()
    del C
    chk = float(chk)
    for seed in data.SEEDS:
        meta = json.loads((data.cache_dir(key, seed) / "build.json").read_text())
        if len(tiles) != meta["n_tiles"] or abs(chk - meta["cov_checksum"]) > 1e-6 * abs(meta["cov_checksum"]):
            raise RuntimeError(f"{key} seed {seed}: covariances differ from the build")
    covs = np.load(d / "covs.npy", mmap_mode="r")
    np.savez(d / "tiles.npz", id=np.array([t.id for t in tiles]), bbox=np.array([t.bbox for t in tiles], float),
             idx=np.concatenate([t.idx for t in tiles]), off=np.cumsum([0] + [len(t.idx) for t in tiles]))

    # per cell: raw counts on the index genes, totals, positions, labels
    gi = adata.var_names.get_indexer([str(g) for g in genes])
    if (gi < 0).any():
        raise RuntimeError("index genes missing from the section")
    from scipy.sparse import csr_matrix
    X = csr_matrix(adata.X[:, gi])
    tot = np.asarray(adata.X.sum(1)).ravel()
    np.savez(d / "cells.npz", X_data=X.data, X_indices=X.indices, X_indptr=X.indptr, X_shape=np.array(X.shape),
             tot=tot, xy=np.asarray(adata.obsm["spatial"], dtype=np.float32), labels=cell_labels(key, adata),
             n_obs=adata.n_obs)
    del X

    # whole-matrix vectors
    vec = whole_vectorizer(p)
    W = np.lib.format.open_memmap(d / "whole.npy", mode="w+", dtype=np.float32, shape=(len(covs), vec.dim))
    for i, c in enumerate(covs):
        W[i] = vec.vector(c, np.float32)
    W.flush()
    del W
    for seed in data.SEEDS:
        s = np.load(data.cache_dir(key, seed) / "split.npz")

        # niche means of the index tiles (imputation baseline of the partial-query accuracy)
        lay = data._load(data.cache_dir(key, seed) / "layout.pkl")
        labels = np.asarray(lay.labels).astype(int)
        niches = sorted(set(labels.tolist()))
        M = np.stack([np.mean([as_cov(covs[s["train"][r]]) for r in np.flatnonzero(labels == k)], axis=0)
                      for k in niches])
        np.savez(data.cache_dir(key, seed) / "niche_means.npz", niches=np.array(niches), M=M)
        print(f"{key} seed {seed}: {len(niches)} niche means", flush=True)

    rec = {"key": key, **st, "n_tiles": len(covs), "p": p, "dim": vec.dim, "cov_checksum": chk,
           "seconds": time.perf_counter() - t0, "peak_gb": peak_gb(), "peak_anon_gb": mon.peak_gb()}
    (d / "whole.json").write_text(json.dumps(rec, indent=1))
    (d / "covs.json").write_text(json.dumps({**rec, "tile_id": [int(t.id) for t in tiles],  # = covs' tile_id, n
                                             "n": [len(t.idx) for t in tiles]}))  # last: stage_done() keys on it
    print(f"{key}: staged in {rec['seconds']:.0f} s, peak {rec['peak_anon_gb']:.2f} GB anonymous "
          f"({rec['peak_gb']:.1f} GB with mapped files); done", flush=True)


def main():
    what, args = sys.argv[1], sys.argv[2:]
    if not args:
        args = [os.environ["SLURM_ARRAY_TASK_ID"]]
    keys = data.indexed_keys()
    if what == "build":
        if len(args) == 1:
            t = int(args[0])
            key, seed = keys[t // len(data.SEEDS)], data.SEEDS[t % len(data.SEEDS)]
        else:
            key, seed = args[0], int(args[1])
        build(key, seed)
    elif what == "stage":
        stage(keys[int(args[0])] if args[0].isdigit() else args[0])
    else:
        raise SystemExit(f"unknown command {what!r}: build | stage")


if __name__ == "__main__":
    main()
