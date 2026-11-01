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

stage (untimed; needs the dataset's builds, checks the covariances against them):
  covs.npy / covs.json, tiles.npz, cells.npz   prepare()'s covariances and tiles, the per-cell inputs (data.py)
  whole.npy / whole.json   whole-matrix vector of every tile: the float32 upper triangle of its p x p log (eigenvalue
                           floor 1e-6), off-diagonals x sqrt 2, / sqrt p, i.e. Spindle-naive's block vector with one
                           block of all genes in panel order; its L2 distance is the whole-matrix d_W, on d_B's scale
  seed<k>/DW.npy           exact d_W, held-out x index tiles (compiled scan, as D.npy)
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
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import spindle_dev as sd  # noqa: E402
from spindle_dev.kernels import naive_scan  # noqa: E402
from spindle_dev.metrics import as_cov  # noqa: E402
from spindle_dev.naive import BlockVectorizer  # noqa: E402

BUILD_FILES = ("build.json", "split.npz", "layout.pkl", "spindle.pkl", "spindle_partial.pkl", "naive.pkl",
               "naive_partial.pkl", "queries.pkl", "D.npy")
STAGE_FILES = ("covs.npy", "covs.json", "tiles.npz", "cells.npz", "whole.npy", "whole.json")
STAGE_SEED_FILES = ("DW.npy", "niche_means.npz")
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

    p, n = len(genes), len(train)
    labels = np.asarray(layout.labels).astype(int)
    rec.update(n_cells=int(adata.n_obs), p=p, n_tiles=len(tiles), n_train=n, n_test=len(test),
               n_niches=len(layout.block_dict), n_blocks=int(sum(len(b) for b in layout.block_dict.values())),
               niche_sizes=np.bincount(labels).tolist(),
               cov_checksum=data.cov_checksum(covs),
               bytes_whole=4 * n * p * (p + 1) // 2, bytes_naive=naive.nbytes(), bytes_spindle=spindle.nbytes(),
               bytes_naive_partial=naive_partial.nbytes(), bytes_spindle_partial=spindle_partial.nbytes(),
               spindle_partial_naive_bytes=spindle_partial.naive_bytes,
               spindle_nodes=spindle.n_nodes(), spindle_partial_nodes=spindle_partial.n_nodes())
    rec["spindle_pct"] = 100 * rec["bytes_spindle"] / rec["bytes_naive"]
    rec["spindle_partial_pct"] = 100 * rec["bytes_spindle_partial"] / spindle_partial.naive_bytes

    omega = sd.variance_quantiles(tr_covs)
    qvs = [naive.query_vectors(covs[i]) for i in test]
    D = np.stack([naive.distances(qv) for qv in qvs])

    data.save_layout(layout, out / "layout.pkl")
    for name, obj in (("spindle", spindle), ("spindle_partial", spindle_partial), ("naive", naive),
                      ("naive_partial", naive_partial), ("queries", qvs)):
        data._dump(obj, out / f"{name}.pkl")
    np.savez(out / "split.npz", train=train, test=test, genes=np.asarray(genes).astype(str), omega=omega)
    np.save(out / "D.npy", D)
    rec["peak_gb"] = peak_gb()
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


def stage(key):
    st = data.stamp()
    t0 = time.perf_counter()
    d = data.CACHE / key
    d.mkdir(parents=True, exist_ok=True)
    adata, tiles, covs, genes = data.prepare(key)
    chk = data.cov_checksum(covs)
    for seed in data.SEEDS:
        meta = json.loads((data.cache_dir(key, seed) / "build.json").read_text())
        if len(covs) != meta["n_tiles"] or abs(chk - meta["cov_checksum"]) > 1e-6 * abs(meta["cov_checksum"]):
            raise RuntimeError(f"{key} seed {seed}: covariances differ from the build")
    p = len(genes)

    # covariances and tiles, exactly as prepare() returns them
    C = np.lib.format.open_memmap(d / "covs.npy", mode="w+", dtype=np.float32, shape=(len(covs), p, p))
    for i, c in enumerate(covs):
        C[i] = c["cov"]
    C.flush()
    del C
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

    # whole-matrix vectors and d_W
    vec = whole_vectorizer(p)
    W = np.empty((len(covs), vec.dim), dtype=np.float32)
    for i, c in enumerate(covs):
        W[i] = vec.vector(c["cov"], np.float32)
    np.save(d / "whole.npy", W)
    for seed in data.SEEDS:
        s = np.load(data.cache_dir(key, seed) / "split.npz")
        X = np.ascontiguousarray(W[s["train"]])
        rows = np.arange(len(X), dtype=np.int64)
        DW = np.empty((len(s["test"]), len(X)))
        for j, t in enumerate(s["test"]):
            naive_scan(X, W[t], rows, DW[j])
        np.save(data.cache_dir(key, seed) / "DW.npy", DW)
        del X

        # niche means of the index tiles (Fig 4 imputation baseline)
        lay = data._load(data.cache_dir(key, seed) / "layout.pkl")
        labels = np.asarray(lay.labels).astype(int)
        niches = sorted(set(labels.tolist()))
        M = np.stack([np.mean([as_cov(covs[s["train"][r]]) for r in np.flatnonzero(labels == k)], axis=0)
                      for k in niches])
        np.savez(data.cache_dir(key, seed) / "niche_means.npz", niches=np.array(niches), M=M)
        print(f"{key} seed {seed}: d_W {DW.shape}, {len(niches)} niche means", flush=True)

    rec = {"key": key, **st, "n_tiles": len(covs), "p": p, "dim": vec.dim, "cov_checksum": chk,
           "seconds": time.perf_counter() - t0, "peak_gb": peak_gb()}
    (d / "whole.json").write_text(json.dumps(rec, indent=1))
    (d / "covs.json").write_text(json.dumps({**rec, "tile_id": [int(c["tile_id"]) for c in covs],
                                             "n": [int(c["n"]) for c in covs]}))  # last: stage_done() keys on it
    print(f"{key}: staged in {rec['seconds']:.0f} s, peak {rec['peak_gb']:.1f} GB; done", flush=True)


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
