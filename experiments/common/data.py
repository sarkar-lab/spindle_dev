"""Datasets, the index cache and the version stamp shared by every experiment (no plotting; safe in sbatch).

    import data
    adata, tiles, covs, genes = data.prepare("breast_cancer")          # recomputed (deterministic, seconds)
    c = data.load_cache("breast_cancer", seed=0)                         # what index_cache.build stored
    c.spindle, c.spindle_partial, c.naive, c.naive_partial, c.layout, c.train, c.test, c.omega, c.D, c.qvs
    data.stamp()                                                        # {"tag": "method-v2", ...}; every result row
    covs, tiles = data.staged_covs("breast_cancer"), data.staged_tiles("breast_cancer")   # prepare()'s, no h5ad

Seeds: 0-4, 10 % of the tiles held out at random (``split_tiles(seed, test_ratio=0.1)``);
seed 0 is the one used for examples, maps and schematics.

Cache layout, results/_cache/<key>/seed<k>/ (gitignored, shared file system), written by
index_cache.build (R0_index_cache.ipynb):
  layout.pkl     the ProcessedData of the index tiles, without spd_matrices and U_list (one p x p matrix per tile;
                 ``load_cache(..., covs=covs)`` puts the index covariances back)
  spindle.pkl, spindle_partial.pkl, naive.pkl, naive_partial.pkl
                 Spindle, SpindlePartial, SpindleNaive, SpindleNaivePartial
  split.npz      train / test tile indices (into prepare()'s tiles), genes, omega (variance_quantiles of the index)
  queries.pkl    SpindleNaive.query_vectors of every held-out tile (test order)
  D.npy          exact d_B, held-out x index tiles (float64, SpindleNaive.distances)
  build.json     build seconds per stage, peak memory, bytes, sizes, host, stamp
  niche_means.npz  mean shrunk covariance of each niche's index tiles (float64, p x p; staged)

Staged inputs, results/_cache/<key>/ (index_cache.stage, run by R0_index_cache.ipynb), so that no other notebook loads an
h5ad or recomputes covariances (16 GB rule):
  covs.npy, covs.json   every tile's shrunk covariance, (n_tiles, p, p) float32 exactly as prepare() returns them
                        (memory-mapped by ``staged_covs``), tile ids, cells per tile, checksum
  tiles.npz             the quadtree tiles (ids, boxes, cell indices)
  cells.npz             per cell: raw counts on the index genes (CSR), total counts, x/y, cell labels, n_obs
  whole.npy, whole.json whole-matrix log vectors of every tile (float32, prepare() tile order)
"""

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import paths  # noqa: E402
import spindle_dev as sd  # noqa: E402

TAG = "method-v2"  # the tag every result carries
SRC_TAG = "method-v3"  # src/ must equal this: method-v2 + Spindle's search split by niche (new functions only)
SEEDS = [0, 1, 2, 3, 4]
EXAMPLE_SEED = 0
TEST_RATIO = 0.1
CACHE = Path(os.environ.get("SPINDLE_CACHE", ROOT / "results" / "_cache"))  # override: smoke tests
RESULTS = ROOT / "results"


def indexed_keys():
    return paths.indexed_keys()


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def stamp():
    """The code version of a result: ``tag`` is TAG only if src/ is identical to SRC_TAG (committed and in the working
    tree); otherwise this raises, because a result of a changed method is not a figure input. method-v2 is method-v1
    with the method renamed (bit-identical outputs); method-v3 is method-v2 plus Spindle's search split by niche
    (kernels.c spindle_probe_niche / spindle_scan_niche, FlatSpindle.search_parallel, Spindle.search_vectors_parallel;
    new functions only, so every method-v2 output is unchanged and results keep the method-v2 tag)."""
    if _git("diff", SRC_TAG, "--", "src/"):
        changed = _git("diff", "--name-only", SRC_TAG, "--", "src/")
        raise RuntimeError(f"src/ differs from {SRC_TAG}: {changed.split()}; no figure input can be made")
    return {"tag": TAG, "describe": _git("describe", "--tags", "--dirty", "--always"),
            "commit": _git("rev-parse", "--short", "HEAD")}


def split(n_tiles, seed):
    """(train, test) tile indices of a seed: 10 % held out at random."""
    return sd.split_tiles(n_tiles, seed=seed, test_ratio=TEST_RATIO)


def prepare(key):
    """(adata, tiles, covs, genes) of a dataset: load_adata + prepare_tiles (the index's gene cap)."""
    adata = sd.load_adata(paths.dataset_path(key))
    tiles, covs, genes = sd.prepare_tiles(adata, max_genes=paths.dataset_max_genes(key))
    return adata, tiles, covs, genes


def cov_checksum(covs):
    """A float of every tile covariance (sum of traces, float64): guards the cache against drift in prepare()."""
    return float(sum(np.trace(np.asarray(c["cov"], dtype=np.float64)) for c in covs))


def cache_dir(key, seed):
    return CACHE / key / f"seed{seed}"


def _dump(obj, path):
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def _load(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def save_layout(layout, path):
    """The layout without its per-tile matrices (spd_matrices, U_list), which load_cache can put back."""
    big = {"spd_matrices": layout.spd_matrices, "U_list": layout.U_list}
    layout.spd_matrices, layout.U_list = None, None
    try:
        _dump(layout, path)
    finally:
        layout.spd_matrices, layout.U_list = big["spd_matrices"], big["U_list"]


def load_cache(key, seed, *, parts=("layout", "spindle", "spindle_partial", "naive", "naive_partial", "queries", "D"), covs=None):
    """The cached build of (key, seed) as a namespace; ``parts`` limits what is unpickled. ``covs`` (prepare()'s
    covariances) are checked against the build and put back into layout.spd_matrices (index tiles)."""
    d = cache_dir(key, seed)
    meta = json.loads((d / "build.json").read_text())
    s = np.load(d / "split.npz", allow_pickle=False)
    c = SimpleNamespace(key=key, seed=seed, meta=meta, train=s["train"], test=s["test"], genes=s["genes"],
                        omega=s["omega"])
    for part in parts:
        if part == "D":
            c.D = np.load(d / "D.npy")
        elif part == "queries":
            c.qvs = _load(d / "queries.pkl")
        else:
            setattr(c, part, _load(d / f"{part}.pkl"))
    if covs is not None:
        chk = covs.checksum if isinstance(covs, StagedCovs) else cov_checksum(covs)
        if len(covs) != meta["n_tiles"] or abs(chk - meta["cov_checksum"]) > 1e-6 * abs(
                meta["cov_checksum"]):
            raise RuntimeError(f"{key}: covariances differ from those the cache was built from")
        if hasattr(c, "layout"):
            c.layout.spd_matrices = [covs[i]["cov"] for i in c.train]
    return c


# ---------------------------------------------------------------- staged inputs (index_cache.stage)
class StagedCovs:
    """prepare()'s covariance list, memory-mapped from covs.npy: ``covs[i]`` is {"tile_id", "cov", "n"} with the same
    float32 values, read from disk only when touched. ``checksum`` is cov_checksum of the full list (stored)."""

    def __init__(self, key):
        d = CACHE / key
        meta = json.loads((d / "covs.json").read_text())
        self.arr = np.load(d / "covs.npy", mmap_mode="r")
        self.tile_id, self.n = np.asarray(meta["tile_id"]), np.asarray(meta["n"])
        self.checksum = meta["cov_checksum"]

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return {"tile_id": int(self.tile_id[i]), "cov": self.arr[i], "n": int(self.n[i])}

    def __iter__(self):
        return (self[i] for i in range(len(self)))


def staged_covs(key):
    return StagedCovs(key)


def staged_tiles(key):
    """prepare()'s tiles (QuadTile objects) from tiles.npz."""
    from spindle_dev.preprocessing import QuadTile
    z = np.load(CACHE / key / "tiles.npz")
    off, idx, bbox = z["off"], z["idx"], z["bbox"]
    return [QuadTile(int(i), tuple(float(v) for v in bbox[t]), idx[off[t]:off[t + 1]])
            for t, i in enumerate(z["id"])]


def staged_cells(key):
    """Per-cell inputs as a dict: X (raw counts on the index genes, dense float32, index gene order), tot (total counts
    over all genes), xy (float32), labels (str; "" = none), n_obs."""
    from scipy.sparse import csr_matrix
    z = np.load(CACHE / key / "cells.npz", allow_pickle=False)
    X = csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    return {"X": np.asarray(X.toarray(), dtype=np.float32), "tot": z["tot"], "xy": z["xy"],
            "labels": z["labels"].astype(str), "n_obs": int(z["n_obs"])}


def staged_whole(key, mmap=True):
    return np.load(CACHE / key / "whole.npy", mmap_mode="r" if mmap else None)


def niche_means(key, seed):
    """{niche: mean shrunk covariance of its index tiles (p x p float64)}, as the Fig 4 baselines use."""
    z = np.load(cache_dir(key, seed) / "niche_means.npz")
    return {int(k): z["M"][i] for i, k in enumerate(z["niches"])}
