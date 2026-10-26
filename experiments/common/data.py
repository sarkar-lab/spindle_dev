"""Datasets, the index cache and the version stamp shared by every experiment (no plotting; safe in sbatch).

    import data
    adata, tiles, covs, genes = data.prepare("breast_cancer")          # recomputed (deterministic, seconds)
    c = data.load_cache("breast_cancer", seed=0)                         # what c_index_cache.py stored
    c.spindle, c.lite, c.partial, c.plite, c.layout, c.train, c.test, c.omega, c.D, c.qvs
    data.stamp()                                                        # {"tag": "method-v1", ...}; every result row

Seeds (paper/RESULTS_PLAN.md §1): 0-4, 10 % of the tiles held out at random (``split_tiles(seed, test_ratio=0.1)``);
seed 0 is the one used for examples, maps and schematics.

Cache layout, results/_cache/<key>/seed<k>/ (gitignored, shared file system), written by c_index_cache.py:
  layout.pkl     the ProcessedData of the index tiles, without spd_matrices and U_list (one p x p matrix per tile;
                 ``load_cache(..., covs=covs)`` puts the index covariances back)
  spindle.pkl, lite.pkl, partial.pkl, plite.pkl   Spindle, SpindleLite, SpindlePartial, SpindleLitePartial
  split.npz      train / test tile indices (into prepare()'s tiles), genes, omega (variance_quantiles of the index)
  queries.pkl    Spindle.query_vectors of every held-out tile (test order)
  D.npy          exact d_B, held-out x index tiles (float64, Spindle.distances)
  build.json     build seconds per stage, peak memory, bytes, sizes, host, stamp
"""

import json
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

TAG = "method-v1"
SEEDS = [0, 1, 2, 3, 4]
EXAMPLE_SEED = 0
TEST_RATIO = 0.1
CACHE = ROOT / "results" / "_cache"
RESULTS = ROOT / "results"


def indexed_keys():
    return paths.indexed_keys()


def _git(*args):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True).stdout.strip()


def stamp():
    """The code version of a result: ``tag`` is TAG only if src/ is identical to it (committed and in the working
    tree); otherwise this raises, because a result of a changed method is not a figure input (FIGURES.md §0)."""
    changed = _git("diff", "--name-only", TAG, "--", "src/")
    if changed:
        raise RuntimeError(f"src/ differs from {TAG}: {changed.split()}; no figure input can be made")
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


def load_cache(key, seed, *, parts=("layout", "spindle", "lite", "partial", "plite", "queries", "D"), covs=None):
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
        if len(covs) != meta["n_tiles"] or abs(cov_checksum(covs) - meta["cov_checksum"]) > 1e-6 * abs(
                meta["cov_checksum"]):
            raise RuntimeError(f"{key}: covariances differ from those the cache was built from")
        if hasattr(c, "layout"):
            c.layout.spd_matrices = [covs[i]["cov"] for i in c.train]
    return c
