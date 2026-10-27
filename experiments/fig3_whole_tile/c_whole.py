"""Whole-matrix log vectors of every tile and the exact whole-matrix distance d_W of every held-out tile (Fig 3A, C).

Usage: sbatch --array=0-7 experiments/common/run.sbatch experiments/fig3_whole_tile/c_whole.py
       python experiments/fig3_whole_tile/c_whole.py [task id 0-7 | key]   (no argument: $SLURM_ARRAY_TASK_ID)
Task t = dataset indexed_keys()[t]. The vectors do not depend on the split, so they are computed once per dataset
(common3.whole_vectorizer on prepare()'s shrunk covariances, the ones the cache was built from).
Writes (gitignored cache):
  results/_cache/<key>/whole.npy         (n_tiles, p(p+1)/2) float32, prepare() tile order
  results/_cache/<key>/whole.json        stamp, shape, cov checksum, seconds
  results/_cache/<key>/seed<k>/DW.npy    exact d_W, held-out x index tiles (float64; compiled scan, as D.npy)
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common3 as c3  # noqa: E402
from common3 import data  # noqa: E402
from spindle_dev.kernels import spindle_scan  # noqa: E402


def run(key):
    rec = {"key": key, **data.stamp()}
    t0 = time.perf_counter()
    _, tiles, covs, genes = data.prepare(key)
    p = len(genes)
    vec = c3.whole_vectorizer(p)
    W = np.empty((len(covs), vec.dim), dtype=np.float32)
    for i, c in enumerate(covs):
        W[i] = vec.vector(c["cov"], np.float32)
    rec.update(n_tiles=len(covs), p=p, dim=vec.dim, cov_checksum=data.cov_checksum(covs),
               vectors_s=time.perf_counter() - t0)
    np.save(c3.whole_path(key), W)
    print(f"{key}: {W.shape} whole vectors in {rec['vectors_s']:.0f} s", flush=True)

    for seed in data.SEEDS:
        meta = json.loads((data.cache_dir(key, seed) / "build.json").read_text())
        if abs(rec["cov_checksum"] - meta["cov_checksum"]) > 1e-6 * abs(meta["cov_checksum"]):
            raise RuntimeError(f"{key} seed {seed}: covariances differ from the cache")
        s = np.load(data.cache_dir(key, seed) / "split.npz")
        X = np.ascontiguousarray(W[s["train"]])
        rows = np.arange(len(X), dtype=np.int64)
        DW = np.empty((len(s["test"]), len(X)))
        for j, t in enumerate(s["test"]):
            spindle_scan(X, W[t], rows, DW[j])
        np.save(data.cache_dir(key, seed) / "DW.npy", DW)
        print(f"{key} seed {seed}: d_W {DW.shape}", flush=True)
    rec["total_s"] = time.perf_counter() - t0
    (c3.whole_path(key).with_suffix(".json")).write_text(json.dumps(rec, indent=1))


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else os.environ["SLURM_ARRAY_TASK_ID"]
    run(data.indexed_keys()[int(a)] if a.isdigit() else a)
