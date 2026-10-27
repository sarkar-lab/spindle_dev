"""Query time of the three whole-tile searches (Fig 3A): whole-matrix log scan, Spindle, Spindle-lite.

Usage: sbatch --array=0-7%1 experiments/common/timing.sbatch experiments/fig3_whole_tile/c_timing.py
       python experiments/fig3_whole_tile/c_timing.py [task id 0-7 | key]   (no argument: $SLURM_ARRAY_TASK_ID)
Needs c_whole.py first (results/_cache/<key>/whole.npy).

Like for like (CLAUDE.md Benchmarks): one thread, compiled searches only, from query vectors
computed beforehand; both exact searches store float32 logs of every index tile and run the same C kernel
(kernels.spindle_scan), so they differ only in what they store (whole matrices or the niches' gene blocks):
  whole    spindle_scan over the whole-matrix vectors + argsort (stable)
  spindle  Spindle.distances + argsort (stable)            (as Spindle.search with c = None)
  lite     SpindleLite.search_vectors, g = 0.25            (returns its tiles sorted)
The query's own logs are timed separately: qlog_whole = the whole-matrix vector (one p x p eigendecomposition),
qlog_block = Spindle.query_vectors (the block logs in every niche; shared by Spindle and Spindle-lite).
Queries: per seed 0-4 the first 200 held-out tiles in test order (all when fewer); 3 untimed warm-up queries per
seed; each step timed once per query (perf_counter). Readout: median ms per seed, mean +/- s.d. over seeds.

Writes results/fig3_whole_tile/timing/<key>.csv: one row per (seed, query) with ms of every step.
"""

import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common3 as c3  # noqa: E402
from common3 import data  # noqa: E402
from spindle_dev.kernels import spindle_scan  # noqa: E402

N_TIMED = 200
N_WARM = 3


def run(key):
    st = data.stamp()
    _, _, covs, genes = data.prepare(key)
    W = np.load(c3.whole_path(key))
    wvec = c3.whole_vectorizer(len(genes))
    out = []
    for seed in data.SEEDS:
        c = data.load_cache(key, seed, parts=("spindle", "lite", "queries"))
        sp, lite = c.spindle, c.lite
        X = np.ascontiguousarray(W[c.train])
        rows_all = np.arange(len(X), dtype=np.int64)
        dw = np.empty(len(X))
        qws = [W[t] for t in c.test[:N_TIMED]]
        idx = list(range(min(N_TIMED, len(c.test))))
        for j in idx[:N_WARM]:
            spindle_scan(X, qws[j], rows_all, dw)
            sp.distances(c.qvs[j])
            lite.search_vectors(c.qvs[j])
        for j in idx:
            cov = covs[c.test[j]]["cov"]
            t0 = time.perf_counter()
            wvec.vector(cov, np.float32)
            t1 = time.perf_counter()
            sp.query_vectors(cov)
            t2 = time.perf_counter()
            spindle_scan(X, qws[j], rows_all, dw)
            np.argsort(dw, kind="stable")
            t3 = time.perf_counter()
            np.argsort(sp.distances(c.qvs[j]), kind="stable")
            t4 = time.perf_counter()
            rows, _, s = lite.search_vectors(c.qvs[j])
            t5 = time.perf_counter()
            out.append({"key": key, "seed": seed, "query": j, "test_tile": int(c.test[j]), "N": len(X),
                        "qlog_whole_ms": 1e3 * (t1 - t0), "qlog_block_ms": 1e3 * (t2 - t1),
                        "whole_ms": 1e3 * (t3 - t2), "spindle_ms": 1e3 * (t4 - t3), "lite_ms": 1e3 * (t5 - t4),
                        "lite_found": len(rows), "lite_work": s["work"], "host": socket.gethostname(),
                        "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "tag": st["tag"],
                        "describe": st["describe"]})
        m = pd.DataFrame(out[-len(idx):]).median(numeric_only=True)
        print(f"{key} seed {seed}: {len(idx)} queries; median ms whole {m.whole_ms:.2f}, spindle {m.spindle_ms:.2f}, "
              f"lite {m.lite_ms:.3f}; query logs whole {m.qlog_whole_ms:.1f}, block {m.qlog_block_ms:.1f}",
              flush=True)
        del X
    dst = c3.OUT / "timing"
    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(dst / f"{key}.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else os.environ["SLURM_ARRAY_TASK_ID"]
    run(data.indexed_keys()[int(a)] if a.isdigit() else a)
