"""Query time of gene-set search (Fig 4E): the exact gene-set scan, Spindle-lite-partial and the two baselines.

Usage: sbatch --array=0-7%1 experiments/common/timing.sbatch experiments/fig4_gene_set/c_timing.py
       python experiments/fig4_gene_set/c_timing.py [task id 0-7 | key]   (no argument: $SLURM_ARRAY_TASK_ID)

Like for like (CLAUDE.md Benchmarks): one thread, compiled searches only:
  exact       SpindlePartial.distances + argsort (stable): eigendecomposes the S x S part of every index tile's
              stored block covariances (stored log when S covers a block); its query logs (one per niche and block
              of S) are computed inside, so they are timed again on their own right after, on the same query
  qlog_exact  common4.exact_query_logs: those query logs, as distances computes them (user, 26 Oct)
  exact_search  exact - qlog_exact per query: the exact scan without its query logs (like for like with lite)
  lite        SpindleLitePartial.search_vectors, g = 1, from the query pieces computed beforehand
  qlog_lite   SpindleLitePartial.query_vectors (the query's piece logs in every niche), reported separately
  padding     fill (common4.pad) + Spindle.query_vectors + Spindle.distances + argsort (stable)
  imputation  niche prediction + fill (common4.impute) + Spindle.query_vectors + Spindle.distances + argsort
              (the baselines' steps are recorded one by one: fill, query logs, scan)
Queries: per seed 0-4 the workload (common4.workload, |S| = 4 / 8 / 16 / 32) of the first 200 held-out tiles in test
order; 3 untimed warm-up queries per seed; each step timed once per query (perf_counter). Readout: median ms per
(seed, size) and per seed with all sizes pooled; mean +/- s.d. over seeds. Expected (19 Oct, CPU2, seed 73):
Spindle-lite-partial 32-66x faster than the exact scan.

Writes results/fig4_gene_set/timing/<key>.csv: one row per (seed, query, size) with ms of every step.
"""

import os
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common4 as c4  # noqa: E402
from common4 import data  # noqa: E402
import spindle_dev as sd  # noqa: E402

N_TIMED = 200
N_WARM = 3


def run(key):
    st = data.stamp()
    _, _, covs, _ = data.prepare(key)
    out = []
    for seed in data.SEEDS:
        c = data.load_cache(key, seed, parts=("layout", "spindle", "partial", "plite", "D"), covs=covs)
        sp, partial, plite = c.spindle, c.partial, c.plite
        p = partial.p
        means = c4.niche_means(c.layout, covs, c.train)
        work = [w for w in c4.workload(c.layout, c.D, seed) if w[0] < N_TIMED]
        qs = [sd.tile_query(covs[c.test[w[0]]]["cov"], w[5]) for w in work]
        for cov_s, idx in qs[:N_WARM]:
            partial.distances(cov_s, idx)
            c4.exact_query_logs(partial, cov_s, idx)
            plite.search_vectors(plite.query_vectors(cov_s, idx))
            sp.distances(sp.query_vectors(c4.pad(cov_s, idx, p)))
        for (j, home, L, b, a, genes), (cov_s, idx) in zip(work, qs):
            t0 = time.perf_counter()
            np.argsort(partial.distances(cov_s, idx), kind="stable")
            t1 = time.perf_counter()
            c4.exact_query_logs(partial, cov_s, idx)
            t1b = time.perf_counter()
            qp = plite.query_vectors(cov_s, idx)
            t2 = time.perf_counter()
            rows, _, s = plite.search_vectors(qp)
            t3 = time.perf_counter()
            full = c4.pad(cov_s, idx, p)
            t4 = time.perf_counter()
            qv = sp.query_vectors(full)
            t5 = time.perf_counter()
            np.argsort(sp.distances(qv), kind="stable")
            t6 = time.perf_counter()
            full = c4.impute(cov_s, idx, means[c4.predict_niche(cov_s, idx, means, partial.blocks)])
            t7 = time.perf_counter()
            qv = sp.query_vectors(full)
            t8 = time.perf_counter()
            np.argsort(sp.distances(qv), kind="stable")
            t9 = time.perf_counter()
            out.append({"key": key, "seed": seed, "query": j, "test_tile": int(c.test[j]), "size": L,
                        "N": partial.n, "exact_ms": 1e3 * (t1 - t0), "qlog_exact_ms": 1e3 * (t1b - t1),
                        "exact_search_ms": 1e3 * ((t1 - t0) - (t1b - t1)), "qlog_lite_ms": 1e3 * (t2 - t1b),
                        "lite_ms": 1e3 * (t3 - t2), "pad_fill_ms": 1e3 * (t4 - t3), "pad_qlog_ms": 1e3 * (t5 - t4),
                        "pad_scan_ms": 1e3 * (t6 - t5), "imp_fill_ms": 1e3 * (t7 - t6),
                        "imp_qlog_ms": 1e3 * (t8 - t7), "imp_scan_ms": 1e3 * (t9 - t8),
                        "lite_found": len(rows), "lite_work": s["work"], "host": socket.gethostname(),
                        "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "tag": st["tag"],
                        "describe": st["describe"]})
        m = pd.DataFrame(out[-len(work):]).median(numeric_only=True)
        print(f"{key} seed {seed}: {len(work)} queries; median ms exact {m.exact_ms:.2f} (search "
              f"{m.exact_search_ms:.2f}, query logs {m.qlog_exact_ms:.2f}), lite {m.lite_ms:.3f} "
              f"(pieces {m.qlog_lite_ms:.2f}); padding {m.pad_fill_ms + m.pad_qlog_ms + m.pad_scan_ms:.1f}, "
              f"imputation {m.imp_fill_ms + m.imp_qlog_ms + m.imp_scan_ms:.1f}", flush=True)
    dst = c4.OUT / "timing"
    dst.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(dst / f"{key}.csv", index=False)


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else os.environ["SLURM_ARRAY_TASK_ID"]
    run(data.indexed_keys()[int(a)] if a.isdigit() else a)
