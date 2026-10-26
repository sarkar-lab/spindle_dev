"""The index cache: one build of the layout and all four indexes per (dataset, seed), timed (RESULTS_PLAN.md §6, S2).

Usage (one array task per dataset x seed, one at a time on one node; the build times are Fig 2C):
  sbatch --array=0-39%1 experiments/common/timing.sbatch experiments/common/c_index_cache.py
  python experiments/common/c_index_cache.py [task id 0-39 | key seed]    (no argument: $SLURM_ARRAY_TASK_ID)
Task t builds dataset indexed_keys()[t // 5] with seed SEEDS[t % 5]. Writes results/_cache/<key>/seed<k>/ (see
data.py) and results/_cache/<key>/seed<k>/build.json; c_index_cache_summary.py collects build.json into
results/index_cache/builds.csv.

Stages, each timed (perf_counter) with the peak resident memory after it (ru_maxrss, monotone over the process):
  load       load_adata
  prepare    prepare_tiles: tiles, top-variance genes, every tile's shrunk covariance (joblib threads, n_jobs = 8)
  layout     build_layout on the index tiles (split_tiles(seed, test_ratio = 0.1))
  spindle    Spindle(layout, covs)                       the exact whole-tile index (block logs)
  lite       SpindleLite.from_spindle                    reuses Spindle's block logs
  partial    SpindlePartial(layout, covs)                the exact gene-set index (block covariances + logs)
  plite      SpindleLitePartial.from_partial             reuses SpindlePartial's block covariances
Then, untimed: omega = variance_quantiles(index covs); the held-out tiles' Spindle.query_vectors and their exact d_B
to every index tile (Spindle.distances). Bytes: whole covariances as float32 upper triangles of the index tiles;
Spindle.nbytes, SpindleLite.nbytes, SpindlePartial.nbytes, SpindleLitePartial.nbytes.
"""

import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import spindle_dev as sd  # noqa: E402


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # kB on Linux


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
    spindle = stage("spindle", lambda: sd.Spindle(layout, tr_covs))
    lite = stage("lite", lambda: sd.SpindleLite.from_spindle(spindle, layout))
    partial = stage("partial", lambda: sd.SpindlePartial(layout, tr_covs))
    plite = stage("plite", lambda: sd.SpindleLitePartial.from_partial(partial, layout))

    p, n = len(genes), len(train)
    labels = np.asarray(layout.labels).astype(int)
    rec.update(n_cells=int(adata.n_obs), p=p, n_tiles=len(tiles), n_train=n, n_test=len(test),
               n_niches=len(layout.block_dict), n_blocks=int(sum(len(b) for b in layout.block_dict.values())),
               niche_sizes=np.bincount(labels).tolist(),
               cov_checksum=data.cov_checksum(covs),
               bytes_whole=4 * n * p * (p + 1) // 2, bytes_spindle=spindle.nbytes(), bytes_lite=lite.nbytes(),
               bytes_partial=partial.nbytes(), bytes_plite=plite.nbytes(), plite_spindle_bytes=plite.spindle_bytes,
               lite_nodes=lite.n_nodes(), plite_nodes=plite.n_nodes())
    rec["lite_pct"] = 100 * rec["bytes_lite"] / rec["bytes_spindle"]
    rec["plite_pct"] = 100 * rec["bytes_plite"] / plite.spindle_bytes

    omega = sd.variance_quantiles(tr_covs)
    qvs = [spindle.query_vectors(covs[i]) for i in test]
    D = np.stack([spindle.distances(qv) for qv in qvs])

    data.save_layout(layout, out / "layout.pkl")
    for name, obj in (("spindle", spindle), ("lite", lite), ("partial", partial), ("plite", plite), ("queries", qvs)):
        data._dump(obj, out / f"{name}.pkl")
    np.savez(out / "split.npz", train=train, test=test, genes=np.asarray(genes).astype(str), omega=omega)
    np.save(out / "D.npy", D)
    rec["peak_gb"] = peak_gb()
    (out / "build.json").write_text(json.dumps(rec, indent=1))
    print(f"{key} seed {seed}: {n} index / {len(test)} held-out tiles, {rec['n_niches']} niches, "
          f"lite {rec['lite_pct']:.2f} %, partial lite {rec['plite_pct']:.2f} % of Spindle; done", flush=True)


def main():
    if len(sys.argv) <= 2:
        t = int(sys.argv[1] if len(sys.argv) == 2 else os.environ["SLURM_ARRAY_TASK_ID"])
        key, seed = data.indexed_keys()[t // len(data.SEEDS)], data.SEEDS[t % len(data.SEEDS)]
    else:
        key, seed = sys.argv[1], int(sys.argv[2])
    build(key, seed)


if __name__ == "__main__":
    main()
