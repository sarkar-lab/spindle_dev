"""Fig 2D/E: build cost of the four indexes against the number of cells (tonsil) and of genes (lymph node 5k).

Usage (one point per process, so its peak memory is that point's alone; timed, one job at a time on one node):
  sbatch --array=0-10%1 experiments/common/timing.sbatch experiments/fig2_cost/c_ladder.py cells
  sbatch --array=0-2%1 --mem=740G experiments/common/timing.sbatch experiments/fig2_cost/c_ladder.py genes_pilot
  sbatch --array=0-5%1 --mem=740G experiments/common/timing.sbatch experiments/fig2_cost/c_ladder.py genes
  python experiments/fig2_cost/c_ladder.py cells|genes|genes_pilot [task id]   (no id: $SLURM_ARRAY_TASK_ID)

Cell ladder (D): tonsil_reactive (1.35 M cells, 377 genes), CELLS[t] cells drawn without replacement (rng seed 0;
None = the whole section), tiled anew, gene cap as the dataset's (800, so all 377 genes).
Gene ladder (E): lymph_node_5k (all cells, 4,624 genes) with the top GENES[t] genes by variance. GENES_PILOT finds
the largest p the current builder fits in the node's memory (no src/ change; the paper reports the range that fits).
Every point: split seed 0 (10 % held out, as the index cache), then the same stages as c_index_cache.py, each timed
with the peak resident memory after it (monotone over the process): load (+ subsample), prepare, layout, spindle,
lite, partial, plite. build_s = prepare + layout + spindle + lite + partial + plite (as builds.csv). Bytes as the
cache. No index is saved.

Writes results/fig2_cost/ladder/<ladder>_<x>.json; c_ladder_summary.py collects them into ladder.csv.
"""

import json
import os
import resource
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402
import spindle_dev as sd  # noqa: E402

CELLS = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, None]
GENES = [250, 400, 500, 750, 1000, 1500]  # 2,000 hits the layout's recursion limit; user: stop at 1,500
GENES_PILOT = [1000, 1500, 2000]
OUT = data.RESULTS / "fig2_cost" / "ladder"
SEED = 0


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20  # kB on Linux


def run(ladder, x):
    key = "tonsil_reactive" if ladder == "cells" else "lymph_node_5k"
    name = f"{ladder}_{'full' if x is None else x}"
    rec = {"ladder": ladder, "key": key, "x": x, **data.stamp(), "host": socket.gethostname(),
           "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "cpus": os.environ.get("SLURM_CPUS_PER_TASK", "")}

    def stage(stage_name, fn):
        t = time.perf_counter()
        res = fn()
        rec[f"{stage_name}_s"] = time.perf_counter() - t
        rec[f"peak_gb_{stage_name}"] = peak_gb()
        print(f"{name}: {stage_name} {rec[f'{stage_name}_s']:.1f} s, peak {rec[f'peak_gb_{stage_name}']:.1f} GB",
              flush=True)
        return res

    def load():
        adata = sd.load_adata(data.paths.dataset_path(key))
        if ladder == "cells" and x is not None:
            keep = np.sort(np.random.default_rng(SEED).choice(adata.n_obs, size=x, replace=False))
            adata = adata[keep].copy()
        return adata

    max_genes = data.paths.dataset_max_genes(key) if ladder == "cells" else x
    adata = stage("load", load)
    tiles, covs, genes = stage("prepare", lambda: sd.prepare_tiles(adata, max_genes=max_genes))
    train, test = data.split(len(tiles), SEED)
    tr_covs = [covs[i] for i in train]
    layout = stage("layout", lambda: sd.build_layout([tiles[i] for i in train], tr_covs, genes, adata.n_obs))
    spindle = stage("spindle", lambda: sd.Spindle(layout, tr_covs))
    lite = stage("lite", lambda: sd.SpindleLite.from_spindle(spindle, layout))
    partial = stage("partial", lambda: sd.SpindlePartial(layout, tr_covs))
    plite = stage("plite", lambda: sd.SpindleLitePartial.from_partial(partial, layout))

    p, n = len(genes), len(train)
    rec.update(n_cells=int(adata.n_obs), p=p, n_tiles=len(tiles), n_train=n, n_test=len(test),
               n_niches=len(layout.block_dict), n_blocks=int(sum(len(b) for b in layout.block_dict.values())),
               bytes_whole=4 * n * p * (p + 1) // 2, bytes_spindle=spindle.nbytes(), bytes_lite=lite.nbytes(),
               bytes_partial=partial.nbytes(), bytes_plite=plite.nbytes(), plite_spindle_bytes=plite.spindle_bytes)
    rec["build_s"] = sum(rec[f"{s}_s"] for s in ("prepare", "layout", "spindle", "lite", "partial", "plite"))
    rec["peak_gb"] = peak_gb()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rec, indent=1))
    print(f"{name}: {n} index tiles, {rec['n_niches']} niches, {rec['n_blocks']} blocks, build {rec['build_s']:.1f} s,"
          f" peak {rec['peak_gb']:.1f} GB; done", flush=True)


def main():
    ladder = sys.argv[1]
    t = int(sys.argv[2] if len(sys.argv) > 2 else os.environ["SLURM_ARRAY_TASK_ID"])
    xs = {"cells": CELLS, "genes": GENES, "genes_pilot": GENES_PILOT}[ladder]
    run("genes" if ladder == "genes_pilot" else ladder, xs[t])


if __name__ == "__main__":
    main()
