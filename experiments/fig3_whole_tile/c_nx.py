"""Spindle-lite against Spindle on every held-out tile (Fig 3B; supplement: per-seed NX, band plots, g trade-off).

Usage: sbatch --array=0-39 experiments/common/run.sbatch experiments/fig3_whole_tile/c_nx.py
       python experiments/fig3_whole_tile/c_nx.py [task id 0-39 | key seed]   (no argument: $SLURM_ARRAY_TASK_ID)
Task t = dataset indexed_keys()[t // 5], seed SEEDS[t % 5]. The cached Spindle-lite (beta = 0.10) is searched from the
cached query vectors (SpindleLite.search_vectors, compiled) at every g in common3.GS; the exact d_B is the cache's
D.npy (Spindle.distances). Readouts per (query, g), common3: NX5, Overlap(5 %, 5 % N), MX10, tiles returned,
short of 10, work (node distances computed) and work / all nodes, A_hat, radius, nearest exact d_B.

Declared before the run (Fig 3B):
- Headline: NX5 at g = 0.25; per (dataset, seed) the mean over held-out queries, then mean +/- s.d. over
  seeds 0-4; the text quotes the median over datasets. Expected (19 Oct, seed 73, 100 queries): NX5 about 11 %
  (8.5-14.4 %); a large mismatch goes to the user.
- Typical query of the band plot (B1): breast_cancer, seed 0, g = 0.25; among the held-out queries that return at
  least 10 tiles, the one whose NX5 is the median (lower middle when even, ties by test order). Base rate stated with
  it: the share of queries that return fewer than 10 tiles. Picked in f_band.py from the seed-0 band file.

Writes results/fig3_whole_tile/per_query/<key>_seed<k>.csv (gitignored) and, for seed 0,
results/fig3_whole_tile/per_query/<key>_band.npz: D (float32, held-out x index), the rows returned at g = 0.25
(concatenated, offsets), test order.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common3 as c3  # noqa: E402
from common3 import data  # noqa: E402


def run(key, seed):
    st = data.stamp()
    c = data.load_cache(key, seed, parts=("lite", "queries", "D"))
    lite, D = c.lite, c.D
    n = D.shape[1]
    c5 = c3.c5(n)
    n_nodes = lite.n_nodes()
    out, found0 = [], []
    for j, qv in enumerate(c.qvs):
        d = D[j]
        for g in c3.GS:
            rows, _, s = lite.search_vectors(qv, g=g)
            out.append({"key": key, "seed": seed, "query": j, "test_tile": int(c.test[j]), "g": g, "N": n,
                        "c5": c5, "n_found": len(rows), "short10": len(rows) < 10, "short5": len(rows) < c5,
                        "NX5": c3.nx(d, rows, c5),
                        "overlap5": c3.overlap(d, rows, c5), "MX10": c3.mx10(d, rows), "work": s["work"],
                        "work_frac": s["work"] / n_nodes, "a_hat": s["a_hat"], "radius": s["radius"],
                        "d_nearest": float(d.min()), "tag": st["tag"], "describe": st["describe"]})
            if g == c3.G_DEFAULT and seed == data.EXAMPLE_SEED:
                found0.append(rows)
    dst = c3.OUT / "per_query"
    dst.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out)
    df.to_csv(dst / f"{key}_seed{seed}.csv", index=False)
    if found0:
        np.savez_compressed(dst / f"{key}_band.npz", D=D.astype(np.float32), found=np.concatenate(found0),
                            off=np.cumsum([0] + [len(r) for r in found0]), test=c.test, tag=st["tag"])
    m = df[df.g == c3.G_DEFAULT]
    print(f"{key} seed {seed}: {len(m)} queries, NX5 {100 * m.NX5.mean():.2f} %, "
          f"short10 {100 * m.short10.mean():.1f} %, work {m.work.median():.0f} / {n_nodes} nodes", flush=True)


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        t = int(sys.argv[1] if len(sys.argv) == 2 else os.environ["SLURM_ARRAY_TASK_ID"])
        run(data.indexed_keys()[t // len(data.SEEDS)], data.SEEDS[t % len(data.SEEDS)])
    else:
        run(sys.argv[1], int(sys.argv[2]))
