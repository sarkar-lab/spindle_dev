"""Fig 8 gradient maps: every index tile's distance to each programme query, seed 0, on the five tumour sections.

Usage (sbatch array, task = section): sbatch --array=0-4 experiments/common/run.sbatch \
    experiments/fig8_pan_cancer/c_gradient.py
Needs results/fig8_pan_cancer/exemplar.{csv,npz} (c_xtumour.py exemplar). Writes
results/fig8_pan_cancer/<key>/seed0/gradient.npz: xy (cells), centre and side of every index tile, and per programme
d_<programme> = Spindle-partial's exact d_B^S of every index tile to the query (index-tile order; the breast exemplar
tile included, it is the query's own source). The query is c_xtumour.py's (R_S at quantile-matched variances).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common8 as c8  # noqa: E402
from common8 import c7, data  # noqa: E402


def main(key):
    seed = data.EXAMPLE_SEED
    E = pd.read_csv(c8.OUT / "exemplar.csv")
    Z = np.load(c8.OUT / "exemplar.npz")
    S = c7.load(key, seed, parts=("layout", "partial"))
    c, covs = S["c"], S["covs"]
    box = np.array([S["tiles"][t].bbox for t in c.train], dtype=np.float64)
    out = {"xy": S["xy"], "tiles": c.train, "centre": (box[:, :2] + box[:, 2:]) / 2,
           "side": np.sqrt((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])), "programs": E.program.to_numpy()}
    for prog in E.program:
        idx, _ = c8.program_idx(S["genes"], prog)
        R, q = Z[f"R_{prog}"], Z[f"q_{prog}"]
        V = np.einsum("nii->ni", c8.sub_covs(covs, c.train, idx))
        sig = np.sqrt([np.quantile(V[:, j], q[j]) for j in range(len(idx))])
        out[f"d_{prog}"] = c.partial.distances(sig[:, None] * R * sig[None, :], idx)
        print(f"{key} {prog}: d {np.min(out[f'd_{prog}']):.3f}-{np.max(out[f'd_{prog}']):.3f}", flush=True)
    np.savez_compressed(c8.out_dir(key, seed) / "gradient.npz", tag=S["stamp"]["tag"], **out)


if __name__ == "__main__":
    main(c8.KEYS[int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))])
