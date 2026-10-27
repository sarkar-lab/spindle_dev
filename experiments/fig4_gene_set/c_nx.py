"""Gene-set search against the exact d_B^S on every held-out tile (Fig 4B, C, D): Spindle-lite-partial, padding and
imputation (common4).

Usage: sbatch --array=0-39 experiments/common/run.sbatch experiments/fig4_gene_set/c_nx.py
       python experiments/fig4_gene_set/c_nx.py [task id 0-39 | key seed]   (no argument: $SLURM_ARRAY_TASK_ID)
Task t = dataset indexed_keys()[t // 5], seed SEEDS[t % 5]. Workload: common4.workload (contiguous sets, |S| = 4 / 8 /
16 / 32). Per (query, size): exact d_B^S of every index tile (SpindlePartial.distances); Spindle-lite-partial from the
query pieces (SpindleLitePartial.query_vectors + search_vectors, compiled, g = 1); padding and imputation as ranked
lists of all index tiles (exact d_B of the completed query, stable sort). Readouts (common3, graded by the exact
d_B^S): NX5, Overlap(5 %, 5 % N); for Spindle-lite-partial also tiles returned, short of 10, work, A_hat, radius;
segments of S in the home niche and the largest number in any niche.

Declared before the run (Fig 4; user 26 Oct):
- Headline: NX5 of Spindle-lite-partial at g = 1; per (dataset, seed, size) the mean over held-out queries, then the
  mean +/- s.d. over seeds 0-4; per size the median over datasets (Fig 4C, with padding and imputation); per dataset
  all sizes pooled (Fig 4D). Expected (19 Oct, seed 73, 100 queries): median NX5 about 12 % (8.4-32.8 %, LN5k
  highest), per size 9.5 / 10.6 / 13.3 / 16.5 %; a large mismatch goes to the user.
- Typical query of the band plot (Fig 4B): breast_cancer, seed 0, |S| = 16; among the held-out queries whose
  Spindle-lite-partial list holds at least 10 tiles, the one whose NX5 is the median (lower middle when even, ties by
  test order). Base rate stated with it: the share of |S| = 16 queries that return fewer than 10 tiles. Picked in
  c_summary.py from the seed-0 band file.

Writes results/fig4_gene_set/per_query/<key>_seed<k>.csv (gitignored) and, for seed 0,
results/fig4_gene_set/per_query/<key>_band.npz: the exact d_B^S (float32, |S| = 16 queries x index tiles), the rows
Spindle-lite-partial returned (concatenated, offsets), the query's test position.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common4 as c4  # noqa: E402
from common4 import c3, data  # noqa: E402
import spindle_dev as sd  # noqa: E402


def run(key, seed):
    st = data.stamp()
    _, _, covs, _ = data.prepare(key)
    c = data.load_cache(key, seed, parts=("layout", "spindle", "partial", "plite", "D"), covs=covs)
    sp, partial, plite = c.spindle, c.partial, c.plite
    n, p = c.D.shape[1], partial.p
    c5 = c3.c5(n)
    means = c4.niche_means(c.layout, covs, c.train)
    out, band_d, band_rows, band_q = [], [], [], []
    for j, home, L, b, a, genes in c4.workload(c.layout, c.D, seed):
        cov_s, idx = sd.tile_query(covs[c.test[j]]["cov"], genes)
        d = partial.distances(cov_s, idx)
        rows, _, s = plite.search_vectors(plite.query_vectors(cov_s, idx))
        pad_rows = np.argsort(sp.distances(sp.query_vectors(c4.pad(cov_s, idx, p))), kind="stable")
        k_hat = c4.predict_niche(cov_s, idx, means, partial.blocks)
        imp_rows = np.argsort(sp.distances(sp.query_vectors(c4.impute(cov_s, idx, means[k_hat]))), kind="stable")
        seg = c4.n_segments(plite, idx)
        base = {"key": key, "seed": seed, "query": j, "test_tile": int(c.test[j]), "size": L, "home": home,
                "block": b, "start": a, "N": n, "c5": c5, "seg_home": seg[home], "seg_max": max(seg.values()),
                "d_nearest": float(d.min()), "tag": st["tag"], "describe": st["describe"]}
        for method, r in (("lite", rows), ("padding", pad_rows), ("imputation", imp_rows)):
            rec = {**base, "method": method, "NX5": c3.nx(d, r, c5), "overlap5": c3.overlap(d, r, c5)}
            if method == "lite":
                rec.update(n_found=len(r), short10=len(r) < 10, short5=len(r) < c5, work=s["work"],
                           a_hat=s["a_hat"], radius=s["radius"])
            if method == "imputation":
                rec.update(k_hat=k_hat, k_hat_home=k_hat == home)
            out.append(rec)
        if seed == data.EXAMPLE_SEED and L == c4.BAND_SIZE:
            band_d.append(d.astype(np.float32))
            band_rows.append(rows)
            band_q.append(j)
    dst = c4.OUT / "per_query"
    dst.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(out)
    df.to_csv(dst / f"{key}_seed{seed}.csv", index=False)
    if band_q:
        np.savez_compressed(dst / f"{key}_band.npz", D=np.stack(band_d), found=np.concatenate(band_rows),
                            off=np.cumsum([0] + [len(r) for r in band_rows]), query=np.asarray(band_q), tag=st["tag"])
    m = df.groupby(["method", "size"]).NX5.mean().unstack()
    print(f"{key} seed {seed}: {df['query'].nunique()} queries, NX5 % by size:\n{(100 * m).round(2).to_string()}",
          flush=True)


if __name__ == "__main__":
    if len(sys.argv) <= 2:
        t = int(sys.argv[1] if len(sys.argv) == 2 else os.environ["SLURM_ARRAY_TASK_ID"])
        run(data.indexed_keys()[t // len(data.SEEDS)], data.SEEDS[t % len(data.SEEDS)])
    else:
        run(sys.argv[1], int(sys.argv[2]))
