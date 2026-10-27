"""Cross-platform search, Xenium <-> Visium (Fig 5C, E; text: Spindle-lite, co-located rank).

Usage: sbatch --array=0-4 experiments/common/run.sbatch experiments/fig5_cross_platform/c_cross.py
       python experiments/fig5_cross_platform/c_cross.py [seed]       (no argument: $SLURM_ARRAY_TASK_ID)
Per seed (0-4; it seeds only build_layout's PCA / UMAP / Leiden, every tile is indexed) and direction (x2v: Xenium
queries against a Visium index; v2x: the reverse), the index platform's tiles get a layout, Spindle and Spindle-lite
(from_spindle, beta = 0.10). Every tile of the query platform is a query (common5: tiles, correction arms).

Methods ranked per query (all index tiles, best first):
  spindle_<arm>   Spindle.distances (d_B) of the corrected query, arm in none / per_niche / global
  lite            SpindleLite.search_vectors (g = 0.25) of the global-corrected query; graded against spindle_global

Declared before the run (Fig 5; user, 26 Oct):
- Headline: tissue agreement of spindle_global = Pearson r over queries between the query tile's tumour fraction
  (invasive + DCIS cells or spots, own platform's annotation) and the mean tumour fraction of its top 10 on the other
  platform; per (direction, seed), then mean +/- s.d. over seeds 0-4. Base rate: the mean absolute difference to
  all index tiles (random) beside the top-10 one. Expected (6 Oct, old code): r about 0.85-0.89.
- Ablation: the same r for none / per_niche / global. The expression and variance-only controls were run once
  (26 Oct, jobs 23272/23279: r 0.93-0.94 and 0.78-0.82) and dropped from the figure and text (user, 26 Oct).
- Spindle-lite (text only): NX5 and Overlap(5 %, 5 % N) against spindle_global, mean over queries per seed.
- Co-located tile rank (hit@1/5/10) is kept in the per-query file; no panel (user, 26 Oct).

Writes results/fig5_cross_platform/per_query/seed<s>.csv (method, direction, query, query / hits tumour fraction,
co-located rank), seed<s>_top10.csv (top 10 per method), seed<s>_lite.csv, seed<s>_index.csv (gitignored dumps).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common5 as c5  # noqa: E402
from common5 import data, sd  # noqa: E402

sys.path.insert(0, str(data.ROOT / "experiments" / "fig3_whole_tile"))
import common3 as c3  # noqa: E402  (nx, overlap, c5: the Fig 3 readouts)


def rank_of(d, t):
    """Tie-aware rank (1 = nearest) of index tile t."""
    return 1 + int((d < d[t]).sum()) + 0.5 * int((d == d[t]).sum() - 1)


def run(seed):
    st = data.stamp()
    S = c5.setup()
    per_q, top, lite_rows, info = [], [], [], []
    for direction, (index_mod, query_mod) in c5.DIRECTIONS.items():
        t_covs, q_covs = S["covs"][index_mod], S["covs"][query_mod]
        layout = sd.build_layout(S["tiles"][index_mod], t_covs, S["genes"], S["n_obs"][index_mod],
                                 random_state=seed)
        spindle = sd.Spindle(layout, t_covs)
        lite = sd.SpindleLite.from_spindle(spindle, layout)
        n = spindle.n
        qv = {"none": [spindle.query_vectors(c["cov"]) for c in q_covs],
              "per_niche": c5.per_niche_vectors(spindle, q_covs),
              "global": [spindle.query_vectors(c) for c in c5.global_corrected(q_covs, t_covs)]}
        D = {f"spindle_{arm}": np.stack([spindle.distances(v) for v in qv[arm]]) for arm in c5.ARMS}
        matched = c5.match_of(S, direction)
        tum_q, tum_t = c5.tumour(S, query_mod), c5.tumour(S, index_mod)
        base = {"seed": seed, "direction": direction, "N": n, "tag": st["tag"], "describe": st["describe"]}
        for method, Dm in D.items():
            for q, d in enumerate(Dm):
                order = np.argsort(d, kind="stable")
                rank = rank_of(d, matched[q]) if matched[q] >= 0 else np.nan
                per_q.append({**base, "method": method, "query": q, "matched_tile": int(matched[q]),
                              "query_tumour": tum_q[q], "hits_tumour": tum_t[order[:c5.TOP]].mean(),
                              "random_mae": float(np.abs(tum_t - tum_q[q]).mean()), "matched_rank": rank,
                              **{f"hit@{k}": float(rank <= k) if matched[q] >= 0 else np.nan for k in (1, 5, 10)}})
                top += [{**base, "method": method, "query": q, "rank": r + 1, "index_tile": int(t), "d": d[t]}
                        for r, t in enumerate(order[:c5.TOP])]
        cc = c3.c5(n)
        for q, v in enumerate(qv["global"]):
            rows, _, s = lite.search_vectors(v)
            d = D["spindle_global"][q]
            lite_rows.append({**base, "query": q, "c5": cc, "n_found": len(rows), "NX5": c3.nx(d, rows, cc),
                              "overlap5": c3.overlap(d, rows, cc), "work": s["work"]})
        info.append({**base, "index_platform": index_mod, "n_query": len(q_covs), "p": len(S["genes"]),
                     "n_niches": len(spindle.layouts), "n_blocks": int(sum(len(v.runs) for _, v in
                                                                          spindle.layouts.values())),
                     "bytes_spindle": spindle.nbytes(), "bytes_lite": lite.nbytes()})
        pq = pd.DataFrame([r for r in per_q if r["direction"] == direction])
        r = pq.groupby("method").apply(lambda g: np.corrcoef(g.query_tumour, g.hits_tumour)[0, 1])
        lr = pd.DataFrame([x for x in lite_rows if x["direction"] == direction])
        print(f"seed {seed} {direction}: N = {n}, {len(spindle.layouts)} niches; tissue r "
              f"{r.round(3).to_dict()}; lite NX5 {100 * lr.NX5.mean():.1f} %, overlap {lr.overlap5.mean():.2f}",
              flush=True)
        if seed == data.EXAMPLE_SEED and direction == "v2x":
            check(spindle, S)
    dst = c5.OUT / "per_query"
    dst.mkdir(parents=True, exist_ok=True)
    for name, rows in (("", per_q), ("_top10", top), ("_lite", lite_rows), ("_index", info)):
        pd.DataFrame(rows).to_csv(dst / f"seed{seed}{name}.csv", index=False)


def check(spindle, S):
    """The Xenium index searched with its own tiles: rank 1 at distance 0; the global shift of a platform onto
    itself is the identity."""
    covs = S["covs"]["Xenium"]
    D = np.stack([spindle.distances(spindle.query_vectors(c["cov"])) for c in covs])
    own = np.arange(len(covs))
    print(f"check: self rank 1 for {np.mean(D.argmin(1) == own):.3f} of tiles, max self d {D[own, own].max():.1e}; "
          f"global shift onto itself: max |d - d_none| = "
          f"{np.abs(np.stack([spindle.distances(spindle.query_vectors(c)) for c in c5.global_corrected(covs, covs)]) - D).max():.1e}",
          flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1] if len(sys.argv) > 1 else os.environ["SLURM_ARRAY_TASK_ID"]))
