"""Fig 8 inputs: four breast gene programs searched as queries in five tumour sections (co-variation framing).

Usage (sbatch): first the exemplars (breast seed 0), then the array (task = 5 * dataset + seed):
  sbatch --job-name=f8_ex experiments/common/run.sbatch experiments/fig8_pan_cancer/c_xtumour.py exemplar
  sbatch --job-name=f8_x --dependency=afterok:<job> --array=0-24 experiments/common/run.sbatch \
      experiments/fig8_pan_cancer/c_xtumour.py
Writes results/fig8_pan_cancer/{exemplar.csv, exemplar.npz} and <key>/seed<s>/{programs.csv, hits.csv}.

Programs: signatures.PAN_CANCER_MODULES (cytotoxic T, myeloid, stromal / perivascular, B / plasma), every gene an index
gene of all five sections (checked; a missing gene stops the run). Sections: breast, lung, pancreas, brain, skin.

RULES (fixed before the run):
- EXEMPLAR, one per program, fixed for every section and seed: among breast's seed-0 index tiles, the tile with the
  largest robust co-variation c on S (common8.robust_c, the gene-set pilot's rule: c = mean off-diagonal correlation
  of S over the tile's cells, raw counts; >= 80 cells, >= 0.5 S counts per cell, no cell > 15 % of the S counts,
  dropping the top 3 cells keeps >= 70 % of c). No robust tile -> the program is dropped (reported). R_S = the
  correlation of its shrunk covariance cut to S (sd.tile_query); q_g = the share of breast seed-0 index tiles whose
  shrunk variance of gene g is <= the exemplar's.
- QUERY in section X, seed s (quantile-matched scale, Fig 8 decision of 9 Oct): sigma_g = sqrt of the q_g quantile
  of gene g's shrunk variance over X's seed-s index tiles; Q = diag(sigma) R_S diag(sigma).
- HITS = the first c = 5 % of the index tiles (common8.n_hits): Spindle-partial ("P", exact d_B^S); Spindle-lite-
  partial ("PL", the first c of search_vectors at the default g; fewer if it returns fewer, recorded). Controls,
  computed and reported, never claimed against: "D" = diag(sigma^2) (variances only, Spindle-partial) and "X" = the c
  index tiles with the highest tile score. The exemplar tile is never a hit and is left out of every background.
- READOUTS per (program, section, seed, method), against the other index tiles (background):
  gain = median over the background of the correlation gap minus its median over the hits (gap = RMS difference
  between a tile's S x S correlation, from its shrunk covariance, and R_S, off-diagonal pairs; > 0 = the hits
  co-vary more like the exemplar than a typical tile) [headline, C];
  delta = Cliff's delta of the tile score (mean over S of the z-scored tile-mean log-normalised expression) of the
  hits vs the background [D; delta <= 0 = programme absent, drawn open];
  lift = median over the hits minus median over all index tiles of c without the robust filter (Figs 6-7) [text];
  Jaccard with P, D and X [text].
- MAPS (B): seed 0, every section's cells and each programme's Spindle-partial hits (tile boxes).
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common8 as c8  # noqa: E402
from common8 import c6, c7, data, sd  # noqa: E402


def exemplars():
    S = c7.load(c8.SOURCE, data.EXAMPLE_SEED)
    c, covs = S["c"], S["covs"]
    rows, arrays = [], {}
    for prog in c8.PROGRAMS:
        idx, found = c8.program_idx(S["genes"], prog)
        cs = np.array([c8.robust_c(S["X"][S["tiles"][t].idx][:, idx]) for t in c.train])
        if not np.isfinite(cs).any():
            print(f"{prog}: no robust tile, dropped", flush=True)
            continue
        r = int(np.nanargmax(cs))
        ex = int(c.train[r])
        subs = c8.sub_covs(covs, c.train, idx)
        V = np.einsum("nii->ni", subs)
        R = c8.to_corr(subs[r])
        q = (V <= V[r]).mean(0)
        iu = np.triu_indices(len(idx), 1)
        rows.append({"program": prog, "genes": " ".join(found), "n_genes": len(idx), "tile": ex,
                     "n_cells": len(S["tiles"][ex].idx), "c": float(cs[r]), "mean_r": float(R[iu].mean()),
                     "n_robust": int(np.isfinite(cs).sum()), "q": " ".join(f"{v:.3f}" for v in q),
                     "tag": S["stamp"]["tag"]})
        arrays[f"R_{prog}"], arrays[f"q_{prog}"] = R, q
        arrays[f"bbox_{prog}"] = np.asarray(S["tiles"][ex].bbox, dtype=np.float64)
        print(f"{prog}: tile {ex} ({rows[-1]['n_cells']} cells), c {cs[r]:.3f}, mean r {R[iu].mean():.3f}, "
              f"{rows[-1]['n_robust']} robust tiles, q {np.round(q, 2)}", flush=True)
    c8.OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(c8.OUT / "exemplar.csv", index=False)
    np.savez_compressed(c8.OUT / "exemplar.npz", programs=np.array([r["program"] for r in rows]), xy=S["xy"],
                        **arrays)


def main(key, seed):
    E = pd.read_csv(c8.OUT / "exemplar.csv")
    Z = np.load(c8.OUT / "exemplar.npz")
    S = c7.load(key, seed, parts=("layout", "partial", "plite"))
    c, covs, st = S["c"], S["covs"], S["stamp"]
    n = len(c.train)
    k = c8.n_hits(n)
    mu = c6.tile_means(S, c.train)
    z = (mu - mu.mean(0)) / (mu.std(0) + 1e-9)
    cells = [S["tiles"][t].idx for t in c.train]
    rows, hits = [], []
    for _, e in E.iterrows():
        prog = e.program
        idx, found = c8.program_idx(S["genes"], prog)
        R, q = Z[f"R_{prog}"], Z[f"q_{prog}"]
        subs = c8.sub_covs(covs, c.train, idx)
        V = np.einsum("nii->ni", subs)
        sig = np.sqrt([np.quantile(V[:, j], q[j]) for j in range(len(idx))])
        Q = sig[:, None] * R * sig[None, :]
        np.linalg.cholesky(Q)
        valid = np.ones(n, bool)
        if key == c8.SOURCE:
            valid[c.train == int(e.tile)] = False
        score = z[:, idx].mean(1)
        gap = c8.corr_gap(subs, R)
        cov_all = np.array([c6.mean_offdiag_corr(S["X"][cl][:, idx]) for cl in cells])

        def first(order):
            return np.asarray([r for r in order if valid[r]][:k], dtype=int)

        tops = {}
        for m, Qm in (("P", Q), ("D", np.diag(sig ** 2))):
            tops[m] = first(np.argsort(c.partial.distances(Qm, idx), kind="stable"))
        lite_rows, _, _ = c.plite.search_vectors(c.plite.query_vectors(Q, idx))
        tops["PL"] = first(lite_rows)
        tops["X"] = first(np.argsort(-score, kind="stable"))
        for m, top in tops.items():
            back = np.setdiff1d(np.flatnonzero(valid), top)
            rows.append({"key": key, "seed": seed, "program": prog, "n_genes": len(idx), "method": m, "n_index": n,
                         "c": k, "n_returned": len(top), "lite_returned": len(lite_rows),
                         "gap_hits": float(np.median(gap[top])), "gap_back": float(np.median(gap[back])),
                         "delta": float(c6.cliffs_delta(score[top], score[back])),
                         "cov_top": float(np.nanmedian(cov_all[top])), "cov_all": float(np.nanmedian(cov_all[valid])),
                         **{f"jacc_{o}": c6.jaccard(top, tops[o]) for o in ("P", "D", "X")}, "tag": st["tag"]})
            hits += [{"key": key, "seed": seed, "program": prog, "method": m, "rank": i + 1, "tile": int(c.train[r]),
                      "gap": float(gap[r]), "score": float(score[r])} for i, r in enumerate(top)]
    T = pd.DataFrame(rows)
    T["gain"] = T.gap_back - T.gap_hits
    T["lift"] = T.cov_top - T.cov_all
    out = c8.out_dir(key, seed)
    T.to_csv(out / "programs.csv", index=False)
    pd.DataFrame(hits).to_csv(out / "hits.csv", index=False)
    print(T[["program", "method", "n_returned", "gain", "delta", "lift", "jacc_P", "jacc_D"]].round(3)
          .to_string(index=False), flush=True)
    print(f"done {key} seed {seed} (c = {k} of {n})", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "exemplar":
        exemplars()
    else:
        main(*c8.task(int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))))
