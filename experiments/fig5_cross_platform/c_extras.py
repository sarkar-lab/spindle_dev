"""Fig 5A, B, F inputs: the section maps, the platform-bias PCA and the example query.

Usage: sbatch --dependency=afterok:<c_cross job> experiments/common/run.sbatch experiments/fig5_cross_platform/c_extras.py
(needs per_query/seed0_top10.csv from c_cross.py).

  maps      tiles.csv (both platforms: box, cells / spots, composition, tumour fraction, co-located Xenium tile) and
            cells.npz (every Xenium cell and Visium spot: x, y, coarse class)
  bias PCA  bias_pca.csv, bias_pca_summary.csv: PCA (5 components) of the shrunk whole-matrix log vectors (upper
            triangle, off-diagonals x sqrt 2, / sqrt p), fit on the index tiles; the query tiles projected before and
            after the global shift (common5); platform silhouette (Euclidean, index vs query tiles) per state.
  example   example.csv (Fig 5F). Rule (user, 26 Oct; replaces the first rule, which picked the Visium query with the
            largest DCIS share, query 12: top 5 at 2-54 % DCIS, co-located tile at rank 54): seed 0, direction v2x
            (Visium query, Xenium index), spindle_global. A query's focal class is the larger of its DCIS and immune
            shares (classes in separate, recognisable foci on both platforms). Among the queries whose focal share is
            > 0.5 and whose co-located Xenium tile is in the top 5, the one whose top 5 have the largest mean share of
            the query's focal class (ties: lower query id). Reported with it: the top 5's share against the Xenium
            section's cell fraction of that class, the co-located tile's rank, the top 5 in seeds 1-4, and the base
            rate = the share of all focal-rich Visium queries (any co-located rank) whose top-5 mean exceeds the
            section's fraction.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common5 as c5  # noqa: E402
from common5 import data  # noqa: E402

N_PC = 5


def maps(S, st):
    rows = []
    for m in c5.PLATFORMS:
        tum = c5.tumour(S, m)
        for i, t in enumerate(S["tiles"][m]):
            rows.append({"platform": m, "tile": i, "x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2], "y1": t.bbox[3],
                         "n_obs": len(t.idx), "matched_xenium": int(S["matched_vi_to_xe"][i]) if m == "Visium" else i,
                         "tumour": tum[i], **dict(zip(c5.CLASSES, S["comp"][m][i])), "tag": st["tag"]})
    pd.DataFrame(rows).to_csv(c5.OUT / "tiles.csv", index=False)
    cells = S["cells"]
    np.savez_compressed(c5.OUT / "cells.npz", classes=np.array(c5.CLASSES),
                        **{f"{m}_xy": cells[m]["xy"].astype(np.float32) for m in c5.PLATFORMS},
                        **{f"{m}_class": np.array([c5.CLASSES.index(c) for c in cells[m]["coarse"]], dtype=np.int8)
                           for m in c5.PLATFORMS})


def vectorize(logs):
    p = logs.shape[1]
    iu = np.triu_indices(p)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)) / np.sqrt(p)
    return logs[:, iu[0], iu[1]] * w


def bias_pca(S, st):
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    logs = {m: c5.whole_logs(S["covs"][m]) for m in c5.PLATFORMS}
    rows, summary = [], []
    for direction, (index_mod, query_mod) in c5.DIRECTIONS.items():
        V_t = vectorize(logs[index_mod])
        states = {False: vectorize(logs[query_mod]), True: vectorize(c5.shift(logs[query_mod], logs[index_mod]))}
        pca = PCA(n_components=N_PC, random_state=0).fit(V_t)
        for k, z in enumerate(pca.transform(V_t)):
            rows.append({"direction": direction, "platform": index_mod, "role": "index", "corrected": False,
                         "tile": k, **{f"PC{j + 1}": z[j] for j in range(N_PC)}})
        for corr, V_q in states.items():
            for k, z in enumerate(pca.transform(V_q)):
                rows.append({"direction": direction, "platform": query_mod, "role": "query", "corrected": corr,
                             "tile": k, **{f"PC{j + 1}": z[j] for j in range(N_PC)}})
            lab = np.r_[np.zeros(len(V_t)), np.ones(len(V_q))]
            summary.append({"direction": direction, "corrected": corr,
                            "silhouette": float(silhouette_score(np.vstack([V_t, V_q]), lab)),
                            "centroid_gap": float(np.linalg.norm(V_t.mean(0) - V_q.mean(0))),
                            "pc12_var": float(pca.explained_variance_ratio_[:2].sum()), "tag": st["tag"]})
    pd.DataFrame(rows).to_csv(c5.OUT / "bias_pca.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(c5.OUT / "bias_pca_summary.csv", index=False)
    print(summ.round(3).to_string(index=False), flush=True)


def example(S, st):
    def top5(seed):
        t = pd.read_csv(c5.OUT / "per_query" / f"seed{seed}_top10.csv")
        t = t[(t.direction == "v2x") & (t.method == "spindle_global") & (t["rank"] <= 5)]
        return t.groupby("query")["index_tile"].apply(lambda x: x.to_numpy())
    focal = ["DCIS", "Immune"]
    j = [c5.CLASSES.index(c) for c in focal]
    comp_v, comp_x = S["comp"]["Visium"][:, j], S["comp"]["Xenium"][:, j]
    section = np.array([np.mean(S["cells"]["Xenium"]["coarse"] == c) for c in focal])
    hits = top5(data.EXAMPLE_SEED)
    matched = S["matched_vi_to_xe"]
    rows = []
    for q, h in hits.items():
        f = int(np.argmax(comp_v[q]))
        rows.append({"query": q, "focal": focal[f], "query_share": comp_v[q, f],
                     "hit_share": comp_x[h, f].mean(), "section": section[f], "found": matched[q] in h})
    R = pd.DataFrame(rows)
    rich = R[R.query_share > 0.5]
    base_rate = float((rich.hit_share > rich.section).mean())
    cand = rich[rich.found].sort_values(["hit_share", "query"], ascending=[False, True])
    q, cls = int(cand["query"].iloc[0]), cand["focal"].iloc[0]
    f = focal.index(cls)
    m = int(matched[q])
    h = hits[q]
    out = pd.DataFrame({"rank": np.arange(1, 6), "index_tile": h, "matched": h == m, "hit_share": comp_x[h, f]})
    out["query"], out["focal_class"], out["query_share"] = q, cls, comp_v[q, f]
    out["matched_tile"], out["section_share"] = m, section[f]
    out["n_rich"], out["n_candidates"], out["rich_above_section"] = len(rich), len(cand), base_rate
    for s_ in data.SEEDS[1:]:
        out[f"top5_seed{s_}"] = " ".join(map(str, top5(s_)[q]))
    out["tag"] = st["tag"]
    out.to_csv(c5.OUT / "example.csv", index=False)
    print(f"example: Visium query {q} ({len(cand)} candidates of {len(rich)} focal-rich queries), {cls} "
          f"{comp_v[q, f]:.2f}; top-5 Xenium {cls} {out.hit_share.min():.2f}-{out.hit_share.max():.2f} (mean "
          f"{out.hit_share.mean():.2f}; section {section[f]:.3f}); co-located tile {m} at rank "
          f"{int(out.loc[out.matched, 'rank'].iloc[0])}; base rate {base_rate:.2f}; seeds 1-4: "
          f"{[out[f'top5_seed{s_}'].iloc[0] for s_ in data.SEEDS[1:]]}", flush=True)


if __name__ == "__main__":
    st = data.stamp()
    S = c5.setup()
    c5.OUT.mkdir(parents=True, exist_ok=True)
    maps(S, st)
    bias_pca(S, st)
    example(S, st)
