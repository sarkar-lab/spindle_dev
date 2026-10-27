"""Every number Fig 6's caption and text quote -> results/fig6_breast/numbers.json.

Usage (login node or anywhere): python experiments/fig6_breast/write_numbers.py
Inputs: results/fig6_breast/full/ (A-C: the whole-section build, c_programs.py full) and seed<0-4>/ (c_programs.py,
c_queries.py). Niches 1-based as in the figure. Across splits: mean and s.d. over seeds 0-4 of the per-seed value.
The controls of the gene-list queries (D = variances only, X = top mean expression) are reported, never claimed
against.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402

RES = data.RESULTS / "fig6_breast"
FULL, S0 = RES / "full", RES / f"seed{data.EXAMPLE_SEED}"
OUT = RES / "numbers.json"
SIG = {"Luminal_Tumor_Core": "luminal", "Macrophage_Myeloid": "macrophage", "Basal_Myoepithelial": "myoepithelial",
       "Endothelial_Vascular": "endothelial", "Proliferation_Signature": "proliferation"}


def r3(x):
    return round(float(x), 3)


def pct(x, nd=0):
    v = round(100 * float(x), nd)
    return int(v) if nd == 0 else v


def ms(s, f=r3):
    return [f(s.mean()), f(s.std())]


def block_abs_r(d, k, s, e, other):
    """Mean |r| over the gene pairs of niche k's block (k's order, positions s:e) in niche ``other``'s mean
    correlation (corr_niche<other>.npy is stored in other's own gene order)."""
    lay = json.loads((d / "layout.json").read_text())
    genes = np.asarray(lay[str(k)]["perm"])[s:e]
    pos = np.empty(len(lay["genes"]), int)
    pos[np.asarray(lay[str(other)]["perm"])] = np.arange(len(lay["genes"]))
    R = np.abs(np.load(d / f"corr_niche{other}.npy")[np.ix_(pos[genes], pos[genes])])
    n = len(genes)
    return (R.sum() - n) / (n * (n - 1))


def seeds(name):
    return pd.concat([pd.read_csv(RES / f"seed{s}" / name).assign(seed=s) for s in data.SEEDS])


def main():
    B = pd.read_csv(FULL / "blocks.csv")
    T = pd.read_csv(FULL / "niche_types.csv")
    A = pd.read_csv(FULL / "acta2.csv")
    Bs, Q, G = seeds("blocks.csv"), seeds("tile_queries.csv"), seeds("gene_lists.csv")
    ex = pd.read_csv(S0 / "example.csv").set_index("set")
    cells = np.load(FULL / "cells_niche.npz")
    tags = set(B.tag) | set(T.tag) | set(A.tag) | set(Bs.tag) | set(Q.tag) | set(G.tag) | set(ex.tag)
    if tags != {data.TAG}:
        raise RuntimeError(f"inputs carry tags {tags}, expected {data.TAG}")

    b = B.assign(both=B.covarying & B.enriched)
    g = b.groupby("niche").agg(both=("both", "sum"), tiles=("n_tiles", "first"))
    main_k = int(g.sort_values(["both", "tiles"], ascending=False).index[0])
    contrast = int(g.drop(main_k).sort_values("tiles", ascending=False).index[0])
    mb = B[B.niche == main_k]
    out = {"tag": data.TAG, "describe": data.stamp()["describe"]}
    out["full"] = {
        "tiles": int(g.tiles.sum()), "niches": len(g), "niche_tiles": g.tiles.astype(int).tolist(),
        "cells": int(len(cells["niche"])), "cells_no_tile": int((cells["niche"] < 0).sum()),
        "genes": int(len(json.loads((FULL / "layout.json").read_text())["genes"])),
        "blocks": int(len(B)), "blocks_per_niche": B.groupby("niche").size().astype(int).tolist(),
        "block_size": [int(B["size"].min()), int(B["size"].max())],
        "covarying": int(B.covarying.sum()), "enriched": int(B.enriched.sum()),
        "enriched_null_pct": pct(B.null_enriched.mean(), 1),
        "main_niche": main_k + 1, "contrast_niche": contrast + 1,
        "main": {"blocks": int(len(mb)), "covarying": int(mb.covarying.sum()),
                 "covarying_enriched": int((mb.covarying & mb.enriched).sum()),
                 "key": [{"hubs": r.hubs, "term": r.key_term if r.enriched else None,
                          "fdr": float(r.key_fdr) if r.enriched else None,
                          "abs_r_main": r3(block_abs_r(FULL, main_k, r.start, r.end, main_k)),
                          "abs_r_contrast": r3(block_abs_r(FULL, main_k, r.start, r.end, contrast))}
                         for r in mb[mb.covarying].sort_values("block").itertuples()]},
    }
    # composition: share of each niche's cells per type and the share of each type's cells in the niche
    comp = {}
    for k, t in T.groupby("niche"):
        top = t.sort_values("frac", ascending=False).head(4)
        comp[str(k + 1)] = {"cells": int(t.niche_cells.iloc[0]),
                            "top_types": {r.type: {"frac_pct": pct(r.frac), "share_of_type_pct": pct(r.share_of_type),
                                                   "log2_enrich": round(float(np.log2(r.frac / r.section_frac)), 2)}
                                          for r in top.itertuples()}}
    out["full"]["niche_composition"] = comp
    out["full"]["type_share_in_niche_pct"] = {
        f"{r.type}|{r.niche + 1}": pct(r.share_of_type) for r in T.itertuples() if r.share_of_type >= 0.5}

    part = {}
    for k in sorted(A.niche.unique()):
        s = A[(A.niche == k) & A.in_block]
        part[str(k + 1)] = {"block_size": int(s.block_size.iloc[0]),
                            "top6": s.nlargest(6, "r").gene.tolist(),
                            "top6_r": [r3(x) for x in s.nlargest(6, "r").r]}
    sets = {k: set(A[(A.niche == k) & A.in_block].gene) for k in A.niche.unique()}
    out["acta2"] = {"partners": part,
                    "jaccard": {f"{a + 1}-{b + 1}": r3(len(sets[a] & sets[b]) / len(sets[a] | sets[b]))
                                for a, b in combinations(sorted(sets), 2)}}

    per = Bs.groupby("seed").agg(cov=("covarying", "mean"), enr=("enriched", "mean"), null=("null_enriched", "mean"),
                                 within=("within", "median"), between=("between", "median"),
                                 random=("null_mean", "median"), niches=("niche", "nunique"), blocks=("block", "size"))
    out["splits_blocks"] = {"covarying_pct": ms(100 * per["cov"], lambda x: round(float(x), 1)),
                            "enriched_pct": ms(100 * per.enr, lambda x: round(float(x), 1)),
                            "enriched_null_pct": ms(100 * per.null, lambda x: round(float(x), 1)),
                            "within_abs_r": ms(per.within), "between_abs_r": ms(per.between),
                            "random_abs_r": ms(per.random),
                            "niches": [int(per.niches.min()), int(per.niches.max())],
                            "blocks": [int(per.blocks.min()), int(per.blocks.max())]}

    pq = Q.groupby("seed").agg(n=("query", "size"), tv_top=("tv_top", "median"), tv_random=("tv_random", "median"),
                               tv_expr=("tv_expr", "median"), lite=("lite_in_top", "median"),
                               below=("tv_top", lambda s: 0))
    pq["below"] = Q.assign(b=Q.tv_top < Q.tv_random).groupby("seed").b.sum()
    out["tile_queries"] = {"n_per_split": int(pq.n.iloc[0]), "tv_top10": ms(pq.tv_top), "tv_random10": ms(pq.tv_random),
                           "below_random": [int(pq.below.min()), int(pq.below.max())],
                           "tv_expression_top10": ms(pq.tv_expr),
                           "lite_in_spindle_top10_median": [int(pq.lite.min()), int(pq.lite.max())]}
    q0 = Q[Q.seed == data.EXAMPLE_SEED].set_index("tile")
    t = int(ex.query_tile.iloc[0])
    out["example"] = {"seed": data.EXAMPLE_SEED, "tile": t, "cells": int(q0.loc[t, "n_cells"]),
                      "n_candidates": int(ex.n_candidates.iloc[0]), "min_cells": 100,
                      "myo_pct": {s: pct(ex.loc[s, "Myoepithelial"]) for s in ex.index},
                      "myo_section_pct": pct(ex.loc["section", "Myoepithelial"], 1),
                      "tv_top10": r3(q0.loc[t, "tv_top"]), "tv_random10": r3(q0.loc[t, "tv_random"])}

    gl = {}
    for name, lab in SIG.items():
        s = G[G.list == name]
        e = {"genes": int(s.n_genes.iloc[0])}
        for qn in ("T", "D", "X"):
            x = s[s["query"] == qn]
            e[qn] = {"delta": ms(x.delta, lambda v: round(float(v), 2)), "cov_top10": ms(x.cov_top, lambda v: round(float(v), 2)),
                     "jacc_D": ms(x.jacc_D, lambda v: round(float(v), 2))}
        e["cov_typical"] = round(float(s.cov_all.mean()), 2)
        gl[lab] = e
    out["gene_lists"] = gl
    tt = G[G["query"] == "T"].groupby("list").mean(numeric_only=True)
    dd = G[G["query"] == "D"].groupby("list").mean(numeric_only=True)
    xx = G[G["query"] == "X"].groupby("list").mean(numeric_only=True)
    out["gene_lists_summary"] = {
        "T_delta": [round(float(tt.delta.min()), 2), round(float(tt.delta.max()), 2)],
        "T_cov_above_typical": int((tt.cov_top > tt.cov_all).sum()),
        "D_delta": [round(float(dd.delta.min()), 2), round(float(dd.delta.max()), 2)],
        "D_cov_below_typical": int((dd.cov_top < dd.cov_all).sum()),
        "X_cov_above_T": int((xx.cov_top > tt.cov_top).sum()),
        "jacc_TD": [round(float(tt.jacc_D.min()), 2), round(float(tt.jacc_D.max()), 2)]}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
