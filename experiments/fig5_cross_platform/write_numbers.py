"""Every number Fig 5's caption and text quote -> results/fig5_cross_platform/numbers.json.

Usage (login node or anywhere): python experiments/fig5_cross_platform/write_numbers.py
Inputs: the small tables of c_summary.py (tissue.csv, lite.csv, index.csv) and c_extras.py (tiles.csv,
bias_pca_summary.csv, example.csv). Per direction (x2v: Xenium queries, Visium index; v2x: the reverse): mean and
s.d. over seeds 0-4 of the per-seed value. Spindle with the global shift unless an arm is named. The co-located rank
(hit@k) is recorded but not quoted (user, 26 Oct: no panel); the controls are not computed (user, 26 Oct).
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402

RES = data.RESULTS / "fig5_cross_platform"
OUT = RES / "numbers.json"
ARMS = {"none": "spindle_none", "per_niche": "spindle_per_niche", "global": "spindle_global"}


def r3(x):
    return round(float(x), 3)


def main():
    t = pd.read_csv(RES / "tissue.csv").set_index(["direction", "method"])
    lite = pd.read_csv(RES / "lite.csv").set_index("direction")
    ix = pd.read_csv(RES / "index.csv").set_index("direction")
    tiles = pd.read_csv(RES / "tiles.csv")
    pca = pd.read_csv(RES / "bias_pca_summary.csv")
    ex = pd.read_csv(RES / "example.csv")
    tags = set(t.tag) | set(lite.tag) | set(ix.tag) | set(tiles.tag) | set(pca.tag) | set(ex.tag)
    if tags != {data.TAG}:
        raise RuntimeError(f"inputs carry tags {tags}, expected {data.TAG}")

    out = {"tag": data.TAG, "describe": data.stamp()["describe"],
           "genes": int(ix.p.iloc[0]),
           "tiles": {"Xenium": int((tiles.platform == "Xenium").sum()), "Visium": int((tiles.platform == "Visium").sum())},
           "cells_per_tile_max": 2000, "min_spots_per_tile": 10,
           "spots_per_visium_tile": [int(tiles[tiles.platform == "Visium"].n_obs.min()),
                                     int(tiles[tiles.platform == "Visium"].n_obs.max())],
           "directions": {}}
    for d in ("x2v", "v2x"):
        g = t.loc[(d, ARMS["global"])]
        out["directions"][d] = {
            "N_index": int(ix.loc[d, "N"]), "n_query": int(ix.loc[d, "n_query"]),
            "niches": [int(ix.loc[d, "niches_min"]), int(ix.loc[d, "niches_max"])],
            "blocks": [int(ix.loc[d, "blocks_min"]), int(ix.loc[d, "blocks_max"])],
            "tissue_r": {a: [r3(t.loc[(d, m), "tissue_r_mean"]), r3(t.loc[(d, m), "tissue_r_std"])]
                         for a, m in ARMS.items()},
            "tumour_mae_top10": [r3(g.mae_mean), r3(g.mae_std)], "tumour_mae_random": r3(g.mae_random_mean),
            "colocated_hit5": [r3(g.hit5_mean), r3(g.hit5_std)], "colocated_hit5_random": r3(g.hit5_random_mean),
            "silhouette": {"uncorrected": r3(pca[(pca.direction == d) & ~pca.corrected].silhouette.iloc[0]),
                           "global": r3(pca[(pca.direction == d) & pca.corrected].silhouette.iloc[0])},
            "lite_NX5_pct": [round(100 * lite.loc[d, "NX5_mean"], 1), round(100 * lite.loc[d, "NX5_std"], 1)],
            "lite_overlap5": [r3(lite.loc[d, "overlap5_mean"]), r3(lite.loc[d, "overlap5_std"])],
            "lite_found": round(float(lite.loc[d, "n_found_mean"]), 1),
            "spindle_MB": round(float(ix.loc[d, "spindle_mb"]), 2), "lite_MB": round(float(ix.loc[d, "lite_mb"]), 2),
        }
    e = ex.iloc[0]
    out["example"] = {"direction": "v2x", "seed": data.EXAMPLE_SEED, "query": int(e.query),
                      "focal_class": e.focal_class, "query_share_pct": round(100 * e.query_share),
                      "top5_share_pct": [round(100 * ex.hit_share.min()), round(100 * ex.hit_share.max())],
                      "top5_mean_pct": round(100 * ex.hit_share.mean()),
                      "section_share_pct": round(100 * e.section_share),
                      "colocated_rank": int(ex.loc[ex.matched, "rank"].iloc[0]),
                      "same_top5_all_seeds": bool(all(e[f"top5_seed{s}"] == " ".join(map(str, ex.index_tile))
                                                      for s in data.SEEDS[1:])),
                      "n_focal_rich": int(e.n_rich), "n_candidates": int(e.n_candidates),
                      "base_rate_pct": round(100 * e.rich_above_section)}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
