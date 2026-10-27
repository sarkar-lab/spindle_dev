"""Every number Fig 2's caption, text and pptx labels quote -> results/fig2_cost/numbers.json (not numbers.py: that
name shadows the standard library's numbers module, which numpy imports).

Usage (login node or anywhere): python experiments/fig2_cost/write_numbers.py
Inputs: results/index_cache/builds.csv (8 datasets x seeds 0-4) and results/fig2_cost/ladder.csv (c_ladder_summary).
Storage folds are of the means over seeds; "range" is over datasets unless it says builds.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402

OUT = data.RESULTS / "fig2_cost" / "numbers.json"


def rng(s, nd=1):
    return [round(float(s.min()), nd), round(float(s.max()), nd)]


def main():
    b = pd.read_csv(data.RESULTS / "index_cache" / "builds.csv")
    b["spindle_build_s"] = b[["prepare_s", "layout_s", "spindle_s"]].sum(axis=1)
    keys = data.indexed_keys()
    m = b.groupby("key").mean(numeric_only=True).loc[keys]
    sd = b.groupby("key").std(numeric_only=True).loc[keys]
    s0 = b[b.seed == 0].set_index("key").loc[keys]
    lad = pd.read_csv(data.RESULTS / "fig2_cost" / "ladder.csv")
    cells, genes = lad[lad.ladder == "cells"].sort_values("n_cells"), lad[lad.ladder == "genes"].sort_values("p")
    full = cells.iloc[-1]

    per = {}
    for k in keys:
        per[k] = {
            "seed0": {c: int(s0.loc[k, c]) for c in ("n_cells", "p", "n_train", "n_niches", "n_blocks")},
            "niches_over_seeds": [int(b[b.key == k].n_niches.min()), int(b[b.key == k].n_niches.max())],
            "blocks_over_seeds": [int(b[b.key == k].n_blocks.min()), int(b[b.key == k].n_blocks.max())],
            "mb": {c: round(m.loc[k, f"bytes_{c}"] / 1e6, 2) for c in ("whole", "spindle", "lite", "partial", "plite")},
            "fold_whole_spindle": round(m.loc[k, "bytes_whole"] / m.loc[k, "bytes_spindle"], 1),
            "fold_spindle_lite": round(m.loc[k, "bytes_spindle"] / m.loc[k, "bytes_lite"], 1),
            "fold_partial_plite": round(m.loc[k, "bytes_partial"] / m.loc[k, "bytes_plite"], 1),
            "lite_pct": [round(m.loc[k, "lite_pct"], 2), round(sd.loc[k, "lite_pct"], 2)],
            "plite_pct": [round(m.loc[k, "plite_pct"], 2), round(sd.loc[k, "plite_pct"], 2)],
            "spindle_build_s": [round(m.loc[k, "spindle_build_s"], 1), round(sd.loc[k, "spindle_build_s"], 1)],
            "adds_s": {c: round(m.loc[k, f"{c}_s"], 2) for c in ("lite", "partial", "plite")},
            "peak_gb": round(m.loc[k, "peak_gb_plite"], 1),
        }
    fold = pd.DataFrame({k: {f: v[f] for f in v if f.startswith("fold")} for k, v in per.items()}).T
    small = cells[cells.n_cells <= 20_000]
    out = {
        "stamp": data.stamp(),
        "builds": int(len(b)), "seeds": data.SEEDS, "tags": sorted(set(b.tag)),
        "datasets": per,
        "range": {
            "cells_seed0": rng(s0.n_cells, 0), "genes": rng(s0.p, 0), "index_tiles_seed0": rng(s0.n_train, 0),
            "niches_seed0": rng(s0.n_niches, 0), "blocks_seed0": rng(s0.n_blocks, 0),
            "niches_builds": rng(b.n_niches, 0), "blocks_builds": rng(b.n_blocks, 0),
            "fold_whole_spindle": rng(fold.fold_whole_spindle), "fold_spindle_lite": rng(fold.fold_spindle_lite),
            "fold_partial_plite": rng(fold.fold_partial_plite),
            "lite_pct_datasets": rng(m.lite_pct, 2), "plite_pct_datasets": rng(m.plite_pct, 2),
            "lite_pct_builds": rng(b.lite_pct, 2), "plite_pct_builds": rng(b.plite_pct, 2),
            "lite_pct_median_builds": round(float(b.lite_pct.median()), 2),
            "plite_pct_median_builds": round(float(b.plite_pct.median()), 2),
            "spindle_build_s": rng(m.spindle_build_s), "spindle_build_sd_max": round(float(sd.spindle_build_s.max()), 1),
            "adds_lite_s": rng(m.lite_s, 2), "adds_partial_s": rng(m.partial_s, 2), "adds_plite_s": rng(m.plite_s, 2),
            "adds_plite_pct_of_spindle_build": rng(100 * m.plite_s / m.spindle_build_s),
            "adds_all_three_s": rng(m.lite_s + m.partial_s + m.plite_s),
            "peak_gb": rng(m.peak_gb_plite),
            "layout_pct_of_spindle_build": rng(100 * m.layout_s / m.spindle_build_s),
            "peak_gb_above_layout_builds": rng(b.peak_gb_plite - b.peak_gb_layout, 2),
        },
        "cell_ladder": {
            "dataset": "tonsil_reactive", "points": int(len(cells)), "cells": rng(cells.n_cells, 0),
            "full": {"cells": int(full.n_cells), "genes": int(full.p), "index_tiles": int(full.n_train),
                     "niches": int(full.n_niches), "blocks": int(full.n_blocks),
                     "spindle_build_s": round(float(full.spindle_build_s), 1),
                     "build_all_s": round(float(full.build_s), 1), "peak_gb": round(float(full.peak_gb_plite), 1),
                     "mb": {c: round(float(full[f"bytes_{c}"]) / 1e6, 1)
                            for c in ("whole", "spindle", "lite", "partial", "plite")}},
            "floor_s_up_to_20k_cells": rng(small.spindle_build_s),
            "from_100k_to_full": {k: round(float(full[c] / cells[cells.n_cells == 100_000].iloc[0][c]), 1) for k, c in
                                  (("cells", "n_cells"), ("index_tiles", "n_train"), ("spindle_build", "spindle_build_s"),
                                   ("peak_gb", "peak_gb_plite"), ("mb_spindle", "bytes_spindle"))},
            "layout_s_1k_cells": round(float(cells.iloc[0].layout_s), 1),
        },
        "gene_ladder": {
            "dataset": "lymph_node_5k", "genes": rng(genes.p, 0), "index_tiles": int(genes.n_train.iloc[0]),
            "points": genes.p.astype(int).tolist(),
            "spindle_build_s": dict(zip(genes.p.astype(int).astype(str), genes.spindle_build_s.round(1))),
            "peak_gb": dict(zip(genes.p.astype(int).astype(str), genes.peak_gb_plite.round(1))),
            "mb_spindle": dict(zip(genes.p.astype(int).astype(str), (genes.bytes_spindle / 1e6).round(1))),
            "mb_whole": dict(zip(genes.p.astype(int).astype(str), (genes.bytes_whole / 1e6).round(1))),
            "fold_whole_spindle": dict(zip(genes.p.astype(int).astype(str),
                                           (genes.bytes_whole / genes.bytes_spindle).round(1))),
            "mb_lite": dict(zip(genes.p.astype(int).astype(str), (genes.bytes_lite / 1e6).round(1))),
            "mb_plite": dict(zip(genes.p.astype(int).astype(str), (genes.bytes_plite / 1e6).round(1))),
            "limit_note": "2,000 genes: the frozen builder's gene ordering (scipy dendrogram, layout.py:96) exceeds "
                          "Python's recursion limit; ladder stops at 1,500 (user, 26 Oct)",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["range"], indent=1))
    print(json.dumps({k: out["cell_ladder"][k] for k in ("full", "floor_s_up_to_20k_cells")}, indent=1))
    print(f"wrote {OUT.relative_to(data.ROOT)}")


if __name__ == "__main__":
    main()
