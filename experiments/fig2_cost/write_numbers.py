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
    b["naive_build_s"] = b[["prepare_s", "layout_s", "naive_s"]].sum(axis=1)
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
            "mb": {c: round(m.loc[k, f"bytes_{c}"] / 1e6, 2)
                   for c in ("whole", "naive", "spindle", "naive_partial", "spindle_partial")},
            "fold_whole_naive": round(m.loc[k, "bytes_whole"] / m.loc[k, "bytes_naive"], 1),
            "fold_naive_spindle": round(m.loc[k, "bytes_naive"] / m.loc[k, "bytes_spindle"], 1),
            "fold_naive_partial_spindle_partial": round(m.loc[k, "bytes_naive_partial"]
                                                        / m.loc[k, "bytes_spindle_partial"], 1),
            "spindle_pct": [round(m.loc[k, "spindle_pct"], 2), round(sd.loc[k, "spindle_pct"], 2)],
            "spindle_partial_pct": [round(m.loc[k, "spindle_partial_pct"], 2),
                                    round(sd.loc[k, "spindle_partial_pct"], 2)],
            "naive_build_s": [round(m.loc[k, "naive_build_s"], 1), round(sd.loc[k, "naive_build_s"], 1)],
            "adds_s": {c: round(m.loc[k, f"{c}_s"], 2) for c in ("spindle", "naive_partial", "spindle_partial")},
            "peak_gb": round(m.loc[k, "peak_gb_spindle_partial"], 1),
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
            "fold_whole_naive": rng(fold.fold_whole_naive), "fold_naive_spindle": rng(fold.fold_naive_spindle),
            "fold_naive_partial_spindle_partial": rng(fold.fold_naive_partial_spindle_partial),
            "spindle_pct_datasets": rng(m.spindle_pct, 2),
            "spindle_partial_pct_datasets": rng(m.spindle_partial_pct, 2),
            "spindle_pct_builds": rng(b.spindle_pct, 2), "spindle_partial_pct_builds": rng(b.spindle_partial_pct, 2),
            "spindle_pct_median_builds": round(float(b.spindle_pct.median()), 2),
            "spindle_partial_pct_median_builds": round(float(b.spindle_partial_pct.median()), 2),
            "naive_build_s": rng(m.naive_build_s), "naive_build_sd_max": round(float(sd.naive_build_s.max()), 1),
            "adds_spindle_s": rng(m.spindle_s, 2), "adds_naive_partial_s": rng(m.naive_partial_s, 2),
            "adds_spindle_partial_s": rng(m.spindle_partial_s, 2),
            "adds_spindle_partial_pct_of_naive_build": rng(100 * m.spindle_partial_s / m.naive_build_s),
            "adds_all_three_s": rng(m.spindle_s + m.naive_partial_s + m.spindle_partial_s),
            "peak_gb": rng(m.peak_gb_spindle_partial),
            "layout_pct_of_naive_build": rng(100 * m.layout_s / m.naive_build_s),
            "peak_gb_above_layout_builds": rng(b.peak_gb_spindle_partial - b.peak_gb_layout, 2),
        },
        "cell_ladder": {
            "dataset": "tonsil_reactive", "points": int(len(cells)), "cells": rng(cells.n_cells, 0),
            "full": {"cells": int(full.n_cells), "genes": int(full.p), "index_tiles": int(full.n_train),
                     "niches": int(full.n_niches), "blocks": int(full.n_blocks),
                     "naive_build_s": round(float(full.naive_build_s), 1),
                     "build_all_s": round(float(full.build_s), 1),
                     "peak_gb": round(float(full.peak_gb_spindle_partial), 1),
                     "mb": {c: round(float(full[f"bytes_{c}"]) / 1e6, 1)
                            for c in ("whole", "naive", "spindle", "naive_partial", "spindle_partial")}},
            "floor_s_up_to_20k_cells": rng(small.naive_build_s),
            "from_100k_to_full": {k: round(float(full[c] / cells[cells.n_cells == 100_000].iloc[0][c]), 1) for k, c in
                                  (("cells", "n_cells"), ("index_tiles", "n_train"), ("naive_build", "naive_build_s"),
                                   ("peak_gb", "peak_gb_spindle_partial"), ("mb_naive", "bytes_naive"))},
            "layout_s_1k_cells": round(float(cells.iloc[0].layout_s), 1),
        },
        "gene_ladder": {
            "dataset": "lymph_node_5k", "genes": rng(genes.p, 0), "index_tiles": int(genes.n_train.iloc[0]),
            "points": genes.p.astype(int).tolist(),
            "naive_build_s": dict(zip(genes.p.astype(int).astype(str), genes.naive_build_s.round(1))),
            "peak_gb": dict(zip(genes.p.astype(int).astype(str), genes.peak_gb_spindle_partial.round(1))),
            "mb_naive": dict(zip(genes.p.astype(int).astype(str), (genes.bytes_naive / 1e6).round(1))),
            "mb_whole": dict(zip(genes.p.astype(int).astype(str), (genes.bytes_whole / 1e6).round(1))),
            "fold_whole_naive": dict(zip(genes.p.astype(int).astype(str),
                                         (genes.bytes_whole / genes.bytes_naive).round(1))),
            "mb_spindle": dict(zip(genes.p.astype(int).astype(str), (genes.bytes_spindle / 1e6).round(1))),
            "mb_spindle_partial": dict(zip(genes.p.astype(int).astype(str),
                                           (genes.bytes_spindle_partial / 1e6).round(1))),
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
