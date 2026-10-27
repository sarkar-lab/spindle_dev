"""Collect the ladder points: results/fig2_cost/ladder/*.json -> results/fig2_cost/ladder.csv.

Usage (login node, after the c_ladder.py arrays): python experiments/fig2_cost/c_ladder_summary.py
Adds spindle_build_s (prepare + layout + spindle: Spindle's full build, file loading excluded). Fails if a point carries
another tag. Prints the points and which planned points are missing (a gene point above the pilot's limit is expected).
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import c_ladder  # noqa: E402
import data  # noqa: E402

OUT = data.RESULTS / "fig2_cost" / "ladder.csv"


def main():
    df = pd.DataFrame([json.loads(f.read_text()) for f in sorted(c_ladder.OUT.glob("*.json"))])
    df["spindle_build_s"] = df[["prepare_s", "layout_s", "spindle_s"]].sum(axis=1)
    df = df.sort_values(["ladder", "n_cells", "p"])
    df.to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(data.ROOT)}: {len(df)} points; tags {sorted(set(df.tag))}")
    cols = ["ladder", "n_cells", "p", "n_train", "n_niches", "n_blocks", "spindle_build_s", "lite_s", "partial_s",
            "plite_s", "peak_gb_plite", "bytes_spindle", "bytes_lite", "bytes_plite"]
    print(df[cols].round(2).to_string(index=False))
    for ladder, xs in (("cells", c_ladder.CELLS), ("genes", c_ladder.GENES)):
        have = {f.stem.split("_", 1)[1] for f in c_ladder.OUT.glob(f"{ladder}_*.json")}
        print(f"{ladder}: missing {[x for x in xs if ('full' if x is None else str(x)) not in have]}")
    if set(df.tag) != {data.TAG}:
        raise SystemExit(f"tags {sorted(set(df.tag))}")


if __name__ == "__main__":
    main()
