"""Collect the index cache's build records: results/_cache/*/seed*/build.json -> results/index_cache/builds.csv.

Usage (login node, after the c_index_cache.py array): python experiments/common/c_index_cache_summary.py
Prints, per dataset over seeds 0-4: index / held-out tiles, niches, blocks, storage of Spindle and
Spindle partial as % of Spindle-naive (against the ranges in paper/SPINDLE_LITE.md and SPINDLE_PARTIAL.md, measured on
the 100-held-out split), build seconds and peak memory. Fails if a build is missing or any build has another tag.
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402

OUT = data.RESULTS / "index_cache"
SPEC = {"spindle_pct": (10.5, 12.4), "spindle_partial_pct": (13.6, 14.4)}  # SPINDLE_LITE.md / SPINDLE_PARTIAL.md ranges


def main():
    rows, missing = [], []
    for key in data.indexed_keys():
        for seed in data.SEEDS:
            f = data.cache_dir(key, seed) / "build.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
            else:
                missing.append(f"{key}/seed{seed}")
    df = pd.DataFrame(rows)
    df["build_s"] = df[["prepare_s", "layout_s", "naive_s", "spindle_s", "naive_partial_s", "spindle_partial_s"]].sum(axis=1)
    OUT.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["niche_sizes"]).to_csv(OUT / "builds.csv", index=False)
    print(f"wrote {(OUT / 'builds.csv').relative_to(data.ROOT)}: {len(df)} builds; tags {sorted(set(df.tag))}, "
          f"hosts {sorted(set(df.host))}")
    g = df.groupby("key", sort=False)
    summ = pd.DataFrame({
        "train": g.n_train.median(), "test": g.n_test.median(), "niches": g.n_niches.agg(lambda s: f"{s.min()}-{s.max()}"),
        "blocks": g.n_blocks.agg(lambda s: f"{s.min()}-{s.max()}"),
        "spindle_pct": g.spindle_pct.mean(), "spindle_partial_pct": g.spindle_partial_pct.mean(),
        "build_s": g.build_s.mean(), "build_sd": g.build_s.std(), "spindle_s": g.spindle_s.mean(),
        "spindle_partial_s": g.spindle_partial_s.mean(), "peak_gb": g.peak_gb_spindle_partial.mean(),
    })
    print(summ.round(2).to_string())
    for col, (lo, hi) in SPEC.items():
        out = df[(df[col] < lo) | (df[col] > hi)]
        print(f"{col}: {df[col].min():.2f}-{df[col].max():.2f} % over all builds (spec {lo}-{hi} %); "
              f"{len(out)} builds outside")
    if missing or set(df.tag) != {data.TAG}:
        raise SystemExit(f"incomplete: missing {missing}, tags {sorted(set(df.tag))}")


if __name__ == "__main__":
    main()
