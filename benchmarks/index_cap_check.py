"""Part 1.3: do the saved indexes respect the niche cap (1,000 tiles)?

For every dataset and seed (73 and 0-4) in results/indexes/, records the number of training tiles,
niches and blocks, and the niche sizes. Prints and writes results/index_cap_check/summary.csv.

  sbatch slurm_jobs/run_index_cap_check.sbatch
"""

import numpy as np
import pandas as pd

import experiment_common as ec

OUT_DIR = ec.PROJECT_ROOT / "results" / "index_cap_check"
CAP = 1000


def main():
    rows = []
    for stem in ec.DATASETS.values():
        for seed in [ec.PRODUCTION_SEED] + ec.MULTISEED:
            path = ec.INDEX_DIR / f"{ec.index_tag(stem, seed)}_spindle_index.pkl"
            if not path.exists():
                rows.append({"dataset": stem, "seed": seed, "missing": True})
                continue
            data = ec.load_index(stem, seed)["data"]
            sizes = np.bincount(np.asarray(data.labels).astype(int))
            sizes = sizes[sizes > 0]
            rows.append({"dataset": stem, "seed": seed, "missing": False, "n_tiles": int(sizes.sum()),
                         "n_niches": len(sizes), "niche_max": int(sizes.max()), "niche_min": int(sizes.min()),
                         "n_over_cap": int((sizes > CAP).sum()),
                         "n_blocks": int(sum(len(data.block_dict[k]) for k in data.block_dict)),
                         "sizes": " ".join(map(str, sorted(sizes.tolist(), reverse=True)))})
            print(rows[-1], flush=True)
            del data
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.drop(columns=["sizes"]).to_string(index=False))
    print(f"\nindexes over the cap: {int((df.n_over_cap > 0).sum())} of {int((~df.missing).sum())}")


if __name__ == "__main__":
    main()
