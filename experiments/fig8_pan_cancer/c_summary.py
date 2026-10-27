"""Fig 8 summary from results/fig8_pan_cancer/<key>/seed<s>/programs.csv (c_xtumour.py; light, login node).

Usage: python experiments/fig8_pan_cancer/c_summary.py
Writes results/fig8_pan_cancer/programs.csv: per section, programme and method (P = Spindle-partial, PL =
Spindle-lite-partial, D = variances only, X = highest tile score), mean and s.d. over seeds 0-4 of gain, delta, lift,
the Jaccard indices and the tiles returned; "absent" = Spindle-partial's mean delta <= 0 (the same for every method
of that section and programme).
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common8 as c8  # noqa: E402
from common8 import data  # noqa: E402


def main():
    g = pd.concat([pd.read_csv(c8.OUT / k / f"seed{s}" / "programs.csv") for k in c8.KEYS for s in data.SEEDS])
    num = ["gain", "delta", "lift", "gap_hits", "gap_back", "jacc_P", "jacc_D", "jacc_X", "n_returned",
           "lite_returned", "c", "n_index"]
    by = ["key", "program", "method"]
    out = g.groupby(by, sort=False)[num].mean().join(g.groupby(by, sort=False)[num].std(), rsuffix="_sd")
    out = out.reset_index()
    p = out[out.method == "P"].set_index(["key", "program"]).delta
    out["absent"] = [bool(p.loc[(k, pr)] <= 0) for k, pr in zip(out.key, out.program)]
    out["tag"] = g.tag.iloc[0]
    out.to_csv(c8.OUT / "programs.csv", index=False)
    print(out[by + ["gain", "gain_sd", "delta", "lift", "jacc_P", "jacc_D", "n_returned", "absent"]].round(3)
          .to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
