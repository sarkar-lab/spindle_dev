"""Fig 5 draft layout (mm, top-left corners on the 180 mm page); read by experiments/common/build_figure.py.
No co-located rank panel and no controls (user, 26 Oct).

A  the two sections, cells / spots by coarse class, tile boxes; the example query (magenta) and its top 5 (black)
B  platform bias: PCA of the whole-matrix logs, Visium tiles before and after the global shift
C  tissue agreement (seed 0, global shift), both directions
D  correction ablation: tissue agreement r per correction
E  the example: the Visium query tile and its top 5 Xenium tiles (labels from results/.../example.csv)
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EX = pd.read_csv(ROOT / "results" / "fig5_cross_platform" / "example.csv")

HEIGHT_MM = 130

FX, FY, TW, TS = 108, 76, 24, 28  # panel E origin, tile width, row step
ATOMS = [
    ("map_xenium", 0, 5, 1), ("map_visium", 44, 5, 1),
    ("pca_raw", 92, 8, 1), ("pca_corr", 136, 8, 1), ("legend_platform", 115, 50, 1),
    ("legend_class", 0, 63, 1),
    ("tissue", 0, 74, 1), ("ablation", 54, 74, 1), ("legend_direction", 17, 121, 1),
    ("ex_query", FX, FY, 1),
] + [(f"ex_hit{r}", FX + TW * (r % 3), FY + TS * (r // 3), 1) for r in range(1, 6)]
POS = {a: (x, y) for a, x, y, _ in ATOMS}

cls = EX["focal_class"].iloc[0].lower()
TEXTS = [
    ("A", 0, 0, 5, 4, "letter"), ("B", 90, 0, 5, 4, "letter"), ("C", 0, 69, 5, 4, "letter"),
    ("D", 52, 69, 5, 4, "letter"), ("E", 106, 69, 5, 4, "letter"),
    ("Xenium", 14, 1, 16, 3, "label"), ("Visium", 58, 1, 16, 3, "label"),
    ("Uncorrected", 104, 4, 30, 3, "label"), ("Global shift", 148, 4, 30, 3, "label"),
    (f"Share of {cls} cells / spots (Xenium section: {100 * EX['section_share'].iloc[0]:.0f} %)", FX + 3, FY - 6.6,
     3 * TW, 3, "small"),
    (f"Query (Visium): {100 * EX['query_share'].iloc[0]:.0f} %", FX, FY - 3.2, TW, 3, "small"),
]
for r in EX.itertuples():
    x, y = POS[f"ex_hit{r.rank}"]
    TEXTS.append((f"{r.rank}{' (co-located)' if r.matched else ''}: {100 * r.hit_share:.0f} %", x, y - 3.2, TW,
                  3, "small"))
