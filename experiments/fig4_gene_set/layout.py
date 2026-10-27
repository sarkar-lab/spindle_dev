"""Fig 4 draft layout (mm, top-left corners on the 180 mm page); read by experiments/common/build_figure.py.
No schematic (user, 26 Oct).

A  the typical breast query's distance bands (|S| = 16)
B  NX at 5 % of N against |S|: Spindle-lite-partial, padding, imputation
C  Overlap(5 %, 5 % of N) against |S|, the same methods
D  time per query (exact gene-set scan, Spindle-lite-partial), speed-ups as text boxes (annotations.json)
E  Spindle-lite-partial NX at 5 % of N per dataset, rows aligned with D
F  stored MB per dataset, rows aligned with D
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = json.loads((ROOT / "figures" / "fig4_gene_set" / "annotations.json").read_text())

HEIGHT_MM = 136

ATOMS = [
    ("band", 0, 4, 1), ("nx_size", 62, 4, 1), ("overlap_size", 124, 4, 1),
    ("legend_nx", 58, 50, 1),
    ("timing", 0, 62, 1), ("nx", 68, 62, 1), ("storage", 114, 62, 1),
    ("legend_timing", 0, 126, 1), ("legend_storage", 72, 126, 1),
]
POS = {a: (x, y) for a, x, y, _ in ATOMS}

TEXTS = [
    ("A", 0, 0, 5, 4, "letter"), ("B", 62, 0, 5, 4, "letter"), ("C", 124, 0, 5, 4, "letter"),
    ("D", 0, 58, 5, 4, "letter"), ("E", 68, 58, 5, 4, "letter"), ("F", 114, 58, 5, 4, "letter"),
    ("Breast, typical query, |S| = 16", 12, 1, 44, 3, "label"),
    ("Speed-up", ANN["timing"][0][1] - 1, 59, 12, 3, "small"),
]
for atom, notes in ANN.items():
    ox, oy = POS[atom]
    for text, x, y in notes:
        TEXTS.append((text, ox + x, oy + y - 1.1, 12, 2.5, "small"))
