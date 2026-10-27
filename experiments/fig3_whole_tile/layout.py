"""Fig 3 draft layout (mm, top-left corners on the 180 mm page); read by experiments/common/build_figure.py.

A  timing (search only) with the query's own logs; speed-up columns as text boxes (annotations.json)
B  Spindle-lite against Spindle per dataset: NX at 5 % of N, rows aligned with A
C  the typical breast query's distance bands
D  memory against fidelity, breast and brain
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANN = json.loads((ROOT / "figures" / "fig3_whole_tile" / "annotations.json").read_text())

HEIGHT_MM = 152

ATOMS = [
    ("timing", 0, 4, 1), ("nx", 110, 4, 1),
    ("legend_timing", 2, 75, 1),
    ("band", 0, 89, 1), ("memory_breast_cancer", 60, 89, 1), ("memory_brain_cancer", 120, 89, 1),
    ("legend_memory", 62, 141, 1),
]
POS = {a: (x, y) for a, x, y, _ in ATOMS}

TEXTS = [
    ("A", 0, 0, 5, 4, "letter"), ("B", 110, 0, 5, 4, "letter"), ("C", 0, 85, 5, 4, "letter"),
    ("D", 60, 85, 5, 4, "letter"),
    ("Speed-up", ANN["timing"][0][1] - 0.5, 1.5, 16, 3, "small"),  # left: Spindle / whole; right: lite / Spindle
    ("Breast, typical query", 12, 86, 40, 3, "label"),
    ("Breast", 85, 86, 20, 3, "label"), ("Brain", 145, 86, 20, 3, "label"),
]
for atom, notes in ANN.items():
    ox, oy = POS[atom]
    for text, x, y in notes:
        TEXTS.append((text, ox + x, oy + y - 1.1, 12, 2.5, "small"))
