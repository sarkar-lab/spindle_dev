"""Fig 6 draft layout (mm, top-left corners on the 180 mm page); read by experiments/common/build_figure.py.
Panels approved 26 Oct (Session 9), then cut (user, 26 Oct: the figure was too big): the co-variation strip and the
signature maps are parked off-page as spares (their numbers go in the text); ACTA2 back in the free top right (user,
27 Oct).

A  niches: cells by niche; log2 enrichment of the 19 cell types per niche, section fractions
B  ACTA2's block partners per niche
C  block programmes of the chosen niche (rule in f_fig6.py) and the contrast niche in its order; numbered co-varying
   blocks with their key (hub genes; Hallmark, else GO BP term and FDR; block genes in the term) from blocks.csv
D  a whole-tile query and Spindle's top 10; compositions; all held-out queries vs random 10
E  gene-list queries, all five signatures: Cliff's delta and co-variation on S
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
S0 = ROOT / "results" / "fig6_breast" / "seed0"
FULL = ROOT / "results" / "fig6_breast" / "full"  # A-C: the whole-section build
B = pd.read_csv(FULL / "blocks.csv")
LAY = json.loads((FULL / "layout.json").read_text())
G0 = pd.read_csv(S0 / "gene_lists.csv")

_b = B.assign(both=B.covarying & B.enriched).groupby("niche").agg(both=("both", "sum"), tiles=("n_tiles", "first"))
MAIN = int(_b.sort_values(["both", "tiles"], ascending=False).index[0])
CONTRAST = int(_b.drop(MAIN).sort_values("tiles", ascending=False).index[0])
MAPS = G0[G0["query"] == "T"].nlargest(2, "delta")["list"].tolist()

HEIGHT_MM = 178

CX, CY, CW = 4, 74, 48  # corr_main origin and size
R3 = 133  # third row
EX0, EY, ES = 0, R3 + 8, 11  # example tiles: origin, first row y, step
SPARE = [("covariation", "legend_cov"),
         (f"sig_map_{MAPS[0]}", f"cbar_sig_{MAPS[0]}"), (f"sig_map_{MAPS[1]}", f"cbar_sig_{MAPS[1]}")]
ATOMS = [
    ("map_niches", 0, 5, 1), ("legend_niche", 0, 50, 1),
    ("niche_enrich", 62, 3, 1), ("cbar_enrich", 88, 59.5, 1),
    ("acta2", 138, 5, 1), ("cbar_acta2", 151, 54.5, 1),
    ("corr_main", CX, CY, 1), ("corr_contrast", CX + CW + 2, CY, 1), ("cbar_corr", 13, 123, 1),
    ("ex_query", EX0, EY, 1),
] + [(f"ex_hit{r}", EX0 + ES * (r if r <= 5 else r - 5), EY + (0 if r <= 5 else 14), 1) for r in range(1, 11)] + [
    ("legend_class", 0, R3 + 36, 1), ("ex_comp", 68, R3 + 3, 1), ("tv_base", 94, R3 + 3, 1),
    ("sig_delta", 121, R3 + 3, 1), ("sig_cov", 150, R3 + 3, 1), ("legend_sig", 135, R3 - 1, 1),
] + [(a, 190, 5 + 60 * i + 50 * j, 1) for i, pair in enumerate(SPARE) for j, a in enumerate(pair)]
POS = {a: (x, y) for a, x, y, _ in ATOMS}

TEXTS = [
    ("A", 0, 0, 5, 4, "letter"), ("B", 128, 0, 5, 4, "letter"), ("C", 0, 67, 5, 4, "letter"),
    ("D", 0, R3 - 1, 5, 4, "letter"), ("E", 119, R3 - 1, 5, 4, "letter"),
    ("r with ACTA2 (○ same block)", 133, 0, 47, 3, "label"),
    ("Niche", 88, -0.5, 18, 3, "small_c"), ("Section", 108, -0.5, 14, 3, "small_c"),
    ("log2 enrichment over section", 118.5, 60, 30, 3, "small"),
    (f"Niche {MAIN + 1}, its blocks", CX, CY - 4.5, CW, 3, "label_c"),
    (f"Niche {CONTRAST + 1}, in niche {MAIN + 1}'s order and blocks", CX + CW + 2, CY - 4.5, CW, 3, "label_c"),
    ("Correlation (niche mean)", 44, 123.8, 40, 3, "small"),
    ("Held-out query and its top 10", 5, R3 - 0.5, 60, 3, "label"),
    ("Query", EX0, EY + 10.2, 10, 3, "small_c"),
]
for r in range(1, 11):
    x, y = POS[f"ex_hit{r}"]
    TEXTS.append((str(r), x, y - 3.0, 10, 3, "small_c"))

# B: numbers beside the co-varying blocks and the key
p = len(LAY["genes"])
main = B[B.niche == MAIN].sort_values("block")
KX, KY, KW = 104, CY - 1, 76
num = 0
key_lines = []
for r in main.itertuples():
    if not r.covarying:
        continue
    num += 1
    mid = (r.start + r.end) / 2 / p
    TEXTS.append((str(num), CX - 4, CY + mid * CW - 1.5, 4, 3, "small_c"))
    line = f"{num}  {', '.join(r.hubs.split())}"
    if r.enriched:
        term = re.sub(r"\s*\(GO:\d+\)", "", r.key_term)
        genes = r.key_genes.split()
        genes = ", ".join(genes[:6]) + (", …" if len(genes) > 6 else "")
        line += f"\n    {term} (FDR {r.key_fdr:.0e}): {genes}"
    key_lines.append(line)
TEXTS.append(("Co-varying blocks: hub genes; term (FDR), its block genes", KX, KY - 4, KW, 3, "label"))
TEXTS.append(("\n".join(key_lines), KX, KY, KW, 52, "small"))
