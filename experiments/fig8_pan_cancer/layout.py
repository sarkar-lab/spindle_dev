"""Fig 8 draft layout (mm, top-left corners on the 180 mm page); read by experiments/common/build_figure.py.
Redesigned 27 Oct (user): distance to a query as a smoothed gradient over the sections; two queries (rule in f_fig8.py).

A  the query (breast exemplar correlation R_S), one row per query
B  seed 0: closeness of every section to the query (breast | lung over pancreas | brain | skin)
C  every section: gain in match to the query of Spindle-partial's and Spindle-lite-partial's hits
D  the same: co-variation lift over the hits' cells (Figs 6-7)
"""

HEIGHT_MM = 142

QUERIES = [("Myeloid", "Myeloid"), ("Stromal_Perivascular", "Stromal / perivascular")]
COLS = [("breast_cancer", 25, 0, "Breast"), ("lung_cancer", 63, 0, "Lung"), ("pancreatic_cancer", 63, 19.2, "Pancreas"),
        ("brain_cancer", 113, 0, "Brain"), ("skin_melanoma", 146, 0, "Skin")]  # (key, x, y within row, label)
Y0, STEP = 8, 35  # first row, row step
CY = 90  # row C letter

ATOMS, TEXTS = [], [("A", 0, 0, 5, 4, "letter"), ("B", 24, 0, 5, 4, "letter"), ("C", 0, CY, 5, 4, "letter"),
                    ("D", 91, CY, 5, 4, "letter")]
for key, x, dy, label in COLS:
    if dy == 0:
        TEXTS.append((label, x, 3.5, 30, 3, "label"))
for i, (name, title) in enumerate(QUERIES):
    y = Y0 + STEP * i
    TEXTS.append((title, 0, y, 24, 3, "small"))
    ATOMS.append((f"rs_{name}", 0, y + 4, 1))
    for key, x, dy, label in COLS:
        ATOMS.append((f"grad_{key}_{name}", x, y + dy, 1))
        if dy:
            TEXTS.append((label, x, y + dy - 3.2, 30, 3, "label"))

ATOMS += [("cbar_rs", 0, 78, 1), ("cbar_grad", 92, 78, 1)]
TEXTS += [("Query correlation", 0, 85, 24, 3, "small_c"), ("Farther", 78, 78.6, 14, 3, "small"),
          ("Nearer", 123, 78.6, 14, 3, "small"), ("Closeness to the query", 92, 85, 30, 3, "small_c")]

ATOMS += [("gain", 0, CY + 4, 1), ("lift", 91, CY + 4, 1), ("legend_cd", 1, CY + 47, 1)]
