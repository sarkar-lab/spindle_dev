"""Fig 7 draft layout (mm, top-left corners on the 180 mm page); read by experiments/common/build_figure.py.
Redesigned around search (user, 27 Oct: not about blocks; full freedom):

A  gene-list search on skin: the three skin signatures, cells by score, Spindle-partial's 10 tiles outlined
B  whole-tile search on skin: a held-out epidermis tile (rule in c_queries.py) and Spindle's top 10 on the section and
   up close; compositions; all skin held-out queries vs random 10
C  every section: composition difference to Spindle's top 10, Spindle-lite's first 10 and a random 10
D  every signature on every section: Cliff's delta, Spindle-partial and Spindle-lite-partial
E  the same: co-variation lift on S
"""

HEIGHT_MM = 146

SIG = [("Melanocyte_Tumor", "Melanocyte (8 genes)"), ("Keratinocyte_Epidermis", "Keratinocyte (4 genes)"),
       ("Immune_Infiltrate", "Immune (5 genes)")]
MX = [0, 61, 122]  # signature map x
BY = 47  # row B letter
TX, TY, TS = 63, BY + 9, 10.5  # thumbnails: hit 1 x, first row y, step
R3 = 92  # row C-E letters

ATOMS = []
TEXTS = [("A", 0, 0, 5, 4, "letter"), ("B", 0, BY, 5, 4, "letter"), ("C", 0, R3, 5, 4, "letter"),
         ("D", 60, R3, 5, 4, "letter"), ("E", 120, R3, 5, 4, "letter")]
for x, (name, title) in zip(MX, SIG):
    ATOMS += [(f"sig_map_{name}", x, 6, 1), (f"cbar_sig_{name}", x + 18, 35, 1)]
    TEXTS.append((title, x + 4, 0.5, 54, 3, "label_c"))
TEXTS.append(("Signature score; Spindle-partial's 10 nearest tiles outlined", 0, 42, 120, 3, "small"))

ATOMS += [("ex_map", 0, BY + 6, 1), ("legend_class", 0, BY + 36, 1),
          ("ex_query", TX, TY + 6.5, 1)]
for r in range(1, 11):
    x, y = TX + 11 + TS * ((r - 1) % 5), TY + (0 if r <= 5 else 13)
    ATOMS.append((f"ex_hit{r}", x, y, 1))
    TEXTS.append((str(r), x, y - 3.0, 9, 3, "small_c"))
TEXTS += [("Held-out query and its top 10", 5, BY + 0.5, 60, 3, "label"),
          ("Query", TX, TY + 16, 9, 3, "small_c")]
ATOMS += [("ex_comp", 127, BY + 2, 1), ("tv_base", 154, BY + 2, 1)]

ATOMS += [("legend_tiles", 2, R3 + 4, 1), ("tiles_all", 0, R3 + 8, 1),
          ("legend_lists", 75, R3 + 4, 1), ("delta_all", 61, R3 + 8, 1), ("lift_all", 121, R3 + 8, 1)]
