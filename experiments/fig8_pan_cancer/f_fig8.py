"""Fig 8 atoms from results/fig8_pan_cancer/ (c_xtumour.py, c_gradient.py, c_summary.py; login node).

Usage: python experiments/fig8_pan_cancer/f_fig8.py
Redesigned 27 Oct (user): the queries, then each query's distance as a smoothed gradient over every section, then one
quantitative panel; cut to two queries (user: a smaller figure). SHOWN RULE (stated in the caption): the programmes whose
Spindle-partial hits gain in match to the query (gain > GAIN_EPS) and score above the other tiles (delta > 0) in all
five sections, means over seeds (programs.csv); the others are reported in the text (numbers.json).
Atoms (figures/fig8_pan_cancer/atoms/), for the shown programmes:
  rs_<programme>, cbar_rs  A: the query, the breast exemplar's correlation R_S on the programme genes (diagonal blank)
  grad_<key>_<programme>, cbar_grad  B: seed 0, closeness of the tissue to the query (gradient_image); a programme
                         absent from the section (Spindle-partial's mean delta <= 0, c_summary.py) is drawn faded
  gain, lift, legend_cd  C, D: every section x shown programme, mean +/- s.d. over seeds 0-4: gain in match to R_S
                         (C) and co-variation lift over the hits' cells (D; c_xtumour.py) of Spindle-partial's and Spindle-lite-partial's
                         hits (5 % of the index tiles), variances only as a grey tick
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import style as ps  # noqa: E402

FIG = "fig8_pan_cancer"
RES = ps.ROOT / "results" / FIG
KEYS = ["breast_cancer", "lung_cancer", "pancreatic_cancer", "brain_cancer", "skin_melanoma"]
PROGS = ["Cytotoxic_T", "Myeloid", "Stromal_Perivascular", "B_Plasma"]
PROG_LABELS = {"Cytotoxic_T": "Cytotoxic T", "Myeloid": "Myeloid", "Stromal_Perivascular": "Stromal / perivascular",
               "B_Plasma": "B / plasma"}
PROG_COLORS = {"Cytotoxic_T": "#CC3311", "Myeloid": "#009988", "Stromal_Perivascular": "#0077BB",
               "B_Plasma": "#EE7733"}
GREY = "#E3E3E3"
CELL_MM = 2.3  # heatmap cell
GRAD_W = {"breast_cancer": 36.0, "lung_cancer": 48.0, "pancreatic_cancer": 48.0, "brain_cancer": 31.0,
          "skin_melanoma": 33.0}  # map widths (mm); lung over pancreas in one column
GAIN_EPS = 1e-3  # as write_numbers.py
SMOOTH = 2.0  # Gaussian s.d. in median tile sides
PIX = 1 / 3  # pixel in median tile sides
ABSENT_ALPHA = 0.3
GRAD_STEPS = 32


def shown():
    T = pd.read_csv(RES / "programs.csv")
    P = T[T.method == "P"]
    ok = P.assign(ok=(P.gain > GAIN_EPS) & (P.delta > 0)).groupby("program").ok.all()
    return [p for p in PROGS if ok[p]]


def absent_pairs():
    T = pd.read_csv(RES / "programs.csv")
    return set(map(tuple, T[(T.method == "P") & T.absent][["key", "program"]].to_numpy()))


# ---------------------------------------------------------------- A
def queries():
    z = np.load(RES / "exemplar.npz")
    E = pd.read_csv(RES / "exemplar.csv").set_index("program")
    cmap = ps.DIVERGING
    for p in shown():
        R = z[f"R_{p}"].copy()
        np.fill_diagonal(R, np.nan)
        genes = E.loc[p, "genes"].split()
        s = len(genes)
        left, bottom = 11.0, 11.0  # room for TNFRSF17
        fig, ax = ps.atom(left + s * CELL_MM + 0.5, bottom + s * CELL_MM + 0.5, left=left, bottom=bottom, right=0.5,
                          top=0.5)
        ax.imshow(R, cmap=cmap, vmin=-0.6, vmax=0.6, interpolation="nearest")
        ax.set_xticks(range(s))
        ax.set_xticklabels(genes, rotation=90)
        ax.set_yticks(range(s))
        ax.set_yticklabels(genes)
        ax.tick_params(length=0, pad=1)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ps.save(fig, FIG, f"rs_{p}", source="results/fig8_pan_cancer/exemplar.npz")
    ps.colorbar_atom(FIG, "cbar_rs", cmap, -0.6, 0.6, 24, 7, ticks=[-0.6, 0, 0.6], orientation="horizontal",
                     source="")


# ---------------------------------------------------------------- B
def gradient_image(z, prog):
    """Closeness of the tissue to a query: every index tile's d_B^S percentile within the section (1 = nearest),
    weighted by tile area, smoothed over tile centres with a Gaussian of SMOOTH tile sides (Nadaraya-Watson on a
    grid of PIX tile sides), masked to the pixels with cells (closed, so held-out and dropped tiles are filled), then
    ranked over the tissue pixels: the share of the section's tissue farther from the query."""
    from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter, label
    from scipy.stats import rankdata

    xy, ctr, side = z["xy"], z["centre"], z["side"]
    d = z[f"d_{prog}"]
    v = 1 - (rankdata(d) - 1) / (len(d) - 1)
    px = PIX * np.median(side)
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    bx, by = np.arange(x0, x1 + px, px), np.arange(y0, y1 + px, px)
    w = side ** 2
    num, _, _ = np.histogram2d(ctr[:, 1], ctr[:, 0], [by, bx], weights=w * v)
    den, _, _ = np.histogram2d(ctr[:, 1], ctr[:, 0], [by, bx], weights=w)
    s = SMOOTH / PIX
    img = gaussian_filter(num, s) / np.maximum(gaussian_filter(den, s), 1e-12)
    occ, _, _ = np.histogram2d(xy[:, 1], xy[:, 0], [by, bx])
    mask = binary_fill_holes(binary_closing(occ > 0, iterations=2)) & (gaussian_filter(den, s) > 1e-3 * den.max())
    lab, _ = label(mask)
    size = np.bincount(lab.ravel())
    mask &= np.isin(lab, np.flatnonzero(size >= 0.01 * mask.sum())) & (lab > 0)  # stray cells off the section
    out = np.full(img.shape, np.nan)
    out[mask] = (rankdata(img[mask]) - 1) / (mask.sum() - 1)
    return out, (x0, x1, y0, y1)


def gradients():
    cmap = ListedColormap(colormaps["viridis"](np.linspace(0, 1, GRAD_STEPS)))  # steps keep the PNGs in budget
    absent = absent_pairs()
    for key in KEYS:
        z = np.load(RES / key / "seed0" / "gradient.npz", allow_pickle=True)
        w = GRAD_W[key]
        for p in shown():
            img, (x0, x1, y0, y1) = gradient_image(z, p)
            fig, ax = ps.atom(w, w * (y1 - y0) / (x1 - x0))
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1, extent=(x0, x1, y1, y0), interpolation="bilinear",
                      origin="upper", alpha=ABSENT_ALPHA if (key, p) in absent else 1.0)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.axis("off")
            ps.save(fig, FIG, f"grad_{key}_{p}", source=f"results/fig8_pan_cancer/{key}/seed0/gradient.npz")
    ps.colorbar_atom(FIG, "cbar_grad", colormaps["viridis"], 0, 1, 30, 7, ticks=[0, 0.5, 1],
                     orientation="horizontal", source="")


# ---------------------------------------------------------------- C, D
def per_section():
    """C gain in match to the query, D co-variation lift over cells (Figs 6-7); variances only as a grey tick."""
    T = pd.read_csv(RES / "programs.csv").set_index(["key", "program", "method"])
    progs = shown()
    off = {p: d for p, d in zip(progs, (-0.17, 0.17))}
    for col, ylab, lim, ticks in (("gain", "Gain in match to query", (0, 0.11), [0, 0.05, 0.1]),
                                  ("lift", "Co-variation lift over cells", (-0.06, 0.13), [-0.05, 0, 0.05, 0.1])):
        fig, ax = ps.atom(88, 42, left=10.0, bottom=5.5, right=1.0, top=1.5)
        ax.axhline(0, color=ps.MUTED, lw=0.5, ls="--", zorder=1)
        for i, key in enumerate(KEYS):
            for p in progs:
                x, c = i + off[p], PROG_COLORS[p]
                ax.plot([x - 0.1, x + 0.1], [T.loc[(key, p, "D"), col]] * 2, color=ps.MUTED, lw=0.8, zorder=2)
                for m, mk, dx in (("P", "o", -0.05), ("PL", "^", 0.05)):
                    r = T.loc[(key, p, m)]
                    ax.errorbar(x + dx, r[col], yerr=r[f"{col}_sd"] if np.isfinite(r[f"{col}_sd"]) else 0, fmt=mk,
                                ms=2.8 if m == "P" else 3.0, color=c, mew=0.6, elinewidth=0.6, capsize=0, zorder=3)
        ax.set_xticks(range(len(KEYS)))
        ax.set_xticklabels([ps.DATASET_LABELS[k] for k in KEYS])
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(-0.5, len(KEYS) - 0.5)
        for i in range(1, len(KEYS)):
            ax.axvline(i - 0.5, color=GREY, lw=0.5, zorder=0)
        ax.set_ylim(*lim)
        ax.set_yticks(ticks)
        ax.set_ylabel(ylab)
        ps.save(fig, FIG, col, source="results/fig8_pan_cancer/programs.csv")
    ps.legend_atom(FIG, "legend_cd",
                   [Patch(fc=PROG_COLORS[p], ec="none", label=PROG_LABELS[p]) for p in progs]
                   + [Line2D([], [], ls="", marker="o", ms=3, color=ps.INK, label="Spindle-partial"),
                      Line2D([], [], ls="", marker="^", ms=3, color=ps.INK, label="Spindle-lite-partial"),
                      Line2D([], [], ls="-", lw=0.8, color=ps.MUTED, label="Variances only")],
                   178, 4, ncol=5, handlelength=1.0, handletextpad=0.4, columnspacing=1.4, source="")


def drop_old_atoms():
    """Atoms of the first Fig 8 design (27 Oct), replaced by the gradient design."""
    import csv
    old = ["ex_map", "legend_prog", "legend_methods", "legend_gain", "delta"] + [f"map_{k}" for k in KEYS]
    hidden = [p for p in PROGS if p not in shown()]
    old += [f"rs_{p}" for p in hidden] + [f"grad_{k}_{p}" for k in KEYS for p in hidden]
    for name in old:
        for f in ps.atom_paths(FIG, name):
            f.unlink(missing_ok=True)
    if ps.MANIFEST.exists():
        with open(ps.MANIFEST) as fh:
            rows = [r for r in csv.DictReader(fh) if not (r["figure"] == FIG and r["atom"] in old)]
        with open(ps.MANIFEST, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=ps.FIELDS)
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    drop_old_atoms()
    queries()
    gradients()
    per_section()
