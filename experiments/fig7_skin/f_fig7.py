"""Fig 7 atoms from results/fig7_skin/ (c_queries.py, c_summary.py; login node).

Usage: python experiments/fig7_skin/f_fig7.py
A-B: the skin worked example (seed 0); C-E: every section, mean +/- s.d. over seeds 0-4.
Atoms (figures/fig7_skin/atoms/):
  sig_map_<list>, cbar_sig_<list>  the three skin signatures: cells by signature score, Spindle-partial's 10 nearest
                            tiles outlined (rule in c_queries.py)
  ex_map                    the example whole-tile query (rule in c_queries.py) and Spindle's top 10 on the section,
                            cells by skin class (common7.skin_classes)
  ex_query, ex_hit1..10     the query and its top 10, cells by class
  legend_class              the 4 skin classes
  ex_comp                   class composition of the query, the top 10, a random 10 (mean of 200 draws), the section
  tv_base                   all skin held-out queries (seed 0): composition difference to the top 10 and to a random 10
  tiles_all, legend_tiles   every section: median composition difference of held-out tiles to Spindle's top 10,
                            Spindle-lite's first 10 and a random 10
  delta_all, lift_all, legend_lists  every tissue signature on every section: Cliff's delta of the signature score
                            and co-variation lift on S of the 10 tiles of Spindle-partial and of Spindle-lite-partial
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import style as ps  # noqa: E402

FIG = "fig7_skin"
RES = ps.ROOT / "results" / FIG
S0 = RES / "skin_melanoma" / "seed0"
SIG_ORDER = ["Melanocyte_Tumor", "Keratinocyte_Epidermis", "Immune_Infiltrate"]
CLASS_COLORS = ["#882255", "#EE7733", "#33BBEE", "#DDDDDD"]  # melanocyte, keratinocyte, immune, other
SPINDLE, LITE, PLITE = ps.TIER_COLORS["spindle"], ps.TIER_COLORS["lite"], ps.TIER_COLORS["plite"]
MAP_W = 58.0
KEYS = ps.DATASET_ORDER
XLAB = [ps.DATASET_LABELS[k] for k in KEYS]


def spatial(ax, x0, x1, y0, y1):
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # image orientation
    ax.set_aspect("equal")
    ax.axis("off")


def extent(xy, w):
    (x0, y0), (x1, y1) = xy.min(0), xy.max(0)
    return x0, x1, y0, y1, w * (y1 - y0) / (x1 - x0)


def dataset_axis(ax):
    ax.set_xticks(range(len(KEYS)))
    ax.set_xticklabels(XLAB, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, len(KEYS) - 0.4)


# ---------------------------------------------------------------- A
def signature_maps():
    z = np.load(S0 / "signature_cells.npz")
    xy = z["xy"]
    x0, x1, y0, y1, h = extent(xy, MAP_W)
    cmap = ListedColormap(colormaps["viridis"](np.linspace(0, 1, 10)))  # 10 steps keep the PNG within budget
    for name in SIG_ORDER:
        s = z[f"score_{name}"].astype(np.float32)
        vmax = float(np.quantile(s, 0.995))
        fig, ax = ps.atom(MAP_W, h)
        o = np.argsort(s, kind="stable")
        ax.scatter(xy[o, 0], xy[o, 1], s=0.12, c=s[o], cmap=cmap, vmin=0, vmax=vmax, lw=0, rasterized=True,
                   antialiased=False)
        for bx0, by0, bx1, by1 in z[f"top_{name}"]:
            ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=ps.QUERY_COLOR, lw=0.7))
        spatial(ax, x0, x1, y0, y1)
        ps.save(fig, FIG, f"sig_map_{name}", source="results/fig7_skin/skin_melanoma/seed0/signature_cells.npz")
        ps.colorbar_atom(FIG, f"cbar_sig_{name}", cmap, 0, vmax, 22, 7, ticks=[0, np.floor(10 * vmax) / 10],
                         orientation="horizontal", source="")


# ---------------------------------------------------------------- B
def example():
    z = np.load(S0 / "example_cells.npz")
    names = [str(n) for n in z["classes"]]
    col = np.array(CLASS_COLORS)
    xy, cls = z["xy"], z["cls"]
    x0, x1, y0, y1, h = extent(xy, 60.0)
    fig, ax = ps.atom(60.0, h)
    o = np.argsort(cls != len(names) - 1, kind="stable")  # "Other" underneath
    ax.scatter(xy[o, 0], xy[o, 1], s=0.12, c=col[cls[o]], lw=0, rasterized=True, antialiased=False)
    for k in range(len(z["tiles"]) - 1, -1, -1):
        bx0, by0, bx1, by1 = z[f"bbox{k}"]
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, lw=0.8,
                               ec=ps.QUERY_COLOR if k == 0 else SPINDLE, zorder=3 if k == 0 else 2))
    spatial(ax, x0, x1, y0, y1)
    ps.save(fig, FIG, "ex_map", source="results/fig7_skin/skin_melanoma/seed0/example_cells.npz")

    for k in range(len(z["tiles"])):
        bx0, by0, bx1, by1 = z[f"bbox{k}"]
        side = max(bx1 - bx0, by1 - by0) * 1.06
        cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        fig, ax = ps.atom(9, 9)
        ax.scatter(*z[f"xy{k}"].T, s=0.8, c=col[z[f"cls{k}"]], lw=0, rasterized=True)
        ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, lw=0.9,
                               ec=ps.QUERY_COLOR if k == 0 else SPINDLE))
        spatial(ax, cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2)
        ps.save(fig, FIG, "ex_query" if k == 0 else f"ex_hit{k}",
                source="results/fig7_skin/skin_melanoma/seed0/example_cells.npz")
    ps.legend_atom(FIG, "legend_class", [Patch(fc=c, ec="none", label=n) for n, c in zip(names, CLASS_COLORS)],
                   60, 4, ncol=4, handlelength=0.8, columnspacing=0.8, handletextpad=0.3, source="")

    ex = pd.read_csv(S0 / "example.csv").set_index("set")
    sets = [("query", "Query"), ("top10", "Top 10"), ("random10", "Random 10"), ("section", "Section")]
    fig, ax = ps.atom(26, 40, left=8.0, bottom=11.0, right=1.0, top=1.5)
    for i, (s, _) in enumerate(sets):
        bottom = 0.0
        for n, c in zip(names, CLASS_COLORS):
            v = ex.loc[s, n]
            ax.bar(i, v, 0.75, bottom=bottom, color=c, lw=0)
            bottom += v
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([l for _, l in sets], rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Fraction of cells")
    ps.save(fig, FIG, "ex_comp", source="results/fig7_skin/skin_melanoma/seed0/example.csv")

    q = pd.read_csv(S0 / "tile_queries.csv")
    fig, ax = ps.atom(24, 40, left=8.0, bottom=11.0, right=1.0, top=1.5)
    rng = np.random.default_rng(0)
    for i, (c, color) in enumerate((("tv_top", SPINDLE), ("tv_random", ps.MUTED))):
        v = q[c].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=1.5, c=color, lw=0, alpha=0.7, zorder=2)
        ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=ps.INK, lw=0.75, zorder=3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Top 10", "Random 10"], rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(0, 1)
    ax.set_yticks([0, 0.5, 1])
    ax.set_ylabel("Composition difference")
    ps.save(fig, FIG, "tv_base", source="results/fig7_skin/skin_melanoma/seed0/tile_queries.csv")


# ---------------------------------------------------------------- C
def tiles_all():
    T = pd.read_csv(RES / "tiles.csv")
    cols = ["tv_top", "tv_lite", "tv_random"]
    m = T.groupby("key")[cols].mean().loc[KEYS]
    s = T.groupby("key")[cols].std().loc[KEYS]
    kinds = [("tv_top", "Spindle, top 10", SPINDLE, "o", -0.2, True), ("tv_lite", "Spindle-lite, first 10", LITE,
             "D", 0.0, True), ("tv_random", "Random 10", ps.MUTED, "o", 0.2, False)]
    fig, ax = ps.atom(58, 44, left=9.0, bottom=13.0, right=1.0, top=1.5)
    for c, _, color, mk, dx, filled in kinds:
        x = np.arange(len(KEYS)) + dx
        ax.errorbar(x, m[c], yerr=s[c], fmt=mk, ms=2.8, mfc=color if filled else "white", mec=color, mew=0.6,
                    ecolor=color, elinewidth=0.6, capsize=0, zorder=2)
    dataset_axis(ax)
    ax.set_ylim(0, 0.7)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    ax.set_ylabel("Composition difference")
    ps.save(fig, FIG, "tiles_all", source="results/fig7_skin/tiles.csv")
    ps.legend_atom(FIG, "legend_tiles",
                   [Line2D([], [], ls="", marker=mk, ms=3, mfc=c if f else "white", mec=c, mew=0.6, label=l)
                    for _, l, c, mk, _, f in kinds], 58, 4, ncol=3, handletextpad=0.2, columnspacing=0.8, source="")


# ---------------------------------------------------------------- D, E
def lists_all():
    L = pd.read_csv(RES / "lists.csv")
    t = L[L["query"] == "T"].set_index(["key", "list"])
    tl = L[L["query"] == "TL"].set_index(["key", "list"]).loc[t.index]
    rng = np.random.default_rng(0)
    xi = np.array([KEYS.index(k) for k, _ in t.index], dtype=float)
    jit = rng.uniform(-0.06, 0.06, len(xi))
    for col, name, ylab, lim, ticks in (("delta", "delta_all", "Cliff's δ", (-1, 1), [-1, 0, 1]),
                                        ("lift", "lift_all", "Co-variation lift on S", (-0.15, 0.5),
                                         [0, 0.2, 0.4])):
        fig, ax = ps.atom(58, 44, left=9.0, bottom=13.0, right=1.0, top=1.5)
        ax.axhline(0, color=ps.MUTED, lw=0.5, ls="--", zorder=1)
        a, b = t[col].to_numpy(), tl[col].to_numpy()
        for x, ya, yb in zip(xi + jit, a, b):
            ax.plot([x - 0.17, x + 0.17], [ya, yb], color=ps.LIGHT, lw=0.5, zorder=1)
        ax.scatter(xi + jit - 0.17, a, s=6, marker="o", c=SPINDLE, lw=0, zorder=2)
        ax.scatter(xi + jit + 0.17, b, s=7, marker="^", c=PLITE, lw=0, zorder=2)
        dataset_axis(ax)
        ax.set_ylim(*lim)
        ax.set_yticks(ticks)
        ax.set_ylabel(ylab)
        ps.save(fig, FIG, name, source="results/fig7_skin/lists.csv")
    ps.legend_atom(FIG, "legend_lists",
                   [Line2D([], [], ls="", marker="o", ms=3, color=SPINDLE, label="Spindle-partial, top 10"),
                    Line2D([], [], ls="", marker="^", ms=3, color=PLITE, label="Spindle-lite-partial, first 10")],
                   90, 4, ncol=2, handletextpad=0.2, columnspacing=1.2, source="")


if __name__ == "__main__":
    signature_maps()
    example()
    tiles_all()
    lists_all()
