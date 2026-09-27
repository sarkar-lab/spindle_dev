"""Fig. 1A tissue only: the breast cells and adaptive quadtree tiles, drawn exactly as in
fig1_subpanels.panel_a but without the zoom inset, its connector lines and the example-tile outline.

Standalone (not part of run_make_figures.sbatch; fig1_subpanels.py is untouched). Writes
figures/pdf/fig1a_tissue_tiles.pdf and figures/png/fig1a_tissue_tiles.png, both saved from the
figure itself. Inputs: results/figure_data/fig1/ (scripts/extract_fig1_data.py).

Usage: python scripts/fig1a_tissue_tiles.py
"""

import matplotlib.pyplot as plt

import figstyle as fs
import fig1_subpanels as f1

NAME = "fig1a_tissue_tiles"


def main():
    d = f1.load()
    t, cells = d["tiles"], d["cells"]
    W = 34.0  # same width, marker size and line widths as panel A
    H = W * f1.tissue_aspect(t)
    fig = f1.part(W, H)
    ax = f1.ax_mm(fig, 0, 0, W, H)
    ax.scatter(cells.x, cells.y, s=0.02, c=f1.CELL_GREY, lw=0, rasterized=True, zorder=1)
    f1.rects(ax, t, facecolor="none", edgecolor=f1.TILE_EDGE, lw=0.12, zorder=2)
    f1.tissue_axes(ax, t)

    fs.PDF_DIR.mkdir(parents=True, exist_ok=True)
    pdf = fs.PDF_DIR / f"{NAME}.pdf"
    fig.savefig(pdf, dpi=600, transparent=True)
    fig.savefig(fs.png_path(pdf), dpi=fs.PNG_DPI, transparent=True)
    plt.close(fig)
    print(f"saved {pdf.relative_to(fs.PROJECT_ROOT)} + {fs.png_path(pdf).relative_to(fs.PROJECT_ROOT)} "
          f"({W:.1f} x {H:.1f} mm)")


if __name__ == "__main__":
    main()
