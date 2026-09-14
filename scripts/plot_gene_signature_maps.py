#!/usr/bin/env python3
"""
plot_gene_signature_maps.py
============================
Regenerates the dual spatial-comparison figure (continuous pathway score vs.
SPINDLE top-match tiles) for every (dataset, module) pair produced by
``benchmarks/gene_signature_search.py``.

Reads ``results/gene_signature_search/<dataset>/<module>_spatial_cells.csv`` and
``..._top_matches.csv``, and writes
``results/gene_signature_search/<dataset>/<module>_spatial_comparison.png`` (+ .pdf),
reusing the shared publication style from ``generate_figure_1.py``.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from generate_figure_1 import (
    set_publication_style,
    LIGHT_GRAY,
    BASE_FONT,
    TICK_FONT,
    TITLE_FONT,
    ANNOT_FONT,
)

RESULTS_DIR = project_root / "results" / "gene_signature_search"


def plot_module_comparison(df_cells: pd.DataFrame, df_matches: pd.DataFrame, module_name: str, dataset_name: str):
    """
    Builds the dual spatial map for one (dataset, module) pair:
      LEFT  — all cells coloured by continuous pathway score (magma)
      RIGHT — SPINDLE top-match tiles highlighted over a greyscale background,
              with rank-ordered bounding boxes
    """
    fig = plt.figure(figsize=(14, 7))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.08,
                            left=0.01, right=0.93, top=0.92, bottom=0.02)
    ax_cont = fig.add_subplot(gs[0])
    ax_match = fig.add_subplot(gs[1])

    x = df_cells['x'].values
    y = df_cells['y'].values
    ps = df_cells['pathway_score'].values

    vmin, vmax = np.percentile(ps, 2), np.percentile(ps, 98)

    # LEFT: continuous pathway score, all cells
    sc = ax_cont.scatter(x, y, c=ps, cmap='magma',
                          vmin=vmin, vmax=vmax,
                          s=0.6, alpha=0.75, rasterized=True,
                          linewidths=0)
    cb = fig.colorbar(sc, ax=ax_cont, shrink=0.65, pad=0.02)
    cb.set_label('Pathway Score', fontsize=BASE_FONT - 1, fontweight='bold')
    cb.ax.tick_params(labelsize=TICK_FONT - 1)
    ax_cont.set_aspect('equal')
    ax_cont.axis('off')

    # RIGHT: greyscale background + SPINDLE tiles highlighted
    ax_match.scatter(x, y, c='#D0D0D0', s=0.4, alpha=0.40,
                      rasterized=True, linewidths=0)

    n_tiles = len(df_matches) if df_matches is not None else 0
    if df_matches is not None and n_tiles > 0:
        in_match = np.zeros(len(x), dtype=bool)
        for _, row in df_matches.iterrows():
            mask = ((x >= row['x0']) & (x <= row['x1']) &
                    (y >= row['y0']) & (y <= row['y1']))
            in_match |= mask

        if in_match.any():
            sc2 = ax_match.scatter(
                x[in_match], y[in_match],
                c=ps[in_match], cmap='magma',
                vmin=vmin, vmax=vmax,
                s=1.8, alpha=0.90, rasterized=True, linewidths=0)
            cb2 = fig.colorbar(sc2, ax=ax_match, shrink=0.65, pad=0.02)
            cb2.set_label('Pathway Score', fontsize=BASE_FONT - 1, fontweight='bold')
            cb2.ax.tick_params(labelsize=TICK_FONT - 1)

    ax_match.set_aspect('equal')
    ax_match.axis('off')

    title_prefix = module_name.replace('_', ' ')
    ax_cont.set_title(f'{title_prefix} — {dataset_name} — Pathway Score',
                       fontsize=TITLE_FONT, pad=6, fontweight='bold')
    ax_match.set_title(f'{title_prefix} — {dataset_name} — SPINDLE Matches (n={n_tiles})',
                        fontsize=TITLE_FONT, pad=6, fontweight='bold')

    return fig


def save_figure(fig, out_dir: Path, module_name: str):
    for fmt in ('png', 'pdf'):
        p = out_dir / f"{module_name}_spatial_comparison.{fmt}"
        fig.savefig(str(p), bbox_inches='tight', dpi=300, format=fmt)
        print(f"    saved -> {p.relative_to(project_root)}")
    plt.close(fig)


def main():
    set_publication_style()

    if not RESULTS_DIR.exists():
        print(f"No results found at {RESULTS_DIR}. Run benchmarks/gene_signature_search.py first.")
        return

    cell_csvs = sorted(RESULTS_DIR.glob("*/*_spatial_cells.csv"))
    if not cell_csvs:
        print(f"No *_spatial_cells.csv files found under {RESULTS_DIR}.")
        return

    for cells_path in cell_csvs:
        dataset_name = cells_path.parent.name
        module_name = cells_path.name[: -len("_spatial_cells.csv")]
        matches_path = cells_path.parent / f"{module_name}_top_matches.csv"

        print(f"Plotting {dataset_name} / {module_name} ...")
        try:
            df_cells = pd.read_csv(cells_path)
            df_matches = pd.read_csv(matches_path) if matches_path.exists() else None
            fig = plot_module_comparison(df_cells, df_matches, module_name, dataset_name)
            save_figure(fig, cells_path.parent, module_name)
        except Exception as e:
            print(f"  WARNING: failed to plot {dataset_name}/{module_name}: {e}")

    print("\nAll gene-signature spatial comparison figures regenerated.")


if __name__ == "__main__":
    main()
