#!/usr/bin/env python3
"""
generate_figure_1.py
====================
Generates publication-quality Figure 1 panels for SPINDLE.

Reads from ``results/panel_data/`` (produced by ``organize_panel_data.py``).
All panels are exported as individual high-resolution standalone files, and a
composite backup is saved to ``results/fig_main_result.pdf/.png``.

Individual exports
------------------
  figures/panels/panel_A.pdf/.png        — Index scalability
  figures/panels/panel_B.pdf/.png        — Query speedup (horizontal bar)
  figures/panels/panel_C.pdf/.png        — Holdout rank distribution
  figures/panels/panel_D.pdf/.png        — Partial-query robustness
  figures/panels/panel_E_xenium.pdf/.png — Xenium tile overlay
  figures/panels/panel_E_visium.pdf/.png — Visium tile overlay
  figures/panels/panel_E_recall.pdf/.png — Cross-modal recall bar chart
  figures/panels/panel_F_map.pdf/.png    — Gene-sig spatial map (dual)
  figures/panels/panel_F_pathway.pdf/.png— Pathway score bar chart
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns

# ── Shared color palette ─────────────────────────────────────────────────────
NAVY       = '#1B365D'
TEAL       = '#2A9D8F'
TERRACOTTA = '#C85A32'
SLATE      = '#4A607A'
PURPLE     = '#6B4C9A'
LIGHT_GRAY = '#F0F2F5'
DARK_GRAY  = '#2C3E50'
GOLD       = '#D4AF37'
STEEL_BLUE = '#3A7CBF'
SAGE       = '#6B8E77'

# ── Publication style ────────────────────────────────────────────────────────
BASE_FONT   = 13.0   # axis labels
TICK_FONT   = 11.5   # tick labels
LEGEND_FONT = 11.0
ANNOT_FONT  = 11.0   # bar / dot annotations
TITLE_FONT  = 14.0
LETTER_FONT = 18.0   # panel letter (A, B, …)


def set_publication_style():
    """Apply clean, publication-ready styling — no grids, larger fonts."""
    sns.set_theme(style='ticks', context='paper')  # 'ticks' removes grid
    plt.rcParams.update({
        'font.family':        'sans-serif',
        'font.sans-serif':    ['Arial', 'Helvetica', 'DejaVu Sans'],
        # axes
        'axes.edgecolor':     '#4A4A4A',
        'axes.linewidth':     1.2,
        'axes.titlesize':     TITLE_FONT,
        'axes.titleweight':   'bold',
        'axes.labelsize':     BASE_FONT,
        'axes.labelweight':   'bold',
        'axes.grid':          False,      # globally disable grids
        'grid.alpha':         0.0,
        # ticks
        'xtick.labelsize':    TICK_FONT,
        'ytick.labelsize':    TICK_FONT,
        'xtick.direction':    'out',
        'ytick.direction':    'out',
        'xtick.major.size':   5,
        'ytick.major.size':   5,
        # legend
        'legend.fontsize':    LEGEND_FONT,
        'legend.framealpha':  0.9,
        # figure
        'figure.titlesize':   16,
        'figure.titleweight': 'bold',
        'savefig.dpi':        300,
    })


def _clean_axes(ax):
    """Remove top and right spines; keep only left and bottom."""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def add_panel_letter(ax, letter: str):
    """Bold panel label in the upper-left corner."""
    ax.text(-0.10, 1.14, letter, transform=ax.transAxes,
            fontsize=LETTER_FONT, fontweight='bold',
            va='top', ha='right', color='#000000')


# ── Helpers ───────────────────────────────────────────────────────────────────
def _read_panel_csv(project_root: Path, filename: str) -> pd.DataFrame | None:
    """Read a panel CSV from results/panel_data/, skipping comment lines."""
    path = project_root / 'results' / 'panel_data' / filename
    if not path.exists():
        print(f'  WARNING: {path.relative_to(project_root)} not found — using fallback values')
        return None
    return pd.read_csv(path, comment='#')


def _save(fig_obj, panels_dir: Path, name: str):
    for fmt in ('pdf', 'png'):
        p = panels_dir / f'{name}.{fmt}'
        fig_obj.savefig(str(p), bbox_inches='tight', dpi=300, format=fmt)
        print(f'    saved -> {p.relative_to(panels_dir.parent.parent)}')
    plt.close(fig_obj)


# ── Panel A: Index Scalability — Dual-axis Bar Chart ────────────────────────
def plot_panel_a_scalability(ax, project_root: Path):
    """
    Panel A — Dual-axis grouped bar chart: Build Time (left, NAVY) and
    Index Size (right, TERRACOTTA) per dataset, sorted by cell count.
    """
    # Fallback values (sorted by cell count)
    datasets = ['Skin', 'Kidney', 'Breast', 'Lung', 'Pancreas', 'Lymph Node']
    build_t  = [25.53,  18.62,   51.67,  45.86,  88.90,  71.51]
    idx_size = [19.41,  18.47,   14.69,  19.98,  30.86,  22.83]

    df = _read_panel_csv(project_root, 'panel_A.csv')
    if df is not None:
        try:
            col_ds = next(c for c in df.columns if 'dataset' in c.lower())
            col_bt = next(c for c in df.columns if 'build' in c.lower())
            col_sz = next(c for c in df.columns if 'size' in c.lower() or 'index' in c.lower())
            datasets = df[col_ds].tolist()
            build_t  = df[col_bt].tolist()
            idx_size = df[col_sz].tolist()
        except Exception as e:
            print(f'  WARNING (Panel A): {e}')

    x     = np.arange(len(datasets))
    width = 0.35
    ax2   = ax.twinx()

    ax.bar( x - width/2, build_t,  width, color=NAVY,       alpha=0.88, edgecolor='none')
    ax2.bar(x + width/2, idx_size, width, color=TERRACOTTA,  alpha=0.88, edgecolor='none')

    ax.set_ylabel('Index Build Time (s)',    color=NAVY,      fontweight='bold', fontsize=BASE_FONT)
    ax2.set_ylabel('Index Size on Disk (MB)', color=TERRACOTTA, fontweight='bold', fontsize=BASE_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha='right', fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelcolor=NAVY,       labelsize=TICK_FONT)
    ax2.tick_params(axis='y', labelcolor=TERRACOTTA, labelsize=TICK_FONT)

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(False)
    ax2.grid(False)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NAVY,       label='Build Time (s)'),
        Patch(facecolor=TERRACOTTA, label='Index Size (MB)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True,
              facecolor='white', fontsize=LEGEND_FONT)



# ── Panel B: Query Acceleration — Horizontal Bar Chart ──────────────────────
def plot_panel_b_speedup(ax, project_root: Path):
    """
    Panel B — Horizontal bar chart of query speedup (x) per dataset.
    Replaces the lollipop chart.  Brute-force baseline shown as a dashed line.
    """
    # Fallback
    lbl_s = ['Skin', 'Lymph Node', 'Lung', 'Pancreas', 'Breast', 'Kidney']
    spd_s = [5.83,    7.38,        7.70,   7.84,       9.31,     9.93]

    df = _read_panel_csv(project_root, 'panel_B.csv')
    if df is not None:
        try:
            col_spd = next(c for c in df.columns if 'speedup' in c.lower())
            col_lbl = 'label' if 'label' in df.columns else df.columns[0]
            lbl_s = df[col_lbl].tolist()
            spd_s = df[col_spd].tolist()
        except Exception as e:
            print(f'  WARNING (Panel B): {e}')

    # Sort descending so tallest bar is on the right
    paired = sorted(zip(lbl_s, spd_s), key=lambda t: t[1])
    lbl_s  = [p[0] for p in paired]
    spd_s  = [p[1] for p in paired]

    x = np.arange(len(lbl_s))

    ax.bar(x, spd_s, width=0.55, color=TEAL, alpha=0.88, edgecolor='none')

    # Value labels above each bar
    for xi, spd in zip(x, spd_s):
        ax.text(xi, spd + 0.3, f'{spd:.1f}\u00d7', ha='center',
                fontsize=ANNOT_FONT, fontweight='bold', color=SLATE)

    ax.axhline(y=1, color='crimson', linestyle='--', alpha=0.70, lw=1.8,
               label='Brute-Force baseline (1\u00d7)')

    ax.set_xticks(x)
    ax.set_xticklabels(lbl_s, rotation=20, ha='right', fontsize=TICK_FONT)
    ax.set_ylabel('Speedup vs. Exact Brute-Force Search (\u00d7)', fontweight='bold',
                  fontsize=BASE_FONT)
    ax.set_ylim(0, max(spd_s) * 1.25)
    ax.legend(loc='upper left', frameon=True, facecolor='white', fontsize=LEGEND_FONT)
    _clean_axes(ax)



# ── Panel C: Holdout Validation — Rank-Bucket Bar Chart ─────────────────────
def plot_panel_c_accuracy(ax, project_root: Path):
    """
    Panel C — Bar chart of rank-bucket percentages.
    """
    labels = ['1st', '2nd', '3rd', '4-5th', '6-10th', '>10th']
    pcts   = [74.77, 13.55, 3.27, 6.07, 1.87, 0.47]   # fallback

    df = _read_panel_csv(project_root, 'panel_C.csv')
    if df is not None:
        try:
            col_cat = next(c for c in df.columns if 'rank' in c.lower() or 'category' in c.lower())
            col_pct = next(c for c in df.columns if 'percent' in c.lower() or 'pct' in c.lower())
            lmap = dict(zip(df[col_cat], df[col_pct]))
            pcts = [lmap.get(l, 0.0) for l in labels]
        except Exception as e:
            print(f'  WARNING (Panel C): {e}')

    ax.bar(labels, pcts, color=PURPLE, alpha=0.88, width=0.55, edgecolor='none')

    for i, (lbl, p) in enumerate(zip(labels, pcts)):
        if p > 0.3:
            ax.text(i, p + 1.2, f'{p:.1f}%',
                    ha='center', va='bottom', fontsize=ANNOT_FONT,
                    fontweight='bold', color=PURPLE)

    ax.set_ylabel('Queries (%)', fontweight='bold', fontsize=BASE_FONT)
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=20, labelsize=TICK_FONT)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    _clean_axes(ax)



# ── Panel D: Partial Query — 4-Bin Line Chart ───────────────────────────────
def plot_panel_d_partial(ax, project_root: Path):
    """
    Panel D — Line chart of Recall@1 and Overlap@10 across gene-coverage bins.
    """
    order_labels   = ['\u22646 genes', '7\u201312 genes', '13\u201316 genes', '>16 genes']
    recall1_vals   = [62.4, 78.5, 88.2, 94.8]
    overlap10_vals = [51.2, 71.4, 83.5, 90.9]

    df = _read_panel_csv(project_root, 'panel_D.csv')
    if df is not None:
        try:
            col_bin = next(c for c in df.columns if 'bin' in c.lower() or 'length' in c.lower())
            col_r1  = next(c for c in df.columns if 'recall' in c.lower())
            col_ov  = next(c for c in df.columns if 'overlap' in c.lower())
            order_labels   = (df[col_bin]
                              .str.replace('<=', '\u2264', regex=False)
                              .str.replace('-', '\u2013', regex=False)
                              .tolist())
            recall1_vals   = df[col_r1].tolist()
            overlap10_vals = df[col_ov].tolist()
        except Exception as e:
            print(f'  WARNING (Panel D): {e}')

    x_pos = np.arange(len(order_labels))
    ax.plot(x_pos, recall1_vals,   marker='o', lw=2.5, markersize=9,
            color=TERRACOTTA, label='Recall@1 (%)')
    ax.plot(x_pos, overlap10_vals, marker='s', lw=2.5, markersize=9,
            color=NAVY,       label='Overlap@10 (%)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(order_labels, fontsize=TICK_FONT)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=BASE_FONT)
    ax.set_xlabel('Partial Query Coverage (Gene Transcript Length)',
                  fontweight='bold', fontsize=BASE_FONT)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=TICK_FONT)
    ax.legend(loc='lower right', frameon=True, facecolor='white', fontsize=LEGEND_FONT)
    _clean_axes(ax)



# ── Panel E: Xenium tile overlay ─────────────────────────────────────────────
def plot_panel_e_xenium(ax, project_root: Path):
    """
    Panel E (left) — Xenium cell scatter + tile bounding boxes.
    Cells coloured by density (2-D KDE-like binning) to improve visibility.
    """
    df_xe    = _read_panel_csv(project_root, 'panel_E_xenium_coords.csv')
    df_boxes = _read_panel_csv(project_root, 'panel_E_tile_boxes.csv')

    rendered = False
    if df_xe is not None:
        try:
            x, y = df_xe['x'].values, df_xe['y'].values

            # Hexbin density instead of plain scatter — visually rich + readable
            hb = ax.hexbin(x, y, gridsize=60, cmap='Blues', mincnt=1,
                           linewidths=0.0, alpha=0.85)
            plt.colorbar(hb, ax=ax, shrink=0.55, label='Cell density',
                         pad=0.02).ax.tick_params(labelsize=TICK_FONT - 1)

            if df_boxes is not None:
                xe_boxes = df_boxes[df_boxes['modality'] == 'Xenium']
                for _, row in xe_boxes.iterrows():
                    w = row['x1'] - row['x0']
                    h = row['y1'] - row['y0']
                    ax.add_patch(patches.FancyBboxPatch(
                        (row['x0'], row['y0']), w, h,
                        boxstyle='square,pad=0',
                        lw=1.5, edgecolor=TERRACOTTA,
                        facecolor=TERRACOTTA, alpha=0.12))
                    ax.add_patch(patches.Rectangle(
                        (row['x0'], row['y0']), w, h,
                        lw=1.5, edgecolor=TERRACOTTA, facecolor='none'))
                n_xe = len(xe_boxes)
            else:
                n_xe = '?'

            ax.set_title(f'Xenium ({n_xe} tiles)', fontsize=TITLE_FONT, pad=8, fontweight='bold')
            # Use 'auto' aspect in the composite so no dead whitespace on sides
            ax.set_aspect('auto')
            ax.axis('off')
            rendered = True
        except Exception as e:
            print(f'  WARNING (Panel E Xenium): {e}')

    if not rendered:
        ax.text(0.5, 0.5, 'Xenium\n[CSV missing]', ha='center', va='center',
                bbox=dict(fc=LIGHT_GRAY), fontsize=ANNOT_FONT)
        ax.axis('off')




def plot_panel_e_visium(ax, project_root: Path):
    """
    Panel E (centre) — Visium spot scatter + tile bounding boxes.
    Uses a perceptually uniform colormap (viridis) instead of flat orange.
    """
    df_vi    = _read_panel_csv(project_root, 'panel_E_visium_coords.csv')
    df_boxes = _read_panel_csv(project_root, 'panel_E_tile_boxes.csv')

    rendered = False
    if df_vi is not None:
        try:
            x, y = df_vi['x'].values, df_vi['y'].values

            # Colour Visium spots by density using hexbin (matches Xenium style)
            hb = ax.hexbin(x, y, gridsize=40, cmap='YlOrRd', mincnt=1,
                           linewidths=0.0, alpha=0.85)
            plt.colorbar(hb, ax=ax, shrink=0.55, label='Spot density',
                         pad=0.02).ax.tick_params(labelsize=TICK_FONT - 1)

            if df_boxes is not None:
                vi_boxes = df_boxes[df_boxes['modality'] == 'Visium']
                for _, row in vi_boxes.iterrows():
                    w = row['x1'] - row['x0']
                    h = row['y1'] - row['y0']
                    ax.add_patch(patches.Rectangle(
                        (row['x0'], row['y0']), w, h,
                        lw=1.5, edgecolor=TEAL, facecolor=TEAL, alpha=0.12))
                    ax.add_patch(patches.Rectangle(
                        (row['x0'], row['y0']), w, h,
                        lw=1.5, edgecolor=TEAL, facecolor='none'))
                n_vi = len(vi_boxes)
            else:
                n_vi = '?'

            ax.set_title(f'Visium ({n_vi} tiles)', fontsize=TITLE_FONT, pad=8, fontweight='bold')
            # Use 'auto' aspect so the tile grid fills its subplot cell
            ax.set_aspect('auto')
            ax.axis('off')
            rendered = True
        except Exception as e:
            print(f'  WARNING (Panel E Visium): {e}')

    if not rendered:
        ax.text(0.5, 0.5, 'Visium\n[CSV missing]', ha='center', va='center',
                bbox=dict(fc=LIGHT_GRAY), fontsize=ANNOT_FONT)
        ax.axis('off')


def plot_panel_e_recall(ax, project_root: Path):
    """
    Panel E (right) — Grouped horizontal bar chart: Recall@1 + Overlap@10
    per cross-modal query direction.
    """
    # Fallback
    directions  = ['Xenium \u2192 Visium', 'Visium \u2192 Xenium']
    recall1     = [100.0, 84.0]
    overlap10   = [90.0,  78.2]

    df = _read_panel_csv(project_root, 'panel_E_recall_metrics.csv')
    if df is not None:
        try:
            col_lbl = next((c for c in df.columns if 'label' in c.lower()), None)
            if col_lbl is None:
                col_dir = next(c for c in df.columns if 'direction' in c.lower())
                dir_map = {'x2v': 'Xenium \u2192 Visium', 'v2x': 'Visium \u2192 Xenium'}
                directions = [dir_map.get(d, d) for d in df[col_dir].tolist()]
            else:
                # Unicode arrows
                directions = [d.replace('->', '\u2192') for d in df[col_lbl].tolist()]

            col_r = next(c for c in df.columns if 'recall' in c.lower())
            col_o = next(c for c in df.columns if 'overlap' in c.lower() and '10' in c)
            recall1   = df[col_r].tolist()
            overlap10 = df[col_o].tolist()
        except Exception as e:
            print(f'  WARNING (Panel E recall): {e}')

    n   = len(directions)
    y   = np.arange(n)
    bh  = 0.30

    # Two grouped bars per direction
    bars_r  = ax.barh(y + bh/2, recall1,   height=bh, color=TEAL,       alpha=0.88, edgecolor='none', label='Recall@1 (%)')
    bars_o  = ax.barh(y - bh/2, overlap10, height=bh, color=TERRACOTTA, alpha=0.88, edgecolor='none', label='Overlap@10 (%)')

    for val, bar in list(zip(recall1, bars_r)) + list(zip(overlap10, bars_o)):
        w = bar.get_width()
        ax.text(w + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}%', va='center', fontsize=ANNOT_FONT,
                fontweight='bold', color=SLATE)

    ax.axvline(x=100, color=SLATE, linestyle=':', alpha=0.55, lw=1.5, label='Perfect recall')
    ax.set_yticks(y)
    ax.set_yticklabels(directions, fontsize=TICK_FONT)
    ax.set_xlabel('Score (%)', fontweight='bold', fontsize=BASE_FONT)
    ax.set_xlim(0, 120)
    ax.set_ylim(-0.7, n - 0.3)
    ax.legend(loc='lower right', frameon=True, facecolor='white', fontsize=LEGEND_FONT)
    _clean_axes(ax)


# ── Panel F: Gene Signature Spatial Map ──────────────────────────────────────
def plot_panel_f_map(fig, gs_left, gs_right, project_root: Path):
    """
    Panel F (left) — Dual spatial map styled like fig_gene_sig_luminal.png:
      LEFT  sub-axis: all cells coloured by continuous pathway score (viridis)
      RIGHT sub-axis: SPINDLE top-match tiles highlighted over greyscale H&E-like bg
    """
    ax_cont  = fig.add_subplot(gs_left)
    ax_match = fig.add_subplot(gs_right)

    df_cells   = _read_panel_csv(project_root, 'panel_F_spatial_cells.csv')
    df_matches = _read_panel_csv(project_root, 'panel_F_top_matches.csv')

    rendered = False
    if df_cells is not None:
        try:
            x  = df_cells['x'].values
            y  = df_cells['y'].values
            ps = df_cells['pathway_score'].values

            vmin, vmax = np.percentile(ps, 2), np.percentile(ps, 98)

            # LEFT: continuous pathway score (viridis, all cells)
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
            # Cells not in top matches: render as light gray
            ax_match.scatter(x, y, c='#D0D0D0', s=0.4, alpha=0.40,
                             rasterized=True, linewidths=0)

            if df_matches is not None and len(df_matches) > 0:
                # Collect all cells inside top-match tiles
                in_match = np.zeros(len(x), dtype=bool)
                for _, row in df_matches.iterrows():
                    mask = ((x >= row['x0']) & (x <= row['x1']) &
                            (y >= row['y0']) & (y <= row['y1']))
                    in_match |= mask

                # Highlight matched cells with pathway score colour
                if in_match.any():
                    sc2 = ax_match.scatter(
                        x[in_match], y[in_match],
                        c=ps[in_match], cmap='magma',
                        vmin=vmin, vmax=vmax,
                        s=1.8, alpha=0.90, rasterized=True, linewidths=0)
                    cb2 = fig.colorbar(sc2, ax=ax_match, shrink=0.65, pad=0.02)
                    cb2.set_label('Pathway Score', fontsize=BASE_FONT - 1, fontweight='bold')
                    cb2.ax.tick_params(labelsize=TICK_FONT - 1)

                # Draw tile bounding boxes
                n_tiles = len(df_matches)
                rank_cmap = plt.get_cmap('plasma', n_tiles)
                for _, row in df_matches.iterrows():
                    r   = int(row['rank']) - 1
                    clr = rank_cmap(r / max(n_tiles - 1, 1))
                    lw  = 2.8 if r == 0 else 1.8
                    ax_match.add_patch(patches.Rectangle(
                        (row['x0'], row['y0']),
                        row['x1'] - row['x0'], row['y1'] - row['y0'],
                        fill=False, edgecolor=clr, lw=lw))

            ax_match.set_aspect('equal')
            ax_match.axis('off')

            # Shared title
            n_q = len(df_matches) if df_matches is not None else 0
            ax_match.set_title(f'Luminal Tumor Core\u2014SPINDLE Matches (n={n_q})',
                               fontsize=TITLE_FONT, pad=6, fontweight='bold')
            ax_cont.set_title('Luminal Tumor Core\u2014Pathway Score',
                              fontsize=TITLE_FONT, pad=6, fontweight='bold')

            rendered = True
        except Exception as e:
            print(f'  WARNING (Panel F map): {e}')

    if not rendered:
        for ax_tmp in [ax_cont, ax_match]:
            ax_tmp.text(0.5, 0.5, 'Gene-sig map\n[Requires CSV data]',
                        ha='center', va='center', bbox=dict(fc=LIGHT_GRAY),
                        fontsize=ANNOT_FONT)
            ax_tmp.axis('off')


    return ax_cont, ax_match


# ── Panel F: Pathway Score Bar Chart ─────────────────────────────────────────
def plot_panel_f_pathway(ax, project_root: Path):
    """
    Panel F (right) — Horizontal grouped bar chart: tissue background vs.
    SPINDLE Top-10 enrichment score per niche.
    """
    # Fallback
    niches  = ['Invasive Tumour\n& DCIS Core',
                'Myoepithelial\n& Basal Layer',
                'Proliferating\nTumour Cells',
                'Vasculature &\nEndothelial',
                'Macrophage &\nDendritic Cell']
    top10   = [5.579, 0.994, 0.287, 0.191, -0.067]
    bg_vals = [2.231, -0.017, 0.029, 0.011, 0.007]

    df = _read_panel_csv(project_root, 'panel_F_pathway_scores.csv')
    if df is not None:
        try:
            col_lbl = next(c for c in df.columns if 'label' in c.lower())
            col_t10 = next(c for c in df.columns if 'enrich' in c.lower() or 'top' in c.lower())
            col_bg  = next(c for c in df.columns if 'background' in c.lower() or 'bg' in c.lower())
            niches  = df[col_lbl].tolist()
            top10   = df[col_t10].tolist()
            bg_vals = df[col_bg].tolist()
        except Exception as e:
            print(f'  WARNING (Panel F pathway): {e}')

    # Sort ascending by top-10 score
    order   = sorted(range(len(niches)), key=lambda i: top10[i])
    niches  = [niches[i]  for i in order]
    top10   = [top10[i]   for i in order]
    bg_vals = [bg_vals[i] for i in order]

    y_pos = np.arange(len(niches))
    bh    = 0.32

    ax.barh(y_pos - bh/2, bg_vals, height=bh,
            color='#CFD8DC', edgecolor='#808080', lw=0.6, label='Tissue Background')
    bars10 = ax.barh(y_pos + bh/2, top10, height=bh,
                     color=TEAL, edgecolor='none', lw=0.0, label='SPINDLE Top-10')

    for b, v in zip(bars10, top10):
        w  = b.get_width()
        ha = 'left' if w >= 0 else 'right'
        xoff = w + 0.06 if w >= 0 else w - 0.06
        ax.text(xoff, b.get_y() + b.get_height() / 2,
                f'{v:.2f}', va='center', fontweight='bold',
                fontsize=ANNOT_FONT, ha=ha, color=SLATE)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(niches, fontsize=TICK_FONT)
    ax.set_xlabel('Pathway Score (Scanpy sc.score_genes)', fontweight='bold', fontsize=BASE_FONT)
    ax.tick_params(axis='x', labelsize=TICK_FONT)
    ax.axvline(x=0, color='gray', lw=0.8)
    ax.legend(loc='lower right', frameon=True, facecolor='white', fontsize=LEGEND_FONT)
    _clean_axes(ax)


# ── Individual Panel Export ───────────────────────────────────────────────────
def export_individual_panels(project_root: Path):
    """Export each sub-panel as a separate standalone high-resolution file."""
    panels_dir = project_root / 'figures' / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)

    # Panel A
    print('  Panel A ...')
    fig_a, ax_a = plt.subplots(figsize=(8, 5.5))
    plot_panel_a_scalability(ax_a, project_root)
    fig_a.tight_layout()
    _save(fig_a, panels_dir, 'panel_A')

    # Panel B
    print('  Panel B ...')
    fig_b, ax_b = plt.subplots(figsize=(7.5, 5.5))
    plot_panel_b_speedup(ax_b, project_root)
    fig_b.tight_layout()
    _save(fig_b, panels_dir, 'panel_B')

    # Panel C
    print('  Panel C ...')
    fig_c, ax_c = plt.subplots(figsize=(7, 5.5))
    plot_panel_c_accuracy(ax_c, project_root)
    fig_c.tight_layout()
    _save(fig_c, panels_dir, 'panel_C')

    # Panel D
    print('  Panel D ...')
    fig_d, ax_d = plt.subplots(figsize=(7.5, 5.5))
    plot_panel_d_partial(ax_d, project_root)
    fig_d.tight_layout()
    _save(fig_d, panels_dir, 'panel_D')

    # Panel E — three separate files
    print('  Panel E (Xenium) ...')
    fig_xe, ax_xe = plt.subplots(figsize=(7, 7))
    plot_panel_e_xenium(ax_xe, project_root)
    fig_xe.tight_layout()
    _save(fig_xe, panels_dir, 'panel_E_xenium')

    print('  Panel E (Visium) ...')
    fig_vi, ax_vi = plt.subplots(figsize=(7, 7))
    plot_panel_e_visium(ax_vi, project_root)
    fig_vi.tight_layout()
    _save(fig_vi, panels_dir, 'panel_E_visium')

    print('  Panel E (Recall bar chart) ...')
    fig_re, ax_re = plt.subplots(figsize=(7.5, 4))
    plot_panel_e_recall(ax_re, project_root)
    fig_re.tight_layout()
    _save(fig_re, panels_dir, 'panel_E_recall')

    # Panel F — two separate files
    print('  Panel F (spatial map) ...')
    fig_fm = plt.figure(figsize=(14, 7))
    gs_fm  = gridspec.GridSpec(1, 2, figure=fig_fm, wspace=0.08,
                               left=0.01, right=0.93, top=0.92, bottom=0.02)
    plot_panel_f_map(fig_fm, gs_fm[0], gs_fm[1], project_root)
    _save(fig_fm, panels_dir, 'panel_F_map')

    print('  Panel F (pathway bar chart) ...')
    fig_fp, ax_fp = plt.subplots(figsize=(8, 5.5))
    plot_panel_f_pathway(ax_fp, project_root)
    fig_fp.tight_layout()
    _save(fig_fp, panels_dir, 'panel_F_pathway')

    print(f'\n  All individual panels saved to {panels_dir.relative_to(project_root)}')


# ── Composite Backup Figure ───────────────────────────────────────────────────
def build_composite_figure(project_root: Path) -> plt.Figure:
    """
    Build a merged multi-panel composite figure.

    Row 0: A | B
    Row 1: C | D
    Row 2: E-Xenium | E-Visium | E-Recall  (full-width)
    Row 3: F-Map-left | F-Map-right         (full-width, 2/3 of width)
    Row 4: F-Pathway                        (full-width, 1/3 used centrally)

    Rows 3+4 are nested inside a single outer row so F-map and F-pathway
    never overlap.
    """
    fig = plt.figure(figsize=(17, 24), dpi=300)

    # Outer grid: 5 logical rows
    gs = gridspec.GridSpec(
        5, 2,
        height_ratios=[1.0, 1.0, 1.4, 1.4, 1.1],
        hspace=0.45, wspace=0.28,
        left=0.07, right=0.97, top=0.96, bottom=0.03,
    )

    # A
    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a_scalability(ax_a, project_root)

    # B
    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b_speedup(ax_b, project_root)

    # C
    ax_c = fig.add_subplot(gs[1, 0])
    plot_panel_c_accuracy(ax_c, project_root)

    # D
    ax_d = fig.add_subplot(gs[1, 1])
    plot_panel_d_partial(ax_d, project_root)

    # E — three tight sub-columns (small wspace to close the gap)
    sub_gs_e = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[2, :],
        width_ratios=[1.0, 1.0, 1.0], wspace=0.06)
    ax_xe_c = fig.add_subplot(sub_gs_e[0])
    plot_panel_e_xenium(ax_xe_c, project_root)
    ax_vi_c = fig.add_subplot(sub_gs_e[1])
    plot_panel_e_visium(ax_vi_c, project_root)
    ax_re_c = fig.add_subplot(sub_gs_e[2])
    plot_panel_e_recall(ax_re_c, project_root)

    # F-Map — occupies full width of row 3
    sub_gs_fm = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[3, :],
        width_ratios=[1.0, 1.0], wspace=0.08)
    plot_panel_f_map(fig, sub_gs_fm[0], sub_gs_fm[1], project_root)

    # F-Pathway — occupies the centre 60% of row 4 to avoid being too wide
    sub_gs_fp = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=gs[4, :],
        width_ratios=[0.5, 4.0, 0.5, 0.5, 0.5], wspace=0.0)
    ax_fp_c = fig.add_subplot(sub_gs_fp[1])
    plot_panel_f_pathway(ax_fp_c, project_root)

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print('Initializing SPINDLE Figure 1 Generator ...')
    set_publication_style()

    current_dir  = Path(__file__).resolve().parent
    project_root = current_dir.parent

    panel_data_dir = project_root / 'results' / 'panel_data'
    if not panel_data_dir.exists():
        print('\n  WARNING: results/panel_data/ not found.')
        print('  Run:  python scripts/organize_panel_data.py')
        print('  Proceeding with hardcoded fallback values.\n')

    # 1. Export individual panels
    print('\n-- Exporting individual panels ---------------------------------------')
    export_individual_panels(project_root)

    # 2. Composite backup
    print('\n-- Building composite backup figure ---------------------------------')
    fig = build_composite_figure(project_root)
    out_stem = project_root / 'results' / 'fig_main_result'
    for fmt in ('pdf', 'png'):
        p = out_stem.with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved -> {p.relative_to(project_root)}')
    plt.close(fig)

    print(f'\n{"="*70}')
    print('SUCCESS: Figure 1 generation complete.')
    print(f'  Panels    -> {project_root / "figures" / "panels"}/')
    print(f'  Composite -> results/fig_main_result.pdf / .png')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
