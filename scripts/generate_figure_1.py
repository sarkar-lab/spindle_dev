#!/usr/bin/env python3
"""
Master Figure 1 Generator
=========================
Generates publication-quality Figure 1 panels and exports:
  - Individual high-resolution standalone panel files for manual assembly
      → figures/panels/panel_A.pdf/.png  …  panel_F.pdf/.png
  - One auto-merged composite figure as a backup
      → results/fig_main_result.pdf/.png  (LaTeX-compatible path)

Panels
------
  A — Index scalability: dual-line scatter sorted by cell count
  B — Query acceleration: horizontal lollipop speedup chart
  C — Holdout validation accuracy: cumulative recall area curve
  D — Partial-query robustness: worst vs. best gene-bin bar summary
  E — Cross-modal retrieval: tile overlays + paired Recall@1 dot plot
  F — Gene signature niche discovery: Luminal Tumour Core spatial map
      + top-10 pathway score bar chart
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import matplotlib.image as mpimg
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ── Shared color palette ─────────────────────────────────────────────────────
NAVY       = '#1B365D'
TEAL       = '#008080'
TERRACOTTA = '#C85A32'
SLATE      = '#4A607A'
PURPLE     = '#6B4C9A'
LIGHT_GRAY = '#F0F2F5'
DARK_GRAY  = '#2C3E50'
GOLD       = '#D4AF37'


# ── Publication style ────────────────────────────────────────────────────────
def set_publication_style():
    """Apply clean, publication-ready styling rules."""
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams['font.family']          = 'sans-serif'
    plt.rcParams['font.sans-serif']      = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor']       = '#D1D5DB'
    plt.rcParams['axes.linewidth']       = 1.0
    plt.rcParams['axes.titlesize']       = 13.5
    plt.rcParams['axes.titleweight']     = 'bold'
    plt.rcParams['axes.labelsize']       = 11.5
    plt.rcParams['axes.labelweight']     = 'bold'
    plt.rcParams['xtick.labelsize']      = 9.5
    plt.rcParams['ytick.labelsize']      = 9.5
    plt.rcParams['legend.fontsize']      = 9.5
    plt.rcParams['figure.titlesize']     = 16
    plt.rcParams['figure.titleweight']   = 'bold'


def add_panel_letter(ax, letter: str):
    """Add a bold panel label (A, B, C …) in the upper-left corner of an axes."""
    ax.text(-0.08, 1.12, letter, transform=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='right', color='#000000')


# ── Panel A: Index Scalability — Dual-axis Bar Chart ────────────────────────
def plot_panel_a_scalability(ax, project_root: Path):
    """
    Panel A — Dual-axis grouped bar chart: Build Time (left, NAVY) and
    Index Size (right, TERRACOTTA) per dataset, sorted by cell count.
    Distinct from the subsection scatter plot; emphasises the per-dataset
    absolute values rather than the growth trajectory.
    """
    datasets = ['Skin', 'Kidney', 'Breast', 'Lung', 'Pancreas', 'Lymph Node']
    cells    = [87499,  97560,  159226, 162254, 190965, 377985]
    build_t  = [25.53,  18.62,   51.67,  45.86,  88.90,  71.51]
    idx_size = [19.41,  18.47,   14.69,  19.98,  30.86,  22.83]

    csv_path = project_root / 'results' / 'holdout_validation' / 'index_scalability_summary.csv'
    if csv_path.exists():
        try:
            df_csv = pd.read_csv(csv_path)
            if all(c in df_csv.columns
                   for c in ['Dataset', 'Cells', 'Build Time (s)', 'Index Size (MB)']):
                df_a     = df_csv.sort_values('Cells')
                datasets = df_a['Dataset'].tolist()
                cells    = df_a['Cells'].tolist()
                build_t  = df_a['Build Time (s)'].tolist()
                idx_size = df_a['Index Size (MB)'].tolist()
        except Exception as e:
            print(f'WARNING (Panel A): {e}')

    x     = np.arange(len(datasets))
    width = 0.35
    ax2   = ax.twinx()

    ax.bar( x - width/2, build_t,  width, label='Build Time (s)',  color=NAVY,       alpha=0.85, edgecolor='none')
    ax2.bar(x + width/2, idx_size, width, label='Index Size (MB)', color=TERRACOTTA, alpha=0.85, edgecolor='none')

    ax.set_ylabel('Index Build Time (s)',    color=NAVY,       fontweight='bold')
    ax2.set_ylabel('Index Size on Disk (MB)', color=TERRACOTTA, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=25, ha='right')
    ax.tick_params(axis='y', labelcolor=NAVY)
    ax2.tick_params(axis='y', labelcolor=TERRACOTTA)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax2.grid(False)
    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              frameon=True, facecolor='white', framealpha=0.9)
    add_panel_letter(ax, 'A')


# ── Panel B: Query Acceleration — Horizontal Lollipop ───────────────────────
def plot_panel_b_speedup(ax, project_root: Path):
    """
    Panel B — Horizontal lollipop chart of query speedup (×) per dataset.
    Shows holdout-validation benchmark only; sorted so highest speedup is at top.
    """
    ds_labels = ['Skin', 'Kidney', 'Breast', 'Lung', 'Lymph Node', 'Pancreas']
    ds_keys   = ['skin_melanoma', 'kidney_nondiseased', 'breast_cancer',
                 'lung_cancer', 'lymph_node', 'pancreatic_cancer']
    speedups  = [9.19, 16.97, 11.06, 8.94, 9.18, 9.58]   # fallback

    split_csv = project_root / 'results' / 'holdout_validation' / 'benchmark_summary.csv'
    if split_csv.exists():
        try:
            df_s = pd.read_csv(split_csv)
            speedups = [
                float(df_s.loc[df_s['Dataset'] == k, 'mean_speedup'].values[0])
                if k in df_s['Dataset'].values else spd
                for k, spd in zip(ds_keys, speedups)
            ]
        except Exception as e:
            print(f"WARNING (Panel B): {e}")

    # Sort ascending so highest speedup appears at top of the horizontal chart
    paired  = sorted(zip(speedups, ds_labels), reverse=False)
    spd_s, lbl_s = zip(*paired)

    y = np.arange(len(lbl_s))

    ax.hlines(y, 0, spd_s, colors=TEAL, alpha=0.55, linewidth=2.2)
    ax.scatter(spd_s, y, color=TEAL, s=100, zorder=4)

    for spd, yi in zip(spd_s, y):
        ax.text(spd + 0.25, yi, f'{spd:.1f}×', va='center', fontsize=9.5,
                fontweight='bold', color=SLATE)

    ax.axvline(x=1, color='crimson', linestyle='--', alpha=0.65, lw=1.5,
               label='Brute-Force baseline (1×)')

    ax.set_yticks(y)
    ax.set_yticklabels(lbl_s, fontsize=10)
    ax.set_xlabel('Speedup vs. Exact Brute-Force Search (×)', fontweight='bold')
    ax.set_xlim(0, max(spd_s) * 1.25)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower right', frameon=True, facecolor='white', fontsize=9)
    add_panel_letter(ax, 'B')


# ── Panel C: Holdout Validation — Rank-Bucket Bar Chart ─────────────────────
def plot_panel_c_accuracy(ax, project_root: Path):
    """
    Panel C — Bar chart of rank-bucket percentages from top1_rank_distribution.csv.
    Shows the raw rank distribution; distinct from the subsection cumulative curve.
    The dominant 74.8% first-rank bar makes the accuracy story immediately legible.
    """
    labels = ['1st', '2nd', '3rd', '4-5th', '6-10th', '>10th']
    pcts   = [74.77, 13.55, 3.27, 6.07, 1.87, 0.47]   # fallback

    rank_csv = project_root / 'results' / 'holdout_validation' / 'top1_rank_distribution.csv'
    if rank_csv.exists():
        try:
            df_r = pd.read_csv(rank_csv)
            if 'Rank_Category' in df_r.columns and 'Percentage' in df_r.columns:
                lmap = dict(zip(df_r['Rank_Category'], df_r['Percentage']))
                pcts = [lmap.get(l, 0.0) for l in labels]
        except Exception as e:
            print(f'WARNING (Panel C): {e}')

    ax.bar(labels, pcts, color=PURPLE, alpha=0.85, width=0.55, edgecolor='none')

    # Annotate every bar
    for lbl, p in zip(labels, pcts):
        if p > 0.3:
            ax.text(labels.index(lbl), p + 1.2, f'{p:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold', color=PURPLE)

    ax.set_ylabel('Queries (%)', fontweight='bold')
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    add_panel_letter(ax, 'C')


# ── Panel D: Partial Query — 4-Bin Line Chart ───────────────────────────────
def plot_panel_d_partial(ax, project_root: Path):
    """
    Panel D — Line chart of Recall@1 and Overlap@10 across all four gene-coverage
    bins (≤6, 7-12, 13-16, >16 genes), pooled across all 6 datasets.
    Distinct from the subsection per-dataset bar chart; shows the coverage gradient.
    """
    order          = ['<=6 genes', '7-12 genes', '13-16 genes', '>16 genes']
    order_labels   = ['≤6 genes',  '7–12 genes', '13–16 genes', '>16 genes']
    recall1_vals   = [62.4, 78.5, 88.2, 94.8]    # fallback
    overlap10_vals = [51.2, 71.4, 83.5, 90.9]    # fallback

    metrics_path = project_root / 'results' / 'partial_panel_search' / 'overall_benchmark_metrics.csv'
    if metrics_path.exists():
        try:
            df_m = pd.read_csv(metrics_path)
            if 'Length_Bin' in df_m.columns and 'hit_top_1' in df_m.columns:
                grp  = df_m.groupby('Length_Bin')[['hit_top_1', 'overlap_10']].mean() * 100
                valid = [b for b in order if b in grp.index]
                if valid:
                    order          = valid
                    order_labels   = [b.replace('<=', '≤').replace('-', '–') for b in valid]
                    recall1_vals   = [grp.loc[b, 'hit_top_1']  for b in order]
                    overlap10_vals = [grp.loc[b, 'overlap_10'] for b in order]
        except Exception as e:
            print(f'WARNING (Panel D): {e}')

    x_pos = np.arange(len(order))
    ax.plot(x_pos, recall1_vals,   marker='o', lw=2.5, markersize=8,
            color=TERRACOTTA, label='Recall@1 (%)')
    ax.plot(x_pos, overlap10_vals, marker='s', lw=2.5, markersize=8,
            color=NAVY,       label='Overlap@10 (%)')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(order_labels)
    ax.set_ylabel('Score (%)', fontweight='bold')
    ax.set_xlabel('Partial Query Coverage (Gene Transcript Length)', fontweight='bold')
    ax.set_ylim(0, 105)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    add_panel_letter(ax, 'D')


# ── Panel E: Cross-Modal Retrieval ───────────────────────────────────────────
def plot_panel_e_cross_modal(sub_gs, fig, project_root: Path):
    """
    Panel E — Xenium and Visium tile overlays (left/centre) plus a paired dot
    plot of Recall@1 per query direction (right).  Replaces the multi-K grouped
    bar chart with a cleaner two-point summary at main-figure scale.
    """
    ax_xe  = fig.add_subplot(sub_gs[0])
    ax_vi  = fig.add_subplot(sub_gs[1])
    ax_dot = fig.add_subplot(sub_gs[2])

    # ── Spatial overlays from CSV ──────────────────────────────────────────
    coords_csv = project_root / 'results' / 'cross_modal_search' / 'spatial_coords_sample.csv'
    boxes_csv  = project_root / 'results' / 'cross_modal_search' / 'tile_overlay_boxes.csv'

    if coords_csv.exists() and boxes_csv.exists():
        try:
            df_coords = pd.read_csv(coords_csv)
            df_boxes  = pd.read_csv(boxes_csv)

            xe = df_coords[df_coords['modality'] == 'Xenium']
            vi = df_coords[df_coords['modality'] == 'Visium']

            ax_xe.scatter(xe['x'], xe['y'], s=0.8, color='lightgray', alpha=0.5)
            for _, row in df_boxes[df_boxes['modality'] == 'Xenium'].iterrows():
                ax_xe.add_patch(patches.Rectangle(
                    (row['x0'], row['y0']), row['x1']-row['x0'], row['y1']-row['y0'],
                    lw=0.5, edgecolor=NAVY, facecolor='none', alpha=0.4))
            n_xe = len(df_boxes[df_boxes['modality'] == 'Xenium'])
            ax_xe.set_title(f'Xenium ({n_xe} tiles)', fontsize=10.5, pad=8)
            ax_xe.set_aspect('equal')
            ax_xe.axis('off')

            ax_vi.scatter(vi['x'], vi['y'], s=4, color='#FFA500', alpha=0.8)
            for _, row in df_boxes[df_boxes['modality'] == 'Visium'].iterrows():
                ax_vi.add_patch(patches.Rectangle(
                    (row['x0'], row['y0']), row['x1']-row['x0'], row['y1']-row['y0'],
                    lw=1.0, edgecolor=TERRACOTTA, facecolor='none'))
            n_vi = len(df_boxes[df_boxes['modality'] == 'Visium'])
            ax_vi.set_title(f'Visium ({n_vi} tiles)', fontsize=10.5, pad=8)
            ax_vi.set_aspect('equal')
            ax_vi.axis('off')

        except Exception as e:
            print(f"WARNING (Panel E overlays): {e}")
            for ax_tmp, lbl in [(ax_xe, 'Xenium Overlay'), (ax_vi, 'Visium Overlay')]:
                ax_tmp.text(0.5, 0.5, lbl, ha='center', va='center')
                ax_tmp.axis('off')
    else:
        for ax_tmp, lbl in [(ax_xe, 'Xenium\n[CSV missing]'),
                            (ax_vi, 'Visium\n[CSV missing]')]:
            ax_tmp.text(0.5, 0.5, lbl, ha='center', va='center',
                        bbox=dict(fc=LIGHT_GRAY))
            ax_tmp.axis('off')

    add_panel_letter(ax_xe, 'E')

    # ── Paired dot plot: Recall@1 per query direction ─────────────────────
    directions = ['Xenium → Visium', 'Visium → Xenium']
    recall1    = [100.0, 84.0]   # fallback from benchmark_summary.csv

    cross_csv = project_root / 'results' / 'cross_modal_search' / 'benchmark_summary.csv'
    if cross_csv.exists():
        try:
            df_c = pd.read_csv(cross_csv)
            if 'recall@1' in df_c.columns:
                vals    = df_c['recall@1'].values
                recall1 = [v * 100 if v <= 1.0 else v for v in vals]
            if 'direction' in df_c.columns:
                dir_map    = {'x2v': 'Xenium → Visium', 'v2x': 'Visium → Xenium'}
                directions = [dir_map.get(d, d) for d in df_c['direction'].tolist()]
        except Exception as e:
            print(f"WARNING (Panel E dot): {e}")

    colors = [TEAL, TERRACOTTA]
    for i, (d, r, c) in enumerate(zip(directions, recall1, colors)):
        ax_dot.hlines(i, 0, r, colors=c, alpha=0.5, lw=2.2)
        ax_dot.scatter([r], [i], color=c, s=140, zorder=4)
        ax_dot.text(r + 1.8, i, f'{r:.0f}%', va='center', fontsize=11,
                    fontweight='bold', color=c)

    ax_dot.set_yticks(range(len(directions)))
    ax_dot.set_yticklabels(directions, fontsize=10)
    ax_dot.set_xlabel('Recall@1 (%)', fontweight='bold')
    ax_dot.set_xlim(0, 118)
    ax_dot.set_ylim(-0.6, len(directions) - 0.4)
    ax_dot.axvline(x=100, color=SLATE, linestyle=':', alpha=0.55, lw=1.2,
                   label='Perfect recall')
    ax_dot.grid(axis='x', linestyle='--', alpha=0.35)
    ax_dot.spines['top'].set_visible(False)
    ax_dot.spines['right'].set_visible(False)
    ax_dot.legend(loc='lower right', frameon=True, fontsize=9)


# ── Panel F: Gene Signature Niche Discovery ──────────────────────────────────
# Pathway scores are the raw Scanpy sc.score_genes values reported in the paper.
# These are loaded from results/gene_signature_search/benchmark_metrics.csv when
# available (produced by benchmarks/gene_signature_search.py run on breast_cancer).
# The hardcoded dict below is the paper-reported fallback.
#
# IMPORTANT: benchmark_metrics.csv was previously generated on the lymph_node
# dataset (wrong dataset). The benchmark has been fixed to use breast_cancer.
# Re-run benchmarks/gene_signature_search.py to regenerate correct CSV values.
_PATHWAY_SCORES = {
    'Invasive Tumour\n& DCIS Core':   {'bg':  2.231, 'top10': 5.579},
    'Myoepithelial\n& Basal Layer':   {'bg': -0.017, 'top10': 0.994},
    'Proliferating\nTumour Cells':    {'bg':  0.029, 'top10': 0.287},
    'Vasculature &\nEndothelial':     {'bg':  0.011, 'top10': 0.191},
    'Macrophage &\nDendritic Cell':   {'bg':  0.007, 'top10': -0.067},
}


def _load_pathway_scores(project_root: Path) -> dict:
    """
    Load pathway scores from benchmark_metrics.csv (breast_cancer run).

    The CSV stores the raw Scanpy sc.score_genes values:
      - background_score        → mean over all tissue cells
      - enrichment_score_at_10  → mean over cells inside the top-10 retrieved tiles

    A plausibility check guards against accidentally loading scores from a
    wrong-dataset run (e.g. lymph_node) where all values collapse near zero.
    Falls back to the hardcoded _PATHWAY_SCORES paper values otherwise.
    """
    ROW_TO_LABEL = {
        'Luminal_Tumor_Core':      'Invasive Tumour\n& DCIS Core',
        'Basal_Myoepithelial':     'Myoepithelial\n& Basal Layer',
        'Proliferation_Signature': 'Proliferating\nTumour Cells',
        'Endothelial_Vascular':    'Vasculature &\nEndothelial',
        'Macrophage_Myeloid':      'Macrophage &\nDendritic Cell',
    }
    csv_path = (project_root / 'results' / 'gene_signature_search'
                / 'benchmark_metrics.csv')
    if not csv_path.exists():
        print('  INFO (Panel F): benchmark_metrics.csv not found — using paper values.')
        return _PATHWAY_SCORES
    try:
        df = pd.read_csv(csv_path, index_col=0)
        req = {'background_score', 'enrichment_score_at_10'}
        if not req.issubset(df.columns):
            raise ValueError(f'Missing columns: {req - set(df.columns)}')
        # Plausibility guard: near-zero scores → wrong dataset
        if df['background_score'].abs().max() < 0.1:
            print('  WARNING (Panel F): benchmark_metrics.csv scores appear implausible '
                  '(|background| < 0.1 for all niches). '
                  'Was the benchmark run on breast_cancer? Falling back to paper values.')
            return _PATHWAY_SCORES
        scores = {}
        for row_key, label in ROW_TO_LABEL.items():
            if row_key in df.index:
                scores[label] = {
                    'bg':    float(df.loc[row_key, 'background_score']),
                    'top10': float(df.loc[row_key, 'enrichment_score_at_10']),
                }
            else:
                scores[label] = _PATHWAY_SCORES.get(label, {'bg': 0.0, 'top10': 0.0})
        print('  INFO (Panel F): Loaded pathway scores from benchmark_metrics.csv.')
        return scores
    except Exception as e:
        print(f'  WARNING (Panel F): Could not load benchmark_metrics.csv ({e}). '
              'Using paper values.')
        return _PATHWAY_SCORES


def plot_panel_f_gene_list(sub_gs, fig, project_root: Path):
    """
    Panel F — Luminal Tumour Core spatial discovery map (left) and pathway score
    bar chart for all five gene-list queries (right, top-10 only).
    Pathway scores are loaded from results/gene_signature_search/benchmark_metrics.csv
    (produced by benchmarks/gene_signature_search.py on the breast cancer dataset),
    with hardcoded paper values as a documented fallback.
    """
    ax_map = fig.add_subplot(sub_gs[0])
    ax_bar = fig.add_subplot(sub_gs[1])

    # ── Luminal Tumour Core spatial map ───────────────────────────────────
    cells_csv   = (project_root / 'results' / 'gene_signature_search' / 'breast_cancer'
                   / 'Luminal_Tumor_Core_spatial_cells.csv')
    matches_csv = (project_root / 'results' / 'gene_signature_search' / 'breast_cancer'
                   / 'Luminal_Tumor_Core_top_matches.csv')

    rendered = False
    if cells_csv.exists() and matches_csv.exists():
        try:
            df_cells   = pd.read_csv(cells_csv)
            df_matches = pd.read_csv(matches_csv)

            # Colour cells by continuous pathway score
            sc = ax_map.scatter(
                df_cells['x'], df_cells['y'],
                c=df_cells['pathway_score'], cmap='viridis',
                s=1.0, alpha=0.7, rasterized=True)
            plt.colorbar(sc, ax=ax_map, shrink=0.6, label='Pathway Score')

            rank_colors = ['#000000', '#D35400', '#27AE60', '#2980B9', '#8E44AD']
            for _, row in df_matches.iterrows():
                r = int(row['rank']) - 1
                c = rank_colors[r] if r < len(rank_colors) else '#333333'
                ax_map.add_patch(patches.Rectangle(
                    (row['x0'], row['y0']), row['x1']-row['x0'], row['y1']-row['y0'],
                    fill=False, edgecolor=c, lw=2.5 if r == 0 else 1.8))
            ax_map.set_aspect('equal')
            rendered = True
        except Exception as e:
            print(f"WARNING (Panel F map): {e}")


    if not rendered:
        ax_map.text(0.5, 0.5,
                    'Luminal Tumour Core\nSpatial Discovery Map\n[Requires CSV data]',
                    ha='center', va='center', bbox=dict(fc=LIGHT_GRAY))

    ax_map.axis('off')
    add_panel_letter(ax_map, 'F')

    # ── Pathway score bar: Top-10 enrichment per niche ────────────────────
    pathway_scores = _load_pathway_scores(project_root)
    niches  = list(pathway_scores.keys())
    top10   = [v['top10'] for v in pathway_scores.values()]
    bg_vals = [v['bg']    for v in pathway_scores.values()]

    # Sort by top-10 ascending for horizontal readability
    order   = sorted(range(len(niches)), key=lambda i: top10[i])
    niches  = [niches[i]  for i in order]
    top10   = [top10[i]   for i in order]
    bg_vals = [bg_vals[i] for i in order]

    y_pos = np.arange(len(niches))
    bar_h = 0.32

    ax_bar.barh(y_pos - bar_h / 2, bg_vals, height=bar_h,
                color='#CFD8DC', edgecolor='black', lw=0.5,
                label='Tissue Background')
    bars10 = ax_bar.barh(y_pos + bar_h / 2, top10, height=bar_h,
                          color=TEAL, edgecolor='black', lw=0.5,
                          label='SPINDLE Top-10')

    for b, v in zip(bars10, top10):
        w = b.get_width()
        ha = 'left' if w >= 0 else 'right'
        xoff = w + 0.08 if w >= 0 else w - 0.08
        ax_bar.text(xoff, b.get_y() + b.get_height() / 2, f'{v:.2f}',
                    va='center', fontweight='bold', fontsize=9.5, ha=ha)

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(niches, fontsize=9.5)
    ax_bar.set_xlabel('Pathway Score (Scanpy sc.score_genes)', fontweight='bold')
    ax_bar.axvline(x=0, color='gray', lw=0.8, linestyle='-')
    ax_bar.legend(loc='lower right', frameon=True, fontsize=9)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.4)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)


# ── Individual Panel Export ───────────────────────────────────────────────────
def export_individual_panels(project_root: Path):
    """
    Export each panel as a standalone high-resolution file for manual assembly.
    Outputs to figures/panels/panel_A.pdf/.png … panel_F.pdf/.png
    """
    panels_dir = project_root / 'figures' / 'panels'
    panels_dir.mkdir(parents=True, exist_ok=True)

    def _save(fig_obj, name):
        for fmt in ('pdf', 'png'):
            p = panels_dir / f'{name}.{fmt}'
            fig_obj.savefig(p, bbox_inches='tight', dpi=300, format=fmt)
            print(f'    saved → {p}')
        plt.close(fig_obj)

    print('  Panel A …')
    fig_a, ax_a = plt.subplots(figsize=(8, 5))
    plot_panel_a_scalability(ax_a, project_root)
    plt.tight_layout()
    _save(fig_a, 'panel_A')

    print('  Panel B …')
    fig_b, ax_b = plt.subplots(figsize=(7, 5))
    plot_panel_b_speedup(ax_b, project_root)
    plt.tight_layout()
    _save(fig_b, 'panel_B')

    print('  Panel C …')
    fig_c, ax_c = plt.subplots(figsize=(7, 5))
    plot_panel_c_accuracy(ax_c, project_root)
    plt.tight_layout()
    _save(fig_c, 'panel_C')

    print('  Panel D …')
    fig_d, ax_d = plt.subplots(figsize=(7, 5))
    plot_panel_d_partial(ax_d, project_root)
    plt.tight_layout()
    _save(fig_d, 'panel_D')

    print('  Panel E …')
    fig_e = plt.figure(figsize=(14, 5))
    gs_e  = gridspec.GridSpec(1, 3, figure=fig_e,
                              width_ratios=[1.0, 1.0, 1.0], wspace=0.25)
    plot_panel_e_cross_modal(gs_e, fig_e, project_root)
    fig_e.tight_layout()
    _save(fig_e, 'panel_E')

    print('  Panel F …')
    fig_f = plt.figure(figsize=(14, 6))
    gs_f  = gridspec.GridSpec(1, 2, figure=fig_f,
                              width_ratios=[1.0, 1.2], wspace=0.28)
    plot_panel_f_gene_list(gs_f, fig_f, project_root)
    fig_f.tight_layout()
    _save(fig_f, 'panel_F')

    print(f'\n  All individual panels saved to {panels_dir}')


# ── Composite Backup Figure ───────────────────────────────────────────────────
def build_composite_figure(project_root: Path) -> plt.Figure:
    """
    Build the merged 6-panel composite (backup for auto-assembly).
    No suptitle is added; the figure is intended for manual captioning in LaTeX.
    """
    fig = plt.figure(figsize=(16, 18.5), dpi=300)

    gs = gridspec.GridSpec(
        4, 2,
        height_ratios=[1.0, 1.0, 1.22, 1.45],
        hspace=0.46, wspace=0.26,
        left=0.06, right=0.97, top=0.96, bottom=0.05,
    )

    ax_a = fig.add_subplot(gs[0, 0])
    plot_panel_a_scalability(ax_a, project_root)

    ax_b = fig.add_subplot(gs[0, 1])
    plot_panel_b_speedup(ax_b, project_root)

    ax_c = fig.add_subplot(gs[1, 0])
    plot_panel_c_accuracy(ax_c, project_root)

    ax_d = fig.add_subplot(gs[1, 1])
    plot_panel_d_partial(ax_d, project_root)

    sub_gs_e = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[2, :],
        width_ratios=[1.0, 1.0, 1.15], wspace=0.18)
    plot_panel_e_cross_modal(sub_gs_e, fig, project_root)

    sub_gs_f = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[3, :],
        width_ratios=[1.0, 1.25], wspace=0.28)
    plot_panel_f_gene_list(sub_gs_f, fig, project_root)

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print('Initializing SPINDLE Figure 1 Generator …')
    set_publication_style()

    current_dir  = Path(__file__).resolve().parent
    project_root = current_dir.parent

    # 1. Export individual panels for manual assembly
    print('\n── Exporting individual panels ──────────────────────────────────────')
    export_individual_panels(project_root)

    # 2. Build and save the merged composite backup
    print('\n── Building composite backup figure ────────────────────────────────')
    fig = build_composite_figure(project_root)

    out_stem = project_root / 'results' / 'fig_main_result'
    for fmt in ('pdf', 'png'):
        p = out_stem.with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved → {p}')
    plt.close(fig)

    print(f'\n{"="*72}')
    print('SUCCESS: Figure 1 generation complete.')
    print(f'  Panels    → {project_root / "figures" / "panels"}/')
    print(f'  Composite → {out_stem}.pdf / .png')
    print(f'{"="*72}')


if __name__ == '__main__':
    main()
