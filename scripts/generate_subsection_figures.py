#!/usr/bin/env python3
"""
Subsection Figure Generator
============================
Generates the per-subsection supplementary figures referenced in results_section.tex.
These figures are **distinct** from the main Figure 1 panels produced by
generate_figure_1.py:
  - Main Figure 1 provides a concise, high-level overview of each result.
  - Subsection figures provide the additional detail discussed in the text.

All data is loaded from the results/ directory; hardcoded values are used only
as documented fallbacks if a CSV is absent (e.g. on a fresh checkout).

Outputs (written to figures/)
-------------------------------
  fig_scalability.pdf/.png          — scatter+trendline (distinct from panel A bar chart)
  fig_query_speed.pdf/.png          — absolute Spindle vs brute-force latency per dataset
  fig_rank_distribution.pdf/.png    — cumulative recall curve across ranks 1-10
  fig_partial_panel_search.pdf/.png — per-dataset accuracy + speedup (2-panel)
  fig_cross_modal.pdf/.png          — Recall/Overlap@K for k=1,5,10,20, both directions
  fig_gene_sig_luminal.pdf/.png     — Luminal Tumour Core spatial comparison (PNG wrap)
  fig_gene_sig_myoepithelial.pdf/.png — Myoepithelial & Basal Layer (PNG wrap)
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# ── Shared colour palette (matches generate_figure_1.py) ─────────────────────
NAVY       = '#1B365D'
TEAL       = '#008080'
TERRACOTTA = '#C85A32'
SLATE      = '#4A607A'
PURPLE     = '#6B4C9A'
LIGHT_GRAY = '#F0F2F5'
GOLD       = '#D4AF37'

DATASET_DISPLAY = {
    'skin_melanoma':      'Skin',
    'kidney_nondiseased': 'Kidney',
    'breast_cancer':      'Breast',
    'lung_cancer':        'Lung',
    'lymph_node':         'Lymph Node',
    'pancreatic_cancer':  'Pancreas',
}


def _set_style():
    """Apply clean, publication-ready styling."""
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams['font.family']        = 'sans-serif'
    plt.rcParams['font.sans-serif']    = ['Arial', 'Helvetica', 'DejaVu Sans']
    plt.rcParams['axes.edgecolor']     = '#D1D5DB'
    plt.rcParams['axes.linewidth']     = 1.0
    plt.rcParams['axes.titlesize']     = 12.5
    plt.rcParams['axes.titleweight']   = 'bold'
    plt.rcParams['axes.labelsize']     = 11.0
    plt.rcParams['axes.labelweight']   = 'bold'
    plt.rcParams['xtick.labelsize']    = 9.5
    plt.rcParams['ytick.labelsize']    = 9.5
    plt.rcParams['legend.fontsize']    = 9.5


def _save(fig, stem: Path):
    """Save a figure as both PDF and 300-dpi PNG."""
    for fmt in ('pdf', 'png'):
        p = stem.with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'    saved → {p}')
    plt.close(fig)


# ── 1. Scalability — Scatter Plot ────────────────────────────────────────────
def generate_scalability_figure(project_root: Path, figures_dir: Path):
    """
    Scatter plot of Build Time (s) and Index Size (MB) vs cell count.
    Emphasises the growth trajectory (sub-linear scaling), making it distinct
    from the main-figure bar chart (panel A) which shows per-dataset absolute
    values.

    Data: results/holdout_validation/index_scalability_summary.csv
    """
    print('Generating Scalability Figure (scatter plot)…')

    # ── fallback hardcoded values ──
    datasets = ['Skin', 'Kidney', 'Breast', 'Lung', 'Pancreas', 'Lymph Node']
    cells     = [87499, 97560, 159226, 162254, 190965, 377985]
    build_t   = [25.53, 18.62,  51.67,  45.86,  88.90,  71.51]
    idx_size  = [19.41, 18.47,  14.69,  19.98,  30.86,  22.83]

    csv = project_root / 'results' / 'holdout_validation' / 'index_scalability_summary.csv'
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            req = {'Dataset', 'Cells', 'Build Time (s)', 'Index Size (MB)'}
            if req.issubset(df.columns):
                df = df.sort_values('Cells')
                datasets = df['Dataset'].tolist()
                cells    = df['Cells'].tolist()
                build_t  = df['Build Time (s)'].tolist()
                idx_size = df['Index Size (MB)'].tolist()
        except Exception as e:
            print(f'  WARNING (scalability CSV): {e}')

    cells_arr = np.array(cells, dtype=float)
    build_arr = np.array(build_t, dtype=float)
    size_arr  = np.array(idx_size, dtype=float)
    sort_idx  = np.argsort(cells_arr)
    cells_arr, build_arr, size_arr = cells_arr[sort_idx], build_arr[sort_idx], size_arr[sort_idx]
    datasets = [datasets[i] for i in sort_idx]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Scatter + connecting line
    ax1.plot(cells_arr, build_arr, marker='o', color=NAVY,       markersize=8,
             linewidth=2, label='Build Time (s)')
    ax2.plot(cells_arr, size_arr,  marker='s', color=TERRACOTTA, markersize=8,
             linewidth=2, label='Index Size (MB)', linestyle='--')

    # Dataset labels
    for i, ds in enumerate(datasets):
        ax1.annotate(ds, (cells_arr[i], build_arr[i] + 2.5),
                     fontsize=8.5, ha='center', color=NAVY)

    ax1.set_xlabel('Number of Cells in Tissue', fontweight='bold')
    ax1.set_ylabel('Build Time (seconds)', color=NAVY, fontweight='bold')
    ax2.set_ylabel('Index Size on Disk (MB)', color=TERRACOTTA, fontweight='bold')
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x/1000)}k'))
    ax1.tick_params(axis='y', labelcolor=NAVY)
    ax2.tick_params(axis='y', labelcolor=TERRACOTTA)
    ax1.grid(axis='both', linestyle='--', alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True,
               facecolor='white', framealpha=0.9)

    fig.tight_layout()
    _save(fig, figures_dir / 'fig_scalability')


# ── 2. Query Speed — Absolute Latency (Spindle vs Brute-Force) ───────────────
def generate_query_speed_figure(project_root: Path, figures_dir: Path):
    """
    Grouped bar chart: Spindle query time (ms) vs brute-force (ms) per dataset,
    with speedup factor annotated.  Main panel B shows only the speedup lollipop;
    this figure adds the absolute latency context discussed in the subsection text.

    Data: results/holdout_validation/benchmark_summary.csv
    """
    print('Generating Query Speed Figure (absolute latency grouped bar)…')

    # ── fallback ──
    ds_keys   = ['skin_melanoma', 'kidney_nondiseased', 'breast_cancer',
                 'lung_cancer', 'lymph_node', 'pancreatic_cancer']
    spindle_ms    = [149.5, 216.7, 169.6, 334.6, 329.7, 783.9]
    bruteforce_ms = [871.8, 2151.2, 1578.7, 2577.2, 2433.1, 6144.9]
    speedups      = [5.83,  9.93,   9.31,   7.70,   7.38,   7.84]

    csv = project_root / 'results' / 'holdout_validation' / 'benchmark_summary.csv'
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            req = {'Dataset', 'mean_spindle_time_ms', 'mean_brute_force_time_ms', 'mean_speedup'}
            if req.issubset(df.columns):
                df['display'] = df['Dataset'].map(
                    lambda k: DATASET_DISPLAY.get(k, k))
                df = df.sort_values('mean_brute_force_time_ms')
                ds_keys       = df['Dataset'].tolist()
                spindle_ms    = df['mean_spindle_time_ms'].tolist()
                bruteforce_ms = df['mean_brute_force_time_ms'].tolist()
                speedups      = df['mean_speedup'].tolist()
        except Exception as e:
            print(f'  WARNING (query speed CSV): {e}')

    ds_labels = [DATASET_DISPLAY.get(k, k) for k in ds_keys]
    x    = np.arange(len(ds_labels))
    w    = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))

    bars_s = ax.bar(x - w/2, spindle_ms,    w, label='SPINDLE (ms)',      color=TEAL,       alpha=0.88, edgecolor='none')
    bars_b = ax.bar(x + w/2, bruteforce_ms, w, label='Brute-Force (ms)',  color=SLATE,      alpha=0.60, edgecolor='none')

    # Speedup annotations above each pair
    for i, (s, b, spd) in enumerate(zip(spindle_ms, bruteforce_ms, speedups)):
        ax.annotate(f'{spd:.1f}×', xy=(x[i], max(s, b) + 30),
                    ha='center', fontsize=9.5, fontweight='bold', color=TERRACOTTA)

    ax.set_xticks(x)
    ax.set_xticklabels(ds_labels, rotation=25, ha='right')
    ax.set_ylabel('Mean Query Time (ms)', fontweight='bold')
    ax.set_ylim(0, max(bruteforce_ms) * 1.22)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.9)

    fig.tight_layout()
    _save(fig, figures_dir / 'fig_query_speed')


# ── 3. Rank Distribution — Cumulative Recall Curve ───────────────────────────
def generate_rank_distribution_figure(project_root: Path, figures_dir: Path):
    """
    Cumulative recall curve (% of queries whose true match is within rank K).
    Main panel C shows a bar chart of raw rank buckets; this curve communicates
    the same data differently, emphasising the rapid accumulation of recall.

    Data: results/holdout_validation/top1_rank_distribution.csv
    """
    print('Generating Rank Distribution Figure (cumulative recall curve)…')

    # ── fallback (from paper text) ──
    ranks      = [1,    2,    3,    5,    10,   11]
    cumulative = [74.77, 88.32, 91.59, 97.66, 99.53, 100.0]

    csv = project_root / 'results' / 'holdout_validation' / 'top1_rank_distribution.csv'
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            if 'Rank_Category' in df.columns and 'Percentage' in df.columns:
                cat_order = ['1st', '2nd', '3rd', '4-5th', '6-10th', '>10th']
                # Map categories to representative rank values for x-axis
                cat_rank  = {'1st': 1, '2nd': 2, '3rd': 3, '4-5th': 5,
                              '6-10th': 10, '>10th': 11}
                df = df.set_index('Rank_Category').reindex(cat_order).reset_index()
                df['Percentage'] = df['Percentage'].fillna(0)
                df['rank_x']     = df['Rank_Category'].map(cat_rank)
                df['cumulative'] = df['Percentage'].cumsum()
                ranks      = df['rank_x'].tolist()
                cumulative = df['cumulative'].tolist()
        except Exception as e:
            print(f'  WARNING (rank distribution CSV): {e}')

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ranks, cumulative, marker='D', markersize=8, color=PURPLE,
            linewidth=2.5)
    ax.fill_between(ranks, cumulative, color=PURPLE, alpha=0.12)

    for x, y in zip(ranks, cumulative):
        ax.text(x, y + 2.0, f'{y:.1f}%', ha='center', va='bottom',
                fontsize=9.5, fontweight='bold', color=PURPLE)

    ax.set_xlabel('Top-K Rank', fontweight='bold')
    ax.set_ylabel('Cumulative Queries Recalled (%)', fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_xticks(ranks)
    ax.set_xticklabels(['1st', '2nd', '3rd', '5th', '10th', '>10th'])
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.tight_layout()
    _save(fig, figures_dir / 'fig_rank_distribution')


# ── 4. Partial Panel Search — Per-Dataset Accuracy + Speedup (2-panel) ───────
def generate_partial_panel_search_figure(project_root: Path, figures_dir: Path):
    """
    Two-panel subsection figure for the partial-query benchmark:
      Left  — Per-dataset Recall@1 and Overlap@10 (all bins pooled), bar chart.
               Shows which datasets are easiest/hardest to search partially.
      Right — Per-dataset query speedup (SPINDLE partial vs brute-force partial).
               The paper text explicitly mentions speedup in this subsection but
               the original figure did not show it.

    Data:
      results/partial_panel_search/benchmark_summary.csv
        → Dataset, recall_at_1, overlap_at_10, mean_speedup
    """
    print('Generating Partial Panel Search Figure (2-panel: accuracy + speedup)…')

    # ── fallback ──
    ds_keys   = ['skin_melanoma', 'kidney_nondiseased', 'breast_cancer',
                 'lung_cancer', 'lymph_node', 'pancreatic_cancer']
    recall1   = [0.98, 1.00, 0.98, 0.98, 1.00, 0.94]
    overlap10 = [0.756, 0.840, 0.728, 0.940, 0.712, 0.774]
    speedups  = [2.76,  3.99,  4.39,  4.43,  5.38,  5.17]

    csv = project_root / 'results' / 'partial_panel_search' / 'benchmark_summary.csv'
    if csv.exists():
        try:
            df = pd.read_csv(csv)
            req = {'Dataset', 'recall_at_1', 'overlap_at_10', 'mean_speedup'}
            if req.issubset(df.columns):
                df['display'] = df['Dataset'].map(lambda k: DATASET_DISPLAY.get(k, k))
                ds_keys   = df['Dataset'].tolist()
                recall1   = df['recall_at_1'].tolist()
                overlap10 = df['overlap_at_10'].tolist()
                speedups  = df['mean_speedup'].tolist()
        except Exception as e:
            print(f'  WARNING (partial benchmark CSV): {e}')

    ds_labels  = [DATASET_DISPLAY.get(k, k) for k in ds_keys]
    recall_pct  = [r * 100 for r in recall1]
    overlap_pct = [o * 100 for o in overlap10]

    fig, (ax_acc, ax_spd) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ── Left: Accuracy ──────────────────────────────────────────────────────
    x   = np.arange(len(ds_labels))
    w   = 0.38
    b1  = ax_acc.bar(x - w/2, recall_pct,  w, label='Recall@1 (%)',       color=TERRACOTTA, alpha=0.88, edgecolor='none')
    b2  = ax_acc.bar(x + w/2, overlap_pct, w, label='Overlap@10 (%)',     color=NAVY,       alpha=0.85, edgecolor='none')

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax_acc.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                    f'{h:.0f}%', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(ds_labels, rotation=25, ha='right')
    ax_acc.set_ylabel('Retrieval Accuracy (%)', fontweight='bold')
    ax_acc.set_ylim(0, 118)
    ax_acc.grid(axis='y', linestyle='--', alpha=0.4)
    ax_acc.spines['top'].set_visible(False)
    ax_acc.spines['right'].set_visible(False)
    ax_acc.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    # ── Right: Speedup ───────────────────────────────────────────────────────
    paired   = sorted(zip(speedups, ds_labels), reverse=False)
    spd_s, lbl_s = zip(*paired)
    y        = np.arange(len(lbl_s))

    ax_spd.hlines(y, 1, spd_s, colors=TEAL, alpha=0.55, linewidth=2.5)
    ax_spd.scatter(spd_s, y, color=TEAL, s=120, zorder=4)

    # Offset label by a fixed fraction of the x-axis range so it never overlaps the dot
    x_range = max(spd_s) * 1.3
    label_offset = x_range * 0.04          # 4% of full axis width
    for spd, yi in zip(spd_s, y):
        ax_spd.text(spd + label_offset, yi, f'{spd:.2f}×', va='center',
                    fontsize=9.5, fontweight='bold', color=SLATE)

    ax_spd.axvline(x=1, color='crimson', linestyle='--', alpha=0.65, lw=1.5,
                   label='Brute-force baseline (1×)')
    ax_spd.set_yticks(y)
    ax_spd.set_yticklabels(lbl_s, fontsize=10)
    ax_spd.set_xlabel('Speedup vs. Brute-Force Partial Search (×)', fontweight='bold')
    ax_spd.set_xlim(0, x_range)
    ax_spd.grid(axis='x', linestyle='--', alpha=0.35)
    ax_spd.spines['top'].set_visible(False)
    ax_spd.spines['right'].set_visible(False)
    ax_spd.legend(loc='lower right', frameon=True, facecolor='white', fontsize=9)

    fig.tight_layout(pad=2.5)
    _save(fig, figures_dir / 'fig_partial_panel_search')


# ── 5. Cross-Modal — Recall/Overlap@K Across k=1,5,10,20 ────────────────────
def generate_cross_modal_figure(project_root: Path, figures_dir: Path):
    """
    Line chart of Recall@1 and Overlap@K (k=5,10,20) for both query directions
    (Xenium→Visium and Visium→Xenium).  Main panel E shows only Recall@1 as a
    dot plot; this figure adds the K-decay detail mentioned in the paper text.

    Data:
      results/cross_modal_search/x2v_query_metrics.csv  (per-query, x2v direction)
      results/cross_modal_search/v2x_query_metrics.csv  (per-query, v2x direction)
      results/cross_modal_search/benchmark_summary.csv  (aggregate fallback)
    """
    print('Generating Cross-Modal Figure (Recall/Overlap@K line chart)…')

    # ── fallback (from paper text) ──
    k_vals     = [1, 5, 10, 20]
    x2v_scores = [100.0, 100.0, 90.0, 90.0]
    v2x_scores = [84.0,  80.4,  78.2, 76.0]

    def _load_per_query(csv_path):
        """Return mean [recall@1, overlap@5, overlap@10, overlap@20] from per-query CSV."""
        df = pd.read_csv(csv_path)
        # recall@1 is already 0/1 per query; overlaps are fractions
        cols = {}
        for col in df.columns:
            cl = col.lower().replace(' ', '_')
            if 'recall_at_1' in cl or 'recall@1' in cl:
                cols['r1'] = col
            elif 'overlap_at_5' in cl or 'overlap@5' in cl:
                cols['o5'] = col
            elif 'overlap_at_10' in cl or 'overlap@10' in cl:
                cols['o10'] = col
            elif 'overlap_at_20' in cl or 'overlap@20' in cl:
                cols['o20'] = col
        if len(cols) < 4:
            return None
        return [
            df[cols['r1']].mean() * 100,
            df[cols['o5']].mean() * 100,
            df[cols['o10']].mean() * 100,
            df[cols['o20']].mean() * 100,
        ]

    x2v_csv = project_root / 'results' / 'cross_modal_search' / 'x2v_query_metrics.csv'
    v2x_csv = project_root / 'results' / 'cross_modal_search' / 'v2x_query_metrics.csv'

    if x2v_csv.exists():
        try:
            scores = _load_per_query(x2v_csv)
            if scores:
                x2v_scores = scores
        except Exception as e:
            print(f'  WARNING (x2v CSV): {e}')

    if v2x_csv.exists():
        try:
            scores = _load_per_query(v2x_csv)
            if scores:
                v2x_scores = scores
        except Exception as e:
            print(f'  WARNING (v2x CSV): {e}')

    # If per-query files failed, try the aggregate summary
    agg_csv = project_root / 'results' / 'cross_modal_search' / 'benchmark_summary.csv'
    if agg_csv.exists() and (x2v_scores == [100.0, 100.0, 90.0, 90.0]):
        try:
            df = pd.read_csv(agg_csv)
            col_map = {'recall@1': 'r1', 'overlap@5': 'o5',
                       'overlap@10': 'o10', 'overlap@20': 'o20'}
            df.columns = [c.lower().strip() for c in df.columns]
            if 'direction' in df.columns:
                row_x2v = df[df['direction'] == 'x2v'].iloc[0]
                row_v2x = df[df['direction'] == 'v2x'].iloc[0]
                def _row_to_scores(row):
                    return [
                        float(row.get('recall@1', 0)) * 100,
                        float(row.get('overlap@5', 0)) * 100,
                        float(row.get('overlap@10', 0)) * 100,
                        float(row.get('overlap@20', 0)) * 100,
                    ]
                x2v_scores = _row_to_scores(row_x2v)
                v2x_scores = _row_to_scores(row_v2x)
        except Exception as e:
            print(f'  WARNING (cross-modal agg CSV): {e}')

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(k_vals, x2v_scores, marker='o', markersize=9, color=TEAL,
            linewidth=2.5, label='Xenium → Visium')
    ax.plot(k_vals, v2x_scores, marker='s', markersize=9, color=TERRACOTTA,
            linewidth=2.5, label='Visium → Xenium')

    # Annotate each point
    for k, xv, vx in zip(k_vals, x2v_scores, v2x_scores):
        ax.annotate(f'{xv:.1f}%', (k, xv + 2.0), ha='center',
                    color=TEAL, fontweight='bold', fontsize=8.5)
        ax.annotate(f'{vx:.1f}%', (k, vx - 4.5), ha='center',
                    color=TERRACOTTA, fontweight='bold', fontsize=8.5)

    ax.set_xlabel('Top-K Candidates', fontweight='bold')
    ax.set_ylabel('Recall / Overlap (%)', fontweight='bold')
    ax.set_xticks(k_vals)
    ax.set_xticklabels(['Recall@1', 'Overlap@5', 'Overlap@10', 'Overlap@20'])
    ax.set_ylim(0, 112)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='lower left', frameon=True, facecolor='white',
              framealpha=0.9, title='Query Direction')

    fig.tight_layout()
    _save(fig, figures_dir / 'fig_cross_modal')


# ── 6 & 7. Gene Signature Figures — Spatial Comparison PNG Wrappers ──────────
def generate_gene_sig_luminal(project_root: Path, figures_dir: Path):
    """
    Subsection figure: Invasive Tumour & DCIS Core spatial niche discovery.
    Wraps the pre-rendered spatial comparison PNG produced by
    benchmarks/gene_signature_search.py in a publication-ready figure.

    Source: results/gene_signature_search/breast_cancer/
              Luminal_Tumor_Core_spatial_comparison.png
    """
    print('Generating Gene Signature Figure: Luminal Tumour Core…')

    img_path = (project_root / 'results' / 'gene_signature_search' / 'breast_cancer'
                / 'Luminal_Tumor_Core_spatial_comparison.png')

    fig, ax = plt.subplots(figsize=(12, 6.5))
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.axis('off')
    else:
        ax.text(0.5, 0.5,
                'Luminal Tumour Core — Spatial Comparison\n'
                '[Requires: Luminal_Tumor_Core_spatial_comparison.png]',
                ha='center', va='center', fontsize=12,
                bbox=dict(facecolor=LIGHT_GRAY, pad=10))
        ax.axis('off')

    fig.tight_layout()
    _save(fig, figures_dir / 'fig_gene_sig_luminal')


def generate_gene_sig_myoepithelial(project_root: Path, figures_dir: Path):
    """
    Subsection figure: Myoepithelial & Basal Layer spatial niche discovery.
    Wraps the pre-rendered spatial comparison PNG produced by
    benchmarks/gene_signature_search.py in a publication-ready figure.

    Source: results/gene_signature_search/breast_cancer/
              Basal_Myoepithelial_spatial_comparison.png
    """
    print('Generating Gene Signature Figure: Myoepithelial & Basal Layer…')

    img_path = (project_root / 'results' / 'gene_signature_search' / 'breast_cancer'
                / 'Basal_Myoepithelial_spatial_comparison.png')

    fig, ax = plt.subplots(figsize=(12, 6.5))
    if img_path.exists():
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        ax.axis('off')
    else:
        ax.text(0.5, 0.5,
                'Myoepithelial & Basal Layer — Spatial Comparison\n'
                '[Requires: Basal_Myoepithelial_spatial_comparison.png]',
                ha='center', va='center', fontsize=12,
                bbox=dict(facecolor=LIGHT_GRAY, pad=10))
        ax.axis('off')

    fig.tight_layout()
    _save(fig, figures_dir / 'fig_gene_sig_myoepithelial')


# ── Entry point ───────────────────────────────────────────────────────────────
def generate_subsection_figures():
    """Generate all subsection figures and save to figures/."""
    _set_style()

    current_dir  = Path(__file__).resolve().parent
    project_root = current_dir.parent
    figures_dir  = project_root / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    print('\n── Subsection Figure Generator ──────────────────────────────────────')
    print(f'   project_root : {project_root}')
    print(f'   figures_dir  : {figures_dir}')
    print()

    generate_scalability_figure(project_root, figures_dir)
    generate_query_speed_figure(project_root, figures_dir)
    generate_rank_distribution_figure(project_root, figures_dir)
    generate_partial_panel_search_figure(project_root, figures_dir)
    generate_cross_modal_figure(project_root, figures_dir)
    generate_gene_sig_luminal(project_root, figures_dir)
    generate_gene_sig_myoepithelial(project_root, figures_dir)

    print(f'\n{"="*68}')
    print('SUCCESS: All subsection figures saved.')
    print(f'  Output → {figures_dir}/')
    print(f'{"="*68}\n')


if __name__ == '__main__':
    generate_subsection_figures()
