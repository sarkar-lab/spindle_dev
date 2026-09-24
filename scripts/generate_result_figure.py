#!/usr/bin/env python3
"""
generate_result_figure.py
==========================
Generates publication-quality main-result figure panels for SPINDLE.

Reads from ``results/panel_data/`` (produced by ``organize_panel_data.py``).
All panels are exported as individual high-resolution standalone files, and a
composite main-figure is saved to ``figures/fig_main_result.pdf/.png``.

Individual exports
------------------
  figures/panels/panel_A.pdf/.png              — Index scalability
  figures/panels/panel_B.pdf/.png              — Query speedup
  figures/panels/panel_C.pdf/.png              — Partial-query efficiency (dist gap & speedup)
  figures/panels/panel_D.pdf/.png              — Recall@1 by dataset
  figures/panels/panel_E.pdf/.png              — Partial-query robustness
  figures/panels/panel_F_xenium.pdf/.png       — Xenium tile overlay (density gradient)
  figures/panels/panel_G_visium.pdf/.png       — Visium tile overlay (density gradient)
  figures/panels/panel_H_recall.pdf/.png       — Cross-modal recall bar chart
  figures/panels/panel_I_map.pdf/.png          — Gene-sig spatial map (dual)
  figures/panels/panel_J_pathway.pdf/.png      — Pathway score bar chart
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
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
    """
    Read a panel CSV from results/panel_data/, skipping the leading '#'
    comment-header block. Uses `skiprows` (not `comment='#'`) so that data
    values containing a literal '#' (e.g. cluster label "DCIS #1") are never
    truncated.
    """
    path = project_root / 'results' / 'panel_data' / filename
    if not path.exists():
        print(f'  WARNING: {path.relative_to(project_root)} not found')
        return None
    with path.open() as fh:
        n_comment_lines = 0
        for line in fh:
            if line.startswith('#'):
                n_comment_lines += 1
            else:
                break
    return pd.read_csv(path, skiprows=n_comment_lines)


def _no_data(ax, csv_name: str = ''):
    """Render a 'Data not found in CSV' placeholder on *ax* and disable its frame."""
    label = f'Data not found in CSV\n({csv_name})' if csv_name else 'Data not found in CSV'
    ax.text(0.5, 0.5, label,
            transform=ax.transAxes,
            ha='center', va='center',
            fontsize=BASE_FONT, color='#6B6B6B',
            fontstyle='italic',
            bbox=dict(boxstyle='round,pad=0.6', facecolor=LIGHT_GRAY,
                      edgecolor='#AAAAAA', lw=1.0))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


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
    df = _read_panel_csv(project_root, 'panel_A.csv')
    if df is None:
        _no_data(ax, 'panel_A.csv')
        return

    try:
        datasets = df['dataset'].tolist()
        build_t  = df['build_time_s'].tolist()
        idx_size = df['index_size_mb'].tolist()
    except Exception as e:
        print(f'  WARNING (Panel A): {e}')
        _no_data(ax, 'panel_A.csv')
        return

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

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=NAVY,       label='Build Time (s)'),
        Patch(facecolor=TERRACOTTA, label='Index Size (MB)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', frameon=True,
              facecolor='white', fontsize=LEGEND_FONT)


# ── Panel B: Query Acceleration — Bar Chart ─────────────────────────────────
def plot_panel_b_speedup(ax, project_root: Path):
    """
    Panel B — Bar chart of query speedup (x) per dataset.
    Brute-force baseline shown as a dashed line.
    """
    df = _read_panel_csv(project_root, 'panel_B.csv')
    if df is None:
        _no_data(ax, 'panel_B.csv')
        return

    try:
        lbl_s = df['dataset'].tolist()
        spd_s = df['mean_speedup'].tolist()
    except Exception as e:
        print(f'  WARNING (Panel B): {e}')
        _no_data(ax, 'panel_B.csv')
        return

    paired = sorted(zip(lbl_s, spd_s), key=lambda t: t[1])
    lbl_s  = [p[0] for p in paired]
    spd_s  = [p[1] for p in paired]

    x = np.arange(len(lbl_s))
    ax.bar(x, spd_s, width=0.55, color=TEAL, alpha=0.88, edgecolor='none')

    for xi, spd in zip(x, spd_s):
        ax.text(xi, spd + 0.3, f'{spd:.1f}×', ha='center',
                fontsize=ANNOT_FONT, fontweight='bold', color=SLATE)

    ax.axhline(y=1, color='crimson', linestyle='--', alpha=0.70, lw=1.8,
               label='Brute-Force baseline (1×)')

    ax.set_xticks(x)
    ax.set_xticklabels(lbl_s, rotation=20, ha='right', fontsize=TICK_FONT)
    ax.set_ylabel('Speedup vs. Exact Brute-Force Search (×)', fontweight='bold',
                  fontsize=BASE_FONT)
    ax.set_ylim(0, max(spd_s) * 1.25)
    ax.legend(loc='upper left', frameon=True, facecolor='white', fontsize=LEGEND_FONT)
    _clean_axes(ax)


# ── Panel C: Partial-Query Efficiency — Distance Gap & Speedup ──────────────
def plot_panel_c_partial_efficiency(ax, project_root: Path):
    """
    Panel C — Dual-axis chart by gene-count bin: mean distance gap from the
    true exact match (left axis, SLATE, lower is better) and mean speedup vs.
    brute-force (right axis, TEAL — same color as Panel B's speedup bars).
    Companion to Panel E, which shows Recall@1/Overlap@10 for the same bins.
    """
    df = _read_panel_csv(project_root, 'panel_C.csv')
    if df is None:
        _no_data(ax, 'panel_C.csv')
        return

    try:
        bins     = (df['length_bin']
                    .str.replace('<=', '≤', regex=False)
                    .str.replace('-', '–', regex=False)
                    .tolist())
        dist_gap = df['mean_dist_gap'].tolist()
        speedup  = df['mean_speedup'].tolist()
    except Exception as e:
        print(f'  WARNING (Panel C): {e}')
        _no_data(ax, 'panel_C.csv')
        return

    x     = np.arange(len(bins))
    width = 0.35
    ax2   = ax.twinx()

    ax.bar( x - width/2, dist_gap, width, color=SLATE,  alpha=0.88, edgecolor='none')
    ax2.bar(x + width/2, speedup,  width, color=TEAL,    alpha=0.88, edgecolor='none')

    ax.set_ylabel('Mean Distance Gap\nfrom Exact Match (lower is better)', color=SLATE,
                  fontweight='bold', fontsize=BASE_FONT)
    ax2.set_ylabel('Speedup vs. Brute-Force (×)', color=TEAL, fontweight='bold',
                   fontsize=BASE_FONT)
    ax.set_xticks(x)
    ax.set_xticklabels(bins, fontsize=TICK_FONT)
    ax.tick_params(axis='y', labelcolor=SLATE, labelsize=TICK_FONT)
    ax2.tick_params(axis='y', labelcolor=TEAL, labelsize=TICK_FONT)
    ax.set_xlabel('Partial Query Coverage (Gene Transcript Length)',
                  fontweight='bold', fontsize=BASE_FONT)

    ax.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.grid(False)
    ax2.grid(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=SLATE, label='Distance Gap'),
        Patch(facecolor=TEAL,  label='Speedup (×)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True,
              facecolor='white', fontsize=LEGEND_FONT)


# ── Panel D: Recall@1 by Dataset ────────────────────────────────────────────
def plot_panel_d_recall_by_dataset(ax, project_root: Path):
    """
    Panel D — Bar chart of Recall@1 (%) per dataset. Companion to Panel C:
    shows consistency of retrieval accuracy across tissue types.
    """
    df = _read_panel_csv(project_root, 'panel_D.csv')
    if df is None:
        _no_data(ax, 'panel_D.csv')
        return

    try:
        labels = df['dataset'].tolist()
        recall = (df['recall_at_1'] * 100).tolist()
    except Exception as e:
        print(f'  WARNING (Panel D): {e}')
        _no_data(ax, 'panel_D.csv')
        return

    x = np.arange(len(labels))
    ax.bar(x, recall, width=0.55, color=TERRACOTTA, alpha=0.88, edgecolor='none')

    for xi, r in zip(x, recall):
        ax.text(xi, r + 1.2, f'{r:.1f}%', ha='center', va='bottom',
                fontsize=ANNOT_FONT, fontweight='bold', color=TERRACOTTA)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=TICK_FONT)
    ax.set_ylabel('Recall@1 (%)', fontweight='bold', fontsize=BASE_FONT)
    ax.set_ylim(0, 112)
    _clean_axes(ax)


# ── Panel E: Partial Query — Gene-Coverage Line Chart ───────────────────────
def plot_panel_e_partial(ax, project_root: Path):
    """
    Panel E — Line chart of Recall@1 and Overlap@10 across gene-coverage bins.
    """
    df = _read_panel_csv(project_root, 'panel_E.csv')
    if df is None:
        _no_data(ax, 'panel_E.csv')
        return

    try:
        col_bin = next(c for c in df.columns if 'bin' in c.lower() or 'length' in c.lower())
        col_r1  = next(c for c in df.columns if 'recall' in c.lower())
        col_ov  = next(c for c in df.columns if 'overlap' in c.lower())
        order_labels   = (df[col_bin]
                          .str.replace('<=', '≤', regex=False)
                          .str.replace('-', '–', regex=False)
                          .tolist())
        recall1_vals   = df[col_r1].tolist()
        overlap10_vals = df[col_ov].tolist()
    except Exception as e:
        print(f'  WARNING (Panel E): {e}')
        _no_data(ax, 'panel_E.csv')
        return

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


# ── Panel F/G: Cross-modal tile overlays, density-gradient scatter ─────────
def _plot_density_scatter(ax, project_root: Path, coords_csv: str, boxes_csv: str,
                           modality: str, box_color: str, cmap: str, density_label: str,
                           gridsize: int):
    df_pts   = _read_panel_csv(project_root, coords_csv)
    df_boxes = _read_panel_csv(project_root, boxes_csv)

    if df_pts is None:
        ax.text(0.5, 0.5, f'{modality}\n[CSV missing]', ha='center', va='center',
                bbox=dict(fc=LIGHT_GRAY), fontsize=ANNOT_FONT)
        ax.axis('off')
        return

    try:
        x, y = df_pts['x'].values, df_pts['y'].values

        hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1,
                       linewidths=0.0, alpha=0.85)
        cb = plt.colorbar(hb, ax=ax, shrink=0.55, pad=0.03, fraction=0.055,
                          label=density_label)
        cb.ax.tick_params(labelsize=TICK_FONT - 1)
        cb.set_label(density_label, fontsize=BASE_FONT - 2, fontweight='bold')

        if df_boxes is not None:
            m_boxes = df_boxes[df_boxes['modality'] == modality]
            for _, row in m_boxes.iterrows():
                w = row['x1'] - row['x0']
                h = row['y1'] - row['y0']
                ax.add_patch(patches.Rectangle(
                    (row['x0'], row['y0']), w, h,
                    lw=1.3, edgecolor=box_color, facecolor='none', alpha=0.9))
            n_tiles = len(m_boxes)
        else:
            n_tiles = '?'

        ax.set_title(f'{modality} ({n_tiles} tiles)', fontsize=TITLE_FONT,
                     pad=8, fontweight='bold')
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
    except Exception as e:
        print(f'  WARNING (Panel {modality}): {e}')
        ax.text(0.5, 0.5, f'{modality}\n[render error]', ha='center', va='center',
                bbox=dict(fc=LIGHT_GRAY), fontsize=ANNOT_FONT)
        ax.axis('off')


def plot_panel_f_xenium(ax, project_root: Path):
    """Panel F — Xenium cell density (hexbin, navy-family 'Blues') + tile boxes."""
    _plot_density_scatter(ax, project_root, 'panel_F_xenium_coords.csv',
                          'panel_F_G_tile_boxes.csv', 'Xenium', TERRACOTTA,
                          cmap='Blues', density_label='Cell density', gridsize=60)


def plot_panel_g_visium(ax, project_root: Path):
    """Panel G — Visium spot density (hexbin, terracotta-family 'OrRd') + tile boxes."""
    _plot_density_scatter(ax, project_root, 'panel_G_visium_coords.csv',
                          'panel_F_G_tile_boxes.csv', 'Visium', TEAL,
                          cmap='OrRd', density_label='Spot density', gridsize=40)


# ── Panel H: Cross-Modal Recall Bar Chart ───────────────────────────────────
def plot_panel_h_recall(ax, project_root: Path):
    """
    Panel H — Grouped vertical bar chart: Recall@eps=0.1 + Overlap@eps=0.5
    per cross-modal query direction (mean ± s.d. across seeds).
    """
    df = _read_panel_csv(project_root, 'panel_H_recall_metrics.csv')
    if df is None:
        _no_data(ax, 'panel_H_recall_metrics.csv')
        return

    try:
        directions = [d.replace(' to ', ' → ') for d in df['direction_label'].tolist()]
        recall1   = df['recall_at_eps_0.1_pct'].tolist()
        recall1_sd = df['recall_at_eps_0.1_sd_pct'].tolist()
        overlap10 = df['overlap_at_eps_0.5_pct'].tolist()
        overlap10_sd = df['overlap_at_eps_0.5_sd_pct'].tolist()
    except Exception as e:
        print(f'  WARNING (Panel H): {e}')
        _no_data(ax, 'panel_H_recall_metrics.csv')
        return

    n   = len(directions)
    x   = np.arange(n)
    bw  = 0.30

    bars_r  = ax.bar(x - bw/2, recall1,   width=bw, yerr=recall1_sd, capsize=3, color=TEAL,
                     alpha=0.88, edgecolor='none', label='Recall@ε=0.1 (%)')
    bars_o  = ax.bar(x + bw/2, overlap10, width=bw, yerr=overlap10_sd, capsize=3, color=TERRACOTTA,
                     alpha=0.88, edgecolor='none', label='Overlap@ε=0.5 (%)')

    for val, sd, bar in list(zip(recall1, recall1_sd, bars_r)) + list(zip(overlap10, overlap10_sd, bars_o)):
        h = bar.get_height() + sd
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f'{val:.0f}%', ha='center', fontsize=ANNOT_FONT,
                fontweight='bold', color=SLATE)

    ax.axhline(y=100, color=SLATE, linestyle=':', alpha=0.55, lw=1.5, label='Perfect recall')
    ax.set_xticks(x)
    ax.set_xticklabels(directions, fontsize=TICK_FONT)
    ax.set_ylabel('Score (%)', fontweight='bold', fontsize=BASE_FONT)
    ax.set_ylim(0, 120)
    ax.set_xlim(-0.7, n - 0.3)
    ax.legend(loc='lower right', frameon=True, facecolor='white', fontsize=LEGEND_FONT)
    _clean_axes(ax)


# ── Panel I: Gene Signature Spatial Map ──────────────────────────────────────
def plot_panel_i_map(fig, gs_left, gs_right, project_root: Path):
    """
    Panel I (left) — Dual spatial map:
      LEFT  sub-axis: all cells coloured by continuous pathway score (magma)
      RIGHT sub-axis: SPINDLE top-match cells highlighted (no tile-boundary
      boxes — colour alone marks the matched region).
    """
    ax_cont  = fig.add_subplot(gs_left)
    ax_match = fig.add_subplot(gs_right)

    df_cells   = _read_panel_csv(project_root, 'panel_I_spatial_cells.csv')
    df_matches = _read_panel_csv(project_root, 'panel_I_top_matches.csv')

    rendered = False
    if df_cells is not None:
        try:
            x  = df_cells['x'].values
            y  = df_cells['y'].values
            ps = df_cells['pathway_score'].values

            vmin, vmax = np.percentile(ps, 2), np.percentile(ps, 98)

            sc = ax_cont.scatter(x, y, c=ps, cmap='magma',
                                 vmin=vmin, vmax=vmax,
                                 s=0.6, alpha=0.75, rasterized=True,
                                 linewidths=0)
            cb = fig.colorbar(sc, ax=ax_cont, shrink=0.65, pad=0.02)
            cb.set_label('Pathway Score', fontsize=BASE_FONT - 1, fontweight='bold')
            cb.ax.tick_params(labelsize=TICK_FONT - 1)
            ax_cont.set_aspect('equal')
            ax_cont.axis('off')

            ax_match.scatter(x, y, c='#D0D0D0', s=0.4, alpha=0.40,
                             rasterized=True, linewidths=0)

            if df_matches is not None and len(df_matches) > 0:
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

            n_q = len(df_matches) if df_matches is not None else 0
            ax_match.set_title(f'Luminal Tumor Core—SPINDLE Matches (n={n_q})',
                               fontsize=TITLE_FONT, pad=6, fontweight='bold')
            ax_cont.set_title('Luminal Tumor Core—Pathway Score',
                              fontsize=TITLE_FONT, pad=6, fontweight='bold')

            rendered = True
        except Exception as e:
            print(f'  WARNING (Panel I map): {e}')

    if not rendered:
        for ax_tmp in [ax_cont, ax_match]:
            ax_tmp.text(0.5, 0.5, 'Gene-sig map\n[Requires CSV data]',
                        ha='center', va='center', bbox=dict(fc=LIGHT_GRAY),
                        fontsize=ANNOT_FONT)
            ax_tmp.axis('off')

    return ax_cont, ax_match


# ── Panel J: Pathway Score Bar Chart ─────────────────────────────────────────
def plot_panel_j_pathway(ax, project_root: Path):
    """
    Panel J — Horizontal grouped bar chart: tissue background vs.
    SPINDLE Top-10 enrichment score per niche.
    """
    df = _read_panel_csv(project_root, 'panel_J_pathway_scores.csv')
    if df is None:
        _no_data(ax, 'panel_J_pathway_scores.csv')
        return

    try:
        niches  = df['niche_label'].tolist()
        top10   = df['enrichment_score_at_10'].tolist()
        bg_vals = df['background_score'].tolist()
    except Exception as e:
        print(f'  WARNING (Panel J): {e}')
        _no_data(ax, 'panel_J_pathway_scores.csv')
        return

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

    print('  Panel A ...')
    fig_a, ax_a = plt.subplots(figsize=(8, 5.5))
    plot_panel_a_scalability(ax_a, project_root)
    fig_a.tight_layout()
    _save(fig_a, panels_dir, 'panel_A')

    print('  Panel B ...')
    fig_b, ax_b = plt.subplots(figsize=(7.5, 5.5))
    plot_panel_b_speedup(ax_b, project_root)
    fig_b.tight_layout()
    _save(fig_b, panels_dir, 'panel_B')

    print('  Panel C ...')
    fig_c, ax_c = plt.subplots(figsize=(7, 5.5))
    plot_panel_c_partial_efficiency(ax_c, project_root)
    fig_c.tight_layout()
    _save(fig_c, panels_dir, 'panel_C')

    print('  Panel D ...')
    fig_d, ax_d = plt.subplots(figsize=(7.5, 5.5))
    plot_panel_d_recall_by_dataset(ax_d, project_root)
    fig_d.tight_layout()
    _save(fig_d, panels_dir, 'panel_D')

    print('  Panel E ...')
    fig_e, ax_e = plt.subplots(figsize=(7.5, 5.5))
    plot_panel_e_partial(ax_e, project_root)
    fig_e.tight_layout()
    _save(fig_e, panels_dir, 'panel_E')

    print('  Panel F (Xenium) ...')
    fig_f, ax_f = plt.subplots(figsize=(8, 7))
    plot_panel_f_xenium(ax_f, project_root)
    fig_f.tight_layout()
    _save(fig_f, panels_dir, 'panel_F_xenium')

    print('  Panel G (Visium) ...')
    fig_g, ax_g = plt.subplots(figsize=(8, 7))
    plot_panel_g_visium(ax_g, project_root)
    fig_g.tight_layout()
    _save(fig_g, panels_dir, 'panel_G_visium')

    print('  Panel H (Recall bar chart) ...')
    fig_h, ax_h = plt.subplots(figsize=(7.5, 4))
    plot_panel_h_recall(ax_h, project_root)
    fig_h.tight_layout()
    _save(fig_h, panels_dir, 'panel_H_recall')

    print('  Panel I (spatial map) ...')
    fig_i = plt.figure(figsize=(14, 7))
    gs_i  = gridspec.GridSpec(1, 2, figure=fig_i, wspace=0.08,
                               left=0.01, right=0.93, top=0.92, bottom=0.02)
    plot_panel_i_map(fig_i, gs_i[0], gs_i[1], project_root)
    _save(fig_i, panels_dir, 'panel_I_map')

    print('  Panel J (pathway bar chart) ...')
    fig_j, ax_j = plt.subplots(figsize=(8, 5.5))
    plot_panel_j_pathway(ax_j, project_root)
    fig_j.tight_layout()
    _save(fig_j, panels_dir, 'panel_J_pathway')

    print(f'\n  All individual panels saved to {panels_dir.relative_to(project_root)}')


# ── Composite Main Figure ────────────────────────────────────────────────────
def build_composite_figure(project_root: Path) -> plt.Figure:
    """
    Build the merged multi-panel main-result composite.

    Row 0: A | B                  — index scalability, query speedup
    Row 1: C | D                  — cumulative top-K accuracy, recall by dataset
    Row 2: E | J                  — partial-query robustness, pathway enrichment
    Row 3: F | G | H              — cross-modal Xenium/Visium tiles + recall
    Row 4: I (full width)         — gene-signature spatial map (dual), hero panel

    Every row holds 2-3 balanced panels so nothing sits alone/stretched.
    """
    fig = plt.figure(figsize=(17, 24), dpi=300)

    gs = gridspec.GridSpec(
        5, 2,
        height_ratios=[1.0, 1.0, 1.0, 1.4, 1.3],
        hspace=0.45, wspace=0.28,
        left=0.07, right=0.97, top=0.96, bottom=0.03,
    )

    ax_a = fig.add_subplot(gs[0, 0]); plot_panel_a_scalability(ax_a, project_root)
    add_panel_letter(ax_a, 'A')
    ax_b = fig.add_subplot(gs[0, 1]); plot_panel_b_speedup(ax_b, project_root)
    add_panel_letter(ax_b, 'B')

    ax_c = fig.add_subplot(gs[1, 0]); plot_panel_c_partial_efficiency(ax_c, project_root)
    add_panel_letter(ax_c, 'C')
    ax_d = fig.add_subplot(gs[1, 1]); plot_panel_d_recall_by_dataset(ax_d, project_root)
    add_panel_letter(ax_d, 'D')

    ax_e = fig.add_subplot(gs[2, 0]); plot_panel_e_partial(ax_e, project_root)
    add_panel_letter(ax_e, 'E')
    ax_j = fig.add_subplot(gs[2, 1]); plot_panel_j_pathway(ax_j, project_root)
    add_panel_letter(ax_j, 'J')

    # F/G/H — cross-modal triplet, shifted left so H (recall) has breathing room
    sub_gs_fgh = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=gs[3, :],
        width_ratios=[0.85, 0.85, 1.2], wspace=0.3)
    ax_f = fig.add_subplot(sub_gs_fgh[0]); plot_panel_f_xenium(ax_f, project_root)
    add_panel_letter(ax_f, 'F')
    ax_g = fig.add_subplot(sub_gs_fgh[1]); plot_panel_g_visium(ax_g, project_root)
    add_panel_letter(ax_g, 'G')
    ax_h = fig.add_subplot(sub_gs_fgh[2]); plot_panel_h_recall(ax_h, project_root)
    add_panel_letter(ax_h, 'H')

    # I — full-width hero panel (intentional, not a leftover)
    sub_gs_i = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[4, :],
        width_ratios=[1.0, 1.0], wspace=0.08)
    ax_i_left, _ = plot_panel_i_map(fig, sub_gs_i[0], sub_gs_i[1], project_root)
    add_panel_letter(ax_i_left, 'I')

    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    print('Initializing SPINDLE Main-Figure Generator ...')
    set_publication_style()

    current_dir  = Path(__file__).resolve().parent
    project_root = current_dir.parent

    panel_data_dir = project_root / 'results' / 'panel_data'
    if not panel_data_dir.exists():
        print('\n  WARNING: results/panel_data/ not found.')
        print('  Run:  python scripts/organize_panel_data.py')
        print('  Panels with missing CSVs will display a placeholder message.\n')

    print('\n-- Exporting individual panels ---------------------------------------')
    export_individual_panels(project_root)

    print('\n-- Building composite main figure ------------------------------------')
    fig = build_composite_figure(project_root)
    out_stem = project_root / 'figures' / 'fig_main_result'
    for fmt in ('pdf', 'png'):
        p = out_stem.with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved -> {p.relative_to(project_root)}')
    plt.close(fig)

    print(f'\n{"="*70}')
    print('SUCCESS: Main figure generation complete.')
    print(f'  Panels    -> {project_root / "figures" / "panels"}/')
    print(f'  Composite -> figures/fig_main_result.pdf / .png')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
