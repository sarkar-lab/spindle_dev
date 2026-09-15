#!/usr/bin/env python3
"""
generate_budget_sweep_figure.py
================================
Renders the recall/overlap-vs-budget tradeoff figure from
``results/budget_sweep_holdout/sweep_summary.csv``, produced by
``benchmarks/budget_sweep_holdout.py``.

For each dataset, plots Recall@ε(0.1×), Recall@ε(0.5×), Overlap@ε(0.5×), and
Overlap@ε(1.0×) (see ``holdout_validation.EPSILON_TOLERANCE_FRACTIONS``)
against the achieved speedup as ``budget_multiplier`` increases, one
small-multiple panel per dataset. Points where the sweep is already
over-retrieving the niche (``over_retrieving_niche_frac`` above
``OVER_RETRIEVAL_DISPLAY_THRESHOLD``) are rendered as hollow markers on a
dashed segment, visually separating the meaningful sweet-spot region from
the degenerate high-budget regime where Spindle is effectively returning
the whole niche rather than doing a meaningful search.

Also renders a small companion figure showing the absolute number of tiles
Overlap@ε(0.5×)/Overlap@ε(1.0×) are actually computed over per dataset
(``true_near_frac_of_dataset_{frac}`` x total indexed tile count) -- context
for how meaningful the overlap fraction is (a fraction over a handful of
tiles reads very differently from one over hundreds).

Saves ``figures/fig_budget_sweep.pdf``/``.png`` and
``figures/fig_budget_sweep_candidate_counts.pdf``/``.png``.
"""

import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

NAVY = '#1B365D'
STEEL_BLUE = '#3A7CBF'
TERRACOTTA = '#C85A32'
GOLD = '#D4AF37'

BASE_FONT = 12.0
TICK_FONT = 10.0
LEGEND_FONT = 9.5
TITLE_FONT = 12.5

OVER_RETRIEVAL_DISPLAY_THRESHOLD = 0.8

# The four series shown in the main tradeoff figure -- Recall@ε at a strict
# (0.1x) and loose (0.5x) tolerance, and Overlap@ε at a loose (0.5x) and the
# loosest (1.0x) tolerance. recall_at_eps_1.0 and overlap_at_eps_0.1 are
# computed (see sweep_summary.csv) but intentionally not plotted here.
SERIES = [
    ('recall_at_eps_0.1', NAVY, 'o', 'Recall@ε (0.1×)'),
    ('recall_at_eps_0.5', STEEL_BLUE, '^', 'Recall@ε (0.5×)'),
    ('overlap_at_eps_0.5', GOLD, 's', 'Overlap@ε (0.5×)'),
    ('overlap_at_eps_1.0', TERRACOTTA, 'D', 'Overlap@ε (1.0×)'),
]


def set_publication_style():
    sns.set_theme(style='ticks', context='paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.edgecolor': '#4A4A4A',
        'axes.linewidth': 1.1,
        'axes.titlesize': TITLE_FONT,
        'axes.titleweight': 'bold',
        'axes.labelsize': BASE_FONT,
        'axes.labelweight': 'bold',
        'axes.grid': False,
        'grid.alpha': 0.0,
        'xtick.labelsize': TICK_FONT,
        'ytick.labelsize': TICK_FONT,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': LEGEND_FONT,
        'legend.framealpha': 0.9,
        'savefig.dpi': 300,
    })


def _clean_axes(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)


def _read_sweep_summary(project_root: Path) -> pd.DataFrame | None:
    path = project_root / 'results' / 'budget_sweep_holdout' / 'sweep_summary.csv'
    if not path.exists():
        print(f'  WARNING: {path.relative_to(project_root)} not found -- run '
              f'benchmarks/budget_sweep_holdout.py first.')
        return None
    return pd.read_csv(path)


def _plot_dataset_panel(ax, df_ds: pd.DataFrame, dataset_label: str):
    """Plot the four Recall@ε/Overlap@ε series vs. speedup for one dataset."""
    df_ds = df_ds.sort_values('budget_multiplier')

    x = df_ds['mean_speedup'].to_numpy()
    over_retrieving = df_ds.get(
        'over_retrieving_niche_frac',
        pd.Series(np.zeros(len(df_ds)), index=df_ds.index),
    ).to_numpy() > OVER_RETRIEVAL_DISPLAY_THRESHOLD

    for col, color, marker, label in SERIES:
        if col not in df_ds.columns:
            continue
        y = df_ds[col].to_numpy()
        # Solid segment for the meaningful (non-over-retrieving) region,
        # dashed + hollow markers once the sweep starts over-retrieving.
        ax.plot(x, y, '-', color=color, linewidth=1.2, alpha=0.3, zorder=1)
        ax.scatter(x[~over_retrieving], y[~over_retrieving], color=color, marker=marker,
                   s=22, label=label, zorder=3, edgecolors='none')
        if over_retrieving.any():
            ax.scatter(x[over_retrieving], y[over_retrieving], facecolors='none',
                       edgecolors=color, marker=marker, s=22, linewidths=1.0, zorder=3)

    ax.set_ylim(-0.05, 1.08)
    ax.set_title(dataset_label, fontsize=TITLE_FONT)
    _clean_axes(ax)


def plot_budget_sweep_grid(project_root: Path, out_stem: Path):
    df = _read_sweep_summary(project_root)
    if df is None:
        return

    datasets = sorted(df['Dataset'].unique())
    n = len(datasets)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.6 * nrows), squeeze=False)

    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        _plot_dataset_panel(ax, df[df['Dataset'] == ds], ds)
        if i % ncols == 0:
            ax.set_ylabel('Recall / Overlap @ ε')
        if i // ncols == nrows - 1:
            ax.set_xlabel('Speedup over brute force (×)')

    # Hide unused axes if datasets don't fill the grid.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
                   frameon=False)

    fig.suptitle('Recall–Speedup Tradeoff Across Budget Multiplier', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        p = out_stem.with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved -> {p.relative_to(project_root)}')
    plt.close(fig)


def _load_dataset_tile_counts(project_root: Path, datasets: list) -> dict:
    """Total indexed training-tile count per dataset (len(data.labels)),
    read from the saved index pickles -- the denominator overlap@ε is
    ultimately a fraction of (via true_near_frac_of_dataset_{frac})."""
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    base_indexed_dir = project_root / 'results' / 'holdout_validation_indexed'
    counts = {}
    for ds in datasets:
        idx_path = base_indexed_dir / f'{ds}_spindle_index.pkl'
        if not idx_path.exists():
            print(f'  WARNING: {idx_path.name} not found, skipping tile count for {ds}')
            continue
        with open(idx_path, 'rb') as f:
            saved_data = pickle.load(f)
        counts[ds] = len(saved_data['data'].labels)
    return counts


def plot_candidate_count_panel(project_root: Path, out_stem: Path):
    """Small companion figure: absolute number of tiles Overlap@ε(0.5x) and
    Overlap@ε(1.0x) are computed over, per dataset (true_near_frac_of_dataset
    x total indexed tile count -- constant across budget_multiplier, so the
    mean across the sweep is used)."""
    df = _read_sweep_summary(project_root)
    if df is None:
        return

    datasets = sorted(df['Dataset'].unique())
    tile_counts = _load_dataset_tile_counts(project_root, datasets)
    if not tile_counts:
        print('  WARNING: no tile counts available, skipping candidate-count figure')
        return

    rows = []
    for ds in datasets:
        if ds not in tile_counts:
            continue
        g = df[df['Dataset'] == ds]
        total = tile_counts[ds]
        for frac, label in [(0.5, 'Overlap@ε (0.5×)'), (1.0, 'Overlap@ε (1.0×)')]:
            col = f'true_near_frac_of_dataset_{frac}'
            if col not in g.columns:
                continue
            frac_val = g[col].mean()
            rows.append({'dataset': ds, 'tolerance': label, 'n_candidates': frac_val * total,
                         'total_tiles': total})

    count_df = pd.DataFrame(rows)
    if count_df.empty:
        print('  WARNING: no true_near_frac_of_dataset columns found, skipping candidate-count figure')
        return

    labels = sorted(count_df['dataset'].unique())
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(1.3 * len(labels) + 2, 4.5))
    for i, (tol, color) in enumerate([('Overlap@ε (0.5×)', GOLD), ('Overlap@ε (1.0×)', TERRACOTTA)]):
        sub = count_df[count_df['tolerance'] == tol].set_index('dataset').reindex(labels)
        offset = (i - 0.5) * width
        ax.bar(x + offset, sub['n_candidates'], width=width, color=color, alpha=0.88,
               edgecolor='none', label=tol)

    ax.set_yscale('linear')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right')
    ax.set_ylabel('# tiles within ε of true best\n(linear scale, out of full indexed dataset)')
    ax.set_title('How many tiles Overlap@ε is actually computed over', fontsize=TITLE_FONT)
    ax.legend(frameon=False, fontsize=LEGEND_FONT)
    _clean_axes(ax)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        p = Path(str(out_stem) + '_candidate_counts').with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved -> {p.relative_to(project_root)}')
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent
    set_publication_style()
    out_stem = project_root / 'figures' / 'fig_budget_sweep'
    plot_budget_sweep_grid(project_root, out_stem)
    plot_candidate_count_panel(project_root, out_stem)


if __name__ == '__main__':
    main()
