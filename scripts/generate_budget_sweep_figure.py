#!/usr/bin/env python3
"""
generate_budget_sweep_figure.py
================================
Renders the recall/overlap-vs-budget tradeoff figure from
``results/budget_sweep_holdout/sweep_summary_{kind}.csv`` (``kind`` in
``"block"``/``"whole"``), produced by ``benchmarks/budget_sweep_holdout.py``.

For each dataset, plots Recall@ε(0.1×), Recall@ε(0.5×), Overlap@ε(0.5×), and
Overlap@ε(1.0×) against ``budget_multiplier`` (log-scaled x-axis) as it
sweeps through the fine grid in ``budget_sweep_holdout.DEFAULT_BUDGET_MULTS``,
one small-multiple panel per dataset. Produces one figure per ground-truth
kind: ``figures/fig_budget_sweep_block.{pdf,png}`` (exact block-diagonalized
Frobenius distance ground truth) and ``figures/fig_budget_sweep_whole.{pdf,png}``
(whole-matrix log-Euclidean distance ground truth).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

NAVY = '#1B365D'
STEEL_BLUE = '#3A7CBF'
TERRACOTTA = '#C85A32'
GOLD = '#D4AF37'

BASE_FONT = 12.0
TICK_FONT = 10.0
LEGEND_FONT = 9.5
TITLE_FONT = 12.5

KIND_TITLES = {
    'block': 'Block-Diagonalized LE Ground Truth',
    'whole': 'Whole-Matrix LE Ground Truth',
}

# The four series shown in the tradeoff figure -- Recall@ε at a strict
# (0.1x) and loose (0.5x) tolerance, and Overlap@ε at a loose (0.5x) and the
# loosest (1.0x) tolerance. recall_at_eps_1.0 and overlap_at_eps_0.1 are
# computed (see sweep_summary_{kind}.csv) but intentionally not plotted here.
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


def _read_sweep_summary(project_root: Path, kind: str) -> pd.DataFrame | None:
    path = project_root / 'results' / 'budget_sweep_holdout' / f'sweep_summary_{kind}.csv'
    if not path.exists():
        print(f'  WARNING: {path.relative_to(project_root)} not found -- run '
              f'benchmarks/budget_sweep_holdout.py first.')
        return None
    return pd.read_csv(path)


def _plot_dataset_panel(ax, df_ds: pd.DataFrame, dataset_label: str):
    """Plot the four Recall@ε/Overlap@ε series vs. budget_multiplier (log
    scale) for one dataset. Every swept budget_multiplier gets its own
    x-tick, labeled with both the multiplier value and its mean speedup vs.
    brute force, so both are readable directly off the figure."""
    df_ds = df_ds.sort_values('budget_multiplier')
    x = df_ds['budget_multiplier'].to_numpy()
    speedup = df_ds['mean_speedup'].to_numpy()

    for col, color, marker, label in SERIES:
        if col not in df_ds.columns:
            continue
        y = df_ds[col].to_numpy()
        ax.plot(x, y, '-', color=color, linewidth=1.2, alpha=0.3, zorder=1)
        ax.scatter(x, y, color=color, marker=marker, s=22, label=label, zorder=3, edgecolors='none')

    ax.set_xscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{xi:g} ({s:.1f}×)' for xi, s in zip(x, speedup)],
                        fontsize=6.6, rotation=90)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(dataset_label, fontsize=TITLE_FONT)
    _clean_axes(ax)


def plot_budget_sweep_grid(project_root: Path, out_stem: Path, kind: str):
    df = _read_sweep_summary(project_root, kind)
    if df is None:
        return

    datasets = sorted(df['Dataset'].unique())
    n = len(datasets)
    ncols = 2
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(9.5 * ncols, 4.4 * nrows), squeeze=False)

    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        _plot_dataset_panel(ax, df[df['Dataset'] == ds], ds)
        if i % ncols == 0:
            ax.set_ylabel('Recall / Overlap @ ε')

    # Hide unused axes if datasets don't fill the grid.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.09),
                   frameon=False)

    fig.text(0.5, -0.05, 'Budget multiplier (log scale)',
              ha='center', fontsize=BASE_FONT, fontweight='bold')

    fig.suptitle(f'Recall–Budget Tradeoff Across Budget Multiplier ({KIND_TITLES[kind]})',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        p = out_stem.with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved -> {p.relative_to(project_root)}')
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent
    set_publication_style()
    for kind in ('block', 'whole'):
        out_stem = project_root / 'figures' / f'fig_budget_sweep_{kind}'
        plot_budget_sweep_grid(project_root, out_stem, kind)


if __name__ == '__main__':
    main()
