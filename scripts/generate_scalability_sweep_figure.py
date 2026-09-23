#!/usr/bin/env python3
"""
generate_scalability_sweep_figure.py
=====================================
Renders the log-log synthetic scalability sweep figure (E6) from
``results/scalability_sweep/<dataset>_synthetic_scaling.csv`` (produced by
``benchmarks/scalability_sweep.py --aggregate <dataset>``).

For each swept base dataset, plots build_time_s and index_size_mb against
actual_cells on log-log axes, annotated with the fitted scaling exponent
(from the matching ``<dataset>_scaling_exponents.json``). Produces
``figures/fig_scalability_sweep.{pdf,png}``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

NAVY = '#1B365D'
TERRACOTTA = '#C85A32'

BASE_FONT = 12.0
TICK_FONT = 10.0
LEGEND_FONT = 9.5
TITLE_FONT = 12.5

METRICS = [
    ('build_time_s', 'Build time (s)', NAVY, 'o'),
    ('index_size_mb', 'Index size (MB)', TERRACOTTA, '^'),
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


def _discover_datasets(project_root: Path):
    scal_dir = project_root / 'results' / 'scalability_sweep'
    if not scal_dir.exists():
        return []
    return sorted(p.stem.replace('_synthetic_scaling', '') for p in scal_dir.glob('*_synthetic_scaling.csv'))


def plot_scalability_sweep(project_root: Path, out_stem: Path):
    scal_dir = project_root / 'results' / 'scalability_sweep'
    datasets = _discover_datasets(project_root)
    if not datasets:
        print(f'  WARNING: no *_synthetic_scaling.csv found under {scal_dir.relative_to(project_root)} '
              f'-- run benchmarks/scalability_sweep.py first.')
        return

    fig, axes = plt.subplots(1, len(METRICS), figsize=(6.5 * len(METRICS), 5.2), squeeze=False)
    axes = axes[0]

    colors = sns.color_palette('deep', n_colors=max(len(datasets), 2))

    for col_idx, (metric, ylabel, _, marker) in enumerate(METRICS):
        ax = axes[col_idx]
        for ds_idx, ds in enumerate(datasets):
            df = pd.read_csv(scal_dir / f'{ds}_synthetic_scaling.csv').sort_values('actual_cells')
            exp_path = scal_dir / f'{ds}_scaling_exponents.json'
            exponent_label = ''
            fit = None
            if exp_path.exists():
                with open(exp_path) as f:
                    exponents = json.load(f)
                if metric in exponents:
                    fit = exponents[metric]
                    exponent_label = f" (exp={fit['scaling_exponent']:.2f}, R²={fit['r_squared']:.2f})"

            color = colors[ds_idx]
            # Observed data: markers connected by a thin, faint line.
            ax.plot(df['actual_cells'], df[metric], '-', color=color, linewidth=1.2, alpha=0.3, zorder=1)
            ax.scatter(df['actual_cells'], df[metric], color=color, marker=marker, s=32,
                       label=f'{ds}{exponent_label}', zorder=3, edgecolors='none')

            # Fitted power law y = 10^intercept * cells^exponent -- a straight
            # line on log-log axes, since the fit itself was a log-log linear
            # regression (scipy.stats.linregress on log10(cells) vs. log10(y)).
            if fit is not None:
                cells_grid = np.logspace(
                    np.log10(df['actual_cells'].min()), np.log10(df['actual_cells'].max()), 100
                )
                fitted_y = 10 ** (fit['intercept'] + fit['scaling_exponent'] * np.log10(cells_grid))
                ax.plot(cells_grid, fitted_y, '--', color=color, linewidth=1.5, alpha=0.85, zorder=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Cell count (log scale)')
        ax.set_ylabel(f'{ylabel} (log scale)')
        ax.legend(loc='upper left', frameon=False)
        _clean_axes(ax)

    fig.suptitle('Synthetic Scalability Sweep: Build Time & Index Size vs. Cell Count',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        p = out_stem.with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved -> {p.relative_to(project_root)}')
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent
    set_publication_style()
    out_stem = project_root / 'figures' / 'fig_scalability_sweep'
    plot_scalability_sweep(project_root, out_stem)


if __name__ == '__main__':
    main()
