#!/usr/bin/env python3
"""
generate_multiseed_holdout_figure.py
======================================
Renders the Stage A / E5 multi-seed holdout-validation figure from
``results/multiseed_holdout/summary.csv`` (produced by
``benchmarks/multiseed_holdout.py --aggregate``): per-dataset mean +/- s.d.
across seeds 0-4 for recall_at_eps_0.1, recall_at_eps_0.5,
overlap_at_eps_0.5, overlap_at_eps_1.0 (left panel, grouped bars, shared
0-1 scale), and mean_speedup (right panel, its own scale, since it is not
bounded to [0,1]). Produces ``figures/fig_multiseed_holdout.{pdf,png}``.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NAVY = '#1B365D'
STEEL_BLUE = '#3A7CBF'
GOLD = '#D4AF37'
TERRACOTTA = '#C85A32'

BASE_FONT = 11.0
TICK_FONT = 9.5
LEGEND_FONT = 9.0
TITLE_FONT = 12.5

DATASET_DISPLAY = {
    'xenium_human_skin_melanoma': 'Skin',
    'xenium_human_kidney_nondiseased': 'Kidney',
    'xenium_human_breast_cancer': 'Breast',
    'xenium_human_lung_cancer': 'Lung',
    'xenium_human_lymph_node': 'Lymph Node',
    'xenium_human_lymph_node_5k': 'Lymph Node (5k)',
    'xenium_human_pancreatic_cancer': 'Pancreas',
    'xenium_human_brain_cancer': 'Brain',
}
DATASET_ORDER = list(DATASET_DISPLAY.keys())

ACCURACY_SERIES = [
    ('recall_at_eps_0.1', NAVY, 'Recall@ε (0.1×)'),
    ('recall_at_eps_0.5', STEEL_BLUE, 'Recall@ε (0.5×)'),
    ('overlap_at_eps_0.5', GOLD, 'Overlap@ε (0.5×)'),
    ('overlap_at_eps_1.0', TERRACOTTA, 'Overlap@ε (1.0×)'),
]


def _set_style():
    sns.set_theme(style='whitegrid', context='paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.edgecolor': '#D1D5DB',
        'axes.linewidth': 1.0,
        'axes.titlesize': TITLE_FONT,
        'axes.titleweight': 'bold',
        'axes.labelsize': BASE_FONT,
        'axes.labelweight': 'bold',
        'xtick.labelsize': TICK_FONT,
        'ytick.labelsize': TICK_FONT,
        'legend.fontsize': LEGEND_FONT,
        'savefig.dpi': 300,
    })


def _save(fig, stem: Path, project_root: Path):
    for fmt in ('pdf', 'png'):
        p = stem.with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved -> {p.relative_to(project_root)}')
    plt.close(fig)


def plot_multiseed_holdout(project_root: Path, out_stem: Path):
    csv_path = project_root / 'results' / 'multiseed_holdout' / 'summary.csv'
    if not csv_path.exists():
        print(f'  WARNING: {csv_path.relative_to(project_root)} not found -- run '
              f'benchmarks/multiseed_holdout.py --aggregate first.')
        return

    df = pd.read_csv(csv_path)
    df['order'] = df['dataset'].map({d: i for i, d in enumerate(DATASET_ORDER)})
    df = df.sort_values('order').dropna(subset=['order'])
    labels = [DATASET_DISPLAY[d] for d in df['dataset']]
    x = np.arange(len(df))

    fig, (ax_acc, ax_speed) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={'width_ratios': [2, 1]})

    n_series = len(ACCURACY_SERIES)
    width = 0.8 / n_series
    for i, (col, color, label) in enumerate(ACCURACY_SERIES):
        offset = (i - (n_series - 1) / 2) * width
        means = df[f'mean_{col}'].to_numpy()
        sds = df[f'sd_{col}'].to_numpy()
        ax_acc.bar(x + offset, means, width, yerr=sds, capsize=2, color=color, label=label,
                   edgecolor='white', linewidth=0.5, error_kw={'linewidth': 1.0, 'ecolor': '#333333'})

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels, rotation=30, ha='right')
    ax_acc.set_ylim(0.85, 1.02)
    ax_acc.set_ylabel('Recall / Overlap @ ε')
    ax_acc.set_title('Recall / Overlap @ ε (mean ± s.d., seeds 0-4)')
    ax_acc.legend(loc='lower left', ncol=2, frameon=True, framealpha=0.9)
    ax_acc.grid(axis='x', visible=False)

    speed_means = df['mean_mean_speedup'].to_numpy()
    speed_sds = df['sd_mean_speedup'].to_numpy()
    ax_speed.bar(x, speed_means, 0.6, yerr=speed_sds, capsize=3, color=NAVY,
                edgecolor='white', linewidth=0.5, error_kw={'linewidth': 1.0, 'ecolor': '#333333'})
    ax_speed.set_xticks(x)
    ax_speed.set_xticklabels(labels, rotation=30, ha='right')
    ax_speed.set_ylabel('Speedup vs. brute force (×)')
    ax_speed.set_title('Mean speedup (mean ± s.d., seeds 0-4)')
    ax_speed.grid(axis='x', visible=False)

    fig.tight_layout()

    _save(fig, out_stem, project_root)


def main():
    project_root = Path(__file__).resolve().parent.parent
    _set_style()
    out_stem = project_root / 'figures' / 'fig_multiseed_holdout'
    plot_multiseed_holdout(project_root, out_stem)


if __name__ == '__main__':
    main()
