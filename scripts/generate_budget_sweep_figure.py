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
one small-multiple panel per dataset (4 columns x 2 rows, shared axes). The
production budget_multiplier=1.0 is marked with a dotted line and annotated
with that dataset's mean speedup there. Produces one figure per ground-truth
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
# Short panel titles, matching the dataset names used in paper/results_tracking.md.
DATASET_LABELS = {
    'xenium_human_breast_cancer': 'Breast Cancer',
    'xenium_human_kidney_nondiseased': 'Kidney Non-diseased',
    'xenium_human_lung_cancer': 'Lung Cancer',
    'xenium_human_lymph_node': 'Lymph node',
    'xenium_human_lymph_node_5k': 'Lymph node (5k)',
    'xenium_human_pancreatic_cancer': 'Pancreatic Cancer',
    'xenium_human_skin_melanoma': 'Skin Melanoma',
    'xenium_human_brain_cancer': 'Brain Cancer',
}

PRODUCTION_BUDGET = 1.0
X_TICKS = [0.05, 0.1, 0.5, 1, 4, 16]
X_LIM = (0.04, 20)

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
    scale) for one dataset, marking the production budget and its speedup."""
    df_ds = df_ds.sort_values('budget_multiplier')
    x = df_ds['budget_multiplier'].to_numpy()

    for col, color, marker, label in SERIES:
        if col not in df_ds.columns:
            continue
        y = df_ds[col].to_numpy()
        ax.plot(x, y, '-', color=color, linewidth=1.2, alpha=0.3, zorder=1)
        ax.scatter(x, y, color=color, marker=marker, s=16, label=label, zorder=3, edgecolors='none')

    ax.axvline(PRODUCTION_BUDGET, color='#8A8A8A', linestyle=':', linewidth=1.1, zorder=0)
    prod = df_ds[np.isclose(df_ds['budget_multiplier'], PRODUCTION_BUDGET)]
    if not prod.empty:
        ax.text(PRODUCTION_BUDGET * 1.12, 0.04, f"{prod['mean_speedup'].iloc[0]:.0f}× speedup at 1.0",
                fontsize=TICK_FONT - 1, color='#4A4A4A', ha='left', va='bottom')

    ax.set_xscale('log')
    ax.set_xlim(*X_LIM)
    ax.set_xticks(X_TICKS)
    ax.set_xticklabels([f'{t:g}' for t in X_TICKS])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(-0.05, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_title(dataset_label, fontsize=TITLE_FONT)
    _clean_axes(ax)


def plot_budget_sweep_grid(project_root: Path, out_stem: Path, kind: str):
    df = _read_sweep_summary(project_root, kind)
    if df is None:
        return

    datasets = sorted(df['Dataset'].unique(), key=lambda d: DATASET_LABELS.get(d, d))
    n = len(datasets)
    ncols = 4
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.0 * nrows), squeeze=False,
                             sharex=True, sharey=True)

    for i, ds in enumerate(datasets):
        ax = axes[i // ncols][i % ncols]
        _plot_dataset_panel(ax, df[df['Dataset'] == ds], DATASET_LABELS.get(ds, ds))
        ax.tick_params(labelbottom=True)
        if i % ncols == 0:
            ax.set_ylabel('Recall / Overlap @ ε')

    # Hide unused axes if datasets don't fill the grid.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    # Manual layout, top to bottom: title, panels, shared x-label, legend.
    fig.tight_layout(rect=(0, 0.11, 1, 0.94))
    fig.suptitle(f'Recall–Budget Tradeoff ({KIND_TITLES[kind]})', fontsize=14, fontweight='bold', y=0.995)
    fig.text(0.5, 0.075, 'Budget multiplier', ha='center', va='center',
             fontsize=BASE_FONT, fontweight='bold')
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4,
                   frameon=False, markerscale=1.4)

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
