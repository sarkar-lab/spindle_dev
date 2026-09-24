#!/usr/bin/env python3
"""
generate_partial_panel_search_figure.py
========================================
Renders the E11 partial-panel-search accuracy-vs-query-length figure from
``results/multiseed_partial_panel_search/<dataset>/seed_*/query_metrics.csv``
(produced by ``benchmarks/multiseed_partial_panel_search.py``'s unit mode,
which copies ``benchmarks/partial_panel_search.py``'s per-seed
``benchmark_interval_metrics.csv`` outputs).

``Length_Bin`` is an exact column in the source CSVs (assigned during query
construction in ``partial_panel_search.py::build_binned_queries`` -- each bin
gets a guaranteed, fixed number of draws, not inferred post-hoc from
``Query_Length``), covering 4 bins: <=6, 7-12, 13-16, >16 genes. Recall@eps=0.1
/ Overlap@eps=0.1 are averaged (mean +/- s.d. across all queries x seeds in
that bin) per dataset, one small-multiple panel per dataset -- mirroring
``generate_budget_sweep_figure.py``'s grid-of-panels convention.

Note: Overlap@ε is denominated by the true-near-set size, which can balloon
for short/low-dimensional gene-interval queries (see
paper/results_tracking.md) -- read it as reflecting the query's own
discriminability as much as search quality.

Produces ``figures/fig_partial_panel_search_by_length.{pdf,png}``.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

NAVY = '#1B365D'
TERRACOTTA = '#C85A32'

BASE_FONT = 12.0
TICK_FONT = 10.0
LEGEND_FONT = 9.5
TITLE_FONT = 12.5

DATASET_DISPLAY = {
    'xenium_human_breast_cancer': 'Breast',
    'xenium_human_kidney_nondiseased': 'Kidney',
    'xenium_human_lung_cancer': 'Lung',
    'xenium_human_lymph_node': 'Lymph Node',
    'xenium_human_lymph_node_5k': 'Lymph Node (5k)',
    'xenium_human_pancreatic_cancer': 'Pancreas',
    'xenium_human_skin_melanoma': 'Skin',
    'xenium_human_brain_cancer': 'Brain',
}
DATASET_ORDER = list(DATASET_DISPLAY.keys())

LENGTH_LABELS = ['<=6', '7-12', '13-16', '>16']

SERIES = [
    ('recall_at_eps_0.1', NAVY, 'o', 'Recall@ε (0.1×)'),
    ('overlap_at_eps_0.1', TERRACOTTA, 's', 'Overlap@ε (0.1×)'),
]

MULTISEED_DIR_NAME = 'multiseed_partial_panel_search'


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


def _load_dataset_queries(project_root: Path, dataset: str) -> pd.DataFrame | None:
    ds_dir = project_root / 'results' / MULTISEED_DIR_NAME / dataset
    if not ds_dir.is_dir():
        return None
    dfs = []
    for seed_dir in sorted(ds_dir.glob('seed_*')):
        csv_path = seed_dir / 'query_metrics.csv'
        if csv_path.exists():
            dfs.append(pd.read_csv(csv_path))
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    return df


def _plot_dataset_panel(ax, df_ds: pd.DataFrame, dataset_label: str):
    x = np.arange(len(LENGTH_LABELS))
    for col, color, marker, label in SERIES:
        means, sds = [], []
        for bin_label in LENGTH_LABELS:
            sub = df_ds[df_ds['Length_Bin'] == bin_label][col]
            means.append(sub.mean() if len(sub) else np.nan)
            sds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
        means = np.array(means)
        sds = np.array(sds)
        ax.plot(x, means, '-', color=color, linewidth=1.2, alpha=0.4, zorder=1)
        ax.errorbar(x, means, yerr=sds, fmt=marker, color=color, markersize=5,
                    label=label, capsize=3, elinewidth=1.0, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(LENGTH_LABELS, fontsize=8.5, rotation=20, ha='right')
    ax.set_ylim(-0.05, 1.08)
    ax.set_title(dataset_label, fontsize=TITLE_FONT)
    _clean_axes(ax)


def plot_partial_panel_search_grid(project_root: Path, out_stem: Path):
    datasets_present = [d for d in DATASET_ORDER
                         if (project_root / 'results' / MULTISEED_DIR_NAME / d).is_dir()]
    if not datasets_present:
        print(f'  WARNING: no dataset directories found under results/{MULTISEED_DIR_NAME}/ -- '
              f'run benchmarks/multiseed_partial_panel_search.py --aggregate first.')
        return

    ncols = 4
    nrows = int(np.ceil(len(datasets_present) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.4 * nrows), squeeze=False)

    for i, ds in enumerate(datasets_present):
        ax = axes[i // ncols][i % ncols]
        df_ds = _load_dataset_queries(project_root, ds)
        _plot_dataset_panel(ax, df_ds, DATASET_DISPLAY[ds])
        if i % ncols == 0:
            ax.set_ylabel('Recall / Overlap')

    for j in range(len(datasets_present), nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03),
                   frameon=False)

    fig.suptitle('Partial-Panel-Search Accuracy vs. Query Length (mean ± s.d., seeds 0-4)',
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
    out_stem = project_root / 'figures' / 'fig_partial_panel_search_by_length'
    plot_partial_panel_search_grid(project_root, out_stem)


if __name__ == '__main__':
    main()
