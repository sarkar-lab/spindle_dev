#!/usr/bin/env python3
"""
generate_partial_panel_search_by_case_figure.py
================================================
Renders the E11 partial-panel-search dataset-wise-by-case figure from
``results/multiseed_partial_panel_search/<dataset>/seed_*/query_metrics.csv``
(same source as ``generate_partial_panel_search_figure.py``).

Unlike that figure (accuracy vs. query length, one panel per dataset), this
one collapses query length entirely and asks a different question: does
Spindle do better on contiguous gene panels ('Contiguous Random' case) than
on non-contiguous ones ('Non-Contiguous Random' case)? One x-axis position
per dataset, grouped bars for the two cases, mean +/- s.d. across all
queries x length bins x seeds -- two panels (Recall@eps=0.1, Overlap@eps=0.1),
mirroring ``generate_multiseed_holdout_figure.py``'s grouped-bar convention.

Produces ``figures/fig_partial_panel_search_by_case.{pdf,png}``.
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

BASE_FONT = 11.0
TICK_FONT = 9.5
LEGEND_FONT = 9.0
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

CASES = ['Contiguous Random', 'Non-Contiguous Random']
CASE_COLORS = {'Contiguous Random': NAVY, 'Non-Contiguous Random': STEEL_BLUE}
CASE_LABELS = {'Contiguous Random': 'Contiguous', 'Non-Contiguous Random': 'Non-Contiguous'}

METRICS = [
    ('recall_at_eps_0.1', 'Recall@ε (0.1×)'),
    ('overlap_at_eps_0.1', 'Overlap@ε (0.1×)'),
]

MULTISEED_DIR_NAME = 'multiseed_partial_panel_search'


def set_publication_style():
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


def _load_dataset_queries(project_root: Path, dataset: str) -> pd.DataFrame | None:
    ds_dir = project_root / 'results' / MULTISEED_DIR_NAME / dataset
    if not ds_dir.is_dir():
        return None
    dfs = []
    for seed_dir in sorted(ds_dir.glob('seed_*')):
        csv_path = seed_dir / 'query_metrics.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['seed_dir'] = seed_dir.name
            dfs.append(df)
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def _per_seed_case_means(df: pd.DataFrame, metric_col: str) -> dict:
    """Mean of metric_col per (case, seed), so error bars reflect variability
    across seeds -- not variability across the many queries within one seed."""
    out = {}
    for case in CASES:
        case_df = df[df['Case'] == case]
        seed_means = case_df.groupby('seed_dir')[metric_col].mean()
        out[case] = (seed_means.mean(), seed_means.std(ddof=1) if len(seed_means) > 1 else 0.0)
    return out


def plot_partial_panel_search_by_case(project_root: Path, out_stem: Path):
    datasets_present = [d for d in DATASET_ORDER
                         if (project_root / 'results' / MULTISEED_DIR_NAME / d).is_dir()]
    if not datasets_present:
        print(f'  WARNING: no dataset directories found under results/{MULTISEED_DIR_NAME}/ -- '
              f'run benchmarks/multiseed_partial_panel_search.py --aggregate first.')
        return

    labels = [DATASET_DISPLAY[d] for d in datasets_present]
    x = np.arange(len(datasets_present))
    width = 0.35

    dataset_dfs = {ds: _load_dataset_queries(project_root, ds) for ds in datasets_present}

    fig, axes = plt.subplots(1, len(METRICS), figsize=(7.5 * len(METRICS), 5.0))
    if len(METRICS) == 1:
        axes = [axes]

    for ax, (metric_col, metric_label) in zip(axes, METRICS):
        for i, case in enumerate(CASES):
            means, sds = [], []
            for ds in datasets_present:
                m, s = _per_seed_case_means(dataset_dfs[ds], metric_col)[case]
                means.append(m)
                sds.append(s)
            offset = (i - 0.5) * width
            ax.bar(x + offset, means, width, yerr=sds, capsize=3, color=CASE_COLORS[case],
                   label=CASE_LABELS[case], edgecolor='white', linewidth=0.5,
                   error_kw={'linewidth': 1.0, 'ecolor': '#333333'})

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(metric_label)
        ax.set_title(f'{metric_label} by Query Pattern', fontsize=TITLE_FONT)
        ax.legend(loc='upper right', frameon=True, framealpha=0.9)
        ax.grid(axis='x', visible=False)

    fig.suptitle('Partial-Panel-Search: Contiguous vs. Non-Contiguous Gene Panels (mean ± s.d., seeds 0-4)',
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
    out_stem = project_root / 'figures' / 'fig_partial_panel_search_by_case'
    plot_partial_panel_search_by_case(project_root, out_stem)


if __name__ == '__main__':
    main()
