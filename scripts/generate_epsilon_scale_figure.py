#!/usr/bin/env python3
"""
generate_epsilon_scale_figure.py
=================================
Diagnostic figure: the empirical DISTRIBUTION of pairwise tile-to-tile
distances, per dataset, with the search-budget epsilon (and its 0.1x/0.5x
tolerance fractions) marked as vertical lines.

Distances are the block-diagonalized log-Euclidean distance -- the same
metric epsilon itself is defined in -- sampled WITHIN each niche (pairs of
training tiles that share that niche's own permutation/block layout, so
the distance is unambiguous; a "cross-niche" distance would depend on
which of the two niches' layouts you project through, which the rest of
the codebase never does symmetrically either). Samples from every niche
are pooled into one dataset-wide distribution. Reuses the ALREADY-CACHED
per-block matrix logs in
``results/ground_truth_cache/{name}_ground_truth_block.pkl``
(``niche_train_cache``) computed during the budget sweep -- no new
eigendecompositions needed, just cheap Frobenius norms between cached logs.

Saves figures/fig_epsilon_scale.{png,pdf}.
"""

from pathlib import Path
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

DIST_COLOR = '#3A7CBF'  # single sequential hue for the distribution itself
EPS_COLORS = {
    '0.1x': '#D4AF37',
    '0.5x': '#C85A32',
    '1.0x': '#173F66',
}

BASE_FONT = 11.5
TITLE_FONT = 12.5


def set_publication_style():
    sns.set_theme(style='ticks', context='paper')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'axes.titlesize': TITLE_FONT,
        'axes.titleweight': 'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.dpi': 300,
    })


def sample_within_niche_distances(niche_train_cache: dict, block_dict: dict,
                                   n_samples_per_niche: int = 4000, seed: int = 0) -> np.ndarray:
    """Sample pairwise block-diagonalized distances between tiles that
    share the same niche, pooled across every niche.

    Reuses each niche's already-cached per-tile, per-block matrix logs
    (``niche_train_cache[niche] = (niche_indices, cached_logs)``, from
    ``holdout_validation.compute_ground_truth``) -- computing a sampled
    pair's distance is then just a handful of cheap Frobenius norms
    (``block_dict[niche]`` gives each block's ``(start, end)`` for the
    ``/ sqrt(block_size)`` normalization), no new eigendecompositions.
    """
    rng = np.random.default_rng(seed)
    all_dists = []
    for niche, (niche_indices, cached_logs) in niche_train_cache.items():
        block_runs = block_dict[niche]
        niche_indices = np.asarray(niche_indices)
        m = len(niche_indices)
        if m < 2:
            continue
        n_pairs = min(n_samples_per_niche, m * (m - 1))  # cap effort on huge niches
        idx_a = rng.choice(niche_indices, size=n_pairs, replace=True)
        idx_b = rng.choice(niche_indices, size=n_pairs, replace=True)
        keep = idx_a != idx_b
        idx_a, idx_b = idx_a[keep], idx_b[keep]

        for a, b in zip(idx_a, idx_b):
            logs_a = cached_logs[a]
            logs_b = cached_logs[b]
            d = 0.0
            for (s, e), la, lb in zip(block_runs, logs_a, logs_b):
                d += np.linalg.norm(la - lb, ord='fro') / np.sqrt(e - s)
            all_dists.append(d)

    return np.asarray(all_dists)


def collect_dataset_distributions(project_root: Path) -> dict:
    cache_dir = project_root / 'results' / 'ground_truth_cache'
    indexed_dir = project_root / 'results' / 'holdout_validation_indexed'

    results = {}
    for cache_path in sorted(cache_dir.glob('*_ground_truth_block.pkl')):
        dataset_name = cache_path.name.removesuffix('_ground_truth_block.pkl')
        idx_path = indexed_dir / f'{dataset_name}_spindle_index.pkl'
        if not idx_path.exists():
            print(f'  WARNING: no index pickle for {dataset_name}, skipping.')
            continue

        print(f'  {dataset_name}: sampling within-niche pairwise distances...')
        with open(cache_path, 'rb') as f:
            cached = pickle.load(f)
        with open(idx_path, 'rb') as f:
            saved = pickle.load(f)

        ground_truth_block = cached['ground_truth']
        data = saved['data']
        config = saved['config']

        distances = sample_within_niche_distances(
            ground_truth_block['niche_train_cache'], data.block_dict,
        )
        epsilon = float(np.median(list(config.epsilon_dict.values()))) if config.epsilon_dict else float('nan')

        print(f'    {len(distances)} sampled pairs, epsilon={epsilon:.3g}, '
              f'distance mean={distances.mean():.3g} std={distances.std():.3g}')
        results[dataset_name] = {'distances': distances, 'epsilon': epsilon}

    return results


def plot_distributions(results: dict, project_root: Path):
    labels = sorted(results.keys())
    n = len(labels)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 3.6 * nrows), squeeze=False)

    for i, ds in enumerate(labels):
        ax = axes[i // ncols][i % ncols]
        d = results[ds]
        distances, epsilon = d['distances'], d['epsilon']

        sns.histplot(distances, stat='density', color=DIST_COLOR, alpha=0.35,
                     edgecolor='none', bins=60, ax=ax)
        sns.kdeplot(distances, color=DIST_COLOR, linewidth=1.6, ax=ax)

        for frac_label, color in EPS_COLORS.items():
            frac = float(frac_label.split('x')[0])
            ax.axvline(frac * epsilon, color=color, linewidth=1.4, linestyle='--', zorder=3)

        ax.set_title(ds.replace('xenium_human_', ''), fontsize=TITLE_FONT)
        ax.set_xlabel('block-diagonalized distance', fontsize=BASE_FONT - 1)
        ax.set_ylabel('density', fontsize=BASE_FONT - 1)
        ax.tick_params(labelsize=BASE_FONT - 2)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    handles = [plt.Line2D([0], [0], color=c, linewidth=1.6, linestyle='--', label=f'{lbl} ε')
               for lbl, c in EPS_COLORS.items()]
    fig.legend(handles=handles, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=False)

    fig.suptitle('Within-niche pairwise tile-to-tile distance distribution vs. search-budget ε',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()

    for fmt in ('pdf', 'png'):
        p = (project_root / 'figures' / 'fig_epsilon_scale').with_suffix(f'.{fmt}')
        fig.savefig(str(p), format=fmt, bbox_inches='tight', dpi=300)
        print(f'  saved -> {p.relative_to(project_root)}')
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parent.parent
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    set_publication_style()

    results = collect_dataset_distributions(project_root)
    if not results:
        print('No datasets with both a cached block ground truth and an index pickle found yet.')
        return
    plot_distributions(results, project_root)


if __name__ == '__main__':
    main()
