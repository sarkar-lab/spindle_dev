"""Shared loaders and helpers for the preprint experiments.

Every experiment here reuses the index builds that already exist in
``results/indexes/`` (never rebuilds them):

* seed 73 -- the production build with a fixed 100-tile holdout (E1); files
  are named ``<stem>_*.pkl`` with no seed suffix.
* seeds 0-4 -- the E5/E11 builds with a 10% holdout; ``<stem>_seed<n>_*.pkl``.

Tiles are not stored for held-out queries, but the quadtree tiling is
deterministic, so ``rebuild_tiles`` regenerates the full tile list exactly as
``build_indexes.prepare_to_index`` did; ``tiles[test_idx]`` are the queries
and ``tiles[train_idx]`` match ``data.metadata['tiles']`` (checked on load).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
for p in (PROJECT_ROOT / "src", BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import spindle_dev.preprocessing as preprocessing  # noqa: E402

DATASET_DIR = Path("/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium")
INDEX_DIR = PROJECT_ROOT / "results" / "indexes"
GT_CACHE_DIR = PROJECT_ROOT / "results" / "ground_truth_cache"

# Short key -> file stem, ordered by cell count (the figure order).
DATASETS = {
    "skin_melanoma": "xenium_human_skin_melanoma",
    "kidney_nondiseased": "xenium_human_kidney_nondiseased",
    "breast_cancer": "xenium_human_breast_cancer",
    "lung_cancer": "xenium_human_lung_cancer",
    "pancreatic_cancer": "xenium_human_pancreatic_cancer",
    "lymph_node": "xenium_human_lymph_node",
    "lymph_node_5k": "xenium_human_lymph_node_5k",
    "brain_cancer": "xenium_human_brain_cancer",
}
PRODUCTION_SEED = 73
MULTISEED = [0, 1, 2, 3, 4]


def resolve_stem(name: str) -> str:
    """Accept a short key, a stem, or a path to the .h5ad file."""
    name = Path(name).stem
    return DATASETS.get(name, name)


def index_tag(stem: str, seed: int) -> str:
    return stem if seed == PRODUCTION_SEED else f"{stem}_seed{seed}"


def load_index(stem: str, seed: int) -> dict:
    """The saved index bundle: data, dag_dict, config, dataset_name, ..."""
    with open(INDEX_DIR / f"{index_tag(stem, seed)}_spindle_index.pkl", "rb") as fh:
        return pickle.load(fh)


def load_raw_covs(stem: str, seed: int) -> dict:
    """train_tile_covs / test_tile_covs (lists of dicts with 'cov'), train_idx, test_idx."""
    with open(INDEX_DIR / f"{index_tag(stem, seed)}_raw_covariances.pkl", "rb") as fh:
        return pickle.load(fh)


def load_interval_index(stem: str, seed: int):
    with open(INDEX_DIR / f"{index_tag(stem, seed)}_interval_index.pkl", "rb") as fh:
        return pickle.load(fh)


def raw_cov(entry) -> np.ndarray:
    return entry if not isinstance(entry, dict) else entry.get("cov", entry.get("matrix", entry))


def load_adata(stem: str):
    """The AnnData exactly as build_indexes.load_and_split_data filtered it."""
    import scanpy as sc

    adata = sc.read_h5ad(DATASET_DIR / f"{stem}.h5ad")
    if "Cluster" in adata.obs.columns:
        adata = adata[adata.obs.loc[adata.obs.Cluster != "Unlabeled"].index, :].copy()
    return adata


def rebuild_tiles(adata):
    """Same call as build_indexes.prepare_to_index."""
    return preprocessing.build_quadtree_tiles(adata.obsm["spatial"], max_pts=200, min_side=0.0, max_depth=40)


def check_tiles_match(tiles, data, train_idx) -> None:
    """Fail loudly if the rebuilt tiling differs from the one the index was built on."""
    saved = data.metadata["tiles"]
    if len(saved) != len(train_idx):
        raise ValueError(f"index has {len(saved)} tiles but train_idx has {len(train_idx)}")
    for k in (0, len(saved) // 2, len(saved) - 1):
        a, b = saved[k], tiles[int(train_idx[k])]
        if a.id != b.id or not np.array_equal(np.sort(a.idx), np.sort(b.idx)):
            raise ValueError(f"rebuilt tile {int(train_idx[k])} does not match saved training tile {k}")


def block_distance(q_blocks_log, t_blocks_log, block_runs) -> float:
    """The block-diagonalized log-Euclidean distance of compute_ground_truth."""
    return float(sum(
        np.linalg.norm(q_blocks_log[b] - t_blocks_log[b], ord="fro") / np.sqrt(e - s)
        for b, (s, e) in enumerate(block_runs)
    ))


def rerank_block(candidates_local, q_blocks_log_by_niche, gt_block, data):
    """Stage-2 exact re-rank of local training indices -> [(dist, local_idx)], ascending.

    Uses the per-block training logs cached in the block ground truth, exactly
    like holdout_core.evaluate_against_ground_truth.
    """
    niche_train_cache = gt_block["niche_train_cache"]
    tile_niche = gt_block["tile_niche"]
    out = []
    for loc in candidates_local:
        niche = tile_niche[loc]
        out.append((block_distance(q_blocks_log_by_niche[niche], niche_train_cache[niche][1][loc],
                                   data.block_dict[niche]), loc))
    out.sort(key=lambda x: x[0])
    return out


def bh_fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def cliffs_delta(x, y) -> float:
    """P(X > Y) - P(X < Y), via ranks (O(n log n))."""
    from scipy.stats import rankdata

    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size == 0 or y.size == 0:
        return float("nan")
    r = rankdata(np.concatenate([x, y]))
    u = r[: x.size].sum() - x.size * (x.size + 1) / 2
    return float(2 * u / (x.size * y.size) - 1)
