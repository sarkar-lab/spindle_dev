"""Extract the small figure inputs that live only in pickles / h5ad files.

The figure scripts (scripts/fig*.py) read CSVs only. Everything they need that
is not already a benchmark CSV is written here, once, to results/figure_data/:

  tiles_seed73.csv            every quadtree tile of the seed-73 production tiling
                              (bbox, cell count, status train/heldout/dropped,
                              niche for training tiles), all 8 datasets
                              -> Fig. 2B niche maps, Fig. S1
  cells_sample.csv            up to 20k random cell coordinates per dataset -> Fig. S1
  block_sizes.csv             genes per block, per (dataset, seed, niche) -> Fig. S2
  niche_epsilons.csv          per-niche epsilon of the seed-73 builds -> Fig. S5b
  within_niche_distances.csv  sampled within-niche block LE distances, seed 73 -> Fig. S5b

Loads index pickles (<= 330 MB) and block ground-truth caches (<= 610 MB), so
run it through SLURM:  sbatch slurm_jobs/figures/run_extract_figure_data.sbatch
"""

import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "benchmarks"))

from spindle_dev.preprocessing import build_quadtree_tiles, _low_density_tile_mask  # noqa: E402
import paths  # noqa: E402  (dataset paths: datasets.yaml)

INDEX_DIR = PROJECT_ROOT / "results" / "indexes"
GT_DIR = PROJECT_ROOT / "results" / "ground_truth_cache"
OUT_DIR = PROJECT_ROOT / "results" / "figure_data"

DATASETS = {k: paths.dataset_path(k).stem for k in paths.indexed_keys()}
SEEDS = [0, 1, 2, 3, 4, 73]
PRODUCTION_SEED = 73


def index_path(stem, seed):
    tag = stem if seed == PRODUCTION_SEED else f"{stem}_seed{seed}"
    return INDEX_DIR / f"{tag}_spindle_index.pkl"


def load_coords(stem):
    """Cell coordinates after the index build's 'Unlabeled' filter (build_indexes.load_and_split_data)."""
    a = ad.read_h5ad(paths.dataset_path(stem), backed="r")
    coords = np.asarray(a.obsm["spatial"])[:, :2]
    if "Cluster" in a.obs.columns:
        coords = coords[(a.obs["Cluster"] != "Unlabeled").to_numpy()]
    return coords


def bbox_key(bbox, n):
    return tuple(round(float(v), 4) for v in bbox) + (int(n),)


def tiles_rows(key, stem, data):
    """Rebuild the full tiling (same call as build_indexes.prepare_to_index) and label each tile."""
    coords = load_coords(stem)
    tiles = build_quadtree_tiles(coords, max_pts=200, min_side=0.0, max_depth=40, filter_low_density=False)
    dropped = _low_density_tile_mask(tiles)

    train = defaultdict(list)  # bbox key -> niche labels (degenerate splits can share a bbox)
    for t, lab in zip(data.metadata["tiles"], data.labels):
        train[bbox_key(t.bbox, t.idx.size)].append(int(lab))

    rows, n_train = [], 0
    for t, drop in zip(tiles, dropped):
        k = bbox_key(t.bbox, t.idx.size)
        if drop:
            status, niche = "dropped", -1
        elif train.get(k):
            status, niche = "train", train[k].pop()
            n_train += 1
        else:
            status, niche = "heldout", -1
        rows.append({"dataset": key, "x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2], "y1": t.bbox[3],
                     "n_cells": t.idx.size, "status": status, "niche": niche})
    if n_train != len(data.metadata["tiles"]):
        raise ValueError(f"{key}: matched {n_train} of {len(data.metadata['tiles'])} training tiles")

    rng = np.random.default_rng(0)
    sel = rng.choice(len(coords), size=min(20000, len(coords)), replace=False)
    cells = pd.DataFrame({"dataset": key, "x": coords[sel, 0], "y": coords[sel, 1]})
    return rows, cells


def sample_within_niche_distances(niche_train_cache, block_dict, n_pairs=3000, seed=0):
    """Random within-niche pairs of training tiles; distance = the block LE sum used by the ground truth."""
    rng = np.random.default_rng(seed)
    rows = []
    for niche, (niche_indices, cached_logs) in niche_train_cache.items():
        niche_indices = np.asarray(niche_indices)
        if len(niche_indices) < 2:
            continue
        a = rng.choice(niche_indices, size=n_pairs)
        b = rng.choice(niche_indices, size=n_pairs)
        for i, j in zip(a[a != b], b[a != b]):
            d = sum(np.linalg.norm(la - lb, ord="fro") / np.sqrt(e - s)
                    for (s, e), la, lb in zip(block_dict[niche], cached_logs[i], cached_logs[j]))
            rows.append((int(niche), d))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--only", nargs="*", default=None,
                        help="Subset of: tiles blocks distances (default: all)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts = set(args.only or ["tiles", "blocks", "distances"])

    tile_rows, cell_frames, block_rows, eps_rows, dist_rows = [], [], [], [], []
    for key, stem in DATASETS.items():
        for seed in SEEDS:
            if seed != PRODUCTION_SEED and parts.isdisjoint({"blocks"}):
                continue
            with open(index_path(stem, seed), "rb") as fh:
                bundle = pickle.load(fh)
            data, config = bundle["data"], bundle["config"]
            print(f"{key} seed={seed}", flush=True)

            if "blocks" in parts:
                for niche, runs in data.block_dict.items():
                    for b, (s, e) in enumerate(runs):
                        block_rows.append({"dataset": key, "seed": seed, "niche": int(niche),
                                           "block": b, "size": e - s})
            if seed != PRODUCTION_SEED:
                continue

            if "tiles" in parts:
                rows, cells = tiles_rows(key, stem, data)
                tile_rows.extend(rows)
                cell_frames.append(cells)
                st = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
                print(f"  tiles: {st}", flush=True)

            if "distances" in parts:
                median_eps = float(np.median([float(v) for v in config.epsilon_dict.values()]))
                for niche, eps in config.epsilon_dict.items():
                    eps_rows.append({"dataset": key, "niche": int(niche), "epsilon": float(eps),
                                     "median_epsilon": median_eps,
                                     "n_tiles": int(np.sum(np.asarray(data.labels) == int(niche)))})
                with open(GT_DIR / f"{stem}_ground_truth_block.pkl", "rb") as fh:
                    cache = pickle.load(fh)["ground_truth"]["niche_train_cache"]
                for niche, d in sample_within_niche_distances(cache, data.block_dict):
                    dist_rows.append({"dataset": key, "niche": niche, "distance": d})
                del cache

    if tile_rows:
        pd.DataFrame(tile_rows).to_csv(args.out_dir / "tiles_seed73.csv", index=False)
        pd.concat(cell_frames).to_csv(args.out_dir / "cells_sample.csv", index=False)
    if block_rows:
        pd.DataFrame(block_rows).to_csv(args.out_dir / "block_sizes.csv", index=False)
    if eps_rows:
        pd.DataFrame(eps_rows).to_csv(args.out_dir / "niche_epsilons.csv", index=False)
        pd.DataFrame(dist_rows).to_csv(args.out_dir / "within_niche_distances.csv", index=False)
    print(f"Wrote figure inputs to {args.out_dir}")


if __name__ == "__main__":
    main()
