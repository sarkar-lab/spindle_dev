"""Synthetic scalability sweep (E6): index build-time/size across 2-3 orders
of magnitude in cell count.

The manuscript's "sub-linear growth" scalability claim was based on real
datasets spanning less than a 10x range in cell count (87,499-377,985
cells). This sweeps ONE real base dataset up to synthetic cell counts of
[1e3, 5e3, 2e4, 5e4, 1e5, 5e5, 1e6] and re-runs the real index-build pipeline
(the same ``prepare_to_index`` -> ``run_index`` -> ``configure_and_build_dag``
steps ``index_datasets.py`` uses) at each size, so build_time_s/index_size_mb
are measured on the actual indexing code path, not simulated.

Synthetic generation, applied to the base dataset's AnnData (after the same
'Cluster' != 'Unlabeled' filter ``index_datasets.py`` applies):
  - target_cells <= real cell count: uniform subsampling without replacement
    (``synthesis_method=subsample``).
  - target_cells > real cell count: bootstrap resampling WITH replacement up
    to target_cells, then i.i.d. Gaussian jitter added to each resampled
    cell's spatial coordinates (std = 1% of that dataset's coordinate range,
    per axis) so replicated cells don't collapse onto identical quadtree
    tile positions (``synthesis_method=bootstrap_replicate_jitter``). Gene
    expression values are NOT perturbed -- only spatial position, since tile
    covariances are computed from expression across the cells *within* a
    tile, and jittering position is what prevents duplicate cells from
    landing in the same tile.

No train/test holdout split is performed here -- this experiment only
measures build cost, not search accuracy, so every synthetic cell is
indexed.

Unit mode (default): one (base dataset, target_cells, seed) combination per
invocation -- meant to be run once per SLURM job (see
``slurm_jobs/run_scalability_sweep.sbatch``). Writes one row to
``results/scalability_sweep/<dataset>/<dataset>_cells<target_cells>_seed<seed>.csv``
(a per-unit file, not a shared append target, so parallel SLURM jobs for the
same dataset never race on the same CSV).

Aggregate mode (``--aggregate``): merges all per-unit CSVs for one dataset
into ``results/scalability_sweep/<dataset>_synthetic_scaling.csv``, fits a
log-log linear regression (``scipy.stats.linregress``) of build_time_s and
index_size_mb against cell count, and writes the fitted exponents to
``results/scalability_sweep/<dataset>_scaling_exponents.json``.
"""

import argparse
import gc
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib
matplotlib.use('Agg')

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import spindle_dev.typing as typing  # type: ignore
from index_datasets import prepare_to_index, run_index, configure_and_build_dag  # type: ignore
from run_logging import RunLogger  # type: ignore

SCALABILITY_DIR = project_root / "results" / "scalability_sweep"
RUN_LOG_DIR = project_root / "results" / "run_logs"

DEFAULT_TARGET_CELLS = [1_000, 5_000, 20_000, 50_000, 100_000, 500_000, 1_000_000]
SPATIAL_JITTER_FRACTION = 0.01


def load_base_adata(dataset_path: Path):
    adata = sc.read_h5ad(dataset_path)
    if 'Cluster' in adata.obs.columns:
        adata = adata[adata.obs.loc[adata.obs.Cluster != "Unlabeled"].index, :].copy()
    return adata


def make_synthetic_adata(base_adata, target_cells: int, seed: int):
    """Return (synthetic_adata, synthesis_method) at exactly target_cells cells."""
    rng = np.random.default_rng(seed)
    n_obs = base_adata.n_obs

    if target_cells <= n_obs:
        idx = rng.choice(n_obs, size=target_cells, replace=False)
        synthetic = base_adata[idx].copy()
        method = "subsample"
    else:
        idx = rng.choice(n_obs, size=target_cells, replace=True)
        synthetic = base_adata[idx].copy()
        coords = synthetic.obsm['spatial'].astype(float)
        coord_range = coords.max(axis=0) - coords.min(axis=0)
        jitter_std = SPATIAL_JITTER_FRACTION * coord_range
        coords = coords + rng.normal(scale=jitter_std, size=coords.shape)
        synthetic.obsm['spatial'] = coords
        method = "bootstrap_replicate_jitter"

    synthetic.obs_names_make_unique()
    return synthetic, method


def run_unit(dataset_path: Path, target_cells: int, seed: int) -> None:
    stem = dataset_path.stem
    unit_name = f"{stem}_cells{target_cells}_seed{seed}"

    with RunLogger(dataset_name=unit_name, stage="scalability_sweep", out_dir=RUN_LOG_DIR,
                    seed=seed, target_cells=target_cells):
        print(f"Loading base dataset {dataset_path}...")
        base_adata = load_base_adata(dataset_path)
        real_n = base_adata.n_obs
        print(f"Base dataset has {real_n} cells; synthesizing {target_cells} cells (seed={seed})...")

        synthetic_adata, synthesis_method = make_synthetic_adata(base_adata, target_cells, seed)
        actual_cells = synthetic_adata.n_obs
        del base_adata
        gc.collect()

        print(f"Preparing {actual_cells} synthetic cells for indexing (method={synthesis_method})...")
        tiles, tile_covs, genes_work = prepare_to_index(synthetic_adata)
        num_tiles = len(tiles)

        print(f"Running index (build_time_s measured from here, matching index_datasets.py)...")
        t0 = time.perf_counter()
        data, out_dict = run_index(tiles, tile_covs, genes_work, synthetic_adata,
                                     resolution=0.2, min_final_size=15, max_niche_size=1000)
        dag_dict, config = configure_and_build_dag(data)
        build_time_s = time.perf_counter() - t0

        index_bundle = typing.DatasetIndex(
            dag_dict=dag_dict,
            metadata=data.metadata,
            latent=data.latent,
            labels=data.labels,
            pca_model=getattr(data, "pca_model", None),
        )
        index_size_mb = round(len(pickle.dumps(index_bundle, protocol=pickle.HIGHEST_PROTOCOL)) / (1024 * 1024), 2)

        print(f"[{unit_name}] build_time_s={build_time_s:.2f}, index_size_mb={index_size_mb}, "
              f"num_tiles={num_tiles}, actual_cells={actual_cells}")

        record = {
            "dataset": stem,
            "target_cells": target_cells,
            "actual_cells": actual_cells,
            "real_base_cells": real_n,
            "synthesis_method": synthesis_method,
            "seed": seed,
            "num_tiles": num_tiles,
            "build_time_s": round(build_time_s, 2),
            "index_size_mb": index_size_mb,
        }

        out_dir = SCALABILITY_DIR / stem
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / f"{stem}_cells{target_cells}_seed{seed}.csv"
        pd.DataFrame([record]).to_csv(out_csv, index=False)
        print(f"Saved unit result to {out_csv}")

        del synthetic_adata, tiles, tile_covs, data, dag_dict, config, index_bundle
        import matplotlib.pyplot as plt
        plt.close('all')
        gc.collect()


def aggregate(dataset_stem: str) -> pd.DataFrame:
    unit_dir = SCALABILITY_DIR / dataset_stem
    unit_csvs = sorted(unit_dir.glob(f"{dataset_stem}_cells*_seed*.csv"))
    if not unit_csvs:
        raise FileNotFoundError(f"No unit CSVs found under {unit_dir}")

    df = pd.concat([pd.read_csv(p) for p in unit_csvs], ignore_index=True)
    df = df.sort_values("actual_cells").reset_index(drop=True)

    out_csv = SCALABILITY_DIR / f"{dataset_stem}_synthetic_scaling.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved merged sweep CSV to {out_csv}")
    print(df.to_string(index=False))

    from scipy import stats
    log_cells = np.log10(df["actual_cells"].to_numpy())
    exponents = {}
    for metric in ("build_time_s", "index_size_mb"):
        log_y = np.log10(df[metric].to_numpy())
        fit = stats.linregress(log_cells, log_y)
        exponents[metric] = {
            "scaling_exponent": round(float(fit.slope), 4),
            "intercept": round(float(fit.intercept), 4),
            "r_squared": round(float(fit.rvalue) ** 2, 4),
        }
        print(f"  {metric}: exponent={fit.slope:.4f} (R^2={fit.rvalue**2:.4f})")

    out_json = SCALABILITY_DIR / f"{dataset_stem}_scaling_exponents.json"
    with open(out_json, "w") as f:
        json.dump(exponents, f, indent=2)
    print(f"Saved scaling exponents to {out_json}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Synthetic scalability sweep (E6).")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to the real base dataset .h5ad file.")
    parser.add_argument("--target-cells", type=int, default=None, help="Target synthetic cell count for this unit.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for subsampling/bootstrap+jitter.")
    parser.add_argument("--aggregate", type=str, default=None,
                         help="Dataset stem to aggregate (e.g. xenium_human_breast_cancer) instead of running a unit.")
    args = parser.parse_args()

    if args.aggregate:
        aggregate(args.aggregate)
        return

    if args.dataset_path is None or args.target_cells is None:
        parser.error("Unit mode requires --dataset-path and --target-cells (or pass --aggregate <dataset_stem>).")

    run_unit(Path(args.dataset_path), args.target_cells, args.seed)


if __name__ == "__main__":
    main()
