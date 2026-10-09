"""Agreement between covariance niches and cell types (Figs 6A-B, 7A; all 8 datasets).

Cell types: breast ``obs['Cluster']`` (20 types after dropping "Unlabeled", exactly as
the index build did); every other dataset the 10x graph-based clusters of its Xenium
analysis bundle (metric_check.cell_type_codes; unlabelled cells are left out). For every
existing index build (seeds 0-4 and the seed-73 production build), compares the
covariance niche of each training tile with the cell types:

* tile level -- niche vs the majority ``Cluster`` label of the tile's cells;
* cell level -- every cell of a training tile inherits the tile's niche
  (``ProcessedData.assign_label_to_spots``) and is compared with its own label.

Held-out tiles have no niche and are excluded. No index is rebuilt.

Breast also gets the DCIS niche (largest DCIS_1 + DCIS_2 share).

Outputs (results/niche_concordance/):
  <key>_ari_nmi.csv           one row per seed (ARI/NMI at both levels)
  <key>_niche_composition.csv niche x cell type cell counts and row fractions, per seed
  <key>_tile_labels.csv       per training tile: seed, niche, majority label, centroid
  <key>_cell_types.csv        per cell type: code, name, n_cells, top marker genes (mean count ratio vs
                              the other labelled cells; for naming the graphclust clusters)

  sbatch slurm_jobs/biology/run_niche_concordance.sbatch [--datasets ...]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import experiment_common as ec
from metric_check import cell_type_codes  # type: ignore

OUT_DIR = ec.PROJECT_ROOT / "results" / "niche_concordance"
DCIS_TYPES = ["DCIS_1", "DCIS_2"]
N_MARKERS = 3


def seed_rows(stem, seed, adata_codes, cell_types, tiles):
    bundle = ec.load_index(stem, seed)
    data = bundle["data"]
    covs = ec.load_raw_covs(stem, seed)
    train_idx = np.asarray(covs["train_idx"])
    ec.check_tiles_match(tiles, data, train_idx)
    niches = np.asarray(data.labels).astype(int)

    tile_records, cell_niche, cell_label = [], [], []
    for k, t in enumerate(data.metadata["tiles"]):
        codes = adata_codes[t.idx]
        codes = codes[codes >= 0]
        counts = np.bincount(codes, minlength=len(cell_types))
        tile_records.append({
            "seed": seed, "tile_id": int(t.id), "niche": int(niches[k]),
            "majority_label": cell_types[int(np.argmax(counts))],
            "majority_fraction": counts.max() / counts.sum(),
            "n_cells": int(counts.sum()),
            "x": float(t.centroid[0]), "y": float(t.centroid[1]),
        })
        cell_niche.append(np.full(codes.size, niches[k]))
        cell_label.append(codes)
    tiles_df = pd.DataFrame(tile_records)
    cell_niche = np.concatenate(cell_niche)
    cell_label = np.concatenate(cell_label)

    # Cross-check against the index's own spot -> niche map.
    data.assign_label_to_spots()
    spot_ids = np.concatenate([t.idx[adata_codes[t.idx] >= 0] for t in data.metadata["tiles"]])
    assert np.array_equal(cell_niche, np.array([data.spot_label[i] for i in spot_ids]))

    metrics = {
        "seed": seed,
        "n_niches": int(len(np.unique(niches))),
        "n_tiles": len(tiles_df),
        "n_cells": int(cell_label.size),
        "ari_tile_majority": adjusted_rand_score(tiles_df["majority_label"], tiles_df["niche"]),
        "nmi_tile_majority": normalized_mutual_info_score(tiles_df["majority_label"], tiles_df["niche"]),
        "ari_cell": adjusted_rand_score(cell_label, cell_niche),
        "nmi_cell": normalized_mutual_info_score(cell_label, cell_niche),
    }

    comp = pd.crosstab(pd.Series(cell_niche, name="niche"),
                       pd.Series(pd.Categorical.from_codes(cell_label, cell_types), name="cell_type"),
                       dropna=False)
    comp_long = comp.stack().rename("n_cells").reset_index()
    comp_long["fraction"] = comp_long["n_cells"] / comp_long.groupby("niche")["n_cells"].transform("sum")
    comp_long.insert(0, "seed", seed)

    if not set(DCIS_TYPES) <= set(cell_types):
        return metrics, comp_long, tiles_df
    dcis = comp_long[comp_long["cell_type"].isin(DCIS_TYPES)].groupby("niche")["fraction"].sum()
    metrics["max_dcis_fraction"] = float(dcis.max())
    metrics["dcis_niche"] = int(dcis.idxmax())
    # Share of all DCIS cells that fall in that one niche.
    dcis_counts = comp_long[comp_long["cell_type"].isin(DCIS_TYPES)].groupby("niche")["n_cells"].sum()
    metrics["dcis_cells_in_dcis_niche"] = float(dcis_counts[metrics["dcis_niche"]] / dcis_counts.sum())
    return metrics, comp_long, tiles_df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets", nargs="+", default=list(ec.DATASETS))
    parser.add_argument("--seeds", type=int, nargs="+", default=ec.MULTISEED + [ec.PRODUCTION_SEED])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key in args.datasets:
        run_dataset(key, args.seeds)


def markers(adata, codes, cell_types, n_top=N_MARKERS):
    """Top genes per cell type by mean count over the mean in the other labelled cells (pseudocount 0.1)."""
    import scipy.sparse as sp

    X = adata.X.tocsr() if sp.issparse(adata.X) else np.asarray(adata.X)
    genes = np.asarray(adata.var_names)
    lab = codes >= 0
    total = np.asarray(X[lab].sum(0)).ravel()
    rows = []
    for c, name in enumerate(cell_types):
        m = codes == c
        if not m.any():
            continue
        inside = np.asarray(X[m].sum(0)).ravel()
        ratio = (inside / m.sum() + 0.1) / ((total - inside) / (lab.sum() - m.sum()) + 0.1)
        rows.append({"code": c, "name": name, "n_cells": int(m.sum()),
                     "markers": " ".join(genes[np.argsort(-ratio)[:n_top]])})
    return pd.DataFrame(rows)


def run_dataset(key, seeds):
    stem = ec.resolve_stem(key)
    key = next((k for k, v in ec.DATASETS.items() if v == stem), stem)
    adata = ec.load_adata(stem)
    codes = np.asarray(cell_type_codes(stem, adata))
    if "Cluster" in adata.obs.columns:
        cell_types = list(adata.obs["Cluster"].astype("category").cat.remove_unused_categories().cat.categories)
    else:
        cell_types = [f"graphclust_{c + 1}" for c in range(int(codes.max()) + 1)]
    markers(adata, codes, cell_types).to_csv(OUT_DIR / f"{key}_cell_types.csv", index=False)
    tiles = ec.rebuild_tiles(adata)
    print(f"{adata.n_obs} labelled cells, {len(cell_types)} cell types, {len(tiles)} tiles")

    all_metrics, all_comp, all_tiles = [], [], []
    for seed in seeds:
        m, comp, tdf = seed_rows(stem, seed, codes, cell_types, tiles)
        print(m)
        all_metrics.append(m)
        all_comp.append(comp)
        all_tiles.append(tdf)

    # Hand-checkable example: the first training tile of the first seed.
    t0 = all_tiles[0].iloc[0]
    print(f"Example tile {t0.tile_id}: majority={t0.majority_label} ({t0.majority_fraction:.2f} of {t0.n_cells} cells)")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_DIR / f"{key}_ari_nmi.csv", index=False)
    pd.concat(all_comp).to_csv(OUT_DIR / f"{key}_niche_composition.csv", index=False)
    pd.concat(all_tiles).to_csv(OUT_DIR / f"{key}_tile_labels.csv", index=False)
    multi = metrics_df[metrics_df["seed"].isin(ec.MULTISEED)]
    print("\nMean +/- s.d. over seeds 0-4:")
    for col in [c for c in ["ari_tile_majority", "nmi_tile_majority", "ari_cell", "nmi_cell", "max_dcis_fraction"]
                if c in multi]:
        print(f"  {col}: {multi[col].mean():.3f} +/- {multi[col].std():.3f}")


if __name__ == "__main__":
    main()
