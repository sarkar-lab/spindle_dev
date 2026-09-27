"""E10-lite: agreement between breast covariance niches and annotated cell types.

Breast is the only dataset with cell-type labels (``obs['Cluster']``, 20
types after dropping "Unlabeled", exactly as the index build did). For every
existing breast index build (seeds 0-4 and the seed-73 production build),
compares the covariance niche of each training tile with the annotation:

* tile level -- niche vs the majority ``Cluster`` label of the tile's cells;
* cell level -- every cell of a training tile inherits the tile's niche
  (``ProcessedData.assign_label_to_spots``) and is compared with its own label.

Held-out tiles have no niche and are excluded. No index is rebuilt.

Outputs (results/niche_concordance/):
  breast_ari_nmi.csv           one row per seed (ARI/NMI at both levels)
  breast_niche_composition.csv niche x cell type cell counts and row fractions, per seed
  breast_tile_labels.csv       per training tile: seed, niche, majority label, centroid

Light (one 160k-cell h5ad, small pickles); run through SLURM with the E13 job:
  sbatch slurm_jobs/run_breast_concordance.sbatch
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import experiment_common as ec

STEM = ec.DATASETS["breast_cancer"]
OUT_DIR = ec.PROJECT_ROOT / "results" / "niche_concordance"
DCIS_TYPES = ["DCIS_1", "DCIS_2"]


def seed_rows(seed, adata_codes, cell_types, tiles):
    bundle = ec.load_index(STEM, seed)
    data = bundle["data"]
    covs = ec.load_raw_covs(STEM, seed)
    train_idx = np.asarray(covs["train_idx"])
    ec.check_tiles_match(tiles, data, train_idx)
    niches = np.asarray(data.labels).astype(int)

    tile_records, cell_niche, cell_label = [], [], []
    for k, t in enumerate(data.metadata["tiles"]):
        codes = adata_codes[t.idx]
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
    spot_ids = np.concatenate([t.idx for t in data.metadata["tiles"]])
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

    dcis = comp_long[comp_long["cell_type"].isin(DCIS_TYPES)].groupby("niche")["fraction"].sum()
    metrics["max_dcis_fraction"] = float(dcis.max())
    metrics["dcis_niche"] = int(dcis.idxmax())
    # Share of all DCIS cells that fall in that one niche.
    dcis_counts = comp_long[comp_long["cell_type"].isin(DCIS_TYPES)].groupby("niche")["n_cells"].sum()
    metrics["dcis_cells_in_dcis_niche"] = float(dcis_counts[metrics["dcis_niche"]] / dcis_counts.sum())
    return metrics, comp_long, tiles_df


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, nargs="+", default=ec.MULTISEED + [ec.PRODUCTION_SEED])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    adata = ec.load_adata(STEM)
    cat = adata.obs["Cluster"].astype("category").cat.remove_unused_categories()
    cell_types = list(cat.cat.categories)
    codes = cat.cat.codes.to_numpy()
    tiles = ec.rebuild_tiles(adata)
    print(f"{adata.n_obs} labelled cells, {len(cell_types)} cell types, {len(tiles)} tiles")

    all_metrics, all_comp, all_tiles = [], [], []
    for seed in args.seeds:
        m, comp, tdf = seed_rows(seed, codes, cell_types, tiles)
        print(m)
        all_metrics.append(m)
        all_comp.append(comp)
        all_tiles.append(tdf)

    # Hand-checkable example: the first training tile of the first seed.
    t0 = all_tiles[0].iloc[0]
    print(f"Example tile {t0.tile_id}: majority={t0.majority_label} ({t0.majority_fraction:.2f} of {t0.n_cells} cells)")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT_DIR / "breast_ari_nmi.csv", index=False)
    pd.concat(all_comp).to_csv(OUT_DIR / "breast_niche_composition.csv", index=False)
    pd.concat(all_tiles).to_csv(OUT_DIR / "breast_tile_labels.csv", index=False)
    multi = metrics_df[metrics_df["seed"].isin(ec.MULTISEED)]
    print("\nMean +/- s.d. over seeds 0-4:")
    for col in ["ari_tile_majority", "nmi_tile_majority", "ari_cell", "nmi_cell", "max_dcis_fraction"]:
        print(f"  {col}: {multi[col].mean():.3f} +/- {multi[col].std():.3f}")


if __name__ == "__main__":
    main()
