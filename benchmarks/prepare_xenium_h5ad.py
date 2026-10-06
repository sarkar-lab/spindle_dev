"""Download a public 10x Xenium dataset and save it as an h5ad in the layout of the existing datasets.

The existing inputs (datasets.yaml, dirs.xenium) are plain conversions of the Xenium outputs:
X = raw counts of the gene-expression features (float32), obs = the per-cell metrics of
cells.csv.gz, obsm["spatial"] = the cell centroids (x, y in microns). This writes the same from
two files of the Xenium output bundle: <prefix>_cell_feature_matrix.h5 and <prefix>_cells.csv.gz.

Datasets (DATASETS): name -> 10x download prefix.
  tonsil_reactive  Human Tonsil Data with Xenium Human Multi-Tissue and Cancer Panel, reactive
                   follicular hyperplasia, FFPE (1,349,620 cells, 377 genes; the largest public
                   single Xenium section we found). Used by the Fig 2D cell ladder.

  python benchmarks/prepare_xenium_h5ad.py tonsil_reactive
"""

import argparse
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

import paths

CF = "https://cf.10xgenomics.com/samples/xenium"
DATASETS = {
    "tonsil_reactive": f"{CF}/1.9.0/Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE/"
                       "Xenium_V1_hTonsil_reactive_follicular_hyperplasia_section_FFPE",
}


def fetch(url, dest):
    if not dest.exists():
        subprocess.run(["curl", "-fL", "--retry", "3", "-o", str(dest), url], check=True)


def build(name):
    prefix = DATASETS[name]
    raw = paths.data_dir("extra") / "raw" / name
    raw.mkdir(parents=True, exist_ok=True)
    fetch(f"{prefix}_cell_feature_matrix.h5", raw / "cell_feature_matrix.h5")
    fetch(f"{prefix}_cells.csv.gz", raw / "cells.csv.gz")

    a = sc.read_10x_h5(raw / "cell_feature_matrix.h5", gex_only=True)
    a.var_names_make_unique()
    cells = pd.read_csv(raw / "cells.csv.gz")
    cells["cell_id"] = cells["cell_id"].astype(str)
    cells = cells.set_index("cell_id").loc[a.obs_names.astype(str)]
    a.obsm["spatial"] = cells[["x_centroid", "y_centroid"]].to_numpy(np.float64)
    a.obs = cells.drop(columns=["x_centroid", "y_centroid"])
    a.X = a.X.astype(np.float32)
    out = paths.dataset_path(name)  # its entry in datasets.yaml
    ad.AnnData(a.X, obs=a.obs, var=a.var, obsm={"spatial": a.obsm["spatial"]}).write_h5ad(out)
    print(f"{out}: {a.n_obs} cells x {a.n_vars} genes")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", choices=sorted(DATASETS))
    build(parser.parse_args().name)


if __name__ == "__main__":
    main()
