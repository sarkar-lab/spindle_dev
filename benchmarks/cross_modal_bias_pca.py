"""Cross-platform bias before/after the global correction, in tangent-space PCA.

Uses the same matched Xenium/Visium breast sections, shared genes, tiling and
tile covariances as benchmarks/cross_modal_search.py (its functions are
imported, not copied). For each direction (x2v: Xenium queries against a
Visium index; v2x: the reverse), the whole-matrix log-covariance of every tile
is vectorized (upper triangle, off-diagonal x sqrt 2, / sqrt p, so L2 = the
log-Euclidean distance), a PCA is fitted on the INDEX-side tiles only, and
index tiles, raw query tiles and globally corrected query tiles
(L_corr = (L_q - mean_q) * min(1, sd_t/sd_q) + mean_t) are projected onto it.

Here the correction statistics use every query-platform tile; the E12 search
itself used the seed's 50 sampled queries, which gives nearly the same shift.

Outputs (results/cross_modal_search/):
  bias_pca.csv          direction, modality, role, corrected, tile_id, PC1..PC5
  bias_pca_summary.csv  per direction and state: centroid gap, NN distance to
                        the index, modality silhouette, explained variance
"""

import argparse

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import experiment_common as ec
import cross_modal_search as cms  # type: ignore
from spindle_dev.preprocessing import build_tile_covs_full
from spindle_dev.utils import log_spd

OUT_DIR = ec.PROJECT_ROOT / "results" / "cross_modal_search"
N_PC = 5


def vectorize(logs):
    p = logs[0].shape[0]
    iu = np.triu_indices(p)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)) / np.sqrt(p)
    return np.stack([L[iu] * w for L in logs])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--xenium-path", default=str(cms.DEFAULT_DATA_DIR / "xenium_rotated.h5ad"))
    parser.add_argument("--visium-path", default=str(cms.DEFAULT_DATA_DIR / "visium_rotated.h5ad"))
    args = parser.parse_args()

    adata_vi = sc.read_h5ad(args.visium_path)
    adata_xe = sc.read_h5ad(args.xenium_path)
    adata_vi.var_names_make_unique()
    adata_xe.var_names_make_unique()
    genes = sorted(set(adata_vi.var_names) & set(adata_xe.var_names))
    adata_xe, adata_vi = adata_xe[:, genes].copy(), adata_vi[:, genes].copy()
    tiles_xe, tiles_vi = cms.build_tiles(adata_xe, adata_vi)
    covs = {"Xenium": [t["cov"] for t in build_tile_covs_full(adata_xe, tiles_xe, n_jobs=1)],
            "Visium": [t["cov"] for t in build_tile_covs_full(adata_vi, tiles_vi, n_jobs=1)]}
    print(f"{len(genes)} shared genes; {len(covs['Xenium'])} Xenium / {len(covs['Visium'])} Visium tiles")
    logs = {m: [log_spd(C) for C in cs] for m, cs in covs.items()}

    rows, summary = [], []
    for direction, (index_mod, query_mod) in {"x2v": ("Visium", "Xenium"), "v2x": ("Xenium", "Visium")}.items():
        corrected = cms.compute_global_corrected_covs(covs[query_mod], covs[index_mod])
        V_index = vectorize(logs[index_mod])
        states = {False: vectorize(logs[query_mod]), True: vectorize([log_spd(C) for C in corrected])}
        pca = PCA(n_components=N_PC, random_state=0).fit(V_index)
        Z_index = pca.transform(V_index)
        for k, z in enumerate(Z_index):
            rows.append({"direction": direction, "modality": index_mod, "role": "index", "corrected": False,
                         "tile_id": k, **{f"PC{j + 1}": z[j] for j in range(N_PC)}})
        for corr, V_q in states.items():
            Z_q = pca.transform(V_q)
            for k, z in enumerate(Z_q):
                rows.append({"direction": direction, "modality": query_mod, "role": "query", "corrected": corr,
                             "tile_id": k, **{f"PC{j + 1}": z[j] for j in range(N_PC)}})
            # Distances in the full tangent space (= log-Euclidean), not just the 2 plotted PCs.
            nn = np.array([np.min(np.linalg.norm(V_index - v, axis=1)) for v in V_q])
            both = np.vstack([V_index, V_q])
            lab = np.r_[np.zeros(len(V_index)), np.ones(len(V_q))]
            summary.append({
                "direction": direction, "corrected": corr,
                "centroid_gap_le": float(np.linalg.norm(V_index.mean(0) - V_q.mean(0))),
                "median_nn_dist_to_index_le": float(np.median(nn)),
                "modality_silhouette": float(silhouette_score(both, lab)),
                "pc1_pc2_explained_var": float(pca.explained_variance_ratio_[:2].sum()),
                "n_index": len(V_index), "n_query": len(V_q),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "bias_pca.csv", index=False)
    summ = pd.DataFrame(summary)
    summ.to_csv(OUT_DIR / "bias_pca_summary.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
