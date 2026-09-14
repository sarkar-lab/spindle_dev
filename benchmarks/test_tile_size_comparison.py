"""
One-off comparison: does shrinking quadtree tiles (and the gene count used to
build the index, to keep n/p healthy) actually sharpen biological localization,
or just add noise? Uses breast_cancer, which has ground-truth obs['Cluster']
labels, to measure cluster-purity lift instead of just continuous pathway score.
"""
import sys
import time
from pathlib import Path
import random
import numpy as np
import scanpy as sc
import pandas as pd

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(current_dir))

import data_helpers
import spindle_dev.interval_index as interval_index

from gene_signature_search import (
    find_best_matching_block,
    construct_partial_query,
)

ADATA_PATH = Path("/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium/xenium_human_breast_cancer.h5ad")

TEST_MODULES = {
    "Luminal_Tumor_Core": {
        "genes": ["ESR1", "PGR", "ERBB2", "FOXA1", "GATA3", "KRT8", "EPCAM", "CDH1"],
        "target_clusters": ["Invasive_Tumor", "DCIS_1", "DCIS_2"],
    },
    "Basal_Myoepithelial": {
        "genes": ["KRT5", "KRT14", "ACTA2", "MYLK"],
        "target_clusters": ["Myoepi_ACTA2+", "Myoepi_KRT15+"],
    },
    "Macrophage_Myeloid": {
        "genes": ["CD68", "CD163", "MRC1", "C1QA", "ITGAX"],
        "target_clusters": ["Macrophages_1", "Macrophages_2", "IRF7+_DCs", "LAMP3+_DCs"],
    },
}

CONFIGS = [
    {"max_pts": 200, "top_genes": 800, "label": "baseline_200pts_800genes"},
    {"max_pts": 100, "top_genes": 300, "label": "small_100pts_300genes"},
    {"max_pts": 50, "top_genes": 150, "label": "smaller_50pts_150genes"},
]


def compute_cluster_enrichment(top_k_results, adata, data, target_clusters, k_val):
    cluster_labels = adata.obs["Cluster"].values
    is_target = np.isin(cluster_labels, target_clusters)
    background_frac = float(np.mean(is_target))

    retrieved = top_k_results[:k_val]
    total_cells = 0
    target_cells = 0
    coords = adata.obsm["spatial"]
    for dist, tids in retrieved:
        for tid in tids:
            if tid in data.spd_ids:
                local_idx = data.spd_ids.index(tid)
                tile = data.metadata["tiles"][local_idx]
                x0, y0, x1, y1 = tile.bbox
                mask = (coords[:, 0] >= x0) & (coords[:, 0] <= x1) & (coords[:, 1] >= y0) & (coords[:, 1] <= y1)
                total_cells += int(mask.sum())
                target_cells += int(is_target[mask].sum())

    frac = (target_cells / total_cells) if total_cells > 0 else 0.0
    lift = (frac / background_frac) if background_frac > 0 else float("nan")
    return frac, background_frac, lift, total_cells


def run_config(cfg):
    print(f"\n{'='*70}\nCONFIG: {cfg['label']} (max_pts={cfg['max_pts']}, top_genes={cfg['top_genes']})\n{'='*70}")
    t0 = time.time()

    np.random.seed(42)
    random.seed(42)

    adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = \
        data_helpers.load_and_split_data(ADATA_PATH, max_pts=cfg["max_pts"], top_genes=cfg["top_genes"])
    num_genes = len(genes_work)

    data, out_dict = data_helpers.run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)
    dag_dict, config = data_helpers.configure_and_build_dag(data)

    config.use_interval_index = True
    config.interval_mode = "dyadic"
    config.interval_max_iters = 5
    ivl_idx = interval_index.build_all_interval_indices(data, config)

    build_time = time.time() - t0
    print(f"Index build took {build_time:.1f}s, {len(train_tiles)} tiles")

    rows = []
    for mod_name, mod_info in TEST_MODULES.items():
        target_genes = mod_info["genes"]
        best_cluster, best_block_idx, overlapping_indices = find_best_matching_block(target_genes, genes_work, data)
        if not overlapping_indices or len(overlapping_indices) < 2:
            print(f"Skipping {mod_name}: insufficient overlap")
            continue

        q_spd, mean_spd_sub = construct_partial_query(best_cluster, overlapping_indices, data, num_genes)
        top_k_results = data_helpers.search_all_clusters_spindle(ivl_idx, data, overlapping_indices, q_spd, top_k=500)
        if not top_k_results:
            print(f"No matches for {mod_name}")
            continue

        sc.tl.score_genes(adata, gene_list=target_genes, score_name='pathway_score', use_raw=False)

        for k_val in (10, 50, 200):
            frac, bg, cluster_lift, n_cells = compute_cluster_enrichment(top_k_results, adata, data, mod_info["target_clusters"], k_val)
            rows.append({
                "config": cfg["label"], "max_pts": cfg["max_pts"], "top_genes": cfg["top_genes"],
                "n_tiles": len(train_tiles), "module": mod_name, "k": k_val,
                "cluster_frac_in_topk": round(frac, 4), "cluster_frac_background": round(bg, 4),
                "cluster_lift": round(cluster_lift, 3) if cluster_lift == cluster_lift else None,
                "n_cells_in_topk": n_cells,
            })
            print(f"  {mod_name} @k={k_val}: cluster-purity lift = {cluster_lift:.2f}x ({frac:.3f} vs bg {bg:.3f}, n={n_cells})")

    rows_df = pd.DataFrame(rows)
    rows_df["build_time_s"] = round(build_time, 1)
    return rows_df


def main():
    all_rows = []
    for cfg in CONFIGS:
        try:
            df = run_config(cfg)
            all_rows.append(df)
        except Exception as e:
            print(f"CONFIG {cfg['label']} FAILED: {e}")
            import traceback
            traceback.print_exc()

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        out_path = project_root / "results" / "gene_signature_search" / "tile_size_comparison.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        print(f"\n\nSaved comparison to {out_path}")
        print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
