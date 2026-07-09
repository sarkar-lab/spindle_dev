import sys
import time
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import spindle_dev.search as search
import data_helper  # type: ignore
import spindle_dev.interval_index as interval_index
from spindle_dev.utils import log_spd, exp_spd

def find_best_matching_block(target_genes, genes_work, data):
    """
    Finds the cluster and block that has the highest overlap with the target genes.
    """
    print(f"Target genes: {target_genes}")
    
    genes_work_list = list(genes_work) if isinstance(genes_work, np.ndarray) else genes_work
    target_idx = []
    for g in target_genes:
        if g in genes_work_list:
            target_idx.append(genes_work_list.index(g))
        else:
            print(f"Warning: Gene {g} not found in the dataset.")
            
    target_idx_set = set(target_idx)
    
    best_overlap_size = -1
    best_cluster = -1
    best_block_idx = -1
    best_overlapping_indices = []
    
    for c_id in set(data.labels):
        if c_id not in data.block_dict:
            continue
            
        perm = data.perm_list[c_id]
        
        for b_idx, (start, end) in enumerate(data.block_dict[c_id]):
            block_perm = perm[start:end]
            overlap = target_idx_set.intersection(set(block_perm))
            
            if len(overlap) > best_overlap_size:
                best_overlap_size = len(overlap)
                best_cluster = c_id
                best_block_idx = b_idx
                # Keep the order of genes as they appear in the block's permutation
                best_overlapping_indices = [idx for idx in block_perm if idx in overlap]
                
    return best_cluster, best_block_idx, best_overlapping_indices

def construct_partial_query(cluster_id, overlapping_indices, data, num_genes):
    """
    Constructs a partial query by taking the log-Euclidean mean of the training 
    covariance matrices for the specific cluster, subsetted to the overlapping genes.
    Returns a full (num_genes x num_genes) dummy SPD where the submatrix is the mean.
    """
    print(f"Constructing partial query for Cluster {cluster_id} with {len(overlapping_indices)} overlapping genes...")
    
    labels = np.asarray(data.labels)
    mask = (labels == cluster_id)
    
    # Get all training SPDs for this cluster
    spds_c = np.asarray(data.spd_matrices)[mask]
    
    if len(spds_c) == 0:
        raise ValueError(f"No training matrices found for cluster {cluster_id}")
        
    # Subset to the overlapping genes
    spds_sub = spds_c[:, overlapping_indices, :][:, :, overlapping_indices]
    
    # Calculate Log-Euclidean Mean
    logs = [log_spd(m) for m in spds_sub]
    mean_log = np.mean(logs, axis=0)
    mean_spd_sub = exp_spd(mean_log)
    
    # Construct a full-size dummy SPD matrix
    # Diagonal is set to 1.0 for genes not in our overlap
    q_spd = np.eye(num_genes)
    
    # Embed the mean sub-matrix into the full query matrix
    q_spd[np.ix_(overlapping_indices, overlapping_indices)] = mean_spd_sub
    
    # Ensure it's perfectly symmetric
    q_spd = 0.5 * (q_spd + q_spd.T)
    
    return q_spd, mean_spd_sub

def compute_tile_enrichment(top_k_results, adata, data, target_clusters):
    coords = adata.obsm["spatial"]
    clusters = adata.obs["Cluster"].values
    bg_mask = np.isin(clusters, target_clusters)
    bg_pct = bg_mask.mean() * 100

    def get_stats_for_k(k_val):
        retrieved = top_k_results[:k_val]
        total_cells = 0
        target_cells = 0
        for dist, tids in retrieved:
            for tid in tids:
                if tid in data.spd_ids:
                    local_idx = data.spd_ids.index(tid)
                    tile = data.metadata["tiles"][local_idx]
                    x0, y0, x1, y1 = tile.bbox
                    mask = (coords[:, 0] >= x0) & (coords[:, 0] <= x1) & (coords[:, 1] >= y0) & (coords[:, 1] <= y1)
                    spot_clusters = clusters[mask]
                    total_cells += len(spot_clusters)
                    target_cells += np.isin(spot_clusters, target_clusters).sum()
        pct = (target_cells / total_cells * 100) if total_cells > 0 else 0.0
        lift = (pct / bg_pct) if bg_pct > 0 else 0.0
        return pct, lift, total_cells, target_cells

    top5_pct, top5_lift, total5, target5 = get_stats_for_k(5)
    top10_pct, top10_lift, total10, target10 = get_stats_for_k(10)
    top20_pct, top20_lift, total20, target20 = get_stats_for_k(20)
    top50_pct, top50_lift, total50, target50 = get_stats_for_k(50)

    return {
        "bg_pct": bg_pct,
        "top5_pct": top5_pct,
        "top5_lift": top5_lift,
        "top10_pct": top10_pct,
        "top10_lift": top10_lift,
        "top20_pct": top20_pct,
        "top20_lift": top20_lift,
        "top50_pct": top50_pct,
        "top50_lift": top50_lift,
        "total10_cells": total10,
        "target10_cells": target10
    }

def plot_biological_module_results(mod_name, mod_info, top_k_results, enrichment_stats, adata, data, overlapping_indices, genes_work, mean_spd_sub, dataset_name):
    out_dir = project_root / "results" / "gene_list_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = adata.obsm["spatial"]
    clusters = adata.obs["Cluster"].values
    is_target = np.isin(clusters, mod_info["target_clusters"])

    np.random.seed(42)
    bg_idx = np.where(~is_target)[0]
    if len(bg_idx) > 15000:
        bg_sub = np.random.choice(bg_idx, size=15000, replace=False)
    else:
        bg_sub = bg_idx
    target_idx = np.where(is_target)[0]
    selected_idx = np.concatenate([bg_sub, target_idx])

    cell_records = pd.DataFrame({
        'x': coords[selected_idx, 0],
        'y': coords[selected_idx, 1],
        'is_target': is_target[selected_idx]
    })
    cells_csv_path = out_dir / f"{mod_name}_spatial_cells.csv"
    cell_records.to_csv(cells_csv_path, index=False)

    match_records = []
    n_plot = min(5, len(top_k_results))
    for i in range(n_plot):
        dist, tids = top_k_results[i]
        for tid in tids:
            if tid in data.spd_ids:
                local_idx = data.spd_ids.index(tid)
                tile = data.metadata["tiles"][local_idx]
                x0, y0, x1, y1 = tile.bbox
                match_records.append({
                    'rank': i + 1,
                    'tile_id': tid,
                    'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1
                })
    matches_df = pd.DataFrame(match_records)
    matches_csv_path = out_dir / f"{mod_name}_top_matches.csv"
    matches_df.to_csv(matches_csv_path, index=False)
    print(f"Exported spatial cells and top matches CSVs for module {mod_name}")

def plot_master_benchmark(benchmark_summary, dataset_name):
    print("Master summary benchmark figures disabled in CSV-only mode.")

def main():
    np.random.seed(42)
    random.seed(42)
    
    dataset_name = "breast_cancer"
    adata_path = project_root.parent / "dataset" / "xenium_human_breast_cancer.h5ad"
    
    if not adata_path.exists():
        print(f"Dataset not found at {adata_path}. Adjust path to test.")
        return
        
    print(f"Loading {dataset_name} dataset...")
    adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = data_helper.load_and_split_data(adata_path)
    num_genes = len(genes_work)
    
    print("\nBuilding Index Data...")
    data, out_dict = data_helper.run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)
    
    dag_dict, config = data_helper.configure_and_build_dag(data)
    
    print("\nBuilding Interval Index...")
    config.use_interval_index = True
    config.interval_mode = "dyadic"
    config.interval_max_iters = 5
    ivl_idx = interval_index.build_all_interval_indices(data, config)
    
    # Curated Biological Modules
    BIOLOGICAL_MODULES = {
        "Luminal_Tumor_Core": {
            "description": "Invasive Tumor & DCIS Epithelial Core",
            "genes": ["ESR1", "PGR", "ERBB2", "FOXA1", "GATA3", "KRT8", "EPCAM", "CDH1"],
            "target_clusters": ["Invasive_Tumor", "DCIS_1", "DCIS_2"],
            "color": "#E64B35"
        },
        "Macrophage_Myeloid": {
            "description": "Macrophage & Dendritic Cell Niche",
            "genes": ["CD68", "CD163", "MRC1", "C1QA", "ITGAX"],
            "target_clusters": ["Macrophages_1", "Macrophages_2", "IRF7+_DCs", "LAMP3+_DCs"],
            "color": "#00A087"
        },
        "Basal_Myoepithelial": {
            "description": "Myoepithelial & Basal Layer",
            "genes": ["KRT5", "KRT14", "ACTA2", "MYLK"],
            "target_clusters": ["Myoepi_ACTA2+", "Myoepi_KRT15+"],
            "color": "#F39B7F"
        },
        "Endothelial_Vascular": {
            "description": "Vasculature & Endothelial Cells",
            "genes": ["PECAM1", "VWF", "KDR"],
            "target_clusters": ["Endothelial", "Perivascular-Like"],
            "color": "#3C5488"
        },
        "Proliferation_Signature": {
            "description": "Actively Proliferating Tumor Cells",
            "genes": ["MKI67", "TOP2A"],
            "target_clusters": ["Prolif_Invasive_Tumor"],
            "color": "#8491B4"
        }
    }
    
    benchmark_summary = {}
    
    print("\n" + "="*70)
    print("BENCHMARKING BIOLOGICALLY MEANINGFUL GENE LISTS (SPINDLE PARTIAL SEARCH)")
    print("="*70)
    
    search_budget = 50
    
    for mod_name, mod_info in BIOLOGICAL_MODULES.items():
        print(f"\n--- Testing Module: {mod_name} ({mod_info['description']}) ---")
        target_genes = mod_info["genes"]
        
        # Step 1: Find best matching block
        best_cluster, best_block_idx, overlapping_indices = find_best_matching_block(target_genes, genes_work, data)
        
        if not overlapping_indices or len(overlapping_indices) < 2:
            print(f"Warning: Insufficient overlap ({len(overlapping_indices)} genes) for {mod_name}. Skipping...")
            continue
            
        print(f"Best Match in Cluster {best_cluster}, Block {best_block_idx} | Overlap: {len(overlapping_indices)} genes -> {[genes_work[i] for i in overlapping_indices]}")
        
        # Step 2: Construct partial query
        q_spd, mean_spd_sub = construct_partial_query(best_cluster, overlapping_indices, data, num_genes)
        
        # Step 3: Search
        top_k_results = data_helper.search_all_clusters_spindle(ivl_idx, data, overlapping_indices, q_spd, top_k=search_budget)
        
        if not top_k_results:
            print(f"No matching spatial patches found for {mod_name}.")
            continue
            
        # Step 4: Compute biological enrichment statistics
        enrichment_stats = compute_tile_enrichment(top_k_results, adata, data, mod_info["target_clusters"])
        
        print(f"Top-10 Target Cell Density: {enrichment_stats['top10_pct']:.2f}% vs Background: {enrichment_stats['bg_pct']:.2f}% | Enrichment Lift: {enrichment_stats['top10_lift']:.2f}x")
        
        # Step 5: Plot individual validation figure
        plot_biological_module_results(mod_name, mod_info, top_k_results, enrichment_stats, adata, data, overlapping_indices, genes_work, mean_spd_sub, dataset_name)
        
        # Record stats
        benchmark_summary[mod_name] = {
            "description": mod_info["description"],
            "color": mod_info["color"],
            "overlap_genes": len(overlapping_indices),
            "bg_pct": enrichment_stats["bg_pct"],
            "top5_pct": enrichment_stats["top5_pct"],
            "top5_lift": enrichment_stats["top5_lift"],
            "top10_pct": enrichment_stats["top10_pct"],
            "top10_lift": enrichment_stats["top10_lift"],
            "top20_pct": enrichment_stats["top20_pct"],
            "top20_lift": enrichment_stats["top20_lift"],
            "top50_pct": enrichment_stats["top50_pct"],
            "top50_lift": enrichment_stats["top50_lift"]
        }
        
    if benchmark_summary:
        # Export metrics to CSV
        df_metrics = pd.DataFrame.from_dict(benchmark_summary, orient='index')
        out_dir = project_root / "results" / "gene_list_search"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "benchmark_metrics.csv"
        df_metrics.to_csv(csv_path)
        print(f"\nExported quantitative benchmark metrics to {csv_path}")
        
        # Master comparison plotting disabled in CSV-only mode
        
    print("\n" + "="*70)
    print("ALL BIOLOGICAL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
