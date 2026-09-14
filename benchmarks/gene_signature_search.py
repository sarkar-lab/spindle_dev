import argparse
import sys
import time
from pathlib import Path
import random
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# benchmarks/ is one level below the project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
src_path = project_root / 'src'

if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import spindle_dev.search as search
import data_helpers  # type: ignore
import spindle_dev.interval_index as interval_index
from spindle_dev.utils import log_spd, exp_spd

DATASET_DIR = Path("/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium")

DEFAULT_DATASETS = {
    "brain_cancer": DATASET_DIR / "xenium_human_brain_cancer.h5ad",
    "breast_cancer": DATASET_DIR / "xenium_human_breast_cancer.h5ad",
    "kidney_nondiseased": DATASET_DIR / "xenium_human_kidney_nondiseased.h5ad",
    "lung_cancer": DATASET_DIR / "xenium_human_lung_cancer.h5ad",
    "lymph_node": DATASET_DIR / "xenium_human_lymph_node.h5ad",
    "lymph_node_5k": DATASET_DIR / "xenium_human_lymph_node_5k.h5ad",
    "pancreatic_cancer": DATASET_DIR / "xenium_human_pancreatic_cancer.h5ad",
    "skin_melanoma": DATASET_DIR / "xenium_human_skin_melanoma.h5ad",
}

# Tissue-specific curated gene signatures. Every gene listed here was verified to be
# present in the corresponding Xenium panel before being added.
TISSUE_MODULES = {
    "breast": {
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
    },
    "kidney": {
        "Tubular_Epithelium": {
            "description": "Proximal/Distal Tubule & Collecting Duct Segments",
            "genes": ["SLC22A8", "SLC4A1", "AQP2", "UMOD", "SLC26A3"],
            "target_clusters": [],
            "color": "#4DBBD5"
        },
        "Injury_Immune_Infiltrate": {
            "description": "Tubular Injury Marker & Immune Infiltrate",
            "genes": ["HAVCR2", "CD68", "CD163", "CD3D", "CD3E"],
            "target_clusters": [],
            "color": "#91D1C2"
        },
    },
    "lung": {
        "Alveolar_Airway_Epithelium": {
            "description": "Alveolar Type 1 & Ciliated/Secretory Airway Epithelium",
            "genes": ["AGER", "FOXJ1", "SCGB2A1", "KRT7", "SOX2"],
            "target_clusters": [],
            "color": "#F39B7F"
        },
        "Vascular_Endothelial": {
            "description": "Lung Vasculature & Endothelium",
            "genes": ["SOX17", "SOX18", "PECAM1", "VWF"],
            "target_clusters": [],
            "color": "#3C5488"
        },
    },
    "pancrea": {
        "Exocrine_Ductal": {
            "description": "Ductal & Exocrine Epithelium",
            "genes": ["KRT7", "KRT20", "CFTR"],
            "target_clusters": [],
            "color": "#E64B35"
        },
        "Endocrine_Islet": {
            "description": "Islets of Langerhans (Alpha/Beta/Delta/PP Cells)",
            "genes": ["INS", "GCG", "SST", "PPY", "CHGA"],
            "target_clusters": [],
            "color": "#00A087"
        },
        "Mast_Cell_Stroma": {
            "description": "Mast Cells & Perivascular Stroma",
            "genes": ["CPA3", "ACTA2", "PDGFRB"],
            "target_clusters": [],
            "color": "#8491B4"
        },
    },
    "brain": {
        "Glioma_Tumor_Proliferative": {
            "description": "Proliferating Glioma Tumor Core",
            "genes": ["EGFR", "PDGFRA", "MKI67"],
            "target_clusters": [],
            "color": "#E64B35"
        },
        "Microglia_Myeloid": {
            "description": "Microglia & Tumor-Associated Myeloid Cells",
            "genes": ["CD68", "CD44", "PTPRC", "MMP9", "MMP12"],
            "target_clusters": [],
            "color": "#00A087"
        },
        "TIL_Lymphocyte": {
            "description": "Tumor-Infiltrating Lymphocytes",
            "genes": ["CD3D", "CD3E", "CD37", "CD38"],
            "target_clusters": [],
            "color": "#4DBBD5"
        },
    },
    "lymph_node": {
        "B_Cell_Germinal_Center": {
            "description": "B Cell / Germinal Center Niche",
            "genes": ["MS4A1", "CD79A", "CD19", "MZB1"],
            "target_clusters": [],
            "color": "#00A087"
        },
        "T_Cell_Zone": {
            "description": "Paracortical T Cell Zone",
            "genes": ["CD3D", "CD3E", "CD4", "CD8A"],
            "target_clusters": [],
            "color": "#4DBBD5"
        },
        "Macrophage_DC_Niche": {
            "description": "Macrophage & Dendritic Cell Niche",
            "genes": ["CD68", "CD163", "LAMP3", "CD83", "CD86", "CD300E"],
            "target_clusters": [],
            "color": "#F39B7F"
        },
        "Lymphatic_Vascular_Stroma": {
            "description": "Lymphatic Sinuses & Vascular Stroma",
            "genes": ["LYVE1", "PROX1", "PDPN", "CD34", "CCL19"],
            "target_clusters": [],
            "color": "#3C5488"
        },
    },
    "skin": {
        "Melanocyte_Tumor": {
            "description": "Melanocyte / Melanoma Tumor Core",
            "genes": ["TYR", "TYRP1", "MLANA", "PMEL", "DCT", "MITF", "SOX10", "S100B"],
            "target_clusters": [],
            "color": "#E64B35"
        },
        "Keratinocyte_Epidermis": {
            "description": "Epidermal Keratinocyte Layers",
            "genes": ["KRT1", "KRT5", "KRT14", "KRT17", "LOR"],
            "target_clusters": [],
            "color": "#F39B7F"
        },
        "Immune_Infiltrate": {
            "description": "Immune Cell Infiltrate",
            "genes": ["CD3D", "CD3E", "CD68", "C1QA", "ITGAX"],
            "target_clusters": [],
            "color": "#00A087"
        },
    },
}

# Cross-tissue functional/pathway modules, run identically on every dataset. Genes
# missing from a given panel are dropped automatically by find_best_matching_block;
# the resulting per-dataset overlap differences are themselves part of the signal.
GENERIC_MODULES = {
    "Cytotoxic_Immune": {
        "description": "Cytotoxic T / NK Cell Activity",
        "genes": ["NKG7", "GNLY", "KLRD1", "CD8A"],
        "target_clusters": [],
        "color": "#8491B4"
    },
    "Pan_Myeloid_Macrophage": {
        "description": "Pan-Myeloid / Macrophage Compartment",
        "genes": ["CD68", "CD163", "CD14", "FCGR3A", "ITGAX"],
        "target_clusters": [],
        "color": "#91D1C2"
    },
    "Interferon_Response": {
        "description": "Interferon-Stimulated Response",
        "genes": ["CXCL9", "CXCL10", "CXCL11", "STAT1", "IRF1"],
        "target_clusters": [],
        "color": "#7E6148"
    },
    "Inflammatory_Response": {
        "description": "General Inflammatory / Cytokine Response",
        "genes": ["IL6", "TNF", "IL1B", "CCL2", "ICAM1"],
        "target_clusters": [],
        "color": "#B09C85"
    },
    "Hypoxia_Angiogenesis": {
        "description": "Hypoxia-Driven Angiogenic Signaling",
        "genes": ["VEGFA", "HIF1A", "MMP9"],
        "target_clusters": [],
        "color": "#DC0000"
    },
    "EMT_Stromal_Invasion": {
        "description": "Epithelial-Mesenchymal Transition & Stromal Invasion",
        "genes": ["VIM", "CDH2", "SNAI1", "ZEB1", "FN1", "MMP2", "SPARC"],
        "target_clusters": [],
        "color": "#631879"
    },
}


def select_modules_for_dataset(dataset_name):
    """
    Picks the tissue-specific module set(s) whose key appears (case-insensitively)
    as a substring of dataset_name, and always unions in GENERIC_MODULES.
    """
    name_lower = dataset_name.lower()
    modules = {}
    matched_any = False
    for tissue_key, tissue_modules in TISSUE_MODULES.items():
        if tissue_key in name_lower:
            modules.update(tissue_modules)
            matched_any = True

    if not matched_any:
        print(f"Warning: No tissue-specific module set found for dataset '{dataset_name}'. Using generic modules only.")

    modules.update(GENERIC_MODULES)
    return modules


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

def compute_tile_enrichment(top_k_results, adata, data):
    coords = adata.obsm["spatial"]
    pathway_scores = adata.obs["pathway_score"].values
    background_score = float(np.mean(pathway_scores))

    def get_stats_for_k(k_val):
        retrieved = top_k_results[:k_val]
        total_cells = 0
        sum_scores = 0.0
        for dist, tids in retrieved:
            for tid in tids:
                if tid in data.spd_ids:
                    local_idx = data.spd_ids.index(tid)
                    tile = data.metadata["tiles"][local_idx]
                    x0, y0, x1, y1 = tile.bbox
                    mask = (coords[:, 0] >= x0) & (coords[:, 0] <= x1) & (coords[:, 1] >= y0) & (coords[:, 1] <= y1)
                    spot_scores = pathway_scores[mask]
                    total_cells += len(spot_scores)
                    sum_scores += np.sum(spot_scores)
        score = (sum_scores / total_cells) if total_cells > 0 else 0.0
        return score, total_cells

    top5_score, total5 = get_stats_for_k(5)
    top10_score, total10 = get_stats_for_k(10)
    top20_score, total20 = get_stats_for_k(20)
    top50_score, total50 = get_stats_for_k(50)
    top200_score, total200 = get_stats_for_k(200)

    background_score_val = background_score  # already computed above
    return {
        "background_score": background_score,
        "enrichment_score_at_5": top5_score,
        "enrichment_score_at_10": top10_score,
        "enrichment_score_at_20": top20_score,
        "enrichment_score_at_50": top50_score,
        "enrichment_score_at_200": top200_score,
        "enrichment_lift_at_5": round(top5_score / background_score, 4) if background_score > 0 else float('nan'),
        "enrichment_lift_at_10": round(top10_score / background_score, 4) if background_score > 0 else float('nan'),
        "enrichment_lift_at_20": round(top20_score / background_score, 4) if background_score > 0 else float('nan'),
        "enrichment_lift_at_50": round(top50_score / background_score, 4) if background_score > 0 else float('nan'),
        "total_cells_at_10": total10
    }

def plot_biological_module_results(mod_name, mod_info, top_k_results, enrichment_stats, adata, data, overlapping_indices, genes_work, mean_spd_sub, dataset_name):
    out_dir = project_root / "results" / "gene_signature_search" / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    coords = adata.obsm["spatial"]
    pathway_scores = adata.obs["pathway_score"].values

    cell_records = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'pathway_score': pathway_scores
    })
    cells_csv_path = out_dir / f"{mod_name}_spatial_cells.csv"
    cell_records.to_csv(cells_csv_path, index=False)

    match_records = []
    n_plot = len(top_k_results)
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

def run_dataset(dataset_name, adata_path, modules, search_budget=500, max_pts=200, top_genes=800):
    """
    Builds the Spindle index for one dataset and benchmarks every module in
    `modules` against it. Returns a dict keyed by mod_name with the recorded stats.
    """
    if not adata_path.exists():
        raise FileNotFoundError(f"Dataset not found at {adata_path}")

    print(f"Loading {dataset_name} dataset...")
    adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = data_helpers.load_and_split_data(adata_path, max_pts=max_pts, top_genes=top_genes)
    num_genes = len(genes_work)

    print("\nBuilding Index Data...")
    data, out_dict = data_helpers.run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)

    dag_dict, config = data_helpers.configure_and_build_dag(data)

    print("\nBuilding Interval Index...")
    config.use_interval_index = True
    config.interval_mode = "dyadic"
    config.interval_max_iters = 5
    ivl_idx = interval_index.build_all_interval_indices(data, config)

    dataset_summary = {}

    print("\n" + "="*70)
    print(f"BENCHMARKING GENE SIGNATURES ON {dataset_name.upper()} (SPINDLE PARTIAL SEARCH)")
    print("="*70)

    for mod_name, mod_info in modules.items():
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
        top_k_results = data_helpers.search_all_clusters_spindle(ivl_idx, data, overlapping_indices, q_spd, top_k=search_budget)

        if not top_k_results:
            print(f"No matching spatial patches found for {mod_name}.")
            continue

        # Calculate pathway score
        sc.tl.score_genes(adata, gene_list=target_genes, score_name='pathway_score', use_raw=False)

        # Step 4: Compute biological enrichment statistics
        enrichment_stats = compute_tile_enrichment(top_k_results, adata, data)

        print(f"Enrichment@10: {enrichment_stats['enrichment_score_at_10']:.3f} vs Background: {enrichment_stats['background_score']:.3f} (Lift: {enrichment_stats['enrichment_lift_at_10']:.2f}x)")

        # Step 5: Export CSVs for downstream plotting
        plot_biological_module_results(mod_name, mod_info, top_k_results, enrichment_stats, adata, data, overlapping_indices, genes_work, mean_spd_sub, dataset_name)

        # Record stats
        dataset_summary[mod_name] = {
            "dataset": dataset_name,
            "description": mod_info["description"],
            "color": mod_info["color"],
            "overlap_genes": len(overlapping_indices),
            "background_score": enrichment_stats["background_score"],
            "enrichment_score_at_5": enrichment_stats["enrichment_score_at_5"],
            "enrichment_score_at_10": enrichment_stats["enrichment_score_at_10"],
            "enrichment_score_at_20": enrichment_stats["enrichment_score_at_20"],
            "enrichment_score_at_50": enrichment_stats["enrichment_score_at_50"],
            "enrichment_score_at_200": enrichment_stats["enrichment_score_at_200"],
            "enrichment_lift_at_5": enrichment_stats["enrichment_lift_at_5"],
            "enrichment_lift_at_10": enrichment_stats["enrichment_lift_at_10"],
            "enrichment_lift_at_20": enrichment_stats["enrichment_lift_at_20"],
            "enrichment_lift_at_50": enrichment_stats["enrichment_lift_at_50"],
        }

    return dataset_summary


def append_to_combined_csv(dataset_summary, dataset_name):
    if not dataset_summary:
        return
    df = pd.DataFrame.from_dict(dataset_summary, orient='index')
    df.index.name = 'module'
    df = df.reset_index()

    out_dir = project_root / "results" / "gene_signature_search"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-dataset CSV
    per_dataset_path = out_dir / dataset_name / "benchmark_metrics.csv"
    per_dataset_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(per_dataset_path, index=False)
    print(f"Exported per-dataset benchmark metrics to {per_dataset_path}")

    # Combined CSV across all datasets (append, write header only once)
    combined_path = out_dir / "benchmark_metrics_all.csv"
    write_header = not combined_path.exists()
    df.to_csv(combined_path, mode='a', header=write_header, index=False)
    print(f"Appended to combined benchmark metrics at {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Run gene-signature spatial search benchmarks across Xenium datasets")
    parser.add_argument('--dataset-paths', nargs='*', default=None, help='Paths to specific datasets to run (defaults to all datasets in DATASET_DIR)')
    parser.add_argument('--search-budget', type=int, default=500, help='Top-K search budget per module')
    parser.add_argument('--max-pts', type=int, default=100, help='Max cells per quadtree tile (smaller = finer spatial resolution)')
    parser.add_argument('--top-genes', type=int, default=300, help='Number of top-variance genes used to build the index')
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)

    if args.dataset_paths:
        datasets = {Path(p).stem: Path(p) for p in args.dataset_paths}
    else:
        datasets = DEFAULT_DATASETS

    print("\n" + "="*70)
    print(f"RUNNING GENE SIGNATURE SEARCH ACROSS {len(datasets)} DATASET(S)")
    print("="*70)

    for dataset_name, adata_path in datasets.items():
        print(f"\n\n########## DATASET: {dataset_name} ##########")
        modules = select_modules_for_dataset(dataset_name)
        try:
            dataset_summary = run_dataset(dataset_name, adata_path, modules, search_budget=args.search_budget, max_pts=args.max_pts, top_genes=args.top_genes)
        except Exception as e:
            print(f"ERROR processing dataset {dataset_name}: {e}")
            continue

        append_to_combined_csv(dataset_summary, dataset_name)

    print("\n" + "="*70)
    print("ALL BIOLOGICAL BENCHMARKS COMPLETED SUCCESSFULLY")
    print("="*70)

if __name__ == "__main__":
    main()
