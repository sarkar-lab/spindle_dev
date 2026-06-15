import sys
import time
from pathlib import Path
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

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




def main():
    np.random.seed(42)
    random.seed(42)
    datasets = {
        # "brain_cancer": project_root.parent / "dataset" / "xenium_human_brain_cancer.h5ad",
        "breast_cancer": project_root.parent / "dataset" / "xenium_human_breast_cancer.h5ad",
        "kidney_nondiseased": project_root.parent / "dataset" / "xenium_human_kidney_nondiseased.h5ad",
        "lymph_node": project_root.parent / "dataset" / "xenium_human_lymph_node.h5ad",
        "lung_cancer": project_root.parent / "dataset" / "xenium_human_lung_cancer.h5ad",
        "skin_melanoma": project_root.parent / "dataset" / "xenium_human_skin_melanoma.h5ad",
        "pancreatic_cancer": project_root.parent / "dataset" / "xenium_human_pancreatic_cancer.h5ad",
        # "lymph_node_5k": project_root.parent / "dataset" / "xenium_human_lymph_node_5k.h5ad"
    }
    
    test_cases = [
        ('Contiguous Random', 'contiguous'),
        ('Non-Contiguous Random', 'non_contiguous')
    ]
    
    search_budget = 15
    test_iterations_per_query = 5
    
    all_dfs = []
    
    for dataset_name, adata_path in datasets.items():
        print(f"\n\n{'='*60}")
        print(f"STARTING BENCHMARK FOR {dataset_name.upper()}")
        print(f"{'='*60}")
        
        print("Preparing the Index Base Dataset...")
        adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = data_helper.load_and_split_data(adata_path)
        
        print("\nBuilding Standard Index Data...")
        data, out_dict = data_helper.run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)
        
        dag_dict, config = data_helper.configure_and_build_dag(data)
        
        print("\nBuilding Dyadic Interval Index...")
        config.use_interval_index = True
        config.interval_mode = "dyadic"
        config.interval_max_iters = 5
        
        ivl_idx = interval_index.build_all_interval_indices(data, config)
        
        print("\nExtracting pristine test queries (unseen tiles)...")
        query_matrices = data_helper.extract_query_matrices(test_tile_covs)
        predicted_clusters = search.assign_clusters_to_new_spds(query_matrices, data)
        
        queries = []
        for j, (q_spd, cluster_id) in enumerate(zip(query_matrices, predicted_clusters)):
            if len(queries) >= 40:
                break
            cluster_id = int(cluster_id)
            block_runs = data.block_dict.get(cluster_id, [])
            if len(block_runs) > 0:
                queries.append({
                    'id': j,
                    'q_spd': q_spd,
                    'cluster_id': cluster_id
                })
                
        print(f"-> Secured {len(queries)} valid test queries.")
        
        benchmark_results_log = data_helper.run_benchmark_suite(
            queries, data, ivl_idx, search_budget, test_cases, test_iterations_per_query
        )

        df = pd.DataFrame(benchmark_results_log)
        df['Dataset'] = dataset_name
        
        results_dir = project_root / "results" / "partial_search" / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(results_dir / "benchmark_interval_metrics.csv", index=False)
        
        # ---------------------------------------------------------
        # Performance Bins, File Writing, and Visual Plots
        # ---------------------------------------------------------
        bins = [0, 6, 12, 16, 1000]
        labels = ['<=6 genes', '7-12 genes', '13-16 genes', '>16 genes']
        df['Length_Bin'] = pd.cut(df['Query_Length'], bins=bins, labels=labels)
        
        data_helper.generate_performance_report(df, results_dir, labels)
        data_helper.generate_visual_plots(df, results_dir, labels)
        
        all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        overall_results_dir = project_root / "results" / "partial_search"
        overall_results_dir.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(overall_results_dir / "overall_benchmark_metrics.csv", index=False)
        data_helper.generate_overall_dataset_plots(combined_df, overall_results_dir)

if __name__ == "__main__":
    main()
