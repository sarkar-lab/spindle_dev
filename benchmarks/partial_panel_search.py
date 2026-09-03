import sys
import time
from pathlib import Path
import random
import pickle
import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

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

# Folder (relative to project_root/results) where index_datasets.py saves its output.
# Keeping this as a named constant avoids the magic string being silently out of sync
# if the output directory is ever renamed.
INDEXED_RESULTS_SUBDIR = "holdout_validation_indexed"




def main():
    parser = argparse.ArgumentParser(description="Spindle Interval Index Partial Search Benchmark")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate pool size retrieved from interval index")
    parser.add_argument("--num-queries", type=int, default=5, help="Number of query tiles per dataset")
    parser.add_argument("--dataset-paths", nargs="*", default=None, help="Paths to specific datasets to benchmark")
    args = parser.parse_args()

    np.random.seed(42)
    random.seed(42)
    if args.dataset_paths:
        datasets = {Path(p).stem: Path(p) for p in args.dataset_paths}
    else:
        datasets = {
            "breast_cancer": project_root / "dataset" / "xenium_human_breast_cancer.h5ad",
            "kidney_nondiseased": project_root / "dataset" / "xenium_human_kidney_nondiseased.h5ad",
            "lymph_node": project_root / "dataset" / "xenium_human_lymph_node.h5ad",
            "lung_cancer": project_root / "dataset" / "xenium_human_lung_cancer.h5ad",
            "skin_melanoma": project_root / "dataset" / "xenium_human_skin_melanoma.h5ad",
            "pancreatic_cancer": project_root / "dataset" / "xenium_human_pancreatic_cancer.h5ad",
        }
    
    test_cases = [
        ('Contiguous Random', 'contiguous'),
        ('Non-Contiguous Random', 'non_contiguous')
    ]
    
    search_budget = args.top_k
    test_iterations_per_query = 5
    
    all_dfs = []
    
    for dataset_name, adata_path in datasets.items():
        print(f"\n\n{'='*60}")
        print(f"STARTING BENCHMARK FOR {dataset_name.upper()}")
        print(f"{'='*60}")
        
        indexed_file = project_root / "results" / INDEXED_RESULTS_SUBDIR / f"{dataset_name}_spindle_index.pkl"
        covs_file = project_root / "results" / INDEXED_RESULTS_SUBDIR / f"{dataset_name}_raw_covariances.pkl"
        if indexed_file.exists() and covs_file.exists():
            print(f"Loading pre-indexed Spindle data from {indexed_file.name} and {covs_file.name}...")
            with open(indexed_file, 'rb') as f:
                saved_data = pickle.load(f)
            with open(covs_file, 'rb') as cf:
                covs_data = pickle.load(cf)
            data = saved_data['data']
            config = saved_data['config']
            test_tile_covs = covs_data['test_tile_covs']
            # Restore raw covariance matrices that were stripped before saving (index_datasets.py
            # clears data.spd_matrices to reduce file size).  train_tile_covs is ordered
            # identically to the local train-tile list built during ProcessedData construction,
            # which is the same ordering as data.spd_ids.  The assertion below guards against any
            # future ordering drift between the two files.
            restored = [t if not isinstance(t, dict) else t.get('cov', t) for t in covs_data['train_tile_covs']]
            data.spd_matrices = restored
            assert len(data.spd_matrices) == len(data.spd_ids), (
                f"SPD matrix/ID count mismatch after restoration for {dataset_name}: "
                f"{len(data.spd_matrices)} matrices vs {len(data.spd_ids)} IDs. "
                "The index file and covariance file may be from different runs."
            )
        else:
            print("Preparing the Index Base Dataset...")
            adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = data_helpers.load_and_split_data(adata_path)
            
            print("\nBuilding Standard Index Data...")
            data, out_dict = data_helpers.run_index(train_tiles, train_tile_covs, genes_work, adata, resolution=0.2, min_final_size=15)
            
            dag_dict, config = data_helpers.configure_and_build_dag(data)
        
        ivl_file = project_root / "results" / INDEXED_RESULTS_SUBDIR / f"{dataset_name}_interval_index.pkl"
        if ivl_file.exists():
            print(f"Loading pre-built Dyadic Interval Index from {ivl_file.name}...")
            with open(ivl_file, 'rb') as f:
                ivl_idx = pickle.load(f)
        else:
            print("\nBuilding Dyadic Interval Index...")
            config.use_interval_index = True
            config.interval_mode = "dyadic"
            config.interval_max_iters = 5
            ivl_idx = interval_index.build_all_interval_indices(data, config)
            print(f"Saving pre-built Dyadic Interval Index to {ivl_file.name}...")
            with open(ivl_file, 'wb') as f:
                pickle.dump(ivl_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print("\nExtracting pristine test queries (unseen tiles)...")
        query_matrices = data_helpers.extract_query_matrices(test_tile_covs)
        predicted_clusters = search.assign_clusters_to_new_spds(query_matrices, data)
        
        queries = []
        for j, (q_spd, cluster_id) in enumerate(zip(query_matrices, predicted_clusters)):
            if len(queries) >= args.num_queries:
                break
            cluster_id = int(cluster_id)
            block_runs = data.block_dict.get(cluster_id, [])
            if len(block_runs) > 0:
                # Rotate block_index round-robin across available blocks so the benchmark
                # samples a representative spread of block sizes / gene orderings rather
                # than always exercising block 0 only.
                block_index = len(queries) % len(block_runs)
                queries.append({
                    'id': j,
                    'q_spd': q_spd,
                    'cluster_id': cluster_id,
                    'block_index': block_index,
                })
                
        print(f"-> Secured {len(queries)} valid test queries.")
        
        benchmark_results_log = data_helpers.run_benchmark_suite(
            queries, data, ivl_idx, search_budget, test_cases, test_iterations_per_query
        )

        df = pd.DataFrame(benchmark_results_log)
        df['Dataset'] = dataset_name
        
        results_dir = project_root / "results" / "partial_panel_search" / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(results_dir / "benchmark_interval_metrics.csv", index=False)
        
        # ---------------------------------------------------------
        # Performance Bins, File Writing, and Visual Plots
        # ---------------------------------------------------------
        bins = [0, 6, 12, 16, 1000]
        labels = ['<=6 genes', '7-12 genes', '13-16 genes', '>16 genes']
        df['Length_Bin'] = pd.cut(df['Query_Length'], bins=bins, labels=labels)
        
        data_helpers.generate_performance_report(df, results_dir, labels)
        # Plotting disabled in CSV-only mode
        
        all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        overall_results_dir = project_root / "results" / "partial_panel_search"
        overall_results_dir.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(overall_results_dir / "overall_benchmark_metrics.csv", index=False)
        
        summary_records = []
        for ds_name, grp in combined_df.groupby('Dataset'):
            mean_sp = round(grp['spindle_time_ms'].mean(), 4)
            mean_bf = round(grp['brute_force_time_ms'].mean(), 4)
            mean_speedup = round(mean_bf / mean_sp, 2) if mean_sp > 0 else np.nan
            summary_records.append({
                'Dataset': ds_name,
                'num_queries': len(grp),
                'mean_spindle_time_ms': mean_sp,
                'mean_brute_force_time_ms': mean_bf,
                'mean_speedup': mean_speedup,
                'recall_at_1': round(grp['recall_at_1'].mean(), 4),
                'overlap_at_5': round(grp['overlap_at_5'].mean(), 4),
                'overlap_at_10': round(grp['overlap_at_10'].mean(), 4),
                'overlap_at_20': round(grp['overlap_at_20'].mean(), 4),
                'overlap_at_50': round(grp['overlap_at_50'].mean(), 4),
            })
        pd.DataFrame(summary_records).to_csv(overall_results_dir / "benchmark_summary.csv", index=False)
        # Plotting disabled in CSV-only mode

if __name__ == "__main__":
    main()
