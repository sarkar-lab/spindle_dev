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

import data_helpers  # type: ignore
import spindle_dev.interval_index as interval_index
from run_logging import RunLogger  # type: ignore

# Folder (relative to project_root/results) where index_datasets.py saves its output.
# Keeping this as a named constant avoids the magic string being silently out of sync
# if the output directory is ever renamed.
INDEXED_RESULTS_SUBDIR = "holdout_validation_indexed"

# Query-length bins: length is now the controlled/primary variable (each bin
# gets exactly --num-queries draws, guaranteeing coverage), not an emergent
# property of round-robin niche/block sampling -- the previous round-robin
# approach left long-query bins completely empty for most datasets, since
# large blocks are rare (~5-6% of all blocks) and the niche-first round-robin
# never sampled deep enough into any one niche's block list to reach them.
# Back to the original 4-bin scheme (<=6/7-12/13-16/>16 genes) this benchmark
# started with; the guaranteed-per-bin-draw mechanism (build_binned_queries)
# still applies, so every bin gets exactly --num-queries draws regardless of
# how wide its range is.
# (label, min_length, max_length_or_None). max_length=None means "as large as
# the dataset's largest available block allows" (the open-ended ">16" bin).
LENGTH_BINS = [
    ('<=6', 4, 6),
    ('7-12', 7, 12),
    ('13-16', 13, 16),
    ('>16', 17, None),
]


def _draw_disjoint_ranges(rng, block_size, target_length, contiguous):
    """Return a list of (start, end) ranges within [0, block_size), summing
    to exactly target_length, either as one contiguous range or two disjoint
    ranges (each >=2 genes). Returns None if no valid placement is found
    (only possible for the non-contiguous case at very small block_size)."""
    if contiguous:
        start = rng.randint(0, block_size - target_length)
        return [(start, start + target_length)]

    if target_length < 4:
        return None  # can't split into two pieces of >=2 genes each
    for _ in range(50):
        len1 = rng.randint(2, target_length - 2)
        len2 = target_length - len1
        a1 = rng.randint(0, block_size - len1)
        b1 = a1 + len1
        a2 = rng.randint(0, block_size - len2)
        b2 = a2 + len2
        if b1 <= a2 or b2 <= a1:
            return [(a1, b1), (a2, b2)] if a1 < a2 else [(a2, b2), (a1, b1)]
    return None


def build_binned_queries(data, query_matrices, queries_per_bin, seed):
    """Build a flat list of fully-specified query dicts, guaranteeing exactly
    ``queries_per_bin`` draws for each bin in LENGTH_BINS, each draw producing
    both a 'Contiguous Random' and 'Non-Contiguous Random' row (so the
    returned list has up to ``2 * queries_per_bin * len(LENGTH_BINS)``
    entries). For each draw: pick a target length uniformly within the bin,
    find every (niche, block) pair large enough to hold that length, and pick
    one at random -- rare bins (e.g. '>30') necessarily reuse the same
    handful of qualifying blocks across draws, since large blocks are scarce
    in every dataset (documented limitation, not a bug).
    """
    rng = random.Random(seed)
    niches_sorted = sorted(int(cid) for cid in data.block_dict.keys())
    all_pairs = [
        (cid, block_index, block_end - block_start)
        for cid in niches_sorted
        for block_index, (block_start, block_end) in enumerate(data.block_dict[cid])
    ]
    if not all_pairs or not query_matrices:
        return []
    max_block_size = max(size for _, _, size in all_pairs)

    queries = []
    tile_cursor = 0
    qid = 0
    for label, lo, hi in LENGTH_BINS:
        bin_hi = hi if hi is not None else max_block_size
        drawn = 0
        attempts = 0
        max_attempts = queries_per_bin * 50
        while drawn < queries_per_bin and attempts < max_attempts:
            attempts += 1
            target_length = rng.randint(lo, min(bin_hi, max_block_size))
            qualifying = [(cid, bidx, sz) for cid, bidx, sz in all_pairs if sz >= target_length]
            if not qualifying:
                continue
            cluster_id, block_index, block_size = rng.choice(qualifying)
            block_start, _ = data.block_dict[cluster_id][block_index]
            perm = data.perm_list[cluster_id]
            block_perm = perm[block_start:block_start + block_size]

            q_spd = query_matrices[tile_cursor % len(query_matrices)]
            tile_id = tile_cursor
            tile_cursor += 1

            drew_any = False
            for case_name, contiguous in [('Contiguous Random', True), ('Non-Contiguous Random', False)]:
                ranges = _draw_disjoint_ranges(rng, block_size, target_length, contiguous)
                if ranges is None:
                    continue
                perm_ivl = []
                for a, b in ranges:
                    perm_ivl.extend(block_perm[a:b])
                queries.append({
                    'id': qid,
                    'tile_id': tile_id,
                    'q_spd': q_spd,
                    'cluster_id': cluster_id,
                    'block_index': block_index,
                    'block_size': block_size,
                    'case_name': case_name,
                    'perm_ivl': perm_ivl,
                    'length_bin': label,
                })
                qid += 1
                drew_any = True
            if drew_any:
                drawn += 1

        if drawn < queries_per_bin:
            print(f"  WARNING: only secured {drawn}/{queries_per_bin} queries for length bin '{label}' "
                  f"(no qualifying block found after {max_attempts} attempts).")

    return queries


def main():
    parser = argparse.ArgumentParser(description="Spindle Interval Index Partial Search Benchmark")
    parser.add_argument("--top-k", type=int, default=50, help="Candidate pool size retrieved from interval index")
    parser.add_argument("--num-queries", type=int, default=50,
                         help="Number of query draws PER LENGTH BIN (see LENGTH_BINS) -- each draw "
                              "produces both a contiguous and non-contiguous test row, so total rows "
                              "per dataset/seed = 2 * num_queries * len(LENGTH_BINS).")
    parser.add_argument("--dataset-paths", nargs="*", default=None, help="Paths to specific datasets to benchmark")
    parser.add_argument('--seed', type=int, default=42,
                         help='Random seed controlling query round-robin assignment and interval sampling.')
    parser.add_argument('--n-holdout', type=int, default=None,
                         help='Fixed holdout tile count used for this dataset\'s split '
                              '(informational -- recorded in the run log; only load-bearing via the '
                              'fallback load_and_split_data path when no pre-built index pickle exists).')
    parser.add_argument('--train-test-ratio', type=float, default=0.05,
                         help='Holdout fraction used for this dataset\'s split (informational, see --n-holdout; '
                              'only load-bearing via the fallback load_and_split_data path).')
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.dataset_paths:
        datasets = {Path(p).stem: Path(p) for p in args.dataset_paths}
    else:
        datasets = {
            "breast_cancer": project_root / "dataset" / "xenium_human_breast_cancer.h5ad",
            "kidney_nondiseased": project_root / "dataset" / "xenium_human_kidney_nondiseased.h5ad",
            "lymph_node": project_root / "dataset" / "xenium_human_lymph_node.h5ad",
            "lymph_node_5k": project_root / "dataset" / "xenium_human_lymph_node_5k.h5ad",
            "lung_cancer": project_root / "dataset" / "xenium_human_lung_cancer.h5ad",
            "skin_melanoma": project_root / "dataset" / "xenium_human_skin_melanoma.h5ad",
            "pancreatic_cancer": project_root / "dataset" / "xenium_human_pancreatic_cancer.h5ad",
            "brain_cancer": project_root / "dataset" / "xenium_human_brain_cancer.h5ad",
        }
    
    search_budget = args.top_k

    all_dfs = []

    run_log_dir = project_root / "results" / "run_logs"

    for dataset_name, adata_path in datasets.items():
        print(f"\n\n{'='*60}")
        print(f"STARTING BENCHMARK FOR {dataset_name.upper()}")
        print(f"{'='*60}")

        with RunLogger(dataset_name=dataset_name, stage="partial_panel_search", out_dir=run_log_dir,
                       seed=args.seed, n_holdout=args.n_holdout, train_test_ratio=args.train_test_ratio):
            df = _run_one_dataset(dataset_name, adata_path, args, search_budget, project_root)
        all_dfs.append(df)

    _write_overall_outputs(all_dfs, project_root)


def _run_one_dataset(dataset_name, adata_path, args, search_budget, project_root):
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
        adata, genes_work, train_tiles, train_tile_covs, test_tiles, test_tile_covs, train_idx, test_idx = (
            data_helpers.load_and_split_data(adata_path, test_ratio=args.train_test_ratio, seed=args.seed)
        )

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

    # Length is the controlled/primary variable: draw exactly args.num_queries
    # queries PER length bin (see LENGTH_BINS), each producing both a
    # contiguous and non-contiguous test row. This guarantees coverage of
    # every bin, including rare long-query ones that pure round-robin
    # niche/block sampling left completely empty (large blocks are only
    # ~5-6% of all blocks in every dataset). No niche prediction involved --
    # replaces the old search.assign_clusters_to_new_spds(...) KNN-vote (a
    # guess: held-out tiles were never clustered, and the search doesn't need
    # a predicted niche -- see holdout_validation.py::compute_ground_truth,
    # which loops over every niche's permutation per query with no
    # prediction step at all).
    queries = build_binned_queries(data, query_matrices, args.num_queries, args.seed)

    print(f"-> Secured {len(queries)} query rows across {len(LENGTH_BINS)} length bins "
          f"({args.num_queries} draws/bin, both test cases per draw).")

    benchmark_results_log = data_helpers.run_benchmark_suite(
        queries, data, ivl_idx, search_budget, config
    )

    df = pd.DataFrame(benchmark_results_log)
    df['Dataset'] = dataset_name

    results_dir = project_root / "results" / "partial_panel_search" / dataset_name
    results_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(results_dir / "benchmark_interval_metrics.csv", index=False)

    # ---------------------------------------------------------
    # Performance Bins, File Writing, and Visual Plots
    # ---------------------------------------------------------
    # Length_Bin is already an exact column (assigned during query
    # construction, not inferred post-hoc from Query_Length) -- every bin in
    # LENGTH_BINS is guaranteed to be represented (barring the WARNING case
    # where a dataset has zero qualifying blocks for some bin).
    bin_labels = [label for label, _, _ in LENGTH_BINS]
    data_helpers.generate_performance_report(df, results_dir, bin_labels)
    # Plotting disabled in CSV-only mode

    return df


def _write_overall_outputs(all_dfs, project_root):
    if not all_dfs:
        return

    new_df = pd.concat(all_dfs, ignore_index=True)
    overall_results_dir = project_root / "results" / "partial_panel_search"
    overall_results_dir.mkdir(parents=True, exist_ok=True)

    # ── overall_benchmark_metrics.csv  (upsert: keep existing rows for OTHER datasets) ──
    overall_metrics_path = overall_results_dir / "overall_benchmark_metrics.csv"
    if overall_metrics_path.exists():
        try:
            existing_metrics = pd.read_csv(overall_metrics_path)
            new_datasets = new_df['Dataset'].unique()
            combined_df = pd.concat(
                [existing_metrics[~existing_metrics['Dataset'].isin(new_datasets)], new_df],
                ignore_index=True
            )
        except Exception as e:
            print(f"Note: Could not merge existing overall metrics ({e}); overwriting.")
            combined_df = new_df
    else:
        combined_df = new_df
    combined_df.to_csv(overall_metrics_path, index=False)

    # ── benchmark_summary.csv  (upsert) ──
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
            'recall_at_eps_0.05': round(grp['recall_at_eps_0.05'].mean(), 4),
            'recall_at_eps_0.1': round(grp['recall_at_eps_0.1'].mean(), 4),
            'overlap_at_eps_0.1': round(grp['overlap_at_eps_0.1'].mean(), 4),
            'overlap_at_eps_0.25': round(grp['overlap_at_eps_0.25'].mean(), 4),
        })
    summary_path = overall_results_dir / "benchmark_summary.csv"
    df_summary = pd.DataFrame(summary_records)
    if summary_path.exists():
        try:
            existing_summary = pd.read_csv(summary_path)
            new_datasets = df_summary['Dataset'].unique()
            df_summary = pd.concat(
                [existing_summary[~existing_summary['Dataset'].isin(new_datasets)], df_summary],
                ignore_index=True
            )
        except Exception as e:
            print(f"Note: Could not merge existing benchmark summary ({e}); overwriting.")
    df_summary.to_csv(summary_path, index=False)
    # Plotting disabled in CSV-only mode


if __name__ == "__main__":
    main()
