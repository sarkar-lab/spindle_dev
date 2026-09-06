"""
diagnose_recall.py
==================
Instruments the search to reveal exactly WHY recall is low for lymph_node_5k,
and why changing --budget-mult has zero effect.

Run on the server where the index lives:
    python diagnose_recall.py --dataset-path /path/to/xenium_human_lymph_node_5k.h5ad

Or if you just want to inspect a pre-saved index:
    python diagnose_recall.py --index-dir results/holdout_validation_indexed --dataset-name xenium_human_lymph_node_5k
"""

from __future__ import annotations
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np

current_dir = Path(__file__).resolve().parent
src_path = current_dir / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
if str(current_dir / 'benchmarks') not in sys.path:
    sys.path.insert(0, str(current_dir / 'benchmarks'))

import spindle_dev.search as search
from spindle_dev.metrics import build_ultrametrics, spd_tree_feature_matrix


def log_spd(M, eps=1e-6):
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T


def diagnose_index_structure(data, dag_dict, config):
    """
    Part 1: Inspect the DAG structure itself.
    Answers: is the block/node structure sane?
    """
    print("\n" + "=" * 70)
    print("PART 1: INDEX STRUCTURE INSPECTION")
    print("=" * 70)

    print(f"\nTotal tiles indexed (train): {len(data.spd_ids)}")
    print(f"Number of Leiden clusters:   {len(dag_dict)}")
    print(f"Cluster label distribution:  {dict(zip(*np.unique(data.labels, return_counts=True)))}")
    print()

    for cluster_id, handle in sorted(dag_dict.items()):
        nodes = handle.nodes
        sorted_blocks = handle.sorted_blocks
        block_runs = handle.block_runs
        block_to_nodes = handle.block_to_node_indices

        n_in_cluster = int((np.array(data.labels) == cluster_id).sum())
        members_per_node = [len(n.metadata.members) for n in nodes]

        print(f"--- Cluster {cluster_id} ---")
        print(f"  Tiles in cluster:     {n_in_cluster}")
        print(f"  Blocks in index:      {len(sorted_blocks)}")
        print(f"  Total nodes:          {len(nodes)}")
        print(f"  Members/node:         min={min(members_per_node)}, "
              f"max={max(members_per_node)}, "
              f"mean={np.mean(members_per_node):.1f}")

        # Check parent→child member overlap (the intersection bug)
        print(f"\n  Checking parent→child member INTERSECTION (the suspected bug):")
        empty_intersections = 0
        total_parent_child_pairs = 0
        for node in nodes:
            if not node.children:
                continue
            parent_spds = {int(sid) for sid, _ in node.metadata.members}
            for child_id in node.children:
                if child_id < 0 or child_id >= len(nodes):
                    continue
                child = nodes[child_id]
                child_spds = {int(sid) for sid, _ in child.metadata.members}
                intersection = parent_spds & child_spds
                total_parent_child_pairs += 1
                if not intersection:
                    empty_intersections += 1

        if total_parent_child_pairs > 0:
            pct_empty = 100 * empty_intersections / total_parent_child_pairs
            print(f"    Parent-child pairs:         {total_parent_child_pairs}")
            print(f"    Empty intersections:        {empty_intersections} ({pct_empty:.1f}%)")
            if pct_empty > 50:
                print(f"    [!] CRITICAL: >{pct_empty:.0f}% of parent→child transitions")
                print(f"        produce EMPTY valid_spds sets.")
                print(f"        The DFS prunes ALL children on these nodes, making")
                print(f"        budget-mult irrelevant — there's nothing left to explore.")
            else:
                print(f"    [OK] Intersection coverage looks healthy.")
        else:
            print("    No parent-child pairs found (index may be flat).")

        # Check if block runs in handle match data.block_dict
        data_block_runs = data.block_dict.get(cluster_id, [])
        handle_block_runs_list = [block_runs[b] for b in sorted_blocks]
        match = (len(data_block_runs) == len(handle_block_runs_list) and
                 all(a == b for a, b in zip(data_block_runs, handle_block_runs_list)))
        print(f"\n  Block runs: data.block_dict has {len(data_block_runs)} blocks, "
              f"index_handle has {len(handle_block_runs_list)} blocks — "
              f"{'MATCH' if match else '[!] MISMATCH'}")

        print(f"  epsilon for cluster: {config.epsilon_dict.get(cluster_id, 'N/A')}")
        print()


def diagnose_single_query(data, dag_dict, config, query_spd, cluster_id,
                          budget_multiplier=1.0, n_trace=5):
    """
    Part 2: Trace a single query through the DFS step by step.
    Reports exactly where the search dies.
    """
    print("=" * 70)
    print(f"PART 2: SINGLE-QUERY TRACE  (cluster={cluster_id}, budget_mult={budget_multiplier})")
    print("=" * 70)

    handle = dag_dict[cluster_id]
    nodes = handle.nodes
    sorted_blocks = handle.sorted_blocks
    block_runs = handle.block_runs
    block_to_nodes = handle.block_to_node_indices
    index_block_runs = handle.block_runs
    query_block_runs = data.block_dict[cluster_id]
    perm = data.perm_list[cluster_id]

    epsilon = config.epsilon_dict[cluster_id]
    num_blocks = len(sorted_blocks)
    budget = epsilon * num_blocks * budget_multiplier

    # Permute query
    q_perm = query_spd[np.ix_(perm, perm)]

    print(f"\n  Budget:     epsilon={epsilon:.4f} × num_blocks={num_blocks} × mult={budget_multiplier} = {budget:.4f}")
    print(f"  Index has {len(index_block_runs)} block runs, query has {len(query_block_runs)} block runs")
    print(f"  Block runs MATCH: {query_block_runs == [index_block_runs[b] for b in sorted_blocks]}")

    # Check the block count comparison (the `len(query_block_runs) == len(index_block_runs)` branch)
    n_query_blocks = len(query_block_runs)
    n_index_blocks = len(index_block_runs)
    print(f"\n  len(query_block_runs)={n_query_blocks}, len(index_block_runs)={n_index_blocks}")
    if n_query_blocks == n_index_blocks:
        print(f"  --> Branch: 'equal lengths' → used_blocks = all sorted_blocks ✓")
        used_blocks = list(sorted_blocks)
    elif n_query_blocks > n_index_blocks:
        print(f"  --> Branch: 'query LONGER than index' → returns EMPTY (bug!)")
        return
    else:
        print(f"  --> Branch: 'query shorter' → determine_active_blocks()")
        used_blocks = search.determine_active_blocks(query_block_runs, [index_block_runs[b] for b in sorted_blocks])

    # Check contiguity
    non_contiguous = any(used_blocks[i] + 1 != used_blocks[i + 1]
                         for i in range(len(used_blocks) - 1))
    if non_contiguous:
        print(f"  [!] NON-CONTIGUOUS used_blocks → search returns EMPTY immediately!")
        return
    print(f"  used_blocks contiguous: OK ({len(used_blocks)} blocks)")

    # Extract query blocks
    q_blocks = []
    for i, b in enumerate(used_blocks):
        s, e = query_block_runs[i]
        q_blocks.append(q_perm[s:e, s:e])
    q_blocks_log = [log_spd(qb) for qb in q_blocks]
    print(f"  Query block sizes: {[qb.shape[0] for qb in q_blocks]}")

    # Trace first-layer nodes
    first_block = used_blocks[0]
    first_nodes = block_to_nodes.get(first_block, [])
    print(f"\n  First block ({first_block}): {len(first_nodes)} candidate start nodes")

    traced = 0
    for node_idx in first_nodes[:n_trace]:
        node = nodes[node_idx]
        p = node.metadata.mean.shape[0]
        diff = q_blocks_log[0] - node.metadata.mean
        dist = float(np.linalg.norm(diff, 'fro') / np.sqrt(p))
        members_here = {int(sid) for sid, _ in node.metadata.members}
        print(f"    Start node {node_idx}: dist={dist:.4f} (budget={budget:.4f}), "
              f"members={len(members_here)}, children={len(node.children)}")

        if dist > budget:
            print(f"      --> PRUNED at first layer (dist > budget)")
            continue

        # Trace one level deeper
        next_block = used_blocks[1] if len(used_blocks) > 1 else None
        if next_block is None:
            print(f"      --> Single-block index, leaf reached")
            continue

        valid_now = members_here
        n_empty = 0
        for child_id in node.children:
            if child_id < 0 or child_id >= len(nodes):
                continue
            child = nodes[child_id]
            if child.block_index != next_block:
                continue
            child_spds = {int(sid) for sid, _ in child.metadata.members}
            intersection = valid_now & child_spds
            if not intersection:
                n_empty += 1
        total_children = len(node.children)
        print(f"      Layer-1 children: {total_children} total, "
              f"{n_empty} have EMPTY intersection with parent → skipped by DFS")
        if n_empty == total_children:
            print(f"      [!] ALL children pruned by intersection! DFS finds nothing from this start.")

        traced += 1
        if traced >= n_trace:
            break

    print()


def main():
    parser = argparse.ArgumentParser(description="Diagnose Spindle recall drop")
    parser.add_argument("--index-dir", type=str,
                        default="results/holdout_validation_indexed",
                        help="Directory containing saved index .pkl files")
    parser.add_argument("--dataset-name", type=str,
                        default="xenium_human_lymph_node_5k",
                        help="Dataset stem name (matches the .pkl filename)")
    parser.add_argument("--budget-mult", type=float, default=1.0)
    parser.add_argument("--n-queries", type=int, default=5,
                        help="Number of test queries to trace in Part 2")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    idx_path = index_dir / f"{args.dataset_name}_spindle_index.pkl"
    cov_path = index_dir / f"{args.dataset_name}_raw_covariances.pkl"

    if not idx_path.exists():
        print(f"[ERROR] Index file not found: {idx_path}")
        sys.exit(1)

    print(f"Loading index from {idx_path.name}...")
    with open(idx_path, 'rb') as f:
        saved = pickle.load(f)
    data = saved['data']
    dag_dict = saved['dag_dict']
    config = saved['config']

    # Part 1: structural inspection
    diagnose_index_structure(data, dag_dict, config)

    # Part 2: trace a few test queries if covariances are available
    if not cov_path.exists():
        print(f"[Skipping Part 2] Covariance file not found: {cov_path}")
        return

    print(f"\nLoading covariances from {cov_path.name}...")
    with open(cov_path, 'rb') as f:
        covs = pickle.load(f)

    test_tile_covs = covs['test_tile_covs'][:args.n_queries]
    query_matrices = [
        t.get('cov', t) if isinstance(t, dict) else t
        for t in test_tile_covs
    ]

    print(f"\nAssigning {len(query_matrices)} test queries to clusters...")
    predicted_clusters = search.assign_clusters_to_new_spds(query_matrices, data)

    for j, (q_spd, cluster_id) in enumerate(zip(query_matrices, predicted_clusters)):
        cluster_id = int(cluster_id)
        print(f"\n--- Query {j} → assigned to cluster {cluster_id} ---")
        diagnose_single_query(data, dag_dict, config, q_spd, cluster_id,
                              budget_multiplier=args.budget_mult)


if __name__ == "__main__":
    main()
