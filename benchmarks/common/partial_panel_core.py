"""Partial-panel (gene-subset) search: the shared core, and one (dataset, seed) run (E11).

Library: ``search_all_clusters_spindle`` (Stage 1: score every tile of every
niche through the dyadic interval index, combining the pieces of the query's
gene set) and ``run_benchmark_suite`` (Stage 2 exact re-rank of the top-k pool,
brute-force reference, Recall@eps / Overlap@eps). gene_signature_search.py
reuses the Stage-1 search.

As a script it loads an already-built index (results/indexes/, from
build_indexes.py), builds or loads that index's dyadic interval index, draws
--num-queries gene-subset queries per length bin from the held-out tiles, and
writes ``<out-dir>/<dataset>_query_metrics.csv``.

Retired pipeline (old Fig 4; its driver partial_panel_search.py was removed in the Fig 4 session, replaced by
partial_search_final.py). Kept only because gene_signature_search.py (Fig 6F-G) imports the Stage-1 search;
delete with interval_index.py when Fig 6 is ported to tiers (Part 4).
"""

import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parents[1]
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import spindle_dev.interval_index as interval_index

INDEX_DIR = project_root / "results" / "indexes"


def extract_query_matrices(test_tile_covs):
    query_matrices = []
    for q_dict in test_tile_covs:
        if isinstance(q_dict, dict):
            query_matrices.append(q_dict.get('cov', q_dict.get('matrix', q_dict)))
        else:
            query_matrices.append(q_dict)
    return query_matrices


def search_all_clusters_spindle(interval_index_obj, data, perm_ivl, q_spd, top_k=20):
    genes_set = set(perm_ivl)
    valid_len = len(perm_ivl)
    total_scale = np.sqrt(valid_len) if valid_len > 1 else 1.0
    
    all_cluster_ranked = []
    
    for c_id in set(data.labels):
        piece_scores = []
        piece_scales = []
        
        for b_idx, (block_start, block_end) in enumerate(data.block_dict.get(c_id, [])):
            block_perm = data.perm_list[c_id][block_start:block_end]
            
            indices = [i for i, g in enumerate(block_perm) if g in genes_set]
            if not indices:
                continue
                
            ranges_c = []
            start = indices[0]
            prev = indices[0]
            for i in indices[1:]:
                if i == prev + 1:
                    prev = i
                else:
                    ranges_c.append((start, prev + 1))
                    start = i
                    prev = i
            ranges_c.append((start, prev + 1))
            
            q_block_c = q_spd[np.ix_(block_perm, block_perm)]
            
            for a, b in ranges_c:
                pieces = interval_index.decompose_to_dyadic(a, b)
                for pa, pb in pieces:
                    q_sub = q_block_c[pa:pb, pa:pb]
                    p = pb - pa
                    scale = np.sqrt(p) if p > 1 else 1.0
                    
                    piece_results = interval_index.query_interval_index(
                        interval_index_obj, c_id, b_idx, (pa, pb), q_sub, top_k=None
                    )
                    
                    if not piece_results:
                        piece_scores.append({})
                        piece_scales.append(scale)
                        continue
                        
                    scores = {}
                    for dist, members in piece_results:
                        for m in members:
                            if m not in scores or dist < scores[m]:
                                scores[m] = dist
                    piece_scores.append(scores)
                    piece_scales.append(scale)
                    
        if not piece_scores:
            continue
            
        common = set(piece_scores[0].keys())
        for ps in piece_scores[1:]:
            common &= ps.keys()
            if not common:
                break
                
        if not common:
            continue
            
        for tid in common:
            total_sq = 0.0
            for i, ps in enumerate(piece_scores):
                unscaled = ps[tid] * piece_scales[i]
                total_sq += unscaled ** 2
            combined_dist = np.sqrt(total_sq) / total_scale
            all_cluster_ranked.append((combined_dist, tid))
            
    all_cluster_ranked = sorted(all_cluster_ranked, key=lambda x: x[0])
    return [(dist, [tid]) for dist, tid in all_cluster_ranked[:top_k]]


def run_benchmark_suite(queries, data, ivl_idx, search_budget, config):
    """Run brute-force + Spindle interval search + metrics for a list of
    fully pre-built query specs (see build_binned_queries).

    Unlike the original version of this function, gene-range construction
    (choosing a target length, picking a qualifying block, drawing the
    contiguous/non-contiguous sub-range) happens entirely upstream, in the
    caller -- each ``queries`` entry already carries a concrete ``perm_ivl``.
    This lets the caller *guarantee* coverage of specific query-length bins
    (including rare long-query bins that pure round-robin block sampling
    would leave empty) rather than discovering whatever length distribution
    falls out of round-robin niche/block sampling after the fact.
    """
    benchmark_results_log = []

    all_spd_matrices = np.asarray(data.spd_matrices)
    all_spd_ids = np.asarray(data.spd_ids)
    sid_to_idx = {int(all_spd_ids[k]): k for k in range(len(all_spd_ids))}
    median_block_epsilon = (
        float(np.median(list(config.epsilon_dict.values()))) if config.epsilon_dict else float('nan')
    )
    # Partial-gene-interval distances are far less discriminative than the
    # full-block/whole-matrix distances holdout_core.py's
    # recall_at_eps_{0.1,0.5}/overlap_at_eps_{0.5,1.0} convention was tuned
    # for -- a fixed absolute radius on a handful of genes sweeps in a huge
    # fraction of the dataset (verified: at frac=0.5 the true-near set already
    # covers 7-40% of all training tiles, vs. Spindle's fixed top_k-sized
    # retrieval pool), so overlap collapses on band size alone rather than
    # search quality. Use smaller fractions here so the near-set stays in a
    # range a bounded retrieval pool can meaningfully cover.
    eps_fracs = [0.05, 0.1, 0.25]

    for q_info in tqdm(queries, desc="Benchmarking"):
        q_spd = q_info['q_spd']
        perm_ivl = q_info['perm_ivl']
        valid_len = len(perm_ivl)
        block_size = q_info['block_size']

        # The block-level epsilon was calibrated (choose_adaptive_epsilons)
        # against full-block distances (fixed size = block_size, both
        # normalized by sqrt(block_size)). Interval queries here are a
        # sub-range of that block (valid_len <= block_size), and their
        # distance is normalized by sqrt(valid_len) instead -- so the raw
        # block epsilon must be rescaled by sqrt(valid_len / block_size)
        # to stay on the same distance scale as what's actually compared,
        # otherwise short intervals get an artificially loose band and
        # long intervals an artificially tight one.
        query_band_epsilon = median_block_epsilon * np.sqrt(valid_len / block_size)

        # =========================================================
        # VECTORIZED BRUTE FORCE
        # =========================================================
        bf_start_time = time.perf_counter()
        q_sub = q_spd[np.ix_(perm_ivl, perm_ivl)]
        q_log = interval_index._log_spd(q_sub)
        scale = np.sqrt(valid_len) if valid_len > 1 else 1.0

        t_subs = all_spd_matrices[:, perm_ivl, :][:, :, perm_ivl]
        t_logs = np.array([interval_index._log_spd(t_sub) for t_sub in t_subs])

        diffs = t_logs - q_log
        dists = np.linalg.norm(diffs, axis=(1, 2)) / scale
        true_min_dist = np.min(dists)

        rounded_dists = np.round(dists, 5)
        unique_dists = np.sort(np.unique(rounded_dists))
        exact_partial_map = {int(all_spd_ids[i]): rounded_dists[i] for i in range(len(dists))}
        bf_time_ms = (time.perf_counter() - bf_start_time) * 1000
        # =========================================================
        # SPINDLE INTERVAL SEARCH
        # =========================================================
        spindle_start = time.perf_counter()
        results = search_all_clusters_spindle(
            ivl_idx, data, perm_ivl, q_spd, top_k=search_budget
        )
        dag_time_ms = (time.perf_counter() - spindle_start) * 1000

        flat_results = []
        for err, ids in results:
            for i in ids:
                flat_results.append((err, i))

        retrieved_sids = [sid for _, sid in flat_results]

        # Real exact re-ranking computation on retrieved candidate spots
        rerank_start = time.perf_counter()
        rerank_scores = []
        for sid in retrieved_sids:
            idx_k = sid_to_idx.get(int(sid))
            if idx_k is not None:
                t_sub = all_spd_matrices[idx_k][np.ix_(perm_ivl, perm_ivl)]
                t_log = interval_index._log_spd(t_sub)
                diff = t_log - q_log
                dist = np.linalg.norm(diff, ord='fro') / scale
                rerank_scores.append((dist, sid))

        rerank_scores.sort(key=lambda x: x[0])
        rerank_time_ms = (time.perf_counter() - rerank_start) * 1000
        spindle_time_ms = dag_time_ms + rerank_time_ms

        spindle_retrieved_ids_all = [sid for _, sid in rerank_scores]

        spindle_best_rank = -1
        spindle_best_partial_dist = float('inf')

        if spindle_retrieved_ids_all:
            spindle_top1_id = spindle_retrieved_ids_all[0]
            spindle_best_partial_dist = exact_partial_map.get(spindle_top1_id, float('inf'))
            if spindle_best_partial_dist != float('inf'):
                spindle_best_rank = np.searchsorted(unique_dists, np.round(spindle_best_partial_dist, 5)) + 1

        record = {
            'Case': q_info['case_name'],
            'Tile': q_info['tile_id'],
            'Niche': q_info['cluster_id'],
            'Block_Index': q_info['block_index'],
            'Length_Bin': q_info['length_bin'],
            'Query_Length': valid_len,
            'rank': spindle_best_rank,
            'dist_gap': (spindle_best_partial_dist - true_min_dist) if spindle_best_rank != -1 else float('inf'),
            'spindle_time_ms': round(spindle_time_ms, 4),
            'brute_force_time_ms': round(bf_time_ms, 4),
            'speedup': round(bf_time_ms / spindle_time_ms, 2) if spindle_time_ms > 0 else np.nan,
            'retrieved': len(flat_results),
        }

        # Recall@epsilon / Overlap@epsilon: distance-tolerance-based hit
        # metrics, matching holdout_core.py's convention. There is
        # no single "true niche" for these queries (they're sampled across
        # every niche/block), so the tolerance band is anchored on the
        # dataset-wide median epsilon, rescaled per query to this interval's
        # length (see query_band_epsilon above).
        for frac in eps_fracs:
            band = frac * query_band_epsilon
            recall_eps = 1 if (spindle_best_rank != -1 and
                (spindle_best_partial_dist - true_min_dist) <= band) else 0
            true_near_ids = {sid for sid, d in exact_partial_map.items() if (d - true_min_dist) <= band}
            spindle_near_ids = {
                sid for sid in spindle_retrieved_ids_all
                if (exact_partial_map.get(sid, float('inf')) - true_min_dist) <= band
            }
            denom_eps = max(1, len(true_near_ids))
            overlap_eps = len(true_near_ids.intersection(spindle_near_ids)) / float(denom_eps)

            record[f'recall_at_eps_{frac}'] = recall_eps
            record[f'overlap_at_eps_{frac}'] = round(overlap_eps, 4)

        benchmark_results_log.append(record)

    return benchmark_results_log


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


def load_index(dataset_name):
    """Index + held-out covariances, with the training covariances restored into ``data``."""
    with open(INDEX_DIR / f"{dataset_name}_spindle_index.pkl", 'rb') as f:
        saved_data = pickle.load(f)
    with open(INDEX_DIR / f"{dataset_name}_raw_covariances.pkl", 'rb') as cf:
        covs_data = pickle.load(cf)
    data = saved_data['data']
    # build_indexes.py strips data.spd_matrices before saving; train_tile_covs is in
    # the same order as data.spd_ids.
    data.spd_matrices = [t if not isinstance(t, dict) else t.get('cov', t) for t in covs_data['train_tile_covs']]
    assert len(data.spd_matrices) == len(data.spd_ids), (
        f"SPD matrix/ID count mismatch for {dataset_name}: {len(data.spd_matrices)} matrices vs "
        f"{len(data.spd_ids)} IDs -- index and covariance files are from different runs."
    )
    return data, saved_data['config'], covs_data['test_tile_covs']


def load_or_build_interval_index(dataset_name, data, config):
    ivl_file = INDEX_DIR / f"{dataset_name}_interval_index.pkl"
    if ivl_file.exists():
        print(f"Loading pre-built dyadic interval index from {ivl_file.name}...")
        with open(ivl_file, 'rb') as f:
            return pickle.load(f)
    print("Building dyadic interval index...")
    config.use_interval_index = True
    config.interval_mode = "dyadic"
    config.interval_max_iters = 5
    ivl_idx = interval_index.build_all_interval_indices(data, config)
    with open(ivl_file, 'wb') as f:
        pickle.dump(ivl_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {ivl_file.name}")
    return ivl_idx


def main():
    parser = argparse.ArgumentParser(description="Partial-panel search on already-built indexes (one seed).")
    parser.add_argument("--dataset-paths", nargs="+", required=True,
                        help="h5ad paths; the stem names the index in results/indexes/")
    parser.add_argument("--out-dir", type=Path, required=True, help="Writes <out-dir>/<dataset>_query_metrics.csv")
    parser.add_argument("--top-k", type=int, default=50, help="Stage-1 candidate pool size for the exact re-rank")
    parser.add_argument("--num-queries", type=int, default=50,
                        help="Query draws PER LENGTH BIN (see LENGTH_BINS); each draw gives a contiguous and "
                             "a non-contiguous row, so rows per dataset = 2 * num_queries * len(LENGTH_BINS).")
    parser.add_argument("--seed", type=int, required=True, help="Seed for query sampling.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for ds_path in args.dataset_paths:
        dataset_name = Path(ds_path).stem
        print(f"\n{'=' * 60}\nPARTIAL-PANEL SEARCH: {dataset_name}\n{'=' * 60}")
        data, config, test_tile_covs = load_index(dataset_name)
        ivl_idx = load_or_build_interval_index(dataset_name, data, config)

        # Length is the controlled variable: exactly --num-queries draws per bin. No
        # niche prediction -- every niche is searched.
        query_matrices = extract_query_matrices(test_tile_covs)
        queries = build_binned_queries(data, query_matrices, args.num_queries, args.seed)
        print(f"-> {len(queries)} query rows across {len(LENGTH_BINS)} length bins "
              f"({args.num_queries} draws/bin, both cases per draw).")

        df = pd.DataFrame(run_benchmark_suite(queries, data, ivl_idx, args.top_k, config))
        df['Dataset'] = dataset_name
        csv_path = args.out_dir / f"{dataset_name}_query_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"Saved query metrics to {csv_path}")


if __name__ == "__main__":
    main()
