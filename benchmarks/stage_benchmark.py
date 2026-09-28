"""Stage 1 vs Stage 1 + Stage 2 vs Stage 2 only (results/stage_benchmark/).

- Stage 1: ``spindle_dev.search.search_top_c`` over every niche (holdout_core.perform_search's
  budgets) returns the c best tiles by DAG path score -- the sum over blocks of the distance
  from the query block's log to the node mean (``dag_path_dist`` in
  HS_notebooks/dag_vs_true_le_distance.ipynb). Needs only the DAG index.
- Stage 1 + 2: the same c tiles re-ranked by exact block-wise LE from every tile's stored
  float32 diagonal-block logs (vectorised per niche and block). Needs DAG + block logs.
- Stage 2 only: nothing precomputed -- every training tile's raw diagonal blocks (stored in
  its niche's layout) are logged by eigendecomposition for every query (batched per niche and
  block), then the exact block LE ranks all tiles. Needs the raw diagonal blocks only. Its
  cost does not depend on the query, so it is timed on the first ``--stage2-only-queries``
  queries only (default 10; NaN for the others).

Per query and c in C_GRID, scored against the cached block-LE ground truth
(holdout_core definitions, band = epsilon of the true best niche): Recall@eps (top-1 within
f*eps of the best distance), Overlap@eps (fraction of the eps-near set among the returned
tiles) and Recall@K (K <= c). Tiles sharing a leaf share a Stage-1 score; ties are resolved
pessimistically (a tile counts only once its whole tie group is inside the prefix).
Time per query on one thread, every query-side log included. Storage in MB (float32 upper
triangles): DAG (the index pickle's index_size_mb), block logs, raw diagonal blocks.

Unit mode (seed-73 production index in results/indexes/):
    python benchmarks/stage_benchmark.py --dataset-path <ds.h5ad> [--max-queries N]
Aggregate mode (cheap):
    python benchmarks/stage_benchmark.py --aggregate
"""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
for p in (current_dir, project_root / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import spindle_dev.search as search
from holdout_core import (EPSILON_TOLERANCE_FRACTIONS, _niche_scale_factor, extract_query_matrices,
                          load_index, load_or_compute_ground_truth, log_spd)

OUT_ROOT = project_root / "results" / "stage_benchmark"
C_GRID = [10, 20, 50, 100, 200, 400, 1000]
TOP_K = [1, 5, 10, 20, 50]
LOG_EPS = 1e-6  # eigenvalue clamp of holdout_core.log_spd, which the ground truth uses
F32 = 4
MB = 1024 * 1024


# =====================================================================
# Metrics
# =====================================================================

class Ranked:
    """A returned list; ``pos[idx]`` = 1-based end of idx's tie group (pessimistic position)."""

    def __init__(self, ids, scores):
        self.ids = ids
        ends, n, i = [0] * len(ids), len(ids), 0
        while i < n:
            j = i
            while j + 1 < n and scores[j + 1] == scores[i]:
                j += 1
            for t in range(i, j + 1):
                ends[t] = j + 1
            i = j + 1
        self.pos = dict(zip(ids, ends))

    def prefix(self, n):
        return {i for i, p in self.pos.items() if p <= n}


def near_sets(gt, band):
    closest = gt['closest_dist']
    return {f: {i for i, d in gt['dist_dict'].items() if d - closest <= f * band} for f in EPSILON_TOLERANCE_FRACTIONS}


def quality(rl, gt, band, near, ks):
    """Recall@eps / Overlap@eps (holdout_core definitions) and Recall@K of one returned list."""
    dd, order, closest = gt['dist_dict'], gt['true_order'], gt['closest_dist']
    first = rl.prefix(1)
    top1 = next(iter(first)) if len(first) == 1 else None  # a tied first place is not one answer
    r = {'n_returned': len(rl.ids), 'best_rank': order.index(top1) + 1 if top1 is not None else -1}
    returned = set(rl.ids)
    for f in EPSILON_TOLERANCE_FRACTIONS:
        r[f'recall_at_eps_{f}'] = int(top1 is not None and dd[top1] - closest <= f * band)
        r[f'overlap_at_eps_{f}'] = len(near[f] & returned) / len(near[f])
    for K in ks:
        r[f'recall_at_{K}'] = len(rl.prefix(K) & set(order[:K])) / K if len(rl.ids) >= K else np.nan
    return r


def batched_log_spd(S, eps=LOG_EPS):
    """log_spd of a stack of symmetric matrices (n, p, p), as holdout_core.log_spd per matrix."""
    S = 0.5 * (S + np.swapaxes(S, 1, 2))
    w, V = np.linalg.eigh(S)
    return np.einsum('nij,nj,nkj->nik', V, np.log(np.maximum(w, eps)), V)


# =====================================================================
# Unit run
# =====================================================================

def run_unit(dataset_path, max_queries, stage2_only_queries, budget_mult=1.0):
    name = Path(dataset_path).stem
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 80}\n[stage_benchmark] {name}\n{'=' * 80}", flush=True)

    saved = load_index(project_root / "results" / "indexes", name)
    data, dag_dict, config = saved['data'], saved['dag_dict'], saved['config']
    test_covs, train_covs = saved['test_tile_covs'], saved['train_tile_covs']
    train_idx, test_idx = saved['train_idx'], saved.get('test_idx')
    gt_block = load_or_compute_ground_truth("block", name, test_covs, train_covs, train_idx, test_idx, data=data,
                                            cache_dir=project_root / "results" / "ground_truth_cache")
    to_local = {g: l for l, g in enumerate(train_idx)}

    # holdout_core.perform_search's budgets
    niches = sorted(set(int(c) for c in data.labels))
    budgets = []
    for c in niches:
        f = _niche_scale_factor(int(np.sum(data.labels == c)))
        budgets.append(float(config.epsilon_dict[c]) * len(dag_dict[c].sorted_blocks) * budget_mult * f)
    perms = [data.perm_list[c] for c in niches]
    runs = [data.block_dict[c] for c in niches]
    inv_sqrt_p = [[1.0 / np.sqrt(e - s) for s, e in r] for r in runs]

    # stored data: float32 block logs (Stage 2 after Stage 1) and raw diagonal blocks (Stage 2 only),
    # both stacked per niche and block in the niche's gene order
    idxs_by, logs_by, raw_by, row_of = [], [], [], {}
    for j, c in enumerate(niches):
        idxs, cached = gt_block['niche_train_cache'][c]
        idxs_by.append(np.asarray(idxs))
        logs_by.append([np.stack([cached[i][b].astype(np.float32).ravel() for i in idxs]) for b in range(len(runs[j]))])
        raws = []
        for s, e in runs[j]:
            g = perms[j][s:e]
            raws.append(np.stack([(train_covs[i]['cov'] if isinstance(train_covs[i], dict) else train_covs[i])[np.ix_(g, g)]
                                  for i in idxs]))
        raw_by.append(raws)
        row_of.update({i: (j, r) for r, i in enumerate(idxs)})
    del gt_block['niche_train_cache'], train_covs

    labels = np.asarray(data.labels).astype(int)
    upper = sum(int(np.sum(labels == c)) * sum((e - s) * (e - s + 1) // 2 for s, e in runs[j]) for j, c in enumerate(niches))
    dag_mb = float(saved.get('index_size_mb', np.nan))
    storage = {'Dataset': name, 'n_train': len(labels), 'n_genes': len(perms[0]), 'n_niches': len(niches),
               'dag_mb': round(dag_mb, 3), 'block_logs_mb': round(upper * F32 / MB, 3),
               'raw_blocks_mb': round(upper * F32 / MB, 3),
               'stage1_mb': round(dag_mb, 3), 'stage1_2_mb': round(dag_mb + upper * F32 / MB, 3),
               'stage2_only_mb': round(upper * F32 / MB, 3)}

    keep = np.arange(len(test_covs))
    if max_queries is not None and len(keep) > max_queries:
        keep = np.sort(np.random.default_rng(42).choice(len(keep), size=max_queries, replace=False))
        print(f"[--max-queries] {len(test_covs)} -> {len(keep)} queries")
    queries = extract_query_matrices([test_covs[i] for i in keep])

    def rerank(ids, q32):
        """Exact block LE of ``ids`` from the stored float32 block logs."""
        by = {}
        for i in ids:
            j, r = row_of[i]
            by.setdefault(j, ([], []))
            by[j][0].append(i)
            by[j][1].append(r)
        d_all, i_all = [], []
        for j, (ii, rows) in by.items():
            rows = np.asarray(rows)
            d = np.zeros(len(rows), dtype=np.float32)
            for b, S in enumerate(logs_by[j]):
                d += np.linalg.norm(S[rows] - q32[j][b], axis=1) * inv_sqrt_p[j][b]
            d_all.append(d)
            i_all.extend(ii)
        d_all = np.concatenate(d_all) if d_all else np.zeros(0)
        return sorted(zip(d_all.tolist(), i_all))

    q_rows, t_rows = [], []
    for n_q, (qi, q) in enumerate(zip(keep, queries)):
        gt = gt_block['per_query'][qi]
        band = float(config.epsilon_dict[gt['true_best_niche']])
        near = near_sets(gt, band)
        base = {'Dataset': name, 'query_idx': int(qi)}
        t = {**base}

        for c_top in C_GRID:
            # ---- Stage 1: DAG top-c by path score
            t0 = time.perf_counter()
            qlist = [(dag_dict[c], q[np.ix_(perms[j], perms[j])], runs[j]) for j, c in enumerate(niches)]
            top, q_logs = search.search_top_c(qlist, c_top, budgets=budgets, return_query_logs=True)
            s1 = sorted(((sc, to_local[sid]) for sc, _, sid in top if sid in to_local), key=lambda x: (x[0], x[1]))
            t[f'stage1_c{c_top}_ms'] = (time.perf_counter() - t0) * 1000
            ids = [i for _, i in s1]
            rl1 = Ranked(ids, [s for s, _ in s1])
            q_rows.append({**base, 'mode': 'stage1', 'top_c': c_top, **quality(rl1, gt, band, near, TOP_K)})

            # ---- Stage 1 + 2: exact re-rank of the same c tiles (stored block logs)
            t0 = time.perf_counter()
            q32 = [None if L is None else [x.astype(np.float32).ravel() for x in L] for L in q_logs]
            rr = rerank(ids, q32)
            t[f'rerank_c{c_top}_ms'] = (time.perf_counter() - t0) * 1000
            rl2 = Ranked([i for _, i in rr], [d for d, _ in rr])
            q_rows.append({**base, 'mode': 'stage1_2', 'top_c': c_top, **quality(rl2, gt, band, near, TOP_K)})
        t['n_candidates_max'] = len(ids)

        # ---- Stage 2 only: eigendecompose every tile's raw diagonal blocks, exact block LE for all tiles
        if n_q >= stage2_only_queries:
            t['stage2_only_ms'] = np.nan
            t['stage2_only_top1_is_gt'] = np.nan
            t_rows.append(t)
            print(f"query {qi}: stage1 c100 {t['stage1_c100_ms']:.1f} ms (+{t['rerank_c100_ms']:.1f} re-rank), "
                  f"c400 {t['stage1_c400_ms']:.1f} (+{t['rerank_c400_ms']:.1f})", flush=True)
            continue
        t0 = time.perf_counter()
        d_parts, i_parts = [], []
        for j in range(len(niches)):
            d = np.zeros(len(idxs_by[j]))
            for b, (s, e) in enumerate(runs[j]):
                g = perms[j][s:e]
                Lq = log_spd(q[np.ix_(g, g)])
                Lt = batched_log_spd(raw_by[j][b])
                d += np.linalg.norm((Lt - Lq).reshape(len(Lt), -1), axis=1) * inv_sqrt_p[j][b]
            d_parts.append(d)
            i_parts.append(idxs_by[j])
        order = np.concatenate(i_parts)[np.argsort(np.concatenate(d_parts), kind='stable')]
        t['stage2_only_ms'] = (time.perf_counter() - t0) * 1000
        t['stage2_only_top1_is_gt'] = int(order[0] == gt['true_order'][0])
        t_rows.append(t)
        print(f"query {qi}: stage1 c100 {t['stage1_c100_ms']:.1f} ms (+{t['rerank_c100_ms']:.1f} re-rank), "
              f"c400 {t['stage1_c400_ms']:.1f} (+{t['rerank_c400_ms']:.1f}), stage2-only {t['stage2_only_ms']:.0f} ms",
              flush=True)

    dq, dt = pd.DataFrame(q_rows), pd.DataFrame(t_rows)
    dq.to_csv(out / "query_metrics.csv", index=False)
    dt.to_csv(out / "timing.csv", index=False)
    pd.DataFrame([storage]).to_csv(out / "storage.csv", index=False)
    summ = dq.drop(columns=['query_idx']).groupby(['Dataset', 'mode', 'top_c'], as_index=False).mean(numeric_only=True)
    summ = summ.sort_values(['mode', 'top_c']).round(4)
    summ.to_csv(out / "summary.csv", index=False)
    tsum = dt.drop(columns=['query_idx']).groupby('Dataset', as_index=False).mean(numeric_only=True).round(3)
    tsum.to_csv(out / "timing_summary.csv", index=False)

    with pd.option_context('display.width', 250, 'display.max_columns', 40):
        cols = ['mode', 'top_c', 'n_returned', 'recall_at_eps_0.1', 'recall_at_eps_0.5', 'overlap_at_eps_0.5',
                'overlap_at_eps_1.0', 'recall_at_10']
        print(summ[cols].to_string(index=False))
        print("\nMean time per query (ms):")
        print(tsum.T.to_string(header=False))
        print("\nStorage (MB):", storage)
    print(f"\n[stage_benchmark] wrote {out}")


def aggregate():
    for fname in ("summary", "timing_summary", "storage"):
        frames = [pd.read_csv(p / f"{fname}.csv") for p in sorted(OUT_ROOT.iterdir())
                  if p.is_dir() and (p / f"{fname}.csv").exists()]
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(OUT_ROOT / f"{fname}_all.csv", index=False)
            print(f"{fname}_all.csv: {len(frames)} datasets")


def main():
    ap = argparse.ArgumentParser(description="Stage 1 vs Stage 1 + Stage 2 vs Stage 2 only.")
    ap.add_argument('--dataset-path', default=None, help='h5ad path; the stem names the index in results/indexes/')
    ap.add_argument('--max-queries', type=int, default=None)
    ap.add_argument('--stage2-only-queries', type=int, default=10,
                    help='Time Stage 2 only on this many queries (its cost does not depend on the query)')
    ap.add_argument('--aggregate', action='store_true')
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    elif args.dataset_path:
        run_unit(args.dataset_path, args.max_queries, args.stage2_only_queries)
    else:
        ap.error('--dataset-path or --aggregate is required')


if __name__ == "__main__":
    main()
