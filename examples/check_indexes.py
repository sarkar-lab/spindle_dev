"""Sanity checks of the three indexes on one dataset (run through sbatch).

1. Spindle self-queries return the tile itself at d_B = 0.
2. Spindle-lite with the effort caps lifted returns every tile with d_B <= budget, and every path's lower
   bound is <= the exact d_B of its tiles.
3. Interval index: on a one-piece gene set, |score - exact d_B^S| <= error_bound.
4. Spindle-lite set size, Overlap(5 %) and recall of the exact top 10 at budget multipliers 0.25-2 (information).

Usage: python examples/check_indexes.py <dataset key>
"""
import sys
import time
from pathlib import Path

import numpy as np

repo = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(repo / "src"), str(repo)]
import paths  # noqa: E402
import spindle_dev as sd  # noqa: E402
from spindle_dev.lite import search_index, SearchConfig, _niche_scale_factor  # noqa: E402
from spindle_dev.metrics import as_cov, query_cov_s  # noqa: E402

key = sys.argv[1]
t0 = time.time()
adata = sd.load_adata(paths.dataset_path(key))
tiles, covs, genes = sd.prepare_tiles(adata, max_genes=paths.dataset_max_genes(key))
tr, te = sd.split_tiles(len(tiles), seed=73, n_holdout=100)
tr_covs = [covs[i] for i in tr]
te_covs = [covs[i] for i in te]
data = sd.build_layout([tiles[i] for i in tr], tr_covs, genes, adata.n_obs)
t_layout = time.time() - t0
spindle = sd.Spindle(data, tr_covs)
t1 = time.time()
lite = sd.build_lite(data)
t_lite = time.time() - t1
labels = np.asarray(data.labels).astype(int)
N = len(labels)
print(f"{key}: {N} training tiles, {len(set(labels.tolist()))} niches, "
      f"blocks {[len(data.block_dict[k]) for k in sorted(data.block_dict)]}; layout {t_layout:.0f}s, lite {t_lite:.0f}s")
print("budget eps per niche:", {k: round(v, 4) for k, v in lite.epsilon_dict.items()})
ok = True

# 1. self-queries
for r in [0, N // 2, N - 1]:
    rows, sc, _ = spindle.search(tr_covs[r], c=1)
    good = rows[0] == r and sc[0] < 1e-5
    ok &= good
    print(f"{'OK  ' if good else 'FAIL'} Spindle self-query row {r}: top {rows[0]} at {sc[0]:.2e}")

# 2. Spindle-lite safety (caps lifted, m = 1) and the per-path lower bound
miss_total = within_total = 0
lb_viol = 0
for q in te_covs[:30]:
    d = spindle.distances(spindle.query_vectors(q))
    found, _ = sd.search_lite(lite, q, budget_multiplier=1.0, effort_multiplier=None)
    found = set(found.tolist())
    for k, h in lite.dag_dict.items():
        budget = lite.epsilon_dict[k] * _niche_scale_factor(lite.niche_sizes[k])
        within = [r for r in np.flatnonzero(labels == k) if d[r] <= budget * (1 - 1e-4)]
        within_total += len(within)
        miss_total += sum(r not in found for r in within)
        qa = as_cov(q)
        res = search_index(h, qa[np.ix_(h.perm, h.perm)], budget, SearchConfig(total_paths_limit=None))
        for path in res.paths[:50]:
            for sid in path.member_ids:
                if path.total_distance > d[lite.row_of_spd_id[sid]] + 1e-4:
                    lb_viol += 1
good = miss_total == 0
ok &= good
print(f"{'OK  ' if good else 'FAIL'} lite safety: {within_total} (query, tile) pairs within budget, {miss_total} missed")
good = lb_viol == 0
ok &= good
print(f"{'OK  ' if good else 'FAIL'} lite path lower bound <= exact d_B: {lb_viol} violations")

# 3. interval index error bound on a one-piece gene set
partial = sd.SpindlePartial(data, tr_covs)
iv = sd.build_interval_index(data, block_covs=partial.covs)
k0 = int(sorted(set(labels.tolist()))[0])
s, e = data.block_dict[k0][0]
gs = np.asarray(data.perm_list[k0])[s:s + min(4, e - s)]
bound = sd.error_bound(iv, gs)[k0]
worst = 0.0
for q in te_covs[:20]:
    rows_i, sc_i, _ = sd.search_interval_index(iv, q, gs)
    d_ex = partial.distances(query_cov_s(q, gs, partial.p), gs)
    in_k0 = np.isin(rows_i, np.flatnonzero(labels == k0))
    worst = max(worst, float(np.abs(sc_i[in_k0] - d_ex[rows_i[in_k0]]).max()))
good = worst <= bound + 1e-6
ok &= good
print(f"{'OK  ' if good else 'FAIL'} interval |score - exact d_B^S| max {worst:.4f} <= bound {bound:.4f} (niche {k0}, 4 genes)")

# 4. information: lite at default caps vs budget
print("m      mean |set|/N   Overlap(5%)   recall@10   ms/query")
for m in [0.25, 0.5, 1.0, 2.0]:
    sizes, ovl, rec, ms = [], [], [], []
    for q in te_covs:
        d = spindle.distances(spindle.query_vectors(q))
        near = np.flatnonzero(d <= 1.05 * d.min())
        top10 = np.argsort(d, kind="stable")[:10]
        found, sec = sd.search_lite(lite, q, budget_multiplier=m)
        sizes.append(len(found) / N)
        ovl.append(np.isin(near, found).mean())
        rec.append(np.isin(top10, found).mean())
        ms.append(sec * 1e3)
    print(f"{m:<6} {np.mean(sizes):>10.3f} {np.mean(ovl):>12.3f} {np.mean(rec):>11.3f} {np.mean(ms):>10.1f}")
print("ALL OK" if ok else "CHECKS FAILED")
