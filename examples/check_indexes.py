"""Sanity checks of the indexes on one dataset (run through sbatch).

1. Spindle self-queries return the tile itself at d_B = 0.
2. The compiled searches equal their numpy / Python references bit for bit: Spindle.distances vs distances_numpy,
   SpindleLite.search_vectors vs search_vectors_dfs (rows, scores, A_hat, radius, work).
3. Spindle-lite: every tile's path lower bound sqrt(sum_b lb_b^2) is <= its exact d_B, and at g = 1 every tile with
   d_B <= radius is returned.
4. Spindle partial: self-queries on a gene set return the tile itself at d_B^S = 0.
5. Spindle-lite partial (contiguous 8-gene runs in a random block, 20 held-out queries): compiled search ==
   Python DFS reference bit for bit at g = 1 and 2; every tile's path lower bound sqrt(sum lb^2) <= its piece
   distance (its own interval logs); at g = 2 every tile with piece distance <= radius is returned.
6. Information: Spindle-lite at g = 0.25 / 0.5 / 1 and Spindle-lite partial at g = 1 (tiles returned, node
   distances, excess mean distance of the returned top 10 over the exact top 10) and search-only times (query logs
   excluded) against the exact scans.
7. Gene-list queries (queries.py; run before 6): gene_indices drops unknown genes and duplicates (case-sensitive and
   not); tile_query self-query; template_query on 20 random gene lists (3-16 genes, across blocks) has variances
   omega, correlations 0.5 and is SPD; Spindle-lite partial compiled search == Python DFS on those queries at
   g = 1 and 2.

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
from spindle_dev.lite_partial import interval_vectors  # noqa: E402
from spindle_dev.metrics import query_cov_s  # noqa: E402

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
lite = sd.SpindleLite.from_spindle(spindle, data)
t_lite = time.time() - t1
labels = np.asarray(data.labels).astype(int)
N = len(labels)
print(f"{key}: {N} training tiles, {len(set(labels.tolist()))} niches, "
      f"blocks {[len(data.block_dict[k]) for k in sorted(data.block_dict)]}; layout {t_layout:.0f}s, lite {t_lite:.0f}s; "
      f"lite storage {100 * lite.nbytes() / lite.spindle_bytes:.2f} % of Spindle")
ok = True


def report(good, msg):
    global ok
    ok &= bool(good)
    print(f"{'OK  ' if good else 'FAIL'} {msg}", flush=True)


# 1. self-queries
for r in [0, N // 2, N - 1]:
    rows, sc, _ = spindle.search(tr_covs[r], c=1)
    report(rows[0] == r and sc[0] < 1e-5, f"Spindle self-query row {r}: top {rows[0]} at {sc[0]:.2e}")

qvs = [spindle.query_vectors(q) for q in te_covs]
D = np.stack([spindle.distances(qv) for qv in qvs])

# 2. compiled == reference
bad_s = sum(not np.array_equal(D[i].view(np.int64), spindle.distances_numpy(qv).view(np.int64))
            for i, qv in enumerate(qvs))
report(bad_s == 0, f"Spindle compiled scan == numpy on {len(qvs) - bad_s}/{len(qvs)} queries")
bad_l = 0
for qv in qvs:
    a, b = lite.search_vectors(qv), lite.search_vectors_dfs(qv)
    bad_l += not (np.array_equal(a[0], b[0]) and np.array_equal(a[1].view(np.int64), b[1].view(np.int64))
                  and a[2] == b[2])
report(bad_l == 0, f"Spindle-lite compiled search == Python DFS on {len(qvs) - bad_l}/{len(qvs)} queries")

# 3. lower bound and safety
lb_viol = miss = within = 0
for i, qv in enumerate(qvs):
    for k, nc in lite.niches.items():
        lb2 = np.zeros(nc.n)
        for C, R, a, (s, e) in zip(nc.centres, nc.radius, nc.node_index, nc.layer_slices):
            d = np.sqrt(((C - qv[k][s:e].astype(np.float64)) ** 2).sum(1))
            lb2 += np.maximum(0.0, d - R.astype(np.float64))[a] ** 2
        lb_viol += int((np.sqrt(lb2) > D[i, nc.rows] + 1e-6).sum())
    rows, _, st = lite.search_vectors(qv, g=1.0)
    near = np.flatnonzero(D[i] <= st["radius"] * (1 - 1e-6))
    within += len(near)
    miss += int((~np.isin(near, rows)).sum())
report(lb_viol == 0, f"Spindle-lite path lower bound <= exact d_B for every tile: {lb_viol} violations")
report(miss == 0, f"Spindle-lite safety at g = 1: {within} (query, tile) pairs within the radius, {miss} missed")

# 4. Spindle partial self-queries
partial = sd.SpindlePartial(data, tr_covs)
for r in [0, N // 2, N - 1]:
    k = int(labels[r])
    s0, e0 = data.block_dict[k][0]
    gs = np.asarray(data.perm_list[k])[s0:s0 + min(12, e0 - s0)]
    rows, sc, _ = partial.search(tr_covs[r], gs, c=1)
    report(rows[0] == r and sc[0] < 1e-5, f"Spindle partial self-query row {r} (12 genes): top {rows[0]} at {sc[0]:.2e}")

# 5. Spindle-lite partial
t1 = time.time()
plite = sd.SpindleLitePartial.from_partial(partial, data)
print(f"Spindle-lite partial built in {time.time() - t1:.0f}s, storage {100 * plite.nbytes() / plite.spindle_bytes:.2f} % "
      f"of Spindle", flush=True)
rng = np.random.default_rng(73)
gsets = []
for i in range(20):
    k = int(rng.choice(sorted(data.block_dict)))
    b = int(rng.integers(len(data.block_dict[k])))
    s0, e0 = data.block_dict[k][b]
    L = min(8, e0 - s0)
    a = int(rng.integers(e0 - s0 - L + 1))
    gsets.append(np.asarray(data.perm_list[k])[s0 + a:s0 + a + L])
qps = [plite.query_vectors(te_covs[i], gs) for i, gs in enumerate(gsets)]
bad = 0
for qp in qps:
    for g in (1.0, 2.0):
        a, b = plite.search_vectors(qp, g=g), plite.search_vectors_dfs(qp, g=g)
        bad += not (np.array_equal(a[0], b[0]) and np.array_equal(a[1].view(np.int64), b[1].view(np.int64))
                    and a[2] == b[2])
report(bad == 0, f"Spindle-lite partial compiled search == Python DFS on {2 * len(qps) - bad}/{2 * len(qps)} "
                 f"(query, g) pairs")
lb_viol = miss = within = 0
for gs, qp in zip(gsets, qps):
    dP = np.empty(N)
    lb2 = np.empty(N)
    for k, nc in plite.niches.items():
        tot, low = np.zeros(nc.n), np.zeros(nc.n)
        for iv, q in qp.layers[k]:
            b, s_, j = nc.keys[iv]
            own = interval_vectors(partial.covs[k][b], s_, j)
            tot += ((own - q.astype(np.float64)) ** 2).sum(1)
            d = np.sqrt(((nc.centres[iv].astype(np.float64) - q.astype(np.float64)) ** 2).sum(1))
            low += np.maximum(0.0, d - nc.radius[iv].astype(np.float64))[nc.node_index[iv]] ** 2
        dP[nc.rows] = np.sqrt(tot)
        lb2[nc.rows] = low
    lb_viol += int((np.sqrt(lb2) > dP + 1e-5).sum())
    rows, _, st = plite.search_vectors(qp, g=2.0)
    near = np.flatnonzero(dP <= st["radius"] * np.sqrt(len(gs)) * (1 - 1e-6))
    within += len(near)
    miss += int((~np.isin(near, rows)).sum())
report(lb_viol == 0, f"Spindle-lite partial path lower bound <= piece distance for every tile: {lb_viol} violations")
report(miss == 0, f"Spindle-lite partial safety at g = 2: {within} (query, tile) pairs within the radius, {miss} missed")

# 7. gene-list queries (queries.py)
names = [str(x) for x in genes]
idx, found_g, missing = sd.gene_indices(names, [names[5], names[2], names[5], "NOT_A_GENE", names[2].lower()])
report(idx.tolist() == [5, 2] and found_g == [names[5], names[2]] and missing == ["NOT_A_GENE", names[2].lower()],
       f"gene_indices: duplicates removed, unknown genes reported ({idx.tolist()}, missing {missing})")
idx, _, missing = sd.gene_indices(names, [names[5].lower(), names[2]], case_sensitive=False)
report(idx.tolist() == [5, 2] and not missing, "gene_indices: case-insensitive lookup")
cq, ci = sd.tile_query(tr_covs[N // 2], gsets[0])
rows, sc, _ = partial.search(cq, ci, c=1)
report(rows[0] == N // 2 and sc[0] < 1e-5, f"tile_query self-query row {N // 2}: top {rows[0]} at {sc[0]:.2e}")
omega = sd.variance_quantiles(tr_covs)
lsets = [rng.choice(partial.p, int(rng.integers(3, 17)), replace=False) for _ in range(20)]
bad_t = 0
tqs = []
for gs in lsets:
    cq, ci = sd.template_query(omega, gs)
    sdv = np.sqrt(np.diag(cq))
    r = cq / np.outer(sdv, sdv)
    bad_t += not (np.allclose(np.diag(cq), omega[gs]) and np.allclose(r[~np.eye(len(gs), dtype=bool)], 0.5)
                  and np.linalg.eigvalsh(cq).min() > 0)
    tqs.append((ci, plite.query_vectors(cq, ci)))
report(bad_t == 0, f"template_query: variances = omega, correlations = 0.5, SPD on {len(lsets) - bad_t}/{len(lsets)} "
                   f"random gene lists (3-16 genes, across blocks)")
bad = 0
for gs, qp in tqs:
    for g in (1.0, 2.0):
        a, b = plite.search_vectors(qp, g=g), plite.search_vectors_dfs(qp, g=g)
        bad += not (np.array_equal(a[0], b[0]) and np.array_equal(a[1].view(np.int64), b[1].view(np.int64))
                    and a[2] == b[2])
report(bad == 0, f"Spindle-lite partial compiled search == Python DFS on template queries: {2 * len(tqs) - bad}/"
                 f"{2 * len(tqs)} (query, g) pairs")

# 6. information (search only: the query's block logs are computed beforehand for both indexes)
spindle.distances(qvs[0])
lite.search_vectors(qvs[0])
t_scan = []
for qv in qvs:
    ts = time.perf_counter()
    np.argsort(spindle.distances(qv), kind="stable")
    t_scan.append(time.perf_counter() - ts)
print(f"Spindle scan: median {1e3 * np.median(t_scan):.3f} ms")
print("g      tiles   node dists   excess mean d (top 10)   short of 10   ms      speed-up")
for g in [0.25, 0.5, 1.0]:
    n_found, work, mx, short, ms = [], [], [], 0, []
    for i, qv in enumerate(qvs):
        ts = time.perf_counter()
        rows, _, st = lite.search_vectors(qv, g=g)
        ms.append(time.perf_counter() - ts)
        n_found.append(st["n_found"])
        work.append(st["work"])
        if len(rows) < 10:
            short += 1
            continue
        exact = np.sort(D[i])[:10].mean()
        mx.append((D[i, rows[:10]].mean() - exact) / exact)
    print(f"{g:<6} {np.median(n_found):>6.0f} {np.median(work):>12.0f} {100 * np.mean(mx):>20.2f} % "
          f"{short:>12d} {1e3 * np.median(ms):>8.3f} {np.median(t_scan) / np.median(ms):>8.1f}x")
t_ex, t_pl, n_pl, w_pl, mx_pl = [], [], [], [], []
for i, (gs, qp) in enumerate(zip(gsets, qps)):
    cov_s = query_cov_s(te_covs[i], gs, partial.p)
    ts = time.perf_counter()
    dS = partial.distances(cov_s, gs)
    np.argsort(dS, kind="stable")
    t_ex.append(time.perf_counter() - ts)
    ts = time.perf_counter()
    rows, _, st = plite.search_vectors(qp)
    t_pl.append(time.perf_counter() - ts)
    n_pl.append(st["n_found"])
    w_pl.append(st["work"])
    if len(rows) >= 10:
        exact = np.sort(dS)[:10].mean()
        mx_pl.append((dS[rows[:10]].mean() - exact) / exact)
print(f"Spindle-lite partial (g = 1, 8-gene runs): median {np.median(n_pl):.0f} tiles, {np.median(w_pl):.0f} node "
      f"distances, excess mean d (top 10) {100 * np.mean(mx_pl):.2f} %, {1e3 * np.median(t_pl):.3f} ms vs exact "
      f"partial scan {1e3 * np.median(t_ex):.2f} ms")
print("ALL OK" if ok else "CHECKS FAILED")
