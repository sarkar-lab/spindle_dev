"""Partial-query search on every dataset: the interval index against the exact tier.

One dataset per run, production index (seed 73), its 100 held-out tiles as queries.

Queries (draw_queries; gene sets defined without reference to any
niche, query covariance = the held-out tile's covariance on the gene set):
random gene sets and gene programs of SIZES genes, and the tissue's curated signatures.

Exact tier (tiers.ExactPartialTier; ground truth and the reference for time and storage): for every niche it
stores each tile's block logs and block covariances (float32 upper triangles).  For a
query S it computes, per niche and block, the query's log on S cap block and the tiles'
logs on S cap block -- from the stored block logs when S covers the whole block, otherwise
by eigendecomposition of the stored block covariances restricted to S cap block -- and
returns the restricted block distance

    d_B^S(q, t) = sqrt(sum_b ||log Sigma_q[S cap b] - log Sigma_t[S cap b]||_F^2) / sqrt(|S|).

Its query time includes those eigendecompositions. Tile and query covariances are shrunk first
(tiers.prepare_cov, metrics.SHRINK_ALPHA).

Interval index (src/spindle_dev/interval_dp.py): CONFIGS (node target K_s = k_target * s^k_alpha,
longest interval max_len, no Lloyd iterations; the first is the paper's), scored by the root of
the summed squared distances to the node means over the fewest-piece dyadic decomposition
(score="mean").  The index-free limit of that score (the tiles' own interval logs,
score="exact") is reported per max_len.

Metrics per (query, method, c): Overlap(delta, c) against d_B^S (expected value under random
tie-breaking), hit@c, near-set sizes, query time.

Outputs (results/interval_index_final/): <stem>_queries.csv, <stem>_metrics.csv, <stem>_configs.csv

  sbatch slurm_jobs/run_interval_index_final.sbatch <dataset>
"""

import argparse
import gc
import time

import numpy as np
import pandas as pd

import experiment_common as ec
import gene_signatures
from spindle_dev import interval_dp as ivd
from spindle_dev import tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "interval_index_final"
SIZES = [4, 8, 16, 32, 48, 64]
DELTAS = [0.01, 0.025, 0.05, 0.10]
C_FRAC = [0.05, 0.10]
# The paper's configuration is the first (K_s = 32 s^0.5, intervals up to 16 genes); the other two
# give more nodes to long intervals (alpha = 1) and index every length.
CONFIGS = [dict(k_target=32, k_alpha=0.5, max_len=16),
           dict(k_target=32, k_alpha=1.0, max_len=16),
           dict(k_target=32, k_alpha=1.0, max_len=None)]
MB = 2 ** 20
TISSUE = {"skin_melanoma": "skin", "kidney_nondiseased": "kidney", "breast_cancer": "breast",
          "lung_cancer": "lung", "pancreatic_cancer": "pancrea", "lymph_node": "lymph_node",
          "lymph_node_5k": "lymph_node", "brain_cancer": "brain"}


def segments_of(data, genes):
    """{niche: number of segments of the gene set in that niche's block order}."""
    out = {}
    for k in sorted(set(np.asarray(data.labels).astype(int).tolist())):
        perm = np.asarray(data.perm_list[k])
        pos = np.empty(len(perm), dtype=int)
        pos[perm] = np.arange(len(perm))
        blk = np.empty(len(perm), dtype=int)
        for b, (s, e) in enumerate(data.block_dict[k]):
            blk[perm[s:e]] = b
        g = np.asarray(genes)
        order = np.argsort(pos[g])
        p_sorted, b_sorted = pos[g][order], blk[g][order]
        out[k] = 1 + int(np.sum((np.diff(p_sorted) > 1) | (np.diff(b_sorted) != 0)))
    return out


def draw_queries(data, tests, genes_all, tissue, seed, families=("random", "program", "signature"), sizes=SIZES):
    rng = np.random.default_rng(seed)
    p = len(genes_all)
    pairs = [(k, b, e - s) for k in sorted(data.block_dict) for b, (s, e) in enumerate(data.block_dict[k])]
    sigs = []
    for name, info in gene_signatures.TISSUE_MODULES.get(tissue, {}).items():
        g = [genes_all.index(x) for x in info["genes"] if x in genes_all]
        if len(g) >= 2:
            sigs.append((name, np.sort(np.asarray(g))))
    queries = []
    for ti in range(len(tests)):
        for m in sizes:
            if "random" in families:
                queries.append(dict(tile=ti, family="random", size=m, label="",
                                    genes=np.sort(rng.choice(p, m, replace=False))))
            ok = [(k, b, n) for k, b, n in pairs if n >= m]
            if ok and "program" in families:
                k, b, n = ok[rng.integers(len(ok))]
                s, _ = data.block_dict[k][b]
                a = int(rng.integers(0, n - m + 1))
                g = np.asarray(data.perm_list[k])[s + a:s + a + m]
                queries.append(dict(tile=ti, family="program", size=m, label=f"niche{k}_block{b}",
                                    source_niche=k, whole_block=int(n == m), genes=np.sort(g)))
        for name, g in (sigs if "signature" in families else []):
            queries.append(dict(tile=ti, family="signature", size=len(g), label=name, genes=g))
    for qi, q in enumerate(queries):
        q["query"] = qi
    return queries


def overlap_rows(d_exact, score, rows, n, d_target=None):
    """Expected Overlap(delta, c) under random tie-breaking at the c-th score, per c."""
    full = np.full(n, np.inf)
    full[rows] = score
    j = int(np.argmin(d_exact))
    near = {dl: np.flatnonzero(d_exact <= (1 + dl) * d_exact[j]) for dl in DELTAS}
    near_t = None if d_target is None else np.flatnonzero(d_target <= 1.05 * d_target.min())
    srt = np.sort(full)
    out = []
    for cf in C_FRAC:
        c = int(max(1, round(cf * n)))
        thr = srt[c - 1]
        below = full < thr
        tied = full == thr
        prob = below.astype(float)
        prob[tied] = (c - below.sum()) / tied.sum()
        rec = {"c_frac": cf, "c": c, "hit": prob[j], "n_zero": int((full == 0).sum()), "n_tied_at_c": int(tied.sum())}
        for dl, m in near.items():
            rec[f"overlap_{dl}"] = prob[m].mean()
            rec[f"near_size_{dl}"] = len(m)
        if near_t is not None:
            rec["overlap_target_0.05"] = prob[near_t].mean()
        out.append(rec)
    return out


def run(stem, short):
    t_load = time.perf_counter()
    data = ec.load_index(stem, ec.PRODUCTION_SEED)["data"]
    raw = ec.load_raw_covs(stem, ec.PRODUCTION_SEED)
    tests = [tiers.prepare_cov(c) for c in raw["test_tile_covs"]]  # shrunk as configured (metrics.SHRINK_ALPHA)
    exact = tiers.ExactPartialTier(data, raw["train_tile_covs"])
    del raw
    gc.collect()
    genes_all = list(data.metadata["genes"])
    n, p = exact.n, len(genes_all)
    labels = exact.labels
    print(f"{stem}: {n} tiles, {p} genes, {len(tests)} held-out tiles; loaded in {time.perf_counter() - t_load:.0f} s",
          flush=True)

    queries = draw_queries(data, tests, genes_all, TISSUE[short], seed=ec.PRODUCTION_SEED, sizes=SIZES)
    print(f"{len(queries)} queries", flush=True)
    d_exact, qrows = [], []
    for q in queries:
        t1 = time.perf_counter()
        d = exact.distances(tests[q["tile"]][np.ix_(q["genes"], q["genes"])], q["genes"])
        exact_ms = 1e3 * (time.perf_counter() - t1)
        d_exact.append(d)
        segs = segments_of(data, q["genes"])
        w = np.bincount(labels)[sorted(segs)]
        nn = int(np.argmin(d))
        qrows.append({k: v for k, v in q.items() if k != "genes"} | {
            "genes": ",".join(genes_all[g] for g in q["genes"]),
            "seg_frac_mean": float(np.average([segs[k] / len(q["genes"]) for k in sorted(segs)], weights=w)),
            "seg_nn_niche": segs[int(labels[nn])], "nn_niche": int(labels[nn]),
            "d_star": float(d[nn]), "exact_tier_ms": exact_ms})
    qdf = pd.DataFrame(qrows)
    print(f"exact tier: median {qdf.exact_tier_ms.median():.1f} ms per query", flush=True)

    ex_bytes = exact.nbytes_parts()
    metrics, configs = [], []
    done_limit = set()
    for cfg in CONFIGS:
        ml = cfg["max_len"]
        need_limit = ml not in done_limit
        idx = ivd.build_interval_index(data, k_target=cfg["k_target"], k_alpha=cfg["k_alpha"], lloyd_iters=0,
                                       max_len=ml, keep_tile_logs=need_limit, block_covs=exact.block_covs())
        nb = idx.nbytes()
        tag = dict(k_target=cfg["k_target"], k_alpha=cfg["k_alpha"], max_len=-1 if ml is None else ml)
        configs.append(tag | {"build_s": idx.build_seconds, "n_nodes": idx.n_nodes(), "index_mb": nb["total"] / MB,
                              "means_mb": nb["means"] / MB, "codes_mb": nb["codes"] / MB})
        runs = [("index", dict(score="mean"))] + ([("limit", dict(score="exact"))] if need_limit else [])
        for method, kw in runs:
            for q, d in zip(queries, d_exact):
                cov = tests[q["tile"]][np.ix_(q["genes"], q["genes"])]
                r = ivd.search_interval_index(idx, q["genes"], cov, **kw)
                for rec in overlap_rows(d, r.scores, r.rows, n):
                    metrics.append({"query": q["query"], "method": method,
                                    **(tag if method == "index" else dict(k_target=-1, k_alpha=-1, max_len=tag["max_len"])),
                                    "query_ms": 1e3 * r.stats["seconds"]} | rec)
        done_limit.add(ml)
        m = pd.DataFrame([x for x in metrics if x["method"] == "index" and x["c_frac"] == 0.05
                          and all(x[k] == v for k, v in tag.items())])
        print(f"{tag}: build {idx.build_seconds:.0f} s, {nb['total'] / MB:.1f} MB, "
              f"overlap(5%,5%) {m['overlap_0.05'].mean():.3f}, median {m.query_ms.median():.1f} ms", flush=True)
        del idx
        gc.collect()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qdf.to_csv(OUT_DIR / f"{stem}_queries.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT_DIR / f"{stem}_metrics.csv", index=False)
    pd.DataFrame(configs).assign(
        dataset=short, n_tiles=n, n_genes=p,
        exact_block_logs_mb=ex_bytes["block_logs"] / MB, exact_block_covs_mb=ex_bytes["block_covs"] / MB,
        exact_tier_mb=(ex_bytes["block_logs"] + ex_bytes["block_covs"]) / MB,
        whole_cov_upper_mb=4 * n * p * (p + 1) / 2 / MB,
    ).to_csv(OUT_DIR / f"{stem}_configs.csv", index=False)
    print("done", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", required=True, help="short key or stem")
    args = ap.parse_args()
    stem = ec.resolve_stem(args.dataset)
    run(stem, next(k for k, v in ec.DATASETS.items() if v == stem))


if __name__ == "__main__":
    main()
