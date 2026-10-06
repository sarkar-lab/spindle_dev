"""Partial (gene-set) search on every dataset (Fig 4, Fig S8): the interval index and two index-free
baselines (padding, imputation) against the exact partial tier.

One (dataset, seed) per run: the saved index's layout, its held-out tiles as queries (at most
--max-tiles). Queries (draw_queries; gene sets defined without reference to the query's niche, query
covariance = the held-out tile's prepared covariance cut to the gene set S): random gene sets and gene
programs (a run of consecutive genes of one niche's block) of SIZES genes, and the tissue's curated
signatures.

Methods, all scored against the ground truth d_B^S of the exact partial tier (tiers.ExactPartialTier):
  interval        the interval index (interval_dp.py), every CONFIG (the first is the paper's);
  limit           its index-free limit (the tiles' own interval logs), per max_len;
  padding         Q_S embedded in a p x p scaled identity, then Spindle-Exact (whole-tile d_B);
  imputation      the conditional-Gaussian fill from the predicted niche's mean covariance, then Spindle-Exact;
Padding and imputation get the same prepared Q_S as the ground truth and are not shrunk again
(src/spindle_dev/partial_search.py).

Readouts per (query, method, c) on dag_eval_common.c_grid(N), with ties at the c-th score broken at random
(expected values; dag_eval_common.retrieval_rows_tied): Overlap(delta, c), hit, recall of the exact top 10.

Outputs (results/partial_search_final/), tag = <stem>_seed<s>:
  <tag>_queries.csv     one row per query (family, size, genes, segments, d*, predicted niche, ...)
  <tag>_per_query.csv   query x method readouts at c = 5 % and 10 % of N
  <tag>_curves.csv      mean readouts per (method, config, family, size, c) over the run's queries
  <tag>_storage.csv     MB of every stored form, interval index per config (build s, nodes)
  <tag>_bound.csv       (S8) per query and niche: |interval score - limit score| vs interval_dp.error_bound
  --timing:   <tag>_timing.csv  gene programs only, the paper's config; ms per query for the 4 methods,
              split into query preparation, query logs and scan (run one at a time, one BLAS thread)
  --schematic (seed 73): schematic_example.json, the real query drawn in Fig 4A
  --aggregate: curves.csv, by_size.csv, summary.csv (c90), storage.csv, timing_summary.csv,
               bound_summary.csv, query_structure.csv

  sbatch slurm_jobs/run_partial_search_final.sbatch <dataset> <seed> [--timing | --schematic] | --aggregate
"""

import argparse
import gc
import json
import time

import numpy as np
import pandas as pd

import dag_eval_common as de
import experiment_common as ec
import gene_signatures
from spindle_dev import interval_dp as ivd
from spindle_dev import partial_search, tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "partial_search_final"
SIZES = [4, 8, 16, 32, 48, 64]
C_FRAC = [0.05, 0.10]
# The paper's configuration is the first (K_s = 32 s^0.5, intervals up to 16 genes); the other two
# give more nodes to long intervals (alpha = 1) and index every length.
CONFIGS = [dict(k_target=32, k_alpha=0.5, max_len=16),
           dict(k_target=32, k_alpha=1.0, max_len=16),
           dict(k_target=32, k_alpha=1.0, max_len=None)]
NO_CFG = dict(k_target=-1, k_alpha=-1, max_len=-1)
BOUND_QUERIES = 20
MAX_TILES = 200
MB = 2 ** 20
TISSUE = {"skin_melanoma": "skin", "kidney_nondiseased": "kidney", "breast_cancer": "breast",
          "lung_cancer": "lung", "pancreatic_cancer": "pancrea", "lymph_node": "lymph_node",
          "lymph_node_5k": "lymph_node", "brain_cancer": "brain"}


def cfg_tag(cfg):
    return dict(k_target=cfg["k_target"], k_alpha=cfg["k_alpha"], max_len=-1 if cfg["max_len"] is None else cfg["max_len"])


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


def draw_queries(data, n_tiles, genes_all, tissue, seed, families=("random", "program", "signature"), sizes=SIZES):
    rng = np.random.default_rng(seed)
    p = len(genes_all)
    pairs = [(k, b, e - s) for k in sorted(data.block_dict) for b, (s, e) in enumerate(data.block_dict[k])]
    sigs = []
    for name, info in gene_signatures.TISSUE_MODULES.get(tissue, {}).items():
        g = [genes_all.index(x) for x in info["genes"] if x in genes_all]
        if len(g) >= 2:
            sigs.append((name, np.sort(np.asarray(g))))
    queries = []
    for ti in range(n_tiles):
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


class Setup:
    """Everything one (dataset, seed) run needs: tiers, prepared and raw queries, layout."""

    def __init__(self, stem, seed, max_tiles, niche_means=True):
        t0 = time.perf_counter()
        self.data = ec.load_index(stem, seed)["data"]
        raw = ec.load_raw_covs(stem, seed)
        self.tests = [tiers.prepare_cov(c) for c in raw["test_tile_covs"][:max_tiles]]  # shrunk over all p genes, as the tiles
        self.ep = tiers.ExactPartialTier(self.data, raw["train_tile_covs"], keep_niche_means=niche_means)
        del raw
        gc.collect()
        self.exact = tiers.ExactTier.from_exact_partial(self.ep, self.data)
        self.means = self.ep.niche_means() if niche_means else None
        self.genes_all = list(self.data.metadata["genes"])
        self.n, self.p = self.ep.n, self.ep.p
        print(f"{stem} seed {seed}: {self.n} tiles, {self.p} genes, {len(self.tests)} query tiles; "
              f"loaded in {time.perf_counter() - t0:.0f} s", flush=True)

    def cov_s(self, q):
        return self.tests[q["tile"]][np.ix_(q["genes"], q["genes"])]

    def padding(self, cov_s, genes):
        """-> (Exact d_B of the padded query, prep s, query-log s, scan s)."""
        t0 = time.perf_counter()
        full = partial_search.pad_query_scaled_identity(cov_s, genes, self.p)
        return self._exact(full, t0)

    def imputation(self, cov_s, genes):
        """-> (Exact d_B of the imputed query, prep s, query-log s, scan s, predicted niche)."""
        t0 = time.perf_counter()
        k_hat, _ = partial_search.predict_niche(cov_s, genes, self.means, self.ep.blocks)
        full = partial_search.impute_query_conditional(cov_s, genes, self.means[k_hat])
        return self._exact(full, t0) + (k_hat,)

    def _exact(self, full, t0):
        t1 = time.perf_counter()
        qv = self.exact.query_vectors(full)  # already prepared: no second shrinkage
        t2 = time.perf_counter()
        d = self.exact.distances(qv)
        t3 = time.perf_counter()
        return d, t1 - t0, t2 - t1, t3 - t2


def full_scores(n, rows, scores):
    out = np.full(n, np.inf)
    out[rows] = scores
    return out


def run(stem, short, seed, max_tiles, max_queries):
    S = Setup(stem, seed, max_tiles)
    data, n, labels = S.data, S.n, S.ep.labels
    tag = f"{stem}_seed{seed}"
    cs = de.c_grid(n)
    c_frac = [int(max(1, round(f * n))) for f in C_FRAC]
    queries = draw_queries(data, len(S.tests), S.genes_all, TISSUE[short], seed=seed)[:max_queries]
    print(f"{len(queries)} queries", flush=True)

    rows, d_exact, qrows = [], [], []

    def score(q, method, cfg, scores, ms):
        for rec in de.retrieval_rows_tied(d_exact[q["query"]], scores, cs):
            rows.append({"query": q["query"], "family": q["family"], "size": q["size"], "method": method, **cfg,
                         "query_ms": ms, **rec})

    for q in queries:
        g = q["genes"]
        cov = S.cov_s(q)
        t0 = time.perf_counter()
        d = S.ep.distances(cov, g)
        ex_ms = 1e3 * (time.perf_counter() - t0)
        d_exact.append(d)
        segs = segments_of(data, g)
        w = np.bincount(labels)[sorted(segs)]
        nn = int(np.argmin(d))
        d_pad, *tp = S.padding(cov, g)
        d_imp, *ti, k_hat = S.imputation(cov, g)
        qrows.append({k: v for k, v in q.items() if k != "genes"} | {
            "genes": ",".join(S.genes_all[x] for x in g),
            "seg_frac_mean": float(np.average([segs[k] / len(g) for k in sorted(segs)], weights=w)),
            "seg_nn_niche": segs[int(labels[nn])], "nn_niche": int(labels[nn]), "d_star": float(d[nn]),
            "near5_size": int((d <= 1.05 * d[nn]).sum()), "pred_niche": int(k_hat),
            "exact_partial_ms": ex_ms, "padding_ms": 1e3 * sum(tp), "imputation_ms": 1e3 * sum(ti)})
        score(q, "padding", NO_CFG, d_pad, 1e3 * sum(tp))
        score(q, "imputation", NO_CFG, d_imp, 1e3 * sum(ti))
    qdf = pd.DataFrame(qrows)
    print(f"exact partial tier: median {qdf.exact_partial_ms.median():.1f} ms; padding "
          f"{qdf.padding_ms.median():.1f} ms; imputation {qdf.imputation_ms.median():.1f} ms per query", flush=True)

    storage, bound = [], []
    ex_bytes = S.ep.nbytes_parts()
    done_limit = set()
    for ci, cfg in enumerate(CONFIGS):
        ml = cfg["max_len"]
        need_limit = ml not in done_limit
        idx = ivd.build_interval_index(data, k_target=cfg["k_target"], k_alpha=cfg["k_alpha"], lloyd_iters=0,
                                       max_len=ml, keep_tile_logs=need_limit, block_covs=S.ep.block_covs())
        nb = idx.nbytes()
        tag_c = cfg_tag(cfg)
        storage.append(tag_c | {"build_s": idx.build_seconds, "n_nodes": idx.n_nodes(), "index_mb": nb["total"] / MB,
                                "means_mb": nb["means"] / MB, "codes_mb": nb["codes"] / MB})
        for q in queries:
            g, cov = q["genes"], S.cov_s(q)
            r = ivd.search_interval_index(idx, g, cov, score="mean")
            sc_mean = full_scores(n, r.rows, r.scores)
            score(q, "interval", tag_c, sc_mean, 1e3 * r.stats["seconds"])
            if need_limit:
                r = ivd.search_interval_index(idx, g, cov, score="exact")
                sc_lim = full_scores(n, r.rows, r.scores)
                score(q, "limit", dict(NO_CFG, max_len=tag_c["max_len"]), sc_lim, 1e3 * r.stats["seconds"])
                if ci == 0 and q["query"] < BOUND_QUERIES:
                    eb = ivd.error_bound(idx, g)
                    gap = np.abs(sc_mean - sc_lim)
                    for k, b in eb.items():
                        m = labels == k
                        bound.append({"query": q["query"], "family": q["family"], "size": q["size"], "niche": k,
                                      "bound": b, "n_tiles": int(m.sum()), "max_gap": float(gap[m].max()),
                                      "median_gap": float(np.median(gap[m])),
                                      "share_within": float(np.mean(gap[m] <= b * (1 + 1e-6) + 1e-9))})
        done_limit.add(ml)
        m = pd.DataFrame([x for x in rows if x["method"] == "interval" and x["c"] == c_frac[0]
                          and x["family"] == "program" and all(x[k] == v for k, v in tag_c.items())])
        print(f"{tag_c}: build {idx.build_seconds:.0f} s, {nb['total'] / MB:.1f} MB, programs "
              f"Overlap(5%, 5%) {m['overlap_0.05'].mean():.3f}", flush=True)
        del idx
        gc.collect()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = dict(dataset=short, stem=stem, seed=seed, n_tiles=n, n_genes=S.p)
    qdf.assign(**base).to_csv(OUT_DIR / f"{tag}_queries.csv", index=False)
    per = pd.DataFrame(rows)
    sel = per[per["c"].isin(c_frac)].copy()
    sel["c_target"] = sel["c"].map(dict(zip(c_frac, C_FRAC)))
    sel.assign(**base).to_csv(OUT_DIR / f"{tag}_per_query.csv", index=False)
    keys = ["method", "k_target", "k_alpha", "max_len", "family", "size", "c"]
    curves = per.drop(columns="query").groupby(keys, dropna=False).mean(numeric_only=True)
    curves["n_queries"] = per.groupby(keys, dropna=False)["query"].nunique()
    curves.reset_index().assign(**base).to_csv(OUT_DIR / f"{tag}_curves.csv", index=False)
    pd.DataFrame(bound).assign(**base).to_csv(OUT_DIR / f"{tag}_bound.csv", index=False)
    pd.DataFrame(storage).assign(
        **base, exact_block_logs_mb=ex_bytes["block_logs"] / MB, exact_block_covs_mb=ex_bytes["block_covs"] / MB,
        exact_partial_mb=(ex_bytes["block_logs"] + ex_bytes["block_covs"]) / MB,
        niche_means_mb=S.ep.niche_means_nbytes() / MB,
        exact_plus_means_mb=(ex_bytes["block_logs"] + S.ep.niche_means_nbytes()) / MB,
        whole_cov_upper_mb=4 * n * S.p * (S.p + 1) / 2 / MB,
    ).to_csv(OUT_DIR / f"{tag}_storage.csv", index=False)
    print("done", flush=True)


def timing(stem, short, seed, max_tiles):
    """Gene programs only, the paper's configuration; one BLAS thread, one job at a time."""
    S = Setup(stem, seed, max_tiles)
    cfg = CONFIGS[0]
    t0 = time.perf_counter()
    idx = ivd.build_interval_index(S.data, k_target=cfg["k_target"], k_alpha=cfg["k_alpha"], lloyd_iters=0,
                                   max_len=cfg["max_len"], block_covs=S.ep.block_covs())
    print(f"interval index built in {time.perf_counter() - t0:.0f} s", flush=True)
    queries = draw_queries(S.data, len(S.tests), S.genes_all, TISSUE[short], seed=seed, families=("program",))
    out = []
    for q in queries:
        g, cov = q["genes"], S.cov_s(q)
        base = {"query": q["query"], "size": q["size"]}
        t1 = time.perf_counter()
        S.ep.distances(cov, g)
        out.append(base | {"method": "exact_partial", "prep_ms": 0.0, "total_ms": 1e3 * (time.perf_counter() - t1)})
        r = ivd.search_interval_index(idx, g, cov)
        out.append(base | {"method": "interval", "prep_ms": 0.0, "total_ms": 1e3 * r.stats["seconds"]})
        _, prep, ql, scan = S.padding(cov, g)
        out.append(base | {"method": "padding", "prep_ms": 1e3 * prep, "query_log_ms": 1e3 * ql,
                           "scan_ms": 1e3 * scan, "total_ms": 1e3 * (prep + ql + scan)})
        _, prep, ql, scan, _ = S.imputation(cov, g)
        out.append(base | {"method": "imputation", "prep_ms": 1e3 * prep, "query_log_ms": 1e3 * ql,
                           "scan_ms": 1e3 * scan, "total_ms": 1e3 * (prep + ql + scan)})
    df = pd.DataFrame(out).assign(dataset=short, stem=stem, seed=seed, n_tiles=S.n)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{stem}_seed{seed}_timing.csv", index=False)
    print(df.groupby("method")["total_ms"].median().to_string(), flush=True)


def schematic(stem, short, seed):
    """The real query of Fig 4A: the first gene program of 12 genes (seed 73) that is one segment in its
    source niche and 3-5 segments in another niche; its pieces, radii and node counts in both niches, and
    the members, node means and radius of its longest piece in the source niche (2-D PCA)."""
    S = Setup(stem, seed, max_tiles=MAX_TILES, niche_means=False)
    data = S.data
    cfg = CONFIGS[0]
    idx = ivd.build_interval_index(data, k_target=cfg["k_target"], k_alpha=cfg["k_alpha"], lloyd_iters=0,
                                   max_len=cfg["max_len"], keep_tile_logs=True, block_covs=S.ep.block_covs())
    queries = draw_queries(data, len(S.tests), S.genes_all, TISSUE[short], seed=seed, families=("program",), sizes=[12])
    for q in queries:
        segs = segments_of(data, q["genes"])
        other = [k for k in sorted(segs) if k != q["source_niche"] and 3 <= segs[k] <= 5]
        if segs[q["source_niche"]] == 1 and other:
            k2 = other[0]
            break
    else:
        raise SystemExit("no query matches the selection rule")
    g, cov = q["genes"], S.cov_s(q)
    d = S.ep.distances(cov, g)
    out = {"dataset": short, "seed": seed, "query": int(q["query"]), "tile": int(q["tile"]),
           "genes": [S.genes_all[x] for x in g], "source_niche": int(q["source_niche"]), "other_niche": int(k2),
           "n_tiles": S.n, "params": cfg_tag(cfg), "niches": {}}
    for k in (int(q["source_niche"]), int(k2)):
        ni = idx.niches[k]
        segs_k = []
        for seg in ivd.query_segments(ni, g):
            blk = ni.blocks[seg.block]
            pieces = ivd.coarsest_decomposition(blk, seg)
            segs_k.append({"block": seg.block, "block_len": int(len(blk.genes)), "a": seg.a, "b": seg.b,
                           "pieces": [{"s": s, "a": a, "n_nodes": blk.intervals[(s, a)].n_nodes,
                                       "eps": blk.eps[s]} for s, a in pieces]})
        exact_blocks = [{"block": b, "block_len": int(len(gb)), "n_in_S": int(np.isin(gb, g).sum())}
                        for b, gb in enumerate(S.ep.blocks[k]) if np.isin(gb, g).any()]
        out["niches"][str(k)] = {"n_tiles": int(ni.n_tiles), "n_blocks": len(ni.blocks), "segments": segs_k,
                                 "exact_blocks": exact_blocks, "error_bound": ivd.error_bound(idx, g)[k]}
    # node-mean inset: the longest piece in the source niche
    k1 = int(q["source_niche"])
    ni = idx.niches[k1]
    seg0 = out["niches"][str(k1)]["segments"][0]
    piece = max(seg0["pieces"], key=lambda x: x["s"])
    blk = ni.blocks[seg0["block"]]
    iv, X = blk.intervals[(piece["s"], piece["a"])], blk.tile_logs[(piece["s"], piece["a"])].astype(np.float64)
    col = np.full(S.p, -1)
    col[g] = np.arange(len(g))
    qv = ivd.sym_to_vec(ivd.batched_log_spd(cov[np.ix_(col[blk.genes[piece["a"]:piece["a"] + piece["s"]]],
                                                       col[blk.genes[piece["a"]:piece["a"] + piece["s"]]])][None]))[0]
    mu = X.mean(0)
    _, _, Vt = np.linalg.svd(X - mu, full_matrices=False)
    P = Vt[:2].T
    out["inset"] = {"s": piece["s"], "a": piece["a"], "block": seg0["block"], "eps": piece["eps"],
                    "members_2d": ((X - mu) @ P).round(4).tolist(), "codes": iv.codes.tolist(),
                    "means_2d": ((iv.means.astype(np.float64) - mu) @ P).round(4).tolist(),
                    "query_2d": ((qv - mu) @ P).round(4).tolist()}
    out["exact_top10"] = [int(x) for x in np.argsort(d, kind="stable")[:10]]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "schematic_example.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "inset"}, indent=1)[:3000], flush=True)


def aggregate():
    stems = {ec.resolve_stem(k): k for k in ec.DATASETS}

    def load(suffix):
        fr = [pd.read_csv(f) for f in sorted(OUT_DIR.glob(f"*_seed[0-9]_{suffix}.csv"))]
        return pd.concat([f for f in fr if len(f) and f["seed"].iloc[0] in ec.MULTISEED], ignore_index=True)

    keys = ["method", "k_target", "k_alpha", "max_len", "family", "size"]
    cur = load("curves")
    # curves: per (dataset, seed) mean over queries (already), then mean / s.d. over seeds
    num = ["hit", "recall_top10"] + [c for c in cur.columns if c.startswith("overlap_")]
    g = cur.groupby(["dataset", "c"] + keys, dropna=False)
    curves = g[num].agg(["mean", "std"])
    curves.columns = [f"{a}_{b}" for a, b in curves.columns]
    curves["n_seeds"] = g["seed"].nunique()
    curves.reset_index().to_csv(OUT_DIR / "curves.csv", index=False)

    # by size at c = 5 % / 10 % of N: per dataset (mean +/- s.d. over seeds) and all datasets (per seed the
    # mean over datasets, then mean +/- s.d. over seeds)
    per = load("per_query")
    pk = ["method", "k_target", "k_alpha", "max_len", "family", "size", "c_target"]
    ds_seed = per.groupby(["dataset", "seed"] + pk, dropna=False)[["overlap_0.05", "hit", "recall_top10"]].mean()
    by_ds = ds_seed.groupby(["dataset"] + pk, dropna=False).agg(["mean", "std"])
    by_ds.columns = [f"{a}_{b}" for a, b in by_ds.columns]
    pooled = ds_seed.groupby(["seed"] + pk, dropna=False).mean().groupby(pk, dropna=False).agg(["mean", "std"])
    pooled.columns = [f"{a}_{b}" for a, b in pooled.columns]
    pd.concat([by_ds.reset_index(), pooled.reset_index().assign(dataset="all")]).to_csv(
        OUT_DIR / "by_size.csv", index=False)

    # c90 per (dataset, seed, method, config, family, size) from the mean curve, then over seeds
    rows = []
    ck = ["dataset", "seed", "method", "k_target", "k_alpha", "max_len", "family", "size"]
    for key, d in cur.groupby(ck, dropna=False):
        d = d.sort_values("c")
        c90 = de.curve_c(d["c"], d["overlap_0.05"])
        rows.append(dict(zip(ck, key)) | {"c90": c90, "c90_pct": 100 * c90 / d["n_tiles"].iloc[0]})
    c90 = pd.DataFrame(rows).groupby([k for k in ck if k != "seed"], dropna=False)[["c90", "c90_pct"]].agg(["mean", "std"])
    c90.columns = [f"{a}_{b}" for a, b in c90.columns]
    c90.reset_index().to_csv(OUT_DIR / "summary.csv", index=False)

    st = load("storage")
    num = st.select_dtypes("number").columns.drop(["seed", "k_target", "k_alpha", "max_len"])
    g = st.groupby(["dataset", "k_target", "k_alpha", "max_len"])
    s = g[list(num)].agg(["mean", "std"])
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s.reset_index().to_csv(OUT_DIR / "storage.csv", index=False)

    # S8: error bound (share of tiles within it, largest gap / bound) and query structure per dataset
    b = load("bound")
    b["gap_over_bound"] = b["max_gap"] / b["bound"]
    b.groupby("dataset").agg(share_within_min=("share_within", "min"), share_within_mean=("share_within", "mean"),
                             gap_over_bound_max=("gap_over_bound", "max"),
                             gap_over_bound_median=("gap_over_bound", "median"),
                             n=("query", "size")).reset_index().to_csv(OUT_DIR / "bound_summary.csv", index=False)
    q = load("queries")
    q["pred_is_nn_niche"] = (q["pred_niche"] == q["nn_niche"]).astype(float)
    q["pred_is_source"] = np.where(q["source_niche"].notna(), (q["pred_niche"] == q["source_niche"]).astype(float), np.nan)
    qs = q.groupby(["dataset", "seed", "family", "size"])[
        ["seg_frac_mean", "seg_nn_niche", "near5_size", "pred_is_nn_niche", "pred_is_source"]].mean()
    qs = qs.groupby(["dataset", "family", "size"]).agg(["mean", "std"])
    qs.columns = [f"{a}_{b}" for a, b in qs.columns]
    qs.reset_index().to_csv(OUT_DIR / "query_structure.csv", index=False)

    tm = [pd.read_csv(f) for f in sorted(OUT_DIR.glob("*_seed[0-9]_timing.csv"))]
    if tm:
        tm = pd.concat(tm, ignore_index=True)
        med = tm.groupby(["dataset", "seed", "method"])["total_ms"].median()
        ts = med.groupby(["dataset", "method"]).agg(["mean", "std", "count"]).reset_index()
        ts.to_csv(OUT_DIR / "timing_summary.csv", index=False)
    order = {k: i for i, k in enumerate(stems.values())}
    p = by_ds.reset_index()
    main_cfg = (((p.method == "interval") & (p.k_alpha == 0.5) & (p.max_len == 16)) | ((p.method == "limit") & (p.max_len == 16))
                | p.method.isin(["padding", "imputation"]))
    p = p[(p.family == "program") & (p.c_target == 0.05) & main_cfg]
    print(p.pivot_table(index=["dataset", "size"], columns="method", values="overlap_0.05_mean")
          .sort_index(key=lambda s: s.map(order) if s.name == "dataset" else s).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", help="short key or stem")
    ap.add_argument("--seed", type=int, default=ec.PRODUCTION_SEED)
    ap.add_argument("--max-tiles", type=int, default=MAX_TILES, help="held-out tiles used as queries")
    ap.add_argument("--max-queries", type=int, default=None, help="pilot: first queries only")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--schematic", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        return aggregate()
    stem = ec.resolve_stem(args.dataset)
    short = next(k for k, v in ec.DATASETS.items() if v == stem)
    if args.timing:
        timing(stem, short, args.seed, args.max_tiles)
    elif args.schematic:
        schematic(stem, short, args.seed)
    else:
        run(stem, short, args.seed, args.max_tiles, args.max_queries)


if __name__ == "__main__":
    main()
