"""Shared pieces for the DAG compactness / scaling experiments.

The DAG is evaluated as a *retrieval* approximator of the exact tier: it returns
an unordered set of c tiles (never re-ranked) and succeeds when that set holds
the exact tier's near set. It is deployed alone, without the block logs.

* Exact distance: the L2 block log-Euclidean distance

      d_B(q, t) = sqrt(sum_b ||log q_b - log t_b||_F^2) / sqrt(p)

  over the niche's permuted gene blocks; every tile is scored in its own
  niche's layout and pooled over niches (the query is projected into each).
* Compact DAG: per niche and block, the **node means** (average of the members'
  block logs) as float32 vectors (upper triangle, off-diagonals x sqrt 2, / sqrt(p),
  so squared L2 is that block's term of d_B^2), one uint8/uint16 code per
  (tile, block), CSR edges and float32 radii. ``nbytes`` is the DAG's whole footprint.
* Configuration: K = 32 nodes per block, alpha = 0, k_min = 32 (DAG_CFG = tiers.DAG_DEFAULTS).
* Covariances are shrunk (tiers.prepare_cov) by the tiers; score_tiers scores spindle_dev.tiers objects.
* DAG result set: approximate d_B^2 = sum_b ||x_b - C_b[code(t, b)]||^2 (one
  distance table per block, then a gather-sum over the codes); the c tiles
  with the smallest value, pooled over niches.
* Overlap@delta: the fraction of the exact near set {t : d_B(t) <= (1 + delta) d*}
  inside the result set.
* Fig 3 retrieval readouts (retrieval_rows, c_needed, curve_c): per c, Overlap(delta, c), hit, recall of
  the exact top 10, the distance ratio (mean exact distance of the returned c over that of the exact top c;
  1 = as close as the exact answer) and the tolerant precision (share of the returned c within 5 % of the
  exact c-th distance); c90 = the smallest c with mean Overlap(5 %, c) >= 0.9. They score any method that
  returns rows in order (a DAG, or a competitor's list), against any exact distance.
"""

from __future__ import annotations

import numpy as np

import experiment_common as ec  # noqa: F401  (sets up sys.path for spindle_dev)

DELTAS = [0.01, 0.025, 0.05, 0.10]
C_FRAC = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
C_ABS = [10, 50, 100]
MB = 1024 ** 2
DAG_CFG = dict(k_target=32, alpha=0.0, k_min=32)


# Library helpers (src/spindle_dev/tiers.py) used here and re-exported for metric_check.py.
from spindle_dev.tiers import ADC, block_log_bytes, build_dag_from_block_logs, n_nodes, nbytes, niche_layouts  # noqa: E402,F401


def c_specs(n):
    """[(kind, spec, c)] -- fractions of N and absolute counts, each clipped to [1, N]."""
    return ([("frac", f, int(min(n, max(1, round(f * n))))) for f in C_FRAC]
            + [("abs", a, int(min(n, a))) for a in C_ABS])


def overlap_rows(d_exact, ids, approx, n):
    """One dict per c spec: overlap@delta, near-set sizes and hit for the DAG's top-c set."""
    order = ids[np.argsort(approx, kind="stable")]
    j = int(np.argmin(d_exact))
    near = {dl: np.flatnonzero(d_exact <= (1 + dl) * d_exact[j]) for dl in DELTAS}
    rows = []
    for kind, spec, c in c_specs(n):
        member = np.zeros(n, bool)
        member[order[:c]] = True
        rec = {"c_kind": kind, "c_spec": spec, "c": c, "c_frac": c / n, "hit": float(member[j])}
        for dl, m in near.items():
            rec[f"near_size_{dl}"] = len(m)
            rec[f"overlap_{dl}"] = float(member[m].mean())
        rows.append(rec)
    return rows


C_TARGET = 0.9   # c90: mean Overlap(5 %, c) >= C_TARGET
K_TOP = 10


def c_grid(n, n_log=40):
    """c values of the retrieval curves: ~n_log log-spaced counts in [1, N] plus every c_specs value."""
    g = np.round(np.logspace(0, np.log10(n), n_log)).astype(int)
    return np.unique(np.clip(np.concatenate([g, [c for _, _, c in c_specs(n)]]), 1, n))


def ranks_of(order, n):
    """Position of every tile in a returned order; tiles not returned get n (never inside any c <= N)."""
    rank = np.full(n, n, dtype=np.int64)
    rank[np.asarray(order)] = np.arange(len(order))
    return rank


def retrieval_rows(d_exact, order, cs, deltas=DELTAS):
    """One dict per c: the Fig 3 readouts of the first c rows of ``order`` against exact distances d_exact.

    ``order`` lists training rows best first (all N, or a method's top max(cs) only)."""
    n = len(d_exact)
    eo = np.argsort(d_exact, kind="stable")
    ds = d_exact[eo]
    cum = np.cumsum(ds)
    rank = ranks_of(order, n)
    near = {dl: np.flatnonzero(d_exact <= (1 + dl) * ds[0]) for dl in deltas}
    out = []
    for c in cs:
        S = np.asarray(order[:c])
        member = rank < c
        rec = {"c": int(c), "c_frac": c / n, "n_returned": len(S), "hit": float(member[eo[0]]),
               f"recall_top{K_TOP}": float(member[eo[:K_TOP]].mean()),
               "dist_ratio": float(d_exact[S].mean() / (cum[c - 1] / c)) if len(S) else np.nan,
               "tol_prec_0.05": float(np.mean(d_exact[S] <= 1.05 * ds[c - 1])) if len(S) else np.nan}
        for dl, m in near.items():
            rec[f"overlap_{dl}"] = float(member[m].mean())
        out.append(rec)
    return out


def retrieval_rows_tied(d_exact, scores, cs, deltas=DELTAS):
    """retrieval_rows for a method that scores every tile (``scores``, lower = better; np.inf = not returned)
    and may tie: the top c holds every tile below the c-th score and a random share of the tiles tied with it,
    so each readout is its expected value under random tie-breaking. Overlap(delta, c), hit and recall of
    the exact top 10 per c."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(d_exact)
    eo = np.argsort(d_exact, kind="stable")
    near = {dl: np.flatnonzero(d_exact <= (1 + dl) * d_exact[eo[0]]) for dl in deltas}
    sets = {"hit": eo[:1], f"recall_top{K_TOP}": eo[:K_TOP]} | {f"overlap_{dl}": m for dl, m in near.items()}
    srt = np.sort(scores)
    cs = np.asarray(cs)
    thr = srt[cs - 1]
    below = np.searchsorted(srt, thr, side="left")
    tied = np.searchsorted(srt, thr, side="right") - below
    share = (cs - below) / tied  # chance that a tile tied at the c-th score is in the top c
    out = [{"c": int(c), "c_frac": c / n} for c in cs]
    for name, members in sets.items():
        s = scores[members][:, None]
        prob = (s < thr[None, :]) + (s == thr[None, :]) * share[None, :]
        for rec, v in zip(out, prob.mean(0)):
            rec[name] = float(v)
    return out


def c_needed(d_exact, order, frac=C_TARGET, delta=0.05):
    """Per query: the smallest c whose first c rows hold >= frac of the exact delta-near set, and of the
    exact top 10 (NaN if the order never gets there)."""
    n = len(d_exact)
    rank = ranks_of(order, n)
    eo = np.argsort(d_exact, kind="stable")
    out = {}
    for name, members in ((f"near_size_{delta}", np.flatnonzero(d_exact <= (1 + delta) * d_exact[eo[0]])),
                          (f"top{K_TOP}", eo[:K_TOP])):
        rk = np.sort(rank[members])
        need = rk[int(np.ceil(frac * len(rk))) - 1] + 1
        key = f"c{int(round(100 * frac))}_" + ("near" if name.startswith("near") else name)
        out[key] = float(need) if need <= n else np.nan
        if name.startswith("near"):
            out[name] = len(members)
    return out


def curve_c(cs, mean_overlap, target=C_TARGET):
    """The c where a mean-overlap curve first reaches ``target`` (interpolated on log c); NaN if never."""
    cs, y = np.asarray(cs, float), np.asarray(mean_overlap, float)
    hit = np.flatnonzero(y >= target)
    if not len(hit):
        return np.nan
    i = hit[0]
    if i == 0 or y[i] == y[i - 1]:
        return float(cs[i])
    t = (target - y[i - 1]) / (y[i] - y[i - 1])
    return float(np.exp(np.log(cs[i - 1]) + t * (np.log(cs[i]) - np.log(cs[i - 1]))))


def peak_rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # Linux: KB


class MemSampler:
    """Peak anonymous and total resident memory of this process, sampled from /proc/self/status.

    ru_maxrss also counts touched pages of memory-mapped files (e.g. an on-disk feature cache), which
    the kernel can drop at any time; RssAnon is the memory the process itself allocated. Child
    processes (joblib workers) are not included. Use as ``with MemSampler() as m: ...; m.anon_gb``.
    """

    def __init__(self, interval=0.5):
        self.interval, self.anon_gb, self.rss_gb = interval, 0.0, 0.0

    def sample(self):
        """Fold the current reading into the peaks (call before reading anon_gb / rss_gb mid-run)."""
        self._read()

    def _read(self):
        vals = {}
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith(("RssAnon:", "VmRSS:")):
                    key, kb = line.split()[:2]
                    vals[key] = int(kb) / 1024 ** 2
        self.anon_gb = max(self.anon_gb, vals.get("RssAnon:", 0.0))
        self.rss_gb = max(self.rss_gb, vals.get("VmRSS:", 0.0))

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._read()

    def __enter__(self):
        import threading

        self._stop = threading.Event()
        self._read()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self._read()
        return False


def score_tiers(exact, dag, query_covs, query_hook=None):
    """score_dag for spindle_dev.tiers objects: the DAG tier's result sets against the exact tier,
    both on shrunk covariances (tiers.prepare_cov). -> (per-query, per-c rows; extra rows)."""
    rows, extra_rows = [], []
    for qi, qc in enumerate(query_covs):
        d = exact.distances(exact._query(qc, None))
        if query_hook is not None:
            extra_rows.append({"query": qi, **query_hook(qi, d)})
        ids, approx = dag.adc.scores(dag._query(qc, None))
        for rec in overlap_rows(d, ids, approx, exact.n):
            rows.append({"query": qi, **rec})
    return rows, extra_rows


def dag_info(compact, data):
    dag_b, logs_b = nbytes(compact), block_log_bytes(data)
    return {"n_niches": len(compact), "n_blocks": int(sum(c["codes"].shape[1] for c in compact.values())),
            "n_nodes": n_nodes(compact), "dag_mb": dag_b / MB, "block_logs_mb": logs_b / MB,
            "block_logs_over_dag": logs_b / dag_b}


def summarize(per_q, base):
    """One summary row: mean overlap / hit per c spec, and c/N needed for Overlap@5% >= 0.95."""
    import pandas as pd

    rec = {**base, "n_queries": int(per_q["query"].nunique())}
    m = per_q.groupby(["c_kind", "c_spec"]).mean(numeric_only=True)
    for (kind, spec), r in m.iterrows():
        tag = f"{100 * spec:g}pct" if kind == "frac" else f"c{int(spec)}"
        rec[f"hit@{tag}"] = r["hit"]
        for dl in DELTAS:
            rec[f"overlap{100 * dl:g}@{tag}"] = r[f"overlap_{dl}"]
    for dl in DELTAS:
        rec[f"near{100 * dl:g}_median"] = float(per_q[f"near_size_{dl}"].median())
    frac = m.loc["frac"].sort_index()
    ok = frac[frac["overlap_0.05"] >= 0.95]
    rec["cfrac_for_overlap5_ge_0.95"] = float(ok.index[0]) if len(ok) else np.nan
    return pd.DataFrame([rec])


def pseudobulk(X, tile_list):
    """Mean expression per tile over the columns of X (cells x index genes)."""
    import scipy.sparse as sp

    out = np.empty((len(tile_list), X.shape[1]))
    for i, t in enumerate(tile_list):
        m = X[t.idx].mean(axis=0)
        out[i] = np.asarray(m).ravel() if sp.issparse(X) else m
    return out


def zscore_pair(E_train, E_query):
    mu, sd = E_train.mean(0), E_train.std(0) + 1e-9
    return (E_train - mu) / sd, (E_query - mu) / sd


def block_fit(cov, perm, runs):
    """Share of a tile's squared off-diagonal correlation that lies inside its niche's blocks."""
    c = np.asarray(cov, dtype=np.float64)[np.ix_(perm, perm)]
    sd = np.sqrt(np.clip(np.diag(c), 1e-12, None))
    r2 = (c / np.outer(sd, sd)) ** 2
    np.fill_diagonal(r2, 0.0)
    inside = sum(r2[s:e, s:e].sum() for s, e in runs)
    total = r2.sum()
    return float(inside / total) if total > 0 else np.nan


def niche_stats(data, train_covs, max_tiles=2000, seed=0):
    """Niche count/sizes and mean block fit (over at most max_tiles random training tiles)."""
    labels = np.asarray(data.labels).astype(int)
    sizes = np.bincount(labels)
    sel = np.random.default_rng(seed).choice(len(labels), min(max_tiles, len(labels)), replace=False)
    fit = [block_fit(ec.raw_cov(train_covs[i]), data.perm_list[labels[i]], data.block_dict[labels[i]]) for i in sel]
    return {"niche_max": int(sizes.max()), "niche_median": float(np.median(sizes)),
            "block_fit_mean": float(np.nanmean(fit))}
