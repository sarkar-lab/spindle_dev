"""Indexes over whole covariances vs Spindle-DAG: memory vs retrieval fidelity (Fig 3D; S6).

Splitting a tile's covariance into niche blocks is part of Spindle, so the competitors index what a
user without Spindle would index: the whole-matrix log of every training tile, as the vector of
tiers.WholeTier (float32 upper triangle, off-diagonals x sqrt 2, / sqrt p; shrunk covariances), whose L2
distance is the whole-matrix log-Euclidean distance d_W. Each competitor is scored against its own
exact metric (d_W, the WholeTier scan); Spindle-DAG against its own (d_B, the ExactTier scan). Scored
against d_B instead (S6 only, labelled): every method, as ``reference = d_B``.

Methods (each a point on a memory axis; stored MB counts every array a query needs):
  hnsw            FAISS IndexHNSWFlat on the whole vectors (M = 32, efConstruction 200,
                  efSearch = max(400, C_MAX)); returns its top C_MAX. MB = serialized index.
  pca<D>_flat     PCA to D in {16, 64, 256} (randomized SVD of the training vectors), exact scan;
                  MB = codes N x D + projection D x d + mean.
  pca256_pq<m>    PCA-256, then product quantization (m in {8, 32} sub-quantizers, 8 bits); MB = the
                  serialized PQ (codes + codebooks) + projection.
  PQ / IVF-PQ on the raw whole vectors is not run: its codebooks alone are 256 x d x 4 bytes (50-118 MB),
  above the PCA-256 + PQ variant at equal codes.
Spindle rows (same queries, for the biology step and the d_B / d_W cross-scoring):
  exact (block top c), whole (WholeTier top c), dag (K = 32).

Every held-out tile is a query. Readouts per (method, reference, c) are dag_eval_common.retrieval_rows
averaged over queries; c90 = dag_eval_common.curve_c on Overlap(5 %, c). Across metrics the
comparison uses c90_top10 (the c where mean recall of the reference's exact top 10 reaches 0.9): the 5 %-near
set of d_W is several times larger than that of d_B (whole-matrix distances are more concentrated), so the
delta-based c90 is not comparable between d_B and d_W. Query time (single-threaded, supplementary only) includes the query's own log (block or whole).

Outputs (results/whole_cov_baselines/):
  <stem>_seed<s>_curves.csv    method x reference x c (means over queries)
  <stem>_seed<s>_methods.csv   per method x reference: MB, build s, query ms, c90, Overlap(5 %, 5 %),
                               distance ratio at c = 10
  <stem>_seed<s>_returned.npz  per method: top-100 training rows per query (input of neighbour_biology.py)
  --aggregate: methods.csv (mean +/- s.d. over seeds 0-4), curves.csv

  sbatch slurm_jobs/whole_tile/run_whole_cov_baselines.sbatch <dataset> <seed> | --aggregate
"""

import argparse
import os
import time
import sys
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import dag_eval_common as de
import experiment_common as ec
from spindle_dev import tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "whole_cov_baselines"
C_MAX = 1000          # hnsw returns its top C_MAX (the other methods return every tile)
TOP_SAVE = 100        # rows kept per query for the biology step
HNSW_M, HNSW_EF_CONSTRUCTION, HNSW_EF_SEARCH = 32, 200, 400
PCA_DIMS = [16, 64, 256]
PQ_M = [8, 32]


def mb(nbytes):
    return nbytes / de.MB


class Method:
    """A competitor: fit on training vectors X (float32), then order(q_vec) -> rows best first."""

    name = ""

    def nbytes(self):
        raise NotImplementedError


class Hnsw(Method):
    def __init__(self, X, n_threads):
        self.name = "hnsw"
        faiss.omp_set_num_threads(n_threads)
        self.index = faiss.IndexHNSWFlat(X.shape[1], HNSW_M)
        self.index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
        self.index.add(X)
        self.k = min(C_MAX, X.shape[0])
        self.index.hnsw.efSearch = max(HNSW_EF_SEARCH, self.k)

    def nbytes(self):
        return faiss.serialize_index(self.index).nbytes

    def order(self, x):
        _, I = self.index.search(x[None, :], self.k)
        return I[0][I[0] >= 0]


class Pca:
    """Top-D principal directions of the centred training vectors (shared by the PCA methods)."""

    def __init__(self, X, D, seed):
        self.mean = X.mean(axis=0)
        _, _, Vt = randomized_svd(X - self.mean, n_components=D, random_state=seed)
        self.W = np.ascontiguousarray(Vt.T, dtype=np.float32)

    def project(self, X):
        return np.ascontiguousarray((X - self.mean) @ self.W, dtype=np.float32)

    def nbytes(self):
        return self.W.nbytes + self.mean.nbytes


class PcaFlat(Method):
    def __init__(self, pca, Z, D):
        self.name, self.pca, self.D = f"pca{D}_flat", pca, D
        self.Z = np.ascontiguousarray(Z[:, :D])

    def nbytes(self):
        return self.Z.nbytes + self.pca.W[:, :self.D].nbytes + self.pca.mean.nbytes

    def order(self, x):
        z = (x - self.pca.mean) @ self.pca.W[:, :self.D]
        return np.argsort(((self.Z - z) ** 2).sum(1), kind="stable")


class PcaPq(Method):
    def __init__(self, pca, Z, m, n_threads):
        self.name, self.pca = f"pca{Z.shape[1]}_pq{m}", pca
        faiss.omp_set_num_threads(n_threads)
        self.index = faiss.IndexPQ(Z.shape[1], m, 8)
        self.index.train(Z)
        self.index.add(Z)
        self.n = Z.shape[0]

    def nbytes(self):
        return faiss.serialize_index(self.index).nbytes + self.pca.nbytes()

    def order(self, x):
        z = self.pca.project(x[None, :])
        _, I = self.index.search(z, self.n)
        return I[0][I[0] >= 0]


def run(stem, seed, max_queries):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    tag = f"{stem}_seed{seed}"
    data = ec.load_index(stem, seed)["data"]
    covs = ec.load_raw_covs(stem, seed)
    train = covs["train_tile_covs"]
    queries = [ec.raw_cov(c) for c in covs["test_tile_covs"]][:max_queries]
    t0 = time.perf_counter()
    exact = tiers.build_tier("exact", data, train)
    dag = tiers.build_tier("dag", data, exact=exact)
    whole = tiers.build_tier("whole", None, train)
    del covs, train
    X, n = whole.X, whole.n
    print(f"{tag}: N = {n}, d = {X.shape[1]}, {len(queries)} queries; tiers {time.perf_counter() - t0:.0f} s",
          flush=True)

    methods, build_s = [], {}

    def add(make, name):
        t = time.perf_counter()
        m = make()
        build_s[m.name] = time.perf_counter() - t + (pca_s if name.startswith("pca") else 0.0)
        methods.append(m)
        print(f"  built {m.name}: {mb(m.nbytes()):.2f} MB in {build_s[m.name]:.0f} s", flush=True)

    pca_s = 0.0
    add(lambda: Hnsw(X, n_threads), "hnsw")
    t = time.perf_counter()
    pca = Pca(X, min(max(PCA_DIMS), n - 1), seed)
    Z = pca.project(X)
    pca_s = time.perf_counter() - t
    for D in PCA_DIMS:
        if D <= Z.shape[1]:
            add(lambda D=D: PcaFlat(pca, Z, D), "pca")
    for m in PQ_M:
        if Z.shape[1] % m == 0 and n >= 256:
            add(lambda m=m: PcaPq(pca, Z, m, n_threads), "pca")
    faiss.omp_set_num_threads(1)

    cs = de.c_grid(n)
    names = ["exact", "whole", "dag"] + [m.name for m in methods]
    curves = {(nm, ref): [] for nm in names for ref in ("own", "d_B", "d_W")}
    times = {nm: [] for nm in names}
    whole_log_s = []  # the query's whole log, which every whole-vector method needs first
    returned = {nm: np.full((len(queries), TOP_SAVE), -1, np.int64) for nm in names}
    for qi, q in enumerate(queries):
        orders = {}
        t = time.perf_counter()
        qv_b = exact._query(q, None)
        d_b = exact.distances(qv_b)
        orders["exact"] = np.argsort(d_b, kind="stable")
        times["exact"].append(time.perf_counter() - t)
        t = time.perf_counter()
        x = whole._query(q, None)
        whole_log_s.append(time.perf_counter() - t)
        d_w = whole.distances(x)
        orders["whole"] = np.argsort(d_w, kind="stable")
        times["whole"].append(time.perf_counter() - t)
        t = time.perf_counter()
        ids, approx = dag.adc.scores(dag._query(q, None))
        orders["dag"] = ids[np.argsort(approx, kind="stable")]
        times["dag"].append(time.perf_counter() - t)
        for m in methods:
            t = time.perf_counter()
            orders[m.name] = m.order(x)
            times[m.name].append(time.perf_counter() - t + whole_log_s[-1])
        for nm, o in orders.items():
            k = min(TOP_SAVE, len(o))
            returned[nm][qi, :k] = o[:k]
            own = d_b if nm in ("exact", "dag") else d_w
            curves[(nm, "own")].append(de.retrieval_rows(own, o, cs, deltas=[0.05]))
            curves[(nm, "d_B")].append(de.retrieval_rows(d_b, o, cs, deltas=[0.05]))
            curves[(nm, "d_W")].append(de.retrieval_rows(d_w, o, cs, deltas=[0.05]))
        if qi % 100 == 0:
            print(f"  query {qi}", flush=True)

    rows, mrows = [], []
    sizes = {"exact": exact.nbytes(), "whole": whole.nbytes(), "dag": dag.nbytes(),
             **{m.name: m.nbytes() for m in methods}}
    builds = {"exact": np.nan, "whole": np.nan, "dag": dag.build_seconds, **build_s}
    for (nm, ref), per_q in curves.items():
        df = pd.concat([pd.DataFrame(r) for r in per_q]).groupby("c").mean(numeric_only=True).reset_index()
        df.insert(0, "reference", ref)
        df.insert(0, "method", nm)
        rows.append(df)
        m = df.set_index("c")
        c5 = int(min(n, max(1, round(0.05 * n))))
        c90 = de.curve_c(m.index, m["overlap_0.05"])
        mrows.append({"method": nm, "reference": ref, "mb": mb(sizes[nm]), "build_s": builds[nm],
                      "query_ms": 1e3 * np.mean(times[nm]),
                      "c90": c90, "c90_pct": 100 * c90 / n,
                      "c90_top10": de.curve_c(m.index, m[f"recall_top{de.K_TOP}"]), "overlap5@5pct": m.loc[c5, "overlap_0.05"],
                      "dist_ratio@c10": m.loc[min(10, n), "dist_ratio"],
                      "recall_top10@c100": m.loc[min(100, n), f"recall_top{de.K_TOP}"],
                      "max_c_returned": int(m["n_returned"].max())})
    base = {"dataset": stem, "seed": seed, "n_tiles": n, "dim": X.shape[1], "n_queries": len(queries)}
    curves_df = pd.concat(rows, ignore_index=True)
    for k, v in reversed(list(base.items())):
        curves_df.insert(0, k, v)
    curves_df.to_csv(OUT_DIR / f"{tag}_curves.csv", index=False)
    mdf = pd.DataFrame([{**base, **r} for r in mrows])
    mdf.to_csv(OUT_DIR / f"{tag}_methods.csv", index=False)
    np.savez_compressed(OUT_DIR / f"{tag}_returned.npz", **returned)
    print(mdf[mdf["reference"] == "own"][["method", "mb", "c90", "c90_pct", "overlap5@5pct", "dist_ratio@c10",
                                          "query_ms"]].round(3).to_string(index=False), flush=True)


def aggregate():
    files = [f for f in sorted(OUT_DIR.glob("*_seed[0-9]_methods.csv"))]
    if not files:
        raise SystemExit("nothing to aggregate")
    df = pd.concat([pd.read_csv(f) for f in files])
    df = df[df["seed"].isin(ec.MULTISEED)]
    num = [c for c in df.select_dtypes("number").columns if c not in ("seed",)]
    g = df.groupby(["dataset", "method", "reference"])
    out = g[num].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    out.reset_index().to_csv(OUT_DIR / "methods.csv", index=False)
    cur = pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob("*_seed[0-9]_curves.csv"))])
    cur = cur[cur["seed"].isin(ec.MULTISEED)]
    g = cur.groupby(["dataset", "method", "reference", "c"])
    cv = g.mean(numeric_only=True).drop(columns="seed")
    cv["overlap_0.05_std"] = g["overlap_0.05"].std()
    cv.reset_index().to_csv(OUT_DIR / "curves.csv", index=False)
    print(out.reset_index().query("reference == 'own'")[["dataset", "method", "mb_mean", "c90_pct_mean",
                                                         "overlap5@5pct_mean"]].round(3).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--aggregate", action="store_true")
    args = ap.parse_args()
    if args.aggregate:
        aggregate()
    else:
        run(ec.resolve_stem(args.dataset), args.seed, args.max_queries)


if __name__ == "__main__":
    main()
