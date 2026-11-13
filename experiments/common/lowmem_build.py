"""Low-memory build: the files of index_cache.build (same values, bit for bit on the same machine), with the per-tile
matrices on local disk instead of in RAM, so that the largest dataset (brain, ~52 GB in memory) builds in < 16 GB.

  python experiments/common/lowmem_build.py build <key> <seed> [--scratch DIR]    writes data.cache_dir(key, seed)
  python experiments/common/lowmem_build.py compare <key> <seed> --ref <reference cache root>
  python experiments/common/lowmem_build.py compare-stage <key> --ref <reference cache root>   (index_cache.py stage)

What differs from index_cache.build (src/ is untouched; only where the data live):
  covariances      computed batch by batch (index_cache.iter_tile_covs: every tile's covariance is computed alone,
                   so batching changes nothing) and written to a memory-mapped file of the training tiles; the
                   held-out tiles (10 %) stay in memory
  tree features    the PCA input of ProcessedData.reduce_dim (one ultrametric upper triangle per training tile, float64)
                   written row by row to a memory-mapped file; sklearn's randomized PCA runs on it with copy=False, which
                   centres it in place instead of in a copy (the same arithmetic; copy is reset to True afterwards)
  U_list           not held: layout.U_list[i] recomputes the tile's ultrametric when a niche's consensus tree needs it
  omega            index_omega: variance_quantiles' diagonals and quantile without its float64 copy of every matrix
                   (np.diag(as_cov(c)) is a view that keeps the copy alive until the stack: 17.6 GB for brain)
  the rest         ProcessedData's own methods (cluster_spds, assign_label_to_spots, get_corr_mean_by_cluster,
                   get_adaptive_runs) and the four index constructors, given the memory-mapped training covariances
The scratch files (brain: 8.8 GB covariances + 8.8 GB tree features) go to --scratch, by default a new directory in the
system temp dir (keep it on a local disk: NFS makes the PCA's passes slow), and are deleted at the end.

build.json gets the usual stage times and ru_maxrss peaks (which also count the mapped files' pages, so they overstate
the need) plus peak_anon_gb_<stage>: the sampled peak of anonymous (non-file) resident memory, the memory the build needs.
Bit-identity holds against a build on the same machine and libraries: BLAS picks its kernels by CPU.
"""

import argparse
import inspect
import json
import os
import resource
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402
import index_cache  # noqa: E402
import spindle_dev as sd  # noqa: E402
from spindle_dev.layout import spd_to_ultrametric  # noqa: E402
from spindle_dev.queries import VAR_QUANTILE  # noqa: E402

# ---------------------------------------------------------------- memory
def cgroup_memory():
    """memory.max / memory.peak (GB) of every cgroup above this process that has them (Slurm job, step), if any."""
    try:
        rel = open("/proc/self/cgroup").read().strip().split("::")[-1]
    except OSError:
        return {}
    out, p = {}, Path("/sys/fs/cgroup" + rel)
    while p != Path("/sys/fs/cgroup") and p != p.parent:
        for f in ("memory.max", "memory.peak"):
            try:
                v = (p / f).read_text().strip()
            except OSError:
                continue
            out[f"{p.name}/{f}"] = v if v == "max" else round(int(v) / 2**30, 2)
        p = p.parent
    return out


# ---------------------------------------------------------------- layout on disk
class LazyUltrametrics:
    """``layout.U_list`` without holding it: item i is spd_to_ultrametric(covs[i])[0], recomputed on access."""

    def __init__(self, covs):
        self.covs = covs

    def __len__(self):
        return len(self.covs)

    def __getitem__(self, i):
        return spd_to_ultrametric(self.covs[i])[0]


def layout_defaults():
    """build_layout's keyword defaults (resolution, min/max_final_size, max_niche_size, random_state)."""
    return {k: v.default for k, v in inspect.signature(sd.build_layout).parameters.items()
            if v.default is not inspect.Parameter.empty}


def stream_layout(tiles, C, tile_ids, genes, num_spots, scratch):
    """sd.build_layout(tiles, covs, genes, num_spots) with ``C`` the (n, p, p) float32 training covariances as an ndarray
    over a memory-mapped file: build_layout and ProcessedData.reduce_dim step by step, the PCA input on disk."""
    from sklearn.decomposition import PCA
    from umap import UMAP

    kw = layout_defaults()
    lay = sd.ProcessedData(tiles, [], genes, num_spots)
    lay.spd_matrices, lay.spd_ids = C, list(tile_ids)

    # reduce_dim(num_pca_components = max(2, min(30, n - 1)), n_components = 2, random_state)
    n, p = C.shape[0], C.shape[1]
    path = scratch / "tree_features.f64"
    F = np.memmap(path, dtype=np.float64, mode="w+", shape=(n, p * (p - 1) // 2))
    for i in range(n):  # = spd_tree_feature_matrix(build_ultrametrics(C)[0])
        F[i] = squareform(spd_to_ultrametric(C[i])[0], checks=False)
    F.flush()
    pca = PCA(n_components=max(2, min(30, len(tiles) - 1)), random_state=kw["random_state"], copy=False)
    lay.latent["pca"] = pca.fit_transform(np.asarray(F))  # centres (then squares) F in place
    pca.set_params(copy=True)  # as reduce_dim's PCA; copy only chose where the centring happened
    lay.pca_model = pca
    del F
    path.unlink()
    lay.U_list = LazyUltrametrics(C)
    umap = UMAP(n_components=2, random_state=kw["random_state"], metric="euclidean", n_neighbors=15, min_dist=0.1)
    lay.latent["umap"] = umap.fit_transform(lay.latent["pca"])

    lay.cluster_spds(resolution=kw["resolution"], max_niche_size=kw["max_niche_size"], random_state=kw["random_state"])
    lay.assign_label_to_spots()
    lay.get_corr_mean_by_cluster()  # C is already an ndarray: no copy
    lay.get_adaptive_runs(min_final_size=kw["min_final_size"], max_final_size=kw["max_final_size"])
    return lay


# ---------------------------------------------------------------- build
def index_omega(C):
    """sd.variance_quantiles(C), the same values: each diagonal in float64 (= the diagonal of as_cov(c)), stacked in the
    same order, the same quantile; without holding a float64 copy of every matrix."""
    return np.quantile(np.stack([np.diag(c).astype(np.float64) for c in C]), VAR_QUANTILE, axis=0)


def stream_covs(adata, max_genes, seed, scratch):
    """prepare_tiles(adata, max_genes) batch by batch, then the seed's split: (tiles, genes, train, test, C, test_covs,
    checksum) with C the training covariances (train order) as an ndarray over a file in ``scratch``, test_covs the
    held-out tiles' {'tile_id', 'cov', 'n'} (test order) and checksum = data.cov_checksum of all covariances."""
    tiles, genes, gene_idx = index_cache.tiles_and_genes(adata, max_genes)
    train, test = data.split(len(tiles), seed)
    p = len(genes)
    C = np.memmap(scratch / "train_covs.f32", dtype=np.float32, mode="w+", shape=(len(train), p, p))
    pos = np.full(len(tiles), -1)
    pos[train] = np.arange(len(train))
    tpos = {int(i): j for j, i in enumerate(test)}
    test_covs = [None] * len(test)
    checksum = 0  # the same sum in the same (tile) order as data.cov_checksum
    for i, c in enumerate(index_cache.iter_tile_covs(adata, tiles, gene_idx)):
        checksum += np.trace(np.asarray(c["cov"], dtype=np.float64))
        if pos[i] >= 0:
            C[pos[i]] = c["cov"]
        else:
            test_covs[tpos[i]] = c
    C.flush()
    return tiles, genes, train, test, np.asarray(C), test_covs, float(checksum)


def build(key, seed, scratch=None):
    rec = {"key": key, "seed": seed, **data.stamp(), "host": socket.gethostname(),
           "slurm_job": os.environ.get("SLURM_JOB_ID", ""), "cpus": os.environ.get("SLURM_CPUS_PER_TASK", ""),
           "lowmem": True}
    out = data.cache_dir(key, seed)
    out.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"spindle_lowmem_{key}_{seed}_", dir=scratch))
    rec["scratch"] = str(scratch)
    mon = index_cache.MemoryMonitor()

    def stage(name, fn):
        t = time.perf_counter()
        res = fn()
        rec[f"{name}_s"] = time.perf_counter() - t
        rec[f"peak_gb_{name}"] = index_cache.peak_gb()
        rec[f"peak_anon_gb_{name}"] = mon.peak_gb()
        print(f"{key} seed {seed}: {name} {rec[f'{name}_s']:.1f} s, peak {rec[f'peak_anon_gb_{name}']:.2f} GB anonymous "
              f"({rec[f'peak_gb_{name}']:.1f} GB with mapped files)", flush=True)
        return res

    try:
        ad = {"adata": stage("load", lambda: sd.load_adata(data.paths.dataset_path(key)))}
        n_cells = int(ad["adata"].n_obs)
        tiles, genes, train, test, C, test_covs, checksum = stage(
            "prepare", lambda: stream_covs(ad["adata"], data.paths.dataset_max_genes(key), seed, scratch))
        ad.clear()  # the section is not needed after the covariances
        layout = stage("layout", lambda: stream_layout([tiles[i] for i in train], C, [tiles[i].id for i in train],
                                                       genes, n_cells, scratch))
        naive = stage("naive", lambda: sd.SpindleNaive(layout, C))
        spindle = stage("spindle", lambda: sd.Spindle.from_naive(naive, layout))
        naive_partial = stage("naive_partial", lambda: sd.SpindleNaivePartial(layout, C))
        spindle_partial = stage("spindle_partial", lambda: sd.SpindlePartial.from_naive_partial(naive_partial, layout))
        rec["cgroup_memory"] = cgroup_memory()
        index_cache.finish_build(rec, out, n_cells=n_cells, n_tiles=len(tiles), cov_checksum=checksum, genes=genes,
                                 train=train, test=test, tr_covs=C, test_covs=test_covs, layout=layout, naive=naive,
                                 spindle=spindle, naive_partial=naive_partial, spindle_partial=spindle_partial,
                                 omega=index_omega(C), mon=mon)
        print(f"{key} seed {seed}: peak {mon.peak_gb():.2f} GB anonymous, "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f} GB with mapped files; "
              f"cgroups {rec['cgroup_memory']}", flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


# ---------------------------------------------------------------- compare
def _diff(x, y, path, out, limit=30):
    """Differences between two unpickled objects, bit for bit for arrays and floats."""
    if len(out) >= limit:
        return
    if type(x) is not type(y):
        out.append(f"{path}: type {type(x).__name__} vs {type(y).__name__}")
    elif isinstance(x, np.ndarray):
        if x.dtype != y.dtype or x.shape != y.shape:
            out.append(f"{path}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}")
        elif x.dtype == object:
            for i, (a, b) in enumerate(zip(x.ravel(), y.ravel())):
                _diff(a, b, f"{path}[{i}]", out, limit)
        elif x.tobytes() != y.tobytes():
            out.append(f"{path}: {int((x != y).sum())} of {x.size} values differ")
    elif isinstance(x, dict):
        if list(x) != list(y):
            out.append(f"{path}: keys differ")
        for k in x:
            if k in y:
                _diff(x[k], y[k], f"{path}[{k!r}]", out, limit)
    elif isinstance(x, (list, tuple)):
        if len(x) != len(y):
            out.append(f"{path}: length {len(x)} vs {len(y)}")
        for i, (a, b) in enumerate(zip(x, y)):
            _diff(a, b, f"{path}[{i}]", out, limit)
    elif isinstance(x, (float, np.floating)):
        if np.asarray(x).tobytes() != np.asarray(y).tobytes():
            out.append(f"{path}: {x!r} vs {y!r}")
    elif hasattr(x, "__dict__") and not isinstance(x, type):
        _diff(vars(x), vars(y), path, out, limit)
    elif x != y:
        out.append(f"{path}: {x!r} vs {y!r}")


IGNORED = ("host", "slurm_job", "cpus", "describe", "commit", "seconds", "scratch", "lowmem", "cgroup_memory")


def _same(a, b):
    """(ok, message) for one file of a cache against the same file of another: identical bytes, or else the same
    unpickled / loaded values (bit for bit); JSON records without their times, memory peaks, hosts and versions."""
    name = a.name
    if name.endswith(".json"):
        ja, jb = json.loads(a.read_text()), json.loads(b.read_text())
        keys = sorted(set(ja) | set(jb))
        bad = [f"{k}: {ja.get(k)!r:.80} vs {jb.get(k)!r:.80}" for k in keys
               if not (k.endswith("_s") or k.startswith("peak_") or k in IGNORED) and ja.get(k) != jb.get(k)]
        return not bad, "\n  ".join(["same records" if not bad else "RECORDS DIFFER", *bad])
    if a.stat().st_size == b.stat().st_size and _bytes_equal(a, b):
        return True, "identical bytes"
    if name.endswith(".pkl"):
        xa, xb = data._load(a), data._load(b)
    elif name.endswith(".npy"):
        xa, xb = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
        xa, xb = (np.asarray(xa), np.asarray(xb)) if xa.dtype == object else (xa, xb)
    else:
        xa, xb = dict(np.load(a)), dict(np.load(b))
    diffs = []
    _diff(xa, xb, name, diffs)
    return not diffs, "\n  ".join(["bytes differ, values identical" if not diffs else "VALUES DIFFER", *diffs])


def _bytes_equal(a, b, chunk=1 << 26):
    with open(a, "rb") as fa, open(b, "rb") as fb:
        while True:
            x, y = fa.read(chunk), fb.read(chunk)
            if x != y:
                return False
            if not x:
                return True


def _report(pairs, label):
    ok = True
    for a, b in pairs:
        same, msg = _same(a, b)
        print(f"{a.relative_to(data.CACHE)}: {msg}", flush=True)
        ok &= same
    print(f"{label}: {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
    return ok


def compare(key, seed, ref):
    """Every build file of data.cache_dir(key, seed) against <ref>/<key>/seed<seed>. Returns True if all match."""
    a, b = data.cache_dir(key, seed), Path(ref) / key / f"seed{seed}"
    return _report([(a / f, b / f) for f in index_cache.BUILD_FILES], f"{key} seed {seed}")


def compare_stage(key, ref):
    """Every staged file of data.CACHE/<key> (and its seeds' niche means) against the same files under <ref>."""
    a, b = data.CACHE / key, Path(ref) / key
    files = list(index_cache.STAGE_FILES) + [f"seed{s}/{f}" for s in data.SEEDS for f in index_cache.STAGE_SEED_FILES]
    return _report([(a / f, b / f) for f in files], f"{key} staged")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=["build", "compare", "compare-stage"])
    ap.add_argument("key")
    ap.add_argument("seed", type=int, nargs="?")
    ap.add_argument("--scratch", default=None, help="build: directory for the scratch files (default: system temp)")
    ap.add_argument("--ref", default=None, help="compare, compare-stage: the reference cache root")
    a = ap.parse_args()
    if a.what == "build":
        build(a.key, a.seed, a.scratch)
    elif a.what == "compare":
        sys.exit(0 if compare(a.key, a.seed, a.ref) else 1)
    else:
        sys.exit(0 if compare_stage(a.key, a.ref) else 1)


if __name__ == "__main__":
    main()
