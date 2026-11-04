"""Compiled search for Spindle-naive, Spindle and Spindle partial: the C kernels (``kernels.c``) through ctypes.

- ``naive_scan``: Spindle-naive's exact d_B scan, bit-identical to ``SpindleNaive.distances_numpy``.
- ``FlatSpindle``: Spindle's flat index arrays and search, the same search as ``spindle.greedy_probe`` +
  ``spindle.radius_search`` breadth-first over the node index, with bit-identical output (rows, order, scores, A_hat,
  radius, work); see the C file for why.
- ``FlatPartial``: Spindle partial's flat arrays and search, bit-identical to
  ``SpindlePartial.search_vectors_dfs`` (the query's pieces are the layers, passed per query).

They are what ``SpindleNaive.search``, ``Spindle.search`` and ``SpindlePartial.search`` run; every benchmark
times these.

The shared library is built with gcc on first use into ``_build/`` next to this file (on the shared file system),
named by the source's hash. Compute nodes have no gcc: after the C file changes, build once on the login node with
``python -c "import spindle_dev.kernels as k; print(k.build())"``. ``-mavx2`` runs on both the login node and the
compute nodes.

Flat index (``FlatSpindle``, derived from ``SpindleNiche`` objects; global ids run over niches in sorted order, then layers,
then nodes): ``cent`` float32 centre vectors back to back, ``node_voff`` each node's start in ``cent``, ``rad``
float64 radii (the stored float32, widened), ``layer_node0`` first node of each layer (+ end), ``layer_dim``,
``layer_qoff`` each layer's start in the query's ``qcat``, ``niche_layer0`` / ``niche_tile0`` first layer / tile of
each niche (+ end), ``codes`` the uint8 node index tile-major per niche (``code0[k] + t * B_k + b``).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_SRC = Path(__file__).with_name("kernels.c")
_FLAGS = ["-O3", "-mavx2", "-ffp-contract=off", "-fno-fast-math", "-fno-builtin-pow", "-fPIC", "-shared", "-std=c11"]


def _library_path() -> Path:
    h = hashlib.sha1(_SRC.read_bytes() + " ".join(_FLAGS).encode()).hexdigest()[:12]
    return _SRC.parent / "_build" / f"kernels-{h}.so"


def build() -> Path:
    """Compile the kernel if this source has no library yet; returns the library's path."""
    lib = _library_path()
    if lib.exists():
        return lib
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError(f"{lib.name} is not built and there is no C compiler here (compute node?); run "
                           f"`python -c \"import spindle_dev.kernels as k; k.build()\"` on the login node once")
    lib.parent.mkdir(exist_ok=True)
    tmp = lib.with_suffix(f".{os.getpid()}.tmp")
    subprocess.run([cc, *_FLAGS, "-o", str(tmp), str(_SRC), "-lm"], check=True)
    os.replace(tmp, lib)
    return lib


_p = ctypes.c_void_p
_i64 = ctypes.c_int64
_LIB = None


def _lib():
    """The kernel library, built (login node) or loaded on first use."""
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(str(build()))
        lib.spindle_search.restype = _i64
        lib.spindle_search.argtypes = [_p] * 10 + [_i64, _i64, _p, ctypes.c_double, _p, _p, _p, _p]
        lib.naive_scan.restype = None
        lib.naive_scan.argtypes = [_p, _p, _i64, _i64, _p, _p]
        lib.partial_search.restype = _i64
        lib.partial_search.argtypes = [_p] * 9 + [_i64] + [_p] * 4 + [ctypes.c_double] + [_p] * 4
        lib.spindle_row_dists.restype = None
        lib.spindle_row_dists.argtypes = [_p, _p, _i64, _i64, _p]
        lib.spindle_probe_niche.restype = _i64  # method-v3: Spindle's search niche by niche
        lib.spindle_probe_niche.argtypes = [_p] * 10 + [_i64, _i64, _p, _i64, _p, _p]
        lib.spindle_scan_niche.restype = _i64
        lib.spindle_scan_niche.argtypes = [_p] * 10 + [_i64, _i64, _p, _i64, ctypes.c_double, _p, _p, _p, _p]
        _LIB = lib
    return _LIB


def _ptr(a: np.ndarray):
    return a.ctypes.data


def naive_scan(X: np.ndarray, q: np.ndarray, rows: np.ndarray, out: np.ndarray) -> None:
    """out[rows] = d_B of each row of float32 X to float32 q, as numpy's
    sqrt(((X - q) ** 2).sum(1, dtype=float64)) bit for bit."""
    X = np.ascontiguousarray(X, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float32)
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    if out.dtype != np.float64 or not out.flags.c_contiguous or len(rows) != len(X) or q.shape != (X.shape[1],):
        raise ValueError("naive_scan: shape or dtype mismatch")
    if len(rows) and (rows.min() < 0 or rows.max() >= len(out)):
        raise IndexError("naive_scan: row outside out")
    _lib().naive_scan(_ptr(X), _ptr(q), len(X), X.shape[1], _ptr(rows), _ptr(out))


def row_dists(C: np.ndarray, q: np.ndarray) -> np.ndarray:
    """The kernel's d of each row of float32 C to float64 q (equal to sqrt(((C - q) ** 2).sum(1)) bit for bit)."""
    C = np.ascontiguousarray(C, dtype=np.float32)
    q = np.ascontiguousarray(q, dtype=np.float64)
    out = np.empty(len(C))
    _lib().spindle_row_dists(_ptr(C), _ptr(q), len(C), C.shape[1], _ptr(out))
    return out


class FlatSpindle:
    """The flat arrays of a Spindle index (derived from its ``SpindleNiche`` objects, nothing new stored)."""

    def __init__(self, niches):
        if any(len(C) > 256 for nc in niches.values() for C in nc.centres):
            raise ValueError("the compiled search needs at most 256 nodes per layer (one-byte node index)")
        cent, rad, voff, node0, dims, qoff, layer0, tile0, codes, code0, rows = [], [], [], [0], [], [], [0], [0], [], [], []
        self._gather = {}
        v = q = c = 0
        for k, nc in niches.items():
            gather = []
            for C, R, (s, e) in zip(nc.centres, nc.radius, nc.layer_slices):
                K, dim = C.shape
                cent.append(np.ascontiguousarray(C, dtype=np.float32).ravel())
                rad.append(np.asarray(R).astype(np.float64))
                voff.append(v + dim * np.arange(K, dtype=np.int64))
                v += K * dim
                node0.append(node0[-1] + K)
                dims.append(dim)
                qoff.append(q)
                q += dim
                gather.append(np.arange(s, e))
            self._gather[k] = np.concatenate(gather)
            layer0.append(layer0[-1] + nc.n_layers)
            tile0.append(tile0[-1] + nc.n)
            codes.append(np.stack(nc.node_index, axis=1).astype(np.uint8).ravel())
            code0.append(c)
            c += nc.n * nc.n_layers
            rows.append(nc.rows)
        i64 = lambda x: np.ascontiguousarray(x, dtype=np.int64)  # noqa: E731
        self.cent = np.concatenate(cent)
        self.rad = np.concatenate(rad)
        self.node_voff = i64(np.concatenate(voff))
        self.layer_node0, self.layer_dim, self.layer_qoff = i64(node0), i64(dims), i64(qoff)
        self.niche_layer0, self.niche_tile0, self.code0 = i64(layer0), i64(tile0), i64(code0)
        self.codes = np.ascontiguousarray(np.concatenate(codes))
        self.rows = i64(np.concatenate(rows))
        self.keys = list(niches.keys())
        self._args = [_ptr(a) for a in (self.cent, self.node_voff, self.rad, self.layer_node0, self.layer_dim,
                                        self.layer_qoff, self.niche_layer0, self.niche_tile0, self.codes, self.code0)]
        self._args += [len(self.keys), len(self.rad)]

    def qcat(self, qv) -> np.ndarray:
        """The query's float64 layer slices, back to back (niches in sorted order, layers in spread order)."""
        return np.concatenate([np.asarray(qv[k])[self._gather[k]] for k in self.keys]).astype(np.float64)

    def search(self, qv, g: float):
        """(training rows by increasing A, A, A_hat, radius, work) for one query's per-niche vectors."""
        qcat = self.qcat(qv)
        N = len(self.rows)
        a2, k, t, st = np.empty(N), np.empty(N, dtype=np.int64), np.empty(N, dtype=np.int64), np.empty(3)
        n = _lib().spindle_search(*self._args, _ptr(qcat), float(g), _ptr(a2), _ptr(k), _ptr(t), _ptr(st))
        if n < 0:
            raise MemoryError("spindle_search: out of memory")
        a2, k, t = a2[:n], k[:n], t[:n]
        order = np.lexsort((t, k, a2))            # = sorting the (A^2, niche, tile) tuples
        rows = self.rows[self.niche_tile0[k[order]] + t[order]]
        return rows, np.sqrt(a2[order]), float(st[0]), float(st[1]), int(st[2])

    def search_many(self, qvs, g: float, threads: int | None = None):
        """``search`` for a list of per-niche query vectors, in parallel threads (the C call releases the GIL)."""
        with ThreadPoolExecutor(max_workers=threads or os.cpu_count()) as ex:
            return list(ex.map(lambda qv: self.search(qv, g), qvs))

    # ---------------------------------------------------------- method-v3: one query's search split by niche
    # ``search`` in two phases, each niche on its own (kernels.c spindle_probe_niche / spindle_scan_niche): probe every
    # niche, A_hat = sqrt(the smallest leaf A^2), r = g * A_hat, scan every niche with r^2, then ``assemble``. Same
    # output as ``search``, bit for bit. A niche's probe and scan must share one ``dcache`` (float64, n_nodes long; the
    # niches use disjoint parts of it).
    @property
    def n_niche(self) -> int:
        return len(self.keys)

    def probe_niche(self, qcat, k: int, dcache) -> tuple[float, int]:
        """(the A^2 of niche k's greedy leaf, node distances computed)."""
        out = np.empty(1)
        w = _lib().spindle_probe_niche(*self._args, _ptr(qcat), int(k), _ptr(dcache), _ptr(out))
        if w < 0:
            raise MemoryError("spindle_probe_niche: out of memory")
        return float(out[0]), int(w)

    def scan_niche(self, qcat, k: int, r2: float, dcache, out_a2=None, out_t=None):
        """(hits' A^2, hits' tiles in niche k, node distances computed); ``out_a2`` / ``out_t`` may be buffers of at
        least the niche's tile count (views of them are returned)."""
        n = int(self.niche_tile0[k + 1] - self.niche_tile0[k])
        a2 = np.empty(n) if out_a2 is None else out_a2
        t = np.empty(n, dtype=np.int64) if out_t is None else out_t
        nh = np.zeros(1, dtype=np.int64)
        w = _lib().spindle_scan_niche(*self._args, _ptr(qcat), int(k), float(r2), _ptr(dcache), _ptr(a2), _ptr(t),
                                      _ptr(nh))
        if w < 0:
            raise MemoryError("spindle_scan_niche: out of memory")
        return a2[:nh[0]], t[:nh[0]], int(w)

    @staticmethod
    def radius(best_a2: float, g: float) -> tuple[float, float, float]:
        """(A_hat, r, r^2) from the smallest leaf A^2 over niches, as spindle_search computes them."""
        a_hat = float(np.sqrt(best_a2))
        r = float(g) * a_hat
        return a_hat, r, r * r

    def assemble(self, hits, a_hat: float, r: float, work: int):
        """``search``'s output from the per-niche hits [(A^2, tiles) for niche 0, 1, ...]."""
        a2 = np.concatenate([h[0] for h in hits])
        t = np.concatenate([h[1] for h in hits]).astype(np.int64)
        k = np.repeat(np.arange(self.n_niche, dtype=np.int64), [len(h[1]) for h in hits])
        order = np.lexsort((t, k, a2))            # = sorting the (A^2, niche, tile) tuples
        rows = self.rows[self.niche_tile0[k[order]] + t[order]]
        return rows, np.sqrt(a2[order]), a_hat, r, int(work)

    def search_parallel(self, qv, g: float, threads: int | None = None, executor=None):
        """``search`` with the niches on ``threads`` threads (or an existing ``executor``); same output."""
        qcat = self.qcat(qv)
        dcache = np.empty(len(self.rad))
        own = executor is None
        ex = ThreadPoolExecutor(max_workers=threads or os.cpu_count()) if own else executor
        try:
            probes = list(ex.map(lambda k: self.probe_niche(qcat, k, dcache), range(self.n_niche)))
            best = np.inf
            for a2, _ in probes:                   # as spindle_search: the first strictly smaller value
                if a2 < best:
                    best = a2
            a_hat, r, r2 = self.radius(best, g)
            hits = list(ex.map(lambda k: self.scan_niche(qcat, k, r2, dcache), range(self.n_niche)))
        finally:
            if own:
                ex.shutdown()
        work = sum(w for _, w in probes) + sum(h[2] for h in hits)
        return self.assemble(hits, a_hat, r, work)



class FlatPartial:
    """The flat arrays of a Spindle partial index (derived from its ``PartialNiche`` objects)."""

    def __init__(self, niches):
        if any(len(C) > 256 for nc in niches.values() for C in nc.centres):
            raise ValueError("the compiled search needs at most 256 nodes per interval (one-byte node index)")
        cent, rad, voff, node0, dims, iv0, tile0, codes, code0, rows = [], [], [], [0], [], [0], [0], [], [], []
        self.iv_base = {}
        v = c = 0
        for k, nc in niches.items():
            self.iv_base[k] = iv0[-1]
            for C, R in zip(nc.centres, nc.radius):
                K, dim = C.shape
                cent.append(np.ascontiguousarray(C, dtype=np.float32).ravel())
                rad.append(np.asarray(R).astype(np.float64))
                voff.append(v + dim * np.arange(K, dtype=np.int64))
                v += K * dim
                node0.append(node0[-1] + K)
                dims.append(dim)
            iv0.append(iv0[-1] + len(nc.centres))
            tile0.append(tile0[-1] + nc.n)
            codes.append(np.stack(nc.node_index, axis=1).astype(np.uint8).ravel())
            code0.append(c)
            c += nc.n * len(nc.centres)
            rows.append(nc.rows)
        i64 = lambda x: np.ascontiguousarray(x, dtype=np.int64)  # noqa: E731
        self.cent = np.concatenate(cent)
        self.rad = np.concatenate(rad)
        self.node_voff = i64(np.concatenate(voff))
        self.iv_node0, self.iv_dim = i64(node0), i64(dims)
        self.niche_iv0, self.niche_tile0, self.code0 = i64(iv0), i64(tile0), i64(code0)
        self.codes = np.ascontiguousarray(np.concatenate(codes))
        self.rows = i64(np.concatenate(rows))
        self.keys = list(niches.keys())
        self._args = [_ptr(a) for a in (self.cent, self.node_voff, self.rad, self.iv_node0, self.iv_dim,
                                        self.niche_iv0, self.niche_tile0, self.codes, self.code0)]
        self._args.append(len(self.keys))

    def search(self, qp, g: float):
        """(training rows by increasing A, A, best A^2 of the probe (unnormalised), work) for one query's pieces
        (``spindle_partial.QueryPieces``)."""
        qlayer, q0, qoff, qcat, o = [], [0], [], [], 0
        for k in self.keys:
            for iv, vec in qp.layers[k]:
                qlayer.append(self.iv_base[k] + iv)
                qoff.append(o)
                o += len(vec)
                qcat.append(vec)
            q0.append(len(qlayer))
        qlayer, q0, qoff = (np.ascontiguousarray(x, dtype=np.int64) for x in (qlayer, q0, qoff))
        qcat = np.ascontiguousarray(np.concatenate(qcat), dtype=np.float64)
        N = len(self.rows)
        a2, k, t, st = np.empty(N), np.empty(N, dtype=np.int64), np.empty(N, dtype=np.int64), np.empty(2)
        n = _lib().partial_search(*self._args, _ptr(qlayer), _ptr(q0), _ptr(qoff), _ptr(qcat), float(g),
                                  _ptr(a2), _ptr(k), _ptr(t), _ptr(st))
        if n < 0:
            raise MemoryError("partial_search: out of memory")
        a2, k, t = a2[:n], k[:n], t[:n]
        order = np.lexsort((t, k, a2))            # = sorting the (A^2, niche, tile) tuples
        rows = self.rows[self.niche_tile0[k[order]] + t[order]]
        return rows, np.sqrt(a2[order] / qp.n_genes), float(st[0]), int(st[1])

    def search_many(self, qps, g: float, threads: int | None = None):
        """``search`` for a list of query pieces, in parallel threads (the C call releases the GIL)."""
        with ThreadPoolExecutor(max_workers=threads or os.cpu_count()) as ex:
            return list(ex.map(lambda qp: self.search(qp, g), qps))


if __name__ == "__main__":
    print(build())
