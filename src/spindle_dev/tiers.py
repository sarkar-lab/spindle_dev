"""Swappable search tiers over a niche/block layout.

Four tiers share one interface, so every experiment can take ``--tier``:

* ``exact``          Spindle-Exact, whole-tile queries. Stores the float32 upper triangles of every
                     training tile's block logs and scans them exactly.
* ``dag``            Spindle-DAG, whole-tile queries. K node means per (niche, block), one code per
                     (tile, block), no logs; approximate d_B^2 from per-block distance tables. Its
                     result is an unordered set of c tiles (returned in score order for convenience).
* ``exact_partial``  Spindle-Exact for a gene set S: stores block logs and block covariances and
                     eigendecomposes sub-blocks S cap block at query time; returns d_B^S exactly.
* ``interval``       The interval index for a gene set S (interval_dp.py): dyadic intervals with
                     eps-bounded node means, coarsest decomposition, rss score.
* ``whole``          The whole-matrix baseline (not Spindle): float32 upper triangles of every training tile's
                     whole p x p log and an exact scan of the whole-matrix log-Euclidean distance
                     d_W(q, t) = ||log q - log t||_F / sqrt(p). Same shrinkage, eigenvalue floor and storage
                     convention as ``exact``, so exact-vs-whole speedups are like for like.

Distances (pooled over niches; each tile is scored in its own niche's gene order and blocks):

    d_B(q, t)   = sqrt(sum_b ||log q_b - log t_b||_F^2) / sqrt(p)
    d_B^S(q, t) = sqrt(sum_b ||log q[S cap b] - log t[S cap b]||_F^2) / sqrt(|S|)

Every covariance (training tile or query) goes through ``prepare_cov`` first, which applies the
covariance shrinkage of ``metrics.shrink_cov``; its strength is ``metrics.SHRINK_ALPHA`` (0 turns it off).
The helpers below the tiers (BlockVectorizer, ExactScorer, compact_dag, ADC, ...) take covariances as
given and never shrink.

Common API::

    tier = build_tier(kind, data, train_covs, **kw)
    rows, scores, seconds = tier.search(query_cov, genes=None, c=None)
    tier.nbytes()

``rows`` are training-tile rows (positions in ``data.labels`` / ``train_covs``), best first; ``c=None``
returns every tile. ``genes=None`` is a whole-tile query (exact, dag); a gene set (global gene indices)
is a partial query (exact_partial, interval), and ``query_cov`` is then either the full p x p matrix or
the |S| x |S| matrix in ``genes`` order. ``tier.last_timing`` splits ``seconds`` into the query's own
logs (``query_s``) and the scan (``scan_s``).
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np

from . import interval_dp as ivd
from . import metrics

LOG_FLOOR = 1e-6  # eigenvalue floor of every block log (the ground-truth convention)
TIER_KINDS = ("exact", "dag", "exact_partial", "interval", "whole")
DAG_DEFAULTS = dict(k_target=32, alpha=0.0, k_min=32)
INTERVAL_DEFAULTS = dict(k_target=32, k_alpha=0.5, max_len=16, lloyd_iters=0)


def raw_cov(entry) -> np.ndarray:
    """The covariance of a stored entry (an array, or a dict with 'cov' / 'matrix')."""
    return entry if not isinstance(entry, dict) else entry.get("cov", entry.get("matrix", entry))


def prepare_cov(entry) -> np.ndarray:
    """float64 covariance with the configured shrinkage (metrics.SHRINK_ALPHA) applied."""
    return metrics.shrink_cov(np.asarray(raw_cov(entry), dtype=np.float64))


# ---------------------------------------------------------------------------------------------- layout
class BlockVectorizer:
    """One niche's layout: block logs -> vector whose squared L2 is d_B^2."""

    def __init__(self, perm, block_runs):
        self.perm = np.asarray(perm)
        self.runs = [(int(s), int(e)) for s, e in block_runs]
        self.p = sum(e - s for s, e in self.runs)
        self.iu = [np.triu_indices(e - s) for s, e in self.runs]
        self.w = [np.where(i == j, 1.0, np.sqrt(2.0)) / np.sqrt(self.p) for i, j in self.iu]
        self.dims = [len(w) for w in self.w]
        self.dim = sum(self.dims)
        self.offsets = np.cumsum([0] + self.dims)

    def block_logs(self, cov):
        c = np.asarray(cov, dtype=np.float64)[np.ix_(self.perm, self.perm)]
        return [metrics.log_spd(c[s:e, s:e], eps=LOG_FLOOR) for s, e in self.runs]

    def pieces(self, logs, dtype=np.float32):
        return [(L[iu] * w).astype(dtype) for L, iu, w in zip(logs, self.iu, self.w)]

    def vector(self, cov, dtype=np.float64):
        return np.concatenate(self.pieces(self.block_logs(cov), dtype))


def niche_layouts(data):
    """{niche: (local tile indices, BlockVectorizer)} in sorted niche order."""
    labels = np.asarray(data.labels).astype(int)
    return {k: (np.flatnonzero(labels == k), BlockVectorizer(data.perm_list[k], data.block_dict[k]))
            for k in sorted(set(labels.tolist()))}


class LayoutShell:
    """Just the niche layout (labels, perm_list, block_dict) that niche_layouts and query_pieces need."""

    def __init__(self, labels, perm_list, block_dict):
        self.labels, self.perm_list, self.block_dict = np.asarray(labels), perm_list, block_dict


def block_log_bytes(data):
    """Exact-tier storage: float32 upper triangles of every training tile's block logs."""
    return int(sum(len(idx) * vec.dim * 4 for idx, vec in niche_layouts(data).values()))


def frob_triu(L, iu):
    """Upper triangle of a symmetric matrix with off-diagonals x sqrt 2 (its L2 norm = Frobenius norm)."""
    return L[iu] * np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0))


class ExactScorer:
    """Exact L2 block distances from a query covariance to every training tile (float64, no shrinkage)."""

    def __init__(self, data, train_covs):
        self.layouts = niche_layouts(data)
        self.n = len(data.labels)
        self.X = {k: np.stack([vec.vector(raw_cov(train_covs[t])) for t in idx])
                  for k, (idx, vec) in self.layouts.items()}

    @classmethod
    def from_vectors(cls, data, X):
        """Build from precomputed per-niche float64 matrices of block vectors (rows = niche tiles)."""
        obj = cls.__new__(cls)
        obj.layouts = niche_layouts(data)
        obj.n = len(data.labels)
        obj.X = X
        return obj

    def query_vectors(self, cov):
        return {k: vec.vector(cov) for k, (_, vec) in self.layouts.items()}

    def distances(self, qv):
        d = np.empty(self.n)
        for k, (idx, _) in self.layouts.items():
            d[idx] = np.sqrt(np.maximum(((self.X[k] - qv[k]) ** 2).sum(1), 0.0))
        return d


# ---------------------------------------------------------------------------------------------- compact DAG
def _rowdist(V, v, p_b):
    """index._fro_dist_norm of every row of V to v (rows are frob_triu vectors)."""
    return np.sqrt(np.maximum(((V - v) ** 2).sum(1), 0.0)) / np.sqrt(p_b)


def _effective_k_target(n, base, alpha, k_min):
    """index._effective_k_target, without importing the index module."""
    from . import index
    return index._effective_k_target(n, base=base, alpha=alpha, k_min=k_min)


def build_dag_from_block_logs(block_vecs, data, k_target, alpha, k_min):
    """The compact DAG built directly from block logs, never touching full covariances.

    Re-implements, for lists of block logs, exactly what dag_eval_common.build_dag (choose_adaptive_epsilons
    with farthest_first_eps_for_k, seed 42, then the epsilon_net branch of index.index_spds) does, and
    returns what compact_dag(build_dag(...)) returns:
      per block, epsilon = the farthest-first cover radius at k = index._effective_k_target;
      epsilon-net centres from tile 0, each tile assigned to its first-nearest centre;
      node mean = members' mean log, radius = max member distance to the centre;
      edges join nodes of consecutive blocks that share tiles.
    ``block_vecs[niche][b]`` is (n_niche, d_b) float64: frob_triu of each tile's block-b log, rows in
    niche order (data.labels == niche).
    """
    out = {}
    for niche, (idx, vec) in niche_layouts(data).items():
        B = len(vec.runs)
        means, radius, codes = [], [], np.empty((len(idx), B), dtype=np.int64)
        for b, (s, e) in enumerate(vec.runs):
            V, p_b, n = block_vecs[niche][b], e - s, len(idx)
            k_eff = _effective_k_target(n, k_target, alpha, k_min)
            # farthest_first_eps_for_k (seed 42)
            first = int(np.random.default_rng(42).integers(0, n))
            delta = _rowdist(V, V[first], p_b)
            delta[first] = 0.0
            deltas_max, n_centres = [float(delta.max())], 1
            while n_centres < min(k_eff, n):
                i_star = int(np.argmax(delta))
                if delta[i_star] <= 0.0:
                    break
                n_centres += 1
                delta = np.minimum(delta, _rowdist(V, V[i_star], p_b))
                deltas_max.append(float(delta.max()))
            eps = deltas_max[min(len(deltas_max), k_eff) - 1]
            # epsilon_net branch of index_spds: centres from tile 0 until everything is within eps
            centres = [0]
            delta = _rowdist(V, V[0], p_b)
            while True:
                i_star = int(np.argmax(delta))
                if delta[i_star] <= eps:
                    break
                centres.append(i_star)
                delta = np.minimum(delta, _rowdist(V, V[i_star], p_b))
            D = np.stack([_rowdist(V, V[c], p_b) for c in centres], axis=1)
            assign = np.argmin(D, axis=1)  # first nearest centre, as the strict '<' scan
            best = D[np.arange(n), assign]
            codes[:, b] = assign
            M = np.stack([V[assign == k].mean(0) for k in range(len(centres))])
            # BlockVectorizer piece = L[iu] * w_frob / sqrt(p) = frob_triu / sqrt(p)
            means.append((M / np.sqrt(vec.p)).astype(np.float32))
            radius.append(np.array([best[assign == k].max() for k in range(len(centres))], np.float32)
                          * np.sqrt(p_b / vec.p))
        width = max(len(m) for m in means)
        indptr, indices = [0], []
        for b in range(B - 1):
            for k in range(len(means[b])):
                indices.extend(sorted(set(codes[codes[:, b] == k, b + 1].tolist())))
                indptr.append(len(indices))
        out[niche] = {"centroids": means, "radius": radius,
                      "codes": codes.astype(np.uint8 if width <= 256 else np.uint16),
                      "edges_indptr": np.asarray(indptr, np.int32), "edges_indices": np.asarray(indices, np.int32),
                      "local_idx": idx.astype(np.int32)}
    return out


def compact_dag(dag_dict, data):
    """The deployable form of an index.index_spds DAG per niche: node means, radii, codes, edges."""
    layouts = niche_layouts(data)
    local_of = {int(s): i for i, s in enumerate(data.spd_ids)}
    out = {}
    for key, handle in dag_dict.items():
        niche = int(key)
        idx, vec = layouts[niche]
        if [tuple(map(int, r)) for r in handle.block_runs] != vec.runs:
            raise ValueError(f"niche {niche}: DAG block runs differ from data.block_dict")
        pos = {int(t): i for i, t in enumerate(idx)}
        B = len(vec.runs)
        means, radius, node_pos = [], [], {}
        for b in range(B):
            nodes = [handle.nodes[g] for g in handle.block_to_node_indices[b]]
            means.append(np.stack([(n.metadata.representative_mean[vec.iu[b]] * vec.w[b]).astype(np.float32)
                                   for n in nodes]))
            s, e = vec.runs[b]
            # index radius (to the epsilon-net centre) is ||.||_F / sqrt(p_b); rescale to d_B units (/ sqrt(p))
            radius.append(np.array([n.metadata.radius for n in nodes], np.float32) * np.sqrt((e - s) / vec.p))
            for k, n in enumerate(nodes):
                node_pos[n.global_node_id] = (b, k)
        codes = np.full((len(idx), B), -1, dtype=np.int64)
        for gid, (b, k) in node_pos.items():
            for spd_id, _ in handle.nodes[gid].metadata.members:
                i = pos[local_of[int(spd_id)]]
                if codes[i, b] != -1:
                    raise ValueError(f"niche {niche}: tile {i} is in two nodes of block {b}")
                codes[i, b] = k
        if (codes < 0).any():
            raise ValueError(f"niche {niche}: {int((codes < 0).sum())} (tile, block) pairs have no node")
        codes = codes.astype(np.uint8 if max(len(c) for c in means) <= 256 else np.uint16)
        indptr, indices = [0], []
        for b in range(B - 1):
            for gid in handle.block_to_node_indices[b]:
                indices.extend(sorted(node_pos[c][1] for c in handle.nodes[gid].children
                                      if node_pos.get(c, (None,))[0] == b + 1))
                indptr.append(len(indices))
        out[niche] = {"centroids": means, "radius": radius, "codes": codes,
                      "edges_indptr": np.asarray(indptr, np.int32), "edges_indices": np.asarray(indices, np.int32),
                      "local_idx": idx.astype(np.int32)}
    return out


def nbytes(compact):
    """The deployed DAG's bytes (node means, radii, codes, edges)."""
    return int(sum(sum(x.nbytes for x in c["centroids"]) + sum(x.nbytes for x in c["radius"]) + c["codes"].nbytes
                   + c["edges_indptr"].nbytes + c["edges_indices"].nbytes for c in compact.values()))


def n_nodes(compact):
    return int(sum(sum(len(x) for x in c["centroids"]) for c in compact.values()))


class ADC:
    """Approximate d_B^2 from a query to every training tile using only the compact DAG."""

    def __init__(self, compact):
        self.niches = []
        for niche, c in sorted(compact.items()):
            C = [np.ascontiguousarray(x) for x in c["centroids"]]
            width = max(len(x) for x in C)
            flat_codes = c["codes"].astype(np.int64) + (np.arange(len(C)) * width)[None, :]
            self.niches.append((niche, C, [np.einsum("ij,ij->i", x, x) for x in C], width, flat_codes,
                                c["local_idx"].astype(np.int64)))

    def scores(self, query_pieces):
        """query_pieces: {niche: [float32 block vector per block]} -> (local ids, approx d_B^2)."""
        ids, out = [], []
        for niche, C, sq, width, flat_codes, local_idx in self.niches:
            x = query_pieces[niche]
            T = np.full(len(C) * width, np.inf, dtype=np.float32)
            for b, (Cb, sqb, xb) in enumerate(zip(C, sq, x)):
                T[b * width: b * width + len(Cb)] = sqb - 2.0 * (Cb @ xb) + xb @ xb
            out.append(T[flat_codes].sum(1))
            ids.append(local_idx)
        return np.concatenate(ids), np.concatenate(out)


def query_pieces(data, cov):
    """Per-niche float32 block vectors of a query (input to ADC.scores)."""
    return {k: vec.pieces(vec.block_logs(cov)) for k, (_, vec) in niche_layouts(data).items()}


# ---------------------------------------------------------------------------------------------- tiers
def _top(scores, c):
    order = np.argsort(scores, kind="stable")
    return order if c is None else order[:c]


class _Tier:
    kind = ""
    partial = False

    def _check(self, genes):
        if self.partial and genes is None:
            raise ValueError(f"tier '{self.kind}' answers gene-set queries; pass genes (or use exact / dag)")
        if not self.partial and genes is not None:
            raise ValueError(f"tier '{self.kind}' answers whole-tile queries; use exact_partial or interval for genes")

    def search(self, query_cov, genes=None, c: Optional[int] = None):
        """-> (training rows best first, scores, seconds). Subclasses implement _query and _scan."""
        self._check(genes)
        t0 = time.perf_counter()
        q = self._query(query_cov, genes)
        t1 = time.perf_counter()
        rows, scores = self._scan(q, genes, c)
        t2 = time.perf_counter()
        self.last_timing = {"query_s": t1 - t0, "scan_s": t2 - t1}
        return rows, scores, t2 - t0


class ExactTier(_Tier):
    """Spindle-Exact: float32 block-log vectors per niche, exact d_B scan."""

    kind = "exact"

    def __init__(self, data, train_covs):
        self.layouts = niche_layouts(data)
        self.n = len(data.labels)
        self.X = {k: np.stack([vec.vector(prepare_cov(train_covs[t]), np.float32) for t in idx])
                  for k, (idx, vec) in self.layouts.items()}

    @classmethod
    def from_block_vectors(cls, data, X):
        """Build from precomputed per-niche float32 matrices (rows = the niche's tiles in niche_layouts
        order) of BlockVectorizer.vector(prepare_cov(cov)), for builds that stream the covariances."""
        obj = cls.__new__(cls)
        obj.layouts = niche_layouts(data)
        obj.n = len(data.labels)
        for k, (idx, vec) in obj.layouts.items():
            if X[k].shape != (len(idx), vec.dim) or X[k].dtype != np.float32:
                raise ValueError(f"niche {k}: expected float32 {(len(idx), vec.dim)}, got {X[k].dtype} {X[k].shape}")
        obj.X = X
        return obj

    @classmethod
    def from_exact_partial(cls, ep, data):
        """Build from an ExactPartialTier's float32 block logs (same layout, rows and triangle order), so the
        block logs are not computed twice. Equal to ExactTier(data, train_covs) up to float32 rounding."""
        return cls.from_block_vectors(data, {k: np.ascontiguousarray(np.concatenate(ep.logs[k], axis=1)
                                                                    / np.float32(np.sqrt(ep.p)))
                                             for k in ep.niches})

    def nbytes(self):
        return int(sum(x.nbytes for x in self.X.values()))

    def query_vectors(self, cov):
        """Per-niche float32 query vectors of an already prepared covariance."""
        return {k: vec.vector(cov, np.float32) for k, (_, vec) in self.layouts.items()}

    def distances(self, qv):
        d = np.empty(self.n)
        for k, (idx, _) in self.layouts.items():
            d[idx] = np.sqrt(np.maximum(((self.X[k] - qv[k]) ** 2).sum(1, dtype=np.float64), 0.0))
        return d

    def _query(self, query_cov, genes):
        return self.query_vectors(prepare_cov(query_cov))

    def _scan(self, qv, genes, c):
        d = self.distances(qv)
        rows = _top(d, c)
        return rows, d[rows]


class DagTier(_Tier):
    """Spindle-DAG: compact DAG (node means + codes) built from the exact tier's block logs."""

    kind = "dag"

    def __init__(self, data, train_covs=None, exact: Optional[ExactTier] = None, k_target=DAG_DEFAULTS["k_target"],
                 alpha=DAG_DEFAULTS["alpha"], k_min=DAG_DEFAULTS["k_min"]):
        exact = exact if exact is not None else ExactTier(data, train_covs)
        self.layouts = exact.layouts
        self.n = exact.n
        t0 = time.perf_counter()
        block_vecs = {k: [exact.X[k][:, vec.offsets[b]:vec.offsets[b + 1]].astype(np.float64) * np.sqrt(vec.p)
                          for b in range(len(vec.runs))] for k, (_, vec) in self.layouts.items()}
        self.compact = build_dag_from_block_logs(block_vecs, data, k_target, alpha, k_min)
        self.build_seconds = time.perf_counter() - t0
        self.params = dict(k_target=k_target, alpha=alpha, k_min=k_min)
        self.adc = ADC(self.compact)

    def nbytes(self):
        return nbytes(self.compact)

    def _query(self, query_cov, genes):
        cov = prepare_cov(query_cov)
        return {k: vec.pieces(vec.block_logs(cov)) for k, (_, vec) in self.layouts.items()}

    def _scan(self, pieces, genes, c):
        ids, approx = self.adc.scores(pieces)
        sel = _top(approx, c)
        return ids[sel], np.sqrt(np.maximum(approx[sel], 0.0))


class ExactPartialTier(_Tier):
    """Spindle-Exact for gene sets: per niche and block, float64 covariance stacks and float32 log vectors."""

    kind = "exact_partial"
    partial = True

    def __init__(self, data, train_covs, keep_niche_means=False):
        labels = self.labels = np.asarray(data.labels).astype(int)
        self.p = len(data.perm_list[labels[0]])
        self.niches = sorted(set(labels.tolist()))
        self.rows = {k: np.flatnonzero(labels == k) for k in self.niches}
        self.blocks = {k: [np.asarray(data.perm_list[k])[s:e] for s, e in data.block_dict[k]] for k in self.niches}
        self.covs, self.logs = {}, {}
        self._niche_means = {} if keep_niche_means else None
        for k in self.niches:
            stacks = [np.empty((len(self.rows[k]), len(g), len(g))) for g in self.blocks[k]]
            total = np.zeros((self.p, self.p)) if keep_niche_means else None
            for i, r in enumerate(self.rows[k]):
                cov = prepare_cov(train_covs[r])
                if total is not None:
                    total += cov
                for stack, g in zip(stacks, self.blocks[k]):
                    stack[i] = cov[np.ix_(g, g)]
            if total is not None:
                self._niche_means[k] = total / len(self.rows[k])
            self.covs[k] = stacks
            self.logs[k] = [ivd.sym_to_vec(ivd.batched_log_spd(s, LOG_FLOOR)).astype(np.float32) for s in stacks]
        self.n = len(labels)

    def nbytes_parts(self):
        """float32 upper triangles of the block logs and of the block covariances."""
        tri = sum(len(self.rows[k]) * sum(len(g) * (len(g) + 1) // 2 for g in self.blocks[k]) for k in self.niches)
        return {"block_logs": 4 * tri, "block_covs": 4 * tri}

    def nbytes(self):
        return int(sum(self.nbytes_parts().values()))

    def block_covs(self):
        return self.covs

    def niche_means(self):
        """{niche: mean prepared (shrunk) p x p covariance of its training tiles} (needs keep_niche_means=True)."""
        if self._niche_means is None:
            raise ValueError("build the tier with keep_niche_means=True")
        return self._niche_means

    def niche_means_nbytes(self):
        """float32 upper triangles of the niche means."""
        return 4 * len(self.niches) * self.p * (self.p + 1) // 2

    def distances(self, cov_s, genes):
        """d_B^S to every training tile; ``cov_s`` is the prepared |S| x |S| query in ``genes`` order."""
        genes = np.asarray(genes)
        col = np.full(self.p, -1, dtype=np.int64)
        col[genes] = np.arange(len(genes))
        d = np.empty(self.n)
        for k in self.niches:
            tot = np.zeros(len(self.rows[k]))
            for g, cov, logv in zip(self.blocks[k], self.covs[k], self.logs[k]):
                pos = np.flatnonzero(col[g] >= 0)
                if len(pos) == 0:
                    continue
                cq = col[g[pos]]
                lq = ivd.sym_to_vec(ivd.batched_log_spd(cov_s[np.ix_(cq, cq)][None], LOG_FLOOR))[0]
                if len(pos) == len(g):
                    lt = logv
                else:
                    lt = ivd.sym_to_vec(ivd.batched_log_spd(cov[:, pos[:, None], pos[None, :]], LOG_FLOOR))
                tot += ((lt - lq) ** 2).sum(1)
            d[self.rows[k]] = np.sqrt(tot / len(genes))
        return d

    def _query(self, query_cov, genes):
        return query_cov_s(query_cov, genes, self.p)

    def _scan(self, cov_s, genes, c):
        d = self.distances(cov_s, genes)
        rows = _top(d, c)
        return rows, d[rows]


class IntervalTier(_Tier):
    """The interval index (interval_dp.build_interval_index), built from the exact partial tier's block covariances."""

    kind = "interval"
    partial = True

    def __init__(self, data, train_covs=None, exact_partial: Optional[ExactPartialTier] = None,
                 k_target=INTERVAL_DEFAULTS["k_target"], k_alpha=INTERVAL_DEFAULTS["k_alpha"],
                 max_len=INTERVAL_DEFAULTS["max_len"], lloyd_iters=INTERVAL_DEFAULTS["lloyd_iters"],
                 keep_tile_logs=False):
        ep = exact_partial if exact_partial is not None else ExactPartialTier(data, train_covs)
        self.p, self.n = ep.p, ep.n
        self.index = ivd.build_interval_index(data, k_target=k_target, k_alpha=k_alpha, lloyd_iters=lloyd_iters,
                                              max_len=max_len, floor=LOG_FLOOR, keep_tile_logs=keep_tile_logs,
                                              block_covs=ep.block_covs())
        self.build_seconds = self.index.build_seconds
        self.params = dict(k_target=k_target, k_alpha=k_alpha, max_len=max_len, lloyd_iters=lloyd_iters)
        self.score = "mean"  # "exact" scores against the tiles' own interval logs (needs keep_tile_logs)

    def nbytes(self):
        return int(self.index.nbytes()["total"])

    def _query(self, query_cov, genes):
        return query_cov_s(query_cov, genes, self.p)

    def _scan(self, cov_s, genes, c):
        r = ivd.search_interval_index(self.index, genes, cov_s, score=self.score)
        sel = slice(None) if c is None else slice(0, c)
        return r.rows[sel], r.scores[sel]


class WholeTier(_Tier):
    """Whole-matrix exact baseline: float32 whole-matrix log vectors (upper triangle, off-diagonals x sqrt 2, / sqrt p)."""

    kind = "whole"

    def __init__(self, data=None, train_covs=None):
        covs = [prepare_cov(c) for c in train_covs]
        self.p = covs[0].shape[0]
        self.iu = np.triu_indices(self.p)
        self.n = len(covs)
        self.X = np.empty((self.n, len(self.iu[0])), dtype=np.float32)
        for i, c in enumerate(covs):
            self.X[i] = self.vector(c)
            covs[i] = None

    def vector(self, cov):
        """Whole-log vector of an already prepared covariance (float32); its L2 distances are d_W."""
        return (frob_triu(metrics.log_spd(cov, eps=LOG_FLOOR), self.iu) / np.sqrt(self.p)).astype(np.float32)

    def nbytes(self):
        return int(self.X.nbytes)

    def distances(self, qv):
        return np.sqrt(np.maximum(((self.X - qv) ** 2).sum(1, dtype=np.float64), 0.0))

    def _query(self, query_cov, genes):
        return self.vector(prepare_cov(query_cov))

    def _scan(self, qv, genes, c):
        d = self.distances(qv)
        rows = _top(d, c)
        return rows, d[rows]


def query_cov_s(query_cov, genes, p):
    """A partial query's prepared |S| x |S| covariance in ``genes`` order.

    Given the full p x p matrix, it is shrunk over all p genes and then cut, exactly as the index's tiles.
    Given an |S| x |S| matrix, it is shrunk on its own (mean variance over S only), which matches the
    tiles' shrinkage only approximately (open point for Fig 4).
    """
    genes = np.asarray(genes)
    cov = np.asarray(raw_cov(query_cov), dtype=np.float64)
    if cov.shape[0] == p:
        return prepare_cov(cov)[np.ix_(genes, genes)]
    if cov.shape[0] != len(genes):
        raise ValueError(f"query covariance is {cov.shape[0]} x {cov.shape[0]}; expected {p} (all genes) "
                         f"or {len(genes)} (the gene set)")
    return metrics.shrink_cov(cov)


_TIERS = {"exact": ExactTier, "dag": DagTier, "exact_partial": ExactPartialTier, "interval": IntervalTier,
          "whole": WholeTier}


def build_tier(kind: str, data, train_covs=None, **kw):
    """Build one tier on ``data``'s niche/block layout from raw training covariances (data.labels order).

    dag accepts ``exact=`` (reuse an ExactTier's logs) and interval accepts ``exact_partial=`` (reuse its
    block covariances); then ``train_covs`` may be None. whole ignores the layout (``data`` may be None).
    """
    if kind not in _TIERS:
        raise ValueError(f"unknown tier '{kind}'; choose from {TIER_KINDS}")
    return _TIERS[kind](data, train_covs, **kw)


__all__ = [
    "TIER_KINDS", "LOG_FLOOR", "build_tier", "prepare_cov", "raw_cov", "query_cov_s",
    "ExactTier", "DagTier", "ExactPartialTier", "IntervalTier", "WholeTier",
    "BlockVectorizer", "niche_layouts", "LayoutShell", "block_log_bytes", "frob_triu", "ExactScorer",
    "build_dag_from_block_logs", "compact_dag", "nbytes", "n_nodes", "ADC", "query_pieces",
]
