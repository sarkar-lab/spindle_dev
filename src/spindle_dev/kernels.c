/* Compiled search kernels for Spindle, Spindle-lite and Spindle-lite partial, bit-identical to their numpy / Python
 * definitions (partial_search: see its own comment further down). Built by
 * kernels.py (gcc -O3 -mavx2 -ffp-contract=off -fno-builtin-pow); called through ctypes.
 *
 * Spindle (spindle_scan): d_B of every stored tile, as Spindle.distances_numpy computes it:
 * sqrt(((X - q) ** 2).sum(1, dtype=float64)) with X and q float32, i.e. the difference and its square in float32,
 * widened to float64, then numpy's pairwise sum in buffered chunks of 8192 elements added in order.
 *
 * Spindle-lite (lite_search): the same search as lite.greedy_probe + lite.radius_search, with the same output (rows,
 * order, scores, A_hat, radius, work). Because lb_b >= 0, a path's sum of lb^2 only grows, so the depth-first search
 * returns tile t exactly when sum_b lb_b(node_b(t))^2 <= r^2, and it computes d for exactly the nodes
 * {node_{b+1}(t): t's path passes through layer b}. Run breadth-first over the node index, the search keeps the tiles
 * still alive and moves them one layer at a time; the greedy probe keeps the members of its path. Node distances are
 * numpy's ((C[idx] - q) ** 2).sum(1) with C float32 and q float64 (difference and square in float64). Where lite.py
 * squares with Python's x ** 2 (the first layer's d and lb), the kernel calls libm pow(x, 2), which is not always
 * x * x (off by one ulp in ~0.1 % of values).
 *
 * numpy's pairwise sum: 8 accumulators over blocks of <= 128, halves at multiples of 8, starting from 0. Here the 8
 * accumulators are two 4-wide double vectors: lane-wise IEEE operations, so the same values as numpy's scalar loop.
 * No FMA contraction.
 */

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef double v4d __attribute__((vector_size(32)));
typedef float v4f __attribute__((vector_size(16)));

/* Spindle-lite terms: (double(c) - q)^2 */
static inline v4d sq4_lite(const float *c, const double *q) {
    v4f cf;
    v4d qd;
    memcpy(&cf, c, sizeof cf);
    memcpy(&qd, q, sizeof qd);
    v4d x = __builtin_convertvector(cf, v4d) - qd;
    return x * x;
}

static inline double sq1_lite(const float *c, const double *q) {
    double x = (double)*c - *q;
    return x * x;
}

/* Spindle terms: double((c - q)^2) with the difference and the square in float32 */
static inline v4d sq4_spindle(const float *c, const float *q) {
    v4f cf, qf;
    memcpy(&cf, c, sizeof cf);
    memcpy(&qf, q, sizeof qf);
    v4f x = cf - qf;
    return __builtin_convertvector(x * x, v4d);
}

static inline double sq1_spindle(const float *c, const float *q) {
    float x = *c - *q;
    return (double)(x * x);
}

/* numpy's pairwise_sum of the terms of c[0:n] against q[0:n] */
#define PAIRWISE(NAME, QT, SQ4, SQ1)                                                                   \
    static double NAME##_leaf(const float *c, const QT *q, int64_t n) {                               \
        if (n < 8) {                                                                                  \
            double res = 0.0;                                                                         \
            for (int64_t i = 0; i < n; i++) res += SQ1(c + i, q + i);                                \
            return res;                                                                               \
        }                                                                                             \
        v4d a = SQ4(c, q), b = SQ4(c + 4, q + 4); /* r0..r3, r4..r7 */                               \
        int64_t m = n - (n % 8), i = 8;                                                               \
        for (; i < m; i += 8) {                                                                       \
            a += SQ4(c + i, q + i);                                                                   \
            b += SQ4(c + i + 4, q + i + 4);                                                           \
        }                                                                                             \
        double res = ((a[0] + a[1]) + (a[2] + a[3])) + ((b[0] + b[1]) + (b[2] + b[3]));               \
        for (; i < n; i++) res += SQ1(c + i, q + i);                                                  \
        return res;                                                                                   \
    }                                                                                                 \
    static double NAME(const float *c, const QT *q, int64_t n) {                                      \
        if (n <= 128) return NAME##_leaf(c, q, n);                                                    \
        int64_t n2 = n / 2;                                                                           \
        n2 -= n2 % 8;                                                                                 \
        return NAME(c, q, n2) + NAME(c + n2, q + n2, n - n2);                                         \
    }

PAIRWISE(pairwise, double, sq4_lite, sq1_lite)
PAIRWISE(pairwise_spindle, float, sq4_spindle, sq1_spindle)

/* Spindle: d_B of each of the n rows of X (row length D) to q, into out[rows[i]] */
void spindle_scan(const float *X, const float *q, int64_t n, int64_t D, const int64_t *rows, double *out) {
    for (int64_t i = 0; i < n; i++) {
        const float *x = X + i * D;
        double acc = 0.0;
        for (int64_t s = 0; s < D; s += 8192) {                /* numpy's casting buffer */
            int64_t m = D - s < 8192 ? D - s : 8192;
            acc += pairwise_spindle(x + s, q + s, m);
        }
        out[rows[i]] = sqrt(acc);
    }
}

typedef struct {
    const float *cent;          /* all centre vectors, back to back */
    const int64_t *node_voff;   /* start of each node's vector in cent */
    const double *rad;          /* node radii (stored float32, widened) */
    const int64_t *layer_node0; /* first global node of each global layer (+ end) */
    const int64_t *layer_dim;   /* vector length of each global layer */
    const int64_t *layer_qoff;  /* start of each global layer's query slice in qcat */
    const int64_t *niche_layer0;/* first global layer of each niche (+ end) */
    const int64_t *niche_tile0; /* first tile of each niche (+ end) */
    const uint8_t *codes;       /* node index, tile-major per niche: code0[k] + t * B_k + b */
    const int64_t *code0;
    int64_t n_niche, n_nodes, n_max;
} Index;

typedef struct {
    const Index *ix;
    const double *qcat;
    double *dcache;             /* < 0: not computed yet */
    int64_t work;
} Query;

static inline double dist(Query *Q, int64_t node, int64_t layer) {
    double d = Q->dcache[node];
    if (d >= 0.0) return d;
    const Index *ix = Q->ix;
    d = sqrt(pairwise(ix->cent + ix->node_voff[node], Q->qcat + ix->layer_qoff[layer], ix->layer_dim[layer]));
    Q->dcache[node] = d;
    Q->work++;
    return d;
}

/* A^2 of niche k's greedy leaf: the closest first-layer node, then at each layer the closest node its path's tiles
 * reach (ties: the smaller node id = first in sorted child order) */
static double probe_niche(Query *Q, int64_t k, int64_t *alive) {
    const Index *ix = Q->ix;
    int64_t L0 = ix->niche_layer0[k], B = ix->niche_layer0[k + 1] - L0;
    int64_t n = ix->niche_tile0[k + 1] - ix->niche_tile0[k];
    const uint8_t *codes = ix->codes + ix->code0[k];
    int64_t g0 = ix->layer_node0[L0], K0 = ix->layer_node0[L0 + 1] - g0;
    double best = INFINITY;
    int64_t bj = -1;
    for (int64_t j = 0; j < K0; j++) {
        double d = dist(Q, g0 + j, L0);
        if (d < best) { best = d; bj = j; }
    }
    double a2 = pow(best, 2.0);                    /* lite: float(d0[j]) ** 2 */
    int64_t na = 0;
    for (int64_t t = 0; t < n; t++)
        if (codes[t * B] == bj) alive[na++] = t;
    for (int64_t b = 1; b < B; b++) {
        int64_t L = L0 + b, g = ix->layer_node0[L];
        best = INFINITY;
        bj = -1;
        for (int64_t i = 0; i < na; i++) {
            int64_t j = codes[alive[i] * B + b];
            double d = dist(Q, g + j, L);
            if (d < best || (d == best && j < bj)) { best = d; bj = j; }
        }
        a2 += best * best;
        int64_t m = 0;
        for (int64_t i = 0; i < na; i++)
            if (codes[alive[i] * B + b] == bj) alive[m++] = alive[i];
        na = m;
    }
    return a2;
}

/* every tile of niche k whose path keeps sum_b lb_b^2 <= r2 at every layer, appended with its A^2 */
static int64_t scan_niche(Query *Q, int64_t k, double r2, int64_t *alive, double *l2, double *a2,
                          double *out_a2, int64_t *out_k, int64_t *out_t, int64_t n_out) {
    const Index *ix = Q->ix;
    int64_t L0 = ix->niche_layer0[k], B = ix->niche_layer0[k + 1] - L0;
    int64_t n = ix->niche_tile0[k + 1] - ix->niche_tile0[k];
    const uint8_t *codes = ix->codes + ix->code0[k];
    int64_t g0 = ix->layer_node0[L0], K0 = ix->layer_node0[L0 + 1] - g0;
    for (int64_t j = 0; j < K0; j++) dist(Q, g0 + j, L0);     /* the whole first layer, as the DFS does */
    int64_t na = 0;
    for (int64_t t = 0; t < n; t++) {
        int64_t node = g0 + codes[t * B];
        double d = Q->dcache[node];
        double lb = d - ix->rad[node];
        if (lb < 0.0) lb = 0.0;
        double lsq = pow(lb, 2.0);                 /* lite: lb[j] ** 2, float(lb[j]) ** 2 */
        if (lsq <= r2) {
            alive[na++] = t;
            l2[t] = lsq;
            a2[t] = pow(d, 2.0);                   /* lite: float(d[j]) ** 2 */
        }
    }
    for (int64_t b = 1; b < B; b++) {
        int64_t L = L0 + b, g = ix->layer_node0[L], m = 0;
        for (int64_t i = 0; i < na; i++) {
            int64_t t = alive[i], node = g + codes[t * B + b];
            double d = dist(Q, node, L);
            double lb = d - ix->rad[node];
            if (lb < 0.0) lb = 0.0;
            double nl = l2[t] + lb * lb;
            if (nl <= r2) {
                l2[t] = nl;
                a2[t] = a2[t] + d * d;
                alive[m++] = t;
            }
        }
        na = m;
    }
    for (int64_t i = 0; i < na; i++) {
        int64_t t = alive[i];
        out_a2[n_out] = a2[t];
        out_k[n_out] = k;
        out_t[n_out] = t;
        n_out++;
    }
    return n_out;
}

/* One query: probe every niche, r = g * A_hat, scan every niche. Hits (unsorted) go to out_*; stats[0..2] =
 * A_hat, r, work. Returns the number of hits, or -1 if out of memory. */
int64_t lite_search(const float *cent, const int64_t *node_voff, const double *rad, const int64_t *layer_node0,
                    const int64_t *layer_dim, const int64_t *layer_qoff, const int64_t *niche_layer0,
                    const int64_t *niche_tile0, const uint8_t *codes, const int64_t *code0, int64_t n_niche,
                    int64_t n_nodes, const double *qcat, double g, double *out_a2, int64_t *out_k, int64_t *out_t,
                    double *stats) {
    Index ix = {cent, node_voff, rad, layer_node0, layer_dim, layer_qoff, niche_layer0, niche_tile0, codes, code0,
                n_niche, n_nodes, 0};
    for (int64_t k = 0; k < n_niche; k++) {
        int64_t n = niche_tile0[k + 1] - niche_tile0[k];
        if (n > ix.n_max) ix.n_max = n;
    }
    double *dcache = malloc(sizeof(double) * (size_t)(n_nodes > 0 ? n_nodes : 1));
    int64_t *alive = malloc(sizeof(int64_t) * (size_t)(ix.n_max + 1));
    double *l2 = malloc(sizeof(double) * (size_t)(ix.n_max + 1));
    double *a2 = malloc(sizeof(double) * (size_t)(ix.n_max + 1));
    if (!dcache || !alive || !l2 || !a2) {
        free(dcache); free(alive); free(l2); free(a2);
        return -1;
    }
    for (int64_t i = 0; i < n_nodes; i++) dcache[i] = -1.0;
    Query Q = {&ix, qcat, dcache, 0};
    double best = INFINITY;
    for (int64_t k = 0; k < n_niche; k++) {
        double v = probe_niche(&Q, k, alive);
        if (v < best) best = v;
    }
    double a_hat = sqrt(best), r = g * a_hat, r2 = r * r;
    int64_t n_out = 0;
    for (int64_t k = 0; k < n_niche; k++)
        n_out = scan_niche(&Q, k, r2, alive, l2, a2, out_a2, out_k, out_t, n_out);
    stats[0] = a_hat;
    stats[1] = r;
    stats[2] = (double)Q.work;
    free(dcache); free(alive); free(l2); free(a2);
    return n_out;
}

/* Spindle-lite partial (partial_search): the same search as lite_partial.greedy_probe + lite_partial.radius_search,
 * with the same output. The layers are the query's own pieces (one stored dyadic interval each, niche by niche, left to
 * right), so the index is addressed through a per-query layer list. Node distances as in lite_search (pairwise, C
 * float32 and q float64); squares are x * x everywhere (the Python reference does the same). The radius is pruned in
 * squares as r2 = (g * g) * (the probe's best A^2), with no square-root round trip.
 *
 * Index (global intervals run over niches in sorted order, then the niche's intervals in (block, s, j) order):
 * cent / node_voff / rad as in lite_search; iv_node0 first global node of each interval (+ end); iv_dim vector length;
 * niche_iv0 first interval of each niche (+ end); niche_tile0 first tile of each niche (+ end); codes the uint8 node
 * index tile-major per niche: code0[k] + t * I_k + (interval - niche_iv0[k]).
 * Query: qlayer the global interval of each query layer, niche by niche; niche_q0 first query layer of each niche (+
 * end); qoff each query layer's start in qcat. */
typedef struct {
    const float *cent;
    const int64_t *node_voff;
    const double *rad;
    const int64_t *iv_node0, *iv_dim, *niche_iv0, *niche_tile0, *code0;
    const uint8_t *codes;
    const int64_t *qlayer, *niche_q0, *qoff, *lnode0;
    const double *qcat;
    double *dcache;             /* per query layer and node; < 0: not computed yet */
    int64_t work;
} PQuery;

static inline double pdist(PQuery *Q, int64_t l, int64_t j) {
    int64_t c = Q->lnode0[l] + j;
    double d = Q->dcache[c];
    if (d >= 0.0) return d;
    int64_t iv = Q->qlayer[l];
    d = sqrt(pairwise(Q->cent + Q->node_voff[Q->iv_node0[iv] + j], Q->qcat + Q->qoff[l], Q->iv_dim[iv]));
    Q->dcache[c] = d;
    Q->work++;
    return d;
}

static inline uint8_t pcode(const PQuery *Q, int64_t k, int64_t t, int64_t l) {
    int64_t I = Q->niche_iv0[k + 1] - Q->niche_iv0[k];
    return Q->codes[Q->code0[k] + t * I + (Q->qlayer[l] - Q->niche_iv0[k])];
}

static double pprobe_niche(PQuery *Q, int64_t k, int64_t *alive) {
    int64_t l0 = Q->niche_q0[k], l1 = Q->niche_q0[k + 1];
    int64_t n = Q->niche_tile0[k + 1] - Q->niche_tile0[k];
    if (l1 <= l0) return INFINITY;
    int64_t iv = Q->qlayer[l0], K0 = Q->iv_node0[iv + 1] - Q->iv_node0[iv];
    double best = INFINITY;
    int64_t bj = -1;
    for (int64_t j = 0; j < K0; j++) {
        double d = pdist(Q, l0, j);
        if (d < best) { best = d; bj = j; }
    }
    double a2 = best * best;
    int64_t na = 0;
    for (int64_t t = 0; t < n; t++)
        if (pcode(Q, k, t, l0) == bj) alive[na++] = t;
    for (int64_t l = l0 + 1; l < l1; l++) {
        best = INFINITY;
        bj = -1;
        for (int64_t i = 0; i < na; i++) {
            int64_t j = pcode(Q, k, alive[i], l);
            double d = pdist(Q, l, j);
            if (d < best || (d == best && j < bj)) { best = d; bj = j; }
        }
        a2 += best * best;
        int64_t m = 0;
        for (int64_t i = 0; i < na; i++)
            if (pcode(Q, k, alive[i], l) == bj) alive[m++] = alive[i];
        na = m;
    }
    return a2;
}

static int64_t pscan_niche(PQuery *Q, int64_t k, double r2, int64_t *alive, double *l2, double *a2,
                           double *out_a2, int64_t *out_k, int64_t *out_t, int64_t n_out) {
    int64_t l0 = Q->niche_q0[k], l1 = Q->niche_q0[k + 1];
    int64_t n = Q->niche_tile0[k + 1] - Q->niche_tile0[k];
    if (l1 <= l0) return n_out;
    int64_t iv = Q->qlayer[l0], g0 = Q->iv_node0[iv], K0 = Q->iv_node0[iv + 1] - g0;
    for (int64_t j = 0; j < K0; j++) pdist(Q, l0, j);        /* the whole first layer, as the DFS does */
    int64_t na = 0;
    for (int64_t t = 0; t < n; t++) {
        int64_t j = pcode(Q, k, t, l0);
        double d = Q->dcache[Q->lnode0[l0] + j];
        double lb = d - Q->rad[g0 + j];
        if (lb < 0.0) lb = 0.0;
        double lsq = lb * lb;
        if (lsq <= r2) {
            alive[na++] = t;
            l2[t] = lsq;
            a2[t] = d * d;
        }
    }
    for (int64_t l = l0 + 1; l < l1; l++) {
        int64_t g = Q->iv_node0[Q->qlayer[l]], m = 0;
        for (int64_t i = 0; i < na; i++) {
            int64_t t = alive[i], j = pcode(Q, k, t, l);
            double d = pdist(Q, l, j);
            double lb = d - Q->rad[g + j];
            if (lb < 0.0) lb = 0.0;
            double nl = l2[t] + lb * lb;
            if (nl <= r2) {
                l2[t] = nl;
                a2[t] = a2[t] + d * d;
                alive[m++] = t;
            }
        }
        na = m;
    }
    for (int64_t i = 0; i < na; i++) {
        int64_t t = alive[i];
        out_a2[n_out] = a2[t];
        out_k[n_out] = k;
        out_t[n_out] = t;
        n_out++;
    }
    return n_out;
}

/* One gene-set query: probe every niche, r2 = (g * g) * best A^2, scan every niche. Hits (unsorted, A^2 unnormalised)
 * go to out_*; stats[0..1] = best A^2 of the probe, work. Returns the number of hits, or -1 if out of memory. */
int64_t partial_search(const float *cent, const int64_t *node_voff, const double *rad, const int64_t *iv_node0,
                       const int64_t *iv_dim, const int64_t *niche_iv0, const int64_t *niche_tile0,
                       const uint8_t *codes, const int64_t *code0, int64_t n_niche, const int64_t *qlayer,
                       const int64_t *niche_q0, const int64_t *qoff, const double *qcat, double g, double *out_a2,
                       int64_t *out_k, int64_t *out_t, double *stats) {
    int64_t n_layers = niche_q0[n_niche], n_max = 0;
    int64_t *lnode0 = malloc(sizeof(int64_t) * (size_t)(n_layers + 1));
    if (!lnode0) return -1;
    lnode0[0] = 0;
    for (int64_t l = 0; l < n_layers; l++)
        lnode0[l + 1] = lnode0[l] + (iv_node0[qlayer[l] + 1] - iv_node0[qlayer[l]]);
    for (int64_t k = 0; k < n_niche; k++) {
        int64_t n = niche_tile0[k + 1] - niche_tile0[k];
        if (n > n_max) n_max = n;
    }
    double *dcache = malloc(sizeof(double) * (size_t)(lnode0[n_layers] > 0 ? lnode0[n_layers] : 1));
    int64_t *alive = malloc(sizeof(int64_t) * (size_t)(n_max + 1));
    double *l2 = malloc(sizeof(double) * (size_t)(n_max + 1));
    double *a2 = malloc(sizeof(double) * (size_t)(n_max + 1));
    if (!dcache || !alive || !l2 || !a2) {
        free(lnode0); free(dcache); free(alive); free(l2); free(a2);
        return -1;
    }
    for (int64_t i = 0; i < lnode0[n_layers]; i++) dcache[i] = -1.0;
    PQuery Q = {cent, node_voff, rad, iv_node0, iv_dim, niche_iv0, niche_tile0, code0, codes,
                qlayer, niche_q0, qoff, lnode0, qcat, dcache, 0};
    double best = INFINITY;
    for (int64_t k = 0; k < n_niche; k++) {
        double v = pprobe_niche(&Q, k, alive);
        if (v < best) best = v;
    }
    double r2 = (g * g) * best;
    int64_t n_out = 0;
    for (int64_t k = 0; k < n_niche; k++)
        n_out = pscan_niche(&Q, k, r2, alive, l2, a2, out_a2, out_k, out_t, n_out);
    stats[0] = best;
    stats[1] = (double)Q.work;
    free(lnode0); free(dcache); free(alive); free(l2); free(a2);
    return n_out;
}

/* numpy-identical d = sqrt(sum((c - q)^2)) of n rows of length dim (row stride dim); for the identity check */
void lite_row_dists(const float *c, const double *q, int64_t n, int64_t dim, double *out) {
    for (int64_t i = 0; i < n; i++) out[i] = sqrt(pairwise(c + i * dim, q, dim));
}
