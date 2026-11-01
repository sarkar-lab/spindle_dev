"""The cross-platform experiment (Fig 5; R5_cross_platform.ipynb): the Xenium / Visium pair, its tiles, the platform
correction, the search per layout seed, the section maps, the platform-bias PCA, the example query and the summaries.
Ported from experiments/fig5_cross_platform/{common5,c_cross,c_extras,c_summary}.py (in git history), themselves from the
6 Oct cross-platform script (git show e514803^:benchmarks/cross_platform/cross_platform_tiers.py), onto the public API.

Data: the two registered serial breast sections of datasets.yaml (cross_modal_xenium, cross_modal_visium), read with
sd.load_adata (cells annotated "Unlabeled" dropped: 4,305 Xenium cells; Visium has none), raw counts on the genes the
two panels share (sorted names). The Xenium section is quadtree-tiled (<= 2,000 cells per tile: a Visium spot is
~55 um and a tile needs >= 10 spots) and the same boxes are laid over the Visium spots, kept if they hold >= 10
spots, so every Visium tile has a co-located Xenium tile. Covariances: build_tile_covs_full (shrunk once, as every
index). Seed-independent; cached in results/_cache/cross_modal/tiles2000.pkl.

Platform correction (statistics over all query tiles and all index tiles; no labels), on the shrunk covariances:
  shift(Lq, Lt) = (Lq - mean_q) * min(1, sd_t / sd_q) + mean_t    (entry-wise means, one scalar s.d. over all entries)
  none       no correction;
  global     shift of the whole p x p log, then exp (the default);
  per_niche  shift per niche x block on Spindle-naive's block-log vectors (the query's vector in that niche's layout
             against the stored vectors of that niche's tiles); the s.d. is over the weighted vector entries.

Search (``search``, per layout seed 0-4; the seed seeds only build_layout's PCA / UMAP / Leiden, every tile is indexed)
and direction (x2v: Xenium queries against a Visium index; v2x: the reverse): the index platform's tiles get a layout,
Spindle-naive and Spindle (from_naive). Every tile of the query platform is a query. Methods ranked per query (all
index tiles, best first):
  naive_<arm>     SpindleNaive.distances (d_B) of the corrected query, arm in none / per_niche / global
  spindle         Spindle.search_vectors (g = 0.25) of the global-corrected query; graded against naive_global

Declared before the run (Fig 5; user, 26 Oct):
- Headline: tissue agreement of naive_global = Pearson r over queries between the query tile's tumour fraction
  (invasive + DCIS cells or spots, own platform's annotation) and the mean tumour fraction of its top 10 on the other
  platform; per (direction, seed), then mean +/- s.d. over seeds 0-4. Base rate: the mean absolute difference to
  all index tiles (random) beside the top-10 one. Expected (6 Oct, old code): r about 0.85-0.89.
- Ablation: the same r for none / per_niche / global. The expression and variance-only controls were run once
  (26 Oct, jobs 23272/23279: r 0.93-0.94 and 0.78-0.82) and dropped from the figure and text (user, 26 Oct).
- Spindle (text only): NX5 and Overlap(5 %, 5 % N) against naive_global, mean over queries per seed.
- Co-located tile rank (hit@1/5/10) is kept in the per-query table; no panel (user, 26 Oct).

Extras (seed 0 / seed-independent): section maps (``maps``), platform-bias PCA (``bias_pca``), example query
(``example``; rule in its docstring).
"""

import pickle

import numpy as np
import pandas as pd

import data  # puts src/ on sys.path
import retrieval as rt
import spindle_dev as sd
from spindle_dev.metrics import LOG_FLOOR, log_spd
from spindle_dev.preprocessing import QuadTile, build_quadtree_tiles, build_tile_covs_full

CACHE = data.CACHE / "cross_modal"
MAX_PTS = 2000
MIN_SPOTS = 10
PLATFORMS = ["Xenium", "Visium"]
KEYS = {"Xenium": "cross_modal_xenium", "Visium": "cross_modal_visium"}
DIRECTIONS = {"x2v": ("Visium", "Xenium"), "v2x": ("Xenium", "Visium")}  # direction -> (index, query)
ARMS = ["none", "per_niche", "global"]
TOP = 10  # hits whose tumour fraction is compared with the query's

# Shared coarse classes of the Xenium cell types and the Visium region labels; labels not listed (the Xenium immune
# types and the hybrids) are "Immune".
COARSE = {
    "invasive": "Invasive tumour", "mixed/invasive": "Invasive tumour",
    "DCIS #1": "DCIS", "DCIS #2": "DCIS",
    "stromal": "Stroma", "stromal/endothelial": "Stroma",
    "immune": "Immune", "stromal/endothelial/immune": "Immune",
    "myoepithelial/stromal/immune": "Myoepithelial",
    "adipocytes": "Adipocytes", "mixed": "Mixed",
    "Invasive_Tumor": "Invasive tumour", "Prolif_Invasive_Tumor": "Invasive tumour",
    "DCIS_1": "DCIS", "DCIS_2": "DCIS",
    "Stromal": "Stroma", "Endothelial": "Stroma", "Perivascular-Like": "Stroma",
    "Myoepi_ACTA2+": "Myoepithelial", "Myoepi_KRT15+": "Myoepithelial",
}
CLASSES = ["Invasive tumour", "DCIS", "Myoepithelial", "Stroma", "Immune", "Adipocytes", "Mixed"]
TUMOUR = ["Invasive tumour", "DCIS"]


def coarse(clusters):
    return np.array([COARSE.get(c, "Immune") for c in np.asarray(clusters).astype(str)])


def load_pair():
    """{platform: AnnData on the shared genes (sorted)}, genes."""
    ads = {}
    for m in PLATFORMS:
        a = sd.load_adata(data.paths.dataset_path(KEYS[m]))
        a.var_names_make_unique()
        ads[m] = a
    genes = sorted(set(ads["Xenium"].var_names) & set(ads["Visium"].var_names))
    return {m: a[:, genes].copy() for m, a in ads.items()}, np.array(genes)


def build_tiles(xy_xe, xy_vi):
    """Xenium quadtree tiles and the same boxes over Visium (>= MIN_SPOTS spots); matched[j] = the Xenium tile of
    Visium tile j."""
    tiles_xe = build_quadtree_tiles(xy_xe, max_pts=MAX_PTS, min_side=0.0, max_depth=40)
    tiles_vi, matched = [], []
    for i, t in enumerate(tiles_xe):
        x0, y0, x1, y1 = t.bbox
        inside = np.flatnonzero((xy_vi[:, 0] >= x0) & (xy_vi[:, 0] < x1) & (xy_vi[:, 1] >= y0) & (xy_vi[:, 1] < y1))
        if len(inside) >= MIN_SPOTS:
            tiles_vi.append(QuadTile(len(tiles_vi), t.bbox, inside))
            matched.append(i)
    for i, t in enumerate(tiles_xe):
        t.id = i
    return tiles_xe, tiles_vi, np.asarray(matched)


def setup():
    """Tiles, shrunk covariances, composition and mean expression of both platforms (cached)."""
    f = CACHE / f"tiles{MAX_PTS}.pkl"
    if f.exists():
        with open(f, "rb") as fh:
            return pickle.load(fh)
    ads, genes = load_pair()
    xy = {m: np.asarray(a.obsm["spatial"])[:, :2] for m, a in ads.items()}
    tiles_xe, tiles_vi, matched = build_tiles(xy["Xenium"], xy["Visium"])
    tiles = {"Xenium": tiles_xe, "Visium": tiles_vi}
    S = {"genes": genes, "matched_vi_to_xe": matched, "tiles": tiles,
         "n_obs": {m: a.n_obs for m, a in ads.items()}, "stamp": data.stamp(), "covs": {}, "comp": {}, "mean": {}}
    for m, a in ads.items():
        S["covs"][m] = build_tile_covs_full(a, tiles[m], n_jobs=8, eps=1e-6)
        cl = coarse(a.obs["Cluster"])
        S["comp"][m] = np.array([[np.mean(cl[t.idx] == c) for c in CLASSES] for t in tiles[m]])
        X = a.X
        S["mean"][m] = np.stack([np.asarray(X[t.idx].mean(0)).ravel() for t in tiles[m]])
    S["cells"] = {m: {"xy": xy[m], "coarse": coarse(a.obs["Cluster"])} for m, a in ads.items()}
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(f, "wb") as fh:
        pickle.dump(S, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"tiles{MAX_PTS}: {len(genes)} genes, {len(tiles_xe)} Xenium / {len(tiles_vi)} Visium tiles", flush=True)
    return S


def tumour(S, m):
    return S["comp"][m][:, [CLASSES.index(c) for c in TUMOUR]].sum(1)


def match_of(S, direction):
    """matched[q] = the co-located index tile of query q (-1 if none)."""
    m = S["matched_vi_to_xe"]
    if direction == "v2x":
        return m.copy()
    out = np.full(len(S["tiles"]["Xenium"]), -1)
    out[m] = np.arange(len(m))
    return out


def shift(q, t):
    """(q - mean_q) * min(1, sd_t / sd_q) + mean_t over the first axis (stacks of queries / index tiles)."""
    sd_q, sd_t = float(q.std()), float(t.std())
    return (q - q.mean(0)) * min(1.0, sd_t / (sd_q if sd_q > 1e-6 else 1.0)) + t.mean(0)


def whole_logs(covs):
    return np.stack([log_spd(np.asarray(c["cov"], dtype=np.float64), eps=LOG_FLOOR) for c in covs])


def exp_sym(L):
    w, V = np.linalg.eigh(0.5 * (L + L.T))
    return (V * np.exp(w)) @ V.T


def global_corrected(q_covs, t_covs):
    """Query covariances after the whole-matrix log shift onto the index tiles."""
    return [exp_sym(x) for x in shift(whole_logs(q_covs), whole_logs(t_covs))]


def per_niche_vectors(naive, q_covs):
    """Per query, {niche: block vector shifted block by block onto that niche's stored vectors}."""
    qvs = [naive.query_vectors(c["cov"]) for c in q_covs]
    out = [dict() for _ in qvs]
    for k, (_, vec) in naive.layouts.items():
        edges = np.cumsum([0] + [len(w) for w in vec.w])
        Q = np.stack([qv[k] for qv in qvs]).astype(np.float64)
        T = naive.X[k].astype(np.float64)
        Z = np.empty_like(Q)
        for s, e in zip(edges[:-1], edges[1:]):
            Z[:, s:e] = shift(Q[:, s:e], T[:, s:e])
        for i in range(len(qvs)):
            out[i][k] = Z[i].astype(np.float32)
    return out



# ---------------------------------------------------------------- search (c_cross.py)
def rank_of(d, t):
    """Tie-aware rank (1 = nearest) of index tile t."""
    return 1 + int((d < d[t]).sum()) + 0.5 * int((d == d[t]).sum() - 1)


def search(S, seed):
    """Both directions for one layout seed: {"per_query", "top10", "spindle", "index"} DataFrames."""
    st = data.stamp()
    per_q, top, spindle_rows, info = [], [], [], []
    for direction, (index_mod, query_mod) in DIRECTIONS.items():
        t_covs, q_covs = S["covs"][index_mod], S["covs"][query_mod]
        layout = sd.build_layout(S["tiles"][index_mod], t_covs, S["genes"], S["n_obs"][index_mod],
                                 random_state=seed)
        naive = sd.SpindleNaive(layout, t_covs)
        spindle = sd.Spindle.from_naive(naive, layout)
        n = naive.n
        qv = {"none": [naive.query_vectors(c["cov"]) for c in q_covs],
              "per_niche": per_niche_vectors(naive, q_covs),
              "global": [naive.query_vectors(c) for c in global_corrected(q_covs, t_covs)]}
        D = {f"naive_{arm}": np.stack([naive.distances(v) for v in qv[arm]]) for arm in ARMS}
        matched = match_of(S, direction)
        tum_q, tum_t = tumour(S, query_mod), tumour(S, index_mod)
        base = {"seed": seed, "direction": direction, "N": n, "tag": st["tag"], "describe": st["describe"]}
        for method, Dm in D.items():
            for q, d in enumerate(Dm):
                order = np.argsort(d, kind="stable")
                rank = rank_of(d, matched[q]) if matched[q] >= 0 else np.nan
                per_q.append({**base, "method": method, "query": q, "matched_tile": int(matched[q]),
                              "query_tumour": tum_q[q], "hits_tumour": tum_t[order[:TOP]].mean(),
                              "random_mae": float(np.abs(tum_t - tum_q[q]).mean()), "matched_rank": rank,
                              **{f"hit@{k}": float(rank <= k) if matched[q] >= 0 else np.nan for k in (1, 5, 10)}})
                top += [{**base, "method": method, "query": q, "rank": r + 1, "index_tile": int(t), "d": d[t]}
                        for r, t in enumerate(order[:TOP])]
        cc = rt.c5(n)
        for q, v in enumerate(qv["global"]):
            rows, _, s = spindle.search_vectors(v)
            d = D["naive_global"][q]
            spindle_rows.append({**base, "query": q, "c5": cc, "n_found": len(rows), "NX5": rt.nx(d, rows, cc),
                                 "overlap5": rt.overlap(d, rows, cc), "work": s["work"]})
        info.append({**base, "index_platform": index_mod, "n_query": len(q_covs), "p": len(S["genes"]),
                     "n_niches": len(naive.layouts), "n_blocks": int(sum(len(v.runs) for _, v in
                                                                        naive.layouts.values())),
                     "bytes_naive": naive.nbytes(), "bytes_spindle": spindle.nbytes()})
        pq = pd.DataFrame([r for r in per_q if r["direction"] == direction])
        r = pq.groupby("method").apply(lambda g: np.corrcoef(g.query_tumour, g.hits_tumour)[0, 1])
        sr = pd.DataFrame([x for x in spindle_rows if x["direction"] == direction])
        print(f"seed {seed} {direction}: N = {n}, {len(naive.layouts)} niches; tissue r "
              f"{r.round(3).to_dict()}; spindle NX5 {100 * sr.NX5.mean():.1f} %, overlap {sr.overlap5.mean():.2f}",
              flush=True)
        if seed == data.EXAMPLE_SEED and direction == "v2x":
            check(naive, S)
    return {"per_query": pd.DataFrame(per_q), "top10": pd.DataFrame(top), "spindle": pd.DataFrame(spindle_rows),
            "index": pd.DataFrame(info)}


def check(naive, S):
    """The Xenium index searched with its own tiles: rank 1 at distance 0; the global shift of a platform onto
    itself is the identity."""
    covs = S["covs"]["Xenium"]
    D = np.stack([naive.distances(naive.query_vectors(c["cov"])) for c in covs])
    own = np.arange(len(covs))
    print(f"check: self rank 1 for {np.mean(D.argmin(1) == own):.3f} of tiles, max self d {D[own, own].max():.1e}; "
          f"global shift onto itself: max |d - d_none| = "
          f"{np.abs(np.stack([naive.distances(naive.query_vectors(c)) for c in global_corrected(covs, covs)]) - D).max():.1e}",
          flush=True)


# ---------------------------------------------------------------- maps, bias PCA, example (c_extras.py)
N_PC = 5


def maps(S):
    """(tiles table of both platforms: box, cells / spots, composition, tumour fraction, co-located Xenium tile;
    {npz array name: array} of every Xenium cell and Visium spot: x, y, coarse class)."""
    st = data.stamp()
    rows = []
    for m in PLATFORMS:
        tum = tumour(S, m)
        for i, t in enumerate(S["tiles"][m]):
            rows.append({"platform": m, "tile": i, "x0": t.bbox[0], "y0": t.bbox[1], "x1": t.bbox[2], "y1": t.bbox[3],
                         "n_obs": len(t.idx), "matched_xenium": int(S["matched_vi_to_xe"][i]) if m == "Visium" else i,
                         "tumour": tum[i], **dict(zip(CLASSES, S["comp"][m][i])), "tag": st["tag"]})
    cells = S["cells"]
    arrays = {"classes": np.array(CLASSES),
              **{f"{m}_xy": cells[m]["xy"].astype(np.float32) for m in PLATFORMS},
              **{f"{m}_class": np.array([CLASSES.index(c) for c in cells[m]["coarse"]], dtype=np.int8)
                 for m in PLATFORMS}}
    return pd.DataFrame(rows), arrays


def vectorize(logs):
    p = logs.shape[1]
    iu = np.triu_indices(p)
    w = np.where(iu[0] == iu[1], 1.0, np.sqrt(2.0)) / np.sqrt(p)
    return logs[:, iu[0], iu[1]] * w


def bias_pca(S):
    """PCA (5 components) of the shrunk whole-matrix log vectors (upper triangle, off-diagonals x sqrt 2, / sqrt p),
    fit on the index tiles; the query tiles projected before and after the global shift; platform silhouette
    (Euclidean, index vs query tiles) per state. Returns (projections, summary)."""
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    st = data.stamp()
    logs = {m: whole_logs(S["covs"][m]) for m in PLATFORMS}
    rows, summary = [], []
    for direction, (index_mod, query_mod) in DIRECTIONS.items():
        V_t = vectorize(logs[index_mod])
        states = {False: vectorize(logs[query_mod]), True: vectorize(shift(logs[query_mod], logs[index_mod]))}
        pca = PCA(n_components=N_PC, random_state=0).fit(V_t)
        for k, z in enumerate(pca.transform(V_t)):
            rows.append({"direction": direction, "platform": index_mod, "role": "index", "corrected": False,
                         "tile": k, **{f"PC{j + 1}": z[j] for j in range(N_PC)}})
        for corr, V_q in states.items():
            for k, z in enumerate(pca.transform(V_q)):
                rows.append({"direction": direction, "platform": query_mod, "role": "query", "corrected": corr,
                             "tile": k, **{f"PC{j + 1}": z[j] for j in range(N_PC)}})
            lab = np.r_[np.zeros(len(V_t)), np.ones(len(V_q))]
            summary.append({"direction": direction, "corrected": corr,
                            "silhouette": float(silhouette_score(np.vstack([V_t, V_q]), lab)),
                            "centroid_gap": float(np.linalg.norm(V_t.mean(0) - V_q.mean(0))),
                            "pc12_var": float(pca.explained_variance_ratio_[:2].sum()), "tag": st["tag"]})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def example(S, top10):
    """Fig 5F example. top10: {seed: the search's "top10" table}. Rule (user, 26 Oct; replaces the first rule, which
    picked the Visium query with the largest DCIS share, query 12: top 5 at 2-54 % DCIS, co-located tile at rank 54):
    seed 0, direction v2x (Visium query, Xenium index), naive_global. A query's focal class is the larger of its DCIS
    and immune shares (classes in separate, recognisable foci on both platforms). Among the queries whose focal share
    is > 0.5 and whose co-located Xenium tile is in the top 5, the one whose top 5 have the largest mean share of the
    query's focal class (ties: lower query id). Reported with it: the top 5's share against the Xenium section's cell
    fraction of that class, the co-located tile's rank, the top 5 in seeds 1-4, and the base rate = the share of all
    focal-rich Visium queries (any co-located rank) whose top-5 mean exceeds the section's fraction."""
    st = data.stamp()

    def top5(seed):
        t = top10[seed]
        t = t[(t.direction == "v2x") & (t.method == "naive_global") & (t["rank"] <= 5)]
        return t.groupby("query")["index_tile"].apply(lambda x: x.to_numpy())
    focal = ["DCIS", "Immune"]
    j = [CLASSES.index(c) for c in focal]
    comp_v, comp_x = S["comp"]["Visium"][:, j], S["comp"]["Xenium"][:, j]
    section = np.array([np.mean(S["cells"]["Xenium"]["coarse"] == c) for c in focal])
    hits = top5(data.EXAMPLE_SEED)
    matched = S["matched_vi_to_xe"]
    rows = []
    for q, h in hits.items():
        f = int(np.argmax(comp_v[q]))
        rows.append({"query": q, "focal": focal[f], "query_share": comp_v[q, f],
                     "hit_share": comp_x[h, f].mean(), "section": section[f], "found": matched[q] in h})
    R = pd.DataFrame(rows)
    rich = R[R.query_share > 0.5]
    base_rate = float((rich.hit_share > rich.section).mean())
    cand = rich[rich.found].sort_values(["hit_share", "query"], ascending=[False, True])
    q, cls = int(cand["query"].iloc[0]), cand["focal"].iloc[0]
    f = focal.index(cls)
    m = int(matched[q])
    h = hits[q]
    out = pd.DataFrame({"rank": np.arange(1, 6), "index_tile": h, "matched": h == m, "hit_share": comp_x[h, f]})
    out["query"], out["focal_class"], out["query_share"] = q, cls, comp_v[q, f]
    out["matched_tile"], out["section_share"] = m, section[f]
    out["n_rich"], out["n_candidates"], out["rich_above_section"] = len(rich), len(cand), base_rate
    for s_ in data.SEEDS[1:]:
        out[f"top5_seed{s_}"] = " ".join(map(str, top5(s_)[q]))
    out["tag"] = st["tag"]
    print(f"example: Visium query {q} ({len(cand)} candidates of {len(rich)} focal-rich queries), {cls} "
          f"{comp_v[q, f]:.2f}; top-5 Xenium {cls} {out.hit_share.min():.2f}-{out.hit_share.max():.2f} (mean "
          f"{out.hit_share.mean():.2f}; section {section[f]:.3f}); co-located tile {m} at rank "
          f"{int(out.loc[out.matched, 'rank'].iloc[0])}; base rate {base_rate:.2f}; seeds 1-4: "
          f"{[out[f'top5_seed{s_}'].iloc[0] for s_ in data.SEEDS[1:]]}", flush=True)
    return out


# ---------------------------------------------------------------- summaries (c_summary.py)
def seed_stats(df, keys):
    num = [c for c in df.columns if c not in keys + ["seed", "tag"]]
    out = df.groupby(keys)[num].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out = out.reset_index()
    out["n_seeds"] = df.groupby(keys)["seed"].nunique().to_numpy()
    out["tag"] = data.TAG
    return out


def tissue_seeds(pq):
    """Per (direction, method, seed): tissue agreement r (query tumour fraction vs the mean of its top 10), mean
    absolute difference (top 10; random = all index tiles), co-located rank (hit@1/5/10, median rank), N."""
    rows = []
    for (d, m, s), g in pq.groupby(["direction", "method", "seed"]):
        h = g[g.matched_tile >= 0]
        rows.append({"direction": d, "method": m, "seed": s, "N": int(g.N.iloc[0]), "n_query": len(g),
                     "tissue_r": float(np.corrcoef(g.query_tumour, g.hits_tumour)[0, 1]),
                     "mae": float((g.query_tumour - g.hits_tumour).abs().mean()), "mae_random": g.random_mae.mean(),
                     "hit1": h["hit@1"].mean(), "hit5": h["hit@5"].mean(), "hit10": h["hit@10"].mean(),
                     "hit5_random": 5 / g.N.iloc[0], "median_rank": h.matched_rank.median(), "n_matched": len(h),
                     "tag": data.TAG})
    return pd.DataFrame(rows)


def spindle_summary(sp):
    """Per direction: Spindle against naive_global, mean over queries per seed, then mean and s.d. over seeds."""
    st = sp.groupby(["direction", "seed"])[["NX5", "overlap5", "n_found", "work"]].mean().reset_index()
    return seed_stats(st, ["direction"])


def index_summary(ix):
    """Per direction: N, query tiles, p, niches and blocks (min-max over seeds), Spindle-naive and Spindle MB."""
    index = ix.groupby("direction").agg(N=("N", "first"), n_query=("n_query", "first"), p=("p", "first"),
                                        niches_min=("n_niches", "min"), niches_max=("n_niches", "max"),
                                        blocks_min=("n_blocks", "min"), blocks_max=("n_blocks", "max"),
                                        naive_mb=("bytes_naive", "mean"), spindle_mb=("bytes_spindle", "mean"))
    index[["naive_mb", "spindle_mb"]] /= 1e6
    index["tag"] = data.TAG
    return index.reset_index()
