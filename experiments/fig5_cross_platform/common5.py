"""Shared by the Fig 5 compute scripts (no plotting; safe in sbatch): the Xenium / Visium pair, its tiles, the
platform correction and the coarse tissue classes. Ported from the 6 Oct cross-platform script
(git show e514803^:benchmarks/cross_platform/cross_platform_tiers.py) onto the method-v1 public API.

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
  per_niche  shift per niche x block on Spindle's block-log vectors (the query's vector in that niche's layout
             against the stored vectors of that niche's tiles); the s.d. is over the weighted vector entries.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import data  # noqa: E402  (puts src/ on sys.path)
import spindle_dev as sd  # noqa: E402
from spindle_dev.metrics import LOG_FLOOR, log_spd  # noqa: E402
from spindle_dev.preprocessing import QuadTile, build_quadtree_tiles, build_tile_covs_full  # noqa: E402

OUT = data.RESULTS / "fig5_cross_platform"
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


def per_niche_vectors(spindle, q_covs):
    """Per query, {niche: block vector shifted block by block onto that niche's stored vectors}."""
    qvs = [spindle.query_vectors(c["cov"]) for c in q_covs]
    out = [dict() for _ in qvs]
    for k, (_, vec) in spindle.layouts.items():
        edges = np.cumsum([0] + [len(w) for w in vec.w])
        Q = np.stack([qv[k] for qv in qvs]).astype(np.float64)
        T = spindle.X[k].astype(np.float64)
        Z = np.empty_like(Q)
        for s, e in zip(edges[:-1], edges[1:]):
            Z[:, s:e] = shift(Q[:, s:e], T[:, s:e])
        for i in range(len(qvs)):
            out[i][k] = Z[i].astype(np.float32)
    return out


if __name__ == "__main__":  # build the tile cache once, before the c_cross array (sbatch run.sbatch common5.py)
    setup()
