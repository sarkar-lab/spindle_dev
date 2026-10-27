"""Shared by the Fig 7 compute scripts (no plotting; safe in sbatch): any of the 8 indexed sections, its cached builds,
cell labels and expression. The statistics are Fig 6's (experiments/fig6_breast/common6.py), imported unchanged, so
breast reproduces Fig 6.

Cell labels for compositions: breast, the 8 classes of common6.CLASSES over the section's annotation (``Cluster``), as
Fig 6; every other section, the 10x graph-based clusters of its Xenium output (analysis.tar.gz,
gene_expression_graphclust/clusters.csv, matched by cell barcode; cells without a cluster are left out). The skin worked
example groups the clusters into classes (skin_classes). Expression as Fig 6: raw counts on the index genes,
log1p(x / total * median total).
"""

import io
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fig6_breast"))
import common6 as c6  # noqa: E402  (puts experiments/common and src/ on sys.path)
from common6 import data, sd  # noqa: E402
from signatures import TISSUE, TISSUE_MODULES  # noqa: E402

KEYS = data.indexed_keys()
EXAMPLE_KEY = "skin_melanoma"
OUT = data.RESULTS / "fig7_skin"
TOP = c6.TOP
N_RANDOM = c6.N_RANDOM
MIN_LIST = 2  # a gene list is searched if at least this many of its genes are index genes
GRAPHCLUST = {
    "skin_melanoma": "xenium_hskin/output-XETG00000__slide_id__hskin/analysis.tar.gz",
    "kidney_nondiseased": "xenium_hkidney/output-XETG0000__slide_id__hkidney/analysis.tar.gz",
    "lung_cancer": "xenium_hlungcancer/output-XETG00000__slide_id__hlungcancer/analysis.tar.gz",
    "pancreatic_cancer": "xenium_hpancreas/output-XETG00000__slide_id__hpancreas/analysis.tar.gz",
    "lymph_node": "xenium_hlymphnode/output-XETG00000__slide_id__hlymphnode/analysis.tar.gz",
    "lymph_node_5k": "xenium_hlymphnode_5k/output-XETG00000__slide_id__hlymphnode5k/analysis.tar.gz",
    "brain_cancer": "xenium_hbraincancer/output-XETG00000__slide_id__hbraincancer/analysis.tar.gz",
}


def task(t):
    """(key, seed) of array task t = 5 * dataset + seed."""
    return KEYS[t // len(data.SEEDS)], data.SEEDS[t % len(data.SEEDS)]


def out_dir(key, seed):
    d = OUT / key / f"seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def graphclust(key, obs_names):
    """10x graph-based cluster of every cell (string, "" = no cluster)."""
    with tarfile.open(data.paths.data_dir("xenium") / GRAPHCLUST[key]) as tf:
        m = next(m for m in tf if m.name.endswith("gene_expression_graphclust/clusters.csv"))
        cl = pd.read_csv(io.BytesIO(tf.extractfile(m).read()))
    lab = cl.set_index(cl.Barcode.astype(str))["Cluster"].reindex(obs_names)
    print(f"{key}: {int(lab.isna().sum())} of {len(lab)} cells without a graph-based cluster", flush=True)
    return np.array(["" if pd.isna(v) else str(int(v)) for v in lab])


def load(key, seed, parts=("layout",)):
    """Section, build and per-cell inputs of one (key, seed), as common6.load."""
    adata, tiles, covs, genes = data.prepare(key)
    c = data.load_cache(key, seed, parts=parts, covs=covs)
    names = [str(g) for g in c.genes]
    gi = adata.var_names.get_indexer(names)
    if (gi < 0).any():
        raise RuntimeError("index genes missing from the section")
    X = adata.X[:, gi]
    X = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)
    tot = np.asarray(adata.X.sum(1)).ravel()
    L = np.log1p(X / np.maximum(tot, 1)[:, None] * np.median(tot)).astype(np.float32)
    if "Cluster" in adata.obs.columns:
        types = np.asarray(adata.obs["Cluster"].astype(str))
        to_class = {t: i for i, (_, _, members) in enumerate(c6.CLASSES) for t in members}
        comp = np.array([to_class.get(t, -1) for t in types])
        if (comp < 0).any():
            raise RuntimeError(f"unmapped cell types: {sorted(set(types[comp < 0]))}")
        comp_names = list(c6.CLASS_NAMES)
    else:
        types = graphclust(key, adata.obs_names)
        clusters = sorted({t for t in types if t}, key=int)
        pos = {t: i for i, t in enumerate(clusters)}
        comp = np.array([pos.get(t, -1) for t in types])
        comp_names = clusters
    return {"key": key, "adata": adata, "tiles": tiles, "covs": covs, "c": c, "genes": names, "X": X, "L": L,
            "types": types, "comp": comp, "comp_names": comp_names,
            "xy": np.asarray(adata.obsm["spatial"], dtype=np.float32), "stamp": data.stamp()}


def comp_counts(S, tile_ids):
    """Labelled cells of each composition label in each tile, (len(tile_ids), n_labels)."""
    n = len(S["comp_names"])
    out = []
    for t in tile_ids:
        v = S["comp"][S["tiles"][t].idx]
        out.append(np.bincount(v[v >= 0], minlength=n))
    return np.stack(out).astype(float)


def shares(counts):
    c = np.asarray(counts).sum(0) if np.ndim(counts) > 1 else np.asarray(counts)
    return c / max(c.sum(), 1)


SKIN_CLASSES = [("Melanocyte", "Melanocyte_Tumor"), ("Keratinocyte", "Keratinocyte_Epidermis"),
                ("Immune", "Immune_Infiltrate"), ("Other", None)]
CLASS_Z = 0.5


def skin_classes(S):
    """Skin cell classes for the worked example (rule fixed before the run): every graph-based cluster takes the skin
    signature (SKIN_CLASSES) with the largest mean standardised cell score over its cells (cell score = mean over S
    of log-normalised expression, standardised over all cells), if that mean exceeds CLASS_Z s.d., else "Other".
    Returns (class index per cell, -1 = no cluster; table of clusters)."""
    sig = {n: idx for n, idx, _, _ in gene_lists(S["key"], S["genes"])}
    Z = []
    for _, name in SKIN_CLASSES[:-1]:
        s = S["L"][:, sig[name]].mean(1)
        Z.append((s - s.mean()) / s.std())
    Z = np.stack(Z, 1)
    types = S["types"]
    cls = np.full(len(types), -1)
    rows = []
    for t in sorted({t for t in types if t}, key=int):
        m = types == t
        z = Z[m].mean(0)
        k = int(np.argmax(z)) if z.max() > CLASS_Z else len(SKIN_CLASSES) - 1
        cls[m] = k
        rows.append({"type": t, "class": SKIN_CLASSES[k][0], "n_cells": int(m.sum()),
                     **{f"z_{n}": float(v) for (n, _), v in zip(SKIN_CLASSES, z)}})
    return cls, pd.DataFrame(rows)


def gene_lists(key, genes):
    """[(name, idx, found, missing)] of the tissue's curated signatures with >= MIN_LIST index genes."""
    out = []
    for name, info in TISSUE_MODULES[TISSUE[key]].items():
        idx, found, missing = sd.gene_indices(genes, info["genes"])
        if len(idx) >= MIN_LIST:
            out.append((name, np.asarray(idx), list(found), list(missing)))
    return out
