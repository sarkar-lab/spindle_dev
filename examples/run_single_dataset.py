"""Build Spindle-naive and Spindle (whole tile and gene set) on one dataset, query each with a held-out tile, then
query the gene-set indexes with a gene list given by names (GENE_LIST, the myeloid program).

Usage: python examples/run_single_dataset.py <dataset key | path.h5ad> [--out DIR]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]
import paths  # noqa: E402
import spindle_dev as sd  # noqa: E402
from spindle_dev.metrics import query_cov_s  # noqa: E402

GENE_LIST = ["CD68", "CD163", "CD14", "AIF1", "MNDA"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="key in datasets.yaml, or an .h5ad path")
    parser.add_argument("--out", type=Path, default=None, help="folder for the pickled layout and indexes")
    args = parser.parse_args()

    path = Path(args.dataset) if args.dataset.endswith(".h5ad") else paths.dataset_path(args.dataset)
    max_genes = paths.dataset_max_genes(args.dataset) if not args.dataset.endswith(".h5ad") else 800

    adata = sd.load_adata(path)
    tiles, covs, genes = sd.prepare_tiles(adata, max_genes=max_genes)
    train, test = sd.split_tiles(len(tiles), seed=73, n_holdout=100)
    train_covs = [covs[i] for i in train]
    data = sd.build_layout([tiles[i] for i in train], train_covs, genes, adata.n_obs)
    print(f"{len(train)} training tiles, {len(set(data.labels.tolist()))} niches")

    naive = sd.SpindleNaive(data, train_covs)
    spindle = sd.Spindle.from_naive(naive, data)
    naive_partial = sd.SpindleNaivePartial(data, train_covs)
    spindle_partial = sd.SpindlePartial.from_naive_partial(naive_partial, data)

    query = covs[test[0]]
    rows, scores, sec = naive.search(query, c=10)
    print(f"Spindle-naive top 10 ({sec * 1e3:.1f} ms):", rows.tolist())
    found, _, sec = spindle.search(query)
    c = min(10, len(found))
    d = naive.distances(naive.query_vectors(query))
    exact = np.sort(d)[:c].mean()
    print(f"Spindle returned {len(found)} tiles ({sec * 1e3:.1f} ms with the query's logs); its top {c} is "
          f"{100 * (d[found[:c]].mean() - exact) / exact:.1f} % farther than the exact top {c}")

    k = int(data.labels[0])
    s, e = data.block_dict[k][0]
    gene_set = np.asarray(data.perm_list[k])[s:min(e, s + 12)]
    d_s = naive_partial.distances(query_cov_s(query, gene_set, naive_partial.p), gene_set)
    found, _, sec = spindle_partial.search(query, gene_set)
    c = min(10, len(found))
    exact = np.sort(d_s)[:c].mean()
    print(f"12-gene set: Spindle (gene sets) returned {len(found)} tiles ({sec * 1e3:.1f} ms with the query's logs); "
          f"its top {c} is {100 * (d_s[found[:c]].mean() - exact) / exact:.1f} % farther than the exact top {c}")

    # a gene list by names (Methods Eq. signature); omega is kept beside the index
    omega = sd.variance_quantiles(train_covs)
    idx, found_genes, missing = sd.gene_indices(genes, GENE_LIST)
    if len(idx) >= 2:
        cov_s, idx = sd.template_query(omega, idx)
        rows, scores, sec = naive_partial.search(cov_s, idx, c=10)
        print(f"gene list {found_genes} (not in the index: {missing}): Spindle-naive (gene sets) top 10 "
              f"({sec * 1e3:.1f} ms):", rows.tolist())
        found, _, sec = spindle_partial.search(cov_s, idx)
        print(f"  Spindle (gene sets) returned {len(found)} tiles ({sec * 1e3:.1f} ms), "
              f"{len(set(found[:10].tolist()) & set(rows.tolist()))} of the exact top 10 in its top 10")
    else:
        print(f"gene list: {len(idx)} of {GENE_LIST} in the index genes, skipped")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        data.spd_matrices, data.U_list = [], None
        for name, obj in [("layout", data), ("naive", naive), ("spindle", spindle), ("naive_partial", naive_partial),
                          ("spindle_partial", spindle_partial)]:
            sd.save_index(obj, args.out / f"{path.stem}_{name}.pkl")
        print("saved to", args.out)


if __name__ == "__main__":
    main()
