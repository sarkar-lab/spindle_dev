"""Build Spindle and Spindle-lite (whole tile and partial) on one dataset and query each with a held-out tile.

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

    spindle = sd.Spindle(data, train_covs)
    lite = sd.SpindleLite.from_spindle(spindle, data)
    partial = sd.SpindlePartial(data, train_covs)
    lite_partial = sd.SpindleLitePartial.from_partial(partial, data)

    query = covs[test[0]]
    rows, scores, sec = spindle.search(query, c=10)
    print(f"Spindle top 10 ({sec * 1e3:.1f} ms):", rows.tolist())
    found, _, sec = lite.search(query)
    c = min(10, len(found))
    d = spindle.distances(spindle.query_vectors(query))
    exact = np.sort(d)[:c].mean()
    print(f"Spindle-lite returned {len(found)} tiles ({sec * 1e3:.1f} ms with the query's logs); its top {c} is "
          f"{100 * (d[found[:c]].mean() - exact) / exact:.1f} % farther than the exact top {c}")

    k = int(data.labels[0])
    s, e = data.block_dict[k][0]
    gene_set = np.asarray(data.perm_list[k])[s:min(e, s + 12)]
    d_s = partial.distances(query_cov_s(query, gene_set, partial.p), gene_set)
    found, _, sec = lite_partial.search(query, gene_set)
    c = min(10, len(found))
    exact = np.sort(d_s)[:c].mean()
    print(f"12-gene set: Spindle-lite partial returned {len(found)} tiles ({sec * 1e3:.1f} ms with the query's logs); "
          f"its top {c} is {100 * (d_s[found[:c]].mean() - exact) / exact:.1f} % farther than the exact top {c}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        data.spd_matrices, data.U_list = [], None
        for name, obj in [("layout", data), ("spindle", spindle), ("lite", lite), ("partial", partial),
                          ("lite_partial", lite_partial)]:
            sd.save_index(obj, args.out / f"{path.stem}_{name}.pkl")
        print("saved to", args.out)


if __name__ == "__main__":
    main()
