"""One worked whole-tile query on breast (Fig 6E; S11 C-D).

Rule: among the seed-73 held-out tiles (100 queries) with >= MIN_CELLS cells, the one with the largest
myoepithelial fraction (Myoepi_ACTA2+ + Myoepi_KRT15+, obs['Cluster']); a duct rim is a structure that a
mean-expression profile mixes with tumour and stroma. The query's full covariance searches Spindle-Exact
(top 10, d_B) and Spindle-DAG (K = 32, an unordered set of c = 10 tiles).

For every held-out query (base rate for the example): the query's myoepithelial fraction and that of its
exact top 10, and the composition JSD (base 2) of the query vs its exact top 10, DAG set and 10 random tiles.

Outputs (results/query_example/):
  breast_example.csv      the query and its returned tiles: method, rank, row, tile id, bbox, d_B (exact),
                          exact rank, n_cells, myoepithelial fraction, composition JSD vs the query
  breast_example_comp.csv cell-type composition of the query, the exact top 10, the DAG set and the section
  breast_all_queries.csv  per held-out query: myoepithelial fraction, JSD exact / DAG / random

  sbatch slurm_jobs/biology/run_query_example.sbatch
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import experiment_common as ec
from spindle_dev import tiers

STEM = ec.DATASETS["breast_cancer"]
SEED = ec.PRODUCTION_SEED
OUT_DIR = ec.PROJECT_ROOT / "results" / "query_example"
MYOEPI = ["Myoepi_ACTA2+", "Myoepi_KRT15+"]
MIN_CELLS = 50
C = 10


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ec.load_adata(STEM)
    cat = adata.obs["Cluster"].astype("category").cat.remove_unused_categories()
    types, codes = list(cat.cat.categories), cat.cat.codes.to_numpy()
    myo = np.isin(np.asarray(types)[codes], MYOEPI)
    tiles_all = ec.rebuild_tiles(adata)
    data = ec.load_index(STEM, SEED)["data"]
    raw = ec.load_raw_covs(STEM, SEED)
    train_idx, test_idx = np.asarray(raw["train_idx"]), np.asarray(raw["test_idx"])
    ec.check_tiles_match(tiles_all, data, train_idx)
    exact = tiers.build_tier("exact", data, raw["train_tile_covs"])
    dag = tiers.build_tier("dag", data, exact=exact)

    def comp(tile_ids):
        idx = np.concatenate([tiles_all[t].idx for t in tile_ids])
        return np.bincount(codes[idx], minlength=len(types)) / idx.size

    def jsd(a, b):
        return float(jensenshannon(a, b, base=2) ** 2)

    rng = np.random.default_rng(SEED)
    rows, results = [], {}
    for qi, t in enumerate(test_idx):
        q = raw["test_tile_covs"][qi]
        cq = comp([t])
        ex_rows, _, _ = exact.search(q, c=C)
        dag_rows, _, _ = dag.search(q, c=C)
        rnd = rng.choice(len(train_idx), C, replace=False)
        rows.append({"query": qi, "tile_id": int(t), "n_cells": len(tiles_all[t].idx),
                     "myoepi_frac": myo[tiles_all[t].idx].mean(),
                     "myoepi_frac_exact10": myo[np.concatenate([tiles_all[train_idx[r]].idx for r in ex_rows])].mean(),
                     "jsd_exact": jsd(cq, comp(train_idx[ex_rows])), "jsd_dag": jsd(cq, comp(train_idx[dag_rows])),
                     "jsd_random": jsd(cq, comp(train_idx[rnd]))})
        results[qi] = (q, ex_rows, dag_rows)
    allq = pd.DataFrame(rows)
    allq.to_csv(OUT_DIR / "breast_all_queries.csv", index=False)

    elig = allq[allq.n_cells >= MIN_CELLS]
    pick = int(elig.loc[elig.myoepi_frac.idxmax(), "query"])
    q, ex_rows, dag_rows = results[pick]
    t = int(test_idx[pick])
    d = exact.distances(exact.query_vectors(tiers.prepare_cov(q)))
    order = np.argsort(d, kind="stable")
    rank = np.empty(len(order), int)
    rank[order] = np.arange(1, len(order) + 1)
    cq = comp([t])

    def row(method, r, k, tile_id):
        tl = tiles_all[tile_id]
        x0, y0, x1, y1 = tl.bbox
        return {"method": method, "rank": k, "row": r, "tile_id": int(tile_id), "x0": x0, "y0": y0, "x1": x1,
                "y1": y1, "d_B": np.nan if r < 0 else d[r], "exact_rank": -1 if r < 0 else int(rank[r]),
                "n_cells": len(tl.idx), "myoepi_frac": myo[tl.idx].mean(), "jsd_vs_query": jsd(cq, comp([tile_id]))}

    ex = [row("query", -1, 0, t)]
    ex += [row("exact", int(r), k, train_idx[r]) for k, r in enumerate(ex_rows, start=1)]
    ex += [row("dag", int(r), k, train_idx[r]) for k, r in enumerate(dag_rows, start=1)]
    pd.DataFrame(ex).to_csv(OUT_DIR / "breast_example.csv", index=False)
    comps = {"query": cq, "exact_top10": comp(train_idx[ex_rows]), "dag_set": comp(train_idx[dag_rows]),
             "section": np.bincount(codes, minlength=len(types)) / codes.size}
    pd.DataFrame(comps, index=types).rename_axis("cell_type").to_csv(OUT_DIR / "breast_example_comp.csv")

    e = allq.loc[allq["query"] == pick].iloc[0]
    print(f"example: query {pick} (tile {t}, {int(e.n_cells)} cells), myoepithelial {e.myoepi_frac:.2f} "
          f"(held-out median {allq.myoepi_frac.median():.2f}, section {myo.mean():.2f}); exact top 10 "
          f"{e.myoepi_frac_exact10:.2f}; DAG set shares {len(set(ex_rows) & set(dag_rows))} of 10 with exact; "
          f"JSD exact {e.jsd_exact:.3f} DAG {e.jsd_dag:.3f} random {e.jsd_random:.3f}")
    print("all queries, mean JSD: " + ", ".join(f"{m} {allq[f'jsd_{m}'].mean():.3f}" for m in ("exact", "dag", "random")))
    print(f"myoepithelial frac of exact top 10 vs query, Spearman over queries: "
          f"{allq[['myoepi_frac', 'myoepi_frac_exact10']].corr('spearman').iloc[0, 1]:.2f}")


if __name__ == "__main__":
    main()
