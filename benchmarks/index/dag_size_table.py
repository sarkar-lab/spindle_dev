"""Per-dataset storage table: whole covariances vs the exact tier's block logs vs the DAG.

For each of the 8 production indexes (seed 73; niches capped at 1000 tiles), the two tiers are
built with spindle_dev.tiers on the index's niche/block layout: the exact tier from the shrunk
covariances (tiers.prepare_cov) and the DAG from its block logs (tiers.DAG_DEFAULTS: K = 32 nodes
per block, node means). Sizes, all float32:

* whole_cov_full_mb   N p^2 x 4 B         (every tile's full covariance matrix)
* whole_cov_upper_mb  N p(p+1)/2 x 4 B    (its upper triangle; the symmetric minimum)
* block_logs_mb       the exact tier: upper triangles of every tile's block logs
* dag_mb              the deployed DAG: node means, radii, codes, edges

Outputs (results/dag_size_table/): <stem>.csv, summary.csv (--aggregate)

  sbatch slurm_jobs/index/run_dag_size_table.sbatch <dataset>
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # benchmarks/
import paths  # noqa: E402  (puts src/ and every benchmarks/ folder on sys.path)
import dag_eval_common as de
import experiment_common as ec
from spindle_dev import tiers

OUT_DIR = ec.PROJECT_ROOT / "results" / "dag_size_table"


def run(stem):
    data = ec.load_index(stem, ec.PRODUCTION_SEED)["data"]
    covs = ec.load_raw_covs(stem, ec.PRODUCTION_SEED)
    exact = tiers.build_tier("exact", data, covs["train_tile_covs"])
    del covs
    dag = tiers.build_tier("dag", data, exact=exact)
    n, p = len(data.labels), len(data.metadata["genes"])
    rec = {"dataset": stem, "n_tiles": n, "n_genes": p,
           "whole_cov_full_mb": n * p * p * 4 / de.MB, "whole_cov_upper_mb": n * p * (p + 1) / 2 * 4 / de.MB,
           **de.dag_info(dag.compact, data), "exact_mb": exact.nbytes() / de.MB, "dag_build_s": dag.build_seconds}
    if abs(rec["exact_mb"] - rec["block_logs_mb"]) > 1e-9:
        raise RuntimeError("exact tier size differs from block_log_bytes")
    rec["full_over_dag"] = rec["whole_cov_full_mb"] / rec["dag_mb"]
    rec["upper_over_dag"] = rec["whole_cov_upper_mb"] / rec["dag_mb"]
    rec["upper_over_block_logs"] = rec["whole_cov_upper_mb"] / rec["block_logs_mb"]
    return pd.DataFrame([rec])


def aggregate():
    s = pd.concat([pd.read_csv(f) for f in sorted(OUT_DIR.glob("xenium_*.csv"))], ignore_index=True)
    order = {v: i for i, v in enumerate(ec.DATASETS.values())}
    s = s.sort_values("dataset", key=lambda c: c.map(order))
    s.to_csv(OUT_DIR / "summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(s.round(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate:
        aggregate()
        return
    stem = ec.resolve_stem(args.dataset)
    df = run(stem)
    df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    print(df.round(2).T.to_string())


if __name__ == "__main__":
    main()
