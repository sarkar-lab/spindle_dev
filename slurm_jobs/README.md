# slurm_jobs/

Jobs that reproduce every result in `results/`. Submit from the repository root.
The subfolders mirror `benchmarks/`: `slurm_jobs/<topic>/run_<script>.sbatch` runs
`benchmarks/<topic>/<script>.py` (one dataset or seed per job, or `--aggregate`), and
`submit_<script>.sh` (where present) runs on the login node and only calls `sbatch` for every unit plus
the aggregate job. Job output goes to `logs/` (gitignored). `results/README.md` maps each output folder
to the figures and tables that use it.

| Folder | Jobs for |
|---|---|
| `index/` | index builds, Fig 2 (storage, build cost, cell and gene ladders) |
| `whole_tile/` | Fig 3 (DAG retrieval, exact vs whole-matrix timing, whole-covariance indexes, neighbour biology) |
| `partial/` | Fig 4 / S8 (partial gene-set search) |
| `cross_platform/` | Fig 5 / S9 (Xenium ↔ Visium) |
| `biology/` | Figs 6–7 inputs computed from the indexes (breast niches and composition, gene signatures) |
| `checks/` | one-off checks behind recorded decisions (metric, tiers, niche cap) |
| `figures/` | figure inputs (`scripts/figure_data/`) and the figures themselves (`scripts/figures/`) |
| `legacy/` | old E4 ANN baselines, kept until Fig 1 is redrawn |

Run order:

1. **Indexes**: `./slurm_jobs/index/submit_build_indexes.sh` builds the seed-73 production index of
   all 8 datasets → `results/indexes/`. The seed 0–4 indexes (10 % held out) come from
   `sbatch slurm_jobs/index/run_rebuild_capped_index.sbatch <stem> <seed>` (builds on the
   seed-suffixed symlink from `python benchmarks/paths.py <stem> --seed-symlink <seed>`).
2. **Experiments** (each needs the indexes; paths below are under `slurm_jobs/`):

   | Exp. | Submit | Unit job | Then |
   |---|---|---|---|
   | Fig 2A–B storage table (seed 73) | – | `index/run_dag_size_table.sbatch <dataset>` (`--mem=128G` for `brain_cancer`, `lymph_node_5k`) | `sbatch slurm_jobs/index/run_dag_size_table.sbatch --aggregate` |
   | Fig 2C end-to-end build time and memory, seeds 0–4 | `[DATASETS=...] index/submit_tier_build_stats.sh` | `index/run_tier_build_stats.sbatch <dataset> <seed>` | aggregate job submitted by the script |
   | Fig 2D cell ladder (first `python benchmarks/index/prepare_xenium_h5ad.py tonsil_reactive` on the login node: downloads ~120 MB) | `index/submit_dag_cell_ladder.sh` | `index/run_dag_cell_ladder.sbatch <dataset> <cells\|full> <seed>` | aggregate job submitted by the script |
   | Fig 2E gene ladder (needs up to ~300 GB of node-local disk under `$SPINDLE_LOCAL_CACHE`, default `/tmp/spindle_cache_$USER`) | `index/submit_dag_gene_ladder.sh` | `index/run_dag_gene_ladder.sbatch <G>` | aggregate job submitted by the script |
   | Rebuild one index in place (niche-cap fix; lymph node 5K at 400 genes) | – | `index/run_rebuild_capped_index.sbatch <stem> <seed> [backup dir]` | – |
   | Fig 3B–C DAG retrieval and K sweep, seeds 0–4 | `whole_tile/submit_dag_holdout.sh [datasets]` | `whole_tile/run_dag_holdout.sbatch <dataset> <seed> [--k-sweep] [--bound]` (S5: `breast_cancer 73 --k-sweep --bound`) | aggregate job submitted by the script |
   | Fig 3A query time (one timing job at a time) | `whole_tile/submit_exact_vs_whole.sh [datasets]` | `whole_tile/run_exact_vs_whole.sbatch <dataset> <seed>` | aggregate job submitted by the script |
   | Fig 3D indexes over whole covariances | `whole_tile/submit_whole_cov_baselines.sh [datasets]` | `whole_tile/run_whole_cov_baselines.sbatch <dataset> <seed>` | aggregate job submitted by the script |
   | Fig 3E neighbour biology (after Fig 3D) | `whole_tile/submit_neighbour_biology.sh [datasets]` | `whole_tile/run_neighbour_biology.sbatch <dataset> <seed>` | aggregate job submitted by the script |
   | Fig 4 / S8 partial search: interval index, padding, imputation vs Spindle-Exact on S, seeds 0–4 (timing one job at a time) | `partial/submit_partial_search_final.sh [accuracy\|timing\|all] [datasets]` | `partial/run_partial_search_final.sbatch <dataset> <seed> [--timing]`; Fig 4A example: `partial/run_partial_search_final.sbatch breast_cancer 73 --schematic` | aggregate job submitted by the script |
   | Fig 5 / S9 cross-platform search (tile caches, seeds 0–4 × tiles 2000/1000, aggregate, bias PCA, example) | `cross_platform/submit_cross_platform_tiers.sh` | `cross_platform/run_cross_platform_tiers.sbatch <seed> <max_pts>` (or `--overlay`, `--check`, `--bias-pca`, `--example`) | aggregate, bias PCA and example jobs submitted by the script |
   | Figs 6–7 niches vs cell types, all datasets | – | `biology/run_niche_concordance.sbatch [--datasets ...] [--seeds ...]` (more memory for `lymph_node_5k`, `brain_cancer`: `--mem=128G`) | – |
   | Figs 6–7, S10 block programs (seeds 73 + 0–4) | – | `biology/run_block_programs.sbatch --datasets <key> --seeds 73 0 1 2 3 4` (one job per dataset; `--mem` 48G, 64G pancreas / lymph node, 96G LN5k, 160G brain) | – |
   | Figs 6–7, S11 signature queries (seeds 73 + 0–4) | – | `biology/run_signature_queries.sbatch --datasets <key> --interval --maps` (one job per dataset; up to 200G for brain) | – |
   | Fig 6E worked whole-tile query | – | `biology/run_query_example.sbatch` | – |
   | Metric check: raw vs shrunk covariances (Part 1.1, seed 73) | – | `checks/run_metric_check.sbatch <dataset>` | `sbatch slurm_jobs/checks/run_metric_check.sbatch --aggregate` |
   | Sanity check of `src/spindle_dev/tiers.py` | – | `checks/run_check_tiers.sbatch [dataset]` | – |
   | Niche-cap check of every saved index (Part 1.3) | – | `checks/run_index_cap_check.sbatch` | – |
   | E4 ANN baselines (old; superseded by Fig 3D, results kept for `extract_fig1_data.py`) | `legacy/submit_ann_baselines.sh` | `legacy/run_ann_baselines.sbatch` | – |

3. **Figure inputs**: `figures/run_collect_index_stats.sbatch`, `figures/run_extract_figure_data.sbatch`,
   `figures/run_extract_fig1_data.sbatch`.
4. **Figures**: `bash slurm_jobs/figures/run_make_figures.sbatch [script ...]` draws every figure twice, into `figures/pdf/` and
   `figures/png/` (one file per figure, plus one per panel in `panels/<figure>/`). Copy the ones you need into the
   Overleaf project by hand.
