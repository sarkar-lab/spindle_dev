# slurm_jobs/

Jobs that reproduce every result in `results/`. Submit from the repository root.
`submit_*.sh` scripts run on the login node and only call `sbatch` (one job per dataset or
seed). Job output goes to `logs/` (gitignored). `results/README.md` maps each output folder
to the figures and tables that use it.

Run order:

1. **Indexes**: `./slurm_jobs/submit_build_indexes.sh` builds the seed-73 production index of
   all 8 datasets → `results/indexes/`. The seed 0–4 indexes (10 % held out) come from `run_rebuild_capped_index.sbatch <stem> <seed>` (or `holdout_search.py` unit mode, which also runs the old E5 search).
2. **Experiments** (each needs the indexes):

   | Exp. | Submit | Unit job | Then |
   |---|---|---|---|
   | E11 partial-panel search, seeds 0–4 | `submit_partial_panel_search.sh` | `run_partial_panel_search.sbatch` | `python benchmarks/partial_panel_search.py --aggregate` |
   | E4 ANN baselines (old; superseded by Fig 3D, results kept for `extract_fig1_data.py`) | `submit_ann_baselines.sh` | `run_ann_baselines.sbatch` | – |
   | E12 cross-platform search | `submit_cross_modal_search.sh` | `run_cross_modal_search.sbatch` | `python benchmarks/aggregate_cross_modal_seeds.py`, then `sbatch run_cross_modal_bias_pca.sbatch` |
   | E10-lite / E13 (breast) | – | `run_breast_concordance.sbatch` | – |
   | E9-lite gene signatures | – | `run_gene_signature_search.sbatch` | – |
   | Niche-cap check of every saved index (Part 1.3) | – | `run_index_cap_check.sbatch` | – |
   | Rebuild one index in place (niche-cap fix; lymph node 5K at 400 genes) | – | `run_rebuild_capped_index.sbatch <stem> <seed> [backup dir]` | – |
   | Metric check: raw vs shrunk covariances (Part 1.1, seed 73) | – | `run_metric_check.sbatch <dataset>` | `sbatch run_metric_check.sbatch --aggregate` |
   | Sanity check of `src/spindle_dev/tiers.py` | – | `run_check_tiers.sbatch [dataset]` | – |
   | Fig 2A–B storage table (seed 73) | – | `run_dag_size_table.sbatch <dataset>` (`--mem=128G` for `brain_cancer`, `lymph_node_5k`) | `sbatch run_dag_size_table.sbatch --aggregate` |
   | Fig 2C end-to-end build time and memory, seeds 0–4 | `[DATASETS=...] submit_tier_build_stats.sh` | `run_tier_build_stats.sbatch <dataset> <seed>` | aggregate job submitted by the script |
   | Fig 2D cell ladder (first `python benchmarks/prepare_xenium_h5ad.py tonsil_reactive` on the login node: downloads ~120 MB) | `submit_dag_cell_ladder.sh` | `run_dag_cell_ladder.sbatch <dataset> <cells\|full> <seed>` | aggregate job submitted by the script |
   | Fig 2E gene ladder (needs up to ~300 GB of node-local disk under `$SPINDLE_LOCAL_CACHE`, default `/tmp/spindle_cache_$USER`) | `submit_dag_gene_ladder.sh` | `run_dag_gene_ladder.sbatch <G>` | aggregate job submitted by the script |
   | Fig 3B–C DAG retrieval and K sweep, seeds 0–4 | `submit_dag_holdout.sh [datasets]` | `run_dag_holdout.sbatch <dataset> <seed> [--k-sweep] [--bound]` (S5: `breast_cancer 73 --k-sweep --bound`) | aggregate job submitted by the script |
   | Fig 3A query time (one timing job at a time) | `submit_exact_vs_whole.sh [datasets]` | `run_exact_vs_whole.sbatch <dataset> <seed>` | aggregate job submitted by the script |
   | Fig 3D indexes over whole covariances | `submit_whole_cov_baselines.sh [datasets]` | `run_whole_cov_baselines.sbatch <dataset> <seed>` | aggregate job submitted by the script |
   | Fig 3E neighbour biology (after Fig 3D) | `submit_neighbour_biology.sh [datasets]` | `run_neighbour_biology.sbatch <dataset> <seed>` | aggregate job submitted by the script |
   | Partial queries: interval index vs exact tier (seed 73; replaces E11) | – | `run_interval_index_final.sbatch <dataset>` (`--mem=128G` for `brain_cancer`, `lymph_node_5k`) | – |

3. **Figure inputs**: `run_collect_index_stats.sbatch`, `run_extract_figure_data.sbatch`,
   `run_extract_fig1_data.sbatch`, `run_bio_modules.sbatch`.
4. **Figures**: `run_make_figures.sbatch` draws every figure twice, into `figures/pdf/` and
   `figures/png/` (one file per figure, plus one per panel in `panels/<figure>/`). Copy the ones you need into the
   Overleaf project by hand.
