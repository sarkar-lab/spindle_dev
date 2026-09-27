# slurm_jobs/

Jobs that reproduce every result in `results/`. Submit from the repository root.
`submit_*.sh` scripts run on the login node and only call `sbatch` (one job per dataset or
seed). Job output goes to `logs/` (gitignored). `results/README.md` maps each output folder
to the figures and tables that use it.

Run order:

1. **Indexes**: `./slurm_jobs/submit_build_indexes.sh` builds the seed-73 production index of
   all 8 datasets → `results/indexes/`. The seed 0–4 indexes are built inside step 2.
2. **Experiments** (each needs the indexes):

   | Exp. | Submit | Unit job | Then |
   |---|---|---|---|
   | E5 whole-tile search, seeds 0–4 | `submit_holdout_search.sh` | `run_holdout_search.sbatch` | `python benchmarks/holdout_search.py --aggregate` |
   | E11 partial-panel search, seeds 0–4 | `submit_partial_panel_search.sh` | `run_partial_panel_search.sbatch` | `python benchmarks/partial_panel_search.py --aggregate` |
   | E1 budget sweep (seed 73) | `submit_budget_sweep.sh` | `run_budget_sweep.sbatch` | – |
   | E4 ANN baselines | `submit_ann_baselines.sh` | `run_ann_baselines.sbatch` | – |
   | E7 noise robustness | `submit_noise_robustness.sh` | `run_noise_robustness.sbatch` | – |
   | E6 scalability sweep | `submit_scalability_sweep.sh` | `run_scalability_sweep.sbatch` | – |
   | E12 cross-platform search | `submit_cross_modal_search.sh` | `run_cross_modal_search.sbatch` | `python benchmarks/aggregate_cross_modal_seeds.py`, then `sbatch run_cross_modal_bias_pca.sbatch` |
   | E14 metric concordance | – | `run_metric_concordance.sbatch <dataset>` | – |
   | E10-lite / E13 (breast) | – | `run_breast_concordance.sbatch` | – |
   | E9-lite gene signatures | – | `run_gene_signature_search.sbatch` | – |

3. **Figure inputs**: `run_collect_index_stats.sbatch`, `run_extract_figure_data.sbatch`,
   `run_extract_fig1_data.sbatch`, `run_bio_modules.sbatch`.
4. **Figures**: `run_make_figures.sbatch` draws every figure twice, into `figures/pdf/` and
   `figures/png/` (one file per figure, plus one per panel in `panels/<figure>/`). Copy the ones you need into the
   Overleaf project by hand.
