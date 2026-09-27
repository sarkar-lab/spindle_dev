# results/

Every number and figure in the paper comes from the CSVs in this folder. The figure scripts
(`scripts/fig*.py`) only read these CSVs; the benchmarks in `benchmarks/` write them.
Each folder has the same name as the script that writes it. New runs go to new folders;
never append to or overwrite a published one.

Datasets: 8 Xenium sections (`xenium_human_<tissue>`). Seed 73 is the single production
build (100 held-out tiles); `_seed0` … `_seed4` are the five builds (10 % held out) that give
the mean ± s.d. error bars.

| Folder | Exp. | Written by (`benchmarks/` unless noted) | Slurm job | Key files | Used in |
|---|---|---|---|---|---|
| `indexes/` (141 GB, not in git) | – | `build_indexes.py` | `submit_build_indexes.sh`, and inside `run_holdout_search` / `run_partial_panel_search` | `<ds>[_seedN]_spindle_index.pkl` (index), `_raw_covariances.pkl` (train/test covariances + split), `_interval_index.pkl` (dyadic index, partial panel) | input to every experiment |
| `ground_truth_cache/` (38 GB, not in git) | – | `holdout_core.load_or_compute_ground_truth` | – | `<ds>_ground_truth_{block,whole}.pkl`: exact rankings, validated against the split | cache; recomputed if missing (slow) |
| `index_stats/` | – | `scripts/collect_index_stats.py` | `run_collect_index_stats` | `index_stats.csv` (per dataset × seed), `index_stats_summary.csv`; inputs `build_run_logs/*.json` (wall time, peak RSS per build) and `log_index_stats.csv` (values from deleted build logs) | Fig 2, S1, S2, Table S1 |
| `scalability_sweep/` | E6 | `scalability_sweep.py` | `submit_scalability_sweep.sh` | `<ds>_synthetic_scaling.csv`, `<ds>_scaling_exponents.json` | Fig 2D |
| `holdout_search/` | E5 | `holdout_search.py` (→ `holdout_core.py`) | `submit_holdout_search.sh` | `summary.csv`; `<ds>/seed_N/query_metrics.csv` | Fig 3A–B, S4, Table S2 |
| `budget_sweep/` | E1 | `budget_sweep.py` | `submit_budget_sweep.sh` | `sweep_summary_{block,whole}.csv`; per-dataset CSVs | Fig 3D, S5, S7 |
| `ann_baselines/` | E4 | `ann_baselines.py` | `submit_ann_baselines.sh` | `summary.csv`, `hnsw_sensitivity.csv` | Fig 3E, S8 |
| `noise_robustness/` | E7 | `noise_robustness.py` | `submit_noise_robustness.sh` | `summary.csv`; `<ds>_<model>_<level>_noise_sweep.csv` | Fig 3C, S6 |
| `partial_panel_search/` | E11 | `partial_panel_search.py` (→ `partial_panel_core.py`) | `submit_partial_panel_search.sh` | `summary.csv`; `<ds>/seed_N/query_metrics.csv` | Fig 4, S9, Table S3 |
| `cross_modal_search/` | E12 | `cross_modal_search.py`, `aggregate_cross_modal_seeds.py`, `cross_modal_bias_pca.py` | `submit_cross_modal_search.sh`, `run_cross_modal_bias_pca` | `summary.csv`, `summary_single_niche_baseline.csv`, `bias_pca*.csv`, `tile_overlay_boxes.csv`, `seed_N/`, `correction_ablation_seed0/` | Fig 5, S10 |
| `metric_concordance/` | E14 | `metric_concordance.py` | `run_metric_concordance` | `summary.csv`; `<ds>_per_query.csv` | Fig S12 |
| `composition_concordance/` | E13 | `composition_concordance.py` | `run_breast_concordance` | `summary.csv`, `breast_jsd.csv` | Fig 6H |
| `niche_concordance/` | E10-lite | `niche_concordance.py` | `run_breast_concordance` | `breast_ari_nmi.csv`, `breast_niche_composition.csv`, `breast_tile_labels.csv` | Fig 6A |
| `gene_signature_search/` | E9-lite | `gene_signature_search.py` (signatures in `gene_signatures.py`) | `run_gene_signature_search` | `breast/signature_stats.csv`, `*_top_matches.csv`, `*_spatial_cells.csv` | Fig 6F–G, Table S4 |
| `biology/` | – | `scripts/bio_modules.py` (configs in `scripts/bio_configs/`) | `run_bio_modules` | `<ds>/{selection,modules,corr,enrichment,node_scores,tile_scores}.csv`; `genesets/*.gmt` are inputs | Fig 6B–E, 7, S3, S11 |
| `figure_data/` | – | `scripts/extract_figure_data.py`, `extract_fig1_data.py`, `extract_cross_modal_example.py` | `run_extract_figure_data`, `run_extract_fig1_data` | `tiles_seed73.csv`, `cells_sample.csv`, `block_sizes.csv`, `niche_epsilons.csv`, `within_niche_distances.csv`, `cross_modal_*.csv`, `fig1/` | Fig 1, 5, 6, 7, S1–S3, S5 |

Metric columns: `recall_at_eps_<f>` = 1 if Spindle's top hit is within f·ε of the exact
nearest distance; `overlap_at_eps_<f>` = fraction of tiles within f·ε that Spindle returns.
Speedup is always mean brute-force time / mean Spindle time (whole-matrix brute force).
