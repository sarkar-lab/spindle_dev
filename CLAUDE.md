# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Spindle** indexes and searches the covariance matrices of spatial-omics tiles. The paper has two indexes,
each for whole tiles and gene sets, all on one shared layout:

- **Common ground:** every tile covariance is shrunk once, when it is computed (`preprocessing._tile_cov`,
  `metrics.SHRINK_ALPHA` = 0.1); the layout and all indexes use only shrunk covariances and never shrink
  again. Every distance is d_B = √(Σ_b ‖log q_b − log t_b‖²_F)/√p (d_B^S with √|S| for gene sets), log floor 1e-6.
- **Names (method-v2, 31 Oct):** the paper's two names are Spindle (the DAG index) and Spindle-naive (the exhaustive
  scan, the accuracy upper bound). Up to method-v1 these were Spindle-lite and Spindle. Keys in code and results:
  `spindle`, `spindle_partial`, `naive`, `naive_partial` (old: lite, plite, spindle, partial).
- **Spindle-naive** (`src/spindle_dev/naive.py`): exact scan over stored block logs. `SpindleNaive` answers whole-tile
  queries with d_B; `SpindleNaivePartial` answers gene-set queries with d_B^S.
- **Spindle** (`spindle.py`, `Spindle`): the first draft's layered block DAG (`Spindle/Spindle_Old.pdf`)
  on d_B: ε-net nodes per block, edges between co-occurring nodes, DFS with an L2 path lower bound, in every
  niche. The final configuration (variance-weighted node counts, spread-ordered layers, radius g × the greedy
  probe's best score) is specified in `paper/SPINDLE_LITE.md`.
- **Compiled search** (`kernels.c`, `kernels.py`): `SpindleNaive.search` / `SpindleNaive.distances` and
  `Spindle.search` / `search_vectors` and `SpindlePartial.search` / `search_vectors` run C kernels (AVX2,
  through ctypes), bit-identical to the numpy / Python references (`SpindleNaive.distances_numpy`,
  `Spindle.search_vectors_dfs`, `SpindlePartial.search_vectors_dfs`). The library builds on first use into
  `src/spindle_dev/_build/`; compute nodes have no gcc, so after editing `kernels.c` build once on the login node
  (`python -c "import spindle_dev.kernels as k; print(k.build())"` with `src/` on the path).
- **Spindle partial** (`spindle_partial.py`, `SpindlePartial`): gene sets on d_B^S, the same design on dyadic
  gene intervals: aligned dyadic intervals per block, centre + radius nodes with counts by the relative-error gain
  at β = 0.10, one-byte node index, no stored edges; the query's pieces left to right, greedy probe, r² = g²·Â²
  (default g = 1). Final configuration and decision log: `paper/SPINDLE_PARTIAL.md`. Its compiled search is
  `kernels.c` `partial_search` (bit-identical to `SpindlePartial.search_vectors_dfs`).
- **Gene-list queries** (`queries.py`): `gene_indices` (names → indices, unknown/duplicate genes dropped and
  reported), `template_query` (Methods Eq. signature: ρ = 0.5, Ω = 90th-percentile shrunk variances from
  `variance_quantiles`, kept beside the index since Spindle partial stores no covariances), `tile_query`. All
  return `(cov_s, idx)` for `SpindleNaivePartial` / `SpindlePartial`. Chosen by a gene-set pilot (25–26 Oct;
  removed in the 27 Oct cleanup, in git history); the exemplar rule stays experiment code (`experiments/common/pan_cancer.py`).
- **Figures** (`experiments/`, `results/`, `figures/`), three tiers (regenerated under method-v2 on 1 Nov):
  1. `experiments/common/*.py`: shared functions. `data.py` (datasets, the index cache in `results/_cache/`, seeds
     0–4 with seed 0 for examples, the `method-v2` stamp, loaders of R0's staged inputs), `index_cache.py` (build +
     stage), `slurm.py`, `results_io.py` (CSV + `results/<topic>/manifest.json`), `style.py` (panels in mm), one
     module per experiment (`whole_tile`, `gene_set(_accuracy)`, `timing`, `cross_platform`, `programs`, `queries`,
     `pan_cancer`, `skin_maps`, `sections`, `retrieval`); `run.sbatch` / `timing.sbatch` (Slurm array workers), `run_nb.py` +
     `run_nb.sbatch` / `timing_nb.sbatch` (execute a notebook).
  2. `experiments/compute/R0`–`R10_*.ipynb`: result notebooks, writing CSVs + a manifest to `results/<topic>/`
     (`index_cost`, `whole_tile`, `gene_set`, `timing`, `cross_platform`, `breast_programs`, `queries`,
     `pan_cancer`, `skin_maps`, `block_structure`); summaries shown, not saved; existing outputs skipped unless `FORCE = True`. Only R0 (index cache
     builds + staging, ~128 GB for brain) uses Slurm; R1–R10 run without it in ≤ 16 GB (`run_nb.sbatch`, `--mem=16G`).
     R4 (timing) runs alone on ROSALIND-CPU1 (`sbatch --exclusive experiments/common/timing_nb.sbatch …`).
  3. `experiments/figures/F2_cost`, `F3_cross_platform`, `F4_skin` (main figures) and `S1_block`, `S2_breast`,
     `S3_scaling`, `S4_accuracy`, `S5_cross_platform` (supplementary results, figures S1–S5), `S6_blocks`, `S7_dyadic`
     (supplementary methods, S6–S7; from R10), `T_supplement` (tables); earlier versions in `_backups/` (gitignored, same relative paths);
     login node, spindle_env kernel; the first cell lists the R notebooks
     needed): one panel per cell → `figures/<fig>/panels/<X>.{pdf,png}`, the full figure → `figures/<fig>/<fig>.{pdf,png}`.
     The PDFs are copied by hand into `Spindle/figures/final/`, which the manuscript includes.
  No pptx, atoms or numbers files. Fig 1 (`figures/fig1_overview/`) is the user's own PDF: never write or delete it.
  `experiments/beyond_expression/` and `query_maps/` are self-contained side studies.
- **Layout** (`layout.py`): tiles → Leiden covariance niches (cap 1,000) → gene order and blocks per niche.
  `preprocessing.py` tiles the section, picks genes, computes covariances and splits train/held-out.

## Rules

- **The method is frozen** at git tag `method-v3` (= `method-v2` + Spindle's search split by niche for multi-core
  queries, new functions only: `kernels.c` `spindle_probe_niche` / `spindle_scan_niche`, `FlatSpindle.search_parallel`,
  `Spindle.search_vectors_parallel`; used by R11 only; every `method-v2` output unchanged, so results keep the
  `method-v2` tag; `method-v2` = `method-v1` renamed). Do not change `src/`: no variants, no retuning, no extra checks.
  A real bug goes to the user first. `data.stamp()` refuses to make a figure input when `src/` differs from `method-v3`.
- Dataset paths live only in `datasets.yaml`; read them with `paths.dataset_path(key)` (shell:
  `python paths.py <key>`). Never hard-code a path.
- Experiment code goes in `experiments/` (the three tiers above) and uses only the public API of `spindle_dev`;
  baselines never go into `src/`.
- **Benchmarks** time the compiled searches only (`SpindleNaive.distances` + sort, `Spindle.search_vectors`,
  `SpindlePartial.search_vectors`; gene sets exact: `SpindleNaivePartial.distances` + sort), from query vectors / pieces
  computed beforehand (`query_vectors`), one thread
  (`OMP_NUM_THREADS=1` etc.), one timing job at a time on an otherwise idle node (R4). Never time the numpy / Python
  references.
- Heavy compute goes through `sbatch` (the shell is a login node), in the `spindle_env` conda env
  (`~/miniforge3/envs/spindle_env`). `spindle_dev` is not installed there: put `src/` on `sys.path`.
  Compute nodes cannot see the login node's `/tmp`, so job scripts and outputs must live on the shared
  file system.
- The manuscript is a separate Overleaf project (local copy `Spindle/`, gitignored); no code writes into it. The
  Results section is written by hand in `Spindle/main_submission.tex`, matching the user's own Results and Methods.

## Run

```bash
python examples/run_single_dataset.py skin_melanoma
python experiments/common/run_nb.py experiments/compute/R0_index_cache.ipynb     # login node: submits the builds
sbatch experiments/common/run_nb.sbatch experiments/compute/R2_whole_tile.ipynb   # any of R1-R3, R5-R9
python experiments/common/run_nb.py experiments/figures/F2_cost.ipynb           # login node
```

There is no test suite; `examples/check_indexes.py <key>` (via sbatch) is the sanity check: Spindle-naive
self-queries (whole tile and gene set), compiled == reference for all three compiled searches, and the lower bound
and safety of both Spindle indexes (whole tile and partial).
