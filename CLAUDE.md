# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Spindle** indexes and searches the covariance matrices of spatial-omics tiles. The paper has two indexes,
each for whole tiles and gene sets, all on one shared layout:

- **Common ground:** every tile covariance is shrunk once, when it is computed (`preprocessing._tile_cov`,
  `metrics.SHRINK_ALPHA` = 0.1); the layout and all indexes use only shrunk covariances and never shrink
  again. Every distance is d_B = √(Σ_b ‖log q_b − log t_b‖²_F)/√p (d_B^S with √|S| for gene sets), log floor 1e-6.
- **Spindle** (`src/spindle_dev/spindle.py`): exact scan over stored block logs. `Spindle` answers whole-tile
  queries with d_B; `SpindlePartial` answers gene-set queries with d_B^S.
- **Spindle-lite** (`lite.py`, `SpindleLite`): the first draft's layered block DAG (`Spindle/Spindle_Old.pdf`)
  on d_B: ε-net nodes per block, edges between co-occurring nodes, DFS with an L2 path lower bound, in every
  niche. The final configuration (variance-weighted node counts, spread-ordered layers, radius g × the greedy
  probe's best score) is specified in `paper/SPINDLE_LITE.md`.
- **Compiled search** (`kernels.c`, `kernels.py`): `Spindle.search` / `Spindle.distances` and
  `SpindleLite.search` / `search_vectors` and `SpindleLitePartial.search` / `search_vectors` run C kernels (AVX2,
  through ctypes), bit-identical to the numpy / Python references (`Spindle.distances_numpy`,
  `SpindleLite.search_vectors_dfs`, `SpindleLitePartial.search_vectors_dfs`). The library builds on first use into
  `src/spindle_dev/_build/`; compute nodes have no gcc, so after editing `kernels.c` build once on the login node
  (`python -c "import spindle_dev.kernels as k; print(k.build())"` with `src/` on the path).
- **Spindle-lite partial** (`lite_partial.py`, `SpindleLitePartial`): gene sets on d_B^S, the same design on dyadic
  gene intervals: aligned dyadic intervals per block, centre + radius nodes with counts by the relative-error gain
  at β = 0.10, one-byte node index, no stored edges; the query's pieces left to right, greedy probe, r² = g²·Â²
  (default g = 1). Final configuration and decision log: `paper/SPINDLE_PARTIAL.md`. Its compiled search is
  `kernels.c` `partial_search` (bit-identical to `SpindleLitePartial.search_vectors_dfs`).
- **Gene-list queries** (`queries.py`): `gene_indices` (names → indices, unknown/duplicate genes dropped and
  reported), `template_query` (Methods Eq. signature: ρ = 0.5, Ω = 90th-percentile shrunk variances from
  `variance_quantiles`, kept beside the index since Spindle-lite partial stores no covariances), `tile_query`. All
  return `(cov_s, idx)` for `SpindlePartial` / `SpindleLitePartial`. Chosen by a gene-set pilot (25–26 Oct;
  removed in the 27 Oct cleanup, in git history); the exemplar rule stays experiment code (`fig8_pan_cancer/common8.py`).
- **Figures** (`experiments/`, `results/`, `figures/`): `experiments/common/` holds the shared pieces (`data.py`:
  datasets, the index cache in `results/_cache/`, seeds 0–4 with seed 0 for examples, the `method-v1` stamp;
  `style.py`, `build_figure.py`, `render.py`, `assemble_all.py`; `run.sbatch` / `timing.sbatch`).
  `experiments/<fig>/run.sh` lists each figure's steps in order (sbatch `c_*.py` → `f_*.py` atoms → draft pptx →
  render → `write_numbers.py` → `results/<fig>/numbers.json`). `figures/<fig>/<fig>.pptx` is the user's edited copy
  and the source of truth (no script writes it); Fig 1 is the user's own PDF.
- **Layout** (`layout.py`): tiles → Leiden covariance niches (cap 1,000) → gene order and blocks per niche.
  `preprocessing.py` tiles the section, picks genes, computes covariances and splits train/held-out.

## Rules

- **The method is frozen** at git tag `method-v1`. Do not change `src/`: no variants, no retuning, no extra checks.
  A real bug goes to the user first. `data.stamp()` refuses to make a figure input when `src/` differs from the tag
  (the only allowed change is the 27 Oct removal of the unused `spindle_lite_legacy.py`).
- Dataset paths live only in `datasets.yaml`; read them with `paths.dataset_path(key)` (shell:
  `python paths.py <key>`). Never hard-code a path.
- Experiment code goes in `experiments/<figure>/` and uses only the public API of `spindle_dev`;
  baselines never go into `src/`.
- **Benchmarks** time the compiled searches only (`Spindle.distances` + sort, `SpindleLite.search_vectors`,
  `SpindleLitePartial.search_vectors`; gene sets exact: `SpindlePartial.distances` + sort), from query vectors / pieces
  computed beforehand (`query_vectors`), one thread
  (`OMP_NUM_THREADS=1` etc.). Never time the numpy / Python references.
- Heavy compute goes through `sbatch` (the shell is a login node), in the `spindle_env` conda env
  (`~/miniforge3/envs/spindle_env`). `spindle_dev` is not installed there: put `src/` on `sys.path`.
  Compute nodes cannot see the login node's `/tmp`, so job scripts and outputs must live on the shared
  file system.
- The manuscript is a separate Overleaf project (local copy `Spindle/`, gitignored); no code writes into it. The
  Results section is written by hand in `Spindle/main_submission.tex`, matching the user's own Results and Methods.

## Run

```bash
python examples/run_single_dataset.py skin_melanoma
```

There is no test suite; `examples/check_indexes.py <key>` (via sbatch) is the sanity check: Spindle
self-queries (whole tile and gene set), compiled == reference for all three compiled searches, and the lower bound
and safety of both Spindle-lites.
