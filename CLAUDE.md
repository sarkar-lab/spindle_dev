# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**Spindle** indexes and searches the covariance matrices of spatial-omics tiles. The paper has two indexes
plus a partial-query index, all on one shared layout:

- **Spindle** (`src/spindle_dev/spindle.py`): exact scan over stored block logs. `Spindle` answers whole-tile
  queries with d_B = √(Σ_b ‖log q_b − log t_b‖²_F)/√p on shrunk covariances (`metrics.SHRINK_ALPHA` = 0.1);
  `SpindlePartial` answers gene-set queries with d_B^S.
- **Spindle-lite** (`lite.py`): the first draft's layered block DAG (`Spindle/Spindle_Old.pdf`): ε-net nodes
  per block, edges between co-occurring nodes, budgeted DFS searched in every niche.
- **Interval index** (`interval.py`): dyadic gene intervals with ε-bounded node means, for gene sets.
- **Layout** (`layout.py`): tiles → Leiden covariance niches (cap 1,000) → gene order and blocks per niche.
  `preprocessing.py` tiles the section, picks genes, computes covariances and splits train/held-out.

## Rules

- **The method is frozen** once `paper/FIGURES.md` Stage A closes (git tag `method-v1`). Do not change
  `src/` while results or figures are being made: no variants, no retuning, no extra checks. A real bug goes
  to the user first; see `paper/FIGURES.md` Section 0. That file is the plan; read it at the start of any
  paper session.
- Dataset paths live only in `datasets.yaml`; read them with `paths.dataset_path(key)` (shell:
  `python paths.py <key>`). Never hard-code a path.
- Experiment code goes in `experiments/<figure>/` and uses only the public API of `spindle_dev`;
  baselines never go into `src/`.
- Heavy compute goes through `sbatch` (the shell is a login node), in the `spindle_env` conda env
  (`~/miniforge3/envs/spindle_env`). `spindle_dev` is not installed there: put `src/` on `sys.path`.
  Compute nodes cannot see the login node's `/tmp`, so job scripts and outputs must live on the shared
  file system.
- The manuscript is a separate Overleaf project (local copy `Spindle/`, gitignored); no code writes into it.
- `backup/` (gitignored) holds the figures from before the 17 Oct 2026 cleanup and the reference
  outputs used to check that the cleanup did not change results (`backup/reference/compare.py`).

## Run

```bash
python examples/run_single_dataset.py skin_melanoma
```

There is no test suite; `backup/reference/ref_new.py` + `compare.py` reproduce the cleanup check.
