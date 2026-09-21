#!/bin/bash
#SBATCH --job-name=spindle_budget_sweep_single
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Runs the recall-vs-budget sweep (benchmarks/budget_sweep_holdout.py) for one
# dataset, reusing the already-built index/covariance pickles under
# results/holdout_validation_indexed/ (built with a fixed seed=1/n_holdout=100
# split by run_single_index_build.sbatch) -- no re-indexing here.
#
# Evaluates every swept budget_multiplier against BOTH the block-diagonalized
# and whole-matrix ground truths (computed/cached once per dataset), using
# the fixed budget_mults grid baked into budget_sweep_holdout.DEFAULT_BUDGET_MULTS.
# The plateau early-stop never fires before budget_multiplier=1.0 and is keyed
# off the whole-matrix ground truth's recall_at_eps_0.25 (see
# EARLY_STOP_GROUND_TRUTH_KIND/EARLY_STOP_METRIC in budget_sweep_holdout.py).
#
# Usage: sbatch run_single_budget_sweep.sbatch <dataset.h5ad>
# Resource overrides for large datasets (lymph_node_5k, brain_cancer) are
# passed via `sbatch --mem=... --time=... --partition=...` at submit time
# (see submit_all_budget_sweep.sh), not hardcoded here.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

echo "Activating Conda environment..."
# Guard against a stray VIRTUAL_ENV/PATH leaking in from the submitting shell
# (e.g. a local .venv activated in an editor session) shadowing the conda env.
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

DATASET_DIR="/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium"
if [ -z "$1" ]; then
    DATASET_NAME="xenium_human_breast_cancer.h5ad"
else
    DATASET_NAME=$1
fi
# Informational only (recorded in the ground-truth cache and run log) -- the
# holdout tile set is already fixed by the index build, so there is nothing
# left to subsample here.
SEED=${2:-1}
N_HOLDOUT=${3:-100}

DATASET_PATH="$DATASET_DIR/$DATASET_NAME"
echo "============================================================"
echo "Budget sweep: $(basename "$DATASET_PATH") (seed=$SEED, n_holdout=$N_HOLDOUT)"
echo "============================================================"

python benchmarks/budget_sweep_holdout.py --dataset-paths "$DATASET_PATH" --seed "$SEED" --n-holdout "$N_HOLDOUT"
echo "Done."
