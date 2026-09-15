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
# results/holdout_validation_indexed/ -- no re-indexing (Phase 1 is skipped
# on purpose: all 8 datasets already have current indexed pickles, and
# re-indexing here would be redundant, multi-GB, multi-hour work for no
# benefit to this experiment).
#
# Usage: sbatch run_single_budget_sweep.sbatch <dataset.h5ad> [stop_metric] [max_queries]
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
# Metric checked for perfect (==1.0) recovery to stop the sweep early.
STOP_METRIC=${2:-recall_at_eps_0.1}
# Cap on the number of held-out test queries evaluated (randomly subsampled,
# fixed seed) -- keeps runtime bounded across ~16-30 budget multipliers per
# dataset. This sweep is for finding the accuracy/speed sweet spot, not the
# final reported numbers (those come later, over the full query set). Empty
# string = no cap.
MAX_QUERIES=${3:-100}

DATASET_PATH="$DATASET_DIR/$DATASET_NAME"
echo "============================================================"
echo "Budget sweep: $(basename "$DATASET_PATH") (stop-metric=$STOP_METRIC, max-queries=$MAX_QUERIES)"
echo "============================================================"

MAX_QUERIES_ARGS=()
if [ -n "$MAX_QUERIES" ]; then
    MAX_QUERIES_ARGS=(--max-queries "$MAX_QUERIES")
fi
python benchmarks/budget_sweep_holdout.py --dataset-paths "$DATASET_PATH" --stop-metric "$STOP_METRIC" "${MAX_QUERIES_ARGS[@]}"
echo "Done."
