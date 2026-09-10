#!/bin/bash
#SBATCH --job-name=spindle_budget_sweep_single
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Limit threading to prevent overloading the node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Ensure logs directory exists
mkdir -p logs

echo "Activating Conda environment..."
# Guard against a stray VIRTUAL_ENV/PATH leaking in from the submitting shell
# (e.g. a local .venv activated in an editor session) shadowing the conda env.
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
# Ensure conda commands work inside the SLURM job
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

# Navigate to the project directory where you submitted the job from
cd $SLURM_SUBMIT_DIR

DATASET_DIR="/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium"
if [ -z "$1" ]; then
    DATASET_NAME="xenium_human_breast_cancer.h5ad"
else
    DATASET_NAME=$1
fi
# Fraction of tiles held out as unseen test queries during PHASE 1 indexing
# (the split happens once, at index-build time -- PHASE 2 only loads the
# already-built index/covariance pickles, so this must be passed here, not
# to budget_sweep_holdout.py).
TRAIN_TEST_RATIO=${2:-0.05}
# Metric checked for perfect (==1.0) recovery to stop the sweep early.
STOP_METRIC=${3:-overlap_at_50}
# Cap on the number of held-out test queries evaluated (randomly subsampled,
# fixed seed) -- keeps runtime bounded for large datasets. Empty = no cap.
MAX_QUERIES=${4:-100}

echo "Selected dataset: $DATASET_NAME"
echo "============================================================"
echo "PHASE 1: INDEXING SINGLE DATASET"
echo "============================================================"
DATASET_PATH="$DATASET_DIR/$DATASET_NAME"
echo "Indexing $(basename "$DATASET_PATH") with train-test-ratio=$TRAIN_TEST_RATIO..."
python benchmarks/index_datasets.py --dataset-paths "$DATASET_PATH" --train-test-ratio "$TRAIN_TEST_RATIO"
echo "Phase 1 Complete."
echo ""

echo "============================================================"
echo "PHASE 2: BUDGET-SWEEP HOLDOUT VALIDATION ON SINGLE DATASET"
echo "============================================================"
echo "Budget sweep for $(basename "$DATASET_PATH")..."
MAX_QUERIES_ARGS=()
if [ -n "$MAX_QUERIES" ]; then
    MAX_QUERIES_ARGS=(--max-queries "$MAX_QUERIES")
fi
python benchmarks/budget_sweep_holdout.py --dataset-paths "$DATASET_PATH" --stop-metric "$STOP_METRIC" "${MAX_QUERIES_ARGS[@]}"
echo "Phase 2 Complete."
echo ""

echo "Single dataset processed successfully!"
