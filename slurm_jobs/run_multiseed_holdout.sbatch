#!/bin/bash
#SBATCH --job-name=spindle_multiseed_holdout
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# One dataset x seed unit of Stage A / E5: rebuilds the index at this seed's
# holdout split (benchmarks/index_datasets.py, --train-test-ratio, not a
# fixed --n-holdout count) and runs the single-point holdout-validation
# benchmark at budget_multiplier=1.0 (benchmarks/holdout_validation.py), via
# benchmarks/multiseed_holdout.py's unit mode. That script points both at a
# seed-suffixed symlink of the real dataset file, so each seed gets its own
# index/covariance/run-log/ground-truth-cache artifacts without touching
# either reused script.
#
# Usage: sbatch run_multiseed_holdout.sbatch <dataset.h5ad> <seed> [train_test_ratio]
# Resource overrides for large datasets (lymph_node_5k, brain_cancer) are
# passed via `sbatch --mem=... --time=...` at submit time.

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

echo "Activating Conda environment..."
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

DATASET_DIR="/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium"
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: sbatch run_multiseed_holdout.sbatch <dataset.h5ad> <seed> [train_test_ratio]"
    exit 1
fi
DATASET_NAME=$1
SEED=$2
TRAIN_TEST_RATIO=${3:-0.10}
DATASET_PATH="$DATASET_DIR/$DATASET_NAME"

echo "============================================================"
echo "Multiseed holdout unit: $(basename "$DATASET_PATH") (seed=$SEED, train-test-ratio=$TRAIN_TEST_RATIO)"
echo "============================================================"
python benchmarks/multiseed_holdout.py --dataset-path "$DATASET_PATH" --seed "$SEED" \
    --train-test-ratio "$TRAIN_TEST_RATIO"
echo "Done."
