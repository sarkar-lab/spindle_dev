#!/bin/bash
#SBATCH --job-name=spindle_index_build
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Builds a single dataset's index (benchmarks/index_datasets.py) with a
# fixed seed=1 / n_holdout=100 tile holdout split. Writes
# *_spindle_index.pkl / *_raw_covariances.pkl under
# results/holdout_validation_indexed/ -- index_datasets.py's
# run_indexing_for_datasets() skips rebuilding if the index file already
# exists, so the caller must remove/move the old index file first for this
# to actually rebuild rather than silently no-op.
#
# Usage: sbatch run_single_index_build.sbatch <dataset.h5ad> [train_test_ratio] [seed] [n_holdout]
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
if [ -z "$1" ]; then
    echo "Usage: sbatch run_single_index_build.sbatch <dataset.h5ad> [train_test_ratio] [seed] [n_holdout]"
    exit 1
fi
DATASET_NAME=$1
TRAIN_TEST_RATIO=${2:-0.05}
SEED=${3:-1}
N_HOLDOUT=${4:-100}
DATASET_PATH="$DATASET_DIR/$DATASET_NAME"

echo "============================================================"
echo "Index build: $(basename "$DATASET_PATH") (seed=$SEED, n_holdout=$N_HOLDOUT, train-test-ratio=$TRAIN_TEST_RATIO)"
echo "============================================================"
python benchmarks/index_datasets.py --dataset-paths "$DATASET_PATH" --train-test-ratio "$TRAIN_TEST_RATIO" \
    --seed "$SEED" --n-holdout "$N_HOLDOUT"
echo "Done."
