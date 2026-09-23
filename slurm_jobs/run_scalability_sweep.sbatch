#!/bin/bash
#SBATCH --job-name=spindle_scalability_sweep
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# One (base dataset, target_cells, seed) unit of the E6 synthetic
# scalability sweep (benchmarks/scalability_sweep.py). Subsamples or
# bootstrap-resamples+jitters the base dataset to target_cells cells and
# runs the real index-build pipeline, recording build_time_s/index_size_mb.
#
# Usage: sbatch run_scalability_sweep.sbatch <dataset.h5ad> <target_cells> [seed]
# Resource overrides for large target_cells (>=5e5) are passed via
# `sbatch --mem=... --time=...` at submit time.

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
    echo "Usage: sbatch run_scalability_sweep.sbatch <dataset.h5ad> <target_cells> [seed]"
    exit 1
fi
DATASET_NAME=$1
TARGET_CELLS=$2
SEED=${3:-0}
DATASET_PATH="$DATASET_DIR/$DATASET_NAME"

echo "============================================================"
echo "Scalability sweep unit: $(basename "$DATASET_PATH") target_cells=$TARGET_CELLS seed=$SEED"
echo "============================================================"
python benchmarks/scalability_sweep.py --dataset-path "$DATASET_PATH" --target-cells "$TARGET_CELLS" --seed "$SEED"
echo "Done."
