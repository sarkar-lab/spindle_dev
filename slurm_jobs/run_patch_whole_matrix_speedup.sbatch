#!/bin/bash
#SBATCH --job-name=spindle_patch_whole_speedup
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# One-off patch for an already-completed multiseed_holdout unit whose
# bf_time_ms/speedup were computed against the block-diagonalized brute-force
# cost instead of the whole-matrix one (see
# benchmarks/patch_whole_matrix_speedup.py's module docstring). Computes only
# the missing whole-matrix ground truth and rewrites bf_time_ms/speedup in
# place -- does not re-run search or the (already-correct) block ground
# truth.
#
# Usage: sbatch run_patch_whole_matrix_speedup.sbatch <dataset.h5ad> <seed>
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
    echo "Usage: sbatch run_patch_whole_matrix_speedup.sbatch <dataset.h5ad> <seed>"
    exit 1
fi
DATASET_NAME=$1
SEED=$2
DATASET_PATH="$DATASET_DIR/$DATASET_NAME"

echo "============================================================"
echo "Patch whole-matrix speedup: $(basename "$DATASET_PATH") (seed=$SEED)"
echo "============================================================"
python benchmarks/patch_whole_matrix_speedup.py --dataset-path "$DATASET_PATH" --seed "$SEED"
echo "Done."
