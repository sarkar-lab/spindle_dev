#!/bin/bash
#SBATCH --job-name=spindle_diagnose_recall
#SBATCH --partition=cpus
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

# Navigate to the project directory where you submitted the job from
cd $SLURM_SUBMIT_DIR

DATASET_DIR="/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium"

# Default to the 5k lymph node dataset; override with $1 if provided
if [ -z "$1" ]; then
    DATASET_NAME="xenium_human_lymph_node_5k.h5ad"
else
    DATASET_NAME=$1
fi

DATASET_STEM="${DATASET_NAME%.h5ad}"
DATASET_PATH="$DATASET_DIR/$DATASET_NAME"
INDEX_DIR="results/holdout_validation_indexed"

echo "============================================================"
echo "DIAGNOSING RECALL DROP FOR: $DATASET_NAME"
echo "============================================================"

python diagnose_recall.py \
    --index-dir "$INDEX_DIR" \
    --dataset-name "$DATASET_STEM" \
    --n-queries 10 \
    --budget-mult 1.0

echo ""
echo "Diagnosis complete. Check logs/ for output."
