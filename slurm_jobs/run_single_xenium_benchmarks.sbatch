#!/bin/bash
#SBATCH --job-name=spindle_benchmarks_all
#SBATCH --partition=cpus
#SBATCH --time=12:00:00               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            
#SBATCH --mem=512G                     
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
echo "Selected dataset: $DATASET_NAME"
echo "============================================================"
echo "PHASE 1: INDEXING SINGLE DATASET"
echo "============================================================"
DATASET_PATH="$DATASET_DIR/$DATASET_NAME"
echo "Indexing $(basename "$DATASET_PATH")..."
python benchmarks/index_datasets.py --dataset-paths "$DATASET_PATH"
echo "Phase 1 Complete."
echo ""

echo "============================================================"
echo "PHASE 2: HOLDOUT VALIDATION ON SINGLE DATASET"
echo "============================================================"
echo "Holdout validation for $(basename "$DATASET_PATH")..."
python benchmarks/holdout_validation.py --dataset-paths "$DATASET_PATH" --train-test-ratio 0.05
echo "Phase 2 Complete."
echo ""

echo "============================================================"
echo "PHASE 3: PARTIAL PANEL SEARCH ON SINGLE DATASET"
echo "============================================================"

echo "Partial panel search for $(basename "$DATASET_PATH")..."
python benchmarks/partial_panel_search.py --dataset-paths "$DATASET_PATH" --num-queries 5
echo "Phase 3 Complete."
echo ""

echo "Single dataset processed successfully!"
