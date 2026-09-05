#!/bin/bash
#SBATCH --job-name=spindle_benchmarks_all
#SBATCH --partition=cpus
#SBATCH --time=12:00:00               
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16            
#SBATCH --mem=256G                     
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

# Loop over all .h5ad files in the dataset directory
for DATASET_PATH in "$DATASET_DIR"/*.h5ad; do
    echo "============================================================"
    echo "STARTING BENCHMARKS FOR: $(basename "$DATASET_PATH")"
    echo "============================================================"

    echo "Running Holdout Validation Benchmark..."
    python benchmarks/holdout_validation.py --dataset-paths "$DATASET_PATH" --train-test-ratio 0.05

    echo "Running Partial Panel Search Benchmark..."
    python benchmarks/partial_panel_search.py --dataset-paths "$DATASET_PATH" --num-queries 5

    echo "Finished benchmarks for $(basename "$DATASET_PATH")"
    echo ""
done

echo "All datasets processed successfully!"
