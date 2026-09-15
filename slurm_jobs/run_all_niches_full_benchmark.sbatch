#!/bin/bash
#SBATCH --job-name=spindle_all_niches_full
#SBATCH --partition=batch
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

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
BUDGET_MULT=${1:-1.0}
MAX_QUERIES=${2:-50}

echo "============================================================"
echo "Full search-all-niches benchmark, all 8 datasets, budget_mult=$BUDGET_MULT, max_queries=$MAX_QUERIES"
echo "============================================================"
python benchmarks/holdout_validation.py --budget-mult "$BUDGET_MULT" --max-queries "$MAX_QUERIES" --dataset-paths \
    "$DATASET_DIR/xenium_human_breast_cancer.h5ad" \
    "$DATASET_DIR/xenium_human_kidney_nondiseased.h5ad" \
    "$DATASET_DIR/xenium_human_lung_cancer.h5ad" \
    "$DATASET_DIR/xenium_human_lymph_node.h5ad" \
    "$DATASET_DIR/xenium_human_lymph_node_5k.h5ad" \
    "$DATASET_DIR/xenium_human_pancreatic_cancer.h5ad" \
    "$DATASET_DIR/xenium_human_brain_cancer.h5ad" \
    "$DATASET_DIR/xenium_human_skin_melanoma.h5ad"
echo "Done."
