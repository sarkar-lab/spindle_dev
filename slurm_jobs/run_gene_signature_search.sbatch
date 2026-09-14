#!/bin/bash
#SBATCH --job-name=spindle_gene_sig
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
# Unset any inherited venv from the submitting shell (e.g. the repo's own .venv),
# which would otherwise shadow the conda env on PATH.
unset VIRTUAL_ENV
# Ensure conda commands work inside the SLURM job
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
export PATH="${HOME}/miniforge3/envs/spindle_env/bin:$PATH"

# Navigate to the project directory where you submitted the job from
cd $SLURM_SUBMIT_DIR

DATASET_DIR="/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium"

echo "============================================================"
echo "GENE SIGNATURE SEARCH ON ALL DATASETS"
echo "============================================================"
for DATASET_PATH in "$DATASET_DIR"/*.h5ad; do
    echo "Gene signature search for $(basename "$DATASET_PATH")..."
    python benchmarks/gene_signature_search.py --dataset-paths "$DATASET_PATH"
done
echo "Gene signature search complete."
echo ""

echo "============================================================"
echo "REGENERATING SPATIAL COMPARISON FIGURES"
echo "============================================================"
python scripts/plot_gene_signature_maps.py

echo "All datasets processed successfully!"
