#!/bin/bash
#SBATCH --job-name=spindle_xenium_human_skin_melanoma
#SBATCH --partition=batch
#SBATCH --account=sarkar_lab
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --mail-user=hirak.sarkar@vanderbilt.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ---- IMPORTANT: limit BLAS threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo -e "Setting up virtual environment for processing /data/sarkar_lab/insitupy_demo_data_xenium/xenium_human_skin_melanoma.h5ad...\n\n"
export PATH="${HOME}/miniforge3/bin:$PATH"
eval "$(${HOME}/miniforge3/bin/mamba shell hook --shell bash)"
source ${HOME}/miniforge3/bin/activate base
mamba activate spatial

echo -e "Processing /data/sarkar_lab/insitupy_demo_data_xenium/xenium_human_skin_melanoma.h5ad...\n\n"

python /data/sarkar_lab/Projects/spindle_dev/ISMB_notebook/spindle_xenium_single.py \
    "/data/sarkar_lab/insitupy_demo_data_xenium/xenium_human_skin_melanoma.h5ad" \
    --resolution 0.5 \
    --min-final-size 15 \
    --top-genes 800 \
    --max-queries 200 \
    --all-genes
