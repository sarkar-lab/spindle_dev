#!/bin/bash
#SBATCH --job-name=spindle_cpu                # Job name
#SBATCH --partition=batch         # Partition
#SBATCH --account=sarkar_lab
#SBATCH --time=1-1:00:00                   # Time limit (hh:mm:ss)
#SBATCH --nodes=1                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB                         # Memory per node
#SBATCH --mail-user=hirak.sarkar@vanderbilt.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# ---- IMPORTANT: limit BLAS threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo -e "Setting up virtual environment for training model...\n\n"
export PATH="${HOME}/miniforge3/bin:$PATH"
eval "$(${HOME}/miniforge3/bin/mamba shell hook --shell bash)"
source ${HOME}/miniforge3/bin/activate base
mamba activate spatial

echo -e "Starting training model...\n\n"

python /data/sarkar_lab/Projects/spindle_dev/ISMB_notebook/spindle_xenium.py