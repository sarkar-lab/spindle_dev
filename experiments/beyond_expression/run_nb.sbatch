#!/bin/bash
# Execute a notebook in place on a compute node (sparta_spatial kernel):
#   sbatch experiments/beyond_expression/run_nb.sbatch <notebook.ipynb>
#SBATCH --job-name=spindle_nb
#SBATCH --partition=cpus
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=/home/NAShome/ghoss18/spindle_dev/results/_logs/%x_%j.out
set -euo pipefail
cd /home/NAShome/ghoss18/spindle_dev
unset VIRTUAL_ENV PYTHONHOME
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sparta_spatial
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
echo "$* on $(hostname)"
python -u experiments/beyond_expression/run_nb.py "$@"
