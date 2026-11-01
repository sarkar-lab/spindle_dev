#!/bin/bash
# Execute a result notebook on a compute node (spindle_env kernel, every thread count = 1, 16 GB: the reviewers' budget):
#   sbatch --job-name=<notebook stem> --output=results/_logs/<notebook stem>/%x_%j.out \
#          experiments/common/run_nb.sbatch experiments/compute/<notebook>.ipynb
#SBATCH --job-name=spindle_nb
#SBATCH --partition=batch
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/home/NAShome/ghoss18/spindle_dev/results/_logs/%x_%j.out
set -euo pipefail
cd /home/NAShome/ghoss18/spindle_dev
unset VIRTUAL_ENV PYTHONHOME
source ~/miniforge3/etc/profile.d/conda.sh
conda activate spindle_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "$* on $(hostname), job ${SLURM_JOB_ID:-}, $(git describe --tags --dirty --always)"
python -u experiments/common/run_nb.py "$@"
