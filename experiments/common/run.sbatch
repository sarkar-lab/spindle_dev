#!/bin/bash
# Any compute step: spindle_env, src/ on sys.path (the scripts add it), every thread count = 1.
#   sbatch [--array=...] [--mem=...] experiments/common/run.sbatch <script.py> [args]
#SBATCH --job-name=spindle_run
#SBATCH --partition=cpus
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/home/NAShome/ghoss18/spindle_dev/results/_logs/%x_%A_%a.out
set -euo pipefail
cd /home/NAShome/ghoss18/spindle_dev
unset VIRTUAL_ENV PYTHONHOME
source ~/miniforge3/etc/profile.d/conda.sh
conda activate spindle_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "$* on $(hostname), task ${SLURM_ARRAY_TASK_ID:-}, $(git describe --tags --dirty --always)"
python -u "$@"
