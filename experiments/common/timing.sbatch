#!/bin/bash
# Every timed job (builds, query timings): as run.sbatch, plus one fixed node and one job at a time. All timing jobs
# share the job name, so --dependency=singleton queues them behind each other; an array also needs %1.
#   sbatch --array=0-39%1 experiments/common/timing.sbatch <script.py> [args]
#SBATCH --job-name=spindle_timing
#SBATCH --dependency=singleton
#SBATCH --partition=cpus
#SBATCH --nodelist=ROSALIND-CPU1
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
echo "load: $(cat /proc/loadavg)"
python -u "$@"
