#!/bin/bash
# Execute the timing notebook: as run_nb.sbatch, plus the fixed node and one timing job at a time (shares the job name
# of timing.sbatch, so --dependency=singleton queues it behind any other timed job):
#   sbatch experiments/common/timing_nb.sbatch experiments/compute/R4_timing.ipynb
#SBATCH --job-name=spindle_timing
#SBATCH --dependency=singleton
#SBATCH --partition=cpus
#SBATCH --nodelist=ROSALIND-CPU1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/home/NAShome/ghoss18/spindle_dev/results/_logs/R4_timing/%x_%j.out
set -euo pipefail
cd /home/NAShome/ghoss18/spindle_dev
unset VIRTUAL_ENV PYTHONHOME
source ~/miniforge3/etc/profile.d/conda.sh
conda activate spindle_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "$* on $(hostname), job ${SLURM_JOB_ID:-}, $(git describe --tags --dirty --always)"
echo "load: $(cat /proc/loadavg)"
python -u experiments/common/run_nb.py "$@"
