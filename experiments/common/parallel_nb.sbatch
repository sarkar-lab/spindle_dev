#!/bin/bash
# Execute the multi-core timing notebook (R11_parallel) on 65 physical cores (64 workers + the main process) of the timing node, one hardware thread per
# core. Shares the job name of the other timing jobs, so --dependency=singleton queues it behind any of them:
#   sbatch --export=ALL,SPINDLE_PARALLEL=pilot experiments/common/parallel_nb.sbatch experiments/compute/R11_parallel.ipynb
#   sbatch --export=ALL,SPINDLE_PARALLEL=full  experiments/common/parallel_nb.sbatch experiments/compute/R11_parallel.ipynb
#SBATCH --job-name=spindle_timing
#SBATCH --dependency=singleton
#SBATCH --partition=cpus
#SBATCH --nodelist=ROSALIND-CPU1
#SBATCH --cpus-per-task=65
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=/home/NAShome/ghoss18/spindle_dev/results/_logs/R11_parallel/%x_%j.out
set -euo pipefail
cd /home/NAShome/ghoss18/spindle_dev
unset VIRTUAL_ENV PYTHONHOME
source ~/miniforge3/etc/profile.d/conda.sh
conda activate spindle_env
# BLAS may use up to 64 threads; the code holds it at 1 (threadpoolctl) except in the timed whole-matrix logarithm
export OPENBLAS_NUM_THREADS=64 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
echo "$* (${SPINDLE_PARALLEL:-pilot}) on $(hostname), job ${SLURM_JOB_ID:-}, $(git describe --tags --dirty --always)"
echo "load: $(cat /proc/loadavg)"
echo "cpus: $(taskset -pc $$)"
lscpu | grep -E "Model name|Socket|Core|Thread|NUMA"
python -u experiments/common/run_nb.py "$@"
