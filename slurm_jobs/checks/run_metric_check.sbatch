#!/bin/bash
#SBATCH --job-name=spindle_metric_check
#SBATCH --partition=cpus
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Part 1.1 metric check: raw vs shrunk vs per-gene-floored covariances under L2 d_B (benchmarks/checks/metric_check.py).
# Usage: sbatch [--mem=...] slurm_jobs/checks/run_metric_check.sbatch <dataset> | --aggregate

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/checks/metric_check.py --aggregate
else
    python benchmarks/checks/metric_check.py --dataset "$1"
fi
status=$?
echo "Done (exit $status)."
exit $status
