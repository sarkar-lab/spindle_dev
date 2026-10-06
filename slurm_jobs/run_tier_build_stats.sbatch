#!/bin/bash
#SBATCH --job-name=spindle_tier_build_stats
#SBATCH --partition=cpus
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Fig 2C: end-to-end index build time and peak memory (benchmarks/tier_build_stats.py).
# Same CPU count and BLAS threads as the saved index builds (run_build_index.sbatch).
# Usage: sbatch [--mem=...] slurm_jobs/run_tier_build_stats.sbatch <dataset> <seed> | --aggregate
#   all runs: bash slurm_jobs/submit_tier_build_stats.sh

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/tier_build_stats.py --aggregate
else
    python benchmarks/tier_build_stats.py --dataset "$1" --seed "$2"
fi
status=$?
echo "Done (exit $status)."
exit $status
