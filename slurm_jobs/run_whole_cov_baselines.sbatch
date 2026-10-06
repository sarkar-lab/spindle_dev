#!/bin/bash
#SBATCH --job-name=spindle_whole_cov
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Fig 3D: indexes over whole covariances (HNSW, PCA + flat, PCA + PQ) vs Spindle-DAG, memory vs fidelity
# (benchmarks/whole_cov_baselines.py). Index builds use the allocated CPUs; searches run on one thread.
# Usage: sbatch [--mem=...] slurm_jobs/run_whole_cov_baselines.sbatch <dataset> <seed> | --aggregate

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/whole_cov_baselines.py --aggregate
else
    python benchmarks/whole_cov_baselines.py --dataset "$1" --seed "$2"
fi
status=$?
echo "Done (exit $status)."
exit $status
