#!/bin/bash
#SBATCH --job-name=spindle_timing
#SBATCH --partition=cpus
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --output=logs/exact_vs_whole_%j.out

# Fig 3A: query time of whole (whole-matrix exact) / Spindle-Exact / Spindle-DAG (benchmarks/whole_tile/exact_vs_whole.py).
# One BLAS thread; submit through submit_exact_vs_whole.sh (singleton: one timing job at a time).
# Usage: sbatch [--mem=...] slurm_jobs/whole_tile/run_exact_vs_whole.sbatch <dataset> <seed> | --aggregate

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/whole_tile/exact_vs_whole.py --aggregate
else
    python benchmarks/whole_tile/exact_vs_whole.py --dataset "$1" --seed "$2"
fi
status=$?
echo "Done (exit $status)."
exit $status
