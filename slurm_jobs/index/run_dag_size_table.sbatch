#!/bin/bash
#SBATCH --job-name=spindle_dag_size_table
#SBATCH --partition=cpus
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Per-dataset storage table: whole covariances vs block logs vs the final DAG (benchmarks/index/dag_size_table.py).
# Usage: sbatch [--mem=...] slurm_jobs/index/run_dag_size_table.sbatch <dataset> | --aggregate

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/index/dag_size_table.py --aggregate
else
    python benchmarks/index/dag_size_table.py --dataset "$1"
fi
status=$?
echo "Done (exit $status)."
exit $status
