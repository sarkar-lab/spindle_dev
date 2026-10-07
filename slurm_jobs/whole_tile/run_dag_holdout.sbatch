#!/bin/bash
#SBATCH --job-name=spindle_dag_holdout
#SBATCH --partition=cpus
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Fig 3B-C: Spindle-DAG retrieval on held-out tiles vs the exact tier (benchmarks/whole_tile/dag_holdout.py).
# Usage: sbatch [--mem=...] slurm_jobs/whole_tile/run_dag_holdout.sbatch <dataset> <seed> [--k-sweep] [--bound] | --aggregate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/whole_tile/dag_holdout.py --aggregate
else
    python benchmarks/whole_tile/dag_holdout.py --dataset "$1" --seed "$2" "${@:3}"
fi
status=$?
echo "Done (exit $status)."
exit $status
