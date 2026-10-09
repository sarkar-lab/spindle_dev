#!/bin/bash
#SBATCH --job-name=spindle_query_example
#SBATCH --partition=cpus
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Fig 6G: one worked breast whole-tile query (exact top 10 + DAG set) and its base rate
# (benchmarks/biology/query_example.py).
# Usage: sbatch slurm_jobs/biology/run_query_example.sbatch

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/biology/query_example.py "$@"
echo "Done."
