#!/bin/bash
#SBATCH --job-name=spindle_signature_queries
#SBATCH --partition=cpus
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Figs 6-7: gene-signature template queries on the partial tiers with the expression control
# (benchmarks/biology/signature_queries.py).
# Usage: sbatch slurm_jobs/biology/run_signature_queries.sbatch [--datasets ...] [--seeds ...] [--interval] [--maps]

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/biology/signature_queries.py "$@"
echo "Done."
