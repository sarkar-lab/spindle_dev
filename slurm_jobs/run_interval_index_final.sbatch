#!/bin/bash
#SBATCH --job-name=spindle_interval_index_final
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Partial-query search on every dataset: interval index vs exact tier (benchmarks/interval_index_final.py).
# Usage: sbatch [--mem=...] slurm_jobs/run_interval_index_final.sbatch <dataset>  (brain_cancer, lymph_node_5k: --mem=128G)

export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/interval_index_final.py --dataset "$1"
status=$?
echo "Done (exit $status)."
exit $status
