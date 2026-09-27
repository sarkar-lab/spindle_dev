#!/bin/bash
#SBATCH --job-name=spindle_metric_concordance
#SBATCH --partition=cpus
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Paper plan 02, E14: block vs whole-matrix LE neighbours vs biology
# (benchmarks/metric_concordance.py).
# Usage: sbatch [--mem=...] slurm_jobs/run_metric_concordance.sbatch <dataset> [--seed N]

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/metric_concordance.py --dataset "$@"
echo "Done."
