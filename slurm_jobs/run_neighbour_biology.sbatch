#!/bin/bash
#SBATCH --job-name=spindle_nbr_bio
#SBATCH --partition=cpus
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Fig 3E: expression r and composition JSD of each method's returned tiles (benchmarks/neighbour_biology.py).
# Needs results/whole_cov_baselines/<stem>_seed<s>_returned.npz.
# Usage: sbatch [--mem=...] slurm_jobs/run_neighbour_biology.sbatch <dataset> <seed> | --aggregate

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/neighbour_biology.py --aggregate
else
    python benchmarks/neighbour_biology.py --dataset "$1" --seed "$2"
fi
status=$?
echo "Done (exit $status)."
exit $status
