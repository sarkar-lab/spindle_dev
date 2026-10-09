#!/bin/bash
#SBATCH --job-name=spindle_niche_concordance
#SBATCH --partition=cpus
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Figs 6-7: covariance niches vs cell types (breast obs Cluster, else 10x graphclust), all builds
# (benchmarks/biology/niche_concordance.py).
# Usage: sbatch slurm_jobs/biology/run_niche_concordance.sbatch [--datasets ...] [--seeds ...]

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/biology/niche_concordance.py "$@"
echo "Done."
