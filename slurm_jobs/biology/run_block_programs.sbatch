#!/bin/bash
#SBATCH --job-name=spindle_block_programs
#SBATCH --partition=cpus
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Figs 6-7: niche x block gene programs (co-variation, enrichment vs size-matched random sets, rewiring,
# recurrence over seeds) from the saved indexes (benchmarks/biology/block_programs.py).
# Usage: sbatch slurm_jobs/biology/run_block_programs.sbatch [--datasets ...] [--seeds ...]

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/biology/block_programs.py "$@"
echo "Done."
