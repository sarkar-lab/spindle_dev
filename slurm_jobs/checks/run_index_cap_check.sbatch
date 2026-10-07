#!/bin/bash
#SBATCH --job-name=spindle_index_cap_check
#SBATCH --partition=cpus
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Part 1.3: niche sizes of every saved index vs the 1,000-tile cap (benchmarks/checks/index_cap_check.py).
# Usage: sbatch slurm_jobs/checks/run_index_cap_check.sbatch

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/checks/index_cap_check.py
status=$?
echo "Done (exit $status)."
exit $status
