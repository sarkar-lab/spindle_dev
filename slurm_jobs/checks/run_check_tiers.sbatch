#!/bin/bash
#SBATCH --job-name=spindle_check_tiers
#SBATCH --partition=cpus
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Sanity check of src/spindle_dev/tiers.py (benchmarks/checks/check_tiers.py).
# Usage: sbatch slurm_jobs/checks/run_check_tiers.sbatch [dataset]

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/checks/check_tiers.py --dataset "${1:-skin_melanoma}"
status=$?
echo "Done (exit $status)."
exit $status
