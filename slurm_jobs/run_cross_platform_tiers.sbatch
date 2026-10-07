#!/bin/bash
#SBATCH --job-name=spindle_cross_platform
#SBATCH --partition=cpus
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

# Fig 5 / S9: cross-platform Xenium <-> Visium search on the tiers (benchmarks/cross_platform_tiers.py).
# Usage: sbatch slurm_jobs/run_cross_platform_tiers.sbatch <seed> <max_pts>
#        sbatch slurm_jobs/run_cross_platform_tiers.sbatch --overlay|--bias-pca|--example|--check [--max-pts N]
#        sbatch slurm_jobs/run_cross_platform_tiers.sbatch --aggregate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

case "$1" in
    --*) python benchmarks/cross_platform_tiers.py "$@" ;;
    *)   python benchmarks/cross_platform_tiers.py --seed "$1" --max-pts "$2" "${@:3}" ;;
esac
status=$?
echo "Done (exit $status)."
exit $status
