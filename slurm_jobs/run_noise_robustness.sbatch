#!/bin/bash
#SBATCH --job-name=spindle_noise_robustness
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Paper plan 02, E7: noisy-query robustness on one dataset's seed-73
# production build (benchmarks/noise_robustness.py). Big datasets are split
# into one job per noise level by slurm_jobs/submit_noise_robustness.sh.
#
# Usage: sbatch slurm_jobs/run_noise_robustness.sbatch <dataset> <spectral|cell_subsample> [levels...]

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

DATASET=$1; MODEL=$2; shift 2
if [ $# -gt 0 ]; then
    python benchmarks/noise_robustness.py --dataset "$DATASET" --model "$MODEL" --levels "$@"
else
    python benchmarks/noise_robustness.py --dataset "$DATASET" --model "$MODEL"
fi
echo "Done."
