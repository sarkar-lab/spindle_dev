#!/bin/bash
#SBATCH --job-name=spindle_ann_baselines
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out

# Paper plan 02, E4: FAISS/PCA-kNN baselines vs Spindle on one dataset's
# seed-73 production build (benchmarks/ann_baselines.py). Index builds use
# all allocated CPUs; every search runs single-threaded (the script sets FAISS
# to 1 thread and BLAS is pinned to 1 thread here), matching how Spindle's
# own query times were measured.
#
# Usage: sbatch [--mem=160G] slurm_jobs/run_ann_baselines.sbatch <dataset> [extra args]
#   e.g. sbatch slurm_jobs/run_ann_baselines.sbatch breast_cancer --sensitivity

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

DATASET=$1; shift
python benchmarks/ann_baselines.py --dataset "$DATASET" "$@"
echo "Done."
