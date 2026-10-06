#!/bin/bash
#SBATCH --job-name=spindle_dag_cell_ladder
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Fig 2D: index build cost (and DAG overlap, supplementary) vs cells, one Xenium section subsampled
# (benchmarks/dag_cell_ladder.py; default the 1.35M-cell tonsil). Same CPUs/threads as run_tier_build_stats.sbatch.
# Usage: sbatch [--mem=... --partition=batch --time=...] slurm_jobs/run_dag_cell_ladder.sbatch \
#            <dataset> <cells|full> [--max-queries N]
#        sbatch slurm_jobs/run_dag_cell_ladder.sbatch --aggregate
#   all units: bash slurm_jobs/submit_dag_cell_ladder.sh

# covariances use joblib threads; numpy stays single-threaded
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/dag_cell_ladder.py --aggregate
else
    DATASET=$1; CELLS=$2; shift 2
    python benchmarks/dag_cell_ladder.py --dataset "$DATASET" --cells "$CELLS" "$@"
fi
status=$?
echo "Done (exit $status)."
exit $status
