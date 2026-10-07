#!/bin/bash
#SBATCH --job-name=spindle_dag_gene_ladder
#SBATCH --partition=batch
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out

# Fig 2E: index build cost (and DAG overlap, supplementary) vs genes on lymph node 5K, all tiles, streaming
# builder (benchmarks/index/dag_gene_ladder.py). 16 CPUs, as the other build-cost runs.
# Usage: sbatch [--mem=...] slurm_jobs/index/run_dag_gene_ladder.sbatch <G> [--max-queries N]
#        sbatch slurm_jobs/index/run_dag_gene_ladder.sbatch --aggregate
#   all points: bash slurm_jobs/index/submit_dag_gene_ladder.sh

# per-tile work runs in joblib processes; numpy stays single-threaded inside each,
# except the Gram/eigendecomposition step, which uses all allocated CPUs
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LOKY_MAX_CPU_COUNT=$SLURM_CPUS_PER_TASK
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/index/dag_gene_ladder.py --aggregate
else
    G=$1; shift 1
    python benchmarks/index/dag_gene_ladder.py --genes "$G" "$@"
fi
status=$?
echo "Done (exit $status)."
exit $status
