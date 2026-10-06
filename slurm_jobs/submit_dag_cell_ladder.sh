#!/bin/bash
# Fig 2D cell ladder (niche cap 1000): one Xenium section subsampled 1K -> full. Default dataset: the
# 1.35M-cell human tonsil (make it first: python benchmarks/prepare_xenium_h5ad.py tonsil_reactive).
# One job per point, then an aggregate job.
# Usage: ./slurm_jobs/submit_dag_cell_ladder.sh [dataset]   (run on the login node)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
mkdir -p logs
# Build times are reported: run one timing job at a time (Slurm singleton on a shared job name), so
# our own jobs never compete for the node.
SERIAL=(--job-name=spindle_timing --dependency=singleton)
RUN=slurm_jobs/run_dag_cell_ladder.sbatch
DATASET=${1:-tonsil_reactive}
JOBS=()
unit() {  # <cells>
    local RES=(--mem=48G)
    case $1 in 500000) RES=(--mem=96G) ;; 1000000|full) RES=(--partition=batch --time=1-00:00:00 --mem=192G) ;; esac
    JOBS+=($(sbatch --parsable "${SERIAL[@]}" "${RES[@]}" "$RUN" "$DATASET" "$1"))
}
for CELLS in 1000 2000 5000 10000 20000 50000 100000 200000 500000 1000000 full; do unit "$CELLS"; done
DEP=$(IFS=:; echo "${JOBS[*]}")
AGG=$(sbatch --parsable --dependency=afterany:$DEP --mem=8G --time=00:30:00 "$RUN" --aggregate)
echo "Submitted ${#JOBS[@]} jobs; aggregate $AGG. Track with: squeue -u $USER"
