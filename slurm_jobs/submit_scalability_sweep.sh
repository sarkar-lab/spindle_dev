#!/bin/bash
# Submits one independent SLURM job per (base dataset, target_cells) unit of
# the E6 synthetic scalability sweep (benchmarks/scalability_sweep.py), so
# all units run in parallel across the cluster. Run this directly on the
# login node (it just calls `sbatch` per unit and exits).
#
# Usage:
#   ./slurm_jobs/submit_scalability_sweep.sh [seed]
#
# Base datasets: breast_cancer (159,226 real cells) and lymph_node (377,985
# real cells), per E6's suggestion. Target cell counts: [1e3, 5e3, 2e4, 5e4,
# 1e5, 5e5, 1e6] -- a synthetic sweep spanning ~3 orders of magnitude.
# Units with target_cells >= 5e5 get bumped --mem/--time overrides at submit
# time, matching the large-real-dataset convention used elsewhere.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SEED=${1:-0}

mkdir -p logs

BASE_DATASETS=(
    xenium_human_breast_cancer.h5ad
    xenium_human_lymph_node.h5ad
)
TARGET_CELLS=(1000 5000 20000 50000 100000 500000 1000000)
LARGE_THRESHOLD=500000

echo "Submitting scalability sweep jobs (seed=$SEED)..."
for DATASET_NAME in "${BASE_DATASETS[@]}"; do
    for CELLS in "${TARGET_CELLS[@]}"; do
        if [ "$CELLS" -ge "$LARGE_THRESHOLD" ]; then
            echo "  Submitting $DATASET_NAME target_cells=$CELLS (bumped resources)..."
            sbatch --mem=256G --time=24:00:00 --partition=batch \
                slurm_jobs/run_scalability_sweep.sbatch "$DATASET_NAME" "$CELLS" "$SEED"
        else
            echo "  Submitting $DATASET_NAME target_cells=$CELLS..."
            sbatch slurm_jobs/run_scalability_sweep.sbatch "$DATASET_NAME" "$CELLS" "$SEED"
        fi
    done
done

echo "All scalability sweep jobs submitted. Track with: squeue -u $USER"
