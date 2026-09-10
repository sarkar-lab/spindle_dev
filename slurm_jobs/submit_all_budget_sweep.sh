#!/bin/bash
# Submits one independent SLURM job per dataset for the budget-sweep holdout
# benchmark, so datasets run in parallel across the cluster instead of
# sequentially in a single job. Run this directly on the login node (it just
# calls `sbatch` per dataset and exits -- it is not itself a SLURM job).
#
# Usage:
#   ./slurm_jobs/submit_all_budget_sweep.sh [train_test_ratio] [stop_metric] [max_queries]
#
# Each dataset's run (indexing + budget sweep) is submitted via
# run_single_budget_sweep.sbatch, one job per dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATASET_DIR="/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium"
TRAIN_TEST_RATIO=${1:-0.1}
STOP_METRIC=${2:-overlap_at_50}
MAX_QUERIES=${3:-100}

mkdir -p logs

echo "Submitting one parallel job per dataset (train-test-ratio=$TRAIN_TEST_RATIO, stop-metric=$STOP_METRIC, max-queries=$MAX_QUERIES)..."
for DATASET_PATH in "$DATASET_DIR"/*.h5ad; do
    DATASET_NAME=$(basename "$DATASET_PATH")
    echo "  Submitting $DATASET_NAME..."
    sbatch slurm_jobs/run_single_budget_sweep.sbatch "$DATASET_NAME" "$TRAIN_TEST_RATIO" "$STOP_METRIC" "$MAX_QUERIES"
done

echo "All jobs submitted. Track with: squeue -u $USER"
