#!/bin/bash
# Submits one independent SLURM job per dataset to (re)build the Spindle
# index with a fixed seed / holdout-tile-count split, so datasets build in
# parallel across the cluster instead of sequentially. Run this directly on
# the login node (it just calls `sbatch` per dataset and exits -- it is not
# itself a SLURM job).
#
# Usage:
#   ./slurm_jobs/submit_reindex_all.sh [seed] [n_holdout] [train_test_ratio]
#
# Only submits the 6 "standard" datasets by default -- the two largest
# (lymph_node_5k, brain_cancer) are submitted separately, with bumped
# resources, only after the 6-dataset run has been validated end-to-end
# (see EXPERIMENT_PLAN.md-style staged rollout in the approved plan).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SEED=${1:-1}
N_HOLDOUT=${2:-100}
TRAIN_TEST_RATIO=${3:-0.05}

mkdir -p logs

DATASETS=(
    xenium_human_breast_cancer.h5ad
    xenium_human_kidney_nondiseased.h5ad
    xenium_human_lung_cancer.h5ad
    xenium_human_lymph_node.h5ad
    xenium_human_pancreatic_cancer.h5ad
    xenium_human_skin_melanoma.h5ad
)

echo "Submitting index-build jobs (seed=$SEED, n_holdout=$N_HOLDOUT, train_test_ratio=$TRAIN_TEST_RATIO)..."
for DATASET_NAME in "${DATASETS[@]}"; do
    echo "  Submitting $DATASET_NAME..."
    sbatch slurm_jobs/run_single_index_build.sbatch "$DATASET_NAME" "$TRAIN_TEST_RATIO" "$SEED" "$N_HOLDOUT"
done

echo "All index-build jobs submitted. Track with: squeue -u $USER"
