#!/bin/bash
# Submits one independent SLURM job per dataset to (re)build the Spindle
# index with a fixed seed / holdout-tile-count split, so datasets build in
# parallel across the cluster instead of sequentially. Run this directly on
# the login node (it just calls `sbatch` per dataset and exits -- it is not
# itself a SLURM job).
#
# Usage:
#   ./slurm_jobs/submit_build_indexes.sh [seed] [n_holdout] [train_test_ratio]
#
# Defaults reproduce the production build used by every seed-73 result
# (seed 73, 100 held-out tiles). Brain and lymph_node_5k need more memory
# (peak RSS 55 GB and 98 GB), so they get their own resource flags below.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SEED=${1:-73}
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
    sbatch slurm_jobs/run_build_index.sbatch "$DATASET_NAME" "$TRAIN_TEST_RATIO" "$SEED" "$N_HOLDOUT"
done
sbatch --mem=160G slurm_jobs/run_build_index.sbatch xenium_human_brain_cancer.h5ad "$TRAIN_TEST_RATIO" "$SEED" "$N_HOLDOUT"
sbatch --mem=220G --partition=batch slurm_jobs/run_build_index.sbatch xenium_human_lymph_node_5k.h5ad "$TRAIN_TEST_RATIO" "$SEED" "$N_HOLDOUT"

echo "All index-build jobs submitted. Track with: squeue -u $USER"
