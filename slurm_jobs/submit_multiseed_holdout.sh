#!/bin/bash
# Submits one independent SLURM job per dataset x seed for Stage A / E5
# (multi-seed holdout-validation reruns), so all 8 datasets x 5 seeds (40
# jobs) run in parallel across the cluster instead of sequentially. Run this
# directly on the login node (it just calls `sbatch` per unit and exits --
# it is not itself a SLURM job).
#
# Usage:
#   ./slurm_jobs/submit_multiseed_holdout.sh [train_test_ratio]
#
# Seeds are fixed at 0-4 (E5's specified seed set). Two datasets
# (lymph_node_5k, brain_cancer) have far larger raw-covariance pickles than
# the rest -- they get bumped-up --mem/--time/--partition overrides at
# submit time rather than a one-size job spec for all 8.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

TRAIN_TEST_RATIO=${1:-0.10}
SEEDS=(0 1 2 3 4)

mkdir -p logs

DATASETS=(
    xenium_human_breast_cancer.h5ad
    xenium_human_kidney_nondiseased.h5ad
    xenium_human_lung_cancer.h5ad
    xenium_human_lymph_node.h5ad
    xenium_human_pancreatic_cancer.h5ad
    xenium_human_skin_melanoma.h5ad
)
LARGE_DATASETS=(
    xenium_human_lymph_node_5k.h5ad
    xenium_human_brain_cancer.h5ad
)

echo "Submitting standard-size jobs (train_test_ratio=$TRAIN_TEST_RATIO, seeds=${SEEDS[*]})..."
for DATASET_NAME in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  Submitting $DATASET_NAME seed=$SEED..."
        sbatch slurm_jobs/run_multiseed_holdout.sbatch "$DATASET_NAME" "$SEED" "$TRAIN_TEST_RATIO"
    done
done

echo "Submitting large-dataset jobs with bumped resources (mem=256G, time=24:00:00, partition=batch)..."
for DATASET_NAME in "${LARGE_DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  Submitting $DATASET_NAME seed=$SEED..."
        sbatch --mem=256G --time=24:00:00 --partition=batch \
            slurm_jobs/run_multiseed_holdout.sbatch "$DATASET_NAME" "$SEED" "$TRAIN_TEST_RATIO"
    done
done

echo "All jobs submitted. Track with: squeue -u $USER"
