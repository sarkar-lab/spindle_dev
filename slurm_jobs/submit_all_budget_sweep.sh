#!/bin/bash
# Submits one independent SLURM job per dataset for the recall-vs-budget
# sweep, so all 8 datasets run in parallel across the cluster instead of
# sequentially in a single job. Run this directly on the login node (it just
# calls `sbatch` per dataset and exits -- it is not itself a SLURM job).
#
# Usage:
#   ./slurm_jobs/submit_all_budget_sweep.sh [seed] [n_holdout]
#
# Two datasets (lymph_node_5k, brain_cancer) have far larger raw-covariance
# pickles (~20GB / ~10GB) than the rest (<2GB) -- they get bumped-up
# --mem/--time/--partition overrides at submit time rather than a one-size
# job spec for all 8.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

SEED=${1:-1}
N_HOLDOUT=${2:-100}

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

echo "Submitting standard-size jobs (seed=$SEED, n_holdout=$N_HOLDOUT)..."
for DATASET_NAME in "${DATASETS[@]}"; do
    echo "  Submitting $DATASET_NAME..."
    sbatch slurm_jobs/run_single_budget_sweep.sbatch "$DATASET_NAME" "$SEED" "$N_HOLDOUT"
done

echo "Submitting large-dataset jobs with bumped resources (mem=256G, time=24:00:00, partition=batch)..."
for DATASET_NAME in "${LARGE_DATASETS[@]}"; do
    echo "  Submitting $DATASET_NAME..."
    sbatch --mem=256G --time=24:00:00 --partition=batch \
        slurm_jobs/run_single_budget_sweep.sbatch "$DATASET_NAME" "$SEED" "$N_HOLDOUT"
done

echo "All jobs submitted. Track with: squeue -u $USER"
