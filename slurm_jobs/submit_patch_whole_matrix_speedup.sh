#!/bin/bash
# Submits one SLURM job per already-completed multiseed_holdout dataset x
# seed unit to patch its bf_time_ms/speedup columns to use the whole-matrix
# ground truth instead of the block-diagonalized one (see
# benchmarks/patch_whole_matrix_speedup.py). Run this directly on the login
# node (it just calls `sbatch` per unit and exits).
#
# Usage:
#   ./slurm_jobs/submit_patch_whole_matrix_speedup.sh <dataset1.h5ad> [dataset2.h5ad ...] -- <seed1> [seed2 ...]
#
# Example (all 7 non-brain_cancer datasets, all 5 seeds):
#   ./slurm_jobs/submit_patch_whole_matrix_speedup.sh \
#       xenium_human_breast_cancer.h5ad xenium_human_kidney_nondiseased.h5ad \
#       xenium_human_lung_cancer.h5ad xenium_human_lymph_node.h5ad \
#       xenium_human_pancreatic_cancer.h5ad xenium_human_skin_melanoma.h5ad \
#       xenium_human_lymph_node_5k.h5ad -- 0 1 2 3 4
#
# Example (brain_cancer, only the seeds already finished under the old code):
#   ./slurm_jobs/submit_patch_whole_matrix_speedup.sh xenium_human_brain_cancer.h5ad -- 0 1 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

DATASETS=()
SEEDS=()
PARSING_SEEDS=false
for ARG in "$@"; do
    if [ "$ARG" == "--" ]; then
        PARSING_SEEDS=true
        continue
    fi
    if [ "$PARSING_SEEDS" == true ]; then
        SEEDS+=("$ARG")
    else
        DATASETS+=("$ARG")
    fi
done

if [ "${#DATASETS[@]}" -eq 0 ] || [ "${#SEEDS[@]}" -eq 0 ]; then
    echo "Usage: $0 <dataset1.h5ad> [dataset2.h5ad ...] -- <seed1> [seed2 ...]"
    exit 1
fi

mkdir -p logs

echo "Submitting patch jobs: datasets=${DATASETS[*]}, seeds=${SEEDS[*]}..."
for DATASET_NAME in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "  Submitting $DATASET_NAME seed=$SEED..."
        sbatch slurm_jobs/run_patch_whole_matrix_speedup.sbatch "$DATASET_NAME" "$SEED"
    done
done

echo "All patch jobs submitted. Track with: squeue -u $USER"
