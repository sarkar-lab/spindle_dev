#!/bin/bash
# Submits one stage-benchmark job per dataset (benchmarks/stage_benchmark.py). Run on the
# login node; it only calls sbatch. Afterwards (cheap, login node):
#   python benchmarks/stage_benchmark.py --aggregate
#   python scripts/stage_benchmark_plots.py
#
# Usage: ./slurm_jobs/submit_stage_benchmark.sh [extra args passed to every job]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."
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

for DATASET_NAME in "${DATASETS[@]}"; do
    sbatch slurm_jobs/run_stage_benchmark.sbatch "$DATASET_NAME" "$@"
done
for DATASET_NAME in "${LARGE_DATASETS[@]}"; do
    sbatch --mem=256G --time=24:00:00 --partition=batch \
        slurm_jobs/run_stage_benchmark.sbatch "$DATASET_NAME" "$@"
done

echo "All jobs submitted. Track with: squeue -u $USER"
