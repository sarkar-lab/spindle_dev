#!/bin/bash
# Paper plan 02, E4: one FAISS-baseline job per dataset (seed-73 production
# build). Breast and lung also run the HNSW M x efSearch sensitivity sweep.
# The two large datasets load ~20-40 GB of cached covariances/ground truth and
# get more memory and the long partition. Run on the login node; it only calls sbatch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p logs
for DS in skin_melanoma kidney_nondiseased pancreatic_cancer lymph_node; do
    sbatch slurm_jobs/legacy/run_ann_baselines.sbatch "$DS"
done
for DS in breast_cancer lung_cancer; do
    sbatch slurm_jobs/legacy/run_ann_baselines.sbatch "$DS" --sensitivity
done
sbatch --mem=160G --time=24:00:00 --partition=batch slurm_jobs/legacy/run_ann_baselines.sbatch brain_cancer
sbatch --mem=220G --time=24:00:00 --partition=batch slurm_jobs/legacy/run_ann_baselines.sbatch lymph_node_5k
