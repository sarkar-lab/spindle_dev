#!/bin/bash
# Paper plan 02, E7: noisy-query sweep on every dataset's seed-73 production
# build, both noise models. Small datasets: one job per (dataset, model).
# brain_cancer and lymph_node_5k (~10 s per query) get one job per noise level
# (level 0 = clean, run once) so each stays well inside the partition limit.
# Run on the login node; it only calls sbatch.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
SMALL=(skin_melanoma kidney_nondiseased breast_cancer lung_cancer pancreatic_cancer lymph_node)
declare -A LEVELS=([spectral]="0.05 0.1 0.2 0.3 0.5" [cell_subsample]="0.9 0.75 0.5")
for MODEL in spectral cell_subsample; do
    for DS in "${SMALL[@]}"; do
        sbatch slurm_jobs/run_noise_robustness.sbatch "$DS" "$MODEL"
    done
    for DS in brain_cancer lymph_node_5k; do
        MEM=$([ "$DS" = lymph_node_5k ] && echo 96G || echo 64G)
        LV="${LEVELS[$MODEL]}"
        [ "$MODEL" = spectral ] && LV="0 $LV"   # clean run once, with the spectral jobs
        for L in $LV; do
            sbatch --mem=$MEM slurm_jobs/run_noise_robustness.sbatch "$DS" "$MODEL" "$L"
        done
    done
done
