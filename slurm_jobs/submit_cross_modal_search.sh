#!/bin/bash
# Stage C / E12: submit one cross-modal search job per seed (0-4).
# Seed 0 also runs the single-niche-routing diagnostic and rewrites the
# seed-independent figure-panel overlay CSVs.
# Afterwards (login node): python benchmarks/multiseed_cross_modal_search.py
set -euo pipefail
cd "$(dirname "$0")/.."

for SEED in 0 1 2 3 4; do
    EXTRA=""
    if [ "$SEED" -eq 0 ]; then
        EXTRA="--single-niche-baseline --write-overlay"
    fi
    sbatch slurm_jobs/run_cross_modal_search.sbatch "$SEED" $EXTRA
done
