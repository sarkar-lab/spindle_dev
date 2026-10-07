#!/bin/bash
#SBATCH --job-name=spindle_breast_concordance
#SBATCH --partition=cpus
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out

# Paper plan 02: E10-lite (niche vs cell-type ARI/NMI) then E13 (composition
# JSD of retrieved tiles), both on breast, reusing the existing index builds
# (seed 73 + seeds 0-4). E13 computes and caches the block ground truth for
# seeds 0-4 on first run.
#
# Usage: sbatch slurm_jobs/biology/run_breast_concordance.sbatch [e10|e13|both]

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

WHAT=${1:-both}
set -e
if [ "$WHAT" = "e10" ] || [ "$WHAT" = "both" ]; then
    python benchmarks/biology/niche_concordance.py
fi
if [ "$WHAT" = "e13" ] || [ "$WHAT" = "both" ]; then
    python benchmarks/biology/composition_concordance.py
fi
echo "Done."
