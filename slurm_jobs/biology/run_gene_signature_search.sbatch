#!/bin/bash
#SBATCH --job-name=spindle_gene_signature_search
#SBATCH --partition=cpus
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out

# Paper plan 02, E9-lite: breast gene-signature queries with all-niche search
# on the existing seed 0-4 index builds (benchmarks/biology/gene_signature_search.py).
# Usage: sbatch slurm_jobs/biology/run_gene_signature_search.sbatch [--seeds ...]

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

python benchmarks/biology/gene_signature_search.py "$@"
echo "Done."
