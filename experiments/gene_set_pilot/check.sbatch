#!/bin/bash
# examples/check_indexes.py on one indexed section per array task (datasets.yaml order).
#SBATCH --job-name=check_idx
#SBATCH --partition=cpus
#SBATCH --array=0-7
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=/home/NAShome/ghoss18/spindle_dev/results/gene_set_pilot/logs/%x_%A_%a.out
set -euo pipefail
cd /home/NAShome/ghoss18/spindle_dev
unset VIRTUAL_ENV PYTHONHOME
source ~/miniforge3/etc/profile.d/conda.sh
conda activate spindle_env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
KEY=$(python -c "import paths; print(paths.indexed_keys()[$SLURM_ARRAY_TASK_ID])")
echo "$KEY on $(hostname), $(git describe --always --dirty --tags)"
python -u examples/check_indexes.py "$KEY"
