#!/bin/bash
#SBATCH --job-name=spindle_extract_figure_data
#SBATCH --partition=cpus
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Writes the pickle/h5ad-derived figure inputs to results/figure_data/
# (see scripts/extract_figure_data.py).
# The figure scripts themselves (scripts/fig*.py) read CSVs only.

mkdir -p logs

echo "Activating Conda environment..."
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

python scripts/extract_figure_data.py "$@" || exit 1
echo "Done."
