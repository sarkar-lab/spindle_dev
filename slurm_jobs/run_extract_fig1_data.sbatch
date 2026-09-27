#!/bin/bash
#SBATCH --job-name=spindle_extract_fig1_data
#SBATCH --partition=cpus
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Writes the breast inputs of the Fig. 1 sub-panels to results/figure_data/fig1/
# (see scripts/extract_fig1_data.py). scripts/fig1_subpanels.py reads CSVs only.

mkdir -p logs

echo "Activating Conda environment..."
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

python scripts/extract_fig1_data.py "$@" || exit 1
echo "Done."
