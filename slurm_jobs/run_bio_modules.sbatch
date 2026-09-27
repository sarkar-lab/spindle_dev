#!/bin/bash
#SBATCH --job-name=spindle_bio_modules
#SBATCH --partition=cpus
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Gene modules, enrichment (panel + Enrichr genome background; needs network) and
# module-distance maps for Figs 6, 7, S10 -> results/biology/<dataset>/ (scripts/bio_modules.py).
# Pass dataset keys to run a subset; default is every scripts/bio_configs/*.yaml.

mkdir -p logs

echo "Activating Conda environment..."
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

python scripts/bio_modules.py "$@" || exit 1
echo "Done."
