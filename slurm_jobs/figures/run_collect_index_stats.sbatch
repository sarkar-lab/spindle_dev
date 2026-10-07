#!/bin/bash
#SBATCH --job-name=spindle_collect_index_stats
#SBATCH --partition=cpus
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Collects per-dataset index statistics (niches, blocks, tiles, build time,
# peak RSS, index size, compression ratio) into results/index_stats/ by
# loading every index pickle -- see scripts/figure_data/collect_index_stats.py.
# Values that came from the deleted index-build SLURM logs are read from
# results/index_stats/log_index_stats.csv; run logs from build_run_logs/

mkdir -p logs

echo "Activating Conda environment..."
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

echo "============================================================"
echo "Collecting index statistics"
echo "============================================================"
python scripts/figure_data/collect_index_stats.py
echo "Done."
