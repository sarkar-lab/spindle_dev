#!/bin/bash
#SBATCH --job-name=spindle_make_figures
#SBATCH --partition=cpus
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Draws every manuscript figure from the CSVs in results/ (about a minute in
# total; also fine to run directly with `bash slurm_jobs/run_make_figures.sbatch`).
# Writes figures/<name>.pdf (exact printed size) and one PDF per panel in
# figures/panels/<name>/. Copy the PDFs you need into the Overleaf project by hand.
#
# Usage: sbatch slurm_jobs/run_make_figures.sbatch [script ...]   (default: all)

mkdir -p logs

unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

if [ $# -gt 0 ]; then
    SCRIPTS=("$@")
else
    SCRIPTS=(scripts/fig[1-7]_*.py scripts/figS[0-9]*_*.py)
fi

for s in "${SCRIPTS[@]}"; do
    echo "== $s"
    python "$s" || exit 1
done
echo "Done."
