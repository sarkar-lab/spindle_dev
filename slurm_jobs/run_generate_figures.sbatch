#!/bin/bash
#SBATCH --job-name=spindle_budget_sweep_figures
#SBATCH --partition=cpus
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Renders the budget-sweep figures (scripts/generate_budget_sweep_figure.py)
# from whatever results/budget_sweep_holdout/sweep_summary_{block,whole}.csv
# currently contain -- cheap (CSV + matplotlib only), but run as its own
# SLURM job (typically --dependency=afterany:<sweep job ids>) so it can run
# unattended after all budget-sweep jobs finish, without needing an
# interactive session to trigger it.

mkdir -p logs

echo "Activating Conda environment..."
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

echo "============================================================"
echo "Generating budget-sweep figures"
echo "============================================================"
python scripts/generate_budget_sweep_figure.py
echo "Done."
