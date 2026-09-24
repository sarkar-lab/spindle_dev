#!/bin/bash
#SBATCH --job-name=spindle_cross_modal
#SBATCH --partition=cpus
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# One seed of Stage C / E12: builds the Visium and Xenium indices at this
# seed and runs both cross-modal search directions (all-niche search), via
# benchmarks/cross_modal_search.py. Extra args are passed through, e.g.
# --single-niche-baseline, --write-overlay, --n-queries 5.
#
# Usage: sbatch run_cross_modal_search.sbatch <seed> [extra cross_modal_search.py args...]

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

echo "Activating Conda environment..."
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

if [ -z "$1" ]; then
    echo "Usage: sbatch run_cross_modal_search.sbatch <seed> [extra args...]"
    exit 1
fi
SEED=$1
shift

echo "============================================================"
echo "Cross-modal search: seed=$SEED extra_args=$*"
echo "============================================================"
python benchmarks/cross_modal_search.py --seed "$SEED" --direction both "$@"
echo "Done."
