#!/bin/bash
#SBATCH --job-name=spindle_check_fast_search
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Checks the rewritten spindle_dev.search.search_index against the committed original
# (benchmarks/check_fast_search.py): identical paths and leaf sets on every held-out query.
#
# Usage: sbatch run_check_fast_search.sbatch <dataset.h5ad> [more datasets ...] [--max-queries N] [--rev HEAD]

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

DATASET_DIR="/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium"
PATHS=()
while [ $# -gt 0 ] && [[ "$1" != --* ]]; do
    PATHS+=("$DATASET_DIR/$1")
    shift
done
if [ ${#PATHS[@]} -eq 0 ]; then
    echo "Usage: sbatch run_check_fast_search.sbatch <dataset.h5ad> [...] [extra args]"
    exit 1
fi
python benchmarks/check_fast_search.py --dataset-paths "${PATHS[@]}" "$@"
echo "Done."
