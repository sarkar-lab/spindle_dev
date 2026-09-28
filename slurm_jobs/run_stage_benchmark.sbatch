#!/bin/bash
#SBATCH --job-name=spindle_stage_benchmark
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Stage 1 (DAG top-c) vs Stage 1 + Stage 2 (exact re-rank of the top-c from stored block logs)
# vs Stage 2 only (every tile's raw diagonal blocks logged per query) on one dataset's seed-73
# production index (benchmarks/stage_benchmark.py) -> results/stage_benchmark/<dataset>/.
#
# Usage: sbatch run_stage_benchmark.sbatch <dataset.h5ad> [--max-queries N] [--stage2-only-queries 10]
# Resource overrides for large datasets (lymph_node_5k, brain_cancer) are
# passed via `sbatch --mem=... --time=... --partition=...` at submit time.

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
if [ -z "$1" ]; then
    echo "Usage: sbatch run_stage_benchmark.sbatch <dataset.h5ad> [extra args]"
    exit 1
fi
DATASET_PATH="$DATASET_DIR/$1"
shift

echo "Stage benchmark: $(basename "$DATASET_PATH") $*"
python benchmarks/stage_benchmark.py --dataset-path "$DATASET_PATH" "$@"
echo "Done."
