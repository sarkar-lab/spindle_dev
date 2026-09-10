#!/bin/bash
#SBATCH --job-name=spindle_budget_sweep_seq
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Single job running all datasets sequentially -- used instead of one job per
# dataset when the scheduler's fair-share priority makes many small jobs
# queue for a very long time (each additional pending job competes for its
# own slot); one job holding one slot for longer is easier to schedule.

# Limit threading to prevent overloading the node
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

echo "Activating Conda environment..."
# Guard against a stray VIRTUAL_ENV/PATH leaking in from the submitting shell
# (e.g. a local .venv activated in an editor session) shadowing the conda env.
unset VIRTUAL_ENV PYTHONHOME
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env

cd $SLURM_SUBMIT_DIR

DATASET_DIR="/home/NAShome/sarkah1/shared_data/insitupy_demo_data_xenium"
TRAIN_TEST_RATIO=${1:-0.05}
STOP_METRIC=${2:-overlap_at_50}
MAX_QUERIES=${3:-100}

MAX_QUERIES_ARGS=()
if [ -n "$MAX_QUERIES" ]; then
    MAX_QUERIES_ARGS=(--max-queries "$MAX_QUERIES")
fi

for DATASET_PATH in "$DATASET_DIR"/*.h5ad; do
    DATASET_NAME=$(basename "$DATASET_PATH")
    echo "============================================================"
    echo "Dataset: $DATASET_NAME"
    echo "============================================================"
    echo "Indexing (train-test-ratio=$TRAIN_TEST_RATIO)..."
    python benchmarks/index_datasets.py --dataset-paths "$DATASET_PATH" --train-test-ratio "$TRAIN_TEST_RATIO"
    echo "Budget sweep..."
    python benchmarks/budget_sweep_holdout.py --dataset-paths "$DATASET_PATH" --stop-metric "$STOP_METRIC" "${MAX_QUERIES_ARGS[@]}"
done

echo "All datasets processed successfully!"
