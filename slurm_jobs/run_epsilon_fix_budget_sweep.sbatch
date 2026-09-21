#!/bin/bash
#SBATCH --job-name=spindle_epsfix_budget_sweep
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

# Fine-grained budget_multiplier sweep for the two datasets rebuilt with the
# eps_hat clustering fix (see /home/NAShome/ghoss18/.claude/plans -- the fix
# that replaced the loose "elbow" epsilon with the tight, k_target_per_block-
# calibrated one for per-block clustering). A quick manual spot-check showed
# recall/overlap is still ~0 at budget_multiplier=0.015-0.05 (the old
# production value is no longer a realistic operating point post-fix), rising
# by 0.1 and saturating to ~95-100% candidate volume by budget_multiplier=0.5.
# This sweep uses a simple, readable step sequence concentrated in the
# 0.05-2.0 rising/saturating region (fine 0.01-0.02 steps from 0.05-0.2,
# coarsening to 0.05/0.1/0.25/1.0-sized steps as the curve flattens out)
# plus a few sparse points (3/4/8/16) to confirm the high-budget plateau
# holds, instead of the old DEFAULT_BUDGET_MULTS grid (30 log-spaced points,
# 0.01-32) or wasting points below 0.05 where nothing is found at all.
#
# Plateau early-stop is left ON (default): once --stop-metric (recall_at_eps_0.1)
# has been flat for 3 consecutive budget multipliers, the sweep stops for that
# dataset. This was disabled for the first kidney/breast_cancer runs because
# the grid then started at 0.01, where recall was still genuinely at 0 (not
# saturated) and the plateau detector couldn't tell the difference. Starting
# the grid at 0.05 (a realistic post-fix floor, already showing partial
# signal) avoids that false trigger, so the default early-stop is safe again
# and saves compute once the curve genuinely saturates.
#
# Reuses the already-rebuilt index/covariance pickles under
# results/holdout_validation_indexed/ for kidney_nondiseased and
# breast_cancer -- no re-indexing here.
#
# Usage: sbatch run_epsilon_fix_budget_sweep.sbatch <dataset.h5ad> [stop_metric] [max_queries]

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
    echo "Usage: sbatch run_epsilon_fix_budget_sweep.sbatch <dataset.h5ad> [stop_metric] [max_queries] [num_workers]"
    exit 1
fi
DATASET_NAME=$1
STOP_METRIC=${2:-recall_at_eps_0.1}
MAX_QUERIES=${3:-100}
# Thread pool size for the per-query loops (ground-truth computation, DAG
# search, Stage-2 re-rank). Measured (A/B test, kidney dataset, 20 queries):
# threading makes this WORSE, not better -- num_workers=1 wall time 7.95s,
# num_workers=4 12.76s, num_workers=16 15.59s, with the *aggregate*
# per-query time ballooning from 6.6s to 45s to 198s. The DFS search
# (search.search_index) is dominated by pure-Python object-graph traversal
# (list/set/dict operations over DAG nodes) interleaved with many small
# numpy calls on tiny per-block matrices -- none of that releases the GIL
# for long enough to pay for thread-switching overhead, so more threads
# just means more GIL contention. Ground-truth computation showed no
# benefit either (flat to slightly worse). Left at 1 (serial) by default;
# do not raise this without re-measuring on the actual workload first.
NUM_WORKERS=${4:-1}

DATASET_PATH="$DATASET_DIR/$DATASET_NAME"

BUDGET_MULTS="0.05 0.06 0.07 0.08 0.09 0.10 0.12 0.14 0.16 0.18 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70 0.80 0.90 1.00 1.25 1.50 2.00 3.00 4.00 8.00 16.00"

echo "============================================================"
echo "Epsilon-fix budget sweep: $(basename "$DATASET_PATH") (stop-metric=$STOP_METRIC, max-queries=$MAX_QUERIES)"
echo "Budget multipliers (28 points): $BUDGET_MULTS"
echo "============================================================"

MAX_QUERIES_ARGS=()
if [ -n "$MAX_QUERIES" ]; then
    MAX_QUERIES_ARGS=(--max-queries "$MAX_QUERIES")
fi

python benchmarks/budget_sweep_holdout.py \
    --dataset-paths "$DATASET_PATH" \
    --stop-metric "$STOP_METRIC" \
    --budget-mults $BUDGET_MULTS \
    --num-workers "$NUM_WORKERS" \
    "${MAX_QUERIES_ARGS[@]}"

echo "Done."
