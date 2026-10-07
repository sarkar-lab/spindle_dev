#!/bin/bash
# Fig 2E gene ladder on lymph node 5K, all tiles, streaming builder, niche cap 1000:
# G = 250, 400 (the production lymph node 5K index), 500, 750, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4624 (the full panel), one job per G,
# then an aggregate job. Each job caches its ultrametric features node-locally under $SPINDLE_LOCAL_CACHE (default /tmp/spindle_cache_$USER; up to
# ~300 GB on disk at G = 4624) and removes them at the end.
# Usage: ./slurm_jobs/submit_dag_gene_ladder.sh [G ...]   (run on the login node; default: the grid)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
mkdir -p logs
# Build times are reported: run one timing job at a time (Slurm singleton on a shared job name), so
# our own jobs never compete for the node.
SERIAL=(--job-name=spindle_timing --dependency=singleton)
RUN=slurm_jobs/index/run_dag_gene_ladder.sbatch
GRID=${@:-250 400 500 750 1000 1500 2000 2500 3000 3500 4000 4624}
JOBS=()
for G in $GRID; do
    if   [ "$G" -le 1000 ]; then MEM=64G
    elif [ "$G" -le 2500 ]; then MEM=160G
    else MEM=360G; fi
    JOBS+=($(sbatch --parsable "${SERIAL[@]}" --mem=$MEM --time=4-00:00:00 "$RUN" "$G"))
done
DEP=$(IFS=:; echo "${JOBS[*]}")
AGG=$(sbatch --parsable --dependency=afterany:$DEP --cpus-per-task=2 --mem=8G --time=00:30:00 "$RUN" --aggregate)
echo "Submitted ${#JOBS[@]} jobs; aggregate $AGG. Track with: squeue -u $USER"
