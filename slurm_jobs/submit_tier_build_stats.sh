#!/bin/bash
# Fig 2C: one tier_build_stats job per dataset x seed (0-4), then the aggregate.
# Usage: [DATASETS="a b"] bash slurm_jobs/submit_tier_build_stats.sh [seeds...]   (default: all 8; seeds 0 1 2 3 4)
cd "$(dirname "$0")/.."
mkdir -p logs
# Build times are reported: run one timing job at a time (Slurm singleton on a shared job name), so
# our own jobs never compete for the node.
SERIAL=(--job-name=spindle_timing --dependency=singleton)
SEEDS=${@:-0 1 2 3 4}
DATASETS=${DATASETS:-skin_melanoma kidney_nondiseased breast_cancer lung_cancer pancreatic_cancer lymph_node lymph_node_5k brain_cancer}
ids=""
for d in $DATASETS; do
    case $d in
        lymph_node_5k) mem=192G ;;
        brain_cancer) mem=128G ;;
        *) mem=48G ;;
    esac
    for s in $SEEDS; do
        j=$(sbatch --parsable "${SERIAL[@]}" --mem=$mem slurm_jobs/run_tier_build_stats.sbatch $d $s)
        echo "$d seed $s: $j"
        ids="$ids:$j"
    done
done
sbatch --parsable --mem=8G --time=00:30:00 --dependency=afterany$ids slurm_jobs/run_tier_build_stats.sbatch --aggregate
