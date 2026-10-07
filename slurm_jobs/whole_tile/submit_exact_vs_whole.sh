#!/bin/bash
# Fig 3A: one exact_vs_whole job per dataset x seed 0-4, one at a time (singleton on the shared
# timing job name, as Fig 2C), then the aggregate.
# Usage: bash slurm_jobs/whole_tile/submit_exact_vs_whole.sh [datasets...]   (default: the 8 indexed datasets)
cd "$(dirname "$0")/../.."
mkdir -p logs
SERIAL=(--job-name=spindle_timing --dependency=singleton)
DATASETS=${@:-skin_melanoma kidney_nondiseased breast_cancer lung_cancer pancreatic_cancer lymph_node lymph_node_5k brain_cancer}
ids=""
for d in $DATASETS; do
    case $d in
        lymph_node_5k|brain_cancer) mem=96G ;;
        *) mem=48G ;;
    esac
    for s in 0 1 2 3 4; do
        j=$(sbatch --parsable "${SERIAL[@]}" --mem=$mem slurm_jobs/whole_tile/run_exact_vs_whole.sbatch $d $s)
        echo "$d seed $s: $j"
        ids="$ids:$j"
    done
done
sbatch --parsable --job-name=spindle_evd_agg --mem=8G --time=00:30:00 --dependency=afterany$ids \
    slurm_jobs/whole_tile/run_exact_vs_whole.sbatch --aggregate
