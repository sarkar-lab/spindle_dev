#!/bin/bash
# Fig 3D: one whole_cov_baselines job per dataset x seed 0-4, then the aggregate.
# Usage: bash slurm_jobs/submit_whole_cov_baselines.sh [datasets...]   (default: the 8 indexed datasets)
cd "$(dirname "$0")/.."
mkdir -p logs
DATASETS=${@:-skin_melanoma kidney_nondiseased breast_cancer lung_cancer pancreatic_cancer lymph_node lymph_node_5k brain_cancer}
ids=""
for d in $DATASETS; do
    case $d in
        lymph_node_5k|brain_cancer) mem=160G ;;
        *) mem=64G ;;
    esac
    for s in 0 1 2 3 4; do
        j=$(sbatch --parsable --mem=$mem slurm_jobs/run_whole_cov_baselines.sbatch $d $s)
        echo "$d seed $s: $j"
        ids="$ids:$j"
    done
done
sbatch --parsable --mem=16G --time=00:30:00 --dependency=afterany$ids slurm_jobs/run_whole_cov_baselines.sbatch --aggregate
