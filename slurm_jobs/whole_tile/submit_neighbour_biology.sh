#!/bin/bash
# Fig 3E: one neighbour_biology job per dataset x seed 0-4 (after whole_cov_baselines), then the aggregate.
# Usage: bash slurm_jobs/whole_tile/submit_neighbour_biology.sh [datasets...]   (default: the 8 indexed datasets)
cd "$(dirname "$0")/../.."
mkdir -p logs
DATASETS=${@:-skin_melanoma kidney_nondiseased breast_cancer lung_cancer pancreatic_cancer lymph_node lymph_node_5k brain_cancer}
ids=""
for d in $DATASETS; do
    case $d in
        lymph_node_5k|brain_cancer) mem=128G ;;
        *) mem=48G ;;
    esac
    for s in 0 1 2 3 4; do
        j=$(sbatch --parsable --mem=$mem slurm_jobs/whole_tile/run_neighbour_biology.sbatch $d $s)
        echo "$d seed $s: $j"
        ids="$ids:$j"
    done
done
sbatch --parsable --mem=16G --time=00:30:00 --dependency=afterany$ids slurm_jobs/whole_tile/run_neighbour_biology.sbatch --aggregate
