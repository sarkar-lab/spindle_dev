#!/bin/bash
# Fig 4 / S8: partial search per dataset x seed 0-4. Accuracy runs go in parallel; timing runs one at a
# time (singleton on the shared timing job name, one BLAS thread, as Figs 2C and 3A); then the aggregate.
# Usage: bash slurm_jobs/partial/submit_partial_search_final.sh [accuracy|timing|all] [datasets...]
cd "$(dirname "$0")/../.."
mkdir -p logs
MODE=${1:-all}
shift
DATASETS=${@:-skin_melanoma kidney_nondiseased breast_cancer lung_cancer pancreatic_cancer lymph_node lymph_node_5k brain_cancer}
ids=""
for d in $DATASETS; do
    case $d in
        lymph_node_5k|brain_cancer) mem=128G ;;
        *) mem=48G ;;
    esac
    for s in 0 1 2 3 4; do
        if [ "$MODE" != timing ]; then
            j=$(sbatch --parsable --mem=$mem slurm_jobs/partial/run_partial_search_final.sbatch $d $s)
            echo "$d seed $s accuracy: $j"; ids="$ids:$j"
        fi
        if [ "$MODE" != accuracy ]; then
            j=$(sbatch --parsable --job-name=spindle_timing --dependency=singleton --cpus-per-task=1 --mem=$mem \
                slurm_jobs/partial/run_partial_search_final.sbatch $d $s --timing)
            echo "$d seed $s timing: $j"; ids="$ids:$j"
        fi
    done
done
sbatch --parsable --job-name=spindle_psf_agg --mem=16G --time=01:00:00 --dependency=afterany$ids \
    slurm_jobs/partial/run_partial_search_final.sbatch --aggregate
