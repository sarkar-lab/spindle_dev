#!/bin/bash
#SBATCH --job-name=spindle_partial_search
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --output=logs/partial_search_final_%j.out

# Fig 4 / S8: partial (gene-set) search -- interval index, padding, imputation vs the exact partial tier
# (benchmarks/partial/partial_search_final.py). Submit through submit_partial_search_final.sh.
# Usage: sbatch [--mem=...] [--cpus-per-task=1 --job-name=spindle_timing --dependency=singleton] \
#          slurm_jobs/partial/run_partial_search_final.sbatch <dataset> <seed> [--timing | --schematic | --max-queries N]
#        sbatch slurm_jobs/partial/run_partial_search_final.sbatch --aggregate

T=${SLURM_CPUS_PER_TASK:-2}
export OMP_NUM_THREADS=$T MKL_NUM_THREADS=$T OPENBLAS_NUM_THREADS=$T NUMEXPR_NUM_THREADS=$T
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR

if [ "$1" = "--aggregate" ]; then
    python benchmarks/partial/partial_search_final.py --aggregate
else
    python benchmarks/partial/partial_search_final.py --dataset "$1" --seed "$2" "${@:3}"
fi
status=$?
echo "Done (exit $status)."
exit $status
