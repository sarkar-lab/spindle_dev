#!/bin/bash
# Fig 5 / S9: tile caches + overlay, then seeds 0-4 x tile sizes 2000 (main) and 1000 (S9), then the
# aggregate, bias PCA and the Fig 5E example.
# Usage: bash slurm_jobs/submit_cross_platform_tiers.sh
cd "$(dirname "$0")/.."
mkdir -p logs
R=slurm_jobs/run_cross_platform_tiers.sbatch
ids=""
for m in 2000 1000; do
    o=$(sbatch --parsable $R --overlay --max-pts $m)
    echo "overlay $m: $o"
    for s in 0 1 2 3 4; do
        j=$(sbatch --parsable --dependency=afterok:$o $R $s $m)
        echo "tiles $m seed $s: $j"
        ids="$ids:$j"
    done
done
a=$(sbatch --parsable --mem=16G --time=00:30:00 --dependency=afterany$ids $R --aggregate)
echo "aggregate: $a"
echo "bias PCA: $(sbatch --parsable --dependency=afterany$ids $R --bias-pca)"
echo "example: $(sbatch --parsable --mem=8G --time=00:15:00 --dependency=afterany$ids $R --example)"
