#!/bin/bash
#SBATCH --job-name=spindle_rebuild_capped
#SBATCH --partition=cpus
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=logs/%x_%j.out

# Rebuild one saved index in place. The old index / covariances / interval index are moved to the
# backup folder first; after the build the train/test split is checked against the backup and the
# niche sizes against the 1,000-tile cap.
# Used for: Part 1.3 (brain indexes over the cap; backup results/indexes/_precap_backup) and the Fig 3
# session (lymph_node_5k at max_genes 400 from datasets.yaml; backup results/indexes/_archive_ln5k_g800).
# Usage: sbatch slurm_jobs/run_rebuild_capped_index.sbatch <stem> <seed> [backup dir]
#   seed 73: production split (100 held-out tiles); seeds 0-4: 10% held out (as holdout_search.py).

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR
set -euo pipefail

STEM=$1
SEED=$2
DATASET_PATH=$(python benchmarks/paths.py "$STEM")  # datasets.yaml
if [ "$SEED" = "73" ]; then TAG=$STEM; else TAG=${STEM}_seed${SEED}; fi
BACKUP=${3:-results/indexes/_precap_backup}
mkdir -p $BACKUP
for f in results/indexes/${TAG}_spindle_index.pkl results/indexes/${TAG}_raw_covariances.pkl results/indexes/${TAG}_interval_index.pkl; do
    if [ -e "$f" ] && [ ! -e "$BACKUP/$(basename $f)" ]; then mv "$f" $BACKUP/; fi
done

if [ "$SEED" = "73" ]; then
    python benchmarks/build_indexes.py --dataset-paths "$DATASET_PATH" --seed 73 --n-holdout 100
else
    LINK=$(python -c "import sys; sys.path.insert(0, 'benchmarks'); from pathlib import Path; import holdout_search as h; print(h.make_seed_symlink(Path('$DATASET_PATH'), $SEED))")
    python benchmarks/build_indexes.py --dataset-paths "$LINK" --seed "$SEED" --train-test-ratio 0.10
fi

python - <<EOF
import pickle, sys, numpy as np
sys.path.insert(0, "src")
new = pickle.load(open("results/indexes/${TAG}_raw_covariances.pkl", "rb"))
old = pickle.load(open("$BACKUP/${TAG}_raw_covariances.pkl", "rb"))
same = all(np.array_equal(np.asarray(new[k]), np.asarray(old[k])) for k in ("train_idx", "test_idx"))
labels = np.asarray(pickle.load(open("results/indexes/${TAG}_spindle_index.pkl", "rb"))["data"].labels).astype(int)
sizes = np.bincount(labels)
print(f"${TAG}: split identical to the old build: {same}; niches {len(sizes)}, max {sizes.max()}, sizes {sorted(sizes.tolist(), reverse=True)}")
assert same and sizes.max() <= 1000
EOF
echo "Done (exit $?)."
