#!/bin/bash
# Fig 2 (index cost), in order. Inputs: the index cache (results/index_cache/builds.csv) and the ladders.
# Compute steps are sbatch (timed, one job at a time on ROSALIND-CPU1); wait for each array before the next step.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=~/miniforge3/envs/spindle_env/bin/python
PPTX="env PYTHONPATH=experiments/common/.pylib $PY"

if [[ "${1:-}" == "compute" ]]; then
  sbatch --array=0-10%1 experiments/common/timing.sbatch experiments/fig2_cost/c_ladder.py cells
  sbatch --array=0-5%1 --mem=740G experiments/common/timing.sbatch experiments/fig2_cost/c_ladder.py genes  # 250-1,500
  exit 0
fi

$PY experiments/fig2_cost/c_ladder_summary.py           # ladder/*.json -> results/fig2_cost/ladder.csv
$PY experiments/fig2_cost/write_numbers.py              # -> results/fig2_cost/numbers.json
$PY experiments/fig2_cost/f_datasets.py                 # atoms A-C
$PY experiments/fig2_cost/f_ladder.py                   # atoms D-E
$PPTX experiments/common/build_figure.py fig2_cost      # -> figures/fig2_cost/fig2_cost_draft.pptx
if [[ -f figures/fig2_cost/fig2_cost.pptx ]]; then
  $PPTX experiments/common/render.py fig2_cost          # the user's copy -> fig2_cost.pdf / .png
else
  $PPTX experiments/common/render.py fig2_cost --draft
fi
