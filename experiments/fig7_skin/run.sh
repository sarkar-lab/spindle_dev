#!/bin/bash
# Fig 7 (search on skin and on all eight sections), in order. 1 is sbatch (compute nodes, from the index cache); 2-5
# run on the login node after the jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=experiments/common
F=experiments/fig7_skin

# 1. whole-tile and gene-list queries with every index (8 datasets x seeds 0-4; task = 5 * dataset + seed)
sbatch --job-name=f7_query --array=0-39 $C/run.sbatch $F/c_queries.py

# after the jobs (login node, spindle_env python):
# 2. summaries              python $F/c_summary.py
# 3. atoms                  python $F/f_fig7.py
# 4. draft pptx             PYTHONPATH=$C/.pylib python $C/build_figure.py fig7_skin
#    then copy figures/fig7_skin/fig7_skin_draft.pptx to fig7_skin.pptx, edit, and
#    PYTHONPATH=$C/.pylib python $C/render.py fig7_skin
# 5. numbers                python $F/write_numbers.py   -> results/fig7_skin/numbers.json (caption, text)
