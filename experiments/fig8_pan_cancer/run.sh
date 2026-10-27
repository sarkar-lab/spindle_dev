#!/bin/bash
# Fig 8 (pan-cancer programme queries), in order. 1 is sbatch (compute nodes, from the index cache); 2-5 run on the
# login node after the jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=experiments/common
F=experiments/fig8_pan_cancer

# 1. exemplars (breast seed 0), then the queries (5 tumour sections x seeds 0-4; task = 5 * section + seed)
j=$(sbatch --parsable --job-name=f8_ex $C/run.sbatch $F/c_xtumour.py exemplar)
sbatch --job-name=f8_x --dependency=afterok:$j --array=0-24 $C/run.sbatch $F/c_xtumour.py
# 1b. seed 0: every index tile's distance to each query (gradient maps, B)
sbatch --job-name=f8_grad --dependency=afterok:$j --array=0-4 $C/run.sbatch $F/c_gradient.py

# after the jobs (login node, spindle_env python):
# 2. summary                python $F/c_summary.py
# 3. atoms                  python $F/f_fig8.py
# 4. draft pptx             PYTHONPATH=$C/.pylib python $C/build_figure.py fig8_pan_cancer
#    then copy figures/fig8_pan_cancer/fig8_pan_cancer_draft.pptx to fig8_pan_cancer.pptx, edit, and
#    PYTHONPATH=$C/.pylib python $C/render.py fig8_pan_cancer
# 5. numbers                python $F/write_numbers.py   -> results/fig8_pan_cancer/numbers.json (caption, text)
