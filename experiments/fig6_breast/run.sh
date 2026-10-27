#!/bin/bash
# Fig 6 (breast block programmes and queries), in order. 1-2 are sbatch (compute nodes, from the index cache);
# 3-5 run on the login node after the jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=experiments/common
F=experiments/fig6_breast

# 1. niches, block programmes, co-variation, enrichment, ACTA2 (seeds 0-4)
sbatch --job-name=f6_prog --array=0-4 $C/run.sbatch $F/c_programs.py
# 2. whole-tile and gene-list queries (seeds 0-4)
sbatch --job-name=f6_query --array=0-4 $C/run.sbatch $F/c_queries.py

# after the jobs (login node):
# 3. atoms                  python $F/f_fig6.py
# 4. draft pptx             PYTHONPATH=$C/.pylib python $C/build_figure.py fig6_breast
#    then copy figures/fig6_breast/fig6_breast_draft.pptx to fig6_breast.pptx, edit, and
#    PYTHONPATH=$C/.pylib python $C/render.py fig6_breast
# 5. numbers                python $F/write_numbers.py   -> results/fig6_breast/numbers.json (caption, text)
