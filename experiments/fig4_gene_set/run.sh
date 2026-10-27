#!/bin/bash
# Fig 4 from the index cache (results/_cache, experiments/common/c_index_cache.py), in order.
# 1-2 are sbatch (compute nodes); 3-6 run on the login node once the jobs are done.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=experiments/common
F=experiments/fig4_gene_set

# 1. gene-set search against the exact d_B^S on every held-out tile (Spindle-lite-partial, padding, imputation)
sbatch --job-name=f4_nx --array=0-39 $C/run.sbatch $F/c_nx.py
# 2. timings: one job at a time on the fixed node (timing.sbatch singleton)
sbatch --array=0-7%1 $C/timing.sbatch $F/c_timing.py

# after the jobs (login node):
# 3. small inputs           python $F/c_summary.py
# 4. atoms                  python $F/f_fig4.py
# 5. draft pptx             PYTHONPATH=$C/.pylib python $C/build_figure.py fig4_gene_set
#    then copy figures/fig4_gene_set/fig4_gene_set_draft.pptx to fig4_gene_set.pptx, edit, and
#    python $C/render.py fig4_gene_set
# 6. numbers                python $F/write_numbers.py   -> results/fig4_gene_set/numbers.json (caption, text)
