#!/bin/bash
# Fig 3 from the index cache (results/_cache, experiments/common/c_index_cache.py), in order.
# 1-4 are sbatch (compute nodes); 5-7 run on the login node once the jobs are done.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=experiments/common
F=experiments/fig3_whole_tile

# 1. whole-matrix log vectors of every tile + exact d_W of the held-out tiles
whole=$(sbatch --parsable --job-name=f3_whole --array=0-7 $C/run.sbatch $F/c_whole.py)
# 2. Spindle-lite vs Spindle on every held-out tile (NX, work, g trade-off, band inputs)
sbatch --job-name=f3_nx --array=0-39 $C/run.sbatch $F/c_nx.py
# 3. memory vs fidelity, breast + brain (Spindle-lite, PCA + flat, PCA-256 + PQ)
sbatch --job-name=f3_memory --array=10-14,35-39 --dependency=afterok:$whole $C/run.sbatch $F/c_memory.py
# 4. timings: one job at a time on the fixed node (timing.sbatch singleton)
sbatch --array=0-7%1 --dependency=afterok:$whole,singleton $C/timing.sbatch $F/c_timing.py

# after the jobs (login node):
# 5. small inputs           python $F/c_summary.py
# 6. atoms                  python $F/f_fig3.py
# 7. draft pptx             PYTHONPATH=$C/.pylib python $C/build_figure.py fig3_whole_tile
#    then copy figures/fig3_whole_tile/fig3_whole_tile_draft.pptx to fig3_whole_tile.pptx, edit, and
#    python $C/render.py fig3_whole_tile
# 8. numbers                python $F/write_numbers.py   -> results/fig3_whole_tile/numbers.json (caption, text)
