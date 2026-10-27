#!/bin/bash
# Fig 5 (Xenium <-> Visium), in order. 1-3 are sbatch (compute nodes); 4-7 run on the login node after the jobs.
set -euo pipefail
cd "$(dirname "$0")/../.."
C=experiments/common
F=experiments/fig5_cross_platform

# 1. tiles + shrunk covariances of both platforms (results/_cache/cross_modal/tiles2000.pkl)
tiles=$(sbatch --parsable --job-name=f5_tiles $C/run.sbatch $F/common5.py)
# 2. search per seed (layout seed): Spindle (3 correction arms), Spindle-lite, expression and diagonal controls
cross=$(sbatch --parsable --job-name=f5_cross --array=0-4 --dependency=afterok:$tiles $C/run.sbatch $F/c_cross.py)
# 3. maps, platform-bias PCA, example query
sbatch --job-name=f5_extras --dependency=afterok:$cross $C/run.sbatch $F/c_extras.py

# after the jobs (login node):
# 4. small inputs           python $F/c_summary.py
# 5. atoms                  python $F/f_fig5.py
# 6. draft pptx             PYTHONPATH=$C/.pylib python $C/build_figure.py fig5_cross_platform
#    then copy figures/fig5_cross_platform/fig5_cross_platform_draft.pptx to fig5_cross_platform.pptx, edit, and
#    python $C/render.py fig5_cross_platform
# 7. numbers                python $F/write_numbers.py   -> results/fig5_cross_platform/numbers.json (caption, text)
