#!/bin/bash
#SBATCH --job-name=spindle_bias_pca
#SBATCH --partition=cpus
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.out

# Paper plan 02, free stats: tangent-space PCA of Xenium vs Visium tiles
# before/after the global bias correction (benchmarks/cross_modal_bias_pca.py).

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
mkdir -p logs
unset VIRTUAL_ENV PYTHONHOME
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
cd $SLURM_SUBMIT_DIR
python benchmarks/cross_modal_bias_pca.py
echo "Done."
