#!/bin/bash
#SBATCH --job-name=spindle_tile_test
#SBATCH --partition=cpus
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ghoss18@vanderbilt.edu

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p logs

echo "Activating Conda environment..."
unset VIRTUAL_ENV
source ${HOME}/miniforge3/bin/activate base
conda activate spindle_env
export PATH="${HOME}/miniforge3/envs/spindle_env/bin:$PATH"

cd $SLURM_SUBMIT_DIR

python benchmarks/test_tile_size_comparison.py
