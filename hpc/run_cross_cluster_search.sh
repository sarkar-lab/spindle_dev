#!/bin/bash
#SBATCH --job-name=spindle_cross_cluster_search
#SBATCH --partition=batch
#SBATCH --account=sarkar_lab
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --mail-user=hirak.sarkar@vanderbilt.edu
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
# Create logs directory if it doesn't exist
mkdir -p logs

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"
echo ""

# Activate conda environment (if needed)
# conda activate your_environment_name

# Run the Python script
python run_cross_cluster_search.py

# Print completion info
echo ""
echo "Job completed at: $(date)"
