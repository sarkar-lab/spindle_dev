#!/usr/bin/env python3
"""
Generate SLURM batch scripts for processing h5ad files with spindle_xenium_single.py

Usage:
    python generate_slurm_jobs.py /path/to/file.h5ad [--output-dir ./jobs]
    python generate_slurm_jobs.py /path/to/dir/*.h5ad [--output-dir ./jobs]
"""

import argparse
import os
from pathlib import Path
import glob

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=spindle_{sample_name}
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

# ---- IMPORTANT: limit BLAS threading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo -e "Setting up virtual environment for processing {h5ad_file}...\\n\\n"
export PATH="${{HOME}}/miniforge3/bin:$PATH"
eval "$(${{HOME}}/miniforge3/bin/mamba shell hook --shell bash)"
source ${{HOME}}/miniforge3/bin/activate base
mamba activate spatial

echo -e "Processing {h5ad_file}...\\n\\n"

python /data/sarkar_lab/Projects/spindle_dev/ISMB_notebook/spindle_xenium_single.py \\
    "{h5ad_file}" \\
    --resolution {resolution} \\
    --min-final-size {min_final_size} \\
    --top-genes {top_genes} \\
    --max-queries {max_queries} \\
    {all_genes_flag}
"""


def generate_job_script(h5ad_file, output_dir, resolution=0.5, min_final_size=15, 
                       top_genes=800, all_genes=True, max_queries=500):
    """Generate a SLURM job script for a single h5ad file"""
    
    h5ad_path = Path(h5ad_file)
    sample_name = h5ad_path.stem
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare template variables
    all_genes_flag = "--all-genes" if all_genes else ""
    
    script_content = SLURM_TEMPLATE.format(
        sample_name=sample_name,
        h5ad_file=str(h5ad_path.absolute()),
        resolution=resolution,
        min_final_size=min_final_size,
        top_genes=top_genes,
        max_queries=max_queries,
        all_genes_flag=all_genes_flag
    )
    
    # Write script to file
    script_path = output_dir / f"spindle_{sample_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    # Make script executable
    os.chmod(script_path, 0o755)
    
    return str(script_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SLURM batch scripts for processing h5ad files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python generate_slurm_jobs.py /path/to/file.h5ad
  
  # Process multiple files from a glob pattern
  python generate_slurm_jobs.py /path/to/dir/*.h5ad --output-dir ./jobs
  
  # Custom parameters
  python generate_slurm_jobs.py /path/to/file.h5ad --resolution 0.3 --max-queries 1000
        """
    )
    
    parser.add_argument("h5ad_files", nargs='+', 
                       help="Path(s) to h5ad file(s) or glob pattern")
    parser.add_argument("--output-dir", default="./slurm_jobs",
                       help="Directory to save generated SLURM scripts (default: ./slurm_jobs)")
    parser.add_argument("--resolution", type=float, default=0.5,
                       help="Clustering resolution (default: 0.5)")
    parser.add_argument("--min-final-size", type=int, default=15,
                       help="Minimum final cluster size (default: 15)")
    parser.add_argument("--top-genes", type=int, default=800,
                       help="Number of top genes (default: 800)")
    parser.add_argument("--all-genes", action="store_true", default=True,
                       help="Use all genes (default: True)")
    parser.add_argument("--no-all-genes", dest="all_genes", action="store_false",
                       help="Use only top genes")
    parser.add_argument("--max-queries", type=int, default=500,
                       help="Maximum number of queries (default: 500)")
    
    args = parser.parse_args()
    
    # Handle glob patterns
    h5ad_files = []
    for pattern in args.h5ad_files:
        expanded = glob.glob(pattern)
        if expanded:
            h5ad_files.extend(expanded)
        else:
            # If no glob match, treat as literal path
            h5ad_files.append(pattern)
    
    if not h5ad_files:
        print("Error: No h5ad files found")
        return 1
    
    print(f"Generating SLURM scripts for {len(h5ad_files)} file(s)...")
    
    scripts_generated = []
    for h5ad_file in h5ad_files:
        if not os.path.exists(h5ad_file):
            print(f"Warning: File not found: {h5ad_file}")
            continue
        
        script_path = generate_job_script(
            h5ad_file,
            args.output_dir,
            resolution=args.resolution,
            min_final_size=args.min_final_size,
            top_genes=args.top_genes,
            all_genes=args.all_genes,
            max_queries=args.max_queries
        )
        scripts_generated.append(script_path)
        print(f"✓ Generated: {script_path}")
    
    print(f"\n{len(scripts_generated)} script(s) generated successfully!")
    print(f"\nTo submit a job:")
    print(f"  sbatch {scripts_generated[0]}")
    
    if len(scripts_generated) > 1:
        print(f"\nTo submit all jobs:")
        print(f"  sbatch {args.output_dir}/spindle_*.sh")
    
    return 0


if __name__ == "__main__":
    exit(main())
