import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def main():
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    dataset_name = "lymph_node"
    results_dir = project_root / "results" / "gene_list_search" / dataset_name
    
    if not results_dir.exists():
        print(f"Results directory not found at {results_dir}")
        return
        
    modules = set()
    for f in results_dir.glob("*_spatial_cells.csv"):
        modules.add(f.name.replace("_spatial_cells.csv", ""))
        
    for mod_name in modules:
        cells_csv = results_dir / f"{mod_name}_spatial_cells.csv"
        matches_csv = results_dir / f"{mod_name}_top_matches.csv"
        
        if not cells_csv.exists() or not matches_csv.exists():
            continue
            
        print(f"Plotting spatial comparison for {mod_name}...")
        
        df_cells = pd.read_csv(cells_csv)
        df_matches = pd.read_csv(matches_csv)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        vmin = df_cells['pathway_score'].min()
        vmax = df_cells['pathway_score'].max()

        # Plot 1: Continuous Pathway Score
        sc_plt1 = ax1.scatter(df_cells['x'], df_cells['y'], c=df_cells['pathway_score'], cmap='magma', vmin=vmin, vmax=vmax, s=1, alpha=0.8)
        fig.colorbar(sc_plt1, ax=ax1, label='Pathway Score')
        
        ax1.set_title(f"{mod_name} - Continuous Pathway Score", fontsize=14)
        ax1.axis('equal')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Plot 2: Spindle Results (Spotlight Method)
        is_matched = np.zeros(len(df_cells), dtype=bool)
        
        for _, row in df_matches.iterrows():
            x0, y0, x1, y1 = row['x0'], row['y0'], row['x1'], row['y1']
            mask = (df_cells['x'] >= x0) & (df_cells['x'] <= x1) & (df_cells['y'] >= y0) & (df_cells['y'] <= y1)
            is_matched = is_matched | mask
            
        bg_cells = df_cells[~is_matched]
        matched_cells = df_cells[is_matched]
        
        ax2.scatter(bg_cells['x'], bg_cells['y'], c='lightgrey', s=1, alpha=0.1)
        if len(matched_cells) > 0:
            sc_plt2 = ax2.scatter(matched_cells['x'], matched_cells['y'], c=matched_cells['pathway_score'], cmap='magma', vmin=vmin, vmax=vmax, s=2, alpha=1.0)
        
        fig.colorbar(sc_plt1, ax=ax2, label='Pathway Score')
        
        ax2.set_title(f"{mod_name} - Spindle Matches (n={len(df_matches)})", fontsize=14)
        ax2.axis('equal')
        ax2.set_xticks([])
        ax2.set_yticks([])
        
        plt.tight_layout()
        out_path = results_dir / f"{mod_name}_spatial_comparison.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()
