import os
import numpy as np
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.spatial import cKDTree

def get_mean_expr(adata):
    if hasattr(adata.X, "toarray"):
        return np.ravel(adata.X.mean(axis=0))
    else:
        return np.ravel(adata.X.mean(axis=0))

def plot_single_gene_spatial(adata_vi, adata_xe, gene, output_path):
    print(f"Plotting spatial expression for {gene}...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Visium
    if gene in adata_vi.var_names:
        expr_vi = adata_vi[:, gene].X.toarray().ravel() if hasattr(adata_vi.X, "toarray") else adata_vi[:, gene].X.ravel()
        scatter_vi = axes[0].scatter(adata_vi.obsm['spatial'][:, 0], adata_vi.obsm['spatial'][:, 1], c=expr_vi, cmap='viridis', s=10)
        axes[0].set_title(f'Visium: {gene} Expression')
        axes[0].axis('equal')
        plt.colorbar(scatter_vi, ax=axes[0], fraction=0.046, pad=0.04)
    else:
        axes[0].set_title(f'Visium: {gene} not found')
        
    # Xenium
    if gene in adata_xe.var_names:
        expr_xe = adata_xe[:, gene].X.toarray().ravel() if hasattr(adata_xe.X, "toarray") else adata_xe[:, gene].X.ravel()
        scatter_xe = axes[1].scatter(adata_xe.obsm['spatial'][:, 0], adata_xe.obsm['spatial'][:, 1], c=expr_xe, cmap='viridis', s=1)
        axes[1].set_title(f'Xenium: {gene} Expression')
        axes[1].axis('equal')
        plt.colorbar(scatter_xe, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].set_title(f'Xenium: {gene} not found')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_gene_pair_correlation(adata_vi, adata_xe, gene1, gene2, output_path):
    print(f"Calculating spatial correlation for pair: {gene1} and {gene2}...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    for ax, adata, name in zip(axes, [adata_vi, adata_xe], ['Visium', 'Xenium']):
        if gene1 in adata.var_names and gene2 in adata.var_names:
            expr1 = adata[:, gene1].X.toarray().ravel() if hasattr(adata.X, "toarray") else adata[:, gene1].X.ravel()
            expr2 = adata[:, gene2].X.toarray().ravel() if hasattr(adata.X, "toarray") else adata[:, gene2].X.ravel()
            
            # Avoid error if constant
            if np.var(expr1) > 0 and np.var(expr2) > 0:
                corr, pval = pearsonr(expr1, expr2)
                title = f'{name}: {gene1} vs {gene2}\nPearson r={corr:.3f} (p={pval:.2e})'
            else:
                title = f'{name}: {gene1} vs {gene2}\n(Zero variance in one or both)'
            
            sns.scatterplot(x=expr1, y=expr2, ax=ax, alpha=0.5, s=10 if name=='Visium' else 2)
            ax.set_title(title)
            ax.set_xlabel(f'{gene1} Expression')
            ax.set_ylabel(f'{gene2} Expression')
        else:
            ax.set_title(f'{name}: Genes not found')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_expression_ranks(adata_vi, adata_xe, common_genes, output_path):
    print("Plotting expression ranks...")
    mean_expr_vi = get_mean_expr(adata_vi[:, common_genes])
    mean_expr_xe = get_mean_expr(adata_xe[:, common_genes])
    
    df = pd.DataFrame({
        'Gene': common_genes,
        'Visium_Mean': mean_expr_vi,
        'Xenium_Mean': mean_expr_xe
    })
    
    df['Visium_Rank'] = df['Visium_Mean'].rank(ascending=False)
    df['Xenium_Rank'] = df['Xenium_Mean'].rank(ascending=False)
    
    corr, pval = spearmanr(df['Visium_Rank'], df['Xenium_Rank'])
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(data=df, x='Visium_Rank', y='Xenium_Rank', alpha=0.6)
    
    # Identity line
    max_rank = max(df['Visium_Rank'].max(), df['Xenium_Rank'].max())
    plt.plot([1, max_rank], [1, max_rank], 'r--', label='Identity')
    
    plt.title(f'Gene Expression Ranks (n={len(common_genes)})\nSpearman rho = {corr:.3f}')
    plt.xlabel('Visium Rank (1=Highest)')
    plt.ylabel('Xenium Rank (1=Highest)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def perform_spatial_overlay_and_tile_check(adata_vi, adata_xe, common_genes, r_matrix_path, overlay_output_path, tile_output_path):
    print("Performing spatial overlay and tile property check...")
    
    if not os.path.exists(r_matrix_path):
        print(f"R matrix not found at {r_matrix_path}. Skipping overlay.")
        return
        
    R = np.load(r_matrix_path)
    
    # Extract coordinates
    coords_vi = adata_vi.obsm['spatial']
    coords_xe = adata_xe.obsm['spatial']
    
    # Transform Visium coordinates to Xenium space using R matrix
    coords_vi_homo = np.hstack([coords_vi, np.ones((coords_vi.shape[0], 1))])
    coords_vi_transformed = (R @ coords_vi_homo.T).T[:, :2]
    
    # 1. Plot spatial overlay
    plt.figure(figsize=(10, 10))
    plt.scatter(coords_xe[:, 0], coords_xe[:, 1], c='lightgray', s=1, alpha=0.5, label='Xenium Cells')
    plt.scatter(coords_vi_transformed[:, 0], coords_vi_transformed[:, 1], c='red', s=10, alpha=0.8, label='Transformed Visium Spots')
    plt.title('Spatial Overlay: Visium mapped to Xenium')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(overlay_output_path, dpi=300)
    plt.close()
    
    # 2. Tile Property Check
    # Build KDTree for Xenium cells
    tree_xe = cKDTree(coords_xe)
    
    # Define a radius for a Visium spot (e.g., half the median distance to nearest neighbor)
    tree_vi = cKDTree(coords_vi_transformed)
    dists, _ = tree_vi.query(coords_vi_transformed, k=2)
    median_dist = np.median(dists[:, 1])
    radius = median_dist / 2.0
    print(f"Estimated Visium spot radius in Xenium space: {radius:.2f}")

    # To speed up, we can pre-extract the dense matrix for common genes
    expr_vi_dense = adata_vi[:, common_genes].X.toarray() if hasattr(adata_vi.X, "toarray") else adata_vi[:, common_genes].X
    expr_xe_dense = adata_xe[:, common_genes].X.toarray() if hasattr(adata_xe.X, "toarray") else adata_xe[:, common_genes].X

    correlations = []
    for i, vi_coord in enumerate(coords_vi_transformed):
        indices = tree_xe.query_ball_point(vi_coord, r=radius)
        if len(indices) >= 5: # Require at least 5 Xenium cells in the spot
            xe_tile_expr = expr_xe_dense[indices].mean(axis=0)
            vi_spot_expr = expr_vi_dense[i]
            
            if np.var(vi_spot_expr) > 0 and np.var(xe_tile_expr) > 0:
                corr, _ = pearsonr(vi_spot_expr, xe_tile_expr)
                if not np.isnan(corr):
                    correlations.append(corr)

    if correlations:
        plt.figure(figsize=(8, 6))
        sns.histplot(correlations, bins=30, kde=True)
        plt.title(f'Tile Property Correlations\n(Visium Spot vs mean of overlapping Xenium cells)\nMean r={np.mean(correlations):.3f}, n_tiles={len(correlations)}')
        plt.xlabel('Pearson Correlation across common genes')
        plt.ylabel('Count of Tiles (Visium spots)')
        plt.tight_layout()
        plt.savefig(tile_output_path, dpi=300)
        plt.close()
        print(f"Calculated correlation for {len(correlations)} valid tiles. Mean correlation: {np.mean(correlations):.3f}")
    else:
        print("No valid tiles found for correlation calculation. Check radius or overlap.")

def main():
    os.makedirs("results", exist_ok=True)

    print("Loading datasets...")
    adata_vi = sc.read_h5ad("dataset/brca-vi.h5ad")
    adata_xe = sc.read_h5ad("dataset/brca-xe.h5ad")

    adata_vi.var_names_make_unique()
    adata_xe.var_names_make_unique()

    common_genes = list(set(adata_vi.var_names).intersection(adata_xe.var_names))
    common_genes.sort()

    print(f"Found {len(common_genes)} common genes.")

    if len(common_genes) == 0:
        print("No common genes found!")
        return

    single_gene = 'FASN' if 'FASN' in common_genes else common_genes[0]
    gene_pair = ['FASN', 'LUM']
    if gene_pair[0] not in common_genes or gene_pair[1] not in common_genes:
        if len(common_genes) >= 2:
            gene_pair = common_genes[:2]
        else:
            gene_pair = None

    # 1. Single Gene Spatial Plot
    plot_single_gene_spatial(adata_vi, adata_xe, single_gene, "results/single_gene_spatial.png")

    # 2. Gene Pair Spatial Correlation
    if gene_pair is not None:
        plot_gene_pair_correlation(adata_vi, adata_xe, gene_pair[0], gene_pair[1], "results/gene_pair_correlation.png")

    # 3. Expression Ranks Plot
    plot_expression_ranks(adata_vi, adata_xe, common_genes, "results/common_genes_rank.png")

    # 4. Spatial Overlay and Tile Check
    perform_spatial_overlay_and_tile_check(
        adata_vi, adata_xe, common_genes, 
        r_matrix_path="dataset/brca-R.npy",
        overlay_output_path="results/spatial_overlay.png",
        tile_output_path="results/tile_property_correlation.png"
    )
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()
