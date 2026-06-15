import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import os

sys.path.append(os.path.join("d:/SPINDLE/spindle_dev", "src"))
from spindle_dev.preprocessing import build_quadtree_tiles

def main():
    print("Loading datasets...")
    adata_vi = sc.read_h5ad(r"d:\SPINDLE\opt_brca\brca\visium_rotated.h5ad")
    adata_xe = sc.read_h5ad(r"d:\SPINDLE\opt_brca\brca\xenium_rotated.h5ad")

    coords_xe = adata_xe.obsm["spatial"]
    coords_vi = adata_vi.obsm["spatial"]

    print("Building Xenium tiles...")
    tiles_xe = build_quadtree_tiles(coords_xe, max_pts=2000, min_side=0.0, max_depth=40)

    print("Overlaying on Visium...")
    tiles_vi = []
    for t in tiles_xe:
        x0, y0, x1, y1 = t.bbox
        mask = (coords_vi[:, 0] >= x0) & (coords_vi[:, 0] < x1) & \
               (coords_vi[:, 1] >= y0) & (coords_vi[:, 1] < y1)
        if np.sum(mask) >= 10:
            tiles_vi.append(t)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot Xenium
    ax = axes[0]
    ax.scatter(coords_xe[:, 0], coords_xe[:, 1], s=1, color='lightgray', alpha=0.5)
    for t in tiles_xe:
        x0, y0, x1, y1 = t.bbox
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=0.5, edgecolor='blue', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
    ax.set_title(f"Xenium ({len(tiles_xe)} tiles)")
    ax.set_aspect('equal')

    # Plot Visium
    ax = axes[1]
    ax.scatter(coords_vi[:, 0], coords_vi[:, 1], s=5, color='orange', alpha=0.8)
    for t in tiles_vi:
        x0, y0, x1, y1 = t.bbox
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
    ax.set_title(f"Visium ({len(tiles_vi)} tiles with >=0 points)")
    ax.set_aspect('equal')

    plt.tight_layout()
    os.makedirs("results/cross_modal_search", exist_ok=True)
    out_path = "results/cross_modal_search/tile_overlay.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

if __name__ == "__main__":
    main()
