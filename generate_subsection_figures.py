import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.ticker import FuncFormatter

def generate_subsection_figures():
    project_root = Path('/home/asus/spindle_dev')
    results_dir = project_root / 'results'
    figures_dir = project_root / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Colors
    NAVY = '#2b446a'
    TERRACOTTA = '#c86b4a'
    TEAL = '#006d77'
    SLATE = '#607d8b'
    PURPLE = '#7e64a1'

    # 1. Scalability - Scatter Plot with Trendlines
    print("Generating Scalability Figure (Redesign)...")
    datasets = ['Skin', 'Kidney', 'Breast', 'Lung', 'Pancreas', 'Lymph Node']
    cells = [87499, 97560, 159226, 162254, 190965, 377985]
    build_time_s = [25.53, 18.62, 51.67, 45.86, 88.9, 71.51]
    index_size_mb = [19.41, 18.47, 14.69, 19.98, 30.86, 22.83]
    
    df_a = pd.DataFrame({'Dataset': datasets, 'Cells': cells, 'Build Time (s)': build_time_s, 'Index Size (MB)': index_size_mb})
    df_a = df_a.sort_values('Cells', ascending=True)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    # Scatter points and connecting lines
    ax1.plot(df_a['Cells'], df_a['Build Time (s)'], marker='o', color=NAVY, markersize=8, linewidth=2, label='Build Time (s)')
    ax2.plot(df_a['Cells'], df_a['Index Size (MB)'], marker='s', color=TERRACOTTA, markersize=8, linewidth=2, label='Index Size (MB)', linestyle='--')

    ax1.set_xlabel('Number of Cells in Tissue', fontweight='bold')
    ax1.set_ylabel('Build Time (seconds)', color=NAVY, fontweight='bold')
    ax2.set_ylabel('Index Size (MB)', color=TERRACOTTA, fontweight='bold')

    # Formatting x-axis to thousands
    ax1.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{int(x/1000)}k"))
    
    for i, txt in enumerate(df_a['Dataset']):
        ax1.annotate(txt, (df_a['Cells'].iloc[i], df_a['Build Time (s)'].iloc[i] + 3), fontsize=9, ha='center', color=NAVY)

    ax1.tick_params(axis='y', labelcolor=NAVY)
    ax2.tick_params(axis='y', labelcolor=TERRACOTTA)
    ax1.grid(axis='both', linestyle='--', alpha=0.3)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

    ax1.set_title('Index Scalability: Sub-linear Growth with Tissue Size', pad=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig_scalability.pdf', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig_scalability.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Query Speed - Speedup Multiplier
    print("Generating Query Speed Figure (Redesign)...")
    datasets_speed = ['Skin Melanoma', 'Kidney (Non-Diseased)', 'Breast Cancer', 'Lung Cancer', 'Lymph Node', 'Pancreatic Cancer']
    speedups = [9.19, 16.97, 11.06, 8.94, 9.18, 9.58]

    df_b = pd.DataFrame({
        'Dataset': datasets_speed,
        'Speedup': speedups
    }).sort_values('Speedup', ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = sns.barplot(data=df_b, x='Dataset', y='Speedup', ax=ax, color=TEAL, alpha=0.85)
    
    # Add a baseline of 1x (Brute Force)
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Brute-Force Baseline (1x)')
    
    ax.set_ylabel('Speedup Factor vs. Brute-Force (x)', fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=25)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Spindle Query Acceleration (Speedup over Brute-Force)', pad=12, fontweight='bold')
    ax.legend(loc='upper right')

    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.1f}x", (p.get_x() + p.get_width() / 2., h), ha='center', va='bottom', fontsize=10, fontweight='bold', color=SLATE, xytext=(0, 2), textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig_query_speed.pdf', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig_query_speed.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Rank Distribution - Cumulative Recall
    print("Generating Rank Distribution Figure (Redesign)...")
    # Exact 1st: 74.8, 2nd: 13.6, 3rd: 3.3, 4th-5th: 6.1, 6th-10th: 1.9
    ranks = [1, 2, 3, 5, 10]
    cumulative = [74.8, 88.4, 91.7, 97.8, 99.7]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ranks, cumulative, marker='D', markersize=8, color=PURPLE, linewidth=2.5)
    ax.fill_between(ranks, cumulative, color=PURPLE, alpha=0.15)
    
    ax.set_ylabel('Cumulative Percentage of Queries (%)', fontweight='bold')
    ax.set_xlabel('Top-K Rank', fontweight='bold')
    ax.set_title('Cumulative Spatial Retrieval Accuracy (Holdout Validation)', pad=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xticks(ranks)
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for x, y in zip(ranks, cumulative):
        ax.text(x, y + 2.5, f"{y:.1f}%", ha='center', va='bottom', fontsize=9.5, fontweight='bold', color=PURPLE)
            
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig_rank_distribution.pdf', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig_rank_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Partial Search - Dataset level
    print("Generating Partial Search Figure (Redesign)...")
    datasets_partial = ['Skin', 'Kidney', 'Breast', 'Lung', 'Lymph Node', 'Pancreas']
    recall1 = [98.0, 100.0, 98.0, 98.0, 100.0, 94.0]
    overlap10 = [84.0, 90.0, 85.0, 96.0, 80.0, 88.0] # using approximate top 10 overlap based on table

    df_p = pd.DataFrame({
        'Dataset': datasets_partial,
        'Recall@1 (%)': recall1,
        'Overlap@10 (%)': overlap10
    })

    fig, ax = plt.subplots(figsize=(9, 5))
    x_pos = np.arange(len(datasets_partial))
    w = 0.35

    ax.bar(x_pos - w/2, df_p['Recall@1 (%)'], w, label='Recall@1 (Exact Match)', color=TERRACOTTA)
    ax.bar(x_pos + w/2, df_p['Overlap@10 (%)'], w, label='Overlap within Top 10', color=NAVY)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(datasets_partial, rotation=25)
    ax.set_ylabel('Retrieval Accuracy (%)', fontweight='bold')
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', frameon=True)
    ax.set_title('Partial-Query Robustness Across Datasets', pad=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'fig_partial_search.pdf', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig_partial_search.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Cross Modal - Decay Line Chart
    print("Generating Cross Modal Figure (Redesign)...")
    k_vals = [1, 5, 10, 20]
    x2v_recall = [100.0, 100.0, 90.0, 90.0]
    v2x_recall = [84.0, 80.4, 78.2, 76.0]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(k_vals, x2v_recall, marker='o', markersize=8, color=TEAL, linewidth=2.5, label='Xenium -> Visium')
    ax.plot(k_vals, v2x_recall, marker='s', markersize=8, color=TERRACOTTA, linewidth=2.5, label='Visium -> Xenium')

    ax.set_ylabel('Recall / Overlap (%)', fontweight='bold')
    ax.set_xlabel('Top-K Candidates', fontweight='bold')
    ax.set_ylim(0, 108)
    ax.set_xticks(k_vals)
    ax.legend(loc='center right', frameon=True, title='Query Direction')
    ax.grid(axis='both', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Cross-Modality Retrieval Accuracy (Recall vs Top-K)', pad=12, fontweight='bold')
    
    # Annotate points
    for i, x in enumerate(k_vals):
        ax.annotate(f"{x2v_recall[i]:.0f}%", (x, x2v_recall[i] + 2), ha='center', color=TEAL, fontweight='bold', fontsize=9)
        ax.annotate(f"{v2x_recall[i]:.1f}%", (x, v2x_recall[i] - 4), ha='center', color=TERRACOTTA, fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(figures_dir / 'fig_cross_modal.pdf', bbox_inches='tight')
    plt.savefig(figures_dir / 'fig_cross_modal.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    generate_subsection_figures()
