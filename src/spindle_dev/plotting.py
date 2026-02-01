import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx


def plot_blocks_from_fcluster(R, perm, runs, title="", ax=None):
    """Plot correlation matrix with block borders.

    Parameters
    ----------
    R : np.ndarray
        Correlation matrix (n, n).
    perm : array-like
        Permutation of indices (length n).
    runs : list of (start, end)
        Block intervals in perm space, [start, end).
    title : str, optional
        Title for the plot.
    ax : matplotlib.axes.Axes, optional
        If provided, draw into this Axes; otherwise create a new figure.
    """
    perm = np.asarray(perm)
    Rp = R[np.ix_(perm, perm)]

    # 5) plot with block borders
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True
    else:
        fig = ax.figure

    im = ax.imshow(Rp, vmin=-1, vmax=1, cmap="coolwarm")
    # Attach colorbar to the figure but anchored to this axes
    fig.colorbar(im, ax=ax)

    for (a, b) in runs:
        # rectangle around diagonal block a..b-1
        rect = Rectangle((a - 0.5, a - 0.5), (b - a), (b - a),
                         fill=False, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)

    ax.set_title(title or f"#blocks={len(runs)}")
    ax.set_xlabel("j (perm)")
    ax.set_ylabel("i (perm)")
    fig.tight_layout()

    if created_fig:
        plt.show()


def visualize_block_dag_nx(nodes, title="", ax = None, figsize=(12,6)):
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n.global_node_id, layer=n.block_index, clust=n.block_cluster_id)
        for ch in n.children:
            G.add_edge(n.global_node_id, ch)

    # simple layered coordinates: x spreads within layer, y is layer
    layers = {}
    for nid, data in G.nodes(data=True):
        layers.setdefault(data["layer"], []).append(nid)

    pos = {}
    for layer, nids in sorted(layers.items()):
        nids = sorted(nids)
        for i, nid in enumerate(nids):
            pos[nid] = (i, -layer)

    labels = {nid: f"N{nid} B{G.nodes[nid]['layer']}\nC{G.nodes[nid]['clust']}" for nid in G.nodes()}
    if ax is None:
        plt.figure(figsize=figsize)
        nx.draw(G, pos, with_labels=False, arrows=True, node_size=1000, alpha=0.7)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)
        plt.axis("off")
        if not title:
            title = "Block Cluster DAG"
        plt.title(title)
        plt.show()
    else:
        nx.draw(G, pos, with_labels=False, arrows=True, node_size=1000, alpha=0.7, ax=ax)
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, ax=ax)
        if not title:
            title = "Block Cluster DAG"
        ax.set_title(title)
        ax.axis("off")
    return ax


from collections import defaultdict
import numpy as np

def _node_spd_set(n):
    # members: List[(spd_id, block_id)]
    return set(spd_id for spd_id, _ in n.metadata.members)

# def visualize_block_dag_sankey(nodes, title="Block Cluster Sankey", block_id_from_members=True):
#     """
#     Sankey view of the block-DAG:
#       - node value = #unique SPDs assigned to that node (thickness)
#       - link value = #SPDs that pass from parent (block i) to child (block i+1)

#     Requires: plotly
#       pip install plotly
#     """
#     import plotly.graph_objects as go

#     # ---- group nodes by layer (block_index)
#     by_layer = defaultdict(list)
#     for n in nodes:
#         by_layer[n.block_index].append(n)
#     layers = sorted(by_layer.keys())

#     # ---- index nodes globally for sankey
#     # Keep stable ordering: by layer then order_id (or global_node_id)
#     ordered_nodes = []
#     for b in layers:
#         ordered_nodes.extend(sorted(by_layer[b], key=lambda n: (n.order_id, n.global_node_id)))

#     idx = {n.global_node_id: i for i, n in enumerate(ordered_nodes)}

#     # ---- node weights (#SPDs)
#     node_spds = {n.global_node_id: len(n.metadata.members) for n in ordered_nodes}
#     node_w = np.array([node_spds[n.global_node_id] for n in ordered_nodes], dtype=float)



#     # ---- build per-SPD assignment per layer (spd_id -> node in that layer)
#     # This makes edge counts correct and ensures "layer width == #SPDs in that block".
#     spd_to_node_in_layer = {b: {} for b in layers}
#     for n in ordered_nodes:
#         b = n.block_index
#         for spd_id, blk_id in n.metadata.members:
#             if (not block_id_from_members) or (blk_id == b):
#                 # If an SPD appears in multiple clusters in same layer, last one wins.
#                 # If that can happen for you, we can instead assert/raise here.
#                 spd_to_node_in_layer[b][spd_id] = n.global_node_id

#     # ---- links: consecutive layers only
#     src, tgt, val = [], [], []
#     for b0, b1 in zip(layers[:-1], layers[1:]):
#         m0 = spd_to_node_in_layer[b0]
#         m1 = spd_to_node_in_layer[b1]
#         common_spds = set(m0.keys()) & set(m1.keys())
#         edge_counts = defaultdict(int)
#         for spd_id in common_spds:
#             u = m0[spd_id]
#             v = m1[spd_id]
#             edge_counts[(u, v)] += 1

#         for (u, v), c in edge_counts.items():
#             src.append(idx[u])
#             tgt.append(idx[v])
#             val.append(c)

#     # ---- positions: fix x by layer; y by cumulative node weight within layer
#     # This makes the layer "stack height" proportional to #SPDs in that layer.
#     x = np.zeros(len(ordered_nodes))
#     y = np.zeros(len(ordered_nodes))

#     layer_to_nodeids = defaultdict(list)
#     for n in ordered_nodes:
#         layer_to_nodeids[n.block_index].append(n.global_node_id)

#     x_by_layer = {b: (i / max(1, (len(layers) - 1))) for i, b in enumerate(layers)}

#     for b in layers:
#         nids = layer_to_nodeids[b]
#         # order nodes within layer consistently (same as ordered_nodes)
#         nids = [n.global_node_id for n in ordered_nodes if n.block_index == b]
#         weights = np.array([node_spds[nid] for nid in nids], dtype=float)
#         tot = weights.sum()

#         # y in [0,1]: center each node in its cumulative segment
#         csum = np.cumsum(weights)
#         starts = csum - weights
#         centers = (starts + weights / 2.0)
#         centers = (centers / tot) if tot > 0 else np.linspace(0.5/len(nids), 1-0.5/len(nids), len(nids))

#         for nid, yc in zip(nids, centers):
#             i = idx[nid]
#             x[i] = x_by_layer[b]
#             y[i] = yc

#     labels = [
#         f"B{n.block_index}  C{n.block_cluster_id} N{n.global_node_id}<br>"
#         f"nSPD={node_spds[n.global_node_id]}"
#         for n in ordered_nodes
#     ]

#     fig = go.Figure(
#         go.Sankey(
#             arrangement="fixed",
#             node=dict(
#                 label=labels,
#                 x=x.tolist(),
#                 y=y.tolist(),
#                 pad=10,
#                 thickness=30,  # visual node bar width (not the "flow width")
#             ),
#             link=dict(
#                 source=src,
#                 target=tgt,
#                 value=val,
#             ),
#         )
#     )
#     fig.update_layout(title_text=title, font_size=11,height=1200)
#     return fig

import numpy as np

def expm_sym(A, eps=1e-10):
    A = 0.5*(A + A.T)
    w, V = np.linalg.eigh(A)
    # exp of symmetric matrix
    S = (V * np.exp(w)) @ V.T
    return 0.5*(S + S.T)

import seaborn as sns
import matplotlib.pyplot as plt
from spindle_dev import metrics

def plot_block_size_distribution(data, cluster_id, dag_dict, node_id, take_representative_mean=False, figsize=(11,5)):
    gene_list = data.metadata['genes']
    gene_array = np.array(gene_list)
    dag = dag_dict[cluster_id]
    perm = dag.perm
    nodes = dag.nodes
    gene_array_perm = gene_array[np.ix_(perm)]
    node = [n for n in nodes if n.global_node_id == node_id][0]
    if take_representative_mean:
        log_spd_mean = node.metadata.representative_mean
    else:
        log_spd_mean = node.metadata.mean
    spd_mean = expm_sym(log_spd_mean)
    spd_ids = node.metadata.members
    #R_mean = data.R_mean_list[cluster_id]
    #R_mean_perm = R_mean[np.ix_(perm, perm)]
    block_index = node.block_index
    block_start, block_end = dag.block_runs[block_index]
    R_mean_block = metrics.spd_to_correlation(spd_mean)
    #R_mean_block = R_mean_perm[block_start:block_end, :][:, block_start:block_end]
    gene_subset = gene_array_perm[block_start:block_end]
    p = len(gene_subset)
    fig, ax = plt.subplots(1, 2, figsize=figsize)
    if p <= 20:
        sns.heatmap(spd_mean, xticklabels=gene_subset, yticklabels=gene_subset, cmap="Reds", annot=True, fmt=".2f", annot_kws={"size": 6}, ax=ax[0])
    else:
        sns.heatmap(spd_mean, xticklabels=gene_subset, yticklabels=gene_subset, cmap="Reds", annot=False, fmt=".2f", annot_kws={"size": 6}, ax=ax[0])
    ax[0].set_title(f'Cluster {cluster_id} Node {node_id} SPD Mean with {len(spd_ids)} SPDs')
    ax[0].set_xlabel("Genes")
    ax[0].set_ylabel("Genes")
    if p <= 20:
        sns.heatmap(R_mean_block, xticklabels=gene_subset, yticklabels=gene_subset, cmap="Reds", annot=True, fmt=".2f", annot_kws={"size": 6}
                , ax=ax[1])
    else:
        sns.heatmap(R_mean_block, xticklabels=gene_subset, yticklabels=gene_subset, cmap="Reds", annot=False, fmt=".2f", annot_kws={"size": 6}
                , ax=ax[1])
    ax[1].set_title(f"Cluster {cluster_id} Node {node_id} Block {block_index} Mean SPD Expression")
    ax[1].set_xlabel("Genes")
    ax[1].set_ylabel("Genes")
    plt.tight_layout()
    plt.show()
    return spd_ids

import scanpy as sc
def plot_spd_ids(spd_ids, tiles, adata, coords, label='Cluster', figsize=(9, 6)):
    spd_tiles = [tiles[spd_id] for spd_id,_ in spd_ids]
    spd_tile_ids = np.concatenate([np.asarray(t.idx).ravel() for t in spd_tiles]).astype(int)
    palette = sc.pl.palettes.default_20
    # make sure Cluster is categorical
    if label not in adata.obs:
        raise ValueError(f"adata.obs must contain a '{label}' column for plotting.")
    cluster_cat = adata.obs[label].astype('category')
    # category labels (one per cluster)
    codes = cluster_cat.cat.categories.tolist()
    # color list aligned with category order (same length as codes)
    color_list = [palette[i % len(palette)] for i, _ in enumerate(codes)]

    # map each cell's Cluster value to its color (one color per spot)
    color_map = dict(zip(codes, color_list))
    point_colors = cluster_cat.map(color_map).values
    # optional: register colors with Scanpy for this field
    # plt.scatter(coords[:, 0], coords[:, 1], s=1, c=point_colors, lw=0, alpha=0.8)
    from matplotlib.lines import Line2D
    from matplotlib.collections import PatchCollection

    # clusters present among highlighted cells
    present_codes = np.unique(cluster_cat.iloc[spd_tile_ids])
    fig, ax = plt.subplots(figsize=figsize)  # <-- control size here

    ax.scatter(coords[:, 0], coords[:, 1], s=1, c='grey', lw=0, alpha=0.04)
    ax.scatter(
        coords[spd_tile_ids, 0],
        coords[spd_tile_ids, 1],
        s=1,
        c=point_colors[spd_tile_ids],
        lw=0,
        alpha=0.8,
    )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o", linestyle="None",
            markerfacecolor=color_map[c], markeredgecolor="none",
            markersize=6, label=str(c),
        )
        for c in present_codes
    ]

    if legend_handles:
        ax.legend(
            handles=legend_handles,
            title="Cluster",
            frameon=False,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            ncol=2,
        )
        fig.subplots_adjust(right=0.8)
    patches = []
    for t in spd_tiles:
        bbox = t.bbox if hasattr(t, "bbox") else t["bbox"]  # (xmin,ymin,xmax,ymax)
        x0, y0, x1, y1 = bbox
        patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False))
    pc = PatchCollection(patches, match_original=True, linewidths=0.2, edgecolors='b', alpha=0.3)
    ax = plt.gca()
    ax.add_collection(pc)
    ax.invert_yaxis()
    ax.axis('off')
    plt.show()


import numpy as np

def _gene_indices(all_genes, module_genes):
    """Return indices of module_genes that are present in all_genes, preserving order."""
    g2i = {g: i for i, g in enumerate(all_genes)}
    idx = [g2i[g] for g in module_genes if g in g2i]
    missing = [g for g in module_genes if g not in g2i]
    return np.array(idx, dtype=int), missing

def _corr_from_cov(S, eps=1e-8):
    """Convert covariance to correlation (safe)."""
    d = np.sqrt(np.maximum(np.diag(S), eps))
    invd = 1.0 / d
    R = (S * invd[None, :]) * invd[:, None]
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return np.clip(R, -1.0, 1.0)

def _offdiag_mask(p):
    m = np.ones((p, p), dtype=bool)
    np.fill_diagonal(m, False)
    return m

def score_tile_cov_submatrix(S_sub, metric="energy", eps=1e-8):
    """
    S_sub: (k,k) covariance for module genes within one tile.
    Returns a scalar score (higher = more coherent covariance structure).
    """
    k = S_sub.shape[0]
    if k < 2:
        return np.nan

    if metric in ("energy", "mean_abs_corr"):
        R = _corr_from_cov(S_sub, eps=eps)
        off = R[_offdiag_mask(k)]
        if metric == "energy":
            return float(np.mean(off * off))          # mean squared off-diagonal corr
        else:
            return float(np.mean(np.abs(off)))        # mean |corr| off-diagonal

    elif metric == "pc1_frac":
        # eigenvalues of covariance: variance explained by first axis
        w = np.linalg.eigvalsh(0.5 * (S_sub + S_sub.T))
        w = np.maximum(w, 0.0)
        s = w.sum()
        return float(w[-1] / s) if s > 0 else np.nan

    else:
        raise ValueError(f"Unknown metric: {metric}")

def tile_module_scores(index_obj, module_genes, metric="energy", min_genes=3, eps=1e-8):
    """
    index_obj has:
      - index_obj.spd_matrices : list of (p,p) cov matrices (same gene order = index_obj.metadata['genes'])
      - index_obj.spd_ids      : list of tile_id (same length)
      - index_obj.metadata['genes'] : array/list of gene names (genes_work)
    Returns:
      scores: dict tile_id -> score
      info: dict with kept_genes, missing_genes, k
    """
    all_genes = list(index_obj.metadata["genes"])
    idx, missing = _gene_indices(all_genes, module_genes)

    if idx.size < min_genes:
        # Not enough genes to score reliably
        scores = {tile_id: np.nan for tile_id in index_obj.spd_ids}
        info = {"kept_genes": [all_genes[i] for i in idx], "missing_genes": missing, "k": int(idx.size)}
        return scores, info

    scores = {}
    for tile_id, S in zip(index_obj.spd_ids, index_obj.spd_matrices):
        S_sub = S[np.ix_(idx, idx)]
        scores[tile_id] = score_tile_cov_submatrix(S_sub, metric=metric, eps=eps)

    info = {"kept_genes": [all_genes[i] for i in idx], "missing_genes": missing, "k": int(idx.size)}
    return scores, info

def assign_module_score_to_spots(index_obj, tile_scores):
    # self.spot_label = np.full(self.num_spots, -1, dtype=int)
    # make it a dictionary
    spot_score = {}
    for i,t in enumerate(index_obj.metadata['tiles']):
        if i not in tile_scores:
            continue
        c = tile_scores[i]
        for id in t.idx:
            spot_score[id] = c
    return spot_score


import plotly.graph_objects as go
from collections import defaultdict
import numpy as np

def _node_spd_set(n):
    # members: List[(spd_id, block_id)]
    return set(spd_id for spd_id, _ in n.metadata.members)

def visualize_block_dag_sankey(nodes, title="Block Cluster Sankey", block_id_from_members=True, thickness=30.0, height=1200):
    """
    Sankey view of the block-DAG:
      - node value = #unique SPDs assigned to that node (thickness)
      - link value = #SPDs that pass from parent (block i) to child (block i+1)

    Requires: plotly
      pip install plotly
    """
    import plotly.graph_objects as go

    # ---- group nodes by layer (block_index)
    by_layer = defaultdict(list)
    for n in nodes:
        by_layer[n.block_index].append(n)
    layers = sorted(by_layer.keys())

    # ---- index nodes globally for sankey
    # Keep stable ordering: by layer then order_id (or global_node_id)
    ordered_nodes = []
    for b in layers:
        ordered_nodes.extend(sorted(by_layer[b], key=lambda n: (n.order_id, n.global_node_id)))

    idx = {n.global_node_id: i for i, n in enumerate(ordered_nodes)}

    # ---- node weights (#SPDs)
    node_spds = {n.global_node_id: len(n.metadata.members) for n in ordered_nodes}
    node_w = np.array([node_spds[n.global_node_id] for n in ordered_nodes], dtype=float)



    # ---- build per-SPD assignment per layer (spd_id -> node in that layer)
    # This makes edge counts correct and ensures "layer width == #SPDs in that block".
    spd_to_node_in_layer = {b: {} for b in layers}
    for n in ordered_nodes:
        b = n.block_index
        for spd_id, blk_id in n.metadata.members:
            if (not block_id_from_members) or (blk_id == b):
                # If an SPD appears in multiple clusters in same layer, last one wins.
                # If that can happen for you, we can instead assert/raise here.
                spd_to_node_in_layer[b][spd_id] = n.global_node_id

    # ---- links: consecutive layers only
    src, tgt, val = [], [], []
    for b0, b1 in zip(layers[:-1], layers[1:]):
        m0 = spd_to_node_in_layer[b0]
        m1 = spd_to_node_in_layer[b1]
        common_spds = set(m0.keys()) & set(m1.keys())
        edge_counts = defaultdict(int)
        for spd_id in common_spds:
            u = m0[spd_id]
            v = m1[spd_id]
            edge_counts[(u, v)] += 1

        for (u, v), c in edge_counts.items():
            src.append(idx[u])
            tgt.append(idx[v])
            val.append(c)

    # ---- positions: fix x by layer; y by cumulative node weight within layer
    # This makes the layer "stack height" proportional to #SPDs in that layer.
    x = np.zeros(len(ordered_nodes))
    y = np.zeros(len(ordered_nodes))

    layer_to_nodeids = defaultdict(list)
    for n in ordered_nodes:
        layer_to_nodeids[n.block_index].append(n.global_node_id)

    x_by_layer = {b: (i / max(1, (len(layers) - 1))) for i, b in enumerate(layers)}

    for b in layers:
        nids = layer_to_nodeids[b]
        # order nodes within layer consistently (same as ordered_nodes)
        nids = [n.global_node_id for n in ordered_nodes if n.block_index == b]
        weights = np.array([node_spds[nid] for nid in nids], dtype=float)
        tot = weights.sum()

        # y in [0,1]: center each node in its cumulative segment
        csum = np.cumsum(weights)
        starts = csum - weights
        centers = (starts + weights / 2.0)
        centers = (centers / tot) if tot > 0 else np.linspace(0.5/len(nids), 1-0.5/len(nids), len(nids))

        for nid, yc in zip(nids, centers):
            i = idx[nid]
            x[i] = x_by_layer[b]
            y[i] = yc

    labels = [
        f"B{n.block_index}  C{n.block_cluster_id} N{n.global_node_id}<br>"
        f"nSPD={node_spds[n.global_node_id]}"
        for n in ordered_nodes
    ]

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                label=labels,
                x=x.tolist(),
                y=y.tolist(),
                pad=10,
                thickness=thickness,  # visual node bar width (not the "flow width")
            ),
            link=dict(
                source=src,
                target=tgt,
                value=val,
            ),
        )
    )
    fig.update_layout(title_text=title, font_size=11,height=height)
    return fig

from collections import defaultdict
import numpy as np
import plotly.graph_objects as go


def visualize_block_dag_sankey_scaled(
    nodes,
    title="Block Cluster Sankey",
    block_id_from_members=True,
    thickness=30.0,
    height=1200,
    # --- NEW: display scaling controls ---
    scale_mode="power",     # "identity" | "sqrt" | "log1p" | "power"
    scale_alpha=0.6,        # used only when scale_mode="power"
    # --- NEW: optional min/max edge display clamp (keeps monotone, not linear) ---
    edge_min_display=None,  # e.g. 1.0
    edge_max_display=None,  # e.g. 50.0
    # --- NEW: nicer hover with true counts ---
    show_true_counts_in_hover=True,
):
    """
    Sankey view of the block-DAG:
      - Nodes correspond to (block_index layer, block_cluster_id).
      - Links connect consecutive layers only.
      - Link "value" controls thickness/flow (and indirectly node heights).

    IMPORTANT (Plotly Sankey):
      - node.x/node.y fix *positions* of rectangle centers, NOT their heights.
      - rectangle heights are determined by link.value + layout constraints.
      - Therefore, to prevent huge blocks from visually dominating while preserving
        proportions (monotone), we SCALE link values for display and show TRUE counts
        in hover/labels.

    Parameters
    ----------
    scale_mode:
      - "identity": exact linear proportions (may blow up)
      - "sqrt", "log1p", "power": compress dynamic range (monotone transform)
    edge_min_display / edge_max_display:
      Optional clamps applied AFTER scaling, purely for legibility. Still monotone.

    Requires: plotly
      pip install plotly
    """

    # ----------------------------
    # helpers
    # ----------------------------
    def _scale_vals(v, mode, alpha=0.6, eps=1e-12):
        v = np.asarray(v, dtype=float)
        if mode == "identity":
            return v
        if mode == "sqrt":
            return np.sqrt(np.maximum(v, 0.0))
        if mode == "log1p":
            return np.log1p(np.maximum(v, 0.0))
        if mode == "power":
            # alpha in (0, 1) compresses; alpha=1 is identity
            return np.power(np.maximum(v, 0.0) + eps, alpha)
        raise ValueError(f"Unknown scale_mode={mode!r}")

    def _clamp(v, vmin=None, vmax=None):
        v = np.asarray(v, dtype=float)
        if vmin is not None:
            v = np.maximum(v, float(vmin))
        if vmax is not None:
            v = np.minimum(v, float(vmax))
        return v

    # ----------------------------
    # group nodes by layer
    # ----------------------------
    by_layer = defaultdict(list)
    for n in nodes:
        by_layer[n.block_index].append(n)
    layers = sorted(by_layer.keys())

    # ----------------------------
    # stable global ordering
    # ----------------------------
    ordered_nodes = []
    for b in layers:
        ordered_nodes.extend(sorted(by_layer[b], key=lambda n: (n.order_id, n.global_node_id)))

    idx = {n.global_node_id: i for i, n in enumerate(ordered_nodes)}

    # ----------------------------
    # true node weights (#members)
    # ----------------------------
    # NOTE: these are TRUE counts; display heights come from display-scaled flows.
    node_spds_true = {n.global_node_id: len(n.metadata.members) for n in ordered_nodes}
    node_radius_true = {n.global_node_id: float(n.metadata.radius) for n in ordered_nodes}

    # ----------------------------
    # per-SPD assignment per layer: spd_id -> node_id (in that layer)
    # ----------------------------
    spd_to_node_in_layer = {b: {} for b in layers}
    for n in ordered_nodes:
        b = n.block_index
        for spd_id, blk_id in n.metadata.members:
            if (not block_id_from_members) or (blk_id == b):
                spd_to_node_in_layer[b][spd_id] = n.global_node_id

    # ----------------------------
    # links (TRUE counts): consecutive layers only
    # ----------------------------
    src, tgt = [], []
    val_true = []  # true SPD counts per edge
    edge_pairs = []  # (u, v) node ids for hover/diagnostics

    for b0, b1 in zip(layers[:-1], layers[1:]):
        m0 = spd_to_node_in_layer[b0]
        m1 = spd_to_node_in_layer[b1]
        common_spds = set(m0.keys()) & set(m1.keys())

        edge_counts = defaultdict(int)
        for spd_id in common_spds:
            u = m0[spd_id]
            v = m1[spd_id]
            edge_counts[(u, v)] += 1

        for (u, v), c in edge_counts.items():
            src.append(idx[u])
            tgt.append(idx[v])
            val_true.append(float(c))
            edge_pairs.append((u, v))

    val_true = np.asarray(val_true, dtype=float)

    # ----------------------------
    # DISPLAY scaling for legibility (monotone)
    # ----------------------------
    val_disp = _scale_vals(val_true, mode=scale_mode, alpha=scale_alpha)
    val_disp = _clamp(val_disp, vmin=edge_min_display, vmax=edge_max_display)

    # ----------------------------
    # positions: fixed x by layer; y by cumulative TRUE weight within layer
    # (This only sets centers. Heights still come from val_disp.)
    # ----------------------------
    x = np.zeros(len(ordered_nodes), dtype=float)
    y = np.zeros(len(ordered_nodes), dtype=float)

    x_by_layer = {b: (i / max(1, (len(layers) - 1))) for i, b in enumerate(layers)}

    for b in layers:
        nids = [n.global_node_id for n in ordered_nodes if n.block_index == b]
        weights = np.array([node_spds_true[nid] for nid in nids], dtype=float)
        tot = weights.sum()

        if tot > 0:
            csum = np.cumsum(weights)
            starts = csum - weights
            centers = (starts + weights / 2.0) / tot
        else:
            centers = np.linspace(0.5 / max(1, len(nids)), 1 - 0.5 / max(1, len(nids)), max(1, len(nids)))

        for nid, yc in zip(nids, centers):
            i = idx[nid]
            x[i] = x_by_layer[b]
            y[i] = float(yc)

    # ----------------------------
    # labels + hover
    # ----------------------------
    # Use true counts in labels so it’s always visible.
    labels = [
        (
            f"B{n.block_index}  C{n.block_cluster_id}  N{n.global_node_id}<br>"
            f"nSPD={node_spds_true[n.global_node_id]} R{n.metadata.radius:.2g}"
        )
        for n in ordered_nodes
    ]

    if show_true_counts_in_hover:
        # Put true + display values in hover via customdata
        # Plotly expects a list; one entry per edge.
        link_customdata = np.vstack([val_true, val_disp]).T  # shape (E,2)
        link_hover = "true=%{customdata[0]:.0f}<br>display=%{customdata[1]:.3g}<extra></extra>"
    else:
        link_customdata = None
        link_hover = None

    # ----------------------------
    # build figure
    # ----------------------------
    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                label=labels,
                x=x.tolist(),
                y=y.tolist(),
                pad=10,
                thickness=float(thickness),
            ),
            link=dict(
                source=src,
                target=tgt,
                value=val_disp.tolist(),  # <-- display-scaled values
                customdata=link_customdata.tolist() if link_customdata is not None else None,
                hovertemplate=link_hover,
            ),
        )
    )

    # Put scaling info in the title so it’s transparent
    scale_note = (
        f" (display scale: {scale_mode}"
        + (f", alpha={scale_alpha}" if scale_mode == "power" else "")
        + (f", clamp=[{edge_min_display},{edge_max_display}]" if (edge_min_display is not None or edge_max_display is not None) else "")
        + ")"
    )

    fig.update_layout(
        title_text="",
        font_size=11,
        height=int(height),
    )

    return fig


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

def _module_color_map(n_modules: int, cmap_name: str = "tab20"):
    cmap = mpl.cm.get_cmap(cmap_name, max(n_modules, 1))
    return [cmap(i) for i in range(n_modules)]


def _build_gene_to_module(modules):
    gene_to_mod = {}
    for mi, m in enumerate(modules):
        for g in m:
            gene_to_mod[g] = mi
    return gene_to_mod


def _order_genes_by_modules(modules, all_genes):
    """
    Order genes by (module_id, gene_name). Genes not in any module go last.
    """
    gene_to_mod = _build_gene_to_module(modules)
    def key(g):
        return (gene_to_mod.get(g, 10**9), g)
    ordered = sorted(list(all_genes), key=key)
    return ordered, gene_to_mod


def seaborn_corr_heatmap_with_modules(R_ord, ordered_genes, gene_to_mod, colors,
                                      vmin=-1, vmax=1, cmap="RdBu_r", figsize=(9,8)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        R_ord,
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        cbar_kws={"label": "Correlation"}
    )

    ax.set_xticks(range(len(ordered_genes)))
    ax.set_yticks(range(len(ordered_genes)))
    ax.set_xticklabels(ordered_genes, rotation=90)
    ax.set_yticklabels(ordered_genes)

    for tick in ax.get_xticklabels():
        mi = gene_to_mod.get(tick.get_text(), None)
        if mi is not None:
            tick.set_color(colors[mi])

    for tick in ax.get_yticklabels():
        mi = gene_to_mod.get(tick.get_text(), None)
        if mi is not None:
            tick.set_color(colors[mi])

    fig.tight_layout()
    return fig, ax


def plot_corr_heatmap_with_modules_old(
    node_stats: dict,
    title: str = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    show_values: bool = False,
    value_fontsize: int = 7,
    boundary_lw: float = 1.5,
    cmap: str = "RdBu_r",
    figsize=(9, 8),
    bold_labels: bool = False,
):
    """
    node_stats must contain:
      - 'R': (p,p) correlation (diag can be 0)
      - 'G': networkx graph with gene names as nodes (preferred)
      - 'modules': list[list[str]] (communities / gene programs)

    Returns: (fig, ax, ordered_genes, gene_to_mod)
    """
    R = node_stats["R"]
    G = node_stats.get("G", None)
    modules = node_stats.get("modules", [])

    # genes: prefer graph nodes (your request), else fall back to modules union
    if G is not None:
        all_genes = list(G.nodes())
    else:
        all_genes = sorted({g for m in modules for g in m})
    if len(all_genes) != R.shape[0]:
        # If ordering mismatch exists, you must pass the correct gene order used to build R.
        # Here we assume R is already aligned to all_genes order.
        # If not, you'll need a 'genes' list saved alongside R.
        pass

    ordered_genes, gene_to_mod = _order_genes_by_modules(modules, all_genes)

    # map gene -> index in current R ordering
    # IMPORTANT: this assumes R rows/cols correspond to all_genes in the same order as list(G.nodes()).
    # If not, store `genes` when you compute R and use that list here.
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    idx = [gene_to_idx[g] for g in ordered_genes]
    R_ord = R[np.ix_(idx, idx)]

    # colors for modules
    n_mod = len(modules)
    colors = _module_color_map(n_mod)
    default_color = (0.25, 0.25, 0.25, 1.0)  # for genes not in any module

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(R_ord, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    # optional numeric values (only sensible for small p)
    if show_values and R_ord.shape[0] <= 40:
        # threshold for switching text color (relative to vmax)
        vmax_abs = max(abs(vmin), abs(vmax))
        thr = 0.5 * vmax_abs  # tweak (0.4–0.7) if needed
        print(thr)

        for i in range(R_ord.shape[0]):
            for j in range(R_ord.shape[1]):
                val = float(R_ord[i, j])
                # choose white text on strong magnitude cells
                txt_color = "white" if abs(val) >= thr else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=value_fontsize,
                    color=txt_color
                )


    ax.set_xticks(range(len(ordered_genes)))
    ax.set_yticks(range(len(ordered_genes)))
    ax.set_xticklabels(ordered_genes, rotation=90)
    ax.set_yticklabels(ordered_genes)

    # color tick labels by module
    for tick in ax.get_xticklabels():
        g = tick.get_text()
        mi = gene_to_mod.get(g, None)
        tick.set_color(colors[mi] if mi is not None and mi < n_mod else default_color)
        if bold_labels:
            tick.set_weight('bold')
    for tick in ax.get_yticklabels():
        g = tick.get_text()
        mi = gene_to_mod.get(g, None)
        tick.set_color(colors[mi] if mi is not None and mi < n_mod else default_color)
        if bold_labels:
            tick.set_weight('bold')

    # # optional numeric values (only sensible for small p)
    # if show_values and R_ord.shape[0] <= 40:
    #     for i in range(R_ord.shape[0]):
    #         for j in range(R_ord.shape[1]):
    #             ax.text(j, i, f"{R_ord[i, j]:.2f}", ha="center", va="center", fontsize=value_fontsize)

    # module boundaries (after reordering, modules become contiguous blocks)
    if n_mod > 0:
        # compute block cut positions
        # count genes per module in ordered list
        counts = []
        seen = set()
        for g in ordered_genes:
            mi = gene_to_mod.get(g, None)
            if mi is None:
                continue
            if mi not in seen:
                seen.add(mi)
                counts.append(mi)
        # derive boundaries by scanning ordered_genes
        boundaries = []
        curr = gene_to_mod.get(ordered_genes[0], None)
        for i, g in enumerate(ordered_genes):
            mi = gene_to_mod.get(g, None)
            if mi != curr:
                boundaries.append(i - 0.5)
                curr = mi
        for b in boundaries:
            ax.axhline(b, color="k", lw=boundary_lw, alpha=0.7)
            ax.axvline(b, color="k", lw=boundary_lw, alpha=0.7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    if title is None:
        title = f"Correlation heatmap (reordered by modules) | node_score={node_stats.get('node_score', np.nan):.3f}"
    ax.set_title(title)
    ax.set_xlabel("Genes")
    ax.set_ylabel("Genes")

    fig.tight_layout()
    return fig, ax, ordered_genes, gene_to_mod


def plot_corr_heatmap_with_modules(
    node_stats: dict,
    title: str = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    show_values: bool = False,
    value_fontsize: int = 7,
    boundary_lw: float = 1.5,
    cmap: str = "RdBu_r",
    figsize=(9, 8),
    bold_labels: bool = False,
    filter_zero_correlation: bool = False,
    zero_corr_threshold: float = 0.0,
):
    """
    node_stats must contain:
      - 'R': (p,p) correlation (diag can be 0)
      - 'G': networkx graph with gene names as nodes (preferred)
      - 'modules': list[list[str]] (communities / gene programs)

    Parameters
    ----------
    filter_zero_correlation : bool, optional
        If True, remove modules with mean absolute within-module correlation <= zero_corr_threshold.
    zero_corr_threshold : float, optional
        Correlation threshold for filtering (default 0.0). Only used if filter_zero_correlation=True.

    Returns: (fig, ax, ordered_genes, gene_to_mod)
    """
    R = node_stats["R"]
    G = node_stats.get("G", None)
    modules = node_stats.get("modules", [])

    # genes: prefer graph nodes (your request), else fall back to modules union
    if G is not None:
        all_genes = list(G.nodes())
    else:
        all_genes = sorted({g for m in modules for g in m})
    if len(all_genes) != R.shape[0]:
        # If ordering mismatch exists, you must pass the correct gene order used to build R.
        # Here we assume R is already aligned to all_genes order.
        # If not, you'll need a 'genes' list saved alongside R.
        pass

    # Keep original mapping to R matrix indices
    original_gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    
    # Optional: filter out zero-correlation modules
    genes_for_plotting = all_genes  # default: use all genes
    if filter_zero_correlation:
        filtered_modules = []
        for m in modules:
            if len(m) < 2:
                # Skip single-gene modules
                continue
            idx = [original_gene_to_idx[g] for g in m if g in original_gene_to_idx]
            if len(idx) < 2:
                continue
            R_sub = R[np.ix_(idx, idx)]
            # Compute mean absolute off-diagonal correlation
            off_diag_mask = np.ones((len(idx), len(idx)), dtype=bool)
            np.fill_diagonal(off_diag_mask, False)
            mean_corr = np.mean(np.abs(R_sub[off_diag_mask]))
            # print(f"Module of size {len(m)}: mean abs corr = {mean_corr:.4f}")
            # Keep only modules with correlation STRICTLY greater than threshold
            if mean_corr > zero_corr_threshold:
                # print(f"Adding module of size {len(m)} with mean abs corr = {mean_corr:.4f}")
                filtered_modules.append(m)
        modules = filtered_modules
        # Only keep genes that are in remaining modules, preserving original order
        genes_in_modules = {g for m in modules for g in m}
        genes_for_plotting = [g for g in all_genes if g in genes_in_modules]

    # print(f"Using {len(modules)} modules for heatmap plotting.")

    ordered_genes, gene_to_mod = _order_genes_by_modules(modules, genes_for_plotting)

    # map gene -> index in ORIGINAL R matrix (before filtering)
    # This preserves the original row/column correspondence with R
    gene_to_idx = original_gene_to_idx
    idx = [gene_to_idx[g] for g in ordered_genes]
    R_ord = R[np.ix_(idx, idx)]

    # colors for modules
    n_mod = len(modules)
    colors = _module_color_map(n_mod)
    default_color = (0.25, 0.25, 0.25, 1.0)  # for genes not in any module

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(R_ord, vmin=vmin, vmax=vmax, cmap=cmap, aspect="equal")
    # optional numeric values (only sensible for small p)
    if show_values and R_ord.shape[0] <= 40:
        # threshold for switching text color (relative to vmax)
        vmax_abs = max(abs(vmin), abs(vmax))
        thr = 0.5 * vmax_abs  # tweak (0.4–0.7) if needed
        #print(thr)

        for i in range(R_ord.shape[0]):
            for j in range(R_ord.shape[1]):
                val = float(R_ord[i, j])
                # choose white text on strong magnitude cells
                txt_color = "white" if abs(val) >= thr else "black"
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    fontsize=value_fontsize,
                    color=txt_color
                )


    ax.set_xticks(range(len(ordered_genes)))
    ax.set_yticks(range(len(ordered_genes)))
    ax.set_xticklabels(ordered_genes, rotation=90)
    ax.set_yticklabels(ordered_genes)

    # color tick labels by module
    for tick in ax.get_xticklabels():
        g = tick.get_text()
        mi = gene_to_mod.get(g, None)
        tick.set_color(colors[mi] if mi is not None and mi < n_mod else default_color)
        if bold_labels:
            tick.set_weight('bold')
    for tick in ax.get_yticklabels():
        g = tick.get_text()
        mi = gene_to_mod.get(g, None)
        tick.set_color(colors[mi] if mi is not None and mi < n_mod else default_color)
        if bold_labels:
            tick.set_weight('bold')

    # # optional numeric values (only sensible for small p)
    # if show_values and R_ord.shape[0] <= 40:
    #     for i in range(R_ord.shape[0]):
    #         for j in range(R_ord.shape[1]):
    #             ax.text(j, i, f"{R_ord[i, j]:.2f}", ha="center", va="center", fontsize=value_fontsize)

    # module boundaries (after reordering, modules become contiguous blocks)
    if n_mod > 0:
        # compute block cut positions
        # count genes per module in ordered list
        counts = []
        seen = set()
        for g in ordered_genes:
            mi = gene_to_mod.get(g, None)
            if mi is None:
                continue
            if mi not in seen:
                seen.add(mi)
                counts.append(mi)
        # derive boundaries by scanning ordered_genes
        boundaries = []
        curr = gene_to_mod.get(ordered_genes[0], None)
        for i, g in enumerate(ordered_genes):
            mi = gene_to_mod.get(g, None)
            if mi != curr:
                boundaries.append(i - 0.5)
                curr = mi
        for b in boundaries:
            ax.axhline(b, color="k", lw=boundary_lw, alpha=0.7)
            ax.axvline(b, color="k", lw=boundary_lw, alpha=0.7)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    if title is None:
        title = f"Correlation heatmap (reordered by modules) | node_score={node_stats.get('node_score', np.nan):.3f}"
    ax.set_title(title)
    ax.set_xlabel("Genes")
    ax.set_ylabel("Genes")

    fig.tight_layout()
    return fig, ax, ordered_genes, gene_to_mod


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
#from matplotlib.colormaps import get_cmap
def plot_spot_module_scores_old(spot_score, tiles, adata_view, ax=None, grid=True, color='continuous', figsize=(8, 6), label=None):
    """Plot spot-level module scores on spatial coordinates.
    
    Parameters
    ----------
    spot_score : dict
        Mapping from spot_id to score.
    tiles : list
        List of tile objects with bbox attributes.
    adata_view : AnnData
        Annotated data object with spatial coordinates in obsm['spatial'].
    ax : matplotlib.axes.Axes, optional
        If provided, plot into this axes. Otherwise create a new figure.
    figsize : tuple, optional
        Figure size if creating new figure (default (8, 6)).
    
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes object for further manipulation or subplotting.
    """
    # Get spot scores and prepare data
    indices = np.array([int(i) for i in spot_score.keys()])
    
    scores = np.array(list(spot_score.values()))
    coords = adata_view.obsm['spatial']

    # Create figure if not provided
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # Normalize scores to [0, 1] range for colormap
    
    norm = Normalize(vmin=scores.min(), vmax=scores.max())
    cmap = plt.cm.get_cmap('Reds')

    # Plot scatter points with continuous colormap
    if len(indices) < adata_view.shape[0]:
        ax.scatter(coords[:, 0], coords[:, 1], s=1, c='lightgray', lw=0, alpha=0.5)

    scatter = ax.scatter(coords[indices, 0], coords[indices, 1], s=1, c=scores, 
                        cmap=cmap, norm=norm, lw=0, alpha=0.8)

    # Add tiles as patches
    if grid:
        patches = []
        for t in tiles:
            bbox = t.bbox if hasattr(t, "bbox") else t["bbox"]
            x0, y0, x1, y1 = bbox
            patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False))

        pc = PatchCollection(patches, match_original=True, linewidths=0.1, edgecolors='b', alpha=0.3)
        ax.add_collection(pc)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    if label:
        cbar.set_label(label, fontsize=11, fontweight='bold')
    else:
        cbar.set_label('Module Score', fontsize=11, fontweight='bold')

    # Format axes
    ax.invert_yaxis()
    ax.axis('off')
    
    if created_fig:
        ax.figure.tight_layout()
        plt.show()
        return fig,ax
    return ax
    

def plot_spot_module_scores(
    spot_score,
    tiles,
    adata_view,
    spot_size=1,
    ax=None,
    grid=True,
    color='continuous',
    figsize=(8, 6),
    label=None
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, ListedColormap, BoundaryNorm
    from matplotlib.patches import Rectangle
    from matplotlib.collections import PatchCollection

    # ---- data prep
    indices = np.array([int(i) for i in spot_score.keys()])
    values = np.array(list(spot_score.values()))
    coords = adata_view.obsm['spatial']

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True

    # ---- background
    if len(indices) < adata_view.shape[0]:
        ax.scatter(
            coords[:, 0], coords[:, 1],
            s=spot_size, c='lightgray', lw=0, alpha=0.5
        )
    # return fig, ax
    # ===============================
    # Continuous (existing behavior)
    # ===============================
    if color == 'continuous':
        norm = Normalize(vmin=values.min(), vmax=values.max())
        cmap = plt.cm.get_cmap('Reds')

        scatter = ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            s=spot_size,
            c=values,
            cmap=cmap,
            norm=norm,
            lw=0,
            alpha=0.8
        )

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(label or 'Module Score', fontsize=11, fontweight='bold')

    # ===============================
    # Discrete (Scanpy-like behavior)
    # ===============================
    elif color == 'discrete':
        
        labels = values.astype(int)
        uniq = np.unique(labels)
        n = len(uniq)

        # Scanpy-style categorical palette
        base_colors = plt.cm.tab20.colors
        cmap = ListedColormap(base_colors[:n])

        scatter = ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            s=1,
            c=labels,
            cmap=cmap,
            lw=0,
            alpha=0.9
        )

        # ---- legend (cluster-style, no colorbar)
        from matplotlib.lines import Line2D
        handles = [
            Line2D(
                [0], [0],
                marker='o',
                linestyle='none',
                markerfacecolor=cmap(i),
                markeredgecolor='none',
                markersize=6,
                label=str(uniq[i])
            )
            for i in range(n)
        ]

        ax.legend(
            handles=handles,
            title=label or "Cluster",
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            fontsize=8
        )


    else:
        raise ValueError("color must be 'continuous' or 'discrete'")
    
    # ---- tile grid
    # Add tiles as patches
    if grid:
        patches = []
        for t in tiles:
            bbox = t.bbox if hasattr(t, "bbox") else t["bbox"]
            x0, y0, x1, y1 = bbox
            patches.append(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False))

        pc = PatchCollection(patches, match_original=True, linewidths=0.1, edgecolors='b', alpha=0.3)
        ax.add_collection(pc)

    # ---- formatting
    ax.invert_yaxis()
    ax.axis('off')

    if created_fig:
        ax.figure.tight_layout()
        plt.show()
        return fig, ax

    return ax


from . import index
def get_module_info(id, node_score_df, dag_stats_df, dag_dict):
    """
    Return metadata + L_mean/index_handle WITHOUT making any plots.
    """
    # Take the top scored node
    cluster_id = int(node_score_df.iloc[id]["cluster_id"])
    node_id    = int(node_score_df.iloc[id]["node_id"])

    block_id = int(
        dag_stats_df.loc[
            (dag_stats_df["cluster_id"] == cluster_id) & (dag_stats_df["node_id"] == node_id),
            "block_id",
        ].values[0]
    )

    print(f"Top scored node is in cluster {cluster_id}, node {node_id}, block {block_id}")

    # Most frequent center (kept for your debug / parity, not strictly needed for L_mean retrieval below)
    _ = dag_stats_df.loc[
        (dag_stats_df["cluster_id"] == cluster_id) & (dag_stats_df["block_id"] == block_id)
    ].sort_values(by="num_spds", ascending=False).node_id.values[0]

    index_handle = dag_dict[cluster_id]

    # Representative log-mean reference for this block
    L_mean = index.get_node_log_ref(index_handle, block_id, use_representative_mean=True)

    return cluster_id, node_id, block_id, L_mean, index_handle


def _fig_to_rgb_array(fig):
    from io import BytesIO
    from PIL import Image
    
    # Ensure figure is rendered
    try:
        fig.canvas.draw()
    except:
        pass
    
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0.05)
    buf.seek(0)
    img = np.array(Image.open(buf))
    if img.shape[2] == 4:
        # If RGBA, convert to RGB
        img = img[..., :3]
    buf.close()
    return img


def _as_figure(fig_or_tuple):
    # Accept fig, or (fig, ax), or (fig, axs), etc.
    if isinstance(fig_or_tuple, tuple):
        return fig_or_tuple[0]
    return fig_or_tuple


def plot_module_heatmap_plus_spatial(id, score_list, node_score_df, dag_stats_df, dag_dict, 
                                     data,
                                     adata,
                                     spot_size=1,
                                     module_ids=(0, 1), zero_corr_threshold=0.05):
    """
    One panel:
      - Left: corr heatmap with modules (auto-sized based on L_mean.shape[0])
      - Right: spatial module score plots for module_ids (stacked, constant size)
    """
    # Get metadata
    cluster_id = int(node_score_df.iloc[id]["cluster_id"])
    node_id = int(node_score_df.iloc[id]["node_id"])
    block_id = int(
        dag_stats_df.loc[
            (dag_stats_df["cluster_id"] == cluster_id) & (dag_stats_df["node_id"] == node_id),
            "block_id",
        ].values[0]
    )
    index_handle = dag_dict[cluster_id]
    L_mean = index.get_node_log_ref(index_handle, block_id, use_representative_mean=True)
    p = int(L_mean.shape[0])

    print(f"Creating combined figure for cluster {cluster_id}, node {node_id}, block {block_id}")

    # Make the heatmap and embed as image
    fig_hm, ax_hm, ordered_genes, gene_to_mod = plot_corr_heatmap_with_modules(
        score_list[id],
        show_values=False,
        value_fontsize=10,
        bold_labels=True,
        figsize=(8, 7),
        filter_zero_correlation=True,
        zero_corr_threshold=zero_corr_threshold,
    )
    hm_img = _fig_to_rgb_array(fig_hm)
    plt.close(fig_hm)

    # Build the combined figure
    heat_w = max(6.5, 0.25 * p)
    heat_h = max(5.0, 0.21 * p)
    spatial_w = 5.5
    total_w = heat_w + spatial_w
    total_h = max(heat_h, 6.5)

    fig = plt.figure(figsize=(total_w, total_h), constrained_layout=False)
    gs = fig.add_gridspec(
        nrows=2, ncols=2,
        width_ratios=[heat_w, spatial_w],
        height_ratios=[1, 1],
        wspace=0.05, hspace=0.10
    )

    ax_heat = fig.add_subplot(gs[:, 0])
    ax_sp0  = fig.add_subplot(gs[0, 1])
    ax_sp1  = fig.add_subplot(gs[1, 1])

    # Place heatmap image
    ax_heat.imshow(hm_img)
    ax_heat.set_axis_off()
    ax_heat.set_title(f"Cluster {cluster_id} | Node {node_id} | Block {block_id}", fontsize=10, pad=6)

    # Create spatial plots directly into axes (NO intermediate figures!)
    scores_final = None

    tiles = data.metadata["tiles"]
    for ax_sp, mid in zip([ax_sp0, ax_sp1], module_ids):
        # print(f"Getting module {mid} score for cluster {cluster_id}, node {node_id}, block {block_id}")
        
        scores_tiles_wrt_ref, info = index.tile_module_scores_from_reference(
            data,
            score_list[id]["modules"][mid],
            index_handle,
            block_id,
            L_mean,
            cluster_id,
        )
        
        spot_score = assign_module_score_to_spots(data, scores_tiles_wrt_ref)
        scores = np.array(list(spot_score.values()))
        scores_final = scores
        
        # Plot directly into the axis - no intermediate figure!
        plot_spot_module_scores(
            spot_score, tiles, adata,
            spot_size=spot_size,
            ax=ax_sp,  # Pass the axis directly!
            grid=False,
            label=f"Module {mid}",
        )
        ax_sp.set_title(f"Module {mid}", fontsize=10, pad=4)

    return scores_final, fig