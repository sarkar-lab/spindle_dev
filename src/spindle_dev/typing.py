"""Core datatypes for the SPD index graph.

This module defines the canonical dataclasses used by the indexing and
search code.  The implementation in :mod:`index` constructs instances
of :class:`IndexHandle`, so the field layout here must stay in sync with
what :func:`index_spds` and :func:`build_index` actually create.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

from .utils import DeterministicConfig


@dataclass
class IndexConfig:
    """Configuration for index construction.

    Attributes
    ----------
    epsilon_dict:
        Dictionary mapping block indices to maximum allowed pairwise log-Euclidean distances within block-clusters.
    epsilon_block_wise_dict:
        Nested dictionary mapping block indices to dictionaries that map cluster indices to their respective epsilon values.
    is_ordered:
        If True, block-clusters within each block are sorted by their
        log-Euclidean distance from a deterministic reference point.
    deterministic:
        Controls global deterministic settings.
    """

    epsilon_dict: Dict[int, float] = field(default_factory=dict)
    epsilon_block_wise_dict: Dict[int, Dict[int, float]] = field(default_factory=dict)
    threshold_type: str = "constant" # "constant" or "block-wise"
    is_ordered: bool = False
    debug: bool = False
    kmean_method: str = "online"
    stabilize_block_cluster: bool = False
    max_iter: int = 5
    deterministic: DeterministicConfig = field(default_factory=DeterministicConfig)


@dataclass
class BlockClusterMetadata:
    """Metadata stored for a single block-cluster.

    members:
        List of (spd_id, block_id) pairs indicating which original SPD
        matrices and which block of each matrix contributed to this
        cluster.
    mean:
        Mean SPD matrix for this block cluster.
    """

    members: List[Tuple[int, int]]
    mean: np.ndarray
    representative_mean: np.ndarray
    radius: float


@dataclass
class BlockClusterNode:
    """Node in the block-level DAG.
    global_node_id:
        Unique identifier of the node within the entire index.
    node_id:
        Unique identifier of the node within a layer.
    block_index:
        Index of the block (i) this node corresponds to.
    block_cluster_id:
        Local ID of the cluster within block i.
    order_id:
        Position in the sorted list of block-cluster means for block i.
    metadata:
        Associated :class:`BlockClusterMetadata`.
    children:
        List of node_ids for block i+1 that co-occur with this node in
        some original SPD matrix.
    """
    global_node_id: int
    node_id: int
    block_index: int
    block_cluster_id: int
    order_id: int
    metadata: BlockClusterMetadata
    children: List[int] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - trivial formatting
        """Compact representation that omits the full mean matrix.

        This is intended for interactive use in notebooks / logs where
        printing the entire SPD block is noisy.  It summarizes the
        structural fields, the number of members, and the mean shape.
        """

        n_members = len(self.metadata.members)
        mean_shape = tuple(self.metadata.mean.shape)
        return (
            "BlockClusterNode("  # do not auto-print the matrix contents
            f"global_node_id={self.global_node_id}, "
            f"node_id={self.node_id}, "
            f"block_index={self.block_index}, "
            f"block_cluster_id={self.block_cluster_id}, "
            f"order_id={self.order_id}, "
            f"n_members={n_members}, "
            f"mean_shape={mean_shape}, "
            f"children={self.children}"
            ")"
        )


@dataclass
class IndexHandle:
    """Top-level handle returned by :func:`build_index`.

    This mirrors the concrete construction performed in
    :func:`index.index_spds` / :func:`index.build_index`.

    nodes:
        List of all nodes in the DAG.
    block_to_nodes:
        Mapping from block index to list of node_ids sorted by
        block-cluster mean distance from a deterministic reference.
    config:
        Index configuration used to build the index.
    """

    nodes: List[BlockClusterNode]
    block_to_nodes: Dict[int, List[int]]
    block_runs: dict[int, int]
    perm: List[int]
    config: IndexConfig
    block_to_node_indices: Dict[int, List[int]]
    sorted_blocks: List[int]

@dataclass
class DatasetIndex:
    """Metadata and models associated with an indexed dataset.

    Attributes
    ----------
    dag_dict:
        Mapping from cluster id to :class:`IndexHandle`.
    metadata:
        Original dataset metadata carried over from :class:`ProcessedData`.
    latent:
        Latent representations (typically a dict containing PCA / UMAP
        embeddings) from :class:`ProcessedData` used during index
        construction.
    labels:
        Cluster labels for each SPD tile used to build the index.
    pca_model:
        The fitted PCA model used to produce the latent PCA features.
        This is stored so that unseen SPDs can be projected into the
        same space for alignment without refitting PCA.
    """

    dag_dict: Dict[int, IndexHandle]
    metadata: Dict[str, Any]
    latent: Any
    labels: np.ndarray
    pca_model: Any | None = None


__all__ = [
    "IndexConfig",
    "BlockClusterMetadata",
    "BlockClusterNode",
    "IndexHandle",
    "DatasetIndex",
]
