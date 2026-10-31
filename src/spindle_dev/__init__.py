"""Spindle: search over spatial-omics tile covariances.

- ``layout``: tiles -> covariance niches -> gene order and blocks per niche (shared by all indexes).
- ``spindle``: Spindle (``Spindle``), the layered block DAG searched with a radius set by the greedy probe
  (paper/SPINDLE_LITE.md).
- ``spindle_partial``: Spindle partial (``SpindlePartial``), gene-set queries on dyadic gene intervals
  (paper/SPINDLE_PARTIAL.md).
- ``naive``: Spindle-naive, the exact block-log scan (``SpindleNaive`` for whole tiles, ``SpindleNaivePartial`` for
  gene sets).
- ``queries``: gene-set queries from gene names (``gene_indices``; ``template_query`` = Methods Eq. (signature) with
  ``variance_quantiles``; ``tile_query``).
- ``kernels``: the compiled searches behind ``Spindle.search``, ``SpindlePartial.search`` and
  ``SpindleNaive.search`` (C, built on first use).
"""

from .layout import ProcessedData, build_layout
from .naive import SpindleNaive, SpindleNaivePartial
from .preprocessing import load_adata, prepare_tiles, split_tiles
from .queries import gene_indices, template_query, tile_query, variance_quantiles
from .spindle import Spindle
from .spindle_partial import SpindlePartial
from .utils import load_index, save_index

__all__ = [
    "build_layout", "ProcessedData", "load_adata", "prepare_tiles", "split_tiles",
    "Spindle", "SpindlePartial", "SpindleNaive", "SpindleNaivePartial",
    "gene_indices", "template_query", "tile_query", "variance_quantiles",
    "save_index", "load_index",
]
