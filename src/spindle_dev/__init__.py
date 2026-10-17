"""Spindle: search over spatial-omics tile covariances.

- ``layout``: tiles -> covariance niches -> gene order and blocks per niche (shared by all indexes).
- ``spindle``: Spindle, the exact block-log scan (``Spindle`` for whole tiles, ``SpindlePartial`` for gene sets).
- ``lite``: Spindle-lite, the layered block DAG with budgeted depth-first search.
- ``interval``: the interval index for gene-set queries.
"""

from .interval import build_interval_index, error_bound, search_interval_index
from .layout import ProcessedData, build_layout
from .lite import LiteIndex, build_lite, load_index, save_index, search_lite
from .preprocessing import load_adata, prepare_tiles, split_tiles
from .spindle import Spindle, SpindlePartial

__all__ = [
    "build_layout", "ProcessedData", "load_adata", "prepare_tiles", "split_tiles",
    "Spindle", "SpindlePartial",
    "build_lite", "search_lite", "LiteIndex",
    "build_interval_index", "search_interval_index", "error_bound",
    "save_index", "load_index",
]
