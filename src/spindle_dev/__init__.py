"""Spindle: search over spatial-omics tile covariances.

- ``layout``: tiles -> covariance niches -> gene order and blocks per niche (shared by all indexes).
- ``spindle``: Spindle, the exact block-log scan (``Spindle`` for whole tiles, ``SpindlePartial`` for gene sets).
- ``lite``: Spindle-lite (``SpindleLite``), the layered block DAG searched with a radius set by the greedy probe
  (paper/SPINDLE_LITE.md).
- ``lite_partial``: Spindle-lite partial (``SpindleLitePartial``), gene-set queries on dyadic gene intervals
  (paper/SPINDLE_PARTIAL.md).
- ``kernels``: the compiled searches behind ``Spindle.search``, ``SpindleLite.search`` and
  ``SpindleLitePartial.search`` (C, built on first use).
- ``spindle_lite_legacy``: the first draft's Spindle-lite (ε elbow budget, effort caps), kept for reference only.
"""

from .layout import ProcessedData, build_layout
from .lite import SpindleLite
from .lite_partial import SpindleLitePartial
from .preprocessing import load_adata, prepare_tiles, split_tiles
from .spindle import Spindle, SpindlePartial
from .utils import load_index, save_index

__all__ = [
    "build_layout", "ProcessedData", "load_adata", "prepare_tiles", "split_tiles",
    "Spindle", "SpindlePartial", "SpindleLite", "SpindleLitePartial",
    "save_index", "load_index",
]
