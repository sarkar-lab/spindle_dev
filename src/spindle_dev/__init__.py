"""spindle_dev: SPD sub-matrix indexing and search.

This package exposes a small, stable public API while keeping most
implementation details internal.
"""

from .index import IndexConfig, IndexHandle, build_index  # noqa: F401
from .search import SearchConfig, SearchResults, query_index  # noqa: F401
from .metrics import log_euclidean_distance, spd_to_correlation, correlation_to_spd  # noqa: F401
from .utils import DeterministicConfig
