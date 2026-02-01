"""Utility helpers: serialization, deterministic config, logging.

These functions are deliberately lightweight to keep the package
focused on core indexing and search logic.
"""

from __future__ import annotations

import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import numpy as np

PathLike = Union[str, os.PathLike]


@dataclass
class DeterministicConfig:
    """Configuration for deterministic behavior.

    Use this to set seeds in index/search routines so that clustering
    and ordering are reproducible across runs.
    """

    seed: int = 0


def configure_determinism(config: DeterministicConfig | None = None) -> None:
    """Apply deterministic settings to the global runtime.

    Currently sets the NumPy random seed. Call this early in index
    construction before any random clustering or ordering.
    """

    if config is None:
        config = DeterministicConfig()
    np.random.seed(config.seed)



def get_logger(name: str = "spindle_dev") -> logging.Logger:
    """Return a module-level logger with a simple default configuration."""

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
