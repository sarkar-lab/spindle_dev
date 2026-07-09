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


# ---------------------------------------------------------------------------
# SPD manifold math helpers
# ---------------------------------------------------------------------------

def log_spd(M: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Matrix logarithm of a Symmetric Positive Definite (SPD) matrix.

    Uses eigendecomposition: log(M) = V diag(log(w)) V^T, with eigenvalues
    clipped to *eps* to guarantee numerical stability.

    Parameters
    ----------
    M : np.ndarray, shape (p, p)
        A symmetric positive (semi-)definite matrix.
    eps : float
        Minimum eigenvalue floor.

    Returns
    -------
    np.ndarray, shape (p, p)
        The matrix logarithm of M.
    """
    M = 0.5 * (M + M.T)          # enforce exact symmetry
    w, V = np.linalg.eigh(M)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T


def exp_spd(M: np.ndarray) -> np.ndarray:
    """Matrix exponential of a symmetric matrix (maps back to SPD manifold).

    Uses eigendecomposition: exp(M) = V diag(exp(w)) V^T, with exponent
    values clipped to [-20, 20] to prevent overflow.

    Parameters
    ----------
    M : np.ndarray, shape (p, p)
        A symmetric matrix (e.g. the output of :func:`log_spd`).

    Returns
    -------
    np.ndarray, shape (p, p)
        The matrix exponential of M (an SPD matrix).
    """
    M = 0.5 * (M + M.T)          # enforce exact symmetry
    w, V = np.linalg.eigh(M)
    w = np.clip(w, -20, 20)
    return (V * np.exp(w)) @ V.T
