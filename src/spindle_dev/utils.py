"""Logging."""

from __future__ import annotations

import logging


def get_logger(name: str = "spindle_dev") -> logging.Logger:
    """A logger with a simple stream handler (added once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
