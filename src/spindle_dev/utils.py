"""Logging and pickling."""

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


def save_index(obj, path) -> None:
    """Pickle any index or layout."""
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path):
    import pickle
    with open(path, "rb") as fh:
        return pickle.load(fh)
