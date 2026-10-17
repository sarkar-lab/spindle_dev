"""Dataset paths, read from datasets.yaml (the only place they are written).

  from paths import dataset_path
  dataset_path("breast_cancer")                 # by key, file stem or file name

Shell: ``python paths.py <name>`` prints the full path; ``python paths.py --dir <name>`` prints a folder.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG = PROJECT_ROOT / "datasets.yaml"
DEFAULT_MAX_GENES = 800


@lru_cache(maxsize=1)
def _config() -> dict:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def data_dir(name: str) -> Path:
    """A folder from ``dirs``."""
    return _resolve(_config()["dirs"][name])


def indexed_keys() -> list[str]:
    """The indexed sections, in figure order."""
    return list(_config()["indexed"])


def _entry(name: str) -> dict:
    for key, e in _config()["datasets"].items():
        if name in (key, e["file"], Path(e["file"]).stem):
            return e
    raise KeyError(f"unknown dataset '{name}'; add it to {CONFIG}")


def dataset_path(name: str) -> Path:
    """The .h5ad of a dataset, given its key, file stem or file name."""
    e = _entry(name)
    return data_dir(e["dir"]) / e["file"]


def dataset_max_genes(name: str) -> int:
    """The index's gene cap for a dataset (default 800)."""
    return int(_entry(name).get("max_genes", DEFAULT_MAX_GENES))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="dataset key, file stem or file name")
    parser.add_argument("--dir", help="print a folder from dirs instead")
    args = parser.parse_args()
    print(data_dir(args.dir) if args.dir else dataset_path(args.name))


if __name__ == "__main__":
    main()
