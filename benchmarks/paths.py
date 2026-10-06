"""Dataset paths, read from datasets.yaml at the repository root (the only place they are written).

  from paths import dataset_path, data_dir
  dataset_path("breast_cancer")                 # by key
  dataset_path("xenium_human_breast_cancer")    # by file stem
  data_dir("xenium")

Shell (slurm scripts): ``python benchmarks/paths.py <key | stem | file name>`` prints the full path;
``python benchmarks/paths.py --dir <name>`` prints a folder.
"""

from __future__ import annotations

import argparse
import re
from functools import lru_cache
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "datasets.yaml"


@lru_cache(maxsize=1)
def _config() -> dict:
    with open(CONFIG) as fh:
        return yaml.safe_load(fh)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def data_dir(name: str) -> Path:
    """A folder from ``dirs`` (xenium, extra, cross_modal)."""
    return _resolve(_config()["dirs"][name])


def dataset_keys() -> list[str]:
    return list(_config()["datasets"])


def indexed_keys() -> list[str]:
    """The 8 indexed sections, in figure order."""
    return list(_config()["indexed"])


def dataset_path(name: str) -> Path:
    """The .h5ad of a dataset, given its key, its file stem or its file name."""
    entries = _config()["datasets"]
    for key, e in entries.items():
        if name in (key, e["file"], Path(e["file"]).stem):
            return data_dir(e["dir"]) / e["file"]
    raise KeyError(f"unknown dataset '{name}'; add it to {CONFIG}")


DEFAULT_MAX_GENES = 800


def dataset_max_genes(name: str) -> int:
    """The index's gene cap for a dataset (key, stem or file name; a ``_seed<n>`` suffix is ignored)."""
    name = re.sub(r"_seed\d+$", "", Path(name).name.removesuffix(".h5ad"))
    for key, e in _config()["datasets"].items():
        if name in (key, Path(e["file"]).stem):
            return int(e.get("max_genes", DEFAULT_MAX_GENES))
    return DEFAULT_MAX_GENES


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="dataset key, file stem or file name")
    parser.add_argument("--dir", help="print a folder from dirs instead")
    args = parser.parse_args()
    print(data_dir(args.dir) if args.dir else dataset_path(args.name))


if __name__ == "__main__":
    main()
