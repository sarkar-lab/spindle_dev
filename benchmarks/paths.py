"""Dataset paths, read from datasets.yaml at the repository root (the only place they are written).

  from paths import dataset_path, data_dir
  dataset_path("breast_cancer")                 # by key
  dataset_path("xenium_human_breast_cancer")    # by file stem
  data_dir("xenium")

Shell (slurm scripts): ``python benchmarks/paths.py <key | stem | file name>`` prints the full path;
``python benchmarks/paths.py --dir <name>`` prints a folder;
``python benchmarks/paths.py <name> --seed-symlink <n>`` prints the seed-n symlink (``make_seed_symlink``).

Importing this module also puts ``src/``, ``benchmarks/`` and every ``benchmarks/<topic>/`` folder on
``sys.path``, so benchmark modules import each other by name (``import experiment_common``) from any
subfolder. A script in ``benchmarks/<topic>/`` starts with
``sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`` and ``import paths``.
"""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from pathlib import Path

import yaml

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
CONFIG = PROJECT_ROOT / "datasets.yaml"

for _p in (PROJECT_ROOT / "src", BENCH_DIR,
           *sorted(d for d in BENCH_DIR.iterdir() if d.is_dir() and not d.name.startswith(("_", ".")))):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


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


SYMLINK_DIR = PROJECT_ROOT / "results" / "indexes" / "_seed_symlinks"


def make_seed_symlink(dataset_path: Path, seed: int) -> Path:
    """Return a symlink named ``{stem}_seed{n}.h5ad`` pointing at ``dataset_path``.

    ``build_indexes.py`` and ``holdout_core.py`` derive ``dataset_name`` from the path stem, so a
    seed-suffixed symlink is all it takes to give each seed (0-4) its own index, covariances,
    run log and ground-truth cache.
    """
    SYMLINK_DIR.mkdir(parents=True, exist_ok=True)
    link_path = SYMLINK_DIR / f"{dataset_path.stem}_seed{seed}.h5ad"
    if link_path.is_symlink() or link_path.exists():
        if link_path.resolve() != dataset_path.resolve():
            link_path.unlink()
            link_path.symlink_to(dataset_path)
    else:
        link_path.symlink_to(dataset_path)
    return link_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("name", nargs="?", help="dataset key, file stem or file name")
    parser.add_argument("--dir", help="print a folder from dirs instead")
    parser.add_argument("--seed-symlink", type=int, metavar="SEED",
                        help="print (and create) the seed-suffixed symlink to the dataset instead")
    args = parser.parse_args()
    if args.dir:
        print(data_dir(args.dir))
    elif args.seed_symlink is not None:
        print(make_seed_symlink(dataset_path(args.name), args.seed_symlink))
    else:
        print(dataset_path(args.name))


if __name__ == "__main__":
    main()
