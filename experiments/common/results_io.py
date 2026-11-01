"""Where the result notebooks write, and the manifest that says what each file is.

    import results_io as rio
    out = rio.topic("whole_tile")                         # results/whole_tile/ (created)
    rio.write_csv(df, "whole_tile", "nx.csv", "NX5 and Overlap per held-out query, method and g")
    rio.write_manifest("whole_tile", "R2_whole_tile.ipynb", inputs=["results/_cache"])

manifest.json (one per topic): notebook, date, code stamp, inputs, peak resident memory of the notebook's process,
and every file written through this module with its description, rows and size. Re-running a notebook adds to or
replaces its entries.
"""

import json
import resource
import time
from pathlib import Path

import data

RESULTS = data.RESULTS
_FILES = {}


def topic(name):
    d = RESULTS / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def per_dataset(name, key):
    """results/<topic>/per_dataset/<key>/ (created); per-seed files are named seed<k>.*."""
    d = topic(name) / "per_dataset" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def register(name, path, description, rows=None):
    path = Path(path)
    _FILES.setdefault(name, {})[str(path.relative_to(topic(name)))] = {
        "description": description, "rows": rows, "bytes": path.stat().st_size}


def write_csv(df, name, file, description):
    path = topic(name) / file
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    register(name, path, description, rows=len(df))
    print(f"wrote {path.relative_to(data.ROOT)}: {len(df)} rows")
    return path


def peak_rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def write_manifest(name, notebook, inputs=(), notes=""):
    path = topic(name) / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    files = {**old.get("files", {}), **_FILES.get(name, {})}
    man = {"topic": name, "notebook": notebook, "date": time.strftime("%Y-%m-%d %H:%M"), **data.stamp(),
           "inputs": list(inputs), "peak_rss_gb": round(peak_rss_gb(), 2), "notes": notes,
           "files": dict(sorted(files.items()))}
    path.write_text(json.dumps(man, indent=1))
    print(f"wrote {path.relative_to(data.ROOT)}: {len(files)} files; peak memory {man['peak_rss_gb']} GB")
    return man
