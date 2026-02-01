#!/usr/bin/env python3
"""Convert notebooks to HTML and generate a docs/index.html linking to them.

Usage:
    python scripts/build_notebooks.py

This will:
  - find all `.ipynb` files under `ISMB_notebook/` and `examples/` (recursively)
  - convert each to HTML and write to `docs/examples/html/<notebook_name>.html`
  - generate `docs/index.html` with links to each generated HTML file
"""
from __future__ import annotations

import os
from pathlib import Path
import nbformat
from nbconvert import HTMLExporter
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
SEARCH_DIRS = [ROOT / "ISMB_notebook", ROOT / "examples"]
OUT_DIR = ROOT / "docs" / "examples" / "html"


def find_notebooks(paths):
    notebooks = []
    for p in paths:
        if not p.exists():
            continue
        for nb in p.rglob("*.ipynb"):
            # Skip checkpoint files
            if nb.name.startswith(".") or nb.name.endswith("-checkpoint.ipynb"):
                continue
            notebooks.append(nb)
    return sorted(notebooks)


def convert_notebook(nb_path: Path, out_path: Path) -> None:
    nb_node = nbformat.read(nb_path, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(nb_node)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(body, encoding="utf-8")


def build_index_html(html_files, out_index: Path) -> None:
    out_index.parent.mkdir(parents=True, exist_ok=True)
    # include README.md content (plain) if available
    readme_txt = ""
    try:
        readme_txt = (ROOT / "README.md").read_text(encoding="utf-8")
    except Exception:
        readme_txt = ""

    lines = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "  <meta charset=\"utf-8\">",
        "  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
        "  <title>spindle_dev examples</title>",
        "  <style>body{font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial;margin:40px} a{display:block;margin:6px 0}</style>",
        "</head>",
        "<body>",
        f"<h1>spindle_dev examples — generated {datetime.utcnow().isoformat()} UTC</h1>",
        "<p>Click an example to open the generated HTML.</p>",
        "<section>",
        "<h2>README</h2>",
        "<pre style=\"white-space:pre-wrap; background:#f8f8f8; padding:12px; border-radius:6px;\">",
        *(readme_txt.splitlines()),
        "</pre>",
        "</section>",
        "<ul>",
    ]

    for rel in html_files:
        safe = rel.replace("&", "&amp;").replace("<", "&lt;")
        lines.append(f"  <li><a href=\"{safe}\">{safe}</a></li>")

    lines += ["</ul>", "</body>", "</html>"]
    out_index.write_text("\n".join(lines), encoding="utf-8")


def main():
    nbs = find_notebooks(SEARCH_DIRS)
    if not nbs:
        print("No notebooks found under ISMB_notebook/ or examples/")
        return
    html_paths = []
    for nb in nbs:
        rel = nb.relative_to(ROOT)
        # preserve subpath under OUT_DIR
        out_rel = rel.with_suffix(".html")
        out_file = OUT_DIR.parent / out_rel
        print(f"Converting {nb} -> {out_file}")
        try:
            convert_notebook(nb, out_file)
            # URL path relative to docs/ (Pages will serve from docs/)
            url = os.path.relpath(out_file, start=ROOT / "docs")
            html_paths.append(url.replace(os.path.sep, "/"))
        except Exception as exc:
            print(f"Failed converting {nb}: {exc}")

    # generate docs/index.html
    index_file = ROOT / "docs" / "index.html"
    build_index_html(html_paths, index_file)
    print(f"Wrote index to {index_file}. {len(html_paths)} files linked.")


if __name__ == "__main__":
    main()
