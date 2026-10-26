"""Render a figure's edited pptx: figures/<fig>/<fig>.pptx -> figures/<fig>/<fig>.pdf + <fig>.png (RESULTS_PLAN.md §2).

Usage (login node): python experiments/common/render.py <fig> [--draft]
--draft renders <fig>_draft.pptx instead (to <fig>_draft.pdf / .png), for a look before the user's copy exists.

LibreOffice (headless, a throwaway profile) exports the slide to PDF with the pictures kept lossless at full
resolution (no downsampling, no JPEG); pdftoppm rasterises that PDF to a 600 dpi PNG. The PDF page must be 180 mm
wide (the slide size); anything else is an error.
"""

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"
DPI = 600
PDF_FILTER = "pdf:impress_pdf_Export:" + json.dumps({
    "ReduceImageResolution": {"type": "boolean", "value": "false"},
    "UseLosslessCompression": {"type": "boolean", "value": "true"},
    "ExportNotesPages": {"type": "boolean", "value": "false"},
})


def to_pdf(pptx, out_pdf):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        subprocess.run(["soffice", f"-env:UserInstallation=file://{tmp / 'profile'}", "--headless", "--convert-to",
                        PDF_FILTER, "--outdir", str(tmp), str(pptx)], check=True, capture_output=True)
        shutil.move(str(tmp / f"{pptx.stem}.pdf"), str(out_pdf))
    info = subprocess.run(["pdfinfo", str(out_pdf)], capture_output=True, text=True, check=True).stdout
    size = next(line for line in info.splitlines() if line.startswith("Page size"))
    w_pt, h_pt = (float(v) for v in size.split()[2:5:2])
    w_mm, h_mm = w_pt / 72 * 25.4, h_pt / 72 * 25.4
    if abs(w_mm - 180) > 0.5:
        raise RuntimeError(f"{out_pdf.name}: page is {w_mm:.1f} mm wide, not 180 mm")
    return w_mm, h_mm


def to_png(pdf, out_png):
    subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-singlefile", str(pdf), str(out_png.with_suffix(""))],
                   check=True)


def render(fig, draft=False):
    stem = f"{fig}_draft" if draft else fig
    pptx = FIGURES / fig / f"{stem}.pptx"
    if not pptx.exists():
        raise SystemExit(f"{pptx.relative_to(ROOT)} does not exist" +
                         ("" if draft else f" (copy {fig}_draft.pptx to it and edit the copy)"))
    pdf, png = pptx.with_suffix(".pdf"), pptx.with_suffix(".png")
    w, h = to_pdf(pptx, pdf)
    to_png(pdf, png)
    print(f"wrote {pdf.relative_to(ROOT)} and {png.relative_to(ROOT)} ({w:.0f} x {h:.0f} mm, {DPI} dpi)")


if __name__ == "__main__":
    render(sys.argv[1], draft="--draft" in sys.argv)
