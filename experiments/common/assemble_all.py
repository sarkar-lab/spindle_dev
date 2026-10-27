"""All figures in one deck: every figures/<fig>/<fig>.pptx -> figures/all_figures.pptx, one slide each.

Usage (login node): PYTHONPATH=experiments/common/.pylib python experiments/common/assemble_all.py [--draft]
--draft takes <fig>_draft.pptx where the user's copy does not exist yet (for a dry run only).

Slides are the 180 x 225 mm page (portrait). Each figure's first slide is deep-copied, shape by shape, so pictures
and text boxes stay editable; pictures are re-added to the new deck (the same bytes). A figure larger than the page
(Fig 1, a 16:9 slide) is scaled to fit: positions, sizes and font sizes are multiplied by the same factor. Figures run
in ORDER, then any other figure folder alphabetically.
"""

import copy
import io
import sys
from pathlib import Path

from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Mm

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "figures"
OUT = FIGURES / "all_figures.pptx"
PAGE_W, PAGE_H = Mm(180), Mm(225)
ORDER = ["fig1_overview", "fig2_cost", "fig3_whole_tile", "fig4_gene_set", "fig5_cross_platform", "fig6_breast",
         "fig7_skin", "fig8_pan_cancer"]
R_EMBED = qn("r:embed")
R_LINK = qn("r:link")


def figure_decks(draft, pdf_only=False):
    """(figure, deck) in ORDER; with pdf_only, a figure that has only its PDF (no pptx, e.g. Fig 1, exported from
    Google Slides by the user) is yielded with that PDF's path, for all_figures.pdf."""
    names = [d.name for d in FIGURES.iterdir() if d.is_dir() and not d.name.startswith(("_", "."))]
    names = [n for n in ORDER if n in names] + sorted(n for n in names if n not in ORDER)
    for fig in names:
        own, dr = FIGURES / fig / f"{fig}.pptx", FIGURES / fig / f"{fig}_draft.pptx"
        if own.exists():
            yield fig, own
        elif draft and dr.exists():
            yield fig, dr
        elif pdf_only and (FIGURES / fig / f"{fig}.pdf").exists():
            yield fig, FIGURES / fig / f"{fig}.pdf"
        else:
            print(f"skipped {fig}: no {own.name}")


def scale_element(el, f):
    """Multiply every offset, extent and font size under a top-level shape element by f."""
    for tag in ("a:off", "a:ext"):
        for node in el.iter(qn(tag)):
            for att in ("x", "y", "cx", "cy"):
                if node.get(att) is not None:
                    node.set(att, str(round(int(node.get(att)) * f)))
    for tag in ("a:chOff", "a:chExt"):  # a group's child frame keeps its own units
        for node in el.iter(qn(tag)):
            for att in ("x", "y", "cx", "cy"):
                if node.get(att) is not None:
                    node.set(att, str(round(int(node.get(att)) * f)))
    for tag in ("a:rPr", "a:defRPr", "a:endParaRPr"):
        for node in el.iter(qn(tag)):
            if node.get("sz") is not None:
                node.set("sz", str(max(100, round(int(node.get("sz")) * f))))
    for node in el.iter(qn("a:ln")):
        if node.get("w") is not None:
            node.set("w", str(round(int(node.get("w")) * f)))


def copy_slide(src_slide, dst_slide, f):
    src_part, dst_part = src_slide.part, dst_slide.part
    for shape in src_slide.shapes:
        el = copy.deepcopy(shape._element)
        for node in el.iter():
            for att in (R_EMBED, R_LINK):
                rid = node.get(att)
                if rid is None:
                    continue
                blob = src_part.related_part(rid).blob
                _, new_rid = dst_part.get_or_add_image_part(io.BytesIO(blob))
                node.set(att, new_rid)
        if f != 1:
            scale_element(el, f)
        dst_slide.shapes._spTree.append(el)


def main():
    draft = "--draft" in sys.argv
    out = Presentation()
    out.slide_width, out.slide_height = PAGE_W, PAGE_H
    n = 0
    for fig, path in figure_decks(draft):
        src = Presentation(str(path))
        f = min(1.0, PAGE_W / src.slide_width, PAGE_H / src.slide_height)
        copy_slide(src.slides[0], out.slides.add_slide(out.slide_layouts[6]), f)
        n += 1
        print(f"{fig}: {path.name}" + (f", scaled x {f:.3f} to fit the page" if f < 1 else ""))
    out.save(OUT)
    print(f"wrote {OUT.relative_to(ROOT)} ({n} slides)")
    assemble_pdf(draft)


def assemble_pdf(draft):
    """figures/all_figures.pdf: every rendered figure PDF (vector, render.py) on a 180 x 225 mm page, in deck order;
    a larger page (Fig 1) is scaled to fit, top-left aligned."""
    from pypdf import PageObject, PdfReader, PdfWriter, Transformation

    pw, ph = 180 / 25.4 * 72, 225 / 25.4 * 72
    out = PdfWriter()
    for fig, path in figure_decks(draft, pdf_only=True):
        pdf = path.with_suffix(".pdf")
        if not pdf.exists():
            print(f"all_figures.pdf: skipped {fig}: {pdf.name} not rendered (render.py)")
            continue
        src = PdfReader(pdf).pages[0]
        w, h = float(src.mediabox.width), float(src.mediabox.height)
        f = min(1.0, pw / w, ph / h)
        page = PageObject.create_blank_page(width=pw, height=ph)
        page.merge_transformed_page(src, Transformation().scale(f, f).translate(0, ph - h * f))
        out.add_page(page)
    out.compress_identical_objects()
    pdf_out = OUT.with_suffix(".pdf")
    with open(pdf_out, "wb") as fh:
        out.write(fh)
    print(f"wrote {pdf_out.relative_to(ROOT)} ({len(out.pages)} pages, {pdf_out.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
