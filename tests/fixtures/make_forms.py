"""Forms: a fillable PDF, a printed form laid out as a ruled grid, and a Word form with content
controls, checkboxes of three kinds, tracked changes and a label/value table."""

from __future__ import annotations

import copy
from pathlib import Path


def fillable_pdf(path: Path) -> Path:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    W, H = letter
    c = canvas.Canvas(str(path), pagesize=letter)
    f = c.acroForm
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, H - 72, "Job Sheet")
    c.setFont("Helvetica", 10)
    c.drawString(72, H - 110, "Lot number:")
    f.textfield(name="lot_no", value="L26-3007", x=160, y=H - 116, width=120, height=18)
    c.drawString(300, H - 110, "Date:")
    f.textfield(name="date", value="2026-11-03", x=340, y=H - 116, width=100, height=18)
    c.drawString(72, H - 145, "Gasket inspected")
    f.checkbox(name="seal_ok", x=200, y=H - 150, size=14, checked=True)
    c.drawString(240, H - 145, "Rework required")
    f.checkbox(name="rework", x=340, y=H - 150, size=14, checked=False)
    c.drawString(72, H - 180, "Disposition:")
    f.radio(name="dispo", value="release", selected=False, x=160, y=H - 185, size=12)
    c.drawString(176, H - 180, "Release")
    f.radio(name="dispo", value="mrb", selected=True, x=230, y=H - 185, size=12)
    c.drawString(246, H - 180, "MRB")
    c.drawString(72, H - 215, "Oven:")
    f.choice(name="oven", value="OV-2", options=["OV-1", "OV-2", "OV-3"], x=160, y=H - 221, width=80, height=18)
    c.drawString(72, H - 250, "Comments:")
    f.textfield(name="comments", value="Leak at corner 3, retest after bake", x=160, y=H - 256, width=300, height=18)
    c.drawString(72, H - 285, "Operator:")
    f.textfield(name="operator", value="", x=160, y=H - 291, width=150, height=18)
    # a ruled grid with fields in its cells
    c.setFont("Helvetica-Bold", 11)
    c.drawString(72, H - 330, "Leak test results")
    c.setFont("Helvetica", 9)
    heads, top = ["Unit", "Slow leak (atm cc/s)", "Fast leak"], H - 345
    for k, hd in enumerate(heads):
        c.rect(72 + 150 * k, top - 20, 150, 20)
        c.drawString(76 + 150 * k, top - 14, hd)
    for r in range(2):
        y = top - 20 * (r + 2)
        for k in range(3):
            c.rect(72 + 150 * k, y, 150, 20)
        c.drawString(76, y + 6, f"Unit {r + 1}")
        f.textfield(name=f"fine_{r}", value=["2.1E-9", "6.0E-8"][r], x=224, y=y + 2, width=140, height=16, borderWidth=0)
        f.checkbox(name=f"gross_{r}", x=376, y=y + 3, size=12, checked=r == 1, borderWidth=0)
    c.showPage()
    c.save()
    return path


def grid_form_pdf(path: Path) -> Path:
    """A printed (flattened) job sheet header: label cells with values beside them."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    W, H = letter
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, H - 72, "Assembly Job Sheet")
    c.setFont("Helvetica", 9)
    rows = [["Lot number:", "L26-3011", "Date:", "2026-11-04"],
            ["Part number:", "PX-7250", "Quantity:", "48"],
            ["Operator:", "J. Ruiz", "Shift:", "2"]]
    for r, row in enumerate(rows):
        for k, val in enumerate(row):
            x, y = 72 + 117 * k, H - 100 - 20 * r
            c.rect(x, y - 20, 117, 20)
            c.drawString(x + 4, y - 14, val)
    c.drawString(72, H - 190, "Assemble the lot within four hours of the job sheet date.")
    c.showPage()
    c.save()
    return path


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W14_NS = "http://schemas.microsoft.com/office/word/2010/wordml"


def _w(tag: str) -> str:
    return f"{{{W_NS}}}{tag}"


def _el(tag: str, attrs: dict | None = None, text: str | None = None, children=()):
    from lxml import etree

    el = etree.Element(_w(tag) if ":" not in tag else tag, nsmap={"w": W_NS, "w14": W14_NS})
    for k, v in (attrs or {}).items():
        el.set(_w(k) if ":" not in k else k, v)
    if text is not None:
        el.text = text
    for ch in children:
        el.append(ch)
    return el


def _run(text: str, font: str | None = None):
    kids = []
    if font:
        kids.append(_el("rPr", children=[_el("rFonts", {"ascii": font, "hAnsi": font})]))
    t = _el("t", text=text)
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return _el("r", children=kids + [t])


def word_form(path: Path) -> Path:
    from docx import Document

    d = Document()
    d.add_heading("Bake Record", 1)
    body = d.element.body
    sect = body[-1]                                          # sectPr stays last

    def add(el):
        sect.addprevious(el)

    # 1. a paragraph with an inline content control holding the operator's name
    add(_el("p", children=[_run("Operator: "), _el("sdt", children=[
        _el("sdtPr", children=[_el("alias", {"val": "Operator"})]),
        _el("sdtContent", children=[_run("M. Okafor")])])]))
    # 2. content-control checkbox (Word 2010+), a Wingdings symbol checkbox, a legacy form checkbox
    cb = _el("sdt", children=[
        _el("sdtPr", children=[_el(f"{{{W14_NS}}}checkbox", children=[
            _el(f"{{{W14_NS}}}checked", {f"{{{W14_NS}}}val": "1"})])]),
        _el("sdtContent", children=[_run("☒", "MS Gothic")])])
    add(_el("p", children=[cb, _run(" Oven door seal checked")]))
    add(_el("p", children=[_el("r", children=[_el("sym", {"font": "Wingdings", "char": "F0FE"})]),
                           _run(" Calibration sticker current"),
                           _run("  "), _run("o", "Wingdings"), _run(" Rework needed")]))
    begin = _el("fldChar", {"fldCharType": "begin"}, children=[_el("ffData", children=[
        _el("name", {"val": "Check1"}), _el("checkBox", children=[_el("sizeAuto"), _el("default", {"val": "1"})])])])
    add(_el("p", children=[_el("r", children=[begin]), _el("r", children=[_el("instrText", text=" FORMCHECKBOX ")]),
                           _el("r", children=[_el("fldChar", {"fldCharType": "end"})]),
                           _run(" Air flow verified")]))
    # 3. tracked changes: inserted text kept, deleted text dropped
    add(_el("p", children=[_run("Bake time: "), _el("del", {"id": "1", "author": "a", "date": "2026-01-01T00:00:00Z"},
                                                    children=[_el("r", children=[_el("delText", text="2 hours")])]),
                           _el("ins", {"id": "2", "author": "a", "date": "2026-01-01T00:00:00Z"}, children=[_run("4 hours")])]))
    # 4. a whole section inside a block-level content control
    inner = d.add_paragraph("Released by QA after the bake log was reviewed.")
    inner_el = copy.deepcopy(inner._p)
    inner._p.getparent().remove(inner._p)
    add(_el("sdt", children=[_el("sdtPr"), _el("sdtContent", children=[inner_el])]))
    # 5. a label/value grid
    t = d.add_table(rows=2, cols=4)
    for r, row in enumerate([["Lot number:", "L26-3020", "Oven:", "OV-3"], ["Start:", "08:15", "End:", "12:15"]]):
        for k, v in enumerate(row):
            t.cell(r, k).text = v
    d.save(path)
    return path


def make_forms(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    return {"fillable": fillable_pdf(root / "Job sheet fillable.pdf"),
            "grid": grid_form_pdf(root / "Job sheet printed.pdf"),
            "word": word_form(root / "Bake record.docx")}
