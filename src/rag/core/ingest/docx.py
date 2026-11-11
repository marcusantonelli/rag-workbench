"""Word documents: .docx natively (python-docx), legacy .doc through LibreOffice."""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

from rag.core.ingest.context import ExtractContext, UnsupportedFile
from rag.core.types import Block, ExtractedDoc
from rag.core.util import long_path, markdown_table

_HEADING_RE = re.compile(r"^heading\s*(\d)", re.I)
_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}


def _style_level(style_name: str) -> int:
    if not style_name:
        return 0
    if style_name.lower() == "title":
        return 1
    m = _HEADING_RE.match(style_name)
    return min(int(m.group(1)), 6) if m else 0


_W = "{%s}" % _NS["w"]
_SKIP = {"del", "moveFrom", "drawing", "pict", "object", "instrText", "delText", "rPr", "pPr", "sdtPr",
         "sdtEndPr", "fldData", "commentReference", "footnoteReference", "endnoteReference"}


def _run_font(r) -> str:
    fonts = r.find(f"{_W}rPr/{_W}rFonts")
    if fonts is None:
        return ""
    return fonts.get(f"{_W}ascii") or fonts.get(f"{_W}hAnsi") or fonts.get(f"{_W}cs") or ""


def _legacy_checkbox(fld) -> str | None:
    """☒ or ☐ for a legacy FORMCHECKBOX field (its state is in the field's data, not its text)."""
    box = fld.find(f"{_W}ffData/{_W}checkBox")
    if box is None:
        return None
    for tag in ("checked", "default"):
        el = box.find(f"{_W}{tag}")
        if el is not None:
            return "☐" if el.get(f"{_W}val") in ("0", "false", "off") else "☒"
    return "☐"


def _para_text(p) -> str:
    """A paragraph's text as Word shows it: runs inside content controls, hyperlinks, tracked
    insertions and simple fields too (python-docx skips those), deleted text left out, tabs and
    breaks kept, Wingdings boxes and ticks as ☐ ☑ ☒ ✓ ✗, legacy form checkboxes by their state."""
    from rag.core.ingest.forms import checkbox_symbol

    out: list[str] = []

    def walk(el) -> None:
        for child in el:
            tag = child.tag.split("}")[-1] if isinstance(child.tag, str) else ""
            if tag in _SKIP:
                continue
            if tag == "r":
                font = _run_font(child)
                for part in child:
                    ptag = part.tag.split("}")[-1] if isinstance(part.tag, str) else ""
                    if ptag == "t":
                        text = part.text or ""
                        if "wingdings" in font.lower():
                            text = "".join(checkbox_symbol(ch, font) or ch for ch in text)
                        out.append(text)
                    elif ptag == "tab":
                        out.append("\t")
                    elif ptag in ("br", "cr"):
                        out.append("\n")
                    elif ptag == "noBreakHyphen":
                        out.append("-")
                    elif ptag == "sym":
                        sym_font = part.get(f"{_W}font") or ""
                        try:
                            ch = chr(int(part.get(f"{_W}char") or "0", 16))
                        except ValueError:
                            continue
                        out.append(checkbox_symbol(ch, sym_font) or ("" if 0xF000 <= ord(ch) <= 0xF0FF else ch))
                    elif ptag == "fldChar" and part.get(f"{_W}fldCharType") == "begin":
                        box = _legacy_checkbox(part)
                        if box:
                            out.append(box)
            elif tag in ("hyperlink", "ins", "smartTag", "sdt", "sdtContent", "customXml", "fldSimple",
                         "moveTo", "dir", "bdo", "AlternateContent", "Choice"):
                walk(child)

    walk(p)
    return "".join(out)


def _block_elements(parent):
    """Paragraphs and tables in document order, including those inside block-level content
    controls (Word forms and templates put whole sections in them)."""
    for child in parent.iterchildren():
        tag = child.tag.split("}")[-1] if isinstance(child.tag, str) else ""
        if tag in ("p", "tbl"):
            yield child
        elif tag in ("sdt", "sdtContent", "customXml"):
            yield from _block_elements(child if tag != "sdt" else (child.find(f"{_W}sdtContent") if child.find(f"{_W}sdtContent") is not None else child))


def _cell_text(cell) -> str:
    paras = [p for p in _block_elements(cell._tc) if p.tag == f"{_W}p"]
    return " ".join(t for t in (_para_text(p).strip() for p in paras) if t)


def _table_rows(table) -> list[list[str]]:
    """Rows of cell text. A merged cell repeats in every column it spans, so columns stay aligned
    (python-docx returns the same cell for each spanned position), except a cell merged across
    the whole row (a title or a part name inside the table), which keeps its text in the first
    column only, as a PDF's spanning cell does."""
    rows: list[list[str]] = []
    for row in table.rows:
        cells = [_cell_text(cell) for cell in row.cells]
        if not any(cells):
            continue
        tcs = [cell._tc for cell in row.cells]
        if len(cells) > 1 and all(tc is tcs[0] for tc in tcs):
            cells = [cells[0]] + [""] * (len(cells) - 1)
        rows.append(cells)
    return rows


def _split_header(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """Header and data rows; a two-row header (merged group cells over sub-columns)
    becomes "Group / Sub" column names."""
    head = rows[0]
    merged = any(a and a == b for a, b in zip(head, head[1:]))
    if merged and len(rows) >= 3:
        sub = rows[1]
        names = []
        for parent, child in zip(head, sub + [""] * (len(head) - len(sub))):
            if child and parent and child != parent:
                names.append(f"{parent} / {child}")
            else:
                names.append(child or parent)
        return names, rows[2:]
    return head, rows[1:]


def extract_docx(path: Path, ctx: ExtractContext, *, file_type: str = "docx") -> ExtractedDoc:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    from rag.core.ingest.figures import make_figure_block

    from rag.core.ingest import docx_shapes as shapes
    from rag.core.ingest.figures import make_diagram_block
    from rag.core.ingest.grids import grid_blocks
    from rag.core.ingest.layout_table import drop_blank_rows

    document = docx.Document(long_path(path))
    blocks: list[Block] = []
    figure_index = 0
    min_px = 64
    has_title = False
    last_level = 0

    elements = list(_block_elements(document.element.body))
    # Shapes first: a flowchart is often drawn as separate floating shapes anchored to several
    # paragraphs, so the single shapes of each section (between headings) are tried together.
    drawings: dict[int, list] = {}
    loose: dict[int, list[list]] = {}              # section -> [(element index, shape), ...]
    section = 0
    for i, el in enumerate(elements):
        if el.tag != f"{_W}p":
            continue
        style = Paragraph(el, document).style
        if _style_level(style.name if style is not None else "") and _para_text(el).strip():
            section += 1
        found = shapes.read_drawings(el, i, _para_text)
        if found:
            drawings[i] = found
            for d in found:
                if len(d.shapes) == 1:
                    loose.setdefault(section, []).append((i, d.shapes[0]))
    together: dict[int, object] = {}               # element index -> diagram drawn there
    consumed: set[int] = set()                     # id() of shapes read as part of one
    for items in loose.values():
        if len(items) >= 3:
            made = shapes.to_diagram([sh for _, sh in items])
            if made is not None:
                d, used = made
                together[next(i for i, sh in items if sh in used)] = d
                consumed.update(id(sh) for sh in used)

    def add_diagram(d) -> None:
        nonlocal figure_index
        figure_index += 1
        blocks.append(make_diagram_block(None, d, ctx, page=None, index=figure_index, caption=None))

    for i, child in enumerate(elements):
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = _para_text(child).strip()
            style = para.style.name if para.style is not None else ""
            level = _style_level(style)
            if text:
                if level:
                    if style.lower() == "title":
                        has_title = True
                    elif has_title:
                        level = min(level + 1, 6)   # Title sits above Heading 1
                    blocks.append(Block(kind="heading", text=text, level=level))
                    last_level = level
                elif "list" in style.lower() or child.find(".//w:numPr", _NS) is not None:
                    blocks.append(Block(kind="paragraph", text=f"- {text}"))
                else:
                    blocks.append(Block(kind="paragraph", text=text))
            if i in together:
                add_diagram(together[i])
            rest = []                     # shapes not in a diagram: their text, in reading order
            for d in drawings.get(i, []):
                mine = [sh for sh in d.shapes if id(sh) not in consumed]
                made = shapes.to_diagram(mine) if len(mine) >= 2 else None
                if made is not None:
                    add_diagram(made[0])
                    mine = [sh for sh in mine if sh not in made[1]]
                rest += mine
            blocks.extend(Block(kind="paragraph", text=t) for t in shapes.loose_text(rest))
            if i not in drawings:
                blocks.extend(Block(kind="paragraph", text=t) for t in shapes.vml_text(child, _para_text))
            for blip in child.findall(".//a:blip", _NS):
                rid = blip.get(f"{{{_NS['r']}}}embed")
                part = document.part.related_parts.get(rid) if rid else None
                if part is None or not getattr(part, "content_type", "").startswith("image/"):
                    continue
                try:
                    img = Image.open(io.BytesIO(part.blob))
                    img.load()
                except Exception:  # noqa: BLE001 - EMF/WMF and friends
                    continue
                if img.width < min_px or img.height < min_px:
                    continue
                figure_index += 1
                blocks.append(make_figure_block(img, ctx, page=None, index=figure_index, caption=None))
        elif tag == "tbl":
            rows = drop_blank_rows(_table_rows(Table(child, document)))
            if not rows:
                continue
            # the same rules as a PDF's ruled tables (title row, part names, panels, forms)
            made = grid_blocks(rows, None, None, split_header=_split_header)
            if made and made[0].meta.get("title_row") and made[0].meta.get("caption"):
                blocks.append(Block(kind="heading", text=made[0].meta["caption"], level=min(last_level + 1, 6)))
            blocks.extend(made)

    title = document.core_properties.title or next(
        (b.text for b in blocks if b.kind == "heading" and b.level == 1), None
    )
    return ExtractedDoc(file_type=file_type, title=title or None, blocks=blocks, extractor="docx-native")


def _find_soffice(configured: str) -> str | None:
    if configured and Path(configured).exists():
        return configured
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    for candidate in (
        r"C:\Program Files\LibreOffice\program\soffice.exe",
        r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
    ):
        if os.path.exists(candidate):
            return candidate
    return None


def extract_doc(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    soffice = _find_soffice(ctx.config["ingest"].get("libreoffice_path", ""))
    if soffice is None:
        raise UnsupportedFile("legacy .doc needs LibreOffice installed (not found)")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [soffice, "--headless", "--convert-to", "docx", "--outdir", tmp, long_path(path)],
            check=True, capture_output=True, timeout=300,
        )
        out = Path(tmp) / (Path(path).stem + ".docx")
        if not out.exists():
            raise UnsupportedFile("LibreOffice did not produce a .docx")
        doc = extract_docx(out, ctx, file_type="doc")
        doc.extractor = "doc-libreoffice"
        return doc
