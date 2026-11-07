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


def _table_rows(table) -> list[list[str]]:
    rows: list[list[str]] = []
    for row in table.rows:
        cells: list[str] = []
        prev_id = None
        for cell in row.cells:
            # Merged cells repeat the same underlying cell object; keep one.
            if id(cell._tc) == prev_id:
                continue
            prev_id = id(cell._tc)
            cells.append(" ".join(p.text.strip() for p in cell.paragraphs if p.text.strip()))
        if any(cells):
            rows.append(cells)
    return rows


def extract_docx(path: Path, ctx: ExtractContext, *, file_type: str = "docx") -> ExtractedDoc:
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    from rag.core.ingest.figures import make_figure_block

    document = docx.Document(long_path(path))
    blocks: list[Block] = []
    figure_index = 0
    min_px = 64
    has_title = False

    for child in document.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            para = Paragraph(child, document)
            text = para.text.strip()
            style = para.style.name if para.style is not None else ""
            level = _style_level(style)
            if text:
                if level:
                    if style.lower() == "title":
                        has_title = True
                    elif has_title:
                        level = min(level + 1, 6)   # Title sits above Heading 1
                    blocks.append(Block(kind="heading", text=text, level=level))
                elif "list" in style.lower() or child.find(".//w:numPr", _NS) is not None:
                    blocks.append(Block(kind="paragraph", text=f"- {text}"))
                else:
                    blocks.append(Block(kind="paragraph", text=text))
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
            rows = _table_rows(Table(child, document))
            if not rows:
                continue
            if len(rows) == 1 or max(len(r) for r in rows) == 1:
                blocks.append(Block(kind="paragraph", text=" / ".join(" ".join(r) for r in rows)))
                continue
            headers, data = rows[0], rows[1:]
            blocks.append(Block(kind="table", text=markdown_table(headers, data),
                                meta={"headers": headers, "rows": data, "caption": None}))

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
