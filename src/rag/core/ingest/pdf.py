"""PDF extraction.

Backends:
  * docling  - layout model, TableFormer tables, reading order, figures (best quality;
               needs `pip install rag-builder[docling]` and its models downloaded once).
  * native   - pdfplumber for digital pages (text lines, font-size headings, ruled
               tables, embedded images) and RapidOCR for scanned pages. No PyTorch.
  * auto     - docling when installed and working, otherwise native.

OCR is decided per page, not per file: a page with too little extractable text,
or mostly garbage characters, is rendered and OCR'd.
"""

from __future__ import annotations

import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block, ExtractedDoc
from rag.core.util import long_path, markdown_table, printable_ratio

_BULLET_RE = re.compile(r"^(•|▪|●|-|\*|\d{1,2}[.)]|[a-z][.)])\s")
_PAGE_NUM_RE = re.compile(r"^(page\s*)?\d{1,4}(\s*(of|/)\s*\d{1,4})?$", re.I)
_CAPTION_RE = re.compile(r"^(figure|fig\.|table)\s*\d", re.I)


class DoclingUnavailable(RuntimeError):
    pass


def extract_pdf(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    backend = ctx.config["pdf"]["backend"]
    warnings: list[str] = []
    if backend in ("auto", "docling"):
        try:
            from rag.core.ingest.pdf_docling import extract_pdf_docling

            return extract_pdf_docling(path, ctx)
        except ImportError as exc:
            if backend == "docling":
                raise DoclingUnavailable("pdf.backend = docling, but docling is not installed") from exc
        except Exception as exc:  # noqa: BLE001 - fall back, but say why
            if backend == "docling":
                raise
            warnings.append(f"docling failed ({type(exc).__name__}: {exc}); used native extractor")
    doc = extract_pdf_native(path, ctx)
    doc.warnings = warnings + doc.warnings
    return doc


# ---------------------------------------------------------------------------
# native backend
# ---------------------------------------------------------------------------
@dataclass
class _Line:
    text: str
    size: float
    top: float
    bottom: float
    x0: float
    bold: bool = False


@dataclass
class _PageItems:
    number: int
    lines: list[_Line] = field(default_factory=list)
    tables: list[tuple[float, list[list[str]]]] = field(default_factory=list)   # (top, rows)
    figures: list[tuple[float, tuple[float, float, float, float]]] = field(default_factory=list)
    ocr: bool = False
    ocr_blocks: list[Block] = field(default_factory=list)
    ocr_conf: float | None = None
    width: float = 0
    height: float = 0


def _inside(word: dict, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    cx = (word["x0"] + word["x1"]) / 2
    cy = (word["top"] + word["bottom"]) / 2
    return x0 - 1 <= cx <= x1 + 1 and top - 1 <= cy <= bottom + 1


def _find_gutter(words: list[dict], width: float) -> float | None:
    """Return the x of a two-column gutter, or None for single-column pages."""
    if len(words) < 40:
        return None
    best_x, best_hits = None, len(words)
    x = width * 0.35
    while x <= width * 0.65:
        hits = sum(1 for w in words if w["x0"] < x < w["x1"])
        if hits < best_hits:
            best_x, best_hits = x, hits
        x += 2
    if best_x is None or best_hits > 0.02 * len(words):
        return None
    left = sum(1 for w in words if w["x1"] <= best_x)
    right = len(words) - left
    if min(left, right) < 0.2 * len(words):
        return None
    return best_x


def _words_to_lines(words: list[dict]) -> list[_Line]:
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    groups: list[list[dict]] = []
    for w in words:
        if groups and abs(groups[-1][0]["top"] - w["top"]) <= 2.5:
            groups[-1].append(w)
        else:
            groups.append([w])
    lines = []
    for g in groups:
        g.sort(key=lambda w: w["x0"])
        sizes = [float(w.get("size") or 0) for w in g]
        fonts = [str(w.get("fontname") or "") for w in g]
        lines.append(
            _Line(
                text=" ".join(w["text"] for w in g),
                size=statistics.median(sizes) if sizes else 0.0,
                top=min(w["top"] for w in g),
                bottom=max(w["bottom"] for w in g),
                x0=min(w["x0"] for w in g),
                bold=sum("bold" in f.lower() for f in fonts) > len(fonts) / 2,
            )
        )
    return lines


def _render_page(pdoc, index: int, scale: float):
    page = pdoc[index]
    try:
        return page.render(scale=scale).to_pil()
    finally:
        page.close()


def extract_pdf_native(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    import pdfplumber
    import pypdfium2 as pdfium

    cfg = ctx.config["pdf"]
    warnings: list[str] = []
    pages: list[_PageItems] = []
    pdoc = pdfium.PdfDocument(long_path(path))
    try:
        with pdfplumber.open(long_path(path)) as pdf:
            title = (pdf.metadata or {}).get("Title") or None
            n_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                ctx.progress("page", {"page": i + 1, "pages": n_pages})
                items = _PageItems(number=i + 1, width=float(page.width), height=float(page.height))
                try:
                    raw = page.extract_text() or ""
                except Exception:  # noqa: BLE001 - broken text layers happen
                    raw = ""
                compact = re.sub(r"\s+", "", raw)
                if len(compact) < cfg["min_text_chars"] or printable_ratio(compact) < cfg["min_printable_ratio"]:
                    items.ocr = True
                    _ocr_page(pdoc, i, items, ctx)
                else:
                    _digital_page(page, items, ctx)
                pages.append(items)
                page.flush_cache()
    except Exception:
        pdoc.close()
        raise

    blocks = _assemble(pages, pdoc, ctx, warnings)
    pdoc.close()
    if not title or title.strip().lower() in {"untitled", "(anonymous)", "anonymous", "untitled document"} \
            or title.lower().startswith("microsoft word - "):
        title = next((b.text for b in blocks if b.kind == "heading" and b.level == 1), None)
    ocr_pages = sum(p.ocr for p in pages)
    if ocr_pages:
        warnings.append(f"{ocr_pages} of {len(pages)} pages read with OCR")
    return ExtractedDoc(
        file_type="pdf",
        title=title,
        blocks=blocks,
        n_pages=len(pages),
        extractor="pdf-native",
        warnings=warnings,
    )


def _digital_page(page, items: _PageItems, ctx: ExtractContext) -> None:
    cfg = ctx.config["pdf"]
    table_bboxes: list[tuple[float, float, float, float]] = []
    try:
        for table in page.find_tables():
            rows = [[(c or "").strip() for c in row] for row in table.extract()]
            rows = [r for r in rows if any(r)]
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                table_bboxes.append(table.bbox)
                items.tables.append((table.bbox[1], rows))
    except Exception:  # noqa: BLE001
        pass
    page_area = items.width * items.height or 1
    for img in page.images:
        bbox = (float(img["x0"]), float(img["top"]), float(img["x1"]), float(img["bottom"]))
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        if area / page_area >= cfg["figure_min_area"]:
            items.figures.append((bbox[1], bbox))
    words = page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False)
    words = [w for w in words if not any(_inside(w, b) for b in table_bboxes)]
    gutter = _find_gutter(words, items.width)
    if gutter is None:
        items.lines = _words_to_lines(words)
    else:
        left = [w for w in words if w["x1"] <= gutter]
        right = [w for w in words if w["x1"] > gutter]
        # Read the left column fully, then the right one; offset the right
        # column's positions so later y-sorting keeps that order.
        items.lines = _words_to_lines(left) + [
            _Line(ln.text, ln.size, ln.top + items.height, ln.bottom + items.height, ln.x0, ln.bold)
            for ln in _words_to_lines(right)
        ]


def _ocr_page(pdoc, index: int, items: _PageItems, ctx: ExtractContext) -> None:
    from rag.core.ingest.ocr import lines_to_paragraphs, ocr_lines

    ctx.progress("ocr", {"page": index + 1})
    image = _render_page(pdoc, index, float(ctx.config["pdf"]["ocr_scale"]))
    lines = ocr_lines(image)
    paras = lines_to_paragraphs(lines)
    items.ocr_conf = (sum(ln.conf for ln in lines) / len(lines)) if lines else 0.0
    for p in paras:
        items.ocr_blocks.append(
            Block(
                kind="heading" if p.is_heading else "paragraph",
                text=p.text,
                level=2 if p.is_heading else 0,
                page=items.number,
                meta={"ocr_conf": round(p.conf, 3), "y": p.y0},
            )
        )


def _furniture(pages: list[_PageItems], min_pages: int) -> set[str]:
    """Lines repeated at the top or bottom of many pages (headers, footers)."""
    digital = [p for p in pages if not p.ocr and p.lines]
    if len(digital) < min_pages:
        return set()

    def norm(t: str) -> str:
        return re.sub(r"\d+", "#", t.strip().lower())

    counts: Counter[str] = Counter()
    for p in digital:
        ordered = sorted(p.lines, key=lambda ln: ln.top)
        edge = {norm(ln.text) for ln in ordered[:2] + ordered[-2:]}
        counts.update(edge)
    threshold = max(min_pages, len(digital) // 2)
    return {t for t, c in counts.items() if c >= threshold}


def _heading_levels(pages: list[_PageItems]) -> tuple[float, dict[float, int]]:
    sizes: Counter[float] = Counter()
    for p in pages:
        for ln in p.lines:
            sizes[round(ln.size, 1)] += len(ln.text)
    if not sizes:
        return 0.0, {}
    body = sizes.most_common(1)[0][0]
    bigger = sorted({s for s in sizes if s >= body * 1.15}, reverse=True)
    return body, {s: min(i + 1, 4) for i, s in enumerate(bigger)}


def _is_heading(ln: _Line, body: float, levels: dict[float, int]) -> int:
    text = ln.text.strip()
    if not text or len(text) > 120 or len(text.split()) > 15 or text.endswith((".", ",", ";")):
        return 0
    level = levels.get(round(ln.size, 1), 0)
    if level:
        return level
    if ln.bold and ln.size >= body * 0.98 and len(text.split()) <= 10 and not _BULLET_RE.match(text):
        return min(len(levels) + 1, 4) if levels else 2
    return 0


def _assemble(pages: list[_PageItems], pdoc, ctx: ExtractContext, warnings: list[str]) -> list[Block]:
    from rag.core.ingest.figures import make_figure_block

    cfg = ctx.config["pdf"]
    furniture = _furniture(pages, int(cfg["furniture_min_pages"]))
    body, levels = _heading_levels(pages)
    blocks: list[Block] = []
    figure_index = 0

    for p in pages:
        if p.ocr:
            for b in p.ocr_blocks:
                b.meta["page_ocr_conf"] = p.ocr_conf
            blocks.extend(p.ocr_blocks)
            continue

        events: list[tuple[float, str, object]] = []
        for ln in p.lines:
            normed = re.sub(r"\d+", "#", ln.text.strip().lower())
            if normed in furniture or _PAGE_NUM_RE.match(ln.text.strip()):
                continue
            events.append((ln.top, "line", ln))
        for top, rows in p.tables:
            events.append((top, "table", rows))
        for top, bbox in p.figures:
            events.append((top, "figure", bbox))
        events.sort(key=lambda e: e[0])

        para: list[_Line] = []

        def flush() -> None:
            if not para:
                return
            text = ""
            for ln in para:
                if text.endswith("-") and ln.text[:1].islower():
                    text = text[:-1] + ln.text
                else:
                    text = f"{text} {ln.text}".strip()
            blocks.append(Block(kind="paragraph", text=text, page=p.number))
            para.clear()

        page_image = None
        consumed: set[int] = set()
        for idx, (top, kind, obj) in enumerate(events):
            if idx in consumed:
                continue
            if kind == "line":
                ln: _Line = obj  # type: ignore[assignment]
                level = _is_heading(ln, body, levels)
                if level:
                    flush()
                    blocks.append(Block(kind="heading", text=ln.text.strip(), level=level, page=p.number))
                    continue
                if para:
                    prev = para[-1]
                    gap = ln.top - prev.bottom
                    if gap > max(prev.bottom - prev.top, 1) * 0.9 or _BULLET_RE.match(ln.text):
                        flush()
                para.append(ln)
            elif kind == "table":
                flush()
                rows: list[list[str]] = obj  # type: ignore[assignment]
                caption = _caption_near(blocks, events, idx, consumed)
                headers, data = rows[0], rows[1:]
                md = markdown_table(headers, data)
                text = (f"**{caption}**\n\n" if caption else "") + md
                blocks.append(Block(kind="table", text=text, page=p.number,
                                    meta={"headers": headers, "rows": data, "caption": caption}))
            elif kind == "figure":
                flush()
                if page_image is None:
                    page_image = _render_page(pdoc, p.number - 1, 2.0)
                x0, t, x1, b = obj  # type: ignore[misc]
                s = 2.0
                crop = page_image.crop((int(x0 * s), int(t * s), int(x1 * s), int(b * s)))
                if crop.width < 20 or crop.height < 20:
                    continue
                figure_index += 1
                caption = _caption_after(events, idx, consumed)
                blocks.append(make_figure_block(crop, ctx, page=p.number, index=figure_index, caption=caption))
        flush()

    return _drop_repeated_figures(blocks, len(pages), int(cfg["furniture_min_pages"]))


def _caption_near(blocks: list[Block], events, idx: int, consumed: set[int]) -> str | None:
    if blocks and blocks[-1].kind == "paragraph" and _CAPTION_RE.match(blocks[-1].text):
        return blocks.pop().text
    return _caption_after(events, idx, consumed)


def _caption_after(events, idx: int, consumed: set[int]) -> str | None:
    if idx + 1 < len(events) and events[idx + 1][1] == "line":
        text = events[idx + 1][2].text.strip()
        if _CAPTION_RE.match(text):
            consumed.add(idx + 1)
            return text
    return None


def _drop_repeated_figures(blocks: list[Block], n_pages: int, min_pages: int) -> list[Block]:
    prints = Counter(b.meta.get("fingerprint") for b in blocks if b.kind == "figure")
    threshold = max(min_pages, n_pages // 2)
    repeated = {fp for fp, c in prints.items() if fp and c >= threshold}
    return [b for b in blocks if not (b.kind == "figure" and b.meta.get("fingerprint") in repeated)]
