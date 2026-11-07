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
_PAGE_REF_RE = re.compile(r"\bpage\s+\d{1,4}(\s*(of|/)\s*\d{1,4})?\b", re.I)
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


def _group_lines(words: list[dict]) -> list[list[dict]]:
    words = sorted(words, key=lambda w: (round(w["top"]), w["x0"]))
    groups: list[list[dict]] = []
    for w in words:
        if groups and abs(groups[-1][0]["top"] - w["top"]) <= 2.5:
            groups[-1].append(w)
        else:
            groups.append([w])
    return groups


def _has_channel(words: list[dict]) -> bool:
    """True when an empty vertical channel splits most lines (a table, not prose)."""
    lines = _group_lines(words)
    if len(lines) < 3:
        return False
    x0 = min(w["x0"] for w in words)
    x1 = max(w["x1"] for w in words)
    span = x1 - x0
    x = x0 + 0.1 * span
    while x <= x1 - 0.1 * span:
        crossing = sum(1 for ln in lines if any(w["x0"] < x < w["x1"] for w in ln))
        both = sum(1 for ln in lines if any(w["x1"] <= x for w in ln) and any(w["x0"] >= x for w in ln))
        if crossing <= 1 and both >= 0.6 * len(lines):
            return True
        x += 2
    return False


def _find_gutter(words: list[dict], width: float) -> float | None:
    """The x of a two-column gutter, or None for single-column (or tabular) pages.

    Strict on purpose: misreading one column as two scrambles every sentence, while
    missing a real second column only joins the two columns' lines.
    """
    lines = _group_lines(words)
    if len(lines) < 16:
        return None
    best_x, best_cross = None, len(lines)
    x = width * 0.35
    while x <= width * 0.65:
        crossing = sum(1 for ln in lines if any(w["x0"] < x < w["x1"] for w in ln))
        if crossing < best_cross or (crossing == best_cross and best_x is not None
                                      and abs(x - width / 2) < abs(best_x - width / 2)):
            best_x, best_cross = x, crossing
        x += 2
    if best_x is None or best_cross > 0.03 * len(lines):
        return None
    left = [w for w in words if w["x1"] <= best_x]
    right = [w for w in words if w["x0"] >= best_x]
    left_lines, right_lines = _group_lines(left), _group_lines(right)
    if len(left_lines) < 8 or len(right_lines) < 8:
        return None
    # The two columns run side by side for most of their height.
    lt, lb = min(w["top"] for w in left), max(w["bottom"] for w in left)
    rt, rb = min(w["top"] for w in right), max(w["bottom"] for w in right)
    overlap = min(lb, rb) - max(lt, rt)
    if overlap < 0.5 * min(lb - lt, rb - rt):
        return None
    # Left-column lines mostly run up to the gutter (wrapped prose), not short entries.
    left_x0 = min(w["x0"] for w in left)
    ends = sorted(max(w["x1"] for w in ln) for ln in left_lines)
    if ends[len(ends) // 2] < best_x - 0.35 * (best_x - left_x0):
        return None
    if _has_channel(left) or _has_channel(right):
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
        size = statistics.median(sizes) if sizes and statistics.median(sizes) > 0 else 10.0
        parts = [g[0]["text"]]
        for prev, w in zip(g, g[1:]):
            gap = float(w["x0"]) - float(prev["x1"])
            parts.append((" | " if gap > 1.8 * size else " ") + w["text"])
        lines.append(
            _Line(
                text="".join(parts),
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
                if _needs_ocr(page, compact, cfg):
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


def _needs_ocr(page, compact: str, cfg) -> bool:
    """OCR a page whose text layer is missing or garbage, or that is mostly one big image.

    A short page with a real text layer (the tail of a table, a signature page with
    a footer) stays digital: OCR would lose its structure.
    """
    if compact and printable_ratio(compact) < cfg["min_printable_ratio"]:
        return True
    if len(compact) >= cfg["min_text_chars"]:
        return False
    if len(compact) < 20:
        return True
    area = float(page.width) * float(page.height) or 1.0
    cover = 0.0
    for img in page.images:
        w = max(0.0, float(img["x1"]) - float(img["x0"]))
        h = max(0.0, float(img["bottom"]) - float(img["top"]))
        cover = max(cover, w * h / area)
    return cover >= 0.5


def _merge_continued_tables(blocks: list[Block]) -> list[Block]:
    """Join a table that continues onto the next page into one table block.

    Continued when two table blocks are adjacent (nothing but stripped page
    furniture between), on consecutive pages, with the same column count; a
    repeated header row is dropped, an unrepeated one is kept as data.
    """
    out: list[Block] = []
    for b in blocks:
        prev = out[-1] if out else None
        # Short page-furniture lines between the two halves (a footer the filters missed).
        between: list[Block] = []
        if b.kind == "table" and prev is not None and prev.kind != "table":
            k = len(out) - 1
            while k >= 0 and len(between) < 2 and out[k].kind == "paragraph" and len(out[k].text.split()) <= 15 \
                    and (_PAGE_REF_RE.search(out[k].text) or _PAGE_NUM_RE.match(out[k].text.strip())):
                between.append(out[k])
                k -= 1
            if between and k >= 0 and out[k].kind == "table":
                prev = out[k]
            else:
                between = []
        if (b.kind == "table" and prev is not None and prev.kind == "table" and b.page and prev.page
                and b.page == (prev.page_end or prev.page) + 1
                and len(b.meta.get("headers") or []) == len(prev.meta.get("headers") or [])
                and not b.meta.get("caption")):
            same_header = [h.strip().lower() for h in b.meta["headers"]] == \
                          [h.strip().lower() for h in prev.meta["headers"]]
            extra = b.meta["rows"] if same_header else [b.meta["headers"]] + b.meta["rows"]
            prev.meta["rows"] = list(prev.meta["rows"]) + list(extra)
            prev.page_end = b.page
            cap = prev.meta.get("caption")
            prev.text = (f"**{cap}**\n\n" if cap else "") + markdown_table(prev.meta["headers"], prev.meta["rows"])
            for _ in between:
                out.pop()   # the furniture lines go; they sat between the halves
            continue
        out.append(b)
    return out


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
    if " | " in text:
        return 0   # a row of aligned columns, not a heading
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
            y = ln.top if ln.top < p.height else ln.top - p.height   # right column is offset by height
            if p.height and y > 0.92 * p.height and _PAGE_REF_RE.search(ln.text) and len(ln.text.split()) <= 15:
                continue   # a footer with a page number, even on a short document
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
            prev_row = False
            for ln in para:
                row = " | " in ln.text
                if text and (row or prev_row):
                    text = f"{text}\n{ln.text}"          # aligned rows keep one line each
                elif text.endswith("-") and ln.text[:1].islower():
                    text = text[:-1] + ln.text
                else:
                    text = f"{text} {ln.text}".strip()
                prev_row = row
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
                    page_image = _render_page(pdoc, p.number - 1, 3.0)
                x0, t, x1, b = obj  # type: ignore[misc]
                s = 3.0
                crop = page_image.crop((int(x0 * s), int(t * s), int(x1 * s), int(b * s)))
                if crop.width < 20 or crop.height < 20:
                    continue
                figure_index += 1
                caption = _caption_after(events, idx, consumed)
                blocks.append(make_figure_block(crop, ctx, page=p.number, index=figure_index, caption=caption))
        flush()

    blocks = _merge_continued_tables(blocks)
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
