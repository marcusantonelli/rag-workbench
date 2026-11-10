"""PDF extraction.

Backends:
  * docling  - layout model, TableFormer tables, reading order, figures (best quality;
               needs `pip install rag-builder[docling]` and its models downloaded once).
  * native   - pdfplumber for digital pages (text lines, font-size headings, ruled
               tables, embedded images, flowcharts drawn as vector shapes) and RapidOCR
               for scanned pages. No PyTorch.
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
    style: str = ""         # dominant font + colour, to tell a wrapped heading from the next one
    group: int = 0          # reading-order group (column) on the page
    key: float = 0.0        # sort key: group * page height + top


@dataclass
class _PageItems:
    number: int
    lines: list[_Line] = field(default_factory=list)
    tables: list[tuple[float, list[list[str]], tuple[float, float, float, float]]] = field(
        default_factory=list)   # (top, rows, bbox)
    figures: list[tuple[float, tuple[float, float, float, float]]] = field(default_factory=list)
    diagrams: list = field(default_factory=list)       # diagram.Diagram, rebuilt from vector shapes
    ocr: bool = False
    ocr_blocks: list[Block] = field(default_factory=list)
    ocr_conf: float | None = None
    width: float = 0
    height: float = 0
    groups: list[tuple[float, float, float, float]] = field(default_factory=list)   # bbox per group


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


_NUMBERED_HEADING_RE = re.compile(r"^(\d+(\.\d+)*\.?|[A-Z]\.|[IVX]+\.)\s+\S")


def _line_spans(lines: list[list[dict]]) -> list[list[tuple[float, float]]]:
    """Word extents padded by about half a space, so an x inside an ordinary word gap
    still counts as crossing the line: a column gutter must be wider than word spacing."""
    out = []
    for ln in lines:
        pad = 0.35 * statistics.median(float(w.get("size") or 10.0) for w in ln)
        out.append([(float(w["x0"]) - pad, float(w["x1"]) + pad) for w in ln])
    return out


def _aligned_starts(words: list[dict]) -> bool:
    """Most lines of a real column start at its left edge (allowing a bullet's hanging
    indent). Half-lines cut out of one column start wherever a word happened to fall."""
    lines = _group_lines(words)
    starts = sorted(min(float(w["x0"]) for w in ln) for ln in lines)
    if len(starts) < 3:
        return True
    size = statistics.median(float(w.get("size") or 10.0) for w in words)
    edge = starts[len(starts) // 10]
    near = sum(1 for t in starts if edge - 3.0 <= t <= edge + 2.0 * size)
    return near >= 0.7 * len(starts)


def _column_channel(words: list[dict], lines: list[list[dict]]) -> float | None:
    """x of the best vertical channel between two text columns in this region, or None.

    A channel may be crossed by a few full-width lines (a title, a footer); those are
    read on their own. Both sides must hold real text (several lines, a usable width).
    """
    x0 = min(float(w["x0"]) for w in words)
    x1 = max(float(w["x1"]) for w in words)
    width = x1 - x0
    if width < 150:
        return None
    n = len(lines)
    allowed = max(1, int(0.25 * n))
    spans = _line_spans(lines)
    xs = [x0 + 0.08 * width + i for i in range(0, int(0.84 * width), 2)]
    crossing = [sum(1 for sp in spans if any(a < x < b for a, b in sp)) for x in xs]
    best: tuple[int, float, float] | None = None     # (crossing, -stretch width, x)
    i = 0
    while i < len(xs):
        if crossing[i] > allowed:
            i += 1
            continue
        j = i
        while j + 1 < len(xs) and crossing[j + 1] <= allowed:
            j += 1
        low = min(crossing[i:j + 1])
        # The centre of the longest stretch at the run's lowest crossing count.
        k, run_best = i, (0, i, i)
        while k <= j:
            if crossing[k] == low:
                m = k
                while m + 1 <= j and crossing[m + 1] == low:
                    m += 1
                if m - k >= run_best[0]:
                    run_best = (m - k, k, m)
                k = m + 1
            else:
                k += 1
        c = (xs[run_best[1]] + xs[run_best[2]]) / 2
        left = [w for w in words if float(w["x1"]) <= c]
        right = [w for w in words if float(w["x0"]) >= c]
        min_side = max(60.0, 0.15 * width)
        if (left and right
                and max(float(w["x1"]) for w in left) - min(float(w["x0"]) for w in left) >= min_side
                and max(float(w["x1"]) for w in right) - min(float(w["x0"]) for w in right) >= min_side
                and len(_group_lines(left)) >= 3 and len(_group_lines(right)) >= 3
                and _aligned_starts(right)):
            cand = (low, -(xs[run_best[2]] - xs[run_best[1]]), c)
            if best is None or cand < best:
                best = cand
        i = j + 1
    return None if best is None else best[2]


def _looks_like_columns(left: list[dict], right: list[dict]) -> bool:
    """False when the two sides are the cells of an unruled table rather than two text columns.

    Table cells sit on shared rows (aligned tops, matching line counts) and are short.
    Text columns either run on their own baselines or both hold full prose lines.
    """
    lines_l, lines_r = _group_lines(left), _group_lines(right)
    if not lines_l or not lines_r:
        return False
    small, big = sorted((lines_l, lines_r), key=len)
    other_tops = [float(ln[0]["top"]) for ln in big]
    paired = sum(1 for ln in small if any(abs(float(ln[0]["top"]) - t) <= 2.0 for t in other_tops)) / len(small)
    ratio = len(small) / len(big)
    words_l = statistics.median(len(ln) for ln in lines_l)
    words_r = statistics.median(len(ln) for ln in lines_r)
    return not (paired >= 0.7 and ratio >= 0.75 and min(words_l, words_r) <= 4)


def _reading_groups(words: list[dict], depth: int = 0) -> list[list[dict]]:
    """Split a page's words into groups in reading order: full-width lines stay in
    place, and the text between them is read one column at a time (recursively, so
    a sidebar plus two columns comes out as three groups)."""
    if not words:
        return []
    lines = _group_lines(words)
    if depth >= 4 or len(lines) < 4:
        return [words]
    ch = _column_channel(words, lines)
    if ch is None:
        return [words]
    out: list[list[dict]] = []
    band: list[dict] = []
    spanning: list[dict] = []

    def flush_band() -> None:
        if not band:
            return
        left = [w for w in band if float(w["x1"]) <= ch]
        right = [w for w in band if float(w["x0"]) >= ch]
        if left and right and _aligned_starts(right) and _looks_like_columns(left, right):
            out.extend(_reading_groups(left, depth + 1))
            out.extend(_reading_groups(right, depth + 1))
        else:
            out.append(list(band))
        band.clear()

    for ln, sp in zip(lines, _line_spans(lines)):
        if any(a < ch < b for a, b in sp):
            flush_band()
            spanning.extend(ln)
        else:
            if spanning:
                out.append(list(spanning))
                spanning.clear()
            band.extend(ln)
    flush_band()
    if spanning:
        out.append(spanning)
    return out


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
        styles = [f"{w.get('fontname', '')}|{w.get('non_stroking_color', '')}" for w in g]
        lines.append(
            _Line(
                text="".join(parts),
                size=statistics.median(sizes) if sizes else 0.0,
                top=min(w["top"] for w in g),
                bottom=max(w["bottom"] for w in g),
                x0=min(w["x0"] for w in g),
                bold=sum("bold" in f.lower() for f in fonts) > len(fonts) / 2,
                style=Counter(styles).most_common(1)[0][0],
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
                and not b.meta.get("from_image") and not prev.meta.get("from_image")
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


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """The share of a's area that lies inside b."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    return w * h / area if area else 0.0


def _digital_page(page, items: _PageItems, ctx: ExtractContext) -> None:
    cfg = ctx.config["pdf"]
    try:
        words = page.extract_words(extra_attrs=["size", "fontname", "non_stroking_color"], keep_blank_chars=False)
    except Exception:  # noqa: BLE001 - odd colour spaces: fall back to font only
        words = page.extract_words(extra_attrs=["size", "fontname"], keep_blank_chars=False)

    # Flowcharts drawn as shapes first: their labels leave the text flow, and the boxes they
    # are drawn with are not tables.
    regions: list[tuple[float, float, float, float]] = []
    if cfg.get("diagrams", True):
        from rag.core.ingest.diagram import find_diagrams

        try:
            items.diagrams = find_diagrams(page, words)
        except Exception:  # noqa: BLE001 - a drawing we can't read is left to the text reader
            items.diagrams = []
        if items.diagrams:
            used = set().union(*(d.word_ids for d in items.diagrams))
            words = [w for i, w in enumerate(words) if i not in used]
            regions = [d.bbox for d in items.diagrams]

    table_bboxes: list[tuple[float, float, float, float]] = []
    try:
        for table in page.find_tables():
            bbox = tuple(float(v) for v in table.bbox)
            if any(_overlap(bbox, r) > 0.5 for r in regions):
                continue
            rows = [[(c or "").strip() for c in row] for row in table.extract()]
            rows = [r for r in rows if any(r)]
            if len(rows) >= 2 and max(len(r) for r in rows) >= 2:
                table_bboxes.append(table.bbox)
                items.tables.append((table.bbox[1], rows, bbox))
    except Exception:  # noqa: BLE001
        pass
    page_area = items.width * items.height or 1
    for img in page.images:
        bbox = (float(img["x0"]), float(img["top"]), float(img["x1"]), float(img["bottom"]))
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        if area / page_area >= cfg["figure_min_area"] and not any(_overlap(bbox, r) > 0.5 for r in regions):
            items.figures.append((bbox[1], bbox))
    words = [w for w in words if not any(_inside(w, b) for b in table_bboxes)]
    height = items.height or 1000.0
    for k, group in enumerate(_reading_groups(words)):
        items.groups.append((min(float(w["x0"]) for w in group), min(float(w["top"]) for w in group),
                             max(float(w["x1"]) for w in group), max(float(w["bottom"]) for w in group)))
        for ln in _words_to_lines(group):
            ln.group = k
            ln.key = k * height + ln.top
            items.lines.append(ln)


def _reading_key(p: _PageItems, bbox: tuple[float, float, float, float]) -> float:
    """Where a table or figure falls in the page's reading order."""
    height = p.height or 1000.0
    x0, top, x1, _ = bbox
    cx = (x0 + x1) / 2
    for k, (gx0, gt, gx1, gb) in enumerate(p.groups):
        if gx0 - 2 <= cx <= gx1 + 2 and gt - 2 <= top <= gb + 2:
            return k * height + top
    # wider than the text around it (a full-width picture beside narrower lines): the group it
    # overlaps most at that height
    best: tuple[float, int] | None = None
    for k, (gx0, gt, gx1, gb) in enumerate(p.groups):
        across = min(x1, gx1) - max(x0, gx0)
        if across > 0 and gt - 2 <= top <= gb + 2 and (best is None or across > best[0]):
            best = (across, k)
    if best is not None:
        return best[1] * height + top
    for k, (_, gt, _, _) in enumerate(p.groups):
        if gt > top:
            return k * height + top
    return len(p.groups) * height + top


def _ocr_page(pdoc, index: int, items: _PageItems, ctx: ExtractContext) -> None:
    from rag.core.ingest.ocr import ocr_page

    ctx.progress("ocr", {"page": index + 1})
    image = _render_page(pdoc, index, float(ctx.config["pdf"]["ocr_scale"]))
    read = ocr_page(image, tables=bool(ctx.config["figures"].get("table_ocr", True)))
    items.ocr_conf = read.conf
    n_tables = 0
    for part in read.parts:
        if part.table is not None:   # a table on a scanned page: rows and columns, with its picture
            n_tables += 1
            t = part.table
            path = ctx.asset_dir / f"p{items.number}-scan-table-{n_tables}.png"
            x0, y0, x1, y1 = part.box
            pad = 6
            image.crop((max(0, int(x0) - pad), max(0, int(y0) - pad), min(image.width, int(x1) + pad),
                        min(image.height, int(y1) + pad))).save(path, format="PNG", optimize=True)
            label = (t.title or f"Table on scanned page {items.number}") + " (read with OCR)"
            items.ocr_blocks.append(Block(
                kind="table", text=f"**{label}**\n\n" + markdown_table(t.headers, t.rows), page=items.number,
                meta={"headers": t.headers, "rows": t.rows, "caption": label, "from_image": True,
                      "asset_path": ctx.asset_rel(path), "ocr_conf": round(t.conf, 3), "y": y0}))
            continue
        p = part.paragraph
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
            if p.height and ln.top > 0.92 * p.height and _PAGE_REF_RE.search(ln.text) and len(ln.text.split()) <= 15:
                continue   # a footer with a page number, even on a short document
            events.append((ln.key, "line", ln))
        for top, rows, bbox in p.tables:
            events.append((_reading_key(p, bbox), "table", rows))
        for top, bbox in p.figures:
            events.append((_reading_key(p, bbox), "figure", bbox))
        for d in p.diagrams:
            events.append((_reading_key(p, d.bbox), "diagram", d))
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
        last_heading: list[_Line | None] = [None]
        for idx, (top, kind, obj) in enumerate(events):
            if idx in consumed:
                continue
            if kind == "line":
                ln: _Line = obj  # type: ignore[assignment]
                level = _is_heading(ln, body, levels)
                if level:
                    flush()
                    prev_h = last_heading[0]
                    if (prev_h is not None and blocks and blocks[-1].kind == "heading"
                            and blocks[-1].level == level and prev_h.group == ln.group
                            and 0 <= ln.top - prev_h.bottom <= 0.6 * max(prev_h.bottom - prev_h.top, 1)
                            and abs(ln.size - prev_h.size) < 0.5 and ln.style == prev_h.style
                            and not _NUMBERED_HEADING_RE.match(ln.text.strip())
                            and len(blocks[-1].text) + len(ln.text) <= 150):
                        blocks[-1].text = f"{blocks[-1].text} {ln.text.strip()}"   # a wrapped heading
                    else:
                        blocks.append(Block(kind="heading", text=ln.text.strip(), level=level, page=p.number))
                    last_heading[0] = ln
                    continue
                last_heading[0] = None
                if para:
                    prev = para[-1]
                    gap = ln.top - prev.bottom
                    if (ln.group != prev.group or gap > max(prev.bottom - prev.top, 1) * 0.9
                            or _BULLET_RE.match(ln.text)):
                        flush()
                para.append(ln)
            elif kind == "table":
                last_heading[0] = None
                flush()
                rows: list[list[str]] = obj  # type: ignore[assignment]
                caption = _caption_near(blocks, events, idx, consumed)
                headers, data = rows[0], rows[1:]
                md = markdown_table(headers, data)
                text = (f"**{caption}**\n\n" if caption else "") + md
                blocks.append(Block(kind="table", text=text, page=p.number,
                                    meta={"headers": headers, "rows": data, "caption": caption}))
            elif kind == "figure":
                last_heading[0] = None
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
            elif kind == "diagram":
                from rag.core.ingest.figures import make_diagram_block

                last_heading[0] = None
                flush()
                if page_image is None:
                    page_image = _render_page(pdoc, p.number - 1, 3.0)
                x0, t, x1, b = obj.bbox  # type: ignore[union-attr]
                s, pad = 3.0, 8.0
                crop = page_image.crop((max(0, int((x0 - pad) * s)), max(0, int((t - pad) * s)),
                                        min(page_image.width, int((x1 + pad) * s)), min(page_image.height, int((b + pad) * s))))
                figure_index += 1
                caption = _caption_near(blocks, events, idx, consumed)
                blocks.append(make_diagram_block(crop, obj, ctx, page=p.number, index=figure_index, caption=caption))
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
