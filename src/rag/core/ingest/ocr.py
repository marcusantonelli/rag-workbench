"""OCR with RapidOCR (PP-OCR models on ONNX Runtime; models ship in the wheel)."""

from __future__ import annotations

import re
import statistics
import threading
import unicodedata
from dataclasses import dataclass

import numpy as np
from PIL import Image

_engine = None
_lock = threading.Lock()


@dataclass
class OcrLine:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class OcrParagraph:
    text: str
    conf: float
    y0: float
    is_heading: bool = False


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        return _engine


_call_lock = threading.Lock()


def _run(arr: np.ndarray, text_score: float):
    """One OCR pass. The engine remembers the thresholds of its last call, so every call states
    its own: 0.5 for ordinary reading, 0.0 for a second look at a single cell."""
    engine = _get_engine()
    with _call_lock:
        result, _ = engine(arr, text_score=text_score, box_thresh=0.5, unclip_ratio=1.6)
    return result


def ocr_fragments(image: Image.Image) -> list[OcrLine]:
    """The text boxes the OCR found, each with its position (one per cell in a table)."""
    arr = np.array(image.convert("RGB"))
    result = _run(arr, 0.5)
    frags: list[OcrLine] = []
    for box, text, score in result or []:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        frags.append(OcrLine(unicodedata.normalize("NFKC", str(text)), float(score), min(xs), min(ys), max(xs), max(ys)))
    return frags


def ocr_lines(image: Image.Image) -> list[OcrLine]:
    """Recognize text and merge fragments that sit on the same visual line."""
    return _merge_lines(ocr_fragments(image))


def _merge_lines(frags: list[OcrLine]) -> list[OcrLine]:
    if not frags:
        return []
    frags = list(frags)
    frags.sort(key=lambda f: ((f.y0 + f.y1) / 2, f.x0))
    lines: list[list[OcrLine]] = []
    for frag in frags:
        mid = (frag.y0 + frag.y1) / 2
        if lines:
            last = lines[-1]
            ly0 = min(f.y0 for f in last)
            ly1 = max(f.y1 for f in last)
            if ly0 <= mid <= ly1:
                last.append(frag)
                continue
        lines.append([frag])
    merged: list[OcrLine] = []
    for group in lines:
        group.sort(key=lambda f: f.x0)
        merged.append(
            OcrLine(
                text=" ".join(f.text for f in group),
                conf=sum(f.conf * len(f.text) for f in group) / max(1, sum(len(f.text) for f in group)),
                x0=min(f.x0 for f in group),
                y0=min(f.y0 for f in group),
                x1=max(f.x1 for f in group),
                y1=max(f.y1 for f in group),
            )
        )
    return merged


def lines_to_paragraphs(lines: list[OcrLine]) -> list[OcrParagraph]:
    """Group lines into paragraphs by vertical gaps; flag tall short lines as headings."""
    if not lines:
        return []
    heights = [ln.height for ln in lines if ln.height > 0]
    median_h = statistics.median(heights) if heights else 10.0
    paras: list[OcrParagraph] = []
    buf: list[OcrLine] = []

    def flush() -> None:
        if buf:
            text = ""
            for ln in buf:
                if text.endswith("-") and ln.text[:1].islower():
                    text = text[:-1] + ln.text
                else:
                    text = f"{text} {ln.text}".strip()
            conf = sum(ln.conf for ln in buf) / len(buf)
            paras.append(OcrParagraph(text=text, conf=conf, y0=buf[0].y0))
            buf.clear()

    prev: OcrLine | None = None
    for ln in lines:
        heading = ln.height >= 1.4 * median_h and len(ln.text) <= 100 and not ln.text.endswith(".")
        if heading:
            flush()
            paras.append(OcrParagraph(text=ln.text, conf=ln.conf, y0=ln.y0, is_heading=True))
            prev = ln
            continue
        if prev is not None and buf:
            gap = ln.y0 - prev.y1
            if gap > 0.9 * median_h or ln.text[:1] in "•▪-*" :
                flush()
        buf.append(ln)
        prev = ln
    flush()
    return paras


def ocr_text(image: Image.Image) -> tuple[str, float]:
    lines = ocr_lines(image)
    if not lines:
        return "", 0.0
    paras = lines_to_paragraphs(lines)
    text = "\n\n".join(p.text for p in paras)
    conf = sum(ln.conf for ln in lines) / len(lines)
    return text, conf


# -- pictures of tables and spreadsheets ----------------------------------------------------------
@dataclass
class OcrTable:
    headers: list[str]
    rows: list[list[str]]
    title: str            # text above the grid (a sheet title), may be empty
    conf: float


@dataclass
class FigureText:
    text: str             # paragraphs, as ocr_text gives
    conf: float
    table: OcrTable | None


def _upscaled(image: Image.Image) -> tuple[Image.Image, float]:
    """Small pictures are enlarged before OCR: the recogniser misses short items (a lone "0") at
    screenshot sizes."""
    side = max(image.size)
    if side >= 1400 or side == 0:
        return image, 1.0
    f = min(3.0, 1600 / side)
    return image.resize((int(image.width * f), int(image.height * f)), Image.LANCZOS), f


def ocr_figure(image: Image.Image, *, tables: bool = True) -> FigureText:
    """OCR a figure; when its text sits in a grid (a table, a spreadsheet screenshot), also
    rebuild the rows and columns."""
    big, _ = _upscaled(image)
    frags = ocr_fragments(big)
    if not frags:
        return FigureText("", 0.0, None)
    # unsure reads (glare, engraving, screen blur) are kept for the grid, which re-reads its
    # cells, but not for the figure's text
    sure = [f for f in frags if f.conf >= 0.4] or frags
    lines = _merge_lines(sure)
    text = "\n\n".join(p.text for p in lines_to_paragraphs(lines))
    conf = sum(f.conf for f in sure) / len(sure)
    return FigureText(text, conf, grid_table(frags, big) if tables else None)


_COL_LETTERS = re.compile(r"^[A-Z]{1,2}$")


_NUMBER = re.compile(r"^[-+(]?[\d.,]+%?\)?$")


def _reread(image: Image.Image, box: tuple[float, float, float, float], gray: np.ndarray | None = None,
            sizes: tuple[int, ...] = (18, 26)) -> tuple[str, float]:
    """Recognise the ink inside one box without the text finder: crop to the ink, scale the
    glyphs to sizes the recogniser reads well (it is sensitive to size), keep the surer read."""
    from PIL import ImageOps

    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return "", 0.0
    region = (gray if gray is not None else np.asarray(image.convert("L")))[y0:y1, x0:x1]
    ys, xs = np.nonzero(region < 150)
    if len(xs) < 8:
        return "", 0.0
    tight = image.crop((x0 + int(xs.min()) - 2, y0 + int(ys.min()) - 2,
                        x0 + int(xs.max()) + 3, y0 + int(ys.max()) + 3)).convert("RGB")
    best, score = "", 0.0
    for glyph in sizes:
        f = glyph / max(1, tight.height)
        sized = tight.resize((max(1, int(tight.width * f)), glyph), Image.LANCZOS)
        padded = ImageOps.expand(sized, int(0.6 * glyph), fill="white")
        try:
            result = _run(np.array(padded), 0.0)
        except Exception:  # noqa: BLE001 - a repair attempt only
            continue
        if result and float(result[0][2]) > score:
            best, score = unicodedata.normalize("NFKC", str(result[0][1])).strip(), float(result[0][2])
            if score >= 0.45:
                break          # sure enough; the other sizes rarely do better
    return best, score


def _read_cell(image: Image.Image, box: tuple[float, float, float, float], gray: np.ndarray | None = None) -> str:
    """A cell the text finder skipped (it misses a lone short item such as "0" or "2"). A number
    is accepted at lower confidence than other text."""
    text, score = _reread(image, box, gray, sizes=(18, 12, 16, 26))
    if not text:
        return ""
    if _NUMBER.match(text):
        return text if score >= 0.3 else ""
    return text if score >= 0.6 else ""


def _ink_gaps(image: Image.Image, box: tuple[float, float, float, float], min_gap: float,
              gray: np.ndarray | None = None) -> list[tuple[float, float]]:
    """Blank vertical stretches at least min_gap wide inside a box, in the image's coordinates."""
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1)
    if x1 - x0 < 4 or y1 - y0 < 2:
        return []
    ink = ((gray if gray is not None else np.asarray(image.convert("L")))[y0:y1, x0:x1] < 160).any(axis=0)
    gaps, i, n = [], 0, len(ink)
    while i < n:
        if not ink[i]:
            j = i
            while j + 1 < n and not ink[j + 1]:
                j += 1
            if i > 0 and j < n - 1 and j - i + 1 >= min_gap:
                gaps.append((x0 + i, x0 + j))
            i = j + 1
        else:
            i += 1
    return gaps


def _rows(frags: list[OcrLine], h: float) -> list[list[OcrLine]]:
    """Text boxes grouped into rows: boxes whose middles share a band about one text height tall."""
    rows: list[list[OcrLine]] = []
    for f in sorted(frags, key=lambda f: ((f.y0 + f.y1) / 2, f.x0)):
        mid = (f.y0 + f.y1) / 2
        if rows:
            last = rows[-1]
            top, bottom = min(g.y0 for g in last), max(g.y1 for g in last)
            if top - 0.2 * h <= mid <= bottom + 0.2 * h and abs(mid - sum((g.y0 + g.y1) / 2 for g in last) / len(last)) <= 0.6 * h:
                last.append(f)
                continue
        rows.append([f])
    return rows


def _median_height(frags: list[OcrLine]) -> float:
    heights = sorted(f.height for f in frags if f.height > 0)
    return heights[len(heights) // 2] if heights else 10.0


def _tighten(f: OcrLine, gray: np.ndarray) -> OcrLine:
    """The text finder's boxes are padded well past the ink; trim one to the ink inside it."""
    x0, y0 = max(0, int(f.x0)), max(0, int(f.y0))
    x1, y1 = min(gray.shape[1], int(f.x1) + 1), min(gray.shape[0], int(f.y1) + 1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return f
    ys, xs = np.nonzero(gray[y0:y1, x0:x1] < 160)
    if len(xs) < 4:
        return f
    return OcrLine(f.text, f.conf, x0 + float(xs.min()), y0 + float(ys.min()), x0 + float(xs.max()) + 1, y0 + float(ys.max()) + 1)


def _split_wide(f: OcrLine, image: Image.Image, gray: np.ndarray, h: float) -> list[OcrLine]:
    """Two cells read as one box ("Hours    Fails"): cut at each blank much wider than a word
    space and read the pieces again. If any piece can't be read, the box stays whole."""
    if " " not in f.text or f.x1 - f.x0 < 2 * h:
        return [f]
    gaps = _ink_gaps(image, (f.x0, f.y0, f.x1, f.y1), 1.1 * h, gray)
    if not gaps:
        return [f]
    bounds = [f.x0] + [(g[0] + g[1]) / 2 for g in gaps] + [f.x1]
    pieces = []
    for a, b in zip(bounds, bounds[1:]):
        text, score = _reread(image, (a, f.y0 - 2, b, f.y1 + 2), gray)
        if not text or score < 0.5:
            return [f]
        pieces.append(_tighten(OcrLine(text, score, a, f.y0, b, f.y1), gray))
    return pieces


def grid_table(frags: list[OcrLine], image: Image.Image | None = None) -> OcrTable | None:
    """Rows and columns from where each text box sits, or None when the text isn't a grid.

    Rows are boxes sharing a baseline band. Column edges are vertical channels that (nearly)
    every row leaves empty; a title or note spanning the width doesn't block them. It is a
    table when most rows have two or more cells and the cells are short (prose is not).
    """
    if len(frags) < 6:
        return None
    gray = np.asarray(image.convert("L")) if image is not None else None
    if gray is not None:
        frags = [_tighten(f, gray) for f in frags]
    h = _median_height(frags)
    if gray is not None:
        frags = [p for f in frags for p in _split_wide(f, image, gray, h)]
    rows = _rows(frags, h)
    if len(rows) < 3:
        return None
    left = min(f.x0 for f in frags)
    right = max(f.x1 for f in frags)
    if gray is not None:
        # a column of single digits the text finder skipped can lie right of every box it found
        reach = min(gray.shape[1], int(right + max(10 * h, 0.3 * (right - left))))
        hits = np.zeros(max(0, reach - int(right)), dtype=np.int32)
        for r in rows:
            y0, y1 = max(0, int(min(f.y0 for f in r))), min(gray.shape[0], int(max(f.y1 for f in r)) + 1)
            if y1 > y0 and len(hits):
                hits += (gray[y0:y1, int(right):reach] < 160).any(axis=0)
        seen = np.nonzero(hits >= max(2, 0.3 * len(rows)))[0]
        if len(seen):
            right = right + float(seen.max()) + 1
    width = right - left
    if width <= 0:
        return None
    # rows that are one wide piece of text (a title, a note) don't decide the columns
    spanning = {i for i, r in enumerate(rows) if len(r) == 1 and (r[0].x1 - r[0].x0) > 0.45 * width}
    grid_rows = [r for i, r in enumerate(rows) if i not in spanning]
    if len(grid_rows) < 3:
        return None
    n = int(width) + 2
    cover = np.zeros(n, dtype=np.int32)
    inks: list[np.ndarray] = []
    for r in grid_rows:
        mask = np.zeros(n, dtype=bool)
        for f in r:
            mask[int(f.x0 - left):int(f.x1 - left) + 1] = True
        cover += mask
        if gray is not None:
            # ink the text finder skipped (single digits) still marks its column
            ink = np.zeros(n, dtype=bool)
            y0 = max(0, int(min(f.y0 for f in r)))
            y1 = min(gray.shape[0], int(max(f.y1 for f in r)) + 1)
            x0, x1 = max(0, int(left)), min(gray.shape[1], int(left) + n)
            if y1 > y0 and x1 > x0:
                ink[x0 - int(left):x1 - int(left)] = (gray[y0:y1, x0:x1] < 160).any(axis=0)
            inks.append(ink & ~mask)
    if inks:
        extra = np.sum(inks, axis=0)
        # a vertical rule: a few pixels wide, inked in (nearly) every row; it is not a column
        ruled = extra >= 0.9 * len(grid_rows)
        i = 0
        while i < n:
            if ruled[i]:
                j = i
                while j + 1 < n and ruled[j + 1]:
                    j += 1
                if j - i + 1 > max(3, 0.2 * h):
                    ruled[i:j + 1] = False
                i = j + 1
            else:
                i += 1
        extra[ruled] = 0
        cover += np.minimum(extra, len(grid_rows))
    allowed = int(0.15 * len(grid_rows))
    empty = cover <= allowed
    strict = np.zeros(n, dtype=bool)          # any text box of a grid row, header included
    for r in grid_rows:
        for f in r:
            strict[int(f.x0 - left):int(f.x1 - left) + 1] = True
    edges: list[float] = []
    i = 0
    min_gap = max(3.0, 0.35 * h)
    while i < n:
        if empty[i] and 0 < i:
            j = i
            while j + 1 < n and empty[j + 1]:
                j += 1
            if j < n - 1 and j - i + 1 >= min_gap:
                # a header in the blank between two columns' data belongs to one of them: cut in
                # the clear stretch left of it (text, headers included, is left-aligned by default)
                clear, k = [], i
                while k <= j:
                    if not strict[k]:
                        m = k
                        while m + 1 <= j and not strict[m + 1]:
                            m += 1
                        if m - k + 1 >= 2:
                            clear.append((k, m))
                        k = m + 1
                    else:
                        k += 1
                a, b = clear[0] if clear else (i, j)
                edges.append(left + (a + b) / 2)
            i = j + 1
        else:
            i += 1
    if not edges:
        return None
    ncols = len(edges) + 1

    def col(f: OcrLine) -> int:
        c = (f.x0 + f.x1) / 2
        return sum(1 for e in edges if c > e)

    table: list[list[str]] = []
    title: list[str] = []
    bands: list[tuple[float, float] | None] = []      # each table row's vertical extent
    for i, r in enumerate(rows):
        if i in spanning:
            if not table:
                title.append(r[0].text)
            else:
                table.append([r[0].text] + [""] * (ncols - 1))
                bands.append(None)
            continue
        cells = [""] * ncols
        for f in sorted(r, key=lambda f: f.x0):
            k = col(f)
            cells[k] = f"{cells[k]} {f.text}".strip()
        table.append(cells)
        bands.append((min(f.y0 for f in r), max(f.y1 for f in r)))
    if image is not None:
        # an empty cell in a column of numbers that still has ink in it: read it on its own
        bounds = [left - 0.5 * h] + edges + [right + 0.5 * h]
        budget = 400          # second reads per table, so one huge scan can't take minutes
        for k in range(ncols):
            for row, band in zip(table[1:], bands[1:]):
                if row[k] or band is None or sum(1 for c in row if c) < 2 or budget <= 0:
                    continue
                budget -= 1
                inset = 0.25 * h
                row[k] = _read_cell(image, (bounds[k] + inset, band[0] - 0.15 * h, bounds[k + 1] - inset, band[1] + 0.15 * h), gray)
    filled = [sum(1 for c in row if c) for row in table]
    multi = sum(1 for k in filled if k >= 2)
    lengths = sorted(len(c) for row in table for c in row if c)
    if multi < 3 or multi < 0.6 * len(table) or not lengths or lengths[len(lengths) // 2] > 30:
        return None
    # Excel's own headers in a screenshot: a row of column letters, a column of row numbers
    if table and sum(1 for c in table[0] if _COL_LETTERS.match(c)) >= max(2, 0.6 * sum(1 for c in table[0] if c)):
        table = table[1:]
    first = [row[0] for row in table]
    if first and sum(1 for c in first if c.isdigit()) >= 0.8 * len(first):
        # row numbers count up by one; allow a misread or two
        start = next(int(c) for c in first if c.isdigit()) - next(i for i, c in enumerate(first) if c.isdigit())
        if sum(1 for i, c in enumerate(first) if c == str(start + i)) >= 0.7 * len(first):
            table = [row[1:] for row in table]
    keep = [k for k in range(len(table[0]) if table else 0) if any(row[k] for row in table)]
    table = [[row[k] for k in keep] for row in table]
    if len(table) < 2 or len(keep) < 2:
        return None
    # a table has a name over most of its columns; a screenshot of a dialog (labels and values
    # scattered in rough rows) does not, and is better kept as a figure with its text
    if sum(1 for c in table[0] if c) < 0.6 * len(keep):
        return None
    if len(keep) == 2 and len(table) < 5:
        return None       # two signs or buttons side by side, not a two-column table
    ragged = sum(1 for row in table[1:] if sum(1 for c in row if c) < 0.5 * len(keep))
    if ragged > 0.5 * (len(table) - 1):
        return None
    return OcrTable(headers=table[0], rows=table[1:], title=" ".join(title),
                    conf=sum(f.conf for f in frags) / len(frags))


# -- scanned pages ----------------------------------------------------------------------------------
@dataclass
class PagePart:
    y: float
    paragraph: OcrParagraph | None = None
    table: OcrTable | None = None
    box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


@dataclass
class PageRead:
    parts: list[PagePart]
    conf: float


def ocr_page(image: Image.Image, *, tables: bool = True) -> PageRead:
    """Read a scanned page: paragraphs and headings, and tables where runs of rows line up in
    columns (a job sheet's data block, a printed spreadsheet)."""
    frags = ocr_fragments(image)
    if not frags:
        return PageRead([], 0.0)
    conf = sum(f.conf for f in frags) / len(frags)
    h = _median_height(frags)
    rows = _rows(frags, h)
    in_table: set[int] = set()
    parts: list[PagePart] = []
    if tables:
        # runs of three or more rows with two or more boxes each (a short one-box row inside a
        # run, such as a section label, doesn't end it) are table candidates
        i = 0
        while i < len(rows):
            if len(rows[i]) < 2:
                i += 1
                continue
            j = i
            while j + 1 < len(rows) and (len(rows[j + 1]) >= 2 or (
                    len(rows[j + 1][0].text) <= 30 and j + 2 < len(rows) and len(rows[j + 2]) >= 2)):
                j += 1
            if j - i + 1 >= 3:
                run = [f for r in rows[i:j + 1] for f in r]
                # a title line right above the run belongs to the table
                if i > 0 and len(rows[i - 1]) == 1 and len(rows[i - 1][0].text) <= 60:
                    run = rows[i - 1] + run
                    first = i - 1
                else:
                    first = i
                table = grid_table(run, image)
                if table is not None:
                    box = (min(f.x0 for f in run), min(f.y0 for f in run), max(f.x1 for f in run), max(f.y1 for f in run))
                    parts.append(PagePart(y=box[1], table=table, box=box))
                    in_table.update(range(first, j + 1))
            i = j + 1
    rest = [f for k, r in enumerate(rows) if k not in in_table for f in r]
    for p in lines_to_paragraphs(_merge_lines(rest)):
        parts.append(PagePart(y=p.y0, paragraph=p))
    parts.sort(key=lambda p: p.y)
    return PageRead(parts, conf)
