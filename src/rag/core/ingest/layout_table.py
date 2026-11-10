"""Tables used as page layout: work instructions and procedures set out as a grid.

A work instruction is often one long table (STEP | PROCEDURE | DESCRIPTION | VISUAL) that runs
for many pages. Its cells hold bullet lists and photographs, a step carries on across rows and
pages, a new step can start part-way down a cell, and small tables sit inside cells. Read as a
grid it gives a few enormous cells and scattered fragments. It is a document, not a data table.

Here such a table is read by position instead of by cell. Each piece of content (a paragraph
or bullet in a column, a picture, a nested table) is placed by its page and height. A short
entry in the first column ("7.2") starts a step; everything level with or below it belongs to
that step until the next entry, on this page or a later one. The reader in pdf.py turns the
stream into a heading per step, its text in order, and its pictures labelled with the step.

A table qualifies when it holds pictures, nested tables, long text or very tall cells, or when it
continues one that does (same column edges at the top of the next page).
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]

# Bullet glyphs, including the Symbol/Wingdings private-use ones Word writes into PDFs.
BULLET = re.compile(r"^\s*([•▪●◦■□–\-\*]|o(?=\s))\s+")
KEY = re.compile(r"^[A-Za-z]?\d+(\.\d+)*[.)]?$|^[A-Za-z][.)]$|^\(?[ivxIVX]+[.)]$")


@dataclass
class Item:
    y: float                     # where it sorts on the page (top of text; upper part of a picture)
    col: int
    kind: str                    # text | image | table
    text: str = ""
    bbox: Box = (0.0, 0.0, 0.0, 0.0)
    rows: list[list[str]] | None = None
    bullet: bool = False
    bold: bool = False
    used: bool = False


@dataclass
class LayoutTable:
    page: int
    bbox: Box
    edges: list[float]
    headers: list[str] | None
    items: list[Item] = field(default_factory=list)
    candidate: bool = False      # looks like a layout table on its own
    is_layout: bool = False      # decided across pages
    continues: bool = False      # carries on the table at the bottom of the previous page
    nested: list[Box] = field(default_factory=list)

    @property
    def keyed(self) -> bool:
        """First column holds short step numbers rather than text."""
        keys = [i.text for i in self.items if i.kind == "text" and i.col == 0]
        return bool(keys) and all(len(k) <= 12 for k in keys) and sum(1 for k in keys if KEY.match(k.strip())) >= 0.6 * len(keys)


def column_edges(table) -> list[float]:
    xs: list[float] = []
    for row in table.rows:
        for c in row.cells:
            if c is not None:
                xs += [float(c[0]), float(c[2])]
    out: list[float] = []
    for x in sorted(xs):
        if not out or x - out[-1] > 3.0:
            out.append(x)
    return out


def same_columns(a: list[float], b: list[float], tol: float = 4.0) -> bool:
    return len(a) == len(b) and all(abs(x - y) <= tol for x, y in zip(a, b))


def _inside(box: Box, x: float, y: float, pad: float = 0.0) -> bool:
    return box[0] - pad <= x <= box[2] + pad and box[1] - pad <= y <= box[3] + pad


def clean_rows(rows: list[list[str | None]]) -> list[list[str]]:
    """Table rows without the empty columns and rows that stray rules create."""
    rows = [[(c or "").strip().replace("\n", " ") for c in r] for r in rows]
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows]
    keep = [k for k in range(width) if any(r[k] for r in rows)]
    return [[r[k] for k in keep] for r in rows if any(r[k] for k in keep)]


def real_nested(rows: list[list[str]]) -> bool:
    """A table inside a cell, as opposed to boxes that merely overlap some text."""
    return len(rows) >= 2 and sum(1 for r in rows if sum(1 for c in r if c) >= 2) >= 2


def _paragraphs(words: list[dict], breaks: list[float]) -> list[tuple[float, Box, str, bool, bool]]:
    """One column's words as paragraphs: (top, bbox, text, is_bullet, is_bold). A new paragraph
    starts at a bullet, after a gap, or across a row rule."""
    lines: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (round(float(w["top"])), float(w["x0"]))):
        if lines and abs(float(lines[-1][0]["top"]) - float(w["top"])) <= 2.5:
            lines[-1].append(w)
        else:
            lines.append([w])
    out: list[tuple[float, Box, str, bool, bool]] = []
    cur: list[list[dict]] = []

    def flush() -> None:
        if not cur:
            return
        text = ""
        for ln in cur:
            t = " ".join(w["text"] for w in sorted(ln, key=lambda w: float(w["x0"])))
            text = text[:-1] + t if text.endswith("-") and t[:1].islower() else f"{text} {t}".strip()
        flat = [w for ln in cur for w in ln]
        bullet = bool(BULLET.match(text))
        text = BULLET.sub("", text).strip()
        bold = sum("bold" in str(w.get("fontname", "")).lower() for w in flat) > len(flat) / 2
        box = (min(float(w["x0"]) for w in flat), min(float(w["top"]) for w in flat),
               max(float(w["x1"]) for w in flat), max(float(w["bottom"]) for w in flat))
        if text:
            out.append((box[1], box, text, bullet, bold))
        cur.clear()

    for ln in lines:
        text = " ".join(w["text"] for w in sorted(ln, key=lambda w: float(w["x0"])))
        top = min(float(w["top"]) for w in ln)
        if cur:
            prev = cur[-1]
            prev_bottom = max(float(w["bottom"]) for w in prev)
            height = max(statistics.median(float(w["bottom"]) - float(w["top"]) for w in prev), 1.0)
            crossed = any(prev_bottom - 1 <= b <= top + 1 for b in breaks)
            if BULLET.match(text) or top - prev_bottom > 0.75 * height or crossed:
                flush()
        cur.append(ln)
    flush()
    return out


def build(page_number: int, table, words: list[dict], images: list[dict], nested: list[tuple[Box, list[list[str]]]],
          page_area: float) -> LayoutTable:
    """Everything inside one table, by column and height."""
    bbox = tuple(float(v) for v in table.bbox)
    edges = column_edges(table)
    ncols = max(1, len(edges) - 1)

    def col_of(x: float) -> int:
        return min(ncols - 1, max(0, sum(1 for e in edges[1:-1] if x >= e)))

    # A first row of short labels in every column is the header.
    headers = None
    header_bottom = bbox[1]
    first = table.rows[0] if table.rows else None
    if first is not None and len(table.rows) >= 1 and all(c is not None for c in first.cells) and len(first.cells) == ncols:
        tops = [float(c[1]) for c in first.cells]
        bottoms = [float(c[3]) for c in first.cells]
        if max(bottoms) - min(tops) <= 34:
            labels = []
            for c in first.cells:
                inside = [w for w in words if _inside(tuple(float(v) for v in c), (float(w["x0"]) + float(w["x1"])) / 2,
                                                      (float(w["top"]) + float(w["bottom"])) / 2)]
                labels.append(" ".join(w["text"] for w in sorted(inside, key=lambda w: (round(float(w["top"])), float(w["x0"])))))
            if all(labels) and all(len(t.split()) <= 4 for t in labels):
                headers, header_bottom = labels, max(bottoms)

    lt = LayoutTable(page=page_number, bbox=bbox, edges=edges, headers=headers, nested=[b for b, _ in nested])
    row_breaks = sorted({float(c[1]) for r in table.rows for c in r.cells if c is not None})
    by_col: dict[int, list[dict]] = {}
    for w in words:
        cx, cy = (float(w["x0"]) + float(w["x1"])) / 2, (float(w["top"]) + float(w["bottom"])) / 2
        if not _inside(bbox, cx, cy, 1.0) or cy <= header_bottom or any(_inside(b, cx, cy, 1.0) for b, _ in nested):
            continue
        by_col.setdefault(col_of(cx), []).append(w)
    for col, ws in by_col.items():
        for top, box, text, bullet, bold in _paragraphs(ws, row_breaks):
            lt.items.append(Item(y=top, col=col, kind="text", text=text, bbox=box, bullet=bullet, bold=bold))
    for box, rows in nested:
        lt.items.append(Item(y=box[1], col=col_of((box[0] + box[2]) / 2), kind="table", bbox=box, rows=rows))
    for img in images:
        b = (float(img["x0"]), float(img["top"]), float(img["x1"]), float(img["bottom"]))
        w, h = b[2] - b[0], b[3] - b[1]
        if not _inside(bbox, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2) or min(w, h) < 28 or w * h < 0.003 * page_area:
            continue
        lt.items.append(Item(y=min(b[1] + 40.0, (b[1] + b[3]) / 2), col=col_of((b[0] + b[2]) / 2), kind="image", bbox=b))
    lt.items.sort(key=lambda i: (i.y, i.col))

    tall = any(c is not None and float(c[3]) - float(c[1]) > 250 for r in table.rows for c in r.cells)
    long_text = any(i.kind == "text" and len(i.text.split()) > 45 for i in lt.items)
    lt.candidate = tall or long_text or any(i.kind in ("image", "table") for i in lt.items)
    return lt
