"""CSV and XLSX: regions become tables for tables.db; stray text becomes notes.

Region detection: a sheet is split at fully blank rows, then each region at
fully blank columns. A region's header row is the first row whose non-empty
cells are all text and fill at least half the width; two stacked text rows
above numeric data become "Parent / Child" headers. Merged cells are
forward-filled. One-row regions are notes (prose), not tables.

Per-workbook overrides go in <workspace>/sheets.toml:

    [[sheet]]
    file = "*Shipments*.xlsx"   # glob on the file name
    sheet = "Summary"           # optional; all sheets when omitted
    skip = false
    header_row = 3              # 1-based row of the header within the sheet
"""

from __future__ import annotations

import csv
import fnmatch
import io
import tomllib
from datetime import date, datetime
from pathlib import Path
from typing import Any

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block, ExtractedDoc, TableData
from rag.core.util import cell_to_str, col_letter, long_path, read_text_file


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def _is_texty(v: Any) -> bool:
    return isinstance(v, str) and not _blank(v) and not v.strip().replace(".", "", 1).replace(",", "").lstrip("-").isdigit()


def _split_regions(grid: list[list[Any]]) -> list[tuple[int, int, int, int]]:
    """Return (r0, r1, c0, c1) inclusive bounds of non-blank regions."""
    regions = []
    n_rows = len(grid)
    r = 0
    while r < n_rows:
        if all(_blank(v) for v in grid[r]):
            r += 1
            continue
        r0 = r
        while r < n_rows and not all(_blank(v) for v in grid[r]):
            r += 1
        r1 = r - 1
        width = max(len(row) for row in grid[r0:r1 + 1])
        col_blank = [all(_blank(grid[i][c]) if c < len(grid[i]) else True for i in range(r0, r1 + 1))
                     for c in range(width)]
        c = 0
        while c < width:
            if col_blank[c]:
                c += 1
                continue
            c0 = c
            while c < width and not col_blank[c]:
                c += 1
            regions.append((r0, r1, c0, c - 1))
    return regions


def _header_rows(rows: list[list[Any]]) -> int:
    """How many leading rows form the header (0, 1 or 2)."""
    if len(rows) < 2:
        return 0
    width = max(len(r) for r in rows)

    def header_like(row: list[Any]) -> bool:
        present = [v for v in row if not _blank(v)]
        return bool(present) and all(_is_texty(v) for v in present) and len(present) >= max(1, width // 2)

    if not header_like(rows[0]):
        return 0
    if len(rows) >= 3 and header_like(rows[1]):
        data_row = rows[2]
        if any(isinstance(v, (int, float, datetime, date)) and not isinstance(v, bool) for v in data_row):
            return 2
    return 1


def _combine_headers(rows: list[list[Any]], n: int, width: int) -> list[str]:
    if n == 0:
        return [f"col_{i + 1}" for i in range(width)]
    if n == 1:
        return [cell_to_str(rows[0][i]) if i < len(rows[0]) else "" for i in range(width)]
    top, sub = rows[0], rows[1]
    out = []
    last_parent = ""
    for i in range(width):
        parent = cell_to_str(top[i]) if i < len(top) else ""
        last_parent = parent or last_parent
        child = cell_to_str(sub[i]) if i < len(sub) else ""
        out.append(f"{last_parent} / {child}" if last_parent and child and last_parent != child else (child or last_parent))
    return out


def _region_to_parts(
    grid: list[list[Any]], bounds: tuple[int, int, int, int], *, sheet: str | None,
    name_hint: str, origin: str, forced_header: int | None = None,
) -> tuple[TableData | None, Block | None]:
    r0, r1, c0, c1 = bounds
    rows = [[grid[r][c] if c < len(grid[r]) else None for c in range(c0, c1 + 1)] for r in range(r0, r1 + 1)]
    width = c1 - c0 + 1
    cell_range = f"{col_letter(c0)}{r0 + 1}:{col_letter(c1)}{r1 + 1}"
    if len(rows) == 1 or width == 1 and len(rows) <= 3:
        text = " ".join(cell_to_str(v) for row in rows for v in row if not _blank(v))
        note = Block(kind="paragraph", text=text, meta={"sheet": sheet, "cell_range": cell_range})
        return None, note
    if forced_header is not None:
        n_head = max(0, min(forced_header - (r0 + 1) + 1, len(rows) - 1))
    else:
        n_head = _header_rows(rows)
    headers = _combine_headers(rows, n_head, width)
    data = rows[n_head:]
    row_numbers = list(range(r0 + 1 + n_head, r1 + 2))
    keep = [i for i, row in enumerate(data) if not all(_blank(v) for v in row)]
    td = TableData(
        name_hint=name_hint, headers=headers, rows=[data[i] for i in keep], sheet=sheet,
        cell_range=cell_range, origin=origin, row_numbers=[row_numbers[i] for i in keep],
    )
    return td, None


def _load_overrides(path: Path, overrides_file: Path | None) -> list[dict[str, Any]]:
    if overrides_file is None or not overrides_file.exists():
        return []
    with open(overrides_file, "rb") as fh:
        rules = tomllib.load(fh).get("sheet", [])
    return [r for r in rules if fnmatch.fnmatch(path.name, r.get("file", "*"))]


def extract_xlsx(path: Path, ctx: ExtractContext, overrides_file: Path | None = None) -> ExtractedDoc:
    import openpyxl

    wb = openpyxl.load_workbook(long_path(path), data_only=True, read_only=False)
    rules = _load_overrides(Path(path), overrides_file)
    tables: list[TableData] = []
    blocks: list[Block] = []
    warnings: list[str] = []
    stem = Path(path).stem
    for ws in wb.worksheets:
        rule = next((r for r in rules if r.get("sheet") in (None, ws.title)), None)
        if rule and rule.get("skip"):
            continue
        if ws.sheet_state != "visible":
            warnings.append(f"skipped hidden sheet {ws.title!r}")
            continue
        grid = [list(row) for row in ws.iter_rows(values_only=True)]
        if not grid:
            continue
        for rng in ws.merged_cells.ranges:
            value = ws.cell(rng.min_row, rng.min_col).value
            for r in range(rng.min_row, rng.max_row + 1):
                for c in range(rng.min_col, rng.max_col + 1):
                    if r - 1 < len(grid) and c - 1 < len(grid[r - 1]):
                        grid[r - 1][c - 1] = value
        sheet_tables: list[TableData] = []
        for bounds in _split_regions(grid):
            td, note = _region_to_parts(grid, bounds, sheet=ws.title, name_hint=f"{stem}__{ws.title}",
                                        origin="sheet", forced_header=(rule or {}).get("header_row"))
            if td is not None and td.rows:
                sheet_tables.append(td)
            elif note is not None and note.text:
                note.text = f"[{ws.title}] {note.text}"
                blocks.append(note)
        if len(sheet_tables) > 1:
            for i, td in enumerate(sheet_tables, start=1):
                td.name_hint = f"{td.name_hint}__{i}"
        tables.extend(sheet_tables)
    wb.close()
    return ExtractedDoc(file_type="xlsx", title=None, blocks=blocks, tables=tables,
                        extractor="xlsx-openpyxl", warnings=warnings)


def extract_csv(path: Path, ctx: ExtractContext, overrides_file: Path | None = None) -> ExtractedDoc:
    text, enc = read_text_file(path)
    sample = text[:20000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = [r for r in csv.reader(io.StringIO(text), dialect)]
    rows = [r for r in rows if any(c.strip() for c in r)]
    if not rows:
        return ExtractedDoc(file_type="csv", title=None, extractor="csv", warnings=["empty CSV"])
    width = max(len(r) for r in rows)
    grid: list[list[Any]] = [r + [""] * (width - len(r)) for r in rows]
    n_head = 1 if _header_rows(grid) else 0
    try:
        if n_head == 0 and csv.Sniffer().has_header(sample):
            n_head = 1
    except csv.Error:
        pass
    headers = _combine_headers(grid, n_head, width)
    td = TableData(
        name_hint=Path(path).stem, headers=headers, rows=grid[n_head:], origin="csv",
        cell_range=f"A1:{col_letter(width - 1)}{len(grid)}",
        row_numbers=list(range(1 + n_head, len(grid) + 1)),
    )
    return ExtractedDoc(file_type="csv", title=None, tables=[td], extractor="csv",
                        warnings=[] if enc.startswith("utf") else [f"decoded as {enc}"])
