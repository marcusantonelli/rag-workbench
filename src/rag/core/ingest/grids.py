"""Ruled grids (tables) as blocks, shared by the PDF and Word readers so both read tables by the
same rules: a title row, part names inside the grid, form-like title blocks, boxes around text."""

from __future__ import annotations

import re
from typing import Callable

from rag.core.types import Block
from rag.core.util import markdown_table

_FORM_CELL_RE = re.compile(r"[:#]")


def grid_blocks(rows: list[list[str]], page: int | None, caption: str | None,
                split_header: Callable[[list[list[str]]], tuple[list[str], list[list[str]]]] | None = None) -> list[Block]:
    """A ruled grid as blocks. Most are one table. A first row with one cell is the table's title
    (meta title_row); a one-cell row further down names the part after it; a grid whose rows never
    have two filled cells is boxes around text (lines in order); a title block ("Label: value"
    cells, then a real table) comes out as lines of text and a captioned table. ``split_header``
    picks the column names from a part's rows (default: its first row)."""
    from rag.core.ingest.forms import form_lines
    out: list[Block] = []
    segment: list[list[str]] = []
    pending = caption
    titled = False          # the caption is the table's own first row

    def flush() -> None:
        nonlocal pending
        if not segment:
            return
        first = segment[0]
        filled = [c for c in first if c]
        form = len(first) >= 2 and len(filled) <= len(first) / 2 and any(_FORM_CELL_RE.search(c) for c in filled)
        pairs = form_lines(segment) if len(segment) > 1 or form else None
        boxes_only = not any(sum(1 for c in r if c) >= 2 for r in segment)
        if form or len(segment) == 1 or pairs is not None or boxes_only:
            lines = pairs if pairs is not None else [c for r in segment for c in r if c]
            text = "\n".join(([pending] if pending else []) + lines)
            out.append(Block(kind="paragraph", text=text, page=page, meta={"form": pairs is not None}))
        else:
            # each part of the grid on its own, so one part's empty columns don't disturb another's
            from rag.core.ingest.layout_table import clean_rows, tidy_grid

            tidy = clean_rows(tidy_grid([list(r) for r in segment]))
            headers, data = split_header(tidy) if split_header and len(tidy) >= 2 else (tidy[0], tidy[1:])
            text = (f"**{pending}**\n\n" if pending else "") + markdown_table(headers, data)
            out.append(Block(kind="table", text=text, page=page,
                             meta={"headers": headers, "rows": data, "caption": pending, "title_row": titled and not out}))
        pending = None
        segment.clear()

    width = max(len(r) for r in rows)
    for n, r in enumerate(rows):
        filled = [c for c in r if c]
        # a one-cell row names the part that follows; at the top, it is the table's title
        title_row = (len(filled) == 1 and len(filled[0].split()) <= 12 and width >= 2 and n + 1 < len(rows)
                     and sum(1 for c in rows[n + 1] if c) >= 2)
        if not segment and title_row and pending is None:
            pending = filled[0]
            titled = True
            continue
        if width >= 3 and len(filled) == 1 and len(filled[0].split()) <= 8 and segment:
            flush()                       # a one-cell row: the name of the part that follows
            pending = filled[0]
            continue
        if segment:
            first = segment[0]
            first_form = len([c for c in first if c]) <= len(first) / 2 and any(_FORM_CELL_RE.search(c) for c in first if c)
            if first_form and len(filled) > len(r) / 2:
                flush()                   # label cells above, a full row below: a table starts here
        segment.append(r)
    flush()
    return out
