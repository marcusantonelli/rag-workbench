"""Structure-first chunking.

Prose is grouped by section (heading trail) and packed to ~400 tokens with a
small overlap; long sections without sub-headings are first split at topic
shifts. Tables, figures, code and SQL statements are chunked on their own
rules and never mixed with prose. Every chunk's embed_text carries a context
header (file > heading trail > symbol) and, for code, split identifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from rag.core.chunk.breakpoints import split_sentences, topic_segments
from rag.core.chunk.identifiers import identifier_terms
from rag.core.structure import heading_number, identity, is_prefix
from rag.core.types import Block, Chunk, ExtractedDoc
from rag.core.util import markdown_table, normalize_for_hash, sha256_text

_FENCE_RE = re.compile(r"^```[\w+#.-]*\n(.*)\n```$", re.S)


@dataclass
class ChunkSettings:
    target_tokens: int = 400
    max_tokens: int = 500
    min_tokens: int = 40
    overlap_tokens: int = 60
    long_section_tokens: int = 800
    topic_shift_percentile: float = 95
    code_max_tokens: int = 800
    table_chunk_max_rows: int = 1000     # larger tables stored in tables.db get a preview chunk only
    # What each chunk's embedded text starts with. 1: file › heading trail. 2: also a line naming
    # the document (title, document number, revision). Fixed per knowledge base when it is first
    # built (meta embed_context), since vectors made with different headers don't mix.
    context: int = 1

    @classmethod
    def from_config(cls, c: dict[str, Any]) -> "ChunkSettings":
        fields = cls.__dataclass_fields__
        return cls(**{k: c[k] for k in fields if k in c})


def _header(display_path: str, heading_path: str | None, symbol: str | None) -> str:
    parts = [display_path]
    if heading_path:
        parts.append(heading_path)
    if symbol:
        parts.append(symbol)
    return " › ".join(parts)


def finalize(chunk: Chunk, display_path: str, count: Callable[[str], int], doc_line: str | None = None) -> Chunk:
    header = _header(display_path, chunk.heading_path, chunk.symbol if chunk.kind in ("code", "sql") else None)
    if doc_line:
        header = f"{doc_line}\n{header}"
    body = chunk.text
    extra = ""
    if chunk.kind in ("code", "sql"):
        terms = identifier_terms(body)
        if terms:
            extra = "\nidentifiers: " + terms
    chunk.embed_text = f"{header}\n{body}{extra}"
    chunk.n_tokens = count(chunk.text)
    # Hash the content, not the header: the same text in two files (a copied SOP)
    # is embedded once and collapsed to one search result.
    chunk.text_hash = sha256_text(chunk.kind + "\n" + normalize_for_hash(chunk.text))
    return chunk


class Chunker:
    def __init__(self, settings: ChunkSettings, count: Callable[[str], int], embedder=None):
        self.s = settings
        self.count = count
        self.embedder = embedder

    # -- prose -------------------------------------------------------------------
    def _split_oversized(self, unit: str) -> list[str]:
        """Cut a unit longer than max_tokens into word windows of ~target_tokens."""
        words = unit.split()
        pieces: list[str] = []
        cur: list[str] = []
        for w in words:
            cur.append(w)
            if len(cur) % 16 == 0 and self.count(" ".join(cur)) >= self.s.target_tokens:
                pieces.append(" ".join(cur))
                cur = []
        if cur:
            pieces.append(" ".join(cur))
        return pieces

    def _pack(self, units: list[str]) -> list[str]:
        """Greedy packing of text units (sentences/paragraphs) with token overlap."""
        expanded: list[str] = []
        for unit in units:
            expanded.extend(self._split_oversized(unit) if self.count(unit) > self.s.max_tokens else [unit])
        out: list[str] = []
        cur: list[str] = []
        cur_tokens = 0
        for unit in expanded:
            t = self.count(unit)
            if cur and cur_tokens + t > self.s.target_tokens:
                out.append(" ".join(cur))
                # overlap: carry trailing units up to overlap_tokens
                carry: list[str] = []
                carry_tokens = 0
                for prev in reversed(cur):
                    pt = self.count(prev)
                    if carry_tokens + pt > self.s.overlap_tokens:
                        break
                    carry.insert(0, prev)
                    carry_tokens += pt
                if not carry and self.s.overlap_tokens > 0:
                    # Last unit is a whole paragraph: carry its trailing sentences.
                    for sent in reversed(split_sentences(cur[-1])):
                        st = self.count(sent)
                        if carry and carry_tokens + st > self.s.overlap_tokens:
                            break
                        if st > self.s.overlap_tokens:
                            break
                        carry.insert(0, sent)
                        carry_tokens += st
                    carry = [" ".join(carry)] if carry else []
                cur, cur_tokens = carry, carry_tokens
            cur.append(unit)
            cur_tokens += t
        if cur:
            out.append(" ".join(cur))
        return out

    def _prose_chunks(self, blocks: list[Block], heading_path: str | None) -> list[Chunk]:
        if not blocks:
            return []
        text = "\n\n".join(b.text for b in blocks)
        pages = [b.page for b in blocks if b.page is not None]
        lines = [x for b in blocks for x in (b.line_start, b.line_end) if x is not None]
        confs = [b.meta.get("ocr_conf") for b in blocks if b.meta.get("ocr_conf") is not None]
        sheet = next((b.meta.get("sheet") for b in blocks if b.meta.get("sheet")), None)

        def mk(body: str) -> Chunk:
            return Chunk(
                kind="prose", text=body, embed_text="", heading_path=heading_path,
                page_start=min(pages) if pages else None, page_end=max(pages) if pages else None,
                line_start=min(lines) if lines else None, line_end=max(lines) if lines else None,
                ocr_conf=round(min(confs), 3) if confs else None, sheet=sheet,
            )

        total = self.count(text)
        if total <= self.s.max_tokens:
            return [mk(text)]

        if total > self.s.long_section_tokens and self.embedder is not None:
            sentences = split_sentences(text)
            segments = topic_segments(sentences, self.embedder, percentile=self.s.topic_shift_percentile)
        else:
            segments = [None]  # type: ignore[list-item]

        bodies: list[str] = []
        for seg in segments:
            if seg is None:
                units: list[str] = []
                for b in blocks:
                    units.extend([b.text] if self.count(b.text) <= self.s.target_tokens else split_sentences(b.text))
            else:
                units = seg
            bodies.extend(self._pack(units))
        return [mk(b) for b in bodies]

    # -- tables --------------------------------------------------------------------
    def _table_chunks(self, block: Block, heading_path: str | None) -> list[Chunk]:
        meta = block.meta
        table_name = meta.get("table_name")
        headers = meta.get("headers") or []
        rows = meta.get("rows") or []
        caption = meta.get("caption")
        base = dict(kind="table", embed_text="", heading_path=heading_path, page_start=block.page,
                    page_end=block.page_end or block.page, line_start=block.line_start, line_end=block.line_end,
                    asset_path=meta.get("asset_path"), ocr_conf=meta.get("ocr_conf"))
        if table_name and len(rows) > self.s.table_chunk_max_rows:
            preview = markdown_table(headers, rows[:5])
            text = ((f"**{caption}**\n\n" if caption else "") + preview +
                    f"\n\nFull table ({len(rows)} rows) is in tables.db as {table_name}; "
                    f"columns: {', '.join(str(h) for h in headers)}.")
            return [Chunk(text=text, table_name=table_name, **base)]
        if table_name:
            # also stored for SQL, and every row searchable: groups of rows, each under the
            # column names, each saying where it sits and where the whole table is
            base["table_name"] = table_name
            note = f"; the whole table is in tables.db as {table_name}"
        else:
            note = ""
        if not table_name and (self.count(block.text) <= self.s.max_tokens or not rows):
            return [Chunk(text=block.text, **base)]
        out = []
        group: list[list[Any]] = []
        start = 0
        prefix = f"**{caption}**\n\n" if caption else ""
        for i, row in enumerate(rows):
            trial = markdown_table(headers, group + [row])
            if group and self.count(prefix + trial) > self.s.max_tokens:
                out.append(Chunk(text=prefix + markdown_table(headers, group) +
                                 f"\n\n(rows {start + 1}-{i} of {len(rows)}{note})", **base))
                group, start = [], i
            group.append(row)
        if group:
            out.append(Chunk(text=prefix + markdown_table(headers, group) +
                             f"\n\n(rows {start + 1}-{len(rows)} of {len(rows)}{note})", **base))
        return out

    # -- code and SQL ------------------------------------------------------------------
    def _code_chunks(self, block: Block, heading_path: str | None) -> list[Chunk]:
        kind = "sql" if block.kind == "sql" else "code"
        base = dict(kind=kind, embed_text="", heading_path=heading_path, symbol=block.symbol,
                    page_start=block.page, page_end=block.page)
        if self.count(block.text) <= self.s.code_max_tokens:
            return [Chunk(text=block.text, line_start=block.line_start, line_end=block.line_end, **base)]
        m = _FENCE_RE.match(block.text)
        body = m.group(1) if m else block.text
        lang = block.lang or ("sql" if kind == "sql" else "")
        comment = "--" if lang == "sql" else ("#" if lang == "python" else "//")
        sig = (block.meta.get("signature") or "").split("\n")[0][:160]
        lines = body.split("\n")
        out: list[Chunk] = []
        cur: list[str] = []
        cur_start = 0
        budget = self.s.code_max_tokens - self.count(sig) - 10

        def emit(end_idx: int) -> None:
            if not cur:
                return
            text = "\n".join(cur)
            if out and sig:
                text = f"{comment} continued: {sig}\n{text}"
            ls = (block.line_start or 1) + cur_start if block.line_start else None
            le = (block.line_start or 1) + end_idx if block.line_start else None
            out.append(Chunk(text=f"```{lang}\n{text}\n```", line_start=ls, line_end=le, **base))

        cur_tokens = 0
        for i, line in enumerate(lines):
            t = self.count(line) + 1
            if cur and cur_tokens + t > budget:
                # Prefer to cut at the last blank line in the current piece.
                cut = max((j for j, ln in enumerate(cur) if not ln.strip()), default=-1)
                if cut > len(cur) // 2:
                    head, tail = cur[:cut], cur[cut + 1:]
                    cur[:] = head
                    emit(cur_start + len(head) - 1)
                    cur_start = cur_start + cut + 1
                    cur[:] = tail
                    cur_tokens = sum(self.count(x) + 1 for x in cur)
                else:
                    emit(i - 1)
                    cur_start = i
                    cur.clear()
                    cur_tokens = 0
            cur.append(line)
            cur_tokens += t
        emit(len(lines) - 1)
        return out

    # -- figures -------------------------------------------------------------------------
    def _figure_chunks(self, block: Block, heading_path: str | None) -> list[Chunk]:
        """One chunk per figure; a long one (a big flowchart) is cut between lines, each piece
        repeating the figure's title line."""
        base = dict(kind="figure", embed_text="", heading_path=heading_path, page_start=block.page,
                    page_end=block.page, asset_path=block.meta.get("asset_path"), ocr_conf=block.meta.get("ocr_conf"))
        if self.count(block.text) <= self.s.max_tokens:
            return [Chunk(text=block.text, **base)]
        head, _, body = block.text.partition("\n")
        out: list[Chunk] = []
        cur: list[str] = []
        for line in body.split("\n"):
            if cur and self.count(head + " (continued)\n" + "\n".join(cur + [line])) > self.s.max_tokens:
                out.append(Chunk(text=(head if not out else head + " (continued)") + "\n" + "\n".join(cur), **base))
                cur = []
            cur.append(line)
        if cur:
            out.append(Chunk(text=(head if not out else head + " (continued)") + "\n" + "\n".join(cur), **base))
        return out

    # -- driver -----------------------------------------------------------------------------
    def chunk(self, doc: ExtractedDoc, display_path: str) -> list[Chunk]:
        stack: list[tuple[int, str, tuple[int, ...] | None]] = []
        section: list[Block] = []
        chunks: list[Chunk] = []

        def path() -> str | None:
            return " › ".join(t for _, t, _ in stack) or None

        figures: list[Block] = []     # pictures met inside the section being collected

        # A heading with nothing under it before the next heading of its level or above (a cover
        # page's title, a label set large) keeps its words as text: a heading alone is only part
        # of a path, and its words would otherwise not be searchable at all.
        blocks_in: list[Block] = []
        is_code = any(b.kind in ("code", "sql") for b in doc.blocks)   # there headings are symbol names
        own_title = " ".join((doc.title or "").lower().split())
        for k, block in enumerate(doc.blocks):
            blocks_in.append(block)
            if block.kind != "heading" or is_code or " ".join(block.text.lower().split()) == own_title:
                continue        # the document's own title is its title, not text to search
            nxt = next((b for b in doc.blocks[k + 1:]), None)
            if nxt is None or (nxt.kind == "heading" and max(1, nxt.level or 1) <= max(1, block.level or 1)):
                blocks_in.append(Block(kind="paragraph", text=block.text.strip(), page=block.page))

        def flush() -> None:
            prose = self._prose_chunks(section, path())
            # A picture inside a section doesn't cut the section's text in two. One with words of
            # its own (caption, labels, readable text) follows as its own chunk; a bare
            # photograph has nothing to search for, so it rides on a text chunk of its section.
            free = [c for c in prose if not c.asset_path]
            extra: list[Chunk] = []
            for fig in figures:
                if fig.meta.get("bare"):
                    if free:
                        free.pop(0).asset_path = fig.meta.get("asset_path")
                else:
                    extra.extend(self._figure_chunks(fig, path()))
            chunks.extend(prose + extra)
            section.clear()
            figures.clear()

        for block in blocks_in:
            if block.kind == "heading":
                flush()
                level = max(1, block.level or 1)
                title = block.text.strip()
                number = heading_number(title)
                if number is not None and any(n is not None for _, _, n in stack):
                    # A numbered heading goes under the heading whose number it extends, whatever
                    # level the file gave it: "8.1" after "7.0 › 7.8" belongs under "8.0", not 7.0.
                    keep = 0
                    for j, (_, _, n) in enumerate(stack):
                        if n is not None:
                            if not is_prefix(n, number):
                                break
                            keep = j + 1
                        elif keep == j:
                            keep = j + 1      # unnumbered headings above the numbering stay
                    del stack[keep:]
                    if stack:
                        level = max(level, stack[-1][0] + 1)
                else:
                    while stack and stack[-1][0] >= level:
                        stack.pop()
                stack.append((level, title, number))
                continue
            if block.kind in ("paragraph", "note"):
                section.append(block)
                continue
            if block.kind == "figure":
                figures.append(block)
                continue
            flush()
            if block.kind == "table":
                chunks.extend(self._table_chunks(block, path()))
            elif block.kind in ("code", "sql"):
                chunks.extend(self._code_chunks(block, path()))
            else:
                section.append(block)
        flush()

        # One- and two-word prose scraps (a leftover nav label, a logo's text such as
        # "ACME") can't answer anything and crowd out real results.
        kept = [c for c in chunks if not (c.kind == "prose" and len(c.text.split()) < 3 and len(c.text) < 25)]
        if kept:
            chunks = kept

        # A document that is only headings (e.g. a title page) still gets one chunk.
        if not chunks and stack:
            chunks.append(Chunk(kind="prose", text=" / ".join(t for _, t, _ in stack), embed_text=""))
        doc_line = None
        if self.s.context >= 2:
            first = [b.text for b in doc.blocks[:8]]
            doc_line = identity(doc.title, first, display_path).line()
        return [finalize(c, display_path, self.count, doc_line) for c in chunks]


def card_chunk(card: str, table_name: str, display_path: str, sheet: str | None, cell_range: str | None,
               page: int | None, count: Callable[[str], int]) -> Chunk:
    ch = Chunk(kind="table_card", text=card, embed_text="", table_name=table_name, sheet=sheet,
               cell_range=cell_range, page_start=page, page_end=page)
    return finalize(ch, display_path, count)


def row_chunks(headers: list[str], rows: list[list[Any]], row_numbers: list[int], table_name: str,
               display_path: str, sheet: str | None, size: int, count: Callable[[str], int]) -> list[Chunk]:
    """`column: value` row groups, for small tables whose cells hold real text."""
    out = []
    for i in range(0, len(rows), size):
        group = rows[i:i + size]
        nums = row_numbers[i:i + size]
        lines = []
        for n, row in zip(nums, group):
            cells = [f"{h}: {v}" for h, v in zip(headers, row) if v not in (None, "")]
            lines.append(f"Row {n}: " + "; ".join(str(c) for c in cells))
        ch = Chunk(kind="rows", text="\n".join(lines), embed_text="", table_name=table_name, sheet=sheet,
                   cell_range=f"rows {nums[0]}-{nums[-1]}" if nums else None)
        out.append(finalize(ch, display_path, count))
    return out
