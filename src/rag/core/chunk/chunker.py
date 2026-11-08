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


def finalize(chunk: Chunk, display_path: str, count: Callable[[str], int]) -> Chunk:
    header = _header(display_path, chunk.heading_path, chunk.symbol if chunk.kind in ("code", "sql") else None)
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
                    page_end=block.page_end or block.page, line_start=block.line_start, line_end=block.line_end)
        if table_name:
            preview = markdown_table(headers, rows[:5])
            text = ((f"**{caption}**\n\n" if caption else "") + preview +
                    f"\n\nFull table ({len(rows)} rows) is in tables.db as {table_name}; "
                    f"columns: {', '.join(str(h) for h in headers)}.")
            return [Chunk(text=text, table_name=table_name, **base)]
        if self.count(block.text) <= self.s.max_tokens or not rows:
            return [Chunk(text=block.text, **base)]
        out = []
        group: list[list[Any]] = []
        start = 0
        prefix = f"**{caption}**\n\n" if caption else ""
        for i, row in enumerate(rows):
            trial = markdown_table(headers, group + [row])
            if group and self.count(prefix + trial) > self.s.max_tokens:
                out.append(Chunk(text=prefix + markdown_table(headers, group) +
                                 f"\n\n(rows {start + 1}-{i} of {len(rows)})", **base))
                group, start = [], i
            group.append(row)
        if group:
            out.append(Chunk(text=prefix + markdown_table(headers, group) +
                             f"\n\n(rows {start + 1}-{len(rows)} of {len(rows)})", **base))
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
    def _figure_chunk(self, block: Block, heading_path: str | None) -> Chunk:
        return Chunk(
            kind="figure", text=block.text, embed_text="", heading_path=heading_path,
            page_start=block.page, page_end=block.page, asset_path=block.meta.get("asset_path"),
            ocr_conf=block.meta.get("ocr_conf"),
        )

    # -- driver -----------------------------------------------------------------------------
    def chunk(self, doc: ExtractedDoc, display_path: str) -> list[Chunk]:
        stack: list[tuple[int, str]] = []
        section: list[Block] = []
        chunks: list[Chunk] = []

        def path() -> str | None:
            return " › ".join(t for _, t in stack) or None

        def flush() -> None:
            chunks.extend(self._prose_chunks(section, path()))
            section.clear()

        for block in doc.blocks:
            if block.kind == "heading":
                flush()
                level = max(1, block.level or 1)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, block.text.strip()))
                continue
            if block.kind in ("paragraph", "note"):
                section.append(block)
                continue
            flush()
            if block.kind == "table":
                chunks.extend(self._table_chunks(block, path()))
            elif block.kind == "figure":
                chunks.append(self._figure_chunk(block, path()))
            elif block.kind in ("code", "sql"):
                chunks.extend(self._code_chunks(block, path()))
            else:
                section.append(block)
        flush()

        # One- and two-word prose scraps (a leftover nav label, a logo's text such as
        # "CAMSTAR") can't answer anything and crowd out real results.
        kept = [c for c in chunks if not (c.kind == "prose" and len(c.text.split()) < 3 and len(c.text) < 25)]
        if kept:
            chunks = kept

        # A document that is only headings (e.g. a title page) still gets one chunk.
        if not chunks and stack:
            chunks.append(Chunk(kind="prose", text=" / ".join(t for _, t in stack), embed_text=""))
        return [finalize(c, display_path, self.count) for c in chunks]


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
