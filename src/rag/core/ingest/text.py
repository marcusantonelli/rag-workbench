"""Plain text and Markdown."""

from __future__ import annotations

import re
from pathlib import Path

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block, ExtractedDoc
from rag.core.util import markdown_table, read_text_file

_ATX_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^(```+|~~~+)\s*([\w+#.-]*)")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


def extract_txt(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    text, enc = read_text_file(path)
    blocks: list[Block] = []
    buf: list[str] = []
    start = 1
    for lineno, line in enumerate(text.split("\n") + [""], start=1):
        if line.strip():
            if not buf:
                start = lineno
            buf.append(line.rstrip())
        elif buf:
            blocks.append(Block(kind="paragraph", text="\n".join(buf), line_start=start, line_end=lineno - 1))
            buf = []
    return ExtractedDoc(file_type="txt", title=None, blocks=blocks, extractor="txt",
                        warnings=[] if enc.startswith("utf") else [f"decoded as {enc}"])


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]
    cells = re.split(r"(?<!\\)\|", line)
    return [c.strip().replace("\\|", "|") for c in cells]


def parse_markdown(text: str) -> tuple[list[Block], dict[str, str]]:
    """Line-based Markdown reader: headings, fences, pipe tables, paragraphs."""
    lines = text.split("\n")
    front: dict[str, str] = {}
    i = 0
    if lines and lines[0].strip() == "---":
        for j in range(1, min(len(lines), 200)):
            if lines[j].strip() in ("---", "..."):
                for fm in lines[1:j]:
                    if ":" in fm:
                        k, v = fm.split(":", 1)
                        front[k.strip().lower()] = v.strip().strip("\"'")
                i = j + 1
                break

    blocks: list[Block] = []
    buf: list[str] = []
    buf_start = 0

    def flush(end: int) -> None:
        nonlocal buf
        if buf:
            blocks.append(Block(kind="paragraph", text="\n".join(buf).strip(),
                                line_start=buf_start, line_end=end))
            buf = []

    n = len(lines)
    while i < n:
        line = lines[i]
        lineno = i + 1
        fence = _FENCE_RE.match(line)
        if fence:
            flush(lineno - 1)
            marker, lang = fence.group(1), fence.group(2)
            j = i + 1
            while j < n and not lines[j].startswith(marker[:3]):
                j += 1
            body = "\n".join(lines[i + 1:j])
            blocks.append(Block(kind="code", text=f"```{lang}\n{body}\n```", lang=lang or None,
                                line_start=lineno, line_end=min(j + 1, n)))
            i = j + 1
            continue
        atx = _ATX_RE.match(line)
        if atx:
            flush(lineno - 1)
            blocks.append(Block(kind="heading", text=atx.group(2), level=len(atx.group(1)),
                                line_start=lineno, line_end=lineno))
            i += 1
            continue
        if (i + 1 < n and line.strip() and not buf and re.match(r"^(=+|-+)\s*$", lines[i + 1])
                and "|" not in line):
            flush(lineno - 1)
            level = 1 if lines[i + 1].startswith("=") else 2
            blocks.append(Block(kind="heading", text=line.strip(), level=level,
                                line_start=lineno, line_end=lineno + 1))
            i += 2
            continue
        if "|" in line and i + 1 < n and _TABLE_SEP_RE.match(lines[i + 1]):
            flush(lineno - 1)
            headers = _split_row(line)
            rows = []
            j = i + 2
            while j < n and "|" in lines[j] and lines[j].strip():
                rows.append(_split_row(lines[j]))
                j += 1
            blocks.append(Block(kind="table", text=markdown_table(headers, rows),
                                line_start=lineno, line_end=j,
                                meta={"headers": headers, "rows": rows, "caption": None}))
            i = j
            continue
        if not line.strip():
            flush(lineno - 1)
        else:
            if not buf:
                buf_start = lineno
            buf.append(line.rstrip())
        i += 1
    flush(n)
    return blocks, front


def extract_markdown(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    text, _ = read_text_file(path)
    blocks, front = parse_markdown(text)
    title = front.get("title") or next((b.text for b in blocks if b.kind == "heading" and b.level == 1), None)
    return ExtractedDoc(file_type="md", title=title, blocks=blocks, extractor="markdown")
