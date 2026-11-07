"""HTML / HTM: strip boilerplate, keep headings, paragraphs, lists, code and tables."""

from __future__ import annotations

import re
from pathlib import Path

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block, ExtractedDoc
from rag.core.util import long_path, markdown_table

_DROP = ["script", "style", "noscript", "nav", "footer", "aside", "form", "iframe", "svg", "template"]
_BLOCK_TAGS = {"p", "li", "pre", "blockquote", "dt", "dd", "figcaption", "caption"}
_HEADINGS = {f"h{i}": i for i in range(1, 7)}


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _own_text(el) -> str:
    """Text of an element without its nested lists (those are read as their own items)."""
    parts = []
    for child in el.children:
        if getattr(child, "name", None) in ("ul", "ol"):
            continue
        parts.append(child.get_text(" ") if hasattr(child, "get_text") else str(child))
    return _clean(" ".join(parts))


def _table_rows(table) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        if tr.find_parent("table") is not table:
            continue  # nested tables are read on their own
        cells = [_clean(c.get_text(" ")) for c in tr.find_all(["th", "td"], recursive=False)]
        if any(cells):
            rows.append(cells)
    return rows


def extract_html(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    from bs4 import BeautifulSoup, UnicodeDammit

    with open(long_path(path), "rb") as fh:
        raw = fh.read()
    dammit = UnicodeDammit(raw, is_html=True)
    soup = BeautifulSoup(dammit.unicode_markup or "", "html.parser")
    title = _clean(soup.title.get_text()) if soup.title else None
    for tag in soup.find_all(_DROP):
        tag.decompose()
    for tag in soup.find_all(attrs={"role": ["navigation", "banner", "contentinfo"]}):
        tag.decompose()
    for tag in soup.find_all("header"):
        if tag.find_parent(["article", "main", "section"]) is None:
            tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup

    blocks: list[Block] = []
    for el in root.find_all(list(_HEADINGS) + list(_BLOCK_TAGS) + ["table", "div"]):
        if el.find_parent("table") is not None and el.name != "table":
            continue
        if el.name == "div":
            # Only bare text sitting directly in a div (no block children).
            if el.find(list(_HEADINGS) + list(_BLOCK_TAGS) + ["table", "div", "ul", "ol"]):
                continue
            text = _clean(el.get_text(" "))
            if text:
                blocks.append(Block(kind="paragraph", text=text))
            continue
        if el.name in _HEADINGS:
            text = _clean(el.get_text(" "))
            if text:
                blocks.append(Block(kind="heading", text=text, level=_HEADINGS[el.name]))
        elif el.name == "table":
            rows = _table_rows(el)
            if len(rows) >= 2:
                caption = el.find("caption")
                cap = _clean(caption.get_text(" ")) if caption else None
                headers, data = rows[0], rows[1:]
                text = (f"**{cap}**\n\n" if cap else "") + markdown_table(headers, data)
                blocks.append(Block(kind="table", text=text, meta={"headers": headers, "rows": data, "caption": cap}))
            elif rows:
                blocks.append(Block(kind="paragraph", text=" | ".join(rows[0])))
        elif el.name == "pre":
            code = el.get_text()
            if code.strip():
                blocks.append(Block(kind="code", text=f"```\n{code.rstrip()}\n```"))
        elif el.name == "li":
            if el.find(["p", "pre"]):
                continue
            text = _own_text(el)
            if text:
                blocks.append(Block(kind="paragraph", text=f"- {text}"))
        elif el.name in ("caption", "figcaption") and el.find_parent("table") is not None:
            continue
        else:
            if el.find(list(_BLOCK_TAGS)):
                continue
            text = _clean(el.get_text(" "))
            if text:
                blocks.append(Block(kind="paragraph", text=text))

    # Merge runs of list items into single paragraphs so lists chunk together.
    merged: list[Block] = []
    for b in blocks:
        if (merged and b.kind == "paragraph" and b.text.startswith("- ")
                and merged[-1].kind == "paragraph" and merged[-1].text.startswith("- ")):
            merged[-1].text += "\n" + b.text
        else:
            merged.append(b)
    return ExtractedDoc(file_type="html", title=title, blocks=merged, extractor="html")
