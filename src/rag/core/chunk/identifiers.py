"""Split code identifiers so keyword search matches both spellings.

``GetLotYield`` -> ``get lot yield``; ``HTTPServer2`` -> ``http server 2``;
``lot_id`` -> ``lot id``.
"""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
_PARTS = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def split_identifier(name: str) -> list[str]:
    parts: list[str] = []
    for piece in name.split("_"):
        parts.extend(p.lower() for p in _PARTS.findall(piece))
    return parts


def identifier_terms(text: str, limit: int = 120) -> str:
    """Space-separated split forms of multi-part identifiers found in text."""
    seen: list[str] = []
    for ident in _IDENT.findall(text):
        parts = split_identifier(ident)
        if len(parts) < 2:
            continue
        form = " ".join(parts)
        if form not in seen:
            seen.append(form)
        if len(seen) >= limit:
            break
    return "; ".join(seen)


def query_identifiers(query: str) -> list[str]:
    """Tokens in a query that look like codes, part numbers or symbols."""
    found = []
    for tok in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.\-/#]*[A-Za-z0-9]", query):
        has_digit = any(c.isdigit() for c in tok)
        has_sep = any(c in "_-./#" for c in tok)
        camel = bool(re.search(r"[a-z][A-Z]", tok))
        if (has_digit and len(tok) >= 3) or (has_sep and len(tok) >= 4) or camel:
            found.append(tok)
    return found
