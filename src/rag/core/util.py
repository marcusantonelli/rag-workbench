"""Small helpers: hashing, paths, slugs, Markdown tables, encodings."""

from __future__ import annotations

import hashlib
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(long_path(path), "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def long_path(path: Path | str) -> str:
    """Windows paths over 260 characters need the extended-length prefix."""
    p = str(Path(path).resolve()) if not str(path).startswith("\\\\?\\") else str(path)
    if sys.platform == "win32" and len(p) >= 240 and not p.startswith("\\\\?\\"):
        if p.startswith("\\\\"):
            return "\\\\?\\UNC\\" + p[2:]
        return "\\\\?\\" + p
    return p


def slugify(text: str, max_len: int = 40) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = _SLUG_RE.sub("_", text.lower()).strip("_")
    return (slug[:max_len].rstrip("_")) or "x"


def decode_bytes(data: bytes) -> tuple[str, str]:
    """Decode text of unknown encoding. Returns (text, encoding)."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace"), "utf-8-sig"
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace"), "utf-16"
    try:
        return data.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        pass
    from charset_normalizer import from_bytes

    best = from_bytes(data).best()
    if best is not None:
        return str(best), best.encoding
    return data.decode("cp1252", errors="replace"), "cp1252"


def read_text_file(path: Path) -> tuple[str, str]:
    with open(long_path(path), "rb") as fh:
        text, enc = decode_bytes(fh.read())
    return text.replace("\r\n", "\n").replace("\r", "\n"), enc


def cell_to_str(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def md_escape_cell(text: str) -> str:
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ").strip()


def markdown_table(headers: list[str], rows: Iterable[list[Any]]) -> str:
    headers = [md_escape_cell(h) or f"col_{i + 1}" for i, h in enumerate(headers)]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        cells = [md_escape_cell(cell_to_str(c)) for c in row]
        cells += [""] * (len(headers) - len(cells))
        lines.append("| " + " | ".join(cells[: len(headers)]) + " |")
    return "\n".join(lines)


def col_letter(index: int) -> str:
    """0-based column index to Excel letters."""
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(65 + rem) + letters
    return letters


def printable_ratio(text: str) -> float:
    if not text:
        return 0.0
    good = sum(1 for ch in text if ch.isprintable() or ch in "\n\t")
    return good / len(text)


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes"}
