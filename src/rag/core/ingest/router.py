"""Detect a file's type and run the right extractor, with a content-hash cache."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from rag.core.ingest.context import ExtractContext, UnsupportedFile
from rag.core.types import ExtractedDoc
from rag.core.util import long_path

EXTENSION_TYPES = {
    ".pdf": "pdf", ".docx": "docx", ".doc": "doc", ".txt": "txt", ".md": "md", ".markdown": "md",
    ".html": "html", ".htm": "html", ".csv": "csv", ".xlsx": "xlsx", ".xlsm": "xlsx",
    ".sql": "sql", ".cs": "cs", ".py": "py",
}

# Bump when an extractor's output changes, so cached extractions are redone.
EXTRACTOR_VERSION = "1"


def detect_type(path: Path) -> str | None:
    ext = Path(path).suffix.lower()
    ftype = EXTENSION_TYPES.get(ext)
    if ftype is None:
        return None
    try:
        with open(long_path(path), "rb") as fh:
            head = fh.read(8)
    except OSError:
        return ftype
    if ftype == "pdf" and not head.startswith(b"%PDF"):
        return "txt" if head and all(32 <= b < 127 or b in (9, 10, 13) for b in head) else None
    if ftype in ("docx", "xlsx") and not head.startswith(b"PK"):
        if head.startswith(b"\xd0\xcf\x11\xe0"):
            return "doc" if ftype == "docx" else None   # legacy .xls renamed: unsupported
        return None
    return ftype


def _extractor(ftype: str) -> Callable[..., ExtractedDoc]:
    if ftype == "pdf":
        from rag.core.ingest.pdf import extract_pdf
        return extract_pdf
    if ftype == "docx":
        from rag.core.ingest.docx import extract_docx
        return extract_docx
    if ftype == "doc":
        from rag.core.ingest.docx import extract_doc
        return extract_doc
    if ftype == "txt":
        from rag.core.ingest.text import extract_txt
        return extract_txt
    if ftype == "md":
        from rag.core.ingest.text import extract_markdown
        return extract_markdown
    if ftype == "html":
        from rag.core.ingest.html import extract_html
        return extract_html
    if ftype == "csv":
        from rag.core.ingest.sheets import extract_csv
        return extract_csv
    if ftype == "xlsx":
        from rag.core.ingest.sheets import extract_xlsx
        return extract_xlsx
    if ftype == "sql":
        from rag.core.ingest.sqlfile import extract_sql
        return extract_sql
    if ftype == "cs":
        from rag.core.ingest.code_csharp import extract_csharp
        return extract_csharp
    if ftype == "py":
        from rag.core.ingest.code_python import extract_python
        return extract_python
    raise UnsupportedFile(f"no extractor for {ftype}")


def extract(path: Path, ftype: str, ctx: ExtractContext, *, cache_dir: Path | None,
            overrides_file: Path | None = None, use_cache: bool = True) -> tuple[ExtractedDoc, bool]:
    """Extract a file, reusing the cached result for identical content.

    Returns (doc, from_cache).
    """
    cache_file = cache_dir / f"{ctx.sha256}.{ftype}.v{EXTRACTOR_VERSION}.json" if cache_dir else None
    if use_cache and cache_file is not None and cache_file.exists():
        try:
            return ExtractedDoc.from_dict(json.loads(cache_file.read_text(encoding="utf-8"))), True
        except (ValueError, KeyError, TypeError):
            pass
    fn = _extractor(ftype)
    if ftype in ("xlsx", "csv"):
        doc = fn(path, ctx, overrides_file=overrides_file)
    else:
        doc = fn(path, ctx)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc.to_dict(), default=str), encoding="utf-8")
        tmp.replace(cache_file)
    return doc, False
