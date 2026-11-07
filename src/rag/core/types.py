"""Data types shared by extractors, the chunker and the store."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Block:
    """One structural unit of a normalized document, as Markdown.

    kind: heading | paragraph | table | figure | code | sql | note
    """

    kind: str
    text: str
    level: int = 0                       # heading level (1 = top)
    page: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    symbol: str | None = None            # code: Namespace.Class.Member
    lang: str | None = None              # code language
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Block":
        return cls(**d)


@dataclass
class TableData:
    """A table destined for tables.db (spreadsheet region or large document table)."""

    name_hint: str
    headers: list[str]
    rows: list[list[Any]]
    sheet: str | None = None
    cell_range: str | None = None
    caption: str | None = None
    page: int | None = None
    origin: str = "sheet"                # sheet | csv | doc_table
    row_numbers: list[int] | None = None # source row number per data row

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TableData":
        return cls(**d)


@dataclass
class ExtractedDoc:
    file_type: str
    title: str | None
    blocks: list[Block] = field(default_factory=list)
    tables: list[TableData] = field(default_factory=list)
    n_pages: int | None = None
    extractor: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_type": self.file_type,
            "title": self.title,
            "blocks": [b.to_dict() for b in self.blocks],
            "tables": [t.to_dict() for t in self.tables],
            "n_pages": self.n_pages,
            "extractor": self.extractor,
            "warnings": self.warnings,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExtractedDoc":
        return cls(
            file_type=d["file_type"],
            title=d.get("title"),
            blocks=[Block.from_dict(b) for b in d.get("blocks", [])],
            tables=[TableData.from_dict(t) for t in d.get("tables", [])],
            n_pages=d.get("n_pages"),
            extractor=d.get("extractor", ""),
            warnings=list(d.get("warnings", [])),
        )


@dataclass
class Chunk:
    kind: str                     # prose | table | figure | code | sql | table_card | rows
    text: str                     # shown and cited
    embed_text: str               # context header + text (+ identifier forms)
    heading_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    line_start: int | None = None
    line_end: int | None = None
    sheet: str | None = None
    cell_range: str | None = None
    table_name: str | None = None
    ocr_conf: float | None = None
    asset_path: str | None = None
    symbol: str | None = None
    n_tokens: int = 0
    text_hash: str = ""
