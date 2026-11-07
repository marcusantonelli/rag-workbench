"""tables.db: load tables with typed columns, keep a catalog, write table cards."""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

from rag.core.db.connect import connect, read_schema
from rag.core.types import TableData
from rag.core.util import slugify

_NUM_RE = re.compile(r"^[-+]?(\d+|\d{1,3}(,\d{3})+)(\.\d+)?([eE][-+]?\d+)?$")
_INT_RE = re.compile(r"^[-+]?(\d+|\d{1,3}(,\d{3})+)$")
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}([ T]\d{2}:\d{2}(:\d{2}(\.\d+)?)?)?$")
_US_DATE_RE = re.compile(r"^(\d{1,2})/(\d{1,2})/(\d{4})$")
_SQL_RESERVED = {
    "select", "from", "where", "group", "order", "by", "limit", "table", "index", "join",
    "on", "as", "and", "or", "not", "null", "is", "in", "case", "when", "then", "else",
    "end", "values", "insert", "update", "delete", "create", "drop", "having", "union",
    "all", "distinct", "default", "check", "primary", "key", "references", "row", "rows",
}


@dataclass
class ColumnInfo:
    name: str
    header: str
    type: str          # integer | real | date | text
    min: Any = None
    max: Any = None
    examples: list[str] | None = None
    nulls: int = 0

    @property
    def sql_type(self) -> str:
        return {"integer": "INTEGER", "real": "REAL"}.get(self.type, "TEXT")


@dataclass
class TableInfo:
    table_name: str
    source_path: str
    origin: str
    sheet: str | None
    cell_range: str | None
    page: int | None
    caption: str | None
    n_rows: int
    columns: list[ColumnInfo]


def sanitize_columns(headers: list[str]) -> list[str]:
    names: list[str] = []
    seen: dict[str, int] = {}
    for i, header in enumerate(headers):
        name = slugify(str(header or ""), max_len=48)
        if name in ("x", ""):
            name = f"col_{i + 1}"
        if name[0].isdigit():
            name = f"c_{name}"
        if name in _SQL_RESERVED or keyword.iskeyword(name) or name == "_row":
            name = f"{name}_"
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 1
        names.append(name)
    return names


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _infer_column(values: list[Any]) -> tuple[str, list[Any]]:
    """Return (type, converted values) for one column."""
    present = [v for v in values if not _blank(v)]
    if not present:
        return "text", [None] * len(values)

    def convert_all(fn):
        return [None if _blank(v) else fn(v) for v in values]

    if all(isinstance(v, bool) for v in present):
        return "integer", convert_all(int)
    if all(isinstance(v, (datetime, date)) and not isinstance(v, bool) for v in present):
        return "date", convert_all(lambda v: v.isoformat())
    if all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        return "integer", convert_all(int)
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        if all(float(v).is_integer() for v in present):
            return "integer", convert_all(lambda v: int(v))
        return "real", convert_all(float)
    if all(isinstance(v, str) for v in present):
        stripped = [v.strip() for v in present]
        if all(_INT_RE.match(s) for s in stripped):
            return "integer", convert_all(lambda v: int(str(v).strip().replace(",", "")))
        if all(_NUM_RE.match(s) for s in stripped):
            return "real", convert_all(lambda v: float(str(v).strip().replace(",", "")))
        if all(_ISO_DATE_RE.match(s) for s in stripped):
            return "date", convert_all(lambda v: str(v).strip().replace(" ", "T"))
        if all(_US_DATE_RE.match(s) for s in stripped):
            def us(v: Any) -> str:
                m = _US_DATE_RE.match(str(v).strip())
                assert m
                return f"{m.group(3)}-{int(m.group(1)):02d}-{int(m.group(2)):02d}"
            return "date", convert_all(us)
    # Mixed: store as text.
    def as_text(v: Any) -> str:
        if isinstance(v, (datetime, date, time)):
            return v.isoformat()
        if isinstance(v, float) and v.is_integer():
            return str(int(v))
        return str(v)
    return "text", convert_all(as_text)


def _shorten(text: str, n: int = 40) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


class TablesStore:
    def __init__(self, path: Path, *, readonly: bool = False):
        self.path = Path(path)
        self.conn = connect(self.path, readonly=readonly)

    @classmethod
    def create(cls, path: Path) -> "TablesStore":
        store = cls(path)
        store.conn.executescript(read_schema("schema_tables.sql"))
        store.conn.commit()
        return store

    def close(self) -> None:
        self.conn.close()

    # -- writes --------------------------------------------------------------
    def delete_for_doc(self, doc_id: int) -> None:
        with self.conn:
            names = [r[0] for r in self.conn.execute(
                "SELECT table_name FROM _catalog WHERE doc_id = ?", (doc_id,))]
            for name in names:
                self.conn.execute(f'DROP TABLE IF EXISTS "{name}"')
            self.conn.execute("DELETE FROM _catalog WHERE doc_id = ?", (doc_id,))

    def _unique_name(self, base: str) -> str:
        name = f"t_{base}"[:60]
        existing = {r[0] for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        candidate, n = name, 2
        while candidate in existing:
            candidate = f"{name}_{n}"
            n += 1
        return candidate

    def load_table(self, doc_id: int, source_path: str, td: TableData) -> TableInfo:
        width = max([len(td.headers)] + [len(r) for r in td.rows]) if td.rows else len(td.headers)
        headers = list(td.headers) + [""] * (width - len(td.headers))
        rows = [list(r) + [None] * (width - len(r)) for r in td.rows]
        names = sanitize_columns(headers)
        row_numbers = td.row_numbers or list(range(1, len(rows) + 1))

        columns: list[ColumnInfo] = []
        converted_cols: list[list[Any]] = []
        for idx, (name, header) in enumerate(zip(names, headers)):
            ctype, values = _infer_column([r[idx] for r in rows])
            present = [v for v in values if v is not None]
            col = ColumnInfo(name=name, header=str(header or "").strip() or name, type=ctype,
                             nulls=len(values) - len(present))
            if present and ctype in ("integer", "real", "date"):
                col.min, col.max = min(present), max(present)
            if present and ctype == "text":
                distinct: list[str] = []
                for v in present:
                    s = _shorten(str(v))
                    if s and s not in distinct:
                        distinct.append(s)
                    if len(distinct) == 3:
                        break
                col.examples = distinct
            columns.append(col)
            converted_cols.append(values)

        table_name = self._unique_name(slugify(td.name_hint, 48))
        col_sql = ", ".join(f'"{c.name}" {c.sql_type}' for c in columns)
        with self.conn:
            self.conn.execute(f'CREATE TABLE "{table_name}" (_row INTEGER PRIMARY KEY, {col_sql})')
            placeholders = ", ".join("?" for _ in range(len(columns) + 1))
            data = [
                [row_numbers[i]] + [converted_cols[c][i] for c in range(len(columns))]
                for i in range(len(rows))
            ]
            self.conn.executemany(f'INSERT INTO "{table_name}" VALUES ({placeholders})', data)
            info = TableInfo(
                table_name=table_name, source_path=source_path, origin=td.origin, sheet=td.sheet,
                cell_range=td.cell_range, page=td.page, caption=td.caption, n_rows=len(rows),
                columns=columns,
            )
            self.conn.execute(
                "INSERT INTO _catalog(table_name, doc_id, source_path, origin, sheet, cell_range, page, "
                "caption, n_rows, columns_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    table_name, doc_id, source_path, td.origin, td.sheet, td.cell_range, td.page,
                    td.caption, len(rows), json.dumps([c.__dict__ for c in columns], default=str),
                ),
            )
        return info

    def set_card_chunk(self, table_name: str, chunk_id: int) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE _catalog SET card_chunk_id = ? WHERE table_name = ?", (chunk_id, table_name)
            )

    # -- reads ---------------------------------------------------------------
    def catalog(self) -> list[dict[str, Any]]:
        rows = self.conn.execute("SELECT * FROM _catalog ORDER BY table_name").fetchall()
        return [dict(r) for r in rows]

    def describe(self, table_name: str, samples: int = 3) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM _catalog WHERE table_name = ?", (table_name,)).fetchone()
        if row is None:
            raise KeyError(f"unknown table: {table_name}")
        ddl = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name = ?", (table_name,)
        ).fetchone()[0]
        cur = self.conn.execute(f'SELECT * FROM "{table_name}" LIMIT ?', (samples,))
        cols = [d[0] for d in cur.description]
        return {
            "table_name": table_name,
            "create_table": ddl,
            "source_path": row["source_path"],
            "sheet": row["sheet"],
            "cell_range": row["cell_range"],
            "page": row["page"],
            "caption": row["caption"],
            "n_rows": row["n_rows"],
            "columns": json.loads(row["columns_json"]),
            "sample_rows": [dict(zip(cols, r)) for r in cur.fetchall()],
        }


def card_text(info: TableInfo) -> str:
    """Plain-text description of a table, embedded so search can find it."""
    where = info.source_path
    bits = []
    if info.sheet:
        bits.append(f'sheet "{info.sheet}"')
    if info.cell_range:
        bits.append(info.cell_range)
    if info.page:
        bits.append(f"page {info.page}")
    loc = f" ({', '.join(bits)})" if bits else ""
    lines = [f"Table {info.table_name} from {where}{loc}, {info.n_rows} rows."]
    if info.caption:
        lines.append(f"Caption: {info.caption}")
    lines.append("Columns:")
    for c in info.columns:
        if c.type in ("integer", "real", "date") and c.min is not None:
            desc = f"{c.type}, {c.min} to {c.max}"
        elif c.examples:
            desc = "text, e.g. " + ", ".join(f'"{e}"' for e in c.examples)
        else:
            desc = c.type
        header = f' (header "{c.header}")' if c.header and slugify(c.header, 48) != c.name else ""
        lines.append(f"- {c.name}{header}: {desc}")
    lines.append(f"Query it with SQL against table {info.table_name}.")
    return "\n".join(lines)
