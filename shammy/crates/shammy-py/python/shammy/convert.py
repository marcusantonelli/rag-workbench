"""Import vectors from other stores into a ``.shammy`` file.

    shammy-convert sqlite kb.db out.shammy --table chunk_vec --id chunk_id --vector embedding
    shammy-convert pgvector "postgresql://user@host/db" out.shammy --table items --id id --vector embedding

or from Python: ``from_sqlite(...)``, ``from_pgvector(...)``, or ``from_rows(...)`` for anything
that can yield ``(id, vector, doc_id, file_type)``.

Ids must be whole numbers. When the source ids are text or UUIDs, rows are numbered from 1 and
the original ids are written to ``<out>.ids.csv`` (number, original id).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from . import Index, storage_dims

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _name(value: str | None, what: str) -> str | None:
    if value is not None and not _NAME.match(value):
        raise ValueError(f"{what} {value!r} is not a plain table or column name")
    return value


def _vector(value: Any) -> np.ndarray:
    """A float32 vector from a float32 blob, a list, or pgvector/JSON text like '[0.1,0.2]'."""
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if raw[:1] == b"[" and raw[-1:] == b"]":   # text that the driver left undecoded
            return np.array(raw[1:-1].split(b","), dtype="<f4")
        return np.frombuffer(raw, dtype="<f4")
    if isinstance(value, str):
        return np.array(value.strip().strip("[]").split(","), dtype="<f4")
    return np.asarray(value, dtype="<f4")


def from_rows(rows: Iterable[tuple[Any, Any, Any, Any]], out: str | Path, *, model: str = "",
              query_instruction: str = "", lossless: bool = True, batch: int = 2000,
              progress=None) -> dict[str, Any]:
    """Write ``(id, vector, doc_id, file_type)`` rows to a new file. ``doc_id`` and ``file_type``
    may be None. Returns counts."""
    out = Path(out)
    it: Iterator = iter(rows)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("the source has no rows") from None
    dims = len(_vector(first[1]))
    numbered = not isinstance(first[0], (int, np.integer))
    mapping = open(str(out) + ".ids.csv", "w", newline="", encoding="utf-8") if numbered else None
    map_writer = csv.writer(mapping) if mapping else None
    n = skipped = 0
    ix = Index.create(out, dims, lossless=lossless, model=model, query_instruction=query_instruction)
    try:
        ids, vecs, docs, types = [], [], [], []

        def flush():
            if ids:
                ix.add(ids, np.stack(vecs), docs, types)
                ids.clear(), vecs.clear(), docs.clear(), types.clear()

        def take(row):
            nonlocal n, skipped
            v = _vector(row[1])
            if len(v) != dims or not np.isfinite(v).all() or not np.any(v):
                skipped += 1
                return
            n += 1
            if map_writer:
                map_writer.writerow([n, row[0].decode("utf-8", "replace") if isinstance(row[0], bytes) else row[0]])
            ids.append(n if numbered else int(row[0]))
            vecs.append(v)
            docs.append(int(row[2]) if row[2] is not None else 0)
            t = row[3]
            types.append("" if t is None else t.decode("utf-8", "replace") if isinstance(t, bytes) else str(t))
            if len(ids) >= batch:
                flush()
                if progress:
                    progress(n)

        take(first)
        for row in it:
            take(row)
        flush()
        ix.commit()
    except BaseException:
        ix.close()
        out.unlink(missing_ok=True)   # no half-written file left behind
        if mapping:
            mapping.close()
            Path(str(out) + ".ids.csv").unlink(missing_ok=True)
        raise
    finally:
        ix.close()
        if mapping:
            mapping.close()
    return {"path": str(out), "rows": n, "skipped": skipped, "dims": dims, "stored_dims": storage_dims(dims),
            "ids_file": str(out) + ".ids.csv" if numbered else None}


def _select(table: str, id: str, vector: str, doc: str | None, file_type: str | None, where: str | None,
            vector_expr: str = "{}") -> str:
    cols = [_name(id, "id column"), vector_expr.format(_name(vector, "vector column")),
            _name(doc, "document column") or "NULL", _name(file_type, "type column") or "NULL"]
    return f"SELECT {', '.join(cols)} FROM {_name(table, 'table')}" + (f" WHERE {where}" if where else "")


def from_sqlite(db: str | Path, out: str | Path, *, table: str = "chunk_vec", id: str = "rowid",
                vector: str = "embedding", doc: str | None = None, file_type: str | None = None,
                where: str | None = None, **kw) -> dict[str, Any]:
    """Import from a SQLite table whose vector column holds float32 blobs or JSON text. That covers
    sqlite-vec ``vec0`` tables (the sqlite-vec package must be installed) and ordinary tables."""
    import sqlite3

    conn = sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)
    try:
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except (ImportError, AttributeError, sqlite3.Error):
            pass  # only needed for vec0 virtual tables; the query below says so if it is
        try:
            cur = conn.execute(_select(table, id, vector, doc, file_type, where))
        except sqlite3.OperationalError as exc:
            if "no such module" in str(exc):
                raise RuntimeError(f"{table} is a sqlite-vec table; install sqlite-vec (pip install sqlite-vec) "
                                   "with a Python whose sqlite3 can load extensions") from exc
            raise
        return from_rows(cur, out, **kw)
    finally:
        conn.close()


def from_pgvector(dsn_or_connection, out: str | Path, *, table: str, id: str = "id", vector: str = "embedding",
                  doc: str | None = None, file_type: str | None = None, where: str | None = None, **kw) -> dict[str, Any]:
    """Import from a PostgreSQL table with a pgvector ``vector`` column. Takes a
    connection string or an open psycopg / psycopg2 connection; rows are streamed, not loaded at once."""
    conn, own = dsn_or_connection, False
    if isinstance(dsn_or_connection, str):
        try:
            import psycopg as driver
        except ImportError:
            try:
                import psycopg2 as driver
            except ImportError:
                raise RuntimeError("install psycopg to read PostgreSQL: pip install 'psycopg[binary]'") from None
        conn, own = driver.connect(dsn_or_connection), True
    sql = _select(table, id, vector, doc, file_type, where, vector_expr="{}::text")
    try:
        cur = conn.cursor(name="shammy_import")  # server-side: streams rows
        cur.itersize = 2000
        cur.execute(sql)
        return from_rows(cur, out, **kw)
    finally:
        if own:
            conn.close()
        else:
            conn.rollback()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="shammy-convert", description="Import vectors into a .shammy file.")
    sub = p.add_subparsers(dest="source", required=True)
    for name, target, default_id in (("sqlite", "database file", "rowid"), ("pgvector", "connection string", "id")):
        s = sub.add_parser(name)
        s.add_argument("source_path", metavar="SOURCE", help=target)
        s.add_argument("out", help="the .shammy file to create")
        s.add_argument("--table", required=name == "pgvector", default="chunk_vec" if name == "sqlite" else None)
        s.add_argument("--id", default=default_id, help=f"id column (default {default_id})")
        s.add_argument("--vector", default="embedding", help="vector column (default embedding)")
        s.add_argument("--doc", help="whole-number column to group rows by (filter and delete by it later)")
        s.add_argument("--type", dest="file_type", help="text column to filter by later")
        s.add_argument("--where", help="SQL condition choosing the rows to import")
        s.add_argument("--model", default="", help="embedding model name to record in the file")
        s.add_argument("--slim", action="store_true", help="leave out the float16 copies (a third of the size; no exact rescoring)")
    a = p.parse_args(argv)
    fn = from_sqlite if a.source == "sqlite" else from_pgvector
    try:
        res = fn(a.source_path, a.out, table=a.table, id=a.id, vector=a.vector, doc=a.doc, file_type=a.file_type,
                 where=a.where, model=a.model, lossless=not a.slim,
                 progress=lambda n: print(f"\r{n:,} rows", end="", file=sys.stderr))
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"shammy-convert: {exc}", file=sys.stderr)
        return 1
    print(f"\r{res['rows']:,} rows written to {res['path']}" + (f" ({res['skipped']} skipped: wrong size, zero or not finite)" if res["skipped"] else ""))
    if res["stored_dims"] != res["dims"]:
        print(f"vectors have {res['dims']} values, stored padded to {res['stored_dims']}")
    if res["ids_file"]:
        print(f"ids were not whole numbers; rows are numbered and the originals are in {res['ids_file']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
