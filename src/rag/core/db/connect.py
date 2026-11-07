"""Opening SQLite connections, with sqlite-vec loaded where needed."""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path


class ExtensionLoadingUnavailable(RuntimeError):
    pass


def _uri(path: Path, readonly: bool) -> str:
    uri = Path(path).resolve().as_uri()
    return uri + ("?mode=ro" if readonly else "?mode=rwc")


def connect(path: Path, *, readonly: bool = False, load_vec: bool = False) -> sqlite3.Connection:
    conn = sqlite3.connect(_uri(path, readonly), uri=True, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    if load_vec:
        load_sqlite_vec(conn)
    if not readonly:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
    return conn


def load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec

    if not hasattr(conn, "enable_load_extension"):
        raise ExtensionLoadingUnavailable(
            "This Python's sqlite3 module cannot load extensions. Use the python.org "
            "build of Python, or install pysqlite3 / apsw."
        )
    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def read_schema(name: str) -> str:
    return resources.files("rag.core.db").joinpath(name).read_text(encoding="utf-8")
