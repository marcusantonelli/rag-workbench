"""Opening SQLite connections, with sqlite-vec loaded where needed.

sqlite-vec is a loadable extension, and some Python builds (pyenv and other
from-source builds without --enable-loadable-sqlite-extensions) compile the
standard sqlite3 module without extension loading. When that happens a
drop-in module is used instead, in this order: pysqlite3, then sqlean
(``pip install sqlean.py``, wheels for Linux and Windows). Set
RAG_SQLITE_MODULE=stdlib|pysqlite3|sqlean to force one.

Everything that touches SQLite must use ``sqlite_module()`` rather than
``import sqlite3``, so constants and exception classes match the connection.
"""

from __future__ import annotations

import os
import sqlite3 as _stdlib_sqlite3
from functools import lru_cache
from importlib import resources
from pathlib import Path
from types import ModuleType

_CANDIDATES = {
    "stdlib": "sqlite3",
    "pysqlite3": "pysqlite3.dbapi2",
    "sqlean": "sqlean",
}


class ExtensionLoadingUnavailable(RuntimeError):
    pass


def _supports_extensions(mod: ModuleType) -> bool:
    return hasattr(mod.Connection, "enable_load_extension")


@lru_cache(maxsize=1)
def sqlite_module() -> ModuleType:
    """The sqlite3-compatible module that can load extensions."""
    forced = os.environ.get("RAG_SQLITE_MODULE", "").strip().lower()
    if forced:
        if forced not in _CANDIDATES:
            raise ExtensionLoadingUnavailable(
                f"RAG_SQLITE_MODULE={forced!r}; choose from {', '.join(_CANDIDATES)}")
        return __import__(_CANDIDATES[forced], fromlist=["connect"])
    if _supports_extensions(_stdlib_sqlite3):
        return _stdlib_sqlite3
    for name in ("pysqlite3", "sqlean"):
        try:
            mod = __import__(_CANDIDATES[name], fromlist=["connect"])
        except ImportError:
            continue
        if _supports_extensions(mod):
            return mod
    raise ExtensionLoadingUnavailable(
        "This Python's sqlite3 module was built without extension loading, which sqlite-vec "
        "needs. Fix it with:\n    pip install sqlean.py\n(a drop-in sqlite3 with extensions "
        "enabled; picked up automatically). Alternatively use a Python built with "
        "--enable-loadable-sqlite-extensions, such as the python.org or distribution build."
    )


def connect(path: Path, *, readonly: bool = False, load_vec: bool = False):
    sq = sqlite_module()
    uri = Path(path).resolve().as_uri() + ("?mode=ro" if readonly else "?mode=rwc")
    conn = sq.connect(uri, uri=True, check_same_thread=False)
    conn.row_factory = sq.Row
    if load_vec:
        load_sqlite_vec(conn)
    if not readonly:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
    return conn


def load_sqlite_vec(conn) -> None:
    import sqlite_vec

    conn.enable_load_extension(True)
    try:
        sqlite_vec.load(conn)
    finally:
        conn.enable_load_extension(False)


def read_schema(name: str) -> str:
    return resources.files("rag.core.db").joinpath(name).read_text(encoding="utf-8")
