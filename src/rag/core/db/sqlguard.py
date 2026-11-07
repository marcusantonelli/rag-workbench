"""Read-only SQL execution against tables.db for agent-written queries.

SQLite enforces the rules instead of a parser: the connection is opened
read-only, an authorizer allows only reads, Python's sqlite3 refuses more than
one statement per call, rows are capped, and a progress handler stops queries
that run past the time limit. The C# server should mirror this design.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rag.core.db.connect import connect, sqlite_module

sqlite3 = sqlite_module()  # constants and exceptions must match the connection's module

_ALLOWED_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "fts3_tokenizer"}


class QueryRejected(ValueError):
    """The query was refused or failed; the message is meant for the agent."""


def _authorizer(action: int, arg1: str | None, arg2: str | None, db: str | None, trigger: str | None) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_READ and arg1 and arg1.lower().startswith("sqlite_"):
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and arg2 and arg2.lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def run_query(tables_db: Path, sql: str, *, row_limit: int = 200, time_limit_ms: int = 2000) -> dict[str, Any]:
    sql = sql.strip().rstrip(";").strip()
    if not sql:
        raise QueryRejected("Empty query.")
    conn = connect(Path(tables_db), readonly=True)
    deadline = time.monotonic() + time_limit_ms / 1000.0
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 2000)
    conn.set_authorizer(_authorizer)
    started = time.monotonic()
    try:
        cur = conn.execute(sql)
        if cur.description is None:
            raise QueryRejected("Only SELECT queries that return rows are allowed.")
        columns = [d[0] for d in cur.description]
        rows = cur.fetchmany(row_limit + 1)
    except sqlite3.ProgrammingError as exc:
        raise QueryRejected(f"{exc} Send a single SELECT statement.") from exc
    except sqlite3.DatabaseError as exc:
        msg = str(exc)
        if "interrupted" in msg:
            msg = f"Query stopped after {time_limit_ms} ms; narrow it with WHERE or aggregate."
        elif "not authorized" in msg:
            msg = "Not allowed: only read-only SELECT queries over the data tables and _catalog."
        raise QueryRejected(msg) from exc
    finally:
        conn.close()
    truncated = len(rows) > row_limit
    rows = rows[:row_limit]
    return {
        "sql": sql,
        "columns": columns,
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "truncated": truncated,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
    }


def rows_to_markdown(result: dict[str, Any]) -> str:
    from rag.core.util import markdown_table

    md = markdown_table(result["columns"], result["rows"])
    if result.get("truncated"):
        md += f"\n\n(first {result['row_count']} rows shown; add filters or aggregates for the rest)"
    return md
