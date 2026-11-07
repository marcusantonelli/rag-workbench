"""SQL files: one block per statement, routines kept whole, bulk INSERTs summarized.

The splitter understands strings, comments, BEGIN/CASE...END nesting, T-SQL
``GO`` batch separators and Oracle ``/`` terminators, so a stored procedure is
never cut at the semicolons inside its body. sqlglot (tolerant mode) supplies
the tables each statement references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block, ExtractedDoc
from rag.core.util import read_text_file

_GO_RE = re.compile(r"^\s*GO\s*(\d+)?\s*;?\s*$", re.I)
_SLASH_RE = re.compile(r"^\s*/\s*$")
_ROUTINE_RE = re.compile(
    r"^\s*(CREATE|ALTER)\s+(OR\s+(ALTER|REPLACE)\s+)?(DEFINER\s*=\s*\S+\s+)?"
    r"(PROC|PROCEDURE|FUNCTION|TRIGGER|PACKAGE(\s+BODY)?|TYPE\s+BODY)\b",
    re.I,
)
_HEAD_RE = re.compile(
    r"^\s*(?P<verb>CREATE|ALTER|DROP|INSERT|UPDATE|DELETE|SELECT|WITH|MERGE|EXEC|EXECUTE|"
    r"GRANT|REVOKE|DECLARE|SET|USE|TRUNCATE|BEGIN|COMMIT|ROLLBACK|CALL|COMMENT)\b",
    re.I,
)
_OBJECT_RE = re.compile(
    r"^\s*(CREATE|ALTER|DROP)\s+(OR\s+(ALTER|REPLACE)\s+)?(UNIQUE\s+|CLUSTERED\s+|NONCLUSTERED\s+|"
    r"GLOBAL\s+TEMPORARY\s+|TEMP(ORARY)?\s+|MATERIALIZED\s+)*"
    r"(?P<kind>TABLE|VIEW|PROC|PROCEDURE|FUNCTION|TRIGGER|INDEX|SEQUENCE|SCHEMA|TYPE|PACKAGE(\s+BODY)?|SYNONYM)"
    r"\s+(IF\s+(NOT\s+)?EXISTS\s+)?(?P<name>[\w\[\]\"`.$#@]+)",
    re.I,
)
_INSERT_RE = re.compile(r"^\s*INSERT\s+(INTO\s+)?(?P<name>[\w\[\]\"`.$#@]+)", re.I)


@dataclass
class _Stmt:
    text: str
    line_start: int
    line_end: int
    meta: dict = field(default_factory=dict)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def split_statements(sql: str) -> list[_Stmt]:
    stmts: list[_Stmt] = []
    lines = sql.split("\n")
    buf: list[str] = []
    buf_start = 1
    depth = 0
    body_closed = False
    in_block_comment = False

    def flush(end_line: int) -> None:
        nonlocal buf, depth, body_closed
        text = "\n".join(buf).strip()
        if _strip_comments(text).strip():
            stmts.append(_Stmt(text=text, line_start=buf_start, line_end=end_line))
        buf = []
        depth = 0
        body_closed = False

    for lineno, line in enumerate(lines, start=1):
        if not in_block_comment and (_GO_RE.match(line) or (_SLASH_RE.match(line) and buf)):
            flush(lineno - 1)
            buf_start = lineno + 1
            continue
        if not buf:
            if not line.strip():
                buf_start = lineno + 1
                continue
            buf_start = lineno
        # Scan the line for strings, comments, keywords and semicolons.
        i = 0
        split_at: list[int] = []
        while i < len(line):
            ch = line[i]
            if in_block_comment:
                end = line.find("*/", i)
                if end < 0:
                    i = len(line)
                    break
                i = end + 2
                in_block_comment = False
                continue
            if line.startswith("--", i):
                break
            if line.startswith("/*", i):
                in_block_comment = True
                i += 2
                continue
            if ch in ("'", '"', "["):
                close = {"'": "'", '"': '"', "[": "]"}[ch]
                j = i + 1
                while j < len(line):
                    if line[j] == close:
                        if close != "]" and j + 1 < len(line) and line[j + 1] == close:
                            j += 2
                            continue
                        break
                    j += 1
                i = j + 1
                continue
            if ch.isalpha() or ch == "_":
                j = i
                while j < len(line) and (line[j].isalnum() or line[j] in "_$#@"):
                    j += 1
                word = line[i:j].upper()
                rest = line[j:].lstrip().upper()
                if word == "BEGIN" and not rest.startswith(("TRAN", "TRANSACTION", ";", "DISTRIBUTED")):
                    depth += 1
                elif word == "CASE":
                    depth += 1
                elif word == "END" and not rest.startswith(("IF", "LOOP", "WHILE")):
                    depth = max(0, depth - 1)
                    if depth == 0:
                        body_closed = True
                i = j
                continue
            if ch == ";" and depth == 0:
                current = "\n".join(buf + [line[: i + 1]])
                if not _ROUTINE_RE.match(_strip_comments(current)) or body_closed:
                    split_at.append(i)
            i += 1
        if split_at:
            start = 0
            for pos in split_at:
                buf.append(line[start: pos + 1])
                flush(lineno)
                buf_start = lineno
                start = pos + 1
            remainder = line[start:]
            if remainder.strip():
                buf.append(remainder)
                buf_start = lineno
            else:
                buf_start = lineno + 1
        else:
            buf.append(line)
    flush(len(lines))
    return stmts


def _referenced_tables(text: str, dialect: str | None) -> list[str]:
    try:
        import sqlglot
        from sqlglot import exp
        from sqlglot.errors import ErrorLevel

        names: list[str] = []
        for dial in ([dialect] if dialect else []) + ["tsql", None]:
            try:
                trees = sqlglot.parse(text, read=dial, error_level=ErrorLevel.IGNORE)
            except Exception:  # noqa: BLE001
                continue
            for tree in trees:
                if tree is None:
                    continue
                for t in tree.find_all(exp.Table):
                    name = ".".join(p for p in (t.catalog, t.db, t.name) if p)
                    if name and name not in names:
                        names.append(name)
            if names:
                break
        return names[:30]
    except ImportError:
        return []


def _describe(stmt: _Stmt, dialect: str | None) -> None:
    clean = _strip_comments(stmt.text).strip()
    head = _HEAD_RE.match(clean)
    stmt.meta["statement_type"] = head.group("verb").upper() if head else "OTHER"
    obj = _OBJECT_RE.match(clean)
    if obj:
        stmt.meta["object_kind"] = re.sub(r"\s+", " ", obj.group("kind").upper())
        stmt.meta["object"] = obj.group("name").strip('[]"`')
    ins = _INSERT_RE.match(clean)
    if ins:
        stmt.meta["object"] = ins.group("name").strip('[]"`')
    stmt.meta["tables"] = _referenced_tables(clean, dialect)


def _summarize_inserts(group: list[_Stmt]) -> _Stmt:
    target = group[0].meta.get("object", "?")
    first = group[0].text
    head = first if len(first) <= 600 else first[:600] + " ..."
    text = (
        f"-- {len(group)} INSERT statements into {target} (lines {group[0].line_start}-{group[-1].line_end}); "
        f"first one shown\n{head}"
    )
    return _Stmt(text=text, line_start=group[0].line_start, line_end=group[-1].line_end,
                 meta={"statement_type": "INSERT", "object": target, "tables": [target], "bulk": len(group)})


def extract_sql(path: Path, ctx: ExtractContext) -> ExtractedDoc:
    text, _ = read_text_file(path)
    dialect = ctx.config["ingest"].get("sql_dialect") or None
    stmts = split_statements(text)
    for s in stmts:
        _describe(s, dialect)
        values_rows = len(re.findall(r"\)\s*,\s*\(", s.text))
        if s.meta["statement_type"] == "INSERT" and values_rows > 20:
            s.text = s.text[:600] + f"\n-- ... {values_rows + 1} value rows in total"
            s.meta["bulk"] = values_rows + 1

    collapsed: list[_Stmt] = []
    run: list[_Stmt] = []
    for s in stmts + [None]:  # type: ignore[list-item]
        if s is not None and s.meta.get("statement_type") == "INSERT" and (
            not run or run[0].meta.get("object") == s.meta.get("object")
        ):
            run.append(s)
            continue
        if run:
            collapsed.extend([_summarize_inserts(run)] if len(run) >= 5 else run)
            run = []
        if s is not None:
            if s.meta.get("statement_type") == "INSERT":
                run.append(s)
            else:
                collapsed.append(s)

    blocks = []
    for s in collapsed:
        label = s.meta.get("object_kind") or s.meta.get("statement_type")
        blocks.append(Block(
            kind="sql",
            text=f"```sql\n{s.text}\n```",
            line_start=s.line_start,
            line_end=s.line_end,
            symbol=s.meta.get("object"),
            lang="sql",
            meta={**s.meta, "label": label, "signature": s.text.split("\n", 1)[0][:200]},
        ))
    return ExtractedDoc(file_type="sql", title=None, blocks=blocks, extractor="sql")
