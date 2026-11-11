"""The spot file: a knowledge base's structure, beside its rag.db (``structure.spot``).

It records which chunks make up a procedure and its steps, the warnings and notes in each step,
the references between steps and documents ("see Section 8.4", "see MF164"), and each document's
number and revision. Search uses it to answer with a whole procedure, or the steps around the one
that matched, instead of a lone fragment.

It is built entirely from the stored chunks (no reading, no embedding), so it can be made for a
knowledge base built by any earlier version, thrown away and rebuilt at any time, and a knowledge
base without one searches exactly as before. It is a plain SQLite file, so the retrieval server
reads it too.

A spot file is only used while it matches its rag.db (the ``chunks`` fingerprint): after any
build it is stale until rebuilt, and a stale one is ignored rather than trusted.

Detection is conservative. A group of numbered headings counts as a procedure only when its title
or its text reads as instructions; numbered lists ("1. Open …") and flowcharts count as procedures
inside one section. Everything else (manual chapters, web pages) stays plain sections, for which
search behaves as it always has.
"""

from __future__ import annotations

import os
import re
import sqlite3
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.core.structure import (_FILE_NUMBER, CAUTION, DOC_REF, LIST_STEP, NOTE, SECTION_REF, heading_number, identity,
                                imperative_share, is_prefix, label, procedural_title)

SPOT_NAME = "structure.spot"
FORMAT = "spot/1"

SCHEMA = """
CREATE TABLE spot_meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE documents (doc_id INTEGER PRIMARY KEY, title TEXT, number TEXT, revision TEXT);
-- kind: numbered (numbered headings), list (a numbered list in one section), flowchart
CREATE TABLE procedures (proc_id INTEGER PRIMARY KEY, doc_id INTEGER NOT NULL, kind TEXT NOT NULL,
                         label TEXT, title TEXT, steps INTEGER NOT NULL, tokens INTEGER NOT NULL);
-- step 0 is the procedure's own text before its first step
CREATE TABLE steps (proc_id INTEGER NOT NULL, step INTEGER NOT NULL, label TEXT, title TEXT,
                    tokens INTEGER NOT NULL, PRIMARY KEY (proc_id, step));
-- role: text | caution | note | figure | table
CREATE TABLE members (chunk_id INTEGER PRIMARY KEY, proc_id INTEGER NOT NULL, step INTEGER NOT NULL,
                      ordinal INTEGER NOT NULL, role TEXT NOT NULL);
CREATE INDEX members_step ON members (proc_id, step, ordinal);
-- kind: section (a step or procedure in the same document) | document (another document)
CREATE TABLE links (chunk_id INTEGER NOT NULL, kind TEXT NOT NULL, ref TEXT NOT NULL, proc_id INTEGER,
                    step INTEGER, doc_id INTEGER);
CREATE INDEX links_chunk ON links (chunk_id);
"""


def spot_path(root: Path) -> Path:
    """Where a knowledge base's spot file lives (root is the folder holding rag.db)."""
    return Path(root) / SPOT_NAME


def fingerprint(conn: sqlite3.Connection) -> str:
    """Which chunks a rag.db holds, cheaply: a spot file is only valid for exactly these."""
    n, hi, total = conn.execute(
        "SELECT COUNT(*), COALESCE(MAX(chunk_id), 0), COALESCE(SUM(chunk_id), 0) FROM chunks").fetchone()
    return f"{n}:{hi}:{total}"


@dataclass
class _Chunk:
    chunk_id: int
    ordinal: int
    kind: str
    text: str
    heading_path: str | None
    tokens: int
    parts: list[str] = field(default_factory=list)
    numbers: list[tuple[int, ...] | None] = field(default_factory=list)


@dataclass
class _Proc:
    doc_id: int
    kind: str
    label: str | None
    title: str | None
    key: tuple[int, ...] | None = None          # a numbered procedure's own number
    steps: dict[Any, dict[str, Any]] = field(default_factory=dict)   # key -> {label, title, chunks}


def _role(c: _Chunk) -> str:
    if c.kind == "figure":
        return "figure"
    if c.kind == "table":
        return "table"
    if CAUTION.search(c.text):
        return "caution"
    if NOTE.match(c.text):
        return "note"
    return "text"


def _numbered(doc_id: int, chunks: list[_Chunk]) -> tuple[list[_Proc], set[int]]:
    """Procedures made of numbered headings: a number with two or more numbered children is a
    procedure, each child a step (deeper numbers belong to their step)."""
    titles: dict[tuple[int, ...], str] = {}
    present: set[tuple[int, ...]] = set()
    for c in chunks:
        for part, num in zip(c.parts, c.numbers):
            if num is not None:
                titles.setdefault(num, part)
                for k in range(1, len(num) + 1):
                    present.add(num[:k])
    children: dict[tuple[int, ...], set[tuple[int, ...]]] = defaultdict(set)
    for n in present:
        if len(n) > 1:
            children[n[:-1]].add(n)

    groups: dict[tuple[int, ...], _Proc] = {}
    assigned: set[int] = set()
    for c in chunks:
        deepest = next((n for n in reversed(c.numbers) if n is not None), None)
        if deepest is None:
            continue
        parent = next((deepest[:k] for k in range(1, len(deepest) + 1) if len(children.get(deepest[:k], ())) >= 2), None)
        if parent is None:
            continue
        proc = groups.get(parent)
        if proc is None:
            proc = groups[parent] = _Proc(doc_id, "numbered", label(parent), titles.get(parent), parent)
        key = parent if deepest == parent else deepest[:len(parent) + 1]
        step = proc.steps.setdefault(key, {"label": label(key), "title": titles.get(key), "chunks": []})
        step["chunks"].append(c)
        assigned.add(c.chunk_id)

    out: list[_Proc] = []
    for parent, proc in groups.items():
        members = [c for s in proc.steps.values() for c in s["chunks"]]
        text = "\n".join(c.text for c in members if c.kind == "prose")
        real_steps = [k for k in proc.steps if k != parent]
        if len(real_steps) >= 2 and (procedural_title(proc.title) or any(procedural_title(s["title"]) for s in proc.steps.values())
                                     or imperative_share(text) >= 0.25):
            out.append(proc)
        else:
            assigned.difference_update(c.chunk_id for c in members)
    return out, assigned


def _sections(doc_id: int, chunks: list[_Chunk], assigned: set[int]) -> list[_Proc]:
    """Procedures inside one section: a numbered list (1., 2., 3. …) or a flowchart."""
    runs: list[list[_Chunk]] = []
    for c in chunks:
        if c.chunk_id in assigned:
            continue
        if runs and runs[-1][-1].heading_path == c.heading_path and runs[-1][-1].ordinal == c.ordinal - 1:
            runs[-1].append(c)
        else:
            runs.append([c])
    out: list[_Proc] = []
    for run in runs:
        title = run[0].parts[-1] if run[0].parts else None
        flow = [c for c in run if c.kind == "figure" and c.text.startswith("**Flowchart**")]
        if len(flow) >= 1 and len(flow) == len([c for c in run if c.kind == "figure"]) and flow:
            proc = _Proc(doc_id, "flowchart", None, title)
            for i, c in enumerate(run, start=1):
                page = re.search(r"\(page (\d+)\)", c.text[:80])
                proc.steps[i] = {"label": f"p.{page.group(1)}" if page and c in flow else None, "title": None, "chunks": [c]}
            if len(flow) >= 2 or len(run) >= 2:
                out.append(proc)
                continue
        prose = "\n".join(c.text for c in run if c.kind == "prose")
        nums = [int(n) for n in LIST_STEP.findall(prose)]
        if {1, 2, 3} <= set(nums) and imperative_share(prose) >= 0.2:
            proc = _Proc(doc_id, "list", None, title)
            for i, c in enumerate(run, start=1):
                found = LIST_STEP.search(c.text)
                proc.steps[i] = {"label": found.group(1) if found and c.kind == "prose" else None, "title": None, "chunks": [c]}
            out.append(proc)
    return out


def _norm_number(s: str) -> str:
    return re.sub(r"[\s_-]", "", s).upper()


def build(rag_db: Path, out: Path | None = None) -> dict[str, int]:
    """Build (or rebuild) the spot file for the rag.db at ``rag_db``. Returns counts."""
    rag_db = Path(rag_db)
    out = Path(out) if out else spot_path(rag_db.parent)
    src = sqlite3.connect(f"file:{rag_db.as_posix()}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row
    try:
        fp = fingerprint(src)
        docs = src.execute("SELECT doc_id, display_path, title FROM documents ORDER BY doc_id").fetchall()
        per_doc: dict[int, list[_Chunk]] = defaultdict(list)
        for r in src.execute("SELECT chunk_id, doc_id, ordinal, kind, text, heading_path, n_tokens FROM chunks "
                             "ORDER BY doc_id, ordinal"):
            c = _Chunk(r["chunk_id"], r["ordinal"], r["kind"], r["text"] or "", r["heading_path"], int(r["n_tokens"] or 0))
            c.parts = [p.strip() for p in (c.heading_path or "").split(" › ") if p.strip()]
            c.numbers = [heading_number(p) for p in c.parts]
            per_doc[r["doc_id"]].append(c)
    finally:
        src.close()

    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    tmp.unlink(missing_ok=True)
    db = sqlite3.connect(tmp)
    counts = {"documents": 0, "procedures": 0, "steps": 0, "members": 0, "links": 0}
    try:
        db.executescript(SCHEMA)
        idents: dict[int, Any] = {}
        for d in docs:
            chunks = per_doc.get(d["doc_id"], [])
            ident = identity(d["title"], [c.text for c in chunks[:4]], d["display_path"] or "")
            idents[d["doc_id"]] = ident
            db.execute("INSERT INTO documents VALUES (?, ?, ?, ?)", (d["doc_id"], ident.title, ident.number, ident.revision))
            counts["documents"] += 1
        by_number: dict[str, int] = {}
        for doc_id, ident in idents.items():
            if ident.number:
                by_number.setdefault(_norm_number(ident.number), doc_id)
        for d in docs:
            name = Path((d["display_path"] or "").replace("\\", "/")).stem
            m = _FILE_NUMBER.match(name)
            if m:
                by_number.setdefault(_norm_number(m.group(1)), d["doc_id"])

        for doc_id, chunks in per_doc.items():
            procs, assigned = _numbered(doc_id, chunks)
            procs += _sections(doc_id, chunks, assigned)
            # where each step and procedure of this document is, for "see Section 8.4"
            where: dict[tuple[int, ...], tuple[int, int]] = {}
            member_of: dict[int, tuple[int, int]] = {}
            for proc in procs:
                if proc.kind == "numbered":
                    # the procedure's own text first (step 0), then its steps in number order
                    ordered = sorted(proc.steps.items(), key=lambda kv: (kv[0] != proc.key, kv[0]))
                else:
                    ordered = sorted(proc.steps.items(), key=lambda kv: min(c.ordinal for c in kv[1]["chunks"]))
                tokens = sum(c.tokens for _, s in ordered for c in s["chunks"])
                cur = db.execute("INSERT INTO procedures (doc_id, kind, label, title, steps, tokens) VALUES (?, ?, ?, ?, ?, ?)",
                                 (doc_id, proc.kind, proc.label, proc.title, 0, tokens))
                proc_id = cur.lastrowid
                n_steps = 0
                for key, step in ordered:
                    if proc.kind == "numbered" and key == proc.key:
                        number = 0
                    else:
                        n_steps += 1
                        number = n_steps
                    db.execute("INSERT INTO steps VALUES (?, ?, ?, ?, ?)",
                               (proc_id, number, step["label"], step["title"], sum(c.tokens for c in step["chunks"])))
                    counts["steps"] += 1
                    if isinstance(key, tuple):
                        where[key] = (proc_id, number)
                    for c in sorted(step["chunks"], key=lambda c: c.ordinal):
                        db.execute("INSERT INTO members VALUES (?, ?, ?, ?, ?)", (c.chunk_id, proc_id, number, c.ordinal, _role(c)))
                        member_of[c.chunk_id] = (proc_id, number)
                        counts["members"] += 1
                if proc.key is not None:
                    where.setdefault(proc.key, (proc_id, 0))
                db.execute("UPDATE procedures SET steps = ? WHERE proc_id = ?", (n_steps, proc_id))
                counts["procedures"] += 1

            for c in chunks:
                seen: set[tuple[str, str]] = set()
                for m in SECTION_REF.finditer(c.text):
                    num = heading_number(m.group(1) + " x")
                    target = where.get(num) if num else None
                    if target is None or target == member_of.get(c.chunk_id) or ("section", m.group(1)) in seen:
                        continue
                    seen.add(("section", m.group(1)))
                    db.execute("INSERT INTO links VALUES (?, 'section', ?, ?, ?, NULL)", (c.chunk_id, m.group(1), *target))
                    counts["links"] += 1
                for m in DOC_REF.finditer(c.text):
                    ref = m.group(1)
                    if ("document", ref) in seen:
                        continue
                    target_doc = by_number.get(_norm_number(ref))
                    if target_doc == doc_id:
                        continue
                    seen.add(("document", ref))
                    db.execute("INSERT INTO links VALUES (?, 'document', ?, NULL, NULL, ?)", (c.chunk_id, ref, target_doc))
                    counts["links"] += 1

        for k, v in {"format": FORMAT, "chunks": fp, "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     **{f"count_{k}": str(v) for k, v in counts.items()}}.items():
            db.execute("INSERT INTO spot_meta VALUES (?, ?)", (k, v))
        db.commit()
    finally:
        db.close()
    _replace(tmp, out)
    return counts


def _replace(tmp: Path, out: Path) -> None:
    """Move the new file into place. On Windows a search holding the old one open blocks this for
    a moment; after a few tries the new file is dropped (the old one stays stale, so it's ignored)."""
    for attempt in range(10):
        try:
            os.replace(tmp, out)
            return
        except PermissionError:
            if attempt == 9:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2)


def status(root: Path) -> dict[str, Any]:
    """Whether a knowledge base has a spot file and whether it matches its rag.db."""
    path = spot_path(root)
    if not path.is_file():
        return {"present": False, "current": False}
    try:
        spot = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        try:
            meta = dict(spot.execute("SELECT key, value FROM spot_meta").fetchall())
        finally:
            spot.close()
        rag = sqlite3.connect(f"file:{(Path(root) / 'rag.db').as_posix()}?mode=ro", uri=True)
        try:
            fp = fingerprint(rag)
        finally:
            rag.close()
    except sqlite3.Error as exc:
        return {"present": True, "current": False, "error": str(exc)}
    out: dict[str, Any] = {"present": True, "current": meta.get("format") == FORMAT and meta.get("chunks") == fp,
                           "built_at": meta.get("built_at")}
    for k in ("documents", "procedures", "steps", "links"):
        if f"count_{k}" in meta:
            out[k] = int(meta[f"count_{k}"])
    return out


# -- reading, for search ------------------------------------------------------------------------
def norm_number(s: str) -> str:
    """How document numbers are compared: case, spaces, dashes and underscores don't matter."""
    return re.sub(r"[\s_-]", "", s).upper()


class SpotIndex:
    """A current spot file opened for searching (the reference search's twin of the shared
    engine's reader)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    @classmethod
    def open(cls, root: Path, rag_conn: sqlite3.Connection) -> "SpotIndex | None":
        path = spot_path(root)
        if not path.is_file():
            return None
        try:
            conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True, check_same_thread=False)
            meta = dict(conn.execute("SELECT key, value FROM spot_meta").fetchall())
        except sqlite3.Error:
            return None
        if meta.get("format") != FORMAT or meta.get("chunks") != fingerprint(rag_conn):
            conn.close()
            return None
        return cls(conn)

    def close(self) -> None:
        self.conn.close()

    def member(self, chunk_id: int) -> tuple[int, int] | None:
        row = self.conn.execute("SELECT proc_id, step FROM members WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return (row[0], row[1]) if row else None

    def procedure(self, proc_id: int) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT doc_id, kind, label, title, steps, tokens FROM procedures WHERE proc_id = ?",
                                (proc_id,)).fetchone()
        if row is None:
            return None
        return {"doc_id": row[0], "kind": row[1], "label": row[2], "title": row[3], "steps": row[4], "tokens": row[5]}

    def steps(self, proc_id: int) -> list[dict[str, Any]]:
        return [{"step": r[0], "label": r[1], "title": r[2], "tokens": r[3]} for r in self.conn.execute(
            "SELECT step, label, title, tokens FROM steps WHERE proc_id = ? ORDER BY step", (proc_id,))]

    def members(self, proc_id: int, lo: int, hi: int) -> list[tuple[int, int]]:
        """(chunk_id, step) of steps lo..hi, in reading order."""
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT chunk_id, step FROM members WHERE proc_id = ? AND step BETWEEN ? AND ? ORDER BY step, ordinal",
            (proc_id, lo, hi))]

    def links(self, chunk_id: int) -> list[dict[str, Any]]:
        return [{"kind": r[0], "ref": r[1], "proc_id": r[2], "step": r[3], "doc_id": r[4]} for r in self.conn.execute(
            "SELECT kind, ref, proc_id, step, doc_id FROM links WHERE chunk_id = ? ORDER BY rowid", (chunk_id,))]

    def step_info(self, proc_id: int, step: int) -> tuple[str | None, str | None] | None:
        row = self.conn.execute("SELECT label, title FROM steps WHERE proc_id = ? AND step = ?", (proc_id, step)).fetchone()
        return (row[0], row[1]) if row else None

    def document(self, doc_id: int) -> tuple[str | None, str | None]:
        row = self.conn.execute("SELECT number, revision FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def numbers(self) -> list[tuple[int, str]]:
        return [(r[0], r[1]) for r in self.conn.execute(
            "SELECT doc_id, number FROM documents WHERE number IS NOT NULL ORDER BY doc_id")]
