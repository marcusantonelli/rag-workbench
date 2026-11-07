"""Crawl state in SQLite (crawl.db in the output folder), so a crawl can resume."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS urls (
  url       TEXT PRIMARY KEY,
  host      TEXT NOT NULL,
  depth     INTEGER NOT NULL,
  offsite   INTEGER NOT NULL DEFAULT 0,
  tunnel    INTEGER NOT NULL DEFAULT 0,
  priority  REAL NOT NULL,
  status    TEXT NOT NULL,     -- queued | active | done | saved | skipped | failed
  reason    TEXT,
  parent    TEXT,
  anchor    TEXT,
  attempts  INTEGER NOT NULL DEFAULT 0,
  updated   REAL
);
CREATE INDEX IF NOT EXISTS urls_status ON urls(status);
CREATE INDEX IF NOT EXISTS urls_host ON urls(host);

CREATE TABLE IF NOT EXISTS files (
  content_hash TEXT PRIMARY KEY,
  url          TEXT NOT NULL,
  path         TEXT NOT NULL,
  type         TEXT NOT NULL,
  bytes        INTEGER NOT NULL,
  relevance    REAL,
  title        TEXT,
  saved_at     REAL
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


class CrawlState:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.executescript(SCHEMA)
        # Anything in flight when the last run stopped goes back in the queue.
        self.conn.execute("UPDATE urls SET status = 'queued' WHERE status = 'active'")

    def close(self) -> None:
        with self.lock:
            self.conn.close()

    # -- urls --------------------------------------------------------------------
    def add_url(self, url: str, host: str, depth: int, offsite: int, tunnel: int, priority: float,
                parent: str | None, anchor: str | None) -> bool:
        with self.lock:
            cur = self.conn.execute(
                "INSERT OR IGNORE INTO urls(url, host, depth, offsite, tunnel, priority, status, parent, anchor, updated)"
                " VALUES (?,?,?,?,?,?,'queued',?,?,?)",
                (url, host, depth, offsite, tunnel, priority, parent, (anchor or "")[:200], time.time()))
            return cur.rowcount == 1

    def known(self, url: str) -> bool:
        with self.lock:
            return self.conn.execute("SELECT 1 FROM urls WHERE url = ?", (url,)).fetchone() is not None

    def set_status(self, url: str, status: str, reason: str | None = None, *, attempt: bool = False) -> None:
        with self.lock:
            self.conn.execute(
                "UPDATE urls SET status = ?, reason = ?, updated = ?, attempts = attempts + ? WHERE url = ?",
                (status, reason, time.time(), 1 if attempt else 0, url))

    def queued(self) -> Iterator[sqlite3.Row]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT url, host, depth, offsite, tunnel, priority, attempts FROM urls WHERE status = 'queued'"
            ).fetchall()
        return iter(rows)

    def get(self, url: str) -> sqlite3.Row | None:
        with self.lock:
            return self.conn.execute("SELECT * FROM urls WHERE url = ?", (url,)).fetchone()

    def host_counts(self) -> dict[str, int]:
        with self.lock:
            return {r[0]: r[1] for r in self.conn.execute(
                "SELECT host, COUNT(*) FROM urls WHERE status NOT IN ('queued') GROUP BY host")}

    # -- files -------------------------------------------------------------------
    def has_content(self, content_hash: str) -> bool:
        with self.lock:
            return self.conn.execute("SELECT 1 FROM files WHERE content_hash = ?", (content_hash,)).fetchone() is not None

    def add_file(self, content_hash: str, url: str, path: str, ftype: str, size: int, relevance: float,
                 title: str) -> None:
        with self.lock:
            self.conn.execute(
                "INSERT INTO files(content_hash, url, path, type, bytes, relevance, title, saved_at)"
                " VALUES (?,?,?,?,?,?,?,?)", (content_hash, url, path, ftype, size, relevance, title, time.time()))

    def saved_bytes(self) -> int:
        with self.lock:
            return int(self.conn.execute("SELECT COALESCE(SUM(bytes), 0) FROM files").fetchone()[0])

    def files(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.conn.execute("SELECT * FROM files ORDER BY saved_at")]

    def status_counts(self) -> dict[str, int]:
        with self.lock:
            return {r[0]: r[1] for r in self.conn.execute("SELECT status, COUNT(*) FROM urls GROUP BY status")}

    def skip_reasons(self, limit: int = 12) -> list[tuple[str, int]]:
        with self.lock:
            return [(r[0], r[1]) for r in self.conn.execute(
                "SELECT reason, COUNT(*) AS n FROM urls WHERE status IN ('skipped','failed') "
                "GROUP BY reason ORDER BY n DESC LIMIT ?", (limit,))]

    # -- meta --------------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self.lock:
            row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self.lock:
            self.conn.execute("INSERT INTO meta(key, value) VALUES (?, ?) "
                              "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))
