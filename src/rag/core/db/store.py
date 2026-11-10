"""rag.db access: per-document transactional writes and deletes.

A document's row, its chunks, their FTS5 rows and their vectors are always
written or removed together in one transaction, so the keyword index and the
vector index can never disagree with the chunks table.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rag import SCHEMA_VERSION
from rag.core.db.connect import connect, read_schema
from rag.core.types import Chunk


class EmbeddingMismatch(RuntimeError):
    pass


EMBED_META_KEYS = ("embed_backend", "embed_model", "embed_dims", "query_instruction")


@dataclass
class DocRecord:
    source_key: str
    display_path: str
    sha256: str
    file_type: str
    title: str | None = None
    library_path: str | None = None
    n_pages: int | None = None
    size_bytes: int | None = None
    mtime: float | None = None
    extractor: str | None = None
    status: str = "ok"
    warnings: list[str] | None = None


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def vec_bytes(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype=np.float32).tobytes()


class RagStore:
    def __init__(self, path: Path, *, readonly: bool = False):
        self.path = Path(path)
        self.readonly = readonly
        self.conn = connect(self.path, readonly=readonly, load_vec=True)
        self._vectors = None

    # -- lifecycle ---------------------------------------------------------
    @classmethod
    def create(cls, path: Path, dims: int, *, vectors: str = "sqlite-vec") -> "RagStore":
        """Create the database (or open the one already there, which keeps its vector store)."""
        from rag.core.db.vectors import BACKENDS, SHAMMY_FILE, ShammyVectors, check_dims

        if vectors not in BACKENDS:
            raise ValueError(f"unknown vector store {vectors!r}; choose from {', '.join(BACKENDS)}")
        if vectors == "shammy" and not Path(path).exists():
            check_dims(int(dims))
        store = cls(path)
        store.conn.executescript(read_schema("schema_rag.sql").replace("{dims}", str(int(dims))))
        if store.get_meta("schema_version") is None:
            store.set_meta("schema_version", str(SCHEMA_VERSION))
            store.set_meta("created_at", now_iso())
            store.set_meta("vector_backend", vectors)
            if vectors == "shammy":
                (store.path.with_name(SHAMMY_FILE)).unlink(missing_ok=True)
                ShammyVectors.create(store.path.with_name(SHAMMY_FILE), int(dims))
        store.conn.commit()
        return store

    @property
    def vector_backend(self) -> str:
        return self.get_meta("vector_backend") or "sqlite-vec"

    @property
    def vectors(self):
        """The workspace's vector store (see rag.core.db.vectors)."""
        if self._vectors is None:
            from rag.core.db.vectors import SHAMMY_FILE, ShammyVectors, SqliteVectors, reconcile

            if self.vector_backend == "shammy":
                self._vectors = ShammyVectors(self.path.with_name(SHAMMY_FILE), readonly=self.readonly)
                if not self.readonly:
                    self.repaired = reconcile(self)
            else:
                self._vectors = SqliteVectors(self.conn)
        return self._vectors

    def close(self) -> None:
        try:
            if self._vectors is not None:
                if not self.readonly:
                    self._vectors.flush()
                self._vectors.close()
                self._vectors = None
        finally:
            self.conn.close()

    # -- meta ----------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def meta(self) -> dict[str, str]:
        return {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta")}

    def ensure_embedding(self, info: dict[str, str], *, adopt: bool) -> None:
        """Record the embedding settings, or refuse to mix with different ones."""
        current = {k: self.get_meta(k) for k in EMBED_META_KEYS}
        if all(v is None for v in current.values()) or adopt:
            for key in EMBED_META_KEYS:
                self.set_meta(key, str(info[key]))
            self.conn.commit()
            return
        diffs = {k: (current[k], str(info[k])) for k in EMBED_META_KEYS if current[k] != str(info[k])}
        if diffs:
            detail = "; ".join(f"{k}: pack={a!r} config={b!r}" for k, (a, b) in diffs.items())
            raise EmbeddingMismatch(
                "This pack was embedded with different settings (" + detail + "). "
                "Rebuild everything with `rag build --force`, or restore the old config."
            )

    # -- documents -----------------------------------------------------------
    def find_document(self, source_key: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM documents WHERE source_key = ?", (source_key,)
        ).fetchone()

    def documents(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM documents ORDER BY display_path").fetchall()

    def _delete_document_rows(self, doc_id: int) -> None:
        ids = [r[0] for r in self.conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,))]
        if ids:
            self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", [(i,) for i in ids])
            self.vectors.delete_doc(doc_id, ids)
        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

    def delete_document(self, doc_id: int) -> None:
        with self.conn:
            self._delete_document_rows(doc_id)

    def reserve_document(self, rec: DocRecord) -> int:
        """Replace any previous version of this source with a fresh, empty row.

        The doc_id is needed before chunks exist (tables.db rows reference it).
        """
        with self.conn:
            old = self.find_document(rec.source_key)
            if old is not None:
                self._delete_document_rows(old["doc_id"])
            cur = self.conn.execute(
                "INSERT INTO documents(source_key, display_path, library_path, sha256, file_type, "
                "title, n_pages, size_bytes, mtime, extractor, status, warnings, n_chunks, ingested_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?)",
                (
                    rec.source_key, rec.display_path, rec.library_path, rec.sha256, rec.file_type,
                    rec.title, rec.n_pages, rec.size_bytes, rec.mtime, rec.extractor,
                    "pending", json.dumps(rec.warnings or []), now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def write_chunks(
        self,
        doc_id: int,
        rec: DocRecord,
        chunks: list[Chunk],
        vectors: np.ndarray,
    ) -> list[int]:
        """Insert chunks, FTS rows and vectors, and mark the document finished.

        Returns the chunk ids in order.
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        ids: list[int] = []
        store = self.vectors
        try:
            self._write_chunks(doc_id, rec, chunks, vectors, ids)
        except BaseException:
            store.failed(doc_id)
            raise
        if hasattr(store, "after_write"):
            store.after_write()
        return ids

    def _write_chunks(self, doc_id: int, rec: DocRecord, chunks: list[Chunk], vectors: np.ndarray, ids: list[int]) -> None:
        with self.conn:
            for ordinal, (ch, _vec) in enumerate(zip(chunks, vectors)):
                cur = self.conn.execute(
                    "INSERT INTO chunks(doc_id, ordinal, kind, text, embed_text, text_hash, heading_path, "
                    "symbol, page_start, page_end, line_start, line_end, sheet, cell_range, table_name, "
                    "ocr_conf, asset_path, n_tokens) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        doc_id, ordinal, ch.kind, ch.text, ch.embed_text, ch.text_hash, ch.heading_path,
                        ch.symbol, ch.page_start, ch.page_end, ch.line_start, ch.line_end, ch.sheet,
                        ch.cell_range, ch.table_name, ch.ocr_conf, ch.asset_path, ch.n_tokens,
                    ),
                )
                chunk_id = int(cur.lastrowid)
                ids.append(chunk_id)
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid, embed_text) VALUES (?, ?)", (chunk_id, ch.embed_text)
                )
            self.vectors.add(ids, vectors, rec.file_type, doc_id)
            self.conn.execute(
                "UPDATE documents SET status = ?, n_chunks = ?, title = ?, n_pages = ?, extractor = ?, "
                "warnings = ? WHERE doc_id = ?",
                (
                    rec.status, len(chunks), rec.title, rec.n_pages, rec.extractor,
                    json.dumps(rec.warnings or []), doc_id,
                ),
            )

    def mark_failed(self, doc_id: int, message: str) -> None:
        with self.conn:
            self.conn.execute(
                "UPDATE documents SET status = 'failed', warnings = ? WHERE doc_id = ?",
                (json.dumps([message]), doc_id),
            )

    # -- vectors ---------------------------------------------------------------
    def vectors_for_hashes(self, hashes: Iterable[str]) -> dict[str, np.ndarray]:
        """Existing vectors for identical text, so it is embedded only once."""
        if self.vector_backend == "sqlite-vec":
            out: dict[str, np.ndarray] = {}
            for h in set(hashes):
                row = self.conn.execute(
                    "SELECT v.embedding FROM chunks c JOIN chunk_vec v ON v.chunk_id = c.chunk_id "
                    "WHERE c.text_hash = ? LIMIT 1",
                    (h,),
                ).fetchone()
                if row is not None:
                    out[h] = np.frombuffer(row[0], dtype=np.float32).copy()
            return out
        by_chunk: dict[int, str] = {}
        for h in set(hashes):
            row = self.conn.execute("SELECT chunk_id FROM chunks WHERE text_hash = ? LIMIT 1", (h,)).fetchone()
            if row is not None:
                by_chunk[int(row[0])] = h
        return {by_chunk[cid]: vec for cid, vec in self.vectors.get(by_chunk).items()}

    # -- errors and stats --------------------------------------------------------
    def log_error(self, path: str, stage: str, message: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO ingest_errors(path, stage, message, at) VALUES (?,?,?,?)",
                (path, stage, message[:4000], now_iso()),
            )

    def counts(self) -> dict[str, Any]:
        q = self.conn.execute
        by_type = {r[0]: r[1] for r in q("SELECT file_type, COUNT(*) FROM documents GROUP BY file_type")}
        by_kind = {r[0]: r[1] for r in q("SELECT kind, COUNT(*) FROM chunks GROUP BY kind")}
        return {
            "documents": q("SELECT COUNT(*) FROM documents").fetchone()[0],
            "failed": q("SELECT COUNT(*) FROM documents WHERE status = 'failed'").fetchone()[0],
            "chunks": q("SELECT COUNT(*) FROM chunks").fetchone()[0],
            "vectors": self.vectors.count(),
            "vector_store": self.vector_backend,
            "documents_by_type": by_type,
            "chunks_by_kind": by_kind,
        }
