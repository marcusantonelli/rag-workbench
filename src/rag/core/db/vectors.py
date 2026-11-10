"""Where a workspace keeps its vectors.

Two stores sit behind one small interface:

* ``sqlite-vec``: the ``chunk_vec`` table inside rag.db (float32, exact search). The default,
  and what the current retrieval server reads.
* ``shammy``: ``vectors.shammy`` next to rag.db (the project's own engine: about a fifth
  smaller with its float16 copies, searched in milliseconds at any size).

Which one a workspace uses is recorded in rag.db (meta ``vector_backend``); ``rag convert``
switches it. Chunk text, the keyword index and tables stay in SQLite either way.

With Shammy the vectors are a second file, so the two can disagree after a crash. They are
brought back in line whenever the workspace is opened for writing (``reconcile``): vectors
without a chunk are dropped, and documents whose vectors never reached the disk are marked to
be built again.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

BACKENDS = ("sqlite-vec", "shammy")
SHAMMY_FILE = "vectors.shammy"
# Vectors added since the last commit before a commit is forced. Each commit adds one segment.
_COMMIT_EVERY = 4096


class VectorStoreError(RuntimeError):
    pass


def shammy_module():
    try:
        import shammy
    except ImportError as exc:
        raise VectorStoreError(
            "this workspace keeps its vectors in Shammy, and the shammy module isn't installed "
            "(build it with shammy/build_wheel.sh, or switch back with `rag convert --to sqlite-vec` "
            "on a machine that has it)") from exc
    return shammy


def check_dims(dims: int) -> None:
    if dims < 64 or dims & (dims - 1):
        raise VectorStoreError(f"Shammy stores vectors whose size is a power of two (64 or more); "
                               f"this workspace uses {dims}. Use sqlite-vec, or a size such as 1024, 2048 or 4096.")


def _f32(vector: np.ndarray) -> bytes:
    return np.ascontiguousarray(vector, dtype=np.float32).tobytes()


class SqliteVectors:
    """The chunk_vec table. Writes join the caller's SQLite transaction."""

    name = "sqlite-vec"

    def __init__(self, conn):
        self.conn = conn

    def add(self, chunk_ids: Sequence[int], vectors, file_type: str, doc_id: int) -> None:
        self.conn.executemany("INSERT INTO chunk_vec(chunk_id, embedding, file_type, doc_id) VALUES (?,?,?,?)",
                              [(int(c), _f32(v), file_type, doc_id) for c, v in zip(chunk_ids, vectors)])

    def delete_doc(self, doc_id: int, chunk_ids: Sequence[int]) -> None:
        self.conn.executemany("DELETE FROM chunk_vec WHERE chunk_id = ?", [(int(i),) for i in chunk_ids])

    def search(self, query: np.ndarray, k: int, file_type: str | None = None) -> list[int]:
        sql = "SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ? AND k = ?"
        args: list[Any] = [_f32(query), k]
        if file_type:
            sql += " AND file_type = ?"
            args.append(file_type)
        return [r[0] for r in self.conn.execute(sql, args).fetchall()]

    def get(self, chunk_ids: Iterable[int]) -> dict[int, np.ndarray]:
        out = {}
        for cid in chunk_ids:
            row = self.conn.execute("SELECT embedding FROM chunk_vec WHERE chunk_id = ?", (int(cid),)).fetchone()
            if row is not None:
                out[int(cid)] = np.frombuffer(row[0], dtype=np.float32).copy()
        return out

    def ids(self) -> np.ndarray:
        return np.array([r[0] for r in self.conn.execute("SELECT chunk_id FROM chunk_vec")], dtype=np.uint64)

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0]

    def clear(self) -> None:
        self.conn.execute("DELETE FROM chunk_vec")

    def failed(self, doc_id: int) -> None:   # the SQLite rollback already undid the vectors
        pass

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


class ShammyVectors:
    """vectors.shammy. Adds and deletes are held in memory and written in batches (each batch is one
    commit and one segment of the file), on ``flush`` and on ``close``."""

    name = "shammy"

    def __init__(self, path: Path, *, readonly: bool):
        self.path = Path(path)
        self.readonly = readonly
        self._index = None
        self._adds: list[tuple[list[int], np.ndarray, str, int]] = []
        self._deletes: set[int] = set()
        self._pending = 0

    @classmethod
    def create(cls, path: Path, dims: int, *, model: str = "", query_instruction: str = "") -> None:
        check_dims(dims)
        shammy_module().Index.create(path, dims, model=model, query_instruction=query_instruction).close()

    @property
    def index(self):
        if self._index is None:
            sh = shammy_module()
            if not self.path.exists():
                raise VectorStoreError(f"{self.path.name} is missing from the workspace (run `rag verify`)")
            try:
                self._index = sh.Index.open(self.path, write=not self.readonly)
            except sh.Locked as exc:
                raise VectorStoreError(f"{exc}. Wait for the other build or merge to finish.") from exc
        return self._index

    def add(self, chunk_ids: Sequence[int], vectors, file_type: str, doc_id: int) -> None:
        if len(chunk_ids) == 0:
            return
        self._adds.append(([int(c) for c in chunk_ids], np.array(vectors, dtype=np.float32), file_type, int(doc_id)))
        self._pending += len(chunk_ids)

    def delete_doc(self, doc_id: int, chunk_ids: Sequence[int] = ()) -> None:
        # rows of this document that are already in the file go at the next flush; rows still
        # waiting in memory go now (a document id can be reused by the next document)
        self._adds = [a for a in self._adds if a[3] != int(doc_id)]
        self._deletes.add(int(doc_id))

    failed = delete_doc   # the SQLite transaction for a document rolled back: take its vectors out

    def after_write(self) -> None:
        if self._pending >= _COMMIT_EVERY:
            self.flush()

    def flush(self) -> None:
        if self.readonly or not (self._adds or self._deletes):
            return
        if self._deletes:
            self.index.delete_docs(sorted(self._deletes))
        for ids, vecs, file_type, doc_id in self._adds:
            self.index.add(ids, vecs, doc_ids=doc_id, file_types=file_type)
        self.index.commit()
        self._adds, self._deletes, self._pending = [], set(), 0

    def search(self, query: np.ndarray, k: int, file_type: str | None = None) -> list[int]:
        self.flush()
        if len(self.index) == 0:
            return []
        return [h.id for h in self.index.search(query, k, file_types=[file_type] if file_type else None)]

    def get(self, chunk_ids: Iterable[int]) -> dict[int, np.ndarray]:
        want = {int(c) for c in chunk_ids}
        out: dict[int, np.ndarray] = {}
        for ids, vecs, _, _ in self._adds:
            for i, cid in enumerate(ids):
                if cid in want:
                    out[cid] = vecs[i].copy()
        rest = sorted(want - out.keys())
        if rest:
            for cid, v in self.index.vectors(rest).items():
                out[cid] = v.copy()
        return out

    def ids(self) -> np.ndarray:
        self.flush()
        return self.index.ids()[0]

    def count(self) -> int:
        self.flush()
        return len(self.index)

    def clear(self) -> None:
        self._adds, self._pending = [], 0
        docs = {int(d) for d in self.index.ids()[1]}
        if docs:
            self.index.delete_docs(sorted(docs))
            self.index.commit()

    def close(self) -> None:
        if self._index is not None:
            self._index.close()
            self._index = None

    def info(self) -> dict[str, Any]:
        self.flush()
        return self.index.info()

    def compact(self, *, force: bool = False) -> dict[str, Any] | None:
        """Rewrite the file without deleted rows when it has many (or many small segments)."""
        info = self.info()
        dead = info["rows"] - info["live_rows"]
        if not force and info["segments"] <= 64 and dead * 4 <= max(info["rows"], 1):
            return None
        sh = shammy_module()
        self.close()
        tmp = self.path.with_name(self.path.name + ".compact")
        tmp.unlink(missing_ok=True)
        try:
            res = sh.compact(str(self.path), str(tmp))
            try:
                tmp.replace(self.path)
            except PermissionError as exc:   # Windows: someone still has the old file open
                raise VectorStoreError(f"can't replace {self.path.name} while a search has it open; "
                                       "close other windows using this workspace and try again") from exc
        finally:
            tmp.unlink(missing_ok=True)
        return res


def reconcile(store) -> dict[str, int]:
    """Bring vectors.shammy and rag.db back in line after an interrupted write. Returns what was fixed."""
    vectors: ShammyVectors = store.vectors
    ids, docs = vectors.index.ids()
    chunk_rows = store.conn.execute("SELECT chunk_id, doc_id FROM chunks").fetchall()
    chunk_ids = np.fromiter((r[0] for r in chunk_rows), dtype=np.uint64, count=len(chunk_rows))
    fixed = {"orphan_vectors": 0, "documents_to_rebuild": 0}
    stray = ~np.isin(ids, chunk_ids)
    if stray.any():
        fixed["orphan_vectors"] = int(stray.sum())
        vectors.index.delete_docs(sorted({int(d) for d in docs[stray]}))
        vectors.flush()
        ids = vectors.index.ids()[0]
    missing = ~np.isin(chunk_ids, ids)
    if missing.any():
        lost = sorted({int(chunk_rows[i][1]) for i in np.nonzero(missing)[0]})
        fixed["documents_to_rebuild"] = len(lost)
        with store.conn:
            for doc_id in lost:   # 'pending' makes the next build read the file again
                store.conn.execute("UPDATE documents SET status = 'pending' WHERE doc_id = ?", (doc_id,))
        vectors.index.delete_docs(lost)
        vectors.flush()
    return fixed
