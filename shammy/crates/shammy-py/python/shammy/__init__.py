"""Shammy: a single-file vector store for read-mostly knowledge bases.

    import numpy as np, shammy

    with shammy.Index.create("kb.shammy", dims=1024, model="my-embedder") as ix:
        ix.add(ids=[1, 2, 3], vectors=np.random.rand(3, 1024), doc_ids=[10, 10, 11], file_types="pdf")
        ix.commit()

    ix = shammy.Index.open("kb.shammy")                 # read-only; safe from any number of processes
    for hit in ix.search(query_vector, k=10, file_types=["pdf"]):
        print(hit.id, hit.doc_id, hit.score)

Vectors are compared by cosine similarity (they are stored at unit length). Stored dimensions must
be a power of two, at least 64; shorter vectors and queries are padded with zeros, which leaves
cosine similarity unchanged, so models with e.g. 768 or 1536 dimensions work in a 1024 or 2048 file.
One process writes at a time; readers see the file as of its last ``commit()``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

from . import _native
from ._native import __version__, compact, default_threads, fts_query, verify

__all__ = ["Index", "Hit", "Locked", "Retriever", "fts_query", "storage_dims", "verify", "compact", "default_threads", "__version__"]


class Hit(NamedTuple):
    id: int
    doc_id: int
    score: float


def storage_dims(dims: int) -> int:
    """The stored size for vectors of ``dims`` values: the next power of two, at least 64."""
    n = 64
    while n < dims:
        n *= 2
    return n


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":  # os.kill would terminate the process on Windows
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Locked(RuntimeError):
    """Another process is writing to this file."""


class _WriteLock:
    """``<file>.lock`` holding the writer's process id; a lock left by a dead process is taken over."""

    _held: set[str] = set()   # files this process is writing

    def __init__(self, path: Path):
        self.path = Path(str(path) + ".lock")
        self.key = str(Path(path).resolve())
        if self.key in self._held:
            raise Locked(f"{path} is already open for writing in this process")
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    pid = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    pid = 0
                if pid != os.getpid() and _pid_alive(pid):
                    raise Locked(f"{path} is being written by another process (pid {pid})") from None
                self.path.unlink(missing_ok=True)
                continue
            with os.fdopen(fd, "w") as f:
                f.write(str(os.getpid()))
            self._held.add(self.key)
            return
        raise Locked(f"can't take the write lock for {path}")

    def release(self) -> None:
        self._held.discard(self.key)
        self.path.unlink(missing_ok=True)


class Index:
    """One ``.shammy`` file, opened for reading or for reading and writing."""

    def __init__(self, path: str | os.PathLike, writer=None, lock: _WriteLock | None = None):
        self.path = Path(path)
        self._writer = writer
        self._lock = lock
        self._reader = None
        self._mutex = threading.Lock()
        self._dirty = False

    # -- opening -----------------------------------------------------------------
    @classmethod
    def create(cls, path: str | os.PathLike, dims: int, *, lossless: bool = True, model: str = "",
               query_instruction: str = "", sketch_dims: int | None = None) -> "Index":
        """A new file for vectors of ``dims`` values (stored at ``storage_dims(dims)``).

        ``lossless`` also keeps each vector as float16, which makes results exact and lets the
        vectors be read back; without it the file is about a third of the size.
        """
        lock = _WriteLock(Path(path))
        try:
            w = _native.Writer.create(str(path), storage_dims(int(dims)), lossless, sketch_dims, model, query_instruction)
            w.commit()
        except BaseException:
            lock.release()
            raise
        return cls(path, w, lock)

    @classmethod
    def open(cls, path: str | os.PathLike, *, write: bool = False) -> "Index":
        if not write:
            ix = cls(path)
            ix._read()  # fail now if the file is missing or damaged
            return ix
        lock = _WriteLock(Path(path))
        try:
            return cls(path, _native.Writer.open(str(path)), lock)
        except BaseException:
            lock.release()
            raise

    def __enter__(self) -> "Index":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001 - nothing useful to do while being collected
            pass

    def close(self) -> None:
        """Commit anything pending and release the file."""
        with self._mutex:
            try:
                if self._writer is not None:
                    self._writer.commit()
                    self._writer.close()
            finally:
                self._writer = None
                if self._reader is not None:
                    self._reader.close()
                    self._reader = None
                if self._lock is not None:
                    self._lock.release()
                    self._lock = None

    # -- writing -----------------------------------------------------------------
    def _need_writer(self):
        if self._writer is None:
            raise ValueError(f"{self.path} is open read-only; use Index.open(path, write=True)")
        return self._writer

    def _fit(self, vectors) -> np.ndarray:
        v = np.asarray(vectors, dtype="<f4")
        if v.ndim == 1:
            v = v[None, :]
        dims = self.dims
        if v.shape[1] < dims:
            v = np.pad(v, ((0, 0), (0, dims - v.shape[1])))
        return np.ascontiguousarray(v)

    def add(self, ids: Sequence[int], vectors, doc_ids: Sequence[int] | int = 0,
            file_types: Sequence[str] | str = "") -> int:
        """Add rows. ``doc_ids`` and ``file_types`` (one per row, or one for all) are what
        searches can filter on and what ``delete_docs`` removes by."""
        w = self._need_writer()
        v = self._fit(vectors)
        n = len(v)
        id_arr = np.asarray(ids, dtype="<u8").reshape(-1)
        doc_arr = np.full(n, doc_ids, dtype="<u4") if np.isscalar(doc_ids) else np.asarray(doc_ids, dtype="<u4")
        types = [file_types] if isinstance(file_types, str) else list(file_types)
        with self._mutex:
            w.add(id_arr.tobytes(), doc_arr.tobytes(), types, v.tobytes())
            self._dirty = True
        return n

    def delete_docs(self, doc_ids: Iterable[int]) -> int:
        """Remove every row of these documents. Returns the number of rows removed."""
        w = self._need_writer()
        with self._mutex:
            n = w.delete_docs([int(d) for d in doc_ids])
            self._dirty = self._dirty or n > 0
            return n

    def commit(self) -> None:
        """Make everything added or deleted so far durable and visible to searches."""
        w = self._need_writer()
        with self._mutex:
            w.commit()
            self._dirty = False
            self._reader = None

    # -- reading -----------------------------------------------------------------
    def _read(self):
        with self._mutex:
            if self._writer is not None and self._dirty:
                self._writer.commit()
                self._dirty = False
                self._reader = None
            if self._reader is None:
                self._reader = _native.Reader(str(self.path))
            return self._reader

    def refresh(self) -> None:
        """Pick up commits made by another process since this index was opened."""
        with self._mutex:
            self._reader = None

    def info(self) -> dict:
        return self._read().info()

    @property
    def dims(self) -> int:
        if getattr(self, "_dims", None) is None:
            self._dims = int(self._read().info()["dims"])
        return self._dims

    def __len__(self) -> int:
        return int(self.info()["live_rows"])

    def search(self, query, k: int = 10, *, file_types: Sequence[str] | None = None,
               doc_ids: Sequence[int] | None = None, exact: bool = False, threads: int = 0,
               stage1: int = 0, stage2: int = 0, finish: int = 0) -> list[Hit]:
        """The ``k`` most similar rows, best first. ``exact`` scores every row in full (slow;
        needs a lossless file). ``threads`` defaults to two fewer than the machine's cores."""
        q = self._fit(query)[0]
        hits = self._read().search(q.tobytes(), int(k), list(file_types) if file_types else None,
                                   [int(d) for d in doc_ids] if doc_ids else None, exact, threads, stage1, stage2, finish)
        return [Hit(*h) for h in hits]

    def vectors(self, ids: Sequence[int]) -> dict[int, np.ndarray]:
        """Stored vectors by id (unit length, float16 precision). Ids that aren't stored are left out."""
        ids = [int(i) for i in ids]
        found = self._read().vectors(ids)
        return {i: np.frombuffer(b, dtype="<f4") for i, b in zip(ids, found) if b is not None}

    def ids(self) -> tuple[np.ndarray, np.ndarray]:
        """(ids, doc ids) of every live row, in file order."""
        a, b = self._read().live()
        return np.frombuffer(a, dtype="<u8"), np.frombuffer(b, dtype="<u4")


class Retriever:
    """The whole retrieval path over a RAG Workbench workspace or unpacked pack folder: vector and
    keyword search, rank fusion, optional reranking, neighbour expansion and citations. The same
    code runs behind the C interface, so every caller gets the same results.

        r = shammy.Retriever("path/to/workspace")
        out = r.search("how is the lid seal tested?", query_vector, k=6,
                       rerank={"enabled": True, "url": "http://127.0.0.1:8081"})
        for hit in out["results"]: ...

    The caller embeds the question (with the model and instruction ``r.info()`` names).
    """

    def __init__(self, folder: str | os.PathLike):
        self._native = _native.Retriever(str(folder))

    def info(self) -> dict:
        import json

        return json.loads(self._native.info())

    def search(self, query: str, vector, **options) -> dict:
        """Options: k, file_type, expand, vec_k, fts_k, rrf_k, neighbors, max_result_tokens,
        threads, markdown, rerank={enabled, url, model, candidates, timeout_ms}."""
        import json

        request = {"query": query, **{k: v for k, v in options.items() if v is not None}}
        v = np.ascontiguousarray(np.asarray(vector, dtype="<f4").reshape(-1))
        return json.loads(self._native.search(json.dumps(request), v.tobytes()))

    def close(self) -> None:
        self._native.close()

    def __enter__(self) -> "Retriever":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
