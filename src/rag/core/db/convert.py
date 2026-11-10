"""Move a workspace's vectors between the two stores (see rag.core.db.vectors).

Nothing is embedded again. Going to sqlite-vec from Shammy writes the float16 copies Shammy
keeps, which differ from the originals only past the fourth significant digit.
"""

from __future__ import annotations

import os
from itertools import groupby
from typing import Any, Callable

from rag.core.db.store import RagStore
from rag.core.db.vectors import BACKENDS, SHAMMY_FILE, ShammyVectors, SqliteVectors, VectorStoreError


def convert_workspace(ws, to: str, *, progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
    if to not in BACKENDS:
        raise VectorStoreError(f"unknown vector store {to!r}; choose from {', '.join(BACKENDS)}")
    store = RagStore(ws.rag_db)
    try:
        current = store.vector_backend
        result: dict[str, Any] = {"from": current, "to": to, "changed": current != to, "vectors": 0, "missing": 0}
        if current == to:
            result["vectors"] = store.vectors.count()
            return result
        dims = int(store.get_meta("embed_dims") or ws.config["embedding"]["dims"])
        rows = store.conn.execute(
            "SELECT c.chunk_id, c.doc_id, d.file_type FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "ORDER BY c.doc_id, c.chunk_id").fetchall()
        src = store.vectors
        shammy_path = ws.root / SHAMMY_FILE
        if to == "shammy":
            new_path = shammy_path.with_name(SHAMMY_FILE + ".new")
            new_path.unlink(missing_ok=True)
            ShammyVectors.create(new_path, dims, model=store.get_meta("embed_model") or "",
                                 query_instruction=store.get_meta("query_instruction") or "")
            dest: Any = ShammyVectors(new_path, readonly=False)
        else:
            dest = SqliteVectors(store.conn)
            dest.clear()
        done = 0
        try:
            for (doc_id, file_type), group in groupby(rows, key=lambda r: (r[1], r[2])):
                ids = [r[0] for r in group]
                found = src.get(ids)
                have = [i for i in ids if i in found]
                result["missing"] += len(ids) - len(have)
                if have:
                    dest.add(have, [found[i] for i in have], file_type, doc_id)
                    result["vectors"] += len(have)
                    if to == "shammy":
                        dest.after_write()
                done += len(ids)
                if progress:
                    progress(done, len(rows))
            if to == "shammy":
                dest.flush()
                dest.close()
                os.replace(new_path, shammy_path)
                SqliteVectors(store.conn).clear()
            store.set_meta("vector_backend", to)
            store.conn.commit()
        except BaseException:
            store.conn.rollback()
            if to == "shammy":
                dest.close()
                new_path.unlink(missing_ok=True)
            raise
        src.close()
        store._vectors = None
        if to == "sqlite-vec":
            shammy_path.unlink(missing_ok=True)
        store.conn.execute("VACUUM")   # give back the space the old vectors used in rag.db
        result["bytes"] = {"rag.db": ws.rag_db.stat().st_size,
                           SHAMMY_FILE: shammy_path.stat().st_size if shammy_path.exists() else 0}
        return result
    finally:
        store.close()


def compact_workspace(ws, *, force: bool = True) -> dict[str, Any] | None:
    """Rewrite vectors.shammy without deleted rows. None when the workspace uses sqlite-vec."""
    store = RagStore(ws.rag_db)
    try:
        if store.vector_backend != "shammy":
            return None
        return store.vectors.compact(force=force) or {}
    finally:
        store.close()
