"""Merge knowledge bases (packs or workspaces) into a workspace without re-embedding.

Sources are compatible when they were embedded the same way (model, vector size, query
instruction): their vectors live in the same space, so chunks, vectors, keyword index rows,
SQL tables, figures and original files are copied as they are.

Per source document:
  * not in the destination                        -> added
  * same file (path) with the same content        -> skipped
  * same file with different content              -> on_conflict: newer (default) | keep | replace
  * same content already present under another path -> skipped, unless keep_copies
Failed documents (nothing extracted) are not copied. Table names that collide are renamed
(t_x -> t_x_2) and the chunk text that mentions them is updated.
"""

from __future__ import annotations

import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rag.core.db.store import EMBED_META_KEYS, RagStore, now_iso
from rag.core.db.tables import TablesStore

ProgressFn = Callable[[dict[str, Any]], None]
CONFLICT_POLICIES = ("newer", "keep", "replace")


class MergeError(RuntimeError):
    pass


@dataclass
class SourcePlan:
    path: str
    name: str
    compatible: bool
    reason: str = ""
    embedding: dict[str, str] = field(default_factory=dict)
    documents: int = 0
    add: int = 0
    replace: int = 0
    identical: int = 0
    copies: int = 0
    kept: int = 0
    failed_skipped: int = 0
    chunks: int = 0
    tables: int = 0
    renamed_tables: list[tuple[str, str]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "renamed_tables": [list(t) for t in self.renamed_tables]}


def _embedding(meta: dict[str, str]) -> dict[str, str]:
    return {k: meta.get(k) for k in EMBED_META_KEYS if meta.get(k) is not None}


def _open_source(path: Path) -> tuple[Path, str, tempfile.TemporaryDirectory | None]:
    """Folder holding rag.db for a workspace or (unpacked) pack, a display name, and the temp dir."""
    import zipfile

    from rag.core.db.pack import unpack_pack

    if path.is_file() and zipfile.is_zipfile(path):
        tmp = tempfile.TemporaryDirectory(prefix="ragmerge_")
        manifest = unpack_pack(path, Path(tmp.name))
        return Path(tmp.name), manifest.get("name") or path.stem, tmp
    from rag.core.workspace import classify, read_marker

    if classify(path) in ("workspace", "legacy"):
        return path, (read_marker(path) or {}).get("name") or path.name, None
    raise MergeError(f"{path} is not a .ragpack file or a workspace folder")


def _decide(src_doc, dest_by_key: dict[str, Any], dest_shas: set[str], on_conflict: str,
            keep_copies: bool) -> str:
    """add | replace | identical | kept | copy | failed."""
    if src_doc["status"] == "failed":
        return "failed"
    existing = dest_by_key.get(src_doc["source_key"])
    if existing is not None:
        if existing["sha256"] == src_doc["sha256"]:
            return "identical"
        if on_conflict == "keep":
            return "kept"
        if on_conflict == "replace":
            return "replace"
        newer = (src_doc["mtime"] or 0, src_doc["ingested_at"] or "") > (existing["mtime"] or 0,
                                                                          existing["ingested_at"] or "")
        return "replace" if newer else "kept"
    if not keep_copies and src_doc["sha256"] in dest_shas:
        return "copy"
    return "add"


def merge(dest: Path, sources: list[Path], *, on_conflict: str = "newer", keep_copies: bool = False,
          dry_run: bool = False, progress: ProgressFn | None = None,
          should_stop: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Merge sources into the workspace at dest (created from the first source's settings if missing)."""
    from rag.core.workspace import Workspace

    if on_conflict not in CONFLICT_POLICIES:
        raise MergeError(f"on_conflict must be one of {', '.join(CONFLICT_POLICIES)}")
    emit = progress or (lambda e: None)
    stop = should_stop or (lambda: False)
    dest = Path(dest)
    for s in sources:
        if Path(s).resolve() == dest.resolve():
            raise MergeError(f"{s} is the destination; pick another knowledge base to merge in")
    opened: list[tuple[Path, Path, str, tempfile.TemporaryDirectory | None]] = []
    try:
        for s in sources:
            root, name, tmp = _open_source(Path(s))
            opened.append((Path(s), root, name, tmp))
        if not opened:
            raise MergeError("nothing to merge")

        created = False
        from rag.core.workspace import classify

        if classify(dest) is None and (dest / "rag.db").exists():
            raise MergeError(f"{dest} has a rag.db but isn't a usable workspace (run `rag verify` on it)")
        if classify(dest) is None:
            if dry_run:
                dest_meta: dict[str, str] = {}
            else:
                if dest.exists() and any(dest.iterdir()):
                    raise MergeError(f"{dest} exists and is not a workspace")
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(opened[0][1] / "config.toml", dest / "config.toml")
                Workspace.init(dest)
                created = True
        ws = Workspace.open(dest) if classify(dest) in ("workspace", "legacy") else None
        store = RagStore(ws.rag_db, readonly=dry_run) if ws else None
        tables = TablesStore(ws.tables_db, readonly=dry_run) if ws else None
        try:
            dest_meta = store.meta() if store else {}
            dest_embedding = _embedding(dest_meta)
            dest_docs = {r["source_key"]: r for r in store.documents()} if store else {}
            dest_shas = {r["sha256"] for r in dest_docs.values()}
            plans: list[SourcePlan] = []

            for src_path, root, name, _ in opened:
                if stop():
                    break
                plan = _merge_one(src_path, root, name, store, tables, ws, dest_embedding, dest_docs, dest_shas,
                                  on_conflict, keep_copies, dry_run, emit, stop)
                plans.append(plan)
                if plan.compatible and not dest_embedding:
                    dest_embedding = plan.embedding   # an empty destination takes the first source's
            if store is not None and not dry_run:
                store.set_meta("last_merge_at", now_iso())
                store.conn.commit()
        finally:
            if store is not None:
                store.close()
            if tables is not None:
                tables.close()
        return {"dest": str(dest), "created": created, "dry_run": dry_run, "on_conflict": on_conflict,
                "keep_copies": keep_copies, "sources": [p.to_dict() for p in plans],
                "cancelled": stop()}
    finally:
        for *_, tmp in opened:
            if tmp is not None:
                tmp.cleanup()


def _merge_one(src_path: Path, root: Path, name: str, store: RagStore | None, tables: TablesStore | None, ws,
               dest_embedding: dict[str, str], dest_docs: dict[str, Any], dest_shas: set[str], on_conflict: str,
               keep_copies: bool, dry_run: bool, emit: ProgressFn, stop: Callable[[], bool]) -> SourcePlan:
    src = RagStore(root / "rag.db", readonly=True)
    src_tables = TablesStore(root / "tables.db", readonly=True)
    try:
        src_embedding = _embedding(src.meta())
        plan = SourcePlan(path=str(src_path), name=name, compatible=True, embedding=src_embedding)
        if not src_embedding:
            plan.compatible, plan.reason = False, "it has never been built (no embedding settings)"
        elif dest_embedding and src_embedding != dest_embedding:
            diffs = [f"{k}: {src_embedding.get(k)!r} here, {dest_embedding.get(k)!r} in the destination"
                     for k in EMBED_META_KEYS if src_embedding.get(k) != dest_embedding.get(k)]
            plan.compatible = False
            plan.reason = "embedded differently (" + "; ".join(diffs) + ")"
        docs = src.documents()
        plan.documents = len(docs)
        if not plan.compatible:
            return plan
        if store is not None and not dest_embedding:
            # An empty destination: its vector table must have room for these vectors.
            dims = store.conn.execute("SELECT sql FROM sqlite_master WHERE name = 'chunk_vec'").fetchone()[0]
            m = re.search(r"float\[(\d+)\]", dims)
            if m and src_embedding.get("embed_dims") and m.group(1) != src_embedding["embed_dims"]:
                plan.compatible = False
                plan.reason = (f"its vectors have {src_embedding['embed_dims']} dimensions and the destination "
                               f"workspace stores {m.group(1)}")
                return plan
            if not dry_run:
                for k, v in src_embedding.items():
                    store.set_meta(k, v)
                store.conn.commit()

        for i, doc in enumerate(docs, start=1):
            if stop():
                break
            decision = _decide(doc, dest_docs, dest_shas, on_conflict, keep_copies)
            existing = dest_docs.get(doc["source_key"])
            if decision in ("replace", "kept"):
                plan.conflicts.append({"path": doc["display_path"], "decision": decision,
                                       "incoming_ingested_at": doc["ingested_at"],
                                       "existing_ingested_at": existing["ingested_at"] if existing else None})
            if decision == "failed":
                plan.failed_skipped += 1
            elif decision == "identical":
                plan.identical += 1
            elif decision == "copy":
                plan.copies += 1
            elif decision == "kept":
                plan.kept += 1
            else:
                n_chunks = int(doc["n_chunks"] or 0)
                n_tables = src_tables.conn.execute("SELECT COUNT(*) FROM _catalog WHERE doc_id = ?",
                                                   (doc["doc_id"],)).fetchone()[0]
                plan.chunks += n_chunks
                plan.tables += n_tables
                if decision == "add":
                    plan.add += 1
                else:
                    plan.replace += 1
                if not dry_run:
                    assert store is not None and tables is not None and ws is not None
                    if existing is not None:
                        tables.delete_for_doc(existing["doc_id"])
                        store.delete_document(existing["doc_id"])
                    renamed = _copy_document(doc, root, src, src_tables, store, tables, ws)
                    plan.renamed_tables += renamed
                    dest_docs[doc["source_key"]] = store.find_document(doc["source_key"])
                    dest_shas.add(doc["sha256"])
                else:
                    dest_docs[doc["source_key"]] = doc
                    dest_shas.add(doc["sha256"])
            emit({"event": "merge_doc", "source": name, "index": i, "total": len(docs),
                  "path": doc["display_path"], "decision": decision})
        return plan
    finally:
        src.close()
        src_tables.close()


def _copy_document(doc, root: Path, src: RagStore, src_tables: TablesStore, store: RagStore,
                   tables: TablesStore, ws) -> list[tuple[str, str]]:
    """Copy one document: tables first (they need the new doc_id), then chunks, FTS rows and vectors."""
    d = dict(doc)
    with store.conn:
        cur = store.conn.execute(
            "INSERT INTO documents(source_key, display_path, library_path, sha256, file_type, title, n_pages, "
            "size_bytes, mtime, extractor, status, warnings, n_chunks, ingested_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (d["source_key"], d["display_path"], d["library_path"], d["sha256"], d["file_type"], d["title"],
             d["n_pages"], d["size_bytes"], d["mtime"], d["extractor"], d["status"], d["warnings"],
             d["n_chunks"], d["ingested_at"]))
        new_doc = int(cur.lastrowid)

    # Tables, renamed when the name is taken.
    renames: dict[str, str] = {}
    for cat in src_tables.conn.execute("SELECT * FROM _catalog WHERE doc_id = ?", (d["doc_id"],)).fetchall():
        old = cat["table_name"]
        new = old
        existing = {r[0] for r in tables.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        n = 2
        while new in existing:
            new = f"{old}_{n}"
            n += 1
        ddl = src_tables.conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name = ?",
                                      (old,)).fetchone()[0]
        ddl = ddl.replace(f'"{old}"', f'"{new}"', 1)
        rows = src_tables.conn.execute(f'SELECT * FROM "{old}"').fetchall()
        with tables.conn:
            tables.conn.execute(ddl)
            if rows:
                marks = ",".join("?" * len(rows[0]))
                tables.conn.executemany(f'INSERT INTO "{new}" VALUES ({marks})', [tuple(r) for r in rows])
            tables.conn.execute(
                "INSERT INTO _catalog(table_name, doc_id, source_path, origin, sheet, cell_range, page, caption, "
                "n_rows, columns_json, card_chunk_id) VALUES (?,?,?,?,?,?,?,?,?,?,NULL)",
                (new, new_doc, cat["source_path"], cat["origin"], cat["sheet"], cat["cell_range"], cat["page"],
                 cat["caption"], cat["n_rows"], cat["columns_json"]))
        if new != old:
            renames[old] = new

    def rename_in(text: str | None) -> str | None:
        if not text or not renames:
            return text
        for old, new in renames.items():
            text = re.sub(rf"\b{re.escape(old)}\b", new, text)
        return text

    chunk_rows = src.conn.execute("SELECT * FROM chunks WHERE doc_id = ? ORDER BY ordinal", (d["doc_id"],)).fetchall()
    src_vecs = src.vectors.get([c["chunk_id"] for c in chunk_rows])
    try:
        with store.conn:
            vec_ids, vecs = [], []
            for c in chunk_rows:
                vec = src_vecs.get(c["chunk_id"])
                embed_text = rename_in(c["embed_text"])
                cur = store.conn.execute(
                    "INSERT INTO chunks(doc_id, ordinal, kind, text, embed_text, text_hash, heading_path, symbol, "
                    "page_start, page_end, line_start, line_end, sheet, cell_range, table_name, ocr_conf, asset_path, "
                    "n_tokens) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (new_doc, c["ordinal"], c["kind"], rename_in(c["text"]), embed_text, c["text_hash"],
                     c["heading_path"], c["symbol"], c["page_start"], c["page_end"], c["line_start"], c["line_end"],
                     c["sheet"], c["cell_range"], renames.get(c["table_name"], c["table_name"]), c["ocr_conf"],
                     c["asset_path"], c["n_tokens"]))
                cid = int(cur.lastrowid)
                store.conn.execute("INSERT INTO chunks_fts(rowid, embed_text) VALUES (?, ?)", (cid, embed_text))
                if vec is not None:
                    vec_ids.append(cid)
                    vecs.append(vec)
                if c["kind"] == "table_card" and c["table_name"]:
                    with tables.conn:
                        tables.conn.execute("UPDATE _catalog SET card_chunk_id = ? WHERE table_name = ?",
                                            (cid, renames.get(c["table_name"], c["table_name"])))
                if c["asset_path"]:
                    _copy_file(root / c["asset_path"], ws.root / c["asset_path"])
            if vec_ids:
                store.vectors.add(vec_ids, vecs, d["file_type"], new_doc)
    except BaseException:
        store.vectors.failed(new_doc)
        raise
    if d["library_path"]:
        _copy_file(root / d["library_path"], ws.root / d["library_path"])
    return list(renames.items())


def _copy_file(src: Path, dest: Path) -> None:
    if src.is_file() and not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
