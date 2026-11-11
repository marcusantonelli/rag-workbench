"""The Builder engine: files in, rag.db + tables.db out.

For each file: hash it; skip it if unchanged (and read by the current extractor
version); otherwise copy it into the
library, extract (cached by hash), load its tables into tables.db, chunk,
embed (reusing vectors for identical text) and write everything for that file
in one transaction. A failure in one file is logged and never stops the build.

Progress is reported as event dicts; `rag build --json` prints them as JSON
lines so the desktop app can drive the engine as a sidecar.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rag import __version__
from rag.core import spots
from rag.core.db.merge import refresh_spot
from rag.core.chunk.chunker import ChunkSettings, Chunker, card_chunk, row_chunks
from rag.core.chunk.tokens import TokenCounter
from rag.core.db.store import DocRecord, RagStore, now_iso
from rag.core.db.tables import TablesStore, card_text
from rag.core.ingest.context import ExtractContext
from rag.core.ingest.router import detect_type, extract, is_current, reading_tag
from rag.core.models.embed import make_embedder
from rag.core.types import Chunk, TableData
from rag.core.util import long_path, sha256_file
from rag.core.workspace import Workspace, reset_workspace_data

Event = dict[str, Any]


@dataclass
class SourceFile:
    path: Path
    source_key: str
    display_path: str
    root_key: str


@dataclass
class BuildStats:
    seen: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    failed: int = 0
    removed: int = 0
    skipped: int = 0
    chunks: int = 0
    tables: int = 0
    embedded: int = 0
    reused_vectors: int = 0
    seconds: float = 0.0
    errors: list[dict[str, str]] = field(default_factory=list)


def source_key(path: Path) -> str:
    return Path(os.path.normcase(str(Path(path).resolve()))).as_posix()


def _excluded(rel_posix: str, name: str, patterns: list[str]) -> bool:
    for pat in patterns:
        if "/" in pat:
            if fnmatch.fnmatch("/" + rel_posix, pat) or fnmatch.fnmatch(rel_posix, pat):
                return True
        elif fnmatch.fnmatch(name, pat):
            return True
    return False


def discover(inputs: list[Path], include: list[str], exclude: list[str], max_mb: float) -> tuple[list[SourceFile], list[str]]:
    """Walk input files and folders. Returns (files, skipped-with-reason)."""
    found: dict[str, SourceFile] = {}
    skipped: list[str] = []
    exts = {e.lower() for e in include}
    for inp in inputs:
        inp = Path(inp)
        if not inp.exists():
            skipped.append(f"{inp}: not found")
            continue
        root = inp.parent if inp.is_file() else inp
        candidates = [inp] if inp.is_file() else sorted(p for p in inp.rglob("*") if p.is_file())
        for p in candidates:
            rel = p.relative_to(root).as_posix()
            display = f"{root.name}/{rel}" if root.name else rel
            if _excluded(rel, p.name, exclude):
                continue
            if p.suffix.lower() not in exts:
                continue
            try:
                size_mb = p.stat().st_size / (1 << 20)
            except OSError as exc:
                skipped.append(f"{display}: {exc}")
                continue
            if size_mb > max_mb:
                skipped.append(f"{display}: larger than {max_mb} MB")
                continue
            key = source_key(p)
            found[key] = SourceFile(path=p, source_key=key, display_path=display, root_key=source_key(root))
    return list(found.values()), skipped


class Builder:
    def __init__(self, ws: Workspace, *, embedder=None, progress: Callable[[Event], None] | None = None,
                 should_stop: Callable[[], bool] | None = None):
        self.ws = ws
        # Checked between files: a cancelled build keeps every file finished so far.
        self.should_stop = should_stop or (lambda: False)
        self.cancelled = False
        self.cfg = ws.config
        self.progress = progress or (lambda e: None)
        self._embedder = embedder
        self.count = TokenCounter(self.cfg["embedding"].get("tokenizer_path") or None)

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = make_embedder(self.cfg["embedding"])
        return self._embedder

    def emit(self, event: str, **info: Any) -> None:
        self.progress({"event": event, **info})

    # ------------------------------------------------------------------------------
    def build(self, inputs: list[Path], *, force: bool = False, prune: bool = False,
              no_cache: bool = False) -> BuildStats:
        started = time.monotonic()
        stats = BuildStats()
        if force:
            reset_workspace_data(self.ws)
        store = RagStore(self.ws.rag_db)
        tables = TablesStore(self.ws.tables_db)
        try:
            store.ensure_embedding(self.embedder.info(), adopt=False)
            ing = self.cfg["ingest"]
            files, skipped = discover(inputs, ing["include"], ing["exclude"], float(ing["max_file_mb"]))
            for s in skipped:
                self.emit("skipped", reason=s)
            stats.skipped = len(skipped)
            self.emit("start", files=len(files), workspace=str(self.ws.root))
            settings = ChunkSettings.from_config(self.cfg["chunking"])
            settings.context = self._embed_context(store, settings.context)
            chunker = Chunker(settings, self.count, self.embedder)

            for i, sf in enumerate(files, start=1):
                if self.should_stop():
                    self.cancelled = True
                    self.emit("cancelled", index=i, total=len(files))
                    break
                stats.seen += 1
                self._process(sf, i, len(files), store, tables, chunker, stats, no_cache)

            if prune and not self.cancelled:
                stats.removed = self._prune(files, inputs, store, tables)
            store.set_meta("last_build_at", now_iso())
            store.set_meta("app_version", __version__)
            store.set_meta("config_fingerprint", self.cfg.fingerprint())
            store.conn.commit()
            if hasattr(store.vectors, "compact"):
                try:              # only when many rows are deleted or the file is in many pieces
                    store.vectors.compact()
                except Exception as exc:  # noqa: BLE001 - e.g. a search holds the file (Windows): next time
                    self.emit("warning", message=f"vectors.shammy wasn't compacted this time: {exc}")
        finally:
            store.close()
            tables.close()
        if stats.added or stats.updated or stats.removed or not spots.status(self.ws.root)["current"]:
            refresh_spot(self.ws.root, lambda e: self.emit(e.pop("event"), **e))
        stats.seconds = round(time.monotonic() - started, 2)
        self.emit("done", **{k: v for k, v in stats.__dict__.items() if k != "errors"},
                  errors=len(stats.errors))
        return stats

    def _embed_context(self, store: RagStore, wanted: int) -> int:
        """The embedded-text header this knowledge base uses. It is fixed when the first document is
        built: vectors made with different headers don't mix, so a knowledge base built before
        context 2 existed keeps context 1 (its vectors stay valid) until it is rebuilt from scratch."""
        recorded = store.get_meta("embed_context")
        if recorded is None:
            has_chunks = store.conn.execute("SELECT 1 FROM chunks LIMIT 1").fetchone() is not None
            recorded = "1" if has_chunks else str(wanted)
            store.set_meta("embed_context", recorded)
            store.conn.commit()
        try:
            current = int(recorded)
        except ValueError:
            current = 1
        if current != wanted:
            self.emit("warning", message=(
                f"this knowledge base embeds with context {current}, the config asks for {wanted}; "
                "the recorded one is kept so its vectors stay comparable (a full rebuild switches)"))
        return current

    # ------------------------------------------------------------------------------
    def _process(self, sf: SourceFile, index: int, total: int, store: RagStore, tables: TablesStore,
                 chunker: Chunker, stats: BuildStats, no_cache: bool) -> None:
        t0 = time.monotonic()
        try:
            digest = sha256_file(sf.path)
        except OSError as exc:
            stats.failed += 1
            stats.errors.append({"path": sf.display_path, "error": str(exc)})
            self.emit("file_failed", path=sf.display_path, error=str(exc), index=index, total=total)
            return
        existing = store.find_document(sf.source_key)
        current = existing is not None and is_current(
            existing["extractor"], self.cfg,
            store.conn.execute("SELECT 1 FROM chunks WHERE doc_id = ? AND asset_path IS NOT NULL LIMIT 1",
                               (existing["doc_id"],)).fetchone() is not None)
        if (existing is not None and existing["sha256"] == digest and existing["status"] in ("ok", "partial")
                and current):
            stats.unchanged += 1
            self.emit("file_unchanged", path=sf.display_path, index=index, total=total)
            return

        ftype = detect_type(sf.path)
        self.emit("file_start", path=sf.display_path, type=ftype, index=index, total=total)
        if ftype is None:
            stats.failed += 1
            msg = "unrecognized or mismatched file signature"
            stats.errors.append({"path": sf.display_path, "error": msg})
            self.emit("file_failed", path=sf.display_path, error=msg, index=index, total=total)
            return

        stat = sf.path.stat()
        rec = DocRecord(
            source_key=sf.source_key, display_path=sf.display_path, sha256=digest, file_type=ftype,
            size_bytes=stat.st_size, mtime=stat.st_mtime,
            library_path=self._copy_to_library(sf.path, digest),
        )
        if existing is not None:
            tables.delete_for_doc(existing["doc_id"])
        doc_id = store.reserve_document(rec)
        stage = "extract"
        try:
            ctx = ExtractContext(
                config=self.cfg, sha256=digest, display_path=sf.display_path, assets_root=self.ws.assets_dir,
                progress=lambda st, info: self.emit("stage", path=sf.display_path, stage=st, **info),
            )
            doc, cached = extract(sf.path, ftype, ctx, cache_dir=self.ws.cache_dir,
                                  overrides_file=self.ws.root / "sheets.toml", use_cache=not no_cache)
            # The version suffix lets a later build re-extract files read by an older extractor.
            rec.title, rec.n_pages, rec.extractor = doc.title, doc.n_pages, f"{doc.extractor}@{reading_tag(self.cfg, ctx)}"
            rec.warnings = list(doc.warnings)

            stage = "tables"
            card_chunks: list[Chunk] = []
            table_jobs: list[TableData] = list(doc.tables)
            inline_max = int(self.cfg["chunking"]["table_inline_max_rows"])
            big_blocks = []
            for n, block in enumerate(b for b in doc.blocks if b.kind == "table"):
                rows = block.meta.get("rows") or []
                if len(rows) > inline_max:
                    td = TableData(name_hint=f"{Path(sf.display_path).stem}__table_{n + 1}",
                                   headers=list(block.meta.get("headers") or []), rows=rows,
                                   caption=block.meta.get("caption"), page=block.page, origin="doc_table")
                    table_jobs.append(td)
                    big_blocks.append((block, td))
            loaded = {}
            for td in table_jobs:
                info = tables.load_table(doc_id, sf.display_path, td)
                loaded[id(td)] = info
                stats.tables += 1
                card_chunks.append(card_chunk(card_text(info), info.table_name, sf.display_path, info.sheet,
                                              info.cell_range, info.page, self.count))
                ck = self.cfg["chunking"]
                if (ck.get("row_chunks") and td.origin != "doc_table" and info.n_rows <= int(ck["row_chunk_max_rows"])
                        and any(c.type == "text" and c.examples and
                                sum(len(e) for e in c.examples) / len(c.examples) >= 25 for c in info.columns)):
                    card_chunks.extend(row_chunks(
                        [c.name for c in info.columns], td.rows, td.row_numbers or list(range(1, len(td.rows) + 1)),
                        info.table_name, sf.display_path, info.sheet, int(ck["row_chunk_size"]), self.count))
            for block, td in big_blocks:
                block.meta["table_name"] = loaded[id(td)].table_name

            stage = "chunk"
            self.emit("stage", path=sf.display_path, stage="chunk")
            chunks = chunker.chunk(doc, sf.display_path) + card_chunks
            if not chunks:
                rec.status = "partial"
                rec.warnings.append("no text extracted")
            elif rec.warnings and any("failed" in w for w in rec.warnings):
                rec.status = "partial"

            stage = "embed"
            vectors = self._embed(chunks, store, stats, sf.display_path)
            stage = "write"
            ids = store.write_chunks(doc_id, rec, chunks, vectors)
            for ch, cid in zip(chunks, ids):
                if ch.kind == "table_card" and ch.table_name:
                    tables.set_card_chunk(ch.table_name, cid)
            stats.chunks += len(chunks)
            if existing is None:
                stats.added += 1
            else:
                stats.updated += 1
            self.emit("file_done", path=sf.display_path, chunks=len(chunks), cached_extract=cached,
                      warnings=rec.warnings, seconds=round(time.monotonic() - t0, 2), index=index, total=total)
        except Exception as exc:  # noqa: BLE001 - one bad file never stops a build
            msg = f"{type(exc).__name__}: {exc}"
            store.log_error(sf.display_path, stage, msg + "\n" + traceback.format_exc(limit=6))
            store.mark_failed(doc_id, f"{stage}: {msg}")
            tables.delete_for_doc(doc_id)
            stats.failed += 1
            stats.errors.append({"path": sf.display_path, "stage": stage, "error": msg})
            self.emit("file_failed", path=sf.display_path, stage=stage, error=msg, index=index, total=total)
            if stage == "embed" and "cannot reach llama-server" in msg:
                raise  # the embedder is down; every remaining file would fail the same way

    def _embed(self, chunks: list[Chunk], store: RagStore, stats: BuildStats, path: str) -> np.ndarray:
        dims = int(self.cfg["embedding"]["dims"])
        if not chunks:
            return np.zeros((0, dims), dtype=np.float32)
        known = store.vectors_for_hashes(c.text_hash for c in chunks)
        todo = [i for i, c in enumerate(chunks) if c.text_hash not in known]
        out = np.zeros((len(chunks), dims), dtype=np.float32)
        for i, c in enumerate(chunks):
            if c.text_hash in known:
                out[i] = known[c.text_hash]
        stats.reused_vectors += len(chunks) - len(todo)
        if todo:
            batch = max(1, int(self.cfg["embedding"]["batch_size"]))
            for start in range(0, len(todo), batch):
                idx = todo[start:start + batch]
                out[idx] = self.embedder.embed_documents([chunks[i].embed_text for i in idx])
                self.emit("stage", path=path, stage="embed", done=min(start + batch, len(todo)), total=len(todo))
            stats.embedded += len(todo)
        return out

    def _copy_to_library(self, path: Path, digest: str) -> str | None:
        if not self.cfg["library"].get("copy_originals", True):
            return None
        dest = self.ws.library_dir / digest[:2] / f"{digest}{path.suffix.lower()}"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(long_path(path), long_path(dest))
        return dest.relative_to(self.ws.root).as_posix()

    def _prune(self, files: list[SourceFile], inputs: list[Path], store: RagStore, tables: TablesStore) -> int:
        """Remove documents under the given folders whose files no longer exist."""
        present = {f.source_key for f in files}
        roots = [source_key(p) for p in inputs if Path(p).is_dir()]
        removed = 0
        for row in store.documents():
            key = row["source_key"]
            if key in present or not any(key.startswith(r.rstrip("/") + "/") for r in roots):
                continue
            tables.delete_for_doc(row["doc_id"])
            store.delete_document(row["doc_id"])
            removed += 1
            self.emit("file_removed", path=row["display_path"])
        return removed

    def remove(self, paths: list[str]) -> int:
        """Remove documents by display path, glob, or original path."""
        store = RagStore(self.ws.rag_db)
        tables = TablesStore(self.ws.tables_db)
        removed = 0
        try:
            for row in store.documents():
                if any(fnmatch.fnmatch(row["display_path"], p) or row["source_key"] == source_key(Path(p))
                       for p in paths):
                    tables.delete_for_doc(row["doc_id"])
                    store.delete_document(row["doc_id"])
                    removed += 1
                    self.emit("file_removed", path=row["display_path"])
        finally:
            store.close()
            tables.close()
        if removed:
            refresh_spot(self.ws.root, lambda e: self.emit(e.pop("event"), **e))
        return removed
