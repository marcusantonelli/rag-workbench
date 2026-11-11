"""`rag` command line: build packs, export them, and test retrieval.

    rag init WS [--name NAME] [--vectors shammy]   create a workspace (marker, config.toml, empty databases)
    rag list [FOLDER]                   workspaces and packs in a folder (default: the app's workspaces)
    rag verify WS|PACK                  check a workspace or pack is whole
    rag build WS PATH... [--json]       ingest files/folders (incremental by content hash)
    rag remove WS PATTERN...            remove documents by display path glob
    rag status WS                       counts, failures, embedding settings
    rag export WS OUT.ragpack           write a verified pack
    rag unpack PACK DEST                extract and verify a pack (e.g. to keep building it)
    rag merge WS SRC...                 copy compatible packs/workspaces into WS (no re-embedding)
    rag convert WS [--to shammy]        move the vectors between sqlite-vec and Shammy (no re-embedding)
    rag compact WS                      rewrite vectors.shammy without deleted rows
    rag search WS|PACK "question"       hybrid search (keyword + vector, RRF)
    rag tables WS|PACK                  list tables in tables.db
    rag describe WS|PACK TABLE          schema, column ranges, sample rows
    rag sql WS|PACK "SELECT ..."        read-only query, as the agent's query_table would
    rag eval WS|PACK golden.jsonl       recall@5 / recall@20 / MRR@10 by question type
    rag check-embedder WS               embed a test string; report dims and latency
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
import time
from pathlib import Path
from typing import Any


def _stage_words(e: dict[str, Any]) -> str:
    st = e.get("stage")
    if st == "page" and e.get("pages"):
        return f"page {e['page']} of {e['pages']}"
    if st == "ocr":
        return f"OCR page {e.get('page')}"
    if st == "describe":
        return f"pictures described: {e.get('done')}"
    if st == "chunked":
        return f"{e.get('chunks')} chunks"
    if st == "embed" and e.get("total"):
        return f"embedding {e.get('done')} of {e.get('total')} chunks"
    return {"chunk": "chunking", "write": "saving", "tables": "tables"}.get(st, st or "")


def _fraction(e: dict[str, Any], before: float) -> float:
    """How far into the current file a stage event is (reading counts 60%, embedding the rest)."""
    st = e.get("stage")
    if st in ("page", "ocr") and e.get("pages"):
        return 0.6 * min(1.0, e["page"] / e["pages"])
    if st == "chunked":
        return 0.62
    if st == "embed" and e.get("total"):
        return 0.65 + 0.35 * min(1.0, e["done"] / e["total"])
    return before


class _Printer:
    """Build progress for a person. On a terminal: one status line that updates in place (bar,
    files, the current file and what is happening to it, time left), with a line per finished
    file above it. Piped or redirected: a line per file, as before."""

    def __init__(self, stream=None):
        self.out = stream or sys.stdout
        self.tty = bool(getattr(self.out, "isatty", lambda: False)())
        self.total = self.done = 0
        self.bytes_total = self.bytes_done = self.bytes_skipped = 0
        self.work0: float | None = None
        self.current = ""
        self.cur_bytes = 0
        self.cur_frac = 0.0
        self.status = False

    def _clear(self) -> None:
        if self.status:
            self.out.write("\r\033[K")
            self.status = False

    def _line(self, text: str) -> None:
        self._clear()
        print(text, file=self.out, flush=True)

    def _left(self) -> str:
        if self.work0 is None or not self.bytes_total:
            return ""
        elapsed = time.monotonic() - self.work0
        # count how far the current file has got, or a large file being read makes the files
        # finished before it look like the whole rate
        work = self.bytes_done - self.bytes_skipped + self.cur_bytes * self.cur_frac
        remaining = self.bytes_total - self.bytes_done - self.cur_bytes * self.cur_frac
        if elapsed < 10 or work <= 0 or remaining <= 0:
            return ""
        secs = elapsed / work * remaining
        return f" · about {secs / 60:.0f} min left" if secs >= 90 else f" · about {secs:.0f} s left"

    def _status(self, words: str) -> None:
        if not self.tty:
            return
        width = 20
        filled = int(width * self.done / self.total) if self.total else 0
        bar = "#" * filled + "-" * (width - filled)
        text = f"[{bar}] {self.done}/{self.total} · {self.current} · {words}{self._left()}"
        cols = shutil.get_terminal_size((100, 20)).columns
        self.out.write("\r\033[K" + text[: max(20, cols - 1)])
        self.out.flush()
        self.status = True

    def __call__(self, event: dict[str, Any]) -> None:
        e = event["event"]
        if e == "start":
            self.total, self.bytes_total = event["files"], int(event.get("bytes") or 0)
            self._line(f"Building {event['files']} files into {event['workspace']}")
        elif e == "file_start":
            if self.work0 is None:
                self.work0 = time.monotonic()
            self.current = event["path"]
            self.cur_bytes, self.cur_frac = int(event.get("bytes") or 0), 0.0
            if self.tty:
                self._status("reading")
            else:
                print(f"[{event['index']}/{event['total']}] {event['path']} ({event['type']})", flush=True)
        elif e == "stage":
            self.cur_frac = _fraction(event, self.cur_frac)
            self._status(_stage_words(event))
        elif e in ("file_done", "file_unchanged", "file_failed"):
            self.done = event.get("index", self.done)
            self.bytes_done += int(event.get("bytes") or 0)
            self.cur_bytes, self.cur_frac = 0, 0.0
            if e == "file_unchanged":
                self.bytes_skipped += int(event.get("bytes") or 0)
                self._status("unchanged")
                return
            if e == "file_done":
                extra = " (cached extraction)" if event.get("cached_extract") else ""
                warn = f"  warnings: {'; '.join(event['warnings'])}" if event.get("warnings") else ""
                head = f"{event['path']}: " if self.tty else "    "
                self._line(f"{head}{event['chunks']} chunks in {event['seconds']}s{extra}{warn}")
            else:
                self._line(f"    FAILED {event['path']}: {event.get('stage', '')} {event['error']}")
        elif e == "skipped":
            self._line(f"skipped {event['reason']}")
        elif e == "file_removed":
            self._line(f"removed {event['path']}")
        elif e == "warning":
            self._line(f"warning: {event.get('message')}")
        elif e == "done":
            self._line(f"\nDone in {event['seconds']}s: {event['added']} added, {event['updated']} updated, "
                       f"{event['unchanged']} unchanged, {event['failed']} failed, {event['removed']} removed; "
                       f"{event['chunks']} chunks, {event['tables']} tables, {event['embedded']} embedded, "
                       f"{event['reused_vectors']} vectors reused.")


def _print_event(event: dict[str, Any]) -> None:
    """A line per file (used where a fresh printer isn't made, e.g. `rag remove`)."""
    _Printer()(event)


def _json_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, default=str), flush=True)


def _overrides(args) -> dict[str, Any]:
    o: dict[str, Any] = {}
    if getattr(args, "embedder", None):
        o.setdefault("embedding", {})["backend"] = args.embedder
    if getattr(args, "dims", None):
        o.setdefault("embedding", {})["dims"] = args.dims
    return o


def cmd_init(args) -> int:
    from rag.core.db.vectors import VectorStoreError
    from rag.core.workspace import Workspace

    overrides = _overrides(args)
    if args.embedder == "hash-test" and not args.dims:
        overrides["embedding"]["dims"] = 256
        overrides["embedding"]["native_dims"] = 256
        overrides["embedding"]["model"] = "hash-test"
    if args.embedder == "qwen3-onnx":
        overrides["embedding"].update({"model": "n24q02m/Qwen3-Embedding-0.6B-ONNX", "native_dims": 1024,
                                       "dims": args.dims or 1024})
    if args.vectors:
        overrides.setdefault("vectors", {})["backend"] = args.vectors
    try:
        ws = Workspace.init(Path(args.workspace), overrides, name=args.name)
    except VectorStoreError as exc:
        print(f"rag: {exc}", file=sys.stderr)
        return 1
    print(f"Workspace \"{ws.name}\" ready at {ws.root}\n  embedding: {ws.config['embedding']['backend']} "
          f"{ws.config['embedding']['model']} ({ws.config['embedding']['dims']} dims)\n"
          f"  vectors: {ws.config['vectors']['backend']}\n"
          f"  edit {ws.config_path} before the first build if needed.")
    return 0


def cmd_build(args) -> int:
    from rag.builder.engine import Builder
    from rag.core.db.store import EmbeddingMismatch
    from rag.core.models.embed import EmbedderError
    from rag.core.workspace import Workspace

    ws = Workspace.open(Path(args.workspace))
    builder = Builder(ws, progress=_json_event if args.json else _Printer())
    try:
        stats = builder.build([Path(p) for p in args.paths], force=args.force, prune=args.prune,
                              no_cache=args.no_cache)
    except (EmbeddingMismatch, EmbedderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 1 if stats.failed and not args.allow_failures else 0


def cmd_remove(args) -> int:
    from rag.builder.engine import Builder
    from rag.core.workspace import Workspace

    n = Builder(Workspace.open(Path(args.workspace)), progress=_print_event).remove(args.patterns)
    print(f"{n} documents removed")
    return 0


def cmd_status(args) -> int:
    from rag.core.db.store import RagStore
    from rag.core.db.tables import TablesStore
    from rag.core.workspace import open_any

    ws, tmp = open_any(Path(args.workspace))
    store = RagStore(ws.rag_db, readonly=True)
    try:
        info = {"workspace": str(args.workspace), "meta": store.meta(), "counts": store.counts()}
        info["failed_documents"] = [
            {"path": r["display_path"], "warnings": json.loads(r["warnings"] or "[]")}
            for r in store.documents() if r["status"] == "failed"]
        info["recent_errors"] = [dict(r) for r in store.conn.execute(
            "SELECT path, stage, substr(message, 1, 300) AS message, at FROM ingest_errors ORDER BY at DESC LIMIT 10")]
    finally:
        store.close()
    t = TablesStore(ws.tables_db, readonly=True)
    info["tables"] = len(t.catalog())
    t.close()
    print(json.dumps(info, indent=2, default=str))
    return 0


def cmd_export(args) -> int:
    from rag.core.db.pack import PackError, export_pack
    from rag.core.workspace import Workspace

    ws = Workspace.open(Path(args.workspace))
    t0 = time.monotonic()
    try:
        manifest = export_pack(ws, Path(args.out), include_originals=args.include_originals, name=args.name,
                               slim=args.slim)
    except PackError as exc:
        print(f"rag: {exc}", file=sys.stderr)
        return 1
    size = Path(args.out).stat().st_size / (1 << 20)
    print(f"Wrote {args.out} ({size:.1f} MB) in {time.monotonic() - t0:.1f}s\n"
          f"  pack_id {manifest['pack_id']}\n  {manifest['counts']['documents']} documents, "
          f"{manifest['counts']['chunks']} chunks, {manifest['counts']['tables']} tables\n"
          f"  embedding {manifest['embedding']['model']} ({manifest['embedding']['dims']} dims)"
          + (f"\n  format {manifest['format']} (vectors in Shammy, {manifest['vector_tier']})"
             if manifest.get("vector_store") == "shammy" else ""))
    return 0


def cmd_unpack(args) -> int:
    from rag.core.db.pack import unpack_pack

    manifest = unpack_pack(Path(args.pack), Path(args.dest))
    print(f"Unpacked and verified {len(manifest['files'])} files into {args.dest}")
    return 0


def cmd_merge(args) -> int:
    from rag.core.db.merge import MergeError, merge

    def progress(e):
        if args.json:
            print(json.dumps(e), flush=True)

    try:
        res = merge(Path(args.dest), [Path(s) for s in args.sources], on_conflict=args.on_conflict,
                    keep_copies=args.keep_copies, dry_run=args.dry_run, progress=progress)
    except MergeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"event": "merged", **res}))
        return 0 if all(s["compatible"] for s in res["sources"]) else 2
    verb = "Would merge" if res["dry_run"] else "Merged"
    if res["created"]:
        print(f"Created workspace {res['dest']}")
    ok = True
    for s in res["sources"]:
        if not s["compatible"]:
            ok = False
            print(f"Skipped {s['name']}: {s['reason']}")
            continue
        print(f"{verb} {s['name']}: {s['add']} new, {s['replace']} replaced "
              f"({_n(s['chunks'], 'chunk')}, {_n(s['tables'], 'table')})")
        skipped = [(s["identical"], "already there"), (s["copies"], "same content under another path"),
                   (s["kept"], "kept the destination's version"), (s["failed_skipped"], "failed to build")]
        for n, why in skipped:
            if n:
                print(f"  skipped {n}: {why}")
        for old, new in s["renamed_tables"]:
            print(f"  table {old} renamed to {new}")
        for c in s["conflicts"][:20]:
            print(f"  {c['decision']:8} {c['path']}")
    if res["dry_run"]:
        print("Nothing was changed (--dry-run).")
    return 0 if ok else 2


def _human(n: int | None) -> str:
    if n is None:
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return ""


def _local_time(iso: str | None) -> str:
    from datetime import datetime

    if not iso:
        return ""
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return iso[:16]


def _n(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def cmd_list(args) -> int:
    from rag.core.listing import default_folder, list_knowledge_bases

    folder = Path(args.folder) if args.folder else default_folder()
    rows = list_knowledge_bases(folder, sizes=not args.no_size)
    if args.json:
        print(json.dumps({"folder": str(folder), "items": rows}, default=str))
        return 0
    if not folder.is_dir():
        print(f"{folder} doesn't exist yet." + ("" if args.folder else
              " Workspaces made in RAG Workbench go there; give a folder to list another one."))
        return 0
    if not rows:
        print(f"No workspaces or packs in {folder}")
        return 0
    print(f"In {folder}:\n")
    headers = ["Name", "Kind", "Docs", "Chunks", "Tables", "Model", "Updated", "Size", "Folder or file"]
    table = [[r["name"], r["kind"], str(r["documents"] if r["documents"] is not None else ""),
              str(r["chunks"] if r["chunks"] is not None else ""), str(r["tables"] if r["tables"] is not None else ""),
              (r["model"] or "").split("/")[-1] + (f" ({r['dims']})" if r["dims"] else ""),
              _local_time(r["updated"]), _human(r["size_bytes"]),
              (Path(r["path"]).name if Path(r["path"]) != folder else ".") + (f"  [{r['note']}]" if r["note"] else "")]
             for r in rows]
    widths = [max(len(h), *(len(row[i]) for row in table)) for i, h in enumerate(headers)]
    right = {2, 3, 4, 7}
    fmt = lambda cells: "  ".join(c.rjust(w) if i in right else c.ljust(w)  # noqa: E731
                                  for i, (c, w) in enumerate(zip(cells, widths))).rstrip()
    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for row in table:
        print(fmt(row))
    return 0


def cmd_convert(args) -> int:
    from rag.core.db.convert import convert_workspace
    from rag.core.db.vectors import VectorStoreError
    from rag.core.workspace import NotAWorkspace, Workspace

    def show(done: int, total: int) -> None:
        if done == total or done % 2000 < 50:
            print(f"\r  {done:,} of {total:,} vectors", end="", file=sys.stderr, flush=True)

    try:
        res = convert_workspace(Workspace.open(Path(args.workspace)), args.to, progress=None if args.json else show)
    except (VectorStoreError, NotAWorkspace) as exc:
        print(f"rag: {exc}", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(res))
    elif not res["changed"]:
        print(f"Already using {res['to']} ({_n(res['vectors'], 'vector')}); nothing to do.")
    else:
        print(f"\rMoved {_n(res['vectors'], 'vector')} from {res['from']} to {res['to']}.          ")
        if res["missing"]:
            print(f"  {_n(res['missing'], 'chunk')} had no vector to move (run `rag verify`).")
        print("  " + ", ".join(f"{name} is now {_human(size)}" for name, size in res["bytes"].items() if size))
        if res["to"] == "shammy":
            print("  Packs exported from this workspace are format ragpack/2, which the retrieval server "
                  "can't read until it has Shammy support.")
    return 0


def cmd_compact(args) -> int:
    from rag.core.db.convert import compact_workspace
    from rag.core.db.vectors import VectorStoreError
    from rag.core.workspace import NotAWorkspace, Workspace

    try:
        res = compact_workspace(Workspace.open(Path(args.workspace)))
    except (VectorStoreError, NotAWorkspace) as exc:
        print(f"rag: {exc}", file=sys.stderr)
        return 1
    if res is None:
        print("This workspace keeps its vectors in rag.db (sqlite-vec); there is nothing to compact.")
    else:
        print(f"vectors.shammy: {res['rows_before'] - res['rows_after']:,} deleted rows dropped, "
              f"{_human(res['bytes_before'])} to {_human(res['bytes_after'])}.")
    return 0


def cmd_verify(args) -> int:
    from rag.core.check import check_path, format_report

    res = check_path(Path(args.target))
    print(json.dumps(res, default=str) if args.json else format_report(res))
    return 0 if res["ok"] else 1


def _searcher(path: Path, rerank: bool | None = None, rerank_url: str | None = None):
    from rag.core.db.store import RagStore
    from rag.core.models.embed import make_embedder
    from rag.core.retrieve.search import Searcher, rerank_settings
    from rag.core.workspace import open_any

    ws, tmp = open_any(path)
    store = RagStore(ws.rag_db, readonly=True)
    meta = store.meta()
    cfg = ws.config
    if meta.get("embed_model") and meta["embed_model"] != cfg["embedding"]["model"]:
        print(f"warning: pack embedded with {meta['embed_model']}, config says {cfg['embedding']['model']}",
              file=sys.stderr)
    searcher = Searcher(store, make_embedder(cfg["embedding"]), cfg["search"],
                        reranker=rerank_settings(cfg.data, rerank, rerank_url))
    return searcher, ws, tmp


def cmd_search(args) -> int:
    searcher, ws, tmp = _searcher(Path(args.workspace), args.rerank, args.rerank_url)
    try:
        res = searcher.search(args.query, k=args.k, file_type=args.type, documents=args.doc or None)
    finally:
        searcher.close()
    if res.rerank and not res.rerank.get("applied"):
        print(f"warning: not reranked: {res.rerank.get('error')}", file=sys.stderr)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    else:
        print(res.to_markdown())
        print(f"\n(timings ms: {res.timings_ms}" + ("; reranked" if res.rerank and res.rerank.get("applied") else "") + ")")
    return 0


def cmd_spots(args) -> int:
    """Build (or rebuild) a workspace's spots file: its procedures, steps and references."""
    from rag.core import spots
    from rag.core.workspace import Workspace

    ws = Workspace.open(Path(args.workspace))
    if args.check:
        st = spots.status(ws.root)
        print(json.dumps(st, indent=2) if args.json else
              ("no spots file" if not st["present"] else
               f"{spots.SPOTS_NAME}: {'current' if st['current'] else 'out of date (run rag spots)'}; "
               f"{st.get('procedures', 0)} procedures, {st.get('steps', 0)} steps, {st.get('links', 0)} references"))
        return 0 if st.get("current") else 1
    counts = spots.build(ws.rag_db)
    if args.json:
        print(json.dumps(counts))
    else:
        print(f"{spots.SPOTS_NAME}: {_n(counts['procedures'], 'procedure')}, {_n(counts['steps'], 'step')}, "
              f"{_n(counts['links'], 'reference')} in {_n(counts['documents'], 'document')}")
    return 0


def cmd_procedure(args) -> int:
    """A page of a procedure, as `get_procedure_steps` returns it to an agent."""
    from rag.core.db.store import RagStore
    from rag.core.retrieve.search import Searcher
    from rag.core.workspace import open_any

    ws, tmp = open_any(Path(args.workspace))
    store = RagStore(ws.rag_db, readonly=True)
    searcher = Searcher(store, None, ws.config["search"])
    try:
        page = searcher.procedure(args.id, from_=args.from_step, after_chunk=args.after_chunk, max_tokens=args.max_tokens)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        searcher.close()
        store.close()
        if tmp is not None:
            tmp.cleanup()
    print(json.dumps(page, indent=2, default=str) if args.json else page.get("markdown", ""))
    return 0


def cmd_update(args) -> int:
    """Say whether a newer release exists. The desktop app installs updates itself."""
    from rag import __version__
    from rag.updates import latest, newer

    try:
        rel = latest()
    except Exception as exc:  # noqa: BLE001 - offline, rate-limited, blocked by a proxy
        print(f"Couldn't check for updates: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"current": __version__, "latest": rel["version"], "newer": newer(rel["version"]), "url": rel["url"]}))
    elif newer(rel["version"]):
        print(f"rag-builder {rel['version']} is available (you have {__version__}): {rel['url']}")
        print("The desktop app offers it on its own; for the command line alone, install the new release.")
    else:
        print(f"rag-builder {__version__} is the latest version.")
    return 0


def cmd_tables(args) -> int:
    from rag.core.db.tables import TablesStore
    from rag.core.workspace import open_any

    ws, tmp = open_any(Path(args.workspace))
    t = TablesStore(ws.tables_db, readonly=True)
    for row in t.catalog():
        loc = " ".join(x for x in (row["sheet"] and f"sheet={row['sheet']}", row["cell_range"],
                                   row["page"] and f"page={row['page']}") if x)
        print(f"{row['table_name']:40} {row['n_rows']:>7} rows  {row['source_path']} {loc}")
    t.close()
    return 0


def cmd_describe(args) -> int:
    from rag.core.db.tables import TablesStore
    from rag.core.workspace import open_any

    ws, tmp = open_any(Path(args.workspace))
    t = TablesStore(ws.tables_db, readonly=True)
    print(json.dumps(t.describe(args.table), indent=2, default=str))
    t.close()
    return 0


def cmd_sql(args) -> int:
    from rag.core.db.sqlguard import QueryRejected, rows_to_markdown, run_query
    from rag.core.workspace import open_any

    ws, tmp = open_any(Path(args.workspace))
    try:
        res = run_query(ws.tables_db, args.sql, row_limit=int(ws.config["sql"]["row_limit"]),
                        time_limit_ms=int(ws.config["sql"]["time_limit_ms"]))
    except QueryRejected as exc:
        print(f"rejected: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2, default=str) if args.json else rows_to_markdown(res))
    return 0


def cmd_eval(args) -> int:
    from rag.eval import run_eval

    searcher, ws, tmp = _searcher(Path(args.workspace))
    show = args.show if args.show is not None else (6 if args.report else 0)
    report = run_eval(searcher, Path(args.golden), show=show)
    report["config_fingerprint"] = ws.config.fingerprint()
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.report:
        from rag.eval import report_markdown

        Path(args.report).write_text(report_markdown(report), encoding="utf-8")
        print(f"report with retrieved chunks: {args.report}")
    o = report["overall"]
    print(f"{o['n']} questions  recall@5 {o['recall@5']}  recall@20 {o['recall@20']}  MRR@10 {o['mrr@10']}")
    for t, s in report["by_type"].items():
        print(f"  {t:16} n={s['n']:<3} recall@5 {s['recall@5']}  recall@20 {s['recall@20']}  MRR@10 {s['mrr@10']}")
    lat = report["latency_ms"]
    print(f"latency ms: embed p50 {lat['embed_p50']} p95 {lat['embed_p95']}; "
          f"search p50 {lat['search_p50']} p95 {lat['search_p95']}")
    misses = [q for q in report["questions"] if not q["hit@20"]]
    for q in misses:
        print(f"  MISS {q['id']}: {q['question']}")
    if report["unanswerable"]:
        print(f"{len(report['unanswerable'])} unanswerable questions not scored here (judge 'not found' via the agent)")
    return 0


def cmd_check_embedder(args) -> int:
    from rag.core.models.embed import make_embedder
    from rag.core.workspace import Workspace

    ws = Workspace.open(Path(args.workspace))
    emb = make_embedder(ws.config["embedding"])
    t0 = time.monotonic()
    d = emb.embed_documents(["Minimum clamp pull force for a 10 mm silicone hose."])
    t1 = time.monotonic()
    q = emb.embed_query("What is the minimum pull strength?")
    t2 = time.monotonic()
    print(f"backend {emb.backend}  model {emb.model}\n  document vector: {d.shape[1]} dims in {1000*(t1-t0):.0f} ms\n"
          f"  query vector: {q.shape[0]} dims in {1000*(t2-t1):.0f} ms\n  cosine(doc, query) = {float(d[0] @ q):.3f}\n"
          f"  query prefix: {emb.info()['query_instruction']!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    import sys

    from rag import __version__

    # Windows consoles may not be UTF-8; print what they can rather than crash on "›".
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            pass
    p = argparse.ArgumentParser(prog="rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"rag {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a workspace")
    s.add_argument("workspace")
    s.add_argument("--embedder", choices=["llama-server", "llama-cpp", "qwen3-onnx", "hash-test"])
    s.add_argument("--dims", type=int, help="stored dims (MRL truncation), e.g. 2048")
    s.add_argument("--name", help="display name (default: the folder name)")
    s.add_argument("--vectors", choices=["sqlite-vec", "shammy"],
                   help="where vectors are kept (default sqlite-vec; shammy is smaller and faster)")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("convert", help="move a workspace's vectors between sqlite-vec and Shammy (nothing is embedded again)")
    s.add_argument("workspace")
    s.add_argument("--to", choices=["shammy", "sqlite-vec"], default="shammy")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_convert)

    s = sub.add_parser("compact", help="rewrite vectors.shammy without deleted rows")
    s.add_argument("workspace")
    s.set_defaults(fn=cmd_compact)

    s = sub.add_parser("list", help="list workspaces and packs in a folder (default: the app's workspaces folder)")
    s.add_argument("folder", nargs="?", help="folder to look in")
    s.add_argument("--json", action="store_true")
    s.add_argument("--no-size", action="store_true", help="skip adding up folder sizes (faster on big folders)")
    s.set_defaults(fn=cmd_list)

    s = sub.add_parser("verify", help="check a workspace or pack is whole (marker, databases, links, checksums)")
    s.add_argument("target", help="workspace folder or .ragpack file")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_verify)

    s = sub.add_parser("build", help="ingest files and folders")
    s.add_argument("workspace")
    s.add_argument("paths", nargs="+")
    s.add_argument("--json", action="store_true", help="JSON-lines progress (for the desktop app)")
    s.add_argument("--force", action="store_true", help="drop everything and rebuild (cache kept)")
    s.add_argument("--no-cache", action="store_true", help="re-extract even when cached")
    s.add_argument("--prune", action="store_true", help="remove documents deleted from the given folders")
    s.add_argument("--allow-failures", action="store_true", help="exit 0 even if some files failed")
    s.set_defaults(fn=cmd_build)

    s = sub.add_parser("remove", help="remove documents")
    s.add_argument("workspace")
    s.add_argument("patterns", nargs="+")
    s.set_defaults(fn=cmd_remove)

    s = sub.add_parser("status", help="show counts and failures")
    s.add_argument("workspace")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("export", help="write a .ragpack")
    s.add_argument("workspace")
    s.add_argument("out")
    s.add_argument("--include-originals", action="store_true")
    s.add_argument("--name")
    s.add_argument("--slim", action="store_true",
                   help="Shammy workspaces: leave out the float16 vector copies (about a third of the vector "
                        "size; results are no longer rescored exactly and the pack can't be merged)")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("unpack", help="extract and verify a .ragpack")
    s.add_argument("pack")
    s.add_argument("dest")
    s.set_defaults(fn=cmd_unpack)

    s = sub.add_parser("merge", help="merge packs or workspaces into a workspace (no re-embedding)",
                       description="Copy the documents, vectors and tables of compatible knowledge bases "
                                   "(same embedding model and vector size) into DEST. DEST is created from "
                                   "the first source's settings if it doesn't exist.")
    s.add_argument("dest", help="workspace folder to merge into")
    s.add_argument("sources", nargs="+", help=".ragpack files or workspace folders")
    s.add_argument("--on-conflict", choices=["newer", "keep", "replace"], default="newer",
                   help="the same file with different content in both: keep the newer build (default), "
                        "keep DEST's, or take the source's")
    s.add_argument("--keep-copies", action="store_true",
                   help="also add files whose content DEST already has under another path")
    s.add_argument("--dry-run", action="store_true", help="report what would happen; change nothing")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_merge)

    s = sub.add_parser("search", help="hybrid search")
    s.add_argument("workspace")
    s.add_argument("query")
    s.add_argument("-k", type=int)
    s.add_argument("--type", help="only this file type, e.g. pdf")
    s.add_argument("--doc", action="append", metavar="DOC",
                   help="only this document: its number (e.g. DOC-12, from the spots file) or part of its path; repeatable")
    s.add_argument("--json", action="store_true")
    s.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                   help="reorder the best candidates with the reranker model (default: the workspace's [rerank] setting)")
    s.add_argument("--rerank-url", help="the reranker's address, e.g. http://127.0.0.1:8082 (implies nothing; use with --rerank)")
    s.set_defaults(fn=cmd_search)

    s = sub.add_parser("spots", help="build the spots file (procedures, steps, references) from the stored chunks; "
                                    "nothing is read or embedded again")
    s.add_argument("workspace")
    s.add_argument("--check", action="store_true", help="only say whether it is there and current")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_spots)

    s = sub.add_parser("procedure", help="read a procedure page by page (the `Read on:` line of a search result names "
                                         "the procedure and where to continue)")
    s.add_argument("workspace")
    s.add_argument("id", type=int, help="the procedure's number")
    s.add_argument("--from", dest="from_step", metavar="STEP", help="start at this step, e.g. 7.35 (default: the first)")
    s.add_argument("--after-chunk", type=int, help="continue inside a long step after this chunk")
    s.add_argument("--max-tokens", type=int, default=4000, help="page size (default 4000)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_procedure)

    s = sub.add_parser("update", help="check whether a newer release exists (the desktop app updates itself)")
    s.add_argument("--check", action="store_true", help="only check (the default; the command line never installs)")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_update)

    s = sub.add_parser("tables", help="list tables")
    s.add_argument("workspace")
    s.set_defaults(fn=cmd_tables)

    s = sub.add_parser("describe", help="describe one table")
    s.add_argument("workspace")
    s.add_argument("table")
    s.set_defaults(fn=cmd_describe)

    s = sub.add_parser("sql", help="read-only SQL over tables.db")
    s.add_argument("workspace")
    s.add_argument("sql")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_sql)

    s = sub.add_parser("eval", help="evaluate retrieval on a golden set")
    s.add_argument("workspace")
    s.add_argument("golden")
    s.add_argument("--out", help="write the full JSON report here")
    s.add_argument("--show", type=int, help="include the top N retrieved chunks per question in --out "
                                            "(default 6 when --report is given)")
    s.add_argument("--report", help="write a readable Markdown report with the retrieved chunks")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("check-embedder", help="test the embedding backend")
    s.add_argument("workspace")
    s.set_defaults(fn=cmd_check_embedder)

    args = p.parse_args(argv)
    from rag.core.db.connect import ExtensionLoadingUnavailable
    from rag.core.db.pack import PackError
    from rag.core.db.store import EmbeddingMismatch
    from rag.core.models.embed import EmbedderError

    try:
        return int(args.fn(args) or 0)
    except (EmbedderError, EmbeddingMismatch, PackError, ExtensionLoadingUnavailable, FileNotFoundError) as exc:
        # Expected problems get one clear line, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
