"""`rag` command line: build packs, export them, and test retrieval.

    rag init WS                         create a workspace (config.toml + empty databases)
    rag build WS PATH... [--json]       ingest files/folders (incremental by content hash)
    rag remove WS PATTERN...            remove documents by display path glob
    rag status WS                       counts, failures, embedding settings
    rag export WS OUT.ragpack           write a verified pack
    rag unpack PACK DEST                extract and verify a pack (e.g. to keep building it)
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
import time
from pathlib import Path
from typing import Any


def _print_event(event: dict[str, Any]) -> None:
    e = event["event"]
    if e == "start":
        print(f"Building {event['files']} files into {event['workspace']}")
    elif e == "file_start":
        print(f"[{event['index']}/{event['total']}] {event['path']} ({event['type']})", flush=True)
    elif e == "file_done":
        extra = " (cached extraction)" if event.get("cached_extract") else ""
        warn = f"  warnings: {'; '.join(event['warnings'])}" if event.get("warnings") else ""
        print(f"    {event['chunks']} chunks in {event['seconds']}s{extra}{warn}")
    elif e == "file_failed":
        print(f"    FAILED {event['path']}: {event.get('stage', '')} {event['error']}", flush=True)
    elif e == "skipped":
        print(f"skipped {event['reason']}")
    elif e == "file_removed":
        print(f"removed {event['path']}")
    elif e == "done":
        print(f"\nDone in {event['seconds']}s: {event['added']} added, {event['updated']} updated, "
              f"{event['unchanged']} unchanged, {event['failed']} failed, {event['removed']} removed; "
              f"{event['chunks']} chunks, {event['tables']} tables, {event['embedded']} embedded, "
              f"{event['reused_vectors']} vectors reused.")


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
    from rag.core.workspace import Workspace

    overrides = _overrides(args)
    if args.embedder == "hash-test" and not args.dims:
        overrides["embedding"]["dims"] = 256
        overrides["embedding"]["native_dims"] = 256
        overrides["embedding"]["model"] = "hash-test"
    if args.embedder == "qwen3-onnx":
        overrides["embedding"].update({"model": "n24q02m/Qwen3-Embedding-0.6B-ONNX", "native_dims": 1024,
                                       "dims": args.dims or 1024})
    ws = Workspace.init(Path(args.workspace), overrides)
    print(f"Workspace ready at {ws.root}\n  embedding: {ws.config['embedding']['backend']} "
          f"{ws.config['embedding']['model']} ({ws.config['embedding']['dims']} dims)\n"
          f"  edit {ws.config_path} before the first build if needed.")
    return 0


def cmd_build(args) -> int:
    from rag.builder.engine import Builder
    from rag.core.db.store import EmbeddingMismatch
    from rag.core.models.embed import EmbedderError
    from rag.core.workspace import Workspace

    ws = Workspace.open(Path(args.workspace))
    builder = Builder(ws, progress=_json_event if args.json else _print_event)
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
    from rag.core.db.pack import export_pack
    from rag.core.workspace import Workspace

    ws = Workspace.open(Path(args.workspace))
    t0 = time.monotonic()
    manifest = export_pack(ws, Path(args.out), include_originals=args.include_originals, name=args.name)
    size = Path(args.out).stat().st_size / (1 << 20)
    print(f"Wrote {args.out} ({size:.1f} MB) in {time.monotonic() - t0:.1f}s\n"
          f"  pack_id {manifest['pack_id']}\n  {manifest['counts']['documents']} documents, "
          f"{manifest['counts']['chunks']} chunks, {manifest['counts']['tables']} tables\n"
          f"  embedding {manifest['embedding']['model']} ({manifest['embedding']['dims']} dims)")
    return 0


def cmd_unpack(args) -> int:
    from rag.core.db.pack import unpack_pack

    manifest = unpack_pack(Path(args.pack), Path(args.dest))
    print(f"Unpacked and verified {len(manifest['files'])} files into {args.dest}")
    return 0


def _searcher(path: Path):
    from rag.core.db.store import RagStore
    from rag.core.models.embed import make_embedder
    from rag.core.retrieve.search import Searcher
    from rag.core.workspace import open_any

    ws, tmp = open_any(path)
    store = RagStore(ws.rag_db, readonly=True)
    meta = store.meta()
    cfg = ws.config
    if meta.get("embed_model") and meta["embed_model"] != cfg["embedding"]["model"]:
        print(f"warning: pack embedded with {meta['embed_model']}, config says {cfg['embedding']['model']}",
              file=sys.stderr)
    searcher = Searcher(store, make_embedder(cfg["embedding"]), cfg["search"])
    return searcher, ws, tmp


def cmd_search(args) -> int:
    searcher, ws, tmp = _searcher(Path(args.workspace))
    res = searcher.search(args.query, k=args.k, file_type=args.type)
    if args.json:
        print(json.dumps(res.to_dict(), indent=2, default=str))
    else:
        print(res.to_markdown())
        print(f"\n(timings ms: {res.timings_ms})")
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
    report = run_eval(searcher, Path(args.golden))
    report["config_fingerprint"] = ws.config.fingerprint()
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
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
    d = emb.embed_documents(["Wire bond pull test minimum strength for 5 mil aluminum wire."])
    t1 = time.monotonic()
    q = emb.embed_query("What is the minimum pull strength?")
    t2 = time.monotonic()
    print(f"backend {emb.backend}  model {emb.model}\n  document vector: {d.shape[1]} dims in {1000*(t1-t0):.0f} ms\n"
          f"  query vector: {q.shape[0]} dims in {1000*(t2-t1):.0f} ms\n  cosine(doc, query) = {float(d[0] @ q):.3f}\n"
          f"  query prefix: {emb.info()['query_instruction']!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create a workspace")
    s.add_argument("workspace")
    s.add_argument("--embedder", choices=["llama-server", "llama-cpp", "qwen3-onnx", "hash-test"])
    s.add_argument("--dims", type=int, help="stored dims (MRL truncation), e.g. 2048")
    s.set_defaults(fn=cmd_init)

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
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("unpack", help="extract and verify a .ragpack")
    s.add_argument("pack")
    s.add_argument("dest")
    s.set_defaults(fn=cmd_unpack)

    s = sub.add_parser("search", help="hybrid search")
    s.add_argument("workspace")
    s.add_argument("query")
    s.add_argument("-k", type=int)
    s.add_argument("--type", help="only this file type, e.g. pdf")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_search)

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
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("check-embedder", help="test the embedding backend")
    s.add_argument("workspace")
    s.set_defaults(fn=cmd_check_embedder)

    args = p.parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
