"""Check that a workspace or pack is whole: marker, settings, databases and the links between them.

Problems are "error" (searching or building will go wrong) or "warning" (works, but something is
missing, e.g. a figure image). Nothing is changed; packs are first checked against their checksums.
"""

from __future__ import annotations

import json
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from rag import SCHEMA_VERSION


def _problem(out: list[dict[str, str]], level: str, message: str) -> None:
    out.append({"level": level, "message": message})


def _ro(db: Path, load_vec: bool) -> sqlite3.Connection:
    from rag.core.db.connect import connect

    return connect(db, readonly=True, load_vec=load_vec)


def check_path(path: Path) -> dict[str, Any]:
    """Check a .ragpack file (checksums, then contents) or a workspace folder."""
    import zipfile

    from rag.core.db.pack import PackError, unpack_pack

    path = Path(path)
    if path.is_file():
        if not zipfile.is_zipfile(path):
            return _result(path, "unknown", [{"level": "error", "message": "not a .ragpack file"}], {})
        with tempfile.TemporaryDirectory(prefix="ragcheck_") as tmp:
            try:
                manifest = unpack_pack(path, Path(tmp))
            except PackError as exc:
                return _result(path, "pack", [{"level": "error", "message": str(exc)}], {})
            res = check_workspace(Path(tmp), pack=True)
            res.update({"path": str(path), "kind": "pack", "name": manifest.get("name"),
                        "checksums": f"all {len(manifest['files'])} files match"})
            counts = manifest.get("counts", {})
            if counts.get("chunks") is not None and res["counts"].get("chunks") != counts["chunks"]:
                _problem(res["problems"], "error", f"the manifest lists {counts['chunks']} chunks, "
                                                   f"the database holds {res['counts'].get('chunks')}")
                res["ok"] = False
            return res
    return check_workspace(path)


def _result(path: Path, kind: str, problems: list[dict[str, str]], counts: dict[str, Any],
            name: str | None = None) -> dict[str, Any]:
    return {"path": str(path), "kind": kind, "name": name, "ok": not any(p["level"] == "error" for p in problems),
            "problems": problems, "counts": counts}


def _check_shammy(root: Path, conn, problems: list[dict[str, str]]) -> tuple[int, int]:
    """Checksums of vectors.shammy and its ids against the chunks. Returns (vectors, chunks without one)."""
    import numpy as np

    from rag.core.db.vectors import SHAMMY_FILE, VectorStoreError, shammy_module

    path = root / SHAMMY_FILE
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if not path.is_file():
        _problem(problems, "error", f"{SHAMMY_FILE} is missing (the workspace keeps its vectors there)")
        return 0, n_chunks
    try:
        sh = shammy_module()
    except VectorStoreError as exc:
        _problem(problems, "error", str(exc))
        return 0, 0
    try:
        rep = sh.verify(str(path))
        for p in rep["problems"]:
            _problem(problems, "error", f"{SHAMMY_FILE}: {p}")
        if rep["uncommitted_bytes"]:
            _problem(problems, "warning", f"{SHAMMY_FILE} ends with an interrupted write; the next build removes it")
        if not rep["ok"]:
            return int(rep["live_rows"]), 0
        ix = sh.Index.open(path)
        try:
            ids = ix.ids()[0]
        finally:
            ix.close()
    except (OSError, ValueError) as exc:
        _problem(problems, "error", f"{SHAMMY_FILE} can't be read: {exc}")
        return 0, n_chunks
    chunk_ids = np.fromiter((r[0] for r in conn.execute("SELECT chunk_id FROM chunks")), dtype=np.uint64, count=n_chunks)
    stray = int((~np.isin(ids, chunk_ids)).sum())
    if stray:
        _problem(problems, "warning", f"{stray} vectors belong to no chunk; the next build removes them")
    return len(ids) - stray, int((~np.isin(chunk_ids, ids)).sum())


def check_workspace(root: Path, *, pack: bool = False) -> dict[str, Any]:
    from rag.config import Config
    from rag.core.workspace import MARKER, classify, read_marker

    root = Path(root)
    problems: list[dict[str, str]] = []
    counts: dict[str, Any] = {}
    if not root.is_dir():
        return _result(root, "unknown", [{"level": "error", "message": "folder not found"}], counts)

    marker = read_marker(root)
    kind = classify(root)
    if marker is None and not pack:
        if (root / MARKER).exists():
            _problem(problems, "error", f"{MARKER} is damaged (not a workspace marker)")
        elif kind == "legacy":
            _problem(problems, "warning", f"no {MARKER} marker (made by an older version); opening it adds one")
        else:
            _problem(problems, "error", f"no {MARKER} marker")
    name = (marker or {}).get("name") or root.name

    cfg = None
    if not (root / "config.toml").exists():
        _problem(problems, "error", "config.toml is missing")
    else:
        try:
            cfg = Config.load(root / "config.toml")
        except Exception as exc:  # noqa: BLE001 - any parse error is the finding
            _problem(problems, "error", f"config.toml can't be read: {exc}")

    rag_db, tables_db = root / "rag.db", root / "tables.db"
    for db in (rag_db, tables_db):
        if not db.exists():
            _problem(problems, "error", f"{db.name} is missing")
    if not rag_db.exists():
        return _result(root, "workspace", problems, counts, name)

    try:
        conn = _ro(rag_db, load_vec=True)
    except Exception as exc:  # noqa: BLE001
        _problem(problems, "error", f"rag.db can't be opened: {exc}")
        return _result(root, "workspace", problems, counts, name)
    conn.row_factory = sqlite3.Row
    q = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    try:
        integrity = conn.execute("PRAGMA quick_check").fetchone()[0]
        if integrity != "ok":
            _problem(problems, "error", f"rag.db is corrupt: {integrity}")
            return _result(root, "workspace", problems, counts, name)
        meta = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM meta")}
        if meta.get("schema_version") != str(SCHEMA_VERSION):
            _problem(problems, "error", f"database schema {meta.get('schema_version')!r}; this version reads "
                                        f"{SCHEMA_VERSION}")
        counts = {"documents": q("SELECT COUNT(*) FROM documents"),
                  "failed_documents": q("SELECT COUNT(*) FROM documents WHERE status = 'failed'"),
                  "chunks": q("SELECT COUNT(*) FROM chunks"), "keyword_rows": q("SELECT COUNT(*) FROM chunks_fts")}
        backend = meta.get("vector_backend") or "sqlite-vec"
        counts["vector_store"] = backend
        missing_vec = 0
        if backend == "shammy":
            counts["vectors"], missing_vec = _check_shammy(root, conn, problems)
        else:
            counts["vectors"] = q("SELECT COUNT(*) FROM chunk_vec")
            missing_vec = q("SELECT COUNT(*) FROM chunks WHERE chunk_id NOT IN (SELECT chunk_id FROM chunk_vec)")
        if not counts["chunks"] == counts["keyword_rows"] == counts["vectors"]:
            _problem(problems, "error", f"{counts['chunks']} chunks but {counts['keyword_rows']} keyword index rows "
                                        f"and {counts['vectors']} vectors (rebuild with --force)")
        orphans = q("SELECT COUNT(*) FROM chunks WHERE doc_id NOT IN (SELECT doc_id FROM documents)")
        if orphans:
            _problem(problems, "error", f"{orphans} chunks belong to no document")
        if missing_vec:
            _problem(problems, "error", f"{missing_vec} chunks have no vector")
        wrong_n = q("SELECT COUNT(*) FROM documents d WHERE n_chunks != (SELECT COUNT(*) FROM chunks c "
                    "WHERE c.doc_id = d.doc_id)")
        if wrong_n:
            _problem(problems, "warning", f"{wrong_n} documents have a stale chunk count")

        if counts["chunks"]:
            if not meta.get("embed_model"):
                _problem(problems, "error", "chunks exist but the embedding settings were never recorded")
            vec_sql = q("SELECT sql FROM sqlite_master WHERE name = 'chunk_vec'") or ""
            m = re.search(r"float\[(\d+)\]", vec_sql)
            if m and meta.get("embed_dims") and m.group(1) != meta["embed_dims"]:
                _problem(problems, "error", f"vectors are stored with {m.group(1)} dimensions but the model "
                                            f"made {meta['embed_dims']}")
            if cfg is not None and meta.get("embed_dims") and str(cfg["embedding"]["dims"]) != meta["embed_dims"]:
                _problem(problems, "warning", f"config.toml says {cfg['embedding']['dims']} dimensions, the "
                                              f"database was built with {meta['embed_dims']}; the next build will refuse")
        counts["embedding"] = {k: meta.get(k) for k in ("embed_backend", "embed_model", "embed_dims")}

        missing_assets = [r[0] for r in conn.execute("SELECT DISTINCT asset_path FROM chunks WHERE asset_path IS NOT NULL")
                          if not (root / r[0]).is_file()]
        if missing_assets:
            _problem(problems, "warning", f"{len(missing_assets)} figure images are missing, e.g. {missing_assets[0]}")
        if not pack:
            missing_lib = [r[0] for r in conn.execute("SELECT library_path FROM documents WHERE library_path IS NOT NULL")
                           if not (root / r[0]).is_file()]
            if missing_lib:
                _problem(problems, "warning", f"{len(missing_lib)} copies of original files are missing from library/")

        table_refs = {r[0]: r[1] for r in conn.execute(
            "SELECT table_name, MAX(kind = 'table_card') FROM chunks WHERE table_name IS NOT NULL GROUP BY table_name")}
        cards = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT chunk_id, table_name, doc_id FROM chunks WHERE kind = 'table_card'")}
    finally:
        conn.close()

    if tables_db.exists():
        try:
            tconn = _ro(tables_db, load_vec=False)
            tconn.row_factory = sqlite3.Row
            try:
                integrity = tconn.execute("PRAGMA quick_check").fetchone()[0]
                if integrity != "ok":
                    _problem(problems, "error", f"tables.db is corrupt: {integrity}")
                else:
                    catalog = tconn.execute("SELECT table_name, doc_id, card_chunk_id, n_rows FROM _catalog").fetchall()
                    real = {r[0] for r in tconn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
                    counts["tables"] = len(catalog)
                    for row in catalog:
                        t = row["table_name"]
                        if t not in real:
                            _problem(problems, "error", f"table {t} is listed but doesn't exist")
                            continue
                        card = cards.get(row["card_chunk_id"])
                        if card is None or card[0] != t:
                            _problem(problems, "warning", f"table {t} has no description chunk, so search won't "
                                                          "suggest it")
                    for t in table_refs:
                        if t not in real:
                            _problem(problems, "error", f"chunks point at table {t}, which doesn't exist")
            finally:
                tconn.close()
        except Exception as exc:  # noqa: BLE001
            _problem(problems, "error", f"tables.db can't be opened: {exc}")
    return _result(root, "pack" if pack else "workspace", problems, counts, name)


def format_report(res: dict[str, Any]) -> str:
    c = res.get("counts", {})
    head = f"{res.get('name') or res['path']} ({res['kind']}): {'OK' if res['ok'] else 'PROBLEMS FOUND'}"
    lines = [head, f"  {res['path']}"]
    if res.get("checksums"):
        lines.append(f"  checksums: {res['checksums']}")
    if c.get("documents") is not None:
        emb = c.get("embedding") or {}
        plural = lambda n, w: f"{n} {w}" + ("" if n == 1 else "s")  # noqa: E731
        lines.append(f"  {plural(c['documents'], 'document')} ({c.get('failed_documents', 0)} failed), "
                     f"{plural(c.get('chunks', 0), 'chunk')}, {plural(c.get('tables', 0), 'table')}" + (f", {emb.get('embed_model')} ({emb.get('embed_dims')} dims)"
                                                     if emb.get("embed_model") else "")
                     + (", vectors in Shammy" if c.get("vector_store") == "shammy" else ""))
    for p in res["problems"]:
        lines.append(f"  {p['level']}: {p['message']}")
    return "\n".join(lines)


def to_json(res: dict[str, Any]) -> str:
    return json.dumps(res, default=str)
