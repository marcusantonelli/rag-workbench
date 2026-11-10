""".ragpack export, verification and unpacking.

A .ragpack is a zip holding consistent snapshots of rag.db and tables.db
(made with VACUUM INTO), vectors.shammy when the workspace keeps its vectors there, figure crops, the workspace config, optionally the
original files, and manifest.json with a SHA-256 for every file.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from rag import PACK_FORMAT, PACK_FORMAT_SHAMMY, PACK_FORMATS, SCHEMA_VERSION, __version__
from rag.core.db.store import RagStore, now_iso
from rag.core.util import sha256_file


class PackError(RuntimeError):
    pass


def _snapshot(src: Path, dest: Path, *, load_vec: bool) -> None:
    from rag.core.db.connect import connect, sqlite_module

    conn = connect(src, readonly=True, load_vec=load_vec)
    try:
        conn.execute("VACUUM INTO ?", (str(dest),))
    finally:
        conn.close()
    out = sqlite_module().connect(dest)
    try:
        out.execute("PRAGMA journal_mode = DELETE")  # read-only consumers need no -wal/-shm files
    finally:
        out.close()


def export_pack(ws, out_path: Path, *, include_originals: bool = False, name: str | None = None) -> dict[str, Any]:
    out_path = Path(out_path)
    store = RagStore(ws.rag_db, readonly=True)
    try:
        meta = store.meta()
        counts = store.counts()
        backend = store.vector_backend
    finally:
        store.close()
    from rag.core.db.tables import TablesStore

    tstore = TablesStore(ws.tables_db, readonly=True)
    try:
        n_tables = len(tstore.catalog())
    finally:
        tstore.close()

    with tempfile.TemporaryDirectory(prefix="ragpack_build_") as tmp:
        tmp_dir = Path(tmp)
        _snapshot(ws.rag_db, tmp_dir / "rag.db", load_vec=True)
        _snapshot(ws.tables_db, tmp_dir / "tables.db", load_vec=False)

        entries: list[tuple[Path, str]] = [
            (tmp_dir / "rag.db", "rag.db"),
            (tmp_dir / "tables.db", "tables.db"),
            (ws.config_path, "config.toml"),
        ]
        if backend == "shammy":
            from rag.core.db.vectors import SHAMMY_FILE, shammy_module

            # a compacted copy: no deleted rows, one read-only file
            shammy_module().compact(str(ws.root / SHAMMY_FILE), str(tmp_dir / SHAMMY_FILE))
            entries.insert(1, (tmp_dir / SHAMMY_FILE, SHAMMY_FILE))
        for folder, include in ((ws.assets_dir, True), (ws.library_dir, include_originals)):
            if include and folder.exists():
                for f in sorted(folder.rglob("*")):
                    if f.is_file():
                        entries.append((f, str(PurePosixPath(folder.name, *f.relative_to(folder).parts))))

        manifest = {
            "format": PACK_FORMAT_SHAMMY if backend == "shammy" else PACK_FORMAT,
            "vector_store": backend,
            "pack_id": str(uuid.uuid4()),
            "name": name or ws.name,
            "built_at": now_iso(),
            "app_version": __version__,
            "schema_version": SCHEMA_VERSION,
            "embedding": {
                "backend": meta.get("embed_backend"),
                "model": meta.get("embed_model"),
                "dims": int(meta["embed_dims"]) if meta.get("embed_dims") else None,
                "query_instruction": meta.get("query_instruction"),
            },
            "counts": {
                "documents": counts["documents"],
                "failed_documents": counts["failed"],
                "chunks": counts["chunks"],
                "tables": n_tables,
                "chunks_by_kind": counts["chunks_by_kind"],
            },
            "includes_originals": include_originals,
            "files": {arc: sha256_file(src) for src, arc in entries},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(out_path.suffix + ".partial")
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for src, arc in entries:
                zf.write(src, arc)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        partial.replace(out_path)
    return manifest


def read_manifest(pack_path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(pack_path) as zf:
        try:
            return json.loads(zf.read("manifest.json"))
        except KeyError as exc:
            raise PackError("not a .ragpack: manifest.json missing") from exc


def unpack_pack(pack_path: Path, dest: Path) -> dict[str, Any]:
    """Extract and verify every file against the manifest."""
    manifest = read_manifest(pack_path)
    if manifest.get("format") not in PACK_FORMATS:
        raise PackError(f"unsupported pack format: {manifest.get('format')}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pack_path) as zf:
        for member in zf.namelist():
            target = (dest / member).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise PackError(f"unsafe path in pack: {member}")
        zf.extractall(dest)
    bad = [arc for arc, digest in manifest["files"].items()
           if not (dest / arc).exists() or sha256_file(dest / arc) != digest]
    if bad:
        shutil.rmtree(dest, ignore_errors=True)
        raise PackError(f"checksum mismatch in pack: {', '.join(bad[:5])}")
    for d in ("cache", "assets", "library"):
        (dest / d).mkdir(exist_ok=True)
    from rag.core.workspace import write_marker

    write_marker(dest, name=manifest.get("name"), from_pack=manifest.get("pack_id"))
    return manifest
