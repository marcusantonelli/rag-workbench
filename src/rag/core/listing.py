"""Find the workspaces and packs in a folder (by default the desktop app's workspaces folder)."""

from __future__ import annotations

import sqlite3
import zipfile
from pathlib import Path
from typing import Any


def default_folder() -> Path:
    """The desktop app's workspaces folder (its setting if changed), without creating anything."""
    from rag.app.paths import Settings, data_dir

    root = data_dir()
    configured = Settings(root).data.get("workspaces_dir")
    return Path(configured) if configured else root / "workspaces"


def _size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def describe(path: Path, *, sizes: bool = True) -> dict[str, Any] | None:
    """One row for a workspace or pack, or None if the path is neither. Reads without changing anything."""
    from rag.core.workspace import classify, read_marker

    path = Path(path)
    kind = classify(path)
    if kind is None:
        return None
    row: dict[str, Any] = {"path": str(path), "kind": "pack" if kind == "pack" else "workspace",
                           "name": path.stem if kind == "pack" else path.name, "marker": kind == "workspace",
                           "documents": None, "chunks": None, "tables": None, "model": None, "dims": None,
                           "updated": None, "size_bytes": _size(path) if sizes else None, "note": ""}
    if kind == "pack":
        from rag.core.db.pack import PackError, read_manifest

        try:
            m = read_manifest(path)
        except (PackError, zipfile.BadZipFile, OSError, ValueError):
            return None
        c, e = m.get("counts", {}), m.get("embedding", {})
        row.update(name=m.get("name") or path.stem, documents=c.get("documents"), chunks=c.get("chunks"),
                   tables=c.get("tables"), model=e.get("model"), dims=e.get("dims"), updated=m.get("built_at"))
        return row

    marker = read_marker(path) or {}
    row["name"] = marker.get("name") or path.name
    if kind == "legacy":
        row["note"] = "no marker (older workspace)"
    try:
        conn = sqlite3.connect(f"{(path / 'rag.db').resolve().as_uri()}?mode=ro", uri=True)
        try:
            q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
            meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
            row.update(documents=q("SELECT COUNT(*) FROM documents"), chunks=q("SELECT COUNT(*) FROM chunks"),
                       model=meta.get("embed_model"), dims=int(meta["embed_dims"]) if meta.get("embed_dims") else None,
                       updated=q("SELECT MAX(ingested_at) FROM documents") or marker.get("created_at"))
        finally:
            conn.close()
        if (path / "tables.db").exists():
            t = sqlite3.connect(f"{(path / 'tables.db').resolve().as_uri()}?mode=ro", uri=True)
            try:
                row["tables"] = t.execute("SELECT COUNT(*) FROM _catalog").fetchone()[0]
            finally:
                t.close()
    except sqlite3.Error as exc:
        row["note"] = f"can't read: {exc}"
    if row["documents"] == 0:
        row["note"] = row["note"] or "empty"
    return row


def list_knowledge_bases(folder: Path, *, packs: bool = True, sizes: bool = True) -> list[dict[str, Any]]:
    """Workspaces directly inside folder (and folder itself, if it is one), plus .ragpack files there."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    rows = []
    candidates = [folder] + sorted(p for p in folder.iterdir() if p.is_dir() or p.suffix.lower() == ".ragpack")
    for p in candidates:
        if p.is_file() and not packs:
            continue
        row = describe(p, sizes=sizes)
        if row:
            rows.append(row)
    return rows
