"""A workspace is the Builder's working folder for one pack.

Layout::

    <workspace>/
      .ragworkspace  marker: format, id, name, creation (hidden on Windows)
      config.toml
      rag.db  tables.db
      cache/     normalized extractions, by source SHA-256
      assets/    figure crops
      library/   copies of the source files (optional)
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.config import DEFAULTS, Config, render_toml
from rag.core.db.store import RagStore
from rag.core.db.tables import TablesStore


MARKER = ".ragworkspace"
MARKER_FORMAT = "rag-workspace"


class NotAWorkspace(FileNotFoundError):
    pass


def _set_hidden(path: Path, hidden: bool) -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))  # type: ignore[attr-defined]
        if attrs == -1 or attrs == 0xFFFFFFFF:
            return
        attrs = attrs | 0x2 if hidden else attrs & ~0x2
        ctypes.windll.kernel32.SetFileAttributesW(str(path), attrs)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - hiding the marker is cosmetic
        pass


def read_marker(root: Path) -> dict[str, Any] | None:
    """The marker's contents, or None when it is missing or isn't a workspace marker."""
    try:
        data = json.loads((Path(root) / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("format") == MARKER_FORMAT else None


def write_marker(root: Path, **fields: Any) -> dict[str, Any]:
    """Create or update the marker (keeping its id and creation time)."""
    from rag import SCHEMA_VERSION, __version__
    from rag.core.db.store import now_iso

    root = Path(root)
    data = read_marker(root) or {"format": MARKER_FORMAT, "version": 1, "id": str(uuid.uuid4()),
                                 "name": root.name, "created_at": now_iso(), "created_with": f"rag {__version__}",
                                 "schema_version": SCHEMA_VERSION}
    data.update({k: v for k, v in fields.items() if v is not None})
    path = root / MARKER
    tmp = root / (MARKER + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    _set_hidden(path, False)          # Windows won't replace a hidden file
    os.replace(tmp, path)
    _set_hidden(path, True)
    return data


def _has_rag_schema(db: Path) -> bool:
    """rag.db is an SQLite database with this program's tables (checked without loading sqlite-vec)."""
    try:
        with open(db, "rb") as f:
            if f.read(16) != b"SQLite format 3\x00":
                return False
        conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
        try:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return False
    return {"meta", "documents", "chunks"} <= names


def classify(path: Path) -> str | None:
    """'workspace' (has a marker), 'legacy' (made before markers existed), 'pack', or None."""
    path = Path(path)
    if path.is_file():
        return "pack" if path.suffix.lower() == ".ragpack" or zipfile.is_zipfile(path) else None
    if not path.is_dir():
        return None
    if read_marker(path) is not None:
        return "workspace"
    if (path / "config.toml").exists() and (path / "rag.db").exists() and _has_rag_schema(path / "rag.db"):
        return "legacy"
    return None


@dataclass
class Workspace:
    root: Path
    config: Config

    @property
    def rag_db(self) -> Path:
        return self.root / "rag.db"

    @property
    def tables_db(self) -> Path:
        return self.root / "tables.db"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def library_dir(self) -> Path:
        return self.root / "library"

    @property
    def config_path(self) -> Path:
        return self.root / "config.toml"

    @property
    def marker(self) -> dict[str, Any]:
        return read_marker(self.root) or {}

    @property
    def name(self) -> str:
        return self.marker.get("name") or self.root.name

    @classmethod
    def init(cls, root: Path, overrides: dict | None = None, *, name: str | None = None) -> "Workspace":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        cfg_path = root / "config.toml"
        if not cfg_path.exists():
            cfg = Config().with_overrides(overrides or {})
            cfg_path.write_text(render_toml(cfg.data), encoding="utf-8")
        ws = cls(root=root, config=Config.load(cfg_path))
        for d in (ws.cache_dir, ws.assets_dir, ws.library_dir):
            d.mkdir(exist_ok=True)
        RagStore.create(ws.rag_db, int(ws.config["embedding"]["dims"])).close()
        TablesStore.create(ws.tables_db).close()
        write_marker(root, name=name)
        return ws

    @classmethod
    def open(cls, root: Path) -> "Workspace":
        """Open a workspace. One made before markers existed gets a marker (when the folder is writable)."""
        root = Path(root)
        kind = classify(root)
        if kind == "legacy":
            try:
                write_marker(root)
            except OSError:
                pass
        elif kind != "workspace":
            if (root / "rag.db").exists() or (root / "config.toml").exists():
                raise NotAWorkspace(f"{root} is not a usable workspace: rag.db or config.toml is missing or "
                                    "isn't one of rag's (run `rag verify` on it for details)")
            raise NotAWorkspace(f"{root} is not a workspace (run `rag init {root}` first)")
        if not (root / "config.toml").exists():
            raise NotAWorkspace(f"{root} is damaged: config.toml is missing")
        if not (root / "rag.db").exists():
            raise NotAWorkspace(f"{root} is damaged: rag.db is missing")
        return cls(root=root, config=Config.load(root / "config.toml"))


def open_any(path: Path) -> tuple[Workspace, tempfile.TemporaryDirectory | None]:
    """Open a workspace folder, or unpack a .ragpack to a temp folder."""
    path = Path(path)
    if path.is_file() and zipfile.is_zipfile(path):
        from rag.core.db.pack import unpack_pack

        tmp = tempfile.TemporaryDirectory(prefix="ragpack_")
        unpack_pack(path, Path(tmp.name))
        return Workspace.open(Path(tmp.name)), tmp
    return Workspace.open(path), None


def reset_workspace_data(ws: Workspace) -> None:
    """Delete databases, cache and assets (keeps config and library)."""
    for p in (ws.rag_db, ws.tables_db):
        for suffix in ("", "-wal", "-shm"):
            q = Path(str(p) + suffix)
            if q.exists():
                q.unlink()
    for d in (ws.cache_dir, ws.assets_dir):
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(exist_ok=True)
    RagStore.create(ws.rag_db, int(ws.config["embedding"]["dims"])).close()
    TablesStore.create(ws.tables_db).close()


__all__ = ["Workspace", "open_any", "reset_workspace_data", "DEFAULTS", "MARKER", "NotAWorkspace", "classify",
           "read_marker", "write_marker"]
