"""A workspace is the Builder's working folder for one pack.

Layout::

    <workspace>/
      config.toml
      rag.db  tables.db
      cache/     normalized extractions, by source SHA-256
      assets/    figure crops
      library/   copies of the source files (optional)
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

from rag.config import DEFAULTS, Config, render_toml
from rag.core.db.store import RagStore
from rag.core.db.tables import TablesStore


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

    @classmethod
    def init(cls, root: Path, overrides: dict | None = None) -> "Workspace":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        cfg_path = root / "config.toml"
        if not cfg_path.exists():
            cfg = Config().with_overrides(overrides or {})
            cfg_path.write_text(render_toml(cfg.data), encoding="utf-8")
        ws = cls.open(root)
        for d in (ws.cache_dir, ws.assets_dir, ws.library_dir):
            d.mkdir(exist_ok=True)
        RagStore.create(ws.rag_db, int(ws.config["embedding"]["dims"])).close()
        TablesStore.create(ws.tables_db).close()
        return ws

    @classmethod
    def open(cls, root: Path) -> "Workspace":
        root = Path(root)
        if not (root / "config.toml").exists() and not (root / "rag.db").exists():
            raise FileNotFoundError(f"{root} is not a workspace (run `rag init {root}` first)")
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


__all__ = ["Workspace", "open_any", "reset_workspace_data", "DEFAULTS"]
