"""What every extractor receives besides the file path."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from rag.config import Config


@dataclass
class ExtractContext:
    config: Config
    sha256: str
    display_path: str
    assets_root: Path                       # <workspace>/assets
    progress: Callable[[str, dict], None] = field(default=lambda stage, info: None)

    @property
    def asset_dir(self) -> Path:
        d = self.assets_root / self.sha256[:16]
        d.mkdir(parents=True, exist_ok=True)
        return d

    def asset_rel(self, path: Path) -> str:
        return path.relative_to(self.assets_root.parent).as_posix()


class UnsupportedFile(RuntimeError):
    """The file type is recognized but cannot be read here (message says why)."""
