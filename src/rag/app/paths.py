"""Where the desktop app keeps its data, and its persisted settings."""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

APP_DIR_NAME = "RAG Workbench"
APP_ID = "com.ragworkbench.desktop"   # matches the Tauri identifier


def data_dir() -> Path:
    """Per-user data folder: models, llama.cpp runtime, opened packs, workspaces, settings.

    RAG_WORKBENCH_HOME overrides it (tests, portable installs).
    """
    override = os.environ.get("RAG_WORKBENCH_HOME")
    if override:
        root = Path(override)
    elif sys.platform == "win32":
        # Not "%LOCALAPPDATA%\RAG Workbench": the per-user installer puts the program there, and
        # an uninstall must not take the models and workspaces with it. The installer's
        # "Delete the application data" option removes this folder (named by the app id).
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP_ID
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    else:
        root = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "rag-workbench"
    root.mkdir(parents=True, exist_ok=True)
    return root


DEFAULT_SETTINGS: dict[str, Any] = {
    "runtime": {
        "mode": "managed",            # managed: the app runs llama-server | external: use runtime.url
        "url": "http://127.0.0.1:8081",
        "accel": "auto",              # auto | gpu | cpu  (cpu = the "Force CPU" switch)
        "variant": None,              # llama.cpp build in use: cuda-12.4 | cuda-13.5 | vulkan | cpu
        "llama_build": None,          # e.g. b12207; None = the app's pinned default
        "model_path": None,           # GGUF file the managed server loads
        "model_id": "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",   # recorded in packs built with it
        "ctx": 8192,
        "port": 0,                    # 0 = pick a free port
        "autostart": True,
    },
    "vision": {
        "enabled": False,             # describe pictures during builds (runs a second llama-server)
        "model_path": None,
        "mmproj_path": None,          # the model's image encoder
        "model_id": None,
        "ctx": 8192,
        "port": 0,                    # 0 = 8083, or a free one
    },
    "workspaces_dir": None,           # default: <data>/workspaces
    "recent": [],                     # [{path, kind, name, opened_at}]
}


class Settings:
    """settings.json in the data folder; every change is written immediately."""

    def __init__(self, root: Path | None = None):
        self.root = root or data_dir()
        self.path = self.root / "settings.json"
        self._lock = threading.Lock()
        self.data = json.loads(json.dumps(DEFAULT_SETTINGS))
        if self.path.exists():
            try:
                saved = json.loads(self.path.read_text(encoding="utf-8"))
                for key, value in saved.items():
                    if isinstance(value, dict) and isinstance(self.data.get(key), dict):
                        self.data[key].update(value)
                    else:
                        self.data[key] = value
            except (OSError, ValueError):
                pass   # a damaged settings file falls back to defaults

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
            tmp.replace(self.path)

    @property
    def runtime(self) -> dict[str, Any]:
        return self.data["runtime"]

    def update_runtime(self, **changes: Any) -> dict[str, Any]:
        self.data["runtime"].update(changes)
        self.save()
        return self.data["runtime"]

    @property
    def vision(self) -> dict[str, Any]:
        return self.data["vision"]

    def update_vision(self, **changes: Any) -> dict[str, Any]:
        self.data["vision"].update(changes)
        self.save()
        return self.data["vision"]

    @property
    def workspaces_dir(self) -> Path:
        d = Path(self.data.get("workspaces_dir") or self.root / "workspaces")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def packs_dir(self) -> Path:
        d = self.root / "packs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def remember(self, path: str, kind: str, name: str, when: str) -> None:
        recent = [r for r in self.data.get("recent", []) if r.get("path") != path]
        recent.insert(0, {"path": path, "kind": kind, "name": name, "opened_at": when})
        self.data["recent"] = recent[:30]
        self.save()

    def forget(self, path: str) -> None:
        self.data["recent"] = [r for r in self.data.get("recent", []) if r.get("path") != path]
        self.save()
