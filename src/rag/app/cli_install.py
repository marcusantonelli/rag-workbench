"""Make the `rag` command that ships with the desktop app usable from a terminal.

* Installed app on Windows: `rag.exe` sits next to the engine. Turning the command on adds
  that folder to the user's PATH (HKCU\\Environment, no admin needed); turning it off
  removes it. New terminals pick it up.
* .deb on Linux: the package installs /usr/bin/rag, so there's nothing to switch.
* AppImage: its files move every launch, so the command can't be linked; use the .deb or pip.
* Development (not frozen): whatever `rag` pip put on the PATH.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Any

EXE = "rag.exe" if sys.platform == "win32" else "rag"
DEB_WRAPPER = Path("/usr/bin/rag")


def bundled_rag() -> Path | None:
    """The rag executable shipped next to the frozen engine, if this is the installed app."""
    if not getattr(sys, "frozen", False):
        return None
    p = Path(sys.executable).resolve().parent / EXE
    return p if p.is_file() else None


def _norm(p: str) -> str:
    return os.path.normcase(os.path.normpath(os.path.expandvars(p.strip().strip('"')))) if p.strip() else ""


def path_add(value: str, folder: str) -> str:
    """PATH string with folder appended (once)."""
    parts = [p for p in value.split(";") if p.strip()]
    if any(_norm(p) == _norm(folder) for p in parts):
        return ";".join(parts)
    return ";".join(parts + [folder])


def path_remove(value: str, folder: str) -> str:
    return ";".join(p for p in value.split(";") if p.strip() and _norm(p) != _norm(folder))


def path_has(value: str, folder: str) -> bool:
    return any(_norm(p) == _norm(folder) for p in value.split(";"))


# -- Windows user PATH ---------------------------------------------------------------------------
def _read_user_path() -> tuple[str, int]:
    import winreg  # type: ignore[import-not-found]

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
        try:
            value, kind = winreg.QueryValueEx(key, "Path")
            return str(value), kind
        except FileNotFoundError:
            return "", winreg.REG_EXPAND_SZ


def _write_user_path(value: str, kind: int) -> None:
    import ctypes
    import winreg  # type: ignore[import-not-found]

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, "Path", 0, kind, value)
    # Tell Explorer (and new terminals it starts) that the environment changed.
    HWND_BROADCAST, WM_SETTINGCHANGE, SMTO_ABORTIFHUNG = 0xFFFF, 0x001A, 0x0002
    result = ctypes.c_ulong()
    ctypes.windll.user32.SendMessageTimeoutW(HWND_BROADCAST, WM_SETTINGCHANGE, 0, "Environment",  # type: ignore[attr-defined]
                                             SMTO_ABORTIFHUNG, 5000, ctypes.byref(result))


def status() -> dict[str, Any]:
    exe = bundled_rag()
    if exe is None:
        found = shutil.which("rag")
        return {"mode": "development", "available": found is not None, "on_path": found is not None,
                "can_toggle": False, "path": found,
                "note": "Using the rag command from your Python environment." if found else
                        "Install rag-builder with pip to get the rag command."}
    if sys.platform == "win32":
        user_path, _ = _read_user_path()
        on = path_has(user_path, str(exe.parent))
        return {"mode": "windows", "available": True, "on_path": on, "can_toggle": True, "path": str(exe),
                "note": "Open a new terminal after turning this on." if on else
                        "Adds the app's folder to your user PATH; no admin rights needed."}
    if os.environ.get("APPIMAGE"):
        return {"mode": "appimage", "available": True, "on_path": False, "can_toggle": False, "path": str(exe),
                "note": "The AppImage can't add commands to the PATH. Install the .deb, or "
                        "pip install rag-builder, to use rag in a terminal."}
    on = DEB_WRAPPER.exists() or shutil.which("rag") is not None
    return {"mode": "linux", "available": True, "on_path": on, "can_toggle": False, "path": str(exe),
            "note": "Installed with the package as /usr/bin/rag." if DEB_WRAPPER.exists() else
                    f"Run it as {exe}, or link it into a folder on your PATH."}


def set_enabled(enabled: bool) -> dict[str, Any]:
    exe = bundled_rag()
    if exe is None or sys.platform != "win32":
        raise RuntimeError("The rag command can only be switched on and off in the installed Windows app.")
    value, kind = _read_user_path()
    folder = str(exe.parent)
    new = path_add(value, folder) if enabled else path_remove(value, folder)
    if new != value:
        _write_user_path(new, kind)
    # This process (and the llama-server it starts) see the change too.
    os.environ["PATH"] = path_add(os.environ.get("PATH", ""), folder) if enabled else \
        path_remove(os.environ.get("PATH", ""), folder)
    return status()
