"""Whether a newer release exists (GitHub releases of the project). The desktop app installs
updates itself; the command line only reports them."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

from rag import __version__

REPO = "marcusantonelli/rag-workbench"
API = f"https://api.github.com/repos/{REPO}/releases/latest"


def _parts(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", version)[:3])


def newer(candidate: str, current: str = __version__) -> bool:
    return _parts(candidate) > _parts(current)


def latest(timeout: float = 10.0) -> dict[str, Any]:
    """{"version", "tag", "url", "name"} of the newest published release."""
    req = urllib.request.Request(API, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": f"rag-builder/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    tag = str(data.get("tag_name") or "")
    return {"version": tag.lstrip("v"), "tag": tag, "url": data.get("html_url"), "name": data.get("name")}
