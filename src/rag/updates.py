"""Whether a newer release exists (GitHub releases of the project). The desktop app installs
updates itself; the command line only reports them."""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

from rag import __version__

REPO = "marcusantonelli/rag-workbench"
API = f"https://api.github.com/repos/{REPO}/releases"
FEED_TAG = "update-feed"      # holds the pre-release channel's feed; not a release


def version_key(version: str) -> tuple:
    """Semantic-version order: 0.4.3-beta.1 < 0.4.3-beta.2 < 0.4.3-rc.1 < 0.4.3 < 0.4.4."""
    v = version.strip().lstrip("v")
    core, _, pre = v.partition("-")
    nums = tuple(int(p) for p in re.findall(r"\d+", core)[:3])
    nums += (0,) * (3 - len(nums))
    if not pre:
        return nums + (1,)
    return nums + (0,) + tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))


def is_prerelease(version: str) -> bool:
    return "-" in version.strip().lstrip("v")


def newer(candidate: str, current: str = __version__) -> bool:
    return version_key(candidate) > version_key(current)


def _get(url: str, timeout: float) -> Any:
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": f"rag-builder/{__version__}"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _out(data: dict[str, Any]) -> dict[str, Any]:
    tag = str(data.get("tag_name") or "")
    return {"version": tag.lstrip("v"), "tag": tag, "url": data.get("html_url"), "name": data.get("name"),
            "prerelease": bool(data.get("prerelease"))}


def latest(timeout: float = 10.0, prerelease: bool = False) -> dict[str, Any]:
    """{"version", "tag", "url", "name", "prerelease"} of the newest published release; with
    ``prerelease``, the newest of normal releases and pre-releases."""
    if not prerelease:
        return _out(_get(f"{API}/latest", timeout))
    found = [r for r in _get(f"{API}?per_page=30", timeout)
             if not r.get("draft") and r.get("tag_name") and r.get("tag_name") != FEED_TAG]
    if not found:
        return _out(_get(f"{API}/latest", timeout))
    return _out(max(found, key=lambda r: version_key(str(r["tag_name"]))))
