"""Decide the pre-release feed: the new release's latest.json, unless the feed already offers a
newer version (publishing an older hotfix after a pre-release must not move the feed back).

    python desktop/scripts/update_feed.py --new latest.json --current prerelease.json --out prerelease.json

Exits 0 when --out was written, 1 when the feed should stay as it is.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def version_key(version: str) -> tuple:
    """Semantic-version order: 0.4.3-beta.1 < 0.4.3-beta.2 < 0.4.3-rc.1 < 0.4.3 < 0.4.4."""
    v = version.strip().lstrip("v")
    core, _, pre = v.partition("-")
    nums = tuple(int(p) for p in re.findall(r"\d+", core)[:3])
    nums += (0,) * (3 - len(nums))
    if not pre:
        return nums + (1,)
    parts = tuple((0, int(p), "") if p.isdigit() else (1, 0, p) for p in pre.split("."))
    return nums + (0,) + parts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--current")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    new = json.loads(Path(a.new).read_text(encoding="utf-8"))
    if a.current and Path(a.current).is_file():
        cur = json.loads(Path(a.current).read_text(encoding="utf-8"))
        if version_key(cur.get("version", "0")) >= version_key(new.get("version", "0")):
            print(f"the feed already offers {cur.get('version')}; {new.get('version')} is not newer", file=sys.stderr)
            return 1
    Path(a.out).write_text(json.dumps(new, indent=2), encoding="utf-8")
    print(f"{a.out}: {new.get('version')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
