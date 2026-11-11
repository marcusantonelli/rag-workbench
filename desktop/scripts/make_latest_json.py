"""Write latest.json, the file the app's updater reads from the newest GitHub release.

    python desktop/scripts/make_latest_json.py --dist DIR --repo OWNER/NAME --tag v0.4.1 \
        --notes docs/release-notes/0.4.1.md --out latest.json

DIR holds the installers built with updater signatures (NSIS .exe, .msi and .AppImage, each with
its .sig beside it). Each platform's entry names the file as GitHub serves it from that release
(spaces in asset names become dots) and carries the signature the app checks against its key.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

KINDS = [  # (suffix, updater platform keys)
    (".exe", ["windows-x86_64-nsis", "windows-x86_64"]),
    (".msi", ["windows-x86_64-msi"]),
    (".AppImage", ["linux-x86_64-appimage", "linux-x86_64"]),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--notes", help="release notes (Markdown); shown in the app before updating")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    base = f"https://github.com/{a.repo}/releases/download/{a.tag}"
    platforms: dict[str, dict[str, str]] = {}
    for suffix, keys in KINDS:
        for sig in sorted(Path(a.dist).rglob(f"*{suffix}.sig")):
            installer = sig.with_name(sig.name[: -len(".sig")])
            if not installer.is_file():
                continue
            entry = {"signature": sig.read_text(encoding="utf-8").strip(),
                     "url": f"{base}/{installer.name.replace(' ', '.')}"}
            for key in keys:
                platforms.setdefault(key, entry)
    if not platforms:
        print("no signed installers found (is TAURI_SIGNING_PRIVATE_KEY set for the build?)", file=sys.stderr)
        return 1
    notes = ""
    if a.notes and Path(a.notes).is_file():
        notes = Path(a.notes).read_text(encoding="utf-8").strip()
        if len(notes) > 6000:
            notes = notes[:6000].rsplit("\n", 1)[0] + "\n\n…"
    doc = {"version": a.tag.lstrip("v"), "notes": notes,
           "pub_date": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "platforms": platforms}
    Path(a.out).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(f"{a.out}: {', '.join(sorted(platforms))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
