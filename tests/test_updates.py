"""The updater manifest the release workflow writes, and the version comparison."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from rag.updates import newer

ROOT = Path(__file__).resolve().parents[1]


def _script():
    spec = importlib.util.spec_from_file_location("make_latest_json", ROOT / "desktop" / "scripts" / "make_latest_json.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_versions_compare_as_numbers():
    assert newer("0.4.2", "0.4.1") and newer("0.10.0", "0.9.9") and newer("v1.0.0", "0.9.9")
    assert not newer("0.4.1", "0.4.1") and not newer("0.4.0", "0.4.1")


def test_latest_json_names_files_as_github_serves_them(tmp_path):
    dist = tmp_path / "dist"
    for rel, sig in [("nsis/RAG Workbench_0.4.1_x64-setup.exe", "SIG-NSIS"), ("msi/RAG Workbench_0.4.1_x64_en-US.msi", "SIG-MSI"),
                     ("appimage/RAG Workbench_0.4.1_amd64.AppImage", "SIG-APP")]:
        f = dist / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
        Path(str(f) + ".sig").write_text(sig + "\n")
    (dist / "deb").mkdir()
    (dist / "deb" / "rag-workbench_0.4.1_amd64.deb").write_bytes(b"x")         # no signature: not offered
    notes = tmp_path / "notes.md"
    notes.write_text("# v0.4.1\n\nProcedures.")
    out = tmp_path / "latest.json"
    assert _script().main(["--dist", str(dist), "--repo", "owner/repo", "--tag", "v0.4.1", "--notes", str(notes),
                           "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["version"] == "0.4.1" and doc["notes"].startswith("# v0.4.1")
    p = doc["platforms"]
    assert p["windows-x86_64"]["url"] == "https://github.com/owner/repo/releases/download/v0.4.1/RAG.Workbench_0.4.1_x64-setup.exe"
    assert p["windows-x86_64"]["signature"] == "SIG-NSIS" and p["windows-x86_64-msi"]["signature"] == "SIG-MSI"
    assert p["linux-x86_64"]["signature"] == "SIG-APP"


def test_no_signed_installers_means_no_update(tmp_path):
    (tmp_path / "d").mkdir()
    assert _script().main(["--dist", str(tmp_path / "d"), "--repo", "o/r", "--tag", "v1.0.0", "--out", str(tmp_path / "l.json")]) == 1
