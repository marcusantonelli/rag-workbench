"""Freeze the Python engine into desktop/engine-dist/rag-engine/ for the installer.

Run from any folder with the Python that has rag-builder installed:
    python desktop/scripts/build_engine.py
then smoke-test it (it checks itself: version, sqlite-vec, a tiny build and search).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
SPEC = DESKTOP / "engine" / "rag-engine.spec"
DIST = DESKTOP / "engine-dist"
WORK = DESKTOP / "engine-build"


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller>=6.10"])
    shutil.rmtree(DIST / "rag-engine", ignore_errors=True)
    subprocess.check_call([sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
                           "--distpath", str(DIST), "--workpath", str(WORK), str(SPEC)], cwd=SPEC.parent)
    exe = DIST / "rag-engine" / ("rag-engine.exe" if os.name == "nt" else "rag-engine")
    print(f"built {exe}")
    return smoke(exe)


def smoke(exe: Path) -> int:
    """Start the frozen engine over stdio and run a tiny build + search with the test embedder."""
    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        docs.mkdir()
        (docs / "note.md").write_text("# Leak test\n\nHelium leak rate limit is 5e-8 atm cc/s for TO-39 packages.\n")
        env = {**os.environ, "RAG_WORKBENCH_HOME": str(Path(tmp) / "home")}
        proc = subprocess.Popen([str(exe), "--stdio"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True, env=env, encoding="utf-8")
        assert proc.stdin and proc.stdout

        def call(rid: int, method: str, params: dict) -> dict:
            proc.stdin.write(json.dumps({"id": rid, "method": method, "params": params}) + "\n")
            proc.stdin.flush()
            while True:
                line = proc.stdout.readline()
                if not line:
                    raise SystemExit("engine exited: " + (proc.stderr.read() if proc.stderr else ""))
                msg = json.loads(line)
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise SystemExit(f"{method} failed: {msg['error']}")
                    return msg["result"]

        try:
            ready = json.loads(proc.stdout.readline())
            print("engine", ready["data"]["version"], "frozen:", ready["data"]["frozen"])
            kb = call(1, "kb.create", {"name": "smoke", "dims": 64, "backend": "hash-test"})
            job = call(2, "kb.build", {"kb": kb["id"], "inputs": [str(docs)]})
            while True:
                j = call(3, "jobs.get", {"job": job["id"]})
                if j["state"] != "running":
                    break
            if j["state"] != "done":
                raise SystemExit(f"build failed: {j['error']}")
            hits = call(4, "kb.search", {"kb": kb["id"], "query": "helium leak rate"})["results"]
            assert hits and "5e-8" in hits[0]["text"], hits
            status = call(5, "runtime.status", {})
            print("smoke test passed; hardware:", status["hardware"]["cpu"], status["hardware"]["gpus"])
            call(6, "app.shutdown", {})
            proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
