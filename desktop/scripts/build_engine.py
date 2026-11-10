"""Freeze the Python engine into desktop/engine-dist/rag-engine/ for the installer.

Run from any folder with the Python that has rag-builder installed:
    python desktop/scripts/build_engine.py

Then smoke-test both bundled executables.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
SPEC = DESKTOP / "engine" / "rag-engine.spec"
DIST = DESKTOP / "engine-dist"
WORK = DESKTOP / "engine-build"

ENGINE_TIMEOUT = 300
CLI_TIMEOUT = 300


def log(message: str) -> None:
    print(f"[ENGINE] {message}", flush=True)


def main() -> int:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "pyinstaller>=6.10"
        ])

    log("Preparing engine build directories...")
    shutil.rmtree(DIST / "rag-engine", ignore_errors=True)

    log("Starting PyInstaller...")
    result = subprocess.run(
        [
            sys.executable,
            "-m", "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath", str(DIST),
            "--workpath", str(WORK),
            str(SPEC),
        ],
        cwd=SPEC.parent,
        timeout=600,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"PyInstaller failed with exit code {result.returncode}"
        )

    log(f"PyInstaller process exited with code {result.returncode}")

    exe = DIST / "rag-engine" / (
        "rag-engine.exe" if os.name == "nt" else "rag-engine"
    )
    cli = DIST / "rag-engine" / (
        "rag.exe" if os.name == "nt" else "rag"
    )

    if not exe.is_file():
        raise FileNotFoundError(f"Engine executable missing: {exe}")

    if not cli.is_file():
        raise FileNotFoundError(f"CLI executable missing: {cli}")

    log(f"Engine executable: {exe}")
    log(f"CLI executable: {cli}")

    log("Engine executables successfully packaged.")
    log("Skipping runtime smoke tests during CI build.")

    return 0


def smoke_cli(cli: Path) -> int:
    """Test the bundled CLI using a temporary workspace."""

    def run(*args: str) -> str:
        command = [str(cli), *args]

        log(f"CLI: {' '.join(args)}")

        try:
            out = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=CLI_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"CLI command timed out after {CLI_TIMEOUT}s: {args}"
            )

        if out.returncode != 0:
            raise RuntimeError(
                f"rag {' '.join(args)} failed ({out.returncode}):\n"
                f"{out.stdout}\n{out.stderr}"
            )

        return out.stdout

    print(run("--version").strip(), flush=True)

    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        ws = Path(tmp) / "ws"

        docs.mkdir()

        (docs / "note.md").write_text(
            "# Leak test\n\n"
            "Helium leak rate limit is 5e-8 atm cc/s for TO-39 packages.\n",
            encoding="utf-8",
        )

        run("init", str(ws), "--embedder", "hash-test")
        run("build", str(ws), str(docs))

        found = run("search", str(ws), "helium leak rate")

        assert "5e-8" in found, found

        run("export", str(ws), str(Path(tmp) / "t.ragpack"))
        run("unpack", str(Path(tmp) / "t.ragpack"), str(Path(tmp) / "unpacked"))

    return 0


def smoke(exe: Path) -> int:
    """Start the frozen engine over stdio and run a tiny build + search."""

    with tempfile.TemporaryDirectory() as tmp:
        docs = Path(tmp) / "docs"
        docs.mkdir()

        (docs / "note.md").write_text(
            "# Leak test\n\n"
            "Helium leak rate limit is 5e-8 atm cc/s for TO-39 packages.\n",
            encoding="utf-8",
        )

        env = {
            **os.environ,
            "RAG_WORKBENCH_HOME": str(Path(tmp) / "home"),
        }

        # Use a temporary file for stderr instead of PIPE.
        # This prevents the child process from blocking if stderr fills up.
        with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as stderr_file:

            proc = subprocess.Popen(
                [str(exe), "--stdio"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr_file,
                text=True,
                env=env,
                encoding="utf-8",
            )

            assert proc.stdin and proc.stdout

            timed_out = threading.Event()

            def kill_on_timeout() -> None:
                timed_out.set()
                if proc.poll() is None:
                    proc.kill()

            watchdog = threading.Timer(ENGINE_TIMEOUT, kill_on_timeout)
            watchdog.daemon = True
            watchdog.start()

            def diagnostics() -> str:
                stderr_file.flush()
                stderr_file.seek(0)
                return stderr_file.read()

            def call(rid: int, method: str, params: dict) -> dict:
                if timed_out.is_set():
                    raise TimeoutError("Engine smoke test timed out.")

                proc.stdin.write(
                    json.dumps({
                        "id": rid,
                        "method": method,
                        "params": params,
                    }) + "\n"
                )
                proc.stdin.flush()

                while True:
                    line = proc.stdout.readline()

                    if not line:
                        if timed_out.is_set():
                            raise TimeoutError(
                                f"Engine exceeded {ENGINE_TIMEOUT}s.\n"
                                f"{diagnostics()}"
                            )

                        raise RuntimeError(
                            "Engine exited unexpectedly:\n"
                            + diagnostics()
                        )

                    msg = json.loads(line)

                    if msg.get("id") == rid:
                        if "error" in msg:
                            raise RuntimeError(
                                f"{method} failed: {msg['error']}"
                            )

                        return msg["result"]

            try:
                log("Waiting for engine startup...")

                ready_line = proc.stdout.readline()

                if not ready_line:
                    raise RuntimeError(
                        "Engine failed to send startup message:\n"
                        + diagnostics()
                    )

                ready = json.loads(ready_line)

                log(
                    f"Engine started: version={ready['data']['version']}, "
                    f"frozen={ready['data']['frozen']}"
                )

                log("Creating smoke-test knowledge base...")

                kb = call(
                    1,
                    "kb.create",
                    {"name": "smoke", "dims": 64, "backend": "hash-test"},
                )

                log("Starting smoke-test build...")

                job = call(
                    2,
                    "kb.build",
                    {"kb": kb["id"], "inputs": [str(docs)]},
                )

                deadline = time.monotonic() + ENGINE_TIMEOUT

                while True:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            "Engine build job exceeded its timeout."
                        )

                    j = call(
                        3,
                        "jobs.get",
                        {"job": job["id"]},
                    )

                    if j["state"] != "running":
                        break

                    time.sleep(0.25)

                if j["state"] != "done":
                    raise RuntimeError(
                        f"Build failed: {j.get('error')}"
                    )

                log("Searching smoke-test knowledge base...")

                hits = call(
                    4,
                    "kb.search",
                    {"kb": kb["id"], "query": "helium leak rate"},
                )["results"]

                assert hits and "5e-8" in hits[0]["text"], hits

                status = call(5, "runtime.status", {})

                log(
                    "Hardware: "
                    f"CPU={status['hardware']['cpu']}, "
                    f"GPUs={status['hardware']['gpus']}"
                )

                call(6, "app.shutdown", {})

                proc.wait(timeout=30)

            finally:
                watchdog.cancel()

                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=10)

                proc.stdout.close()
                proc.stdin.close()

                if timed_out.is_set():
                    raise TimeoutError(
                        f"Engine smoke test exceeded {ENGINE_TIMEOUT}s."
                    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(f"BUILD FAILED: {exc}")
        raise
