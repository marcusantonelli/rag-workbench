"""The desktop app's terminal: runs `rag` commands, and nothing else.

The command line is split into arguments here (no shell is involved, so pipes, `&&`,
redirection and variables don't exist) and the first word must be one of rag's own
subcommands. The program started is always the bundled `rag` (or `python -m rag.cli`
in development), never something named by the user.
"""

from __future__ import annotations

import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from rag.app.cli_install import bundled_rag

SUBCOMMANDS = ("init", "list", "verify", "build", "remove", "status", "export", "unpack", "merge", "convert", "compact", "search", "tables", "describe", "sql",
               "eval", "check-embedder")
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class CommandRejected(ValueError):
    pass


def split_command(line: str, *, windows: bool | None = None) -> list[str]:
    """Arguments after `rag`. Quotes group words; on Windows backslashes are path separators, not escapes."""
    windows = sys.platform == "win32" if windows is None else windows
    lex = shlex.shlex(line.strip(), posix=True)
    lex.whitespace_split = True
    lex.commenters = ""
    if windows:
        lex.escape = ""
    try:
        args = list(lex)
    except ValueError as exc:   # unbalanced quotes
        raise CommandRejected(f"Can't read that command: {exc}. Check the quotes.") from exc
    if args and args[0].lower() in ("rag", "rag.exe"):
        args = args[1:]
    if not args:
        raise CommandRejected("Type a rag command, for example: search my-kb.ragpack \"leak test limits\"")
    first = args[0]
    if first in ("help", "-h", "--help"):
        return ["--help"] if len(args) == 1 else [*args[1:], "--help"]
    if first == "--version":
        return ["--version"]
    if first not in SUBCOMMANDS:
        raise CommandRejected(
            f"Only rag commands run here, and {first!r} isn't one. Commands: {', '.join(SUBCOMMANDS)}.")
    for a in args:
        if "\n" in a or "\0" in a:
            raise CommandRejected("Commands must be on one line.")
    return args


def rag_program() -> list[str]:
    exe = bundled_rag()
    return [str(exe)] if exe else [sys.executable, "-m", "rag.cli"]


class Run:
    def __init__(self, run_id: str, argv: list[str], cwd: Path):
        self.id, self.argv, self.cwd = run_id, argv, cwd
        self.proc: subprocess.Popen | None = None
        self.started = time.monotonic()
        self.stopped = False


class Terminal:
    def __init__(self, emit: Callable[[str, dict[str, Any]], None], embedding_url: Callable[[], str | None]):
        self.emit = emit
        self.embedding_url = embedding_url
        self.current: Run | None = None
        self.lock = threading.Lock()

    def run(self, command: str, cwd: str | None = None) -> dict[str, Any]:
        args = split_command(command)
        folder = Path(cwd).expanduser() if cwd else Path.home()
        if not folder.is_dir():
            raise CommandRejected(f"The working folder {folder} doesn't exist.")
        with self.lock:
            if self.current is not None and self.current.proc is not None and self.current.proc.poll() is None:
                raise CommandRejected("A command is still running. Stop it first.")
            run = Run("run_" + uuid.uuid4().hex[:10], args, folder)
            self.current = run
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"}
        url = self.embedding_url()
        if url:
            env["RAG_EMBEDDING_URL"] = url
        try:
            run.proc = subprocess.Popen(rag_program() + args, cwd=str(folder), env=env, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW)
        except OSError as exc:
            self.current = None
            raise CommandRejected(f"Couldn't start rag: {exc}") from exc
        out: queue.Queue = queue.Queue()
        readers = [threading.Thread(target=self._read, args=(run.proc.stdout, "stdout", out), daemon=True),
                   threading.Thread(target=self._read, args=(run.proc.stderr, "stderr", out), daemon=True)]
        for t in readers:
            t.start()
        threading.Thread(target=self._pump, args=(run, out, readers), daemon=True).start()
        shown = "rag " + " ".join(shlex.quote(a) if sys.platform != "win32" else (f'"{a}"' if " " in a else a)
                                  for a in args)
        return {"run_id": run.id, "argv": args, "display": shown, "cwd": str(folder), "embedding_url": url}

    @staticmethod
    def _read(stream, name: str, out: queue.Queue) -> None:
        for raw in iter(stream.readline, b""):
            out.put((name, raw.decode("utf-8", errors="replace").rstrip("\r\n")))
        out.put((name, None))

    def _pump(self, run: Run, out: queue.Queue, readers: list[threading.Thread]) -> None:
        """Send output in small batches (a build prints thousands of lines)."""
        open_streams = 2
        batch: list[dict[str, str]] = []
        last = time.monotonic()
        while open_streams:
            try:
                name, text = out.get(timeout=0.1)
                if text is None:
                    open_streams -= 1
                else:
                    batch.append({"stream": name, "text": text})
            except queue.Empty:
                pass
            if batch and (len(batch) >= 200 or time.monotonic() - last > 0.1):
                self.emit("terminal", {"run_id": run.id, "type": "output", "lines": batch})
                batch, last = [], time.monotonic()
        if batch:
            self.emit("terminal", {"run_id": run.id, "type": "output", "lines": batch})
        assert run.proc is not None
        code = run.proc.wait()
        self.emit("terminal", {"run_id": run.id, "type": "exit", "code": code, "stopped": run.stopped,
                               "seconds": round(time.monotonic() - run.started, 1)})

    def stop(self, run_id: str) -> dict[str, Any]:
        run = self.current
        if run is None or run.id != run_id or run.proc is None or run.proc.poll() is not None:
            return {"stopped": False}
        run.stopped = True
        run.proc.terminate()
        try:
            run.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            run.proc.kill()
        return {"stopped": True}

    def shutdown(self) -> None:
        run = self.current
        if run is not None and run.proc is not None and run.proc.poll() is None:
            run.proc.kill()
