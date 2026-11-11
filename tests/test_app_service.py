"""The desktop app's engine: knowledge bases, jobs, runtime management and transports."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest

from rag.app.paths import Settings
from rag.app.service import Service, ServiceError

FAKE_SERVER = r'''
import hashlib, json, math, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

args = sys.argv[1:]
if "--list-devices" in args:
    print("Available devices:")
    print("  Vulkan0: Fake Radeon (8192 MiB, 7900 MiB free)")
    sys.exit(0)
port = int(args[args.index("--port") + 1])
born = time.monotonic()
print("fake llama-server args:", " ".join(args), flush=True)

def vec(text):
    out = []
    for i in range(64):
        h = hashlib.sha256(f"{i}:{text.lower()}".encode()).digest()
        out.append((h[0] - 127.5) / 127.5)
    for w in text.lower().split():
        h = hashlib.sha256(w.encode()).digest()
        out[h[0] % 64] += 3.0
    n = math.sqrt(sum(x * x for x in out)) or 1.0
    return [x / n for x in out]

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass
    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        if self.path == "/health":
            self._send(200 if time.monotonic() - born > 0.4 else 503, {"status": "ok"})
        else:
            self._send(404, {})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
        self._send(200, {"data": [{"index": i, "embedding": vec(t)} for i, t in enumerate(inputs)]})

ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
'''

HASH = {"backend": "hash-test", "dims": 256}


def wait_job(svc: Service, job: dict, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        j = svc.jobs_get(job["id"])
        if j["state"] != "running":
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job['id']} did not finish")


@pytest.fixture()
def svc(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_WORKBENCH_HOME", str(tmp_path / "home"))
    events: list[tuple[str, dict]] = []
    s = Service(Settings(tmp_path / "home"), emit=lambda n, d: events.append((n, d)))
    s.events = events  # type: ignore[attr-defined]
    yield s
    s.shutdown()


def test_create_build_browse_search_export_and_reopen_as_pack(svc, corpus, tmp_path):
    kb = svc.call("kb.create", {"name": "Plant docs", "dims": 256, "backend": "hash-test"})
    assert kb["kind"] == "workspace" and kb["name"] == "Plant docs" and kb["counts"]["documents"] == 0
    assert kb["runtime"]["needs_runtime"] is False

    job = wait_job(svc, svc.call("kb.build", {"kb": kb["id"], "inputs": [str(corpus)]}))
    assert job["state"] == "done", job["error"]
    assert job["result"]["added"] == 11 and job["progress"]["files_done"] == 11
    assert any(n == "job" and d["state"] == "done" for n, d in svc.events)

    summary = svc.call("kb.summary", {"kb": kb["id"]})
    assert summary["counts"]["documents"] == 11 and summary["counts"]["tables"] >= 1
    docs = svc.call("kb.documents", {"kb": kb["id"]})
    pdf = next(d for d in docs if d["file_type"] == "pdf")
    assert svc.call("kb.documents", {"kb": kb["id"], "query": pdf["display_path"][-12:]})[0]["doc_id"] == pdf["doc_id"]
    doc = svc.call("kb.document", {"kb": kb["id"], "doc_id": pdf["doc_id"]})
    assert doc["chunks"] and doc["chunks"][0]["ordinal"] == 0
    chunk = svc.call("kb.chunk", {"kb": kb["id"], "chunk_id": doc["chunks"][0]["chunk_id"]})
    assert chunk["chunk"]["file"] == pdf["display_path"] and chunk["context"]

    res = svc.call("kb.search", {"kb": kb["id"], "query": "minimum pull strength 5 mil aluminum"})
    assert res["results"] and res["candidates"] and res["markdown"].startswith("[S1]")
    assert res["results"][0]["ranks"]["rank"] >= 1 and "fused" in res["results"][0]["ranks"]

    tables = svc.call("kb.tables", {"kb": kb["id"]})
    name = tables[0]["table_name"]
    assert svc.call("kb.table", {"kb": kb["id"], "name": name})["columns"]
    assert svc.call("kb.sql", {"kb": kb["id"], "sql": f'SELECT COUNT(*) AS n FROM "{name}"'})["rows"][0][0] >= 1
    with pytest.raises(ServiceError) as err:
        svc.call("kb.sql", {"kb": kb["id"], "sql": f'DELETE FROM "{name}"'})
    assert err.value.code == "sql_rejected"

    golden = tmp_path / "g.jsonl"
    golden.write_text("\n".join(json.dumps(q) for q in [
        {"id": "a", "type": "t", "question": "minimum pull strength 5 mil aluminum",
         "expect": [{"file": "*QP-114*"}]},
        {"id": "z", "question": "price of a hose crimper", "expect": []},
    ]), encoding="utf-8")
    ev = wait_job(svc, svc.call("eval.run", {"kb": kb["id"], "golden": str(golden), "show": 3}))
    assert ev["state"] == "done", ev["error"]
    assert ev["result"]["overall"]["n"] == 1 and ev["result"]["questions"][0]["retrieved"]
    assert "# Retrieval report" in svc.call("eval.markdown", {"report": ev["result"]})

    out = tmp_path / "out" / "plant"
    ex = wait_job(svc, svc.call("kb.export", {"kb": kb["id"], "out": str(out)}))
    assert ex["state"] == "done", ex["error"]
    pack_path = Path(ex["result"]["path"])
    assert pack_path.suffix == ".ragpack" and pack_path.exists()

    pack = svc.call("kb.open", {"path": str(pack_path)})
    assert pack["kind"] == "pack" and pack["readonly"] and pack["name"] == "Plant docs"
    assert pack["counts"]["documents"] == 11 and pack["manifest"]["counts"]["documents"] == 11
    assert svc.call("kb.search", {"kb": pack["id"], "query": "oven zone 3"})["results"]
    with pytest.raises(ServiceError) as err:
        svc.call("kb.build", {"kb": pack["id"], "inputs": [str(corpus)]})
    assert err.value.code == "readonly"
    ver = wait_job(svc, svc.call("kb.verify", {"kb": pack["id"]}))
    assert ver["result"]["ok"] is True

    # Reopening reuses the verified unpacked copy; both show up in recents.
    svc.call("kb.close", {"kb": pack["id"]})
    again = svc.call("kb.open", {"path": str(pack_path)})
    assert again["root"] == pack["root"]
    recent = [r["path"] for r in svc.call("kb.recent", {})]
    assert str(pack_path) in recent and kb["path"] in recent
    assert len(svc.call("kb.list", {})) == 2


def test_errors_are_coded(svc, tmp_path):
    with pytest.raises(ServiceError) as err:
        svc.call("kb.open", {"path": str(tmp_path / "missing.ragpack")})
    assert err.value.code == "not_found"
    bad = tmp_path / "bad.ragpack"
    bad.write_text("not a zip")
    with pytest.raises(ServiceError) as err:
        svc.call("kb.open", {"path": str(bad)})
    assert err.value.code == "not_pack"
    (tmp_path / "folder").mkdir()
    with pytest.raises(ServiceError) as err:
        svc.call("kb.open", {"path": str(tmp_path / "folder")})
    assert err.value.code == "not_workspace"
    with pytest.raises(ServiceError) as err:
        svc.call("kb.nope", {})
    assert err.value.code == "unknown_method"
    with pytest.raises(ServiceError) as err:
        svc.call("kb.search", {"kb": "kb_x", "query": "q"})
    assert err.value.code == "kb_not_open"


def test_llama_server_kb_needs_a_ready_matching_runtime(svc):
    kb = svc.call("kb.create", {"name": "Needs model", "dims": 4096})
    assert kb["runtime"] == {"needs_runtime": True, "ready": False, "match": True,
                             "kb_model": "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",
                             "runtime_model": "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M"}
    with pytest.raises(ServiceError) as err:
        svc.call("kb.search", {"kb": kb["id"], "query": "anything"})
    assert err.value.code == "runtime_not_ready"


def test_cancelled_build_keeps_finished_files(svc, corpus, monkeypatch):
    kb = svc.call("kb.create", {"name": "Cancel me", "dims": 256, "backend": "hash-test"})
    from rag.builder import engine

    original = engine.Builder._process

    def slow(self, *a, **kw):
        time.sleep(0.3)
        return original(self, *a, **kw)

    monkeypatch.setattr(engine.Builder, "_process", slow)
    job = svc.call("kb.build", {"kb": kb["id"], "inputs": [str(corpus)]})
    time.sleep(0.5)
    svc.call("jobs.cancel", {"job": job["id"]})
    done = wait_job(svc, job)
    assert done["state"] == "cancelled" and done["result"]["cancelled"] is True
    n = svc.call("kb.summary", {"kb": kb["id"]})["counts"]["documents"]
    assert 0 < n < 11


@pytest.mark.skipif(sys.platform == "win32", reason="fake llama-server is a shebang script")
def test_managed_runtime_with_fake_llama_server(svc, corpus, tmp_path):
    rt = svc.runtime
    build = rt.build_dir("vulkan")
    build.mkdir(parents=True)
    exe = build / "llama-server"
    exe.write_text(f"#!{sys.executable}\n" + FAKE_SERVER)
    exe.chmod(0o755)
    model = tmp_path / "Qwen3-Embedding-8B-Q4_K_M.gguf"
    model.write_bytes(b"GGUF" + b"\0" * 1024)
    svc.settings.update_runtime(variant="vulkan", model_path=str(model))

    status = svc.call("runtime.status", {})
    assert status["needs_setup"] is False and status["state"] == "stopped"
    rt.start(wait=True)
    status = svc.call("runtime.status", {})
    assert status["ready"] and status["url"].startswith("http://127.0.0.1:")
    assert status["devices"][0]["name"] == "Fake Radeon" and "Fake Radeon" in status["device_note"]
    assert "-ngl 99" in " ".join(status["log_tail"])

    kb = svc.call("kb.create", {"name": "Real model", "dims": 64})
    svc.call("runtime.configure", {})   # no-op
    ws_cfg = (Path(kb["root"]) / "config.toml").read_text()
    assert "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M" in ws_cfg
    # native_dims stays 4096 in config; dims 64 truncates the fake server's 64-dim output.
    job = wait_job(svc, svc.call("kb.build", {"kb": kb["id"], "inputs": [str(corpus)]}))
    assert job["state"] == "done", job["error"]
    assert svc.call("kb.search", {"kb": kb["id"], "query": "pressure leak"})["results"]

    # Force CPU restarts the server without offload.
    svc.call("runtime.configure", {"accel": "cpu"})
    deadline = time.monotonic() + 20
    while not svc.call("runtime.status", {})["ready"] and time.monotonic() < deadline:
        time.sleep(0.1)
    status = svc.call("runtime.status", {})
    assert status["ready"] and "--device none" in " ".join(status["log_tail"]) and status["device_note"] == "Force CPU is on."

    # A different model id makes searching this knowledge base a coded error.
    svc.settings.update_runtime(model_id="something-else")
    with pytest.raises(ServiceError) as err:
        svc.call("kb.search", {"kb": kb["id"], "query": "pressure leak"})
    assert err.value.code == "model_mismatch"
    assert svc.call("kb.search", {"kb": kb["id"], "query": "pressure leak", "allow_mismatch": True})["results"]

    svc.call("runtime.stop", {})
    assert svc.call("runtime.status", {})["state"] == "stopped"


def _engine_cmd() -> list[str]:
    return [sys.executable, "-m", "rag.app"]


def test_stdio_transport(tmp_path, corpus):
    env = {**os.environ, "RAG_WORKBENCH_HOME": str(tmp_path / "home"), "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(_engine_cmd() + ["--stdio"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, env=env)
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready["event"] == "ready" and ready["data"]["version"]

        def call(rid, method, params):
            proc.stdin.write(json.dumps({"id": rid, "method": method, "params": params}) + "\n")
            proc.stdin.flush()
            while True:
                msg = json.loads(proc.stdout.readline())
                if msg.get("id") == rid:
                    return msg

        assert call(1, "app.info", {})["result"]["data_dir"] == str(tmp_path / "home")
        assert call(2, "kb.open", {"path": str(tmp_path / "nope")})["error"]["code"] == "not_found"
        kb = call(3, "kb.create", {"name": "S", "dims": 256, "backend": "hash-test"})["result"]
        job = call(4, "kb.build", {"kb": kb["id"], "inputs": [str(corpus)]})["result"]
        # Events arrive on the same stream until the job finishes.
        while True:
            msg = json.loads(proc.stdout.readline())
            if msg.get("event") == "job" and msg["data"]["id"] == job["id"] and msg["data"]["state"] != "running":
                assert msg["data"]["state"] == "done" and msg["data"]["result"]["added"] == 11
                break
        assert call(5, "app.shutdown", {})["result"]["ok"] is True
        assert proc.wait(timeout=20) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_http_transport(tmp_path):
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "RAG_WORKBENCH_HOME": str(tmp_path / "home")}
    proc = subprocess.Popen(_engine_cmd() + ["--http", str(port)], env=env, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except OSError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(0.2)

        def call(method, params):
            req = urllib.request.Request(f"http://127.0.0.1:{port}/rpc", method="POST",
                                         data=json.dumps({"method": method, "params": params}).encode(),
                                         headers={"Content-Type": "application/json",
                                                  "Origin": "http://localhost:5173"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                assert resp.headers["Access-Control-Allow-Origin"] == "http://localhost:5173"
                return json.load(resp)

        assert call("app.info", {})["result"]["version"]
        assert call("kb.open", {"path": "/nope"})["error"]["code"] == "not_found"
        status = call("runtime.status", {})["result"]
        assert status["needs_setup"] is True and status["hardware"]["os"] in ("linux", "windows", "darwin")
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def test_variant_choice():
    from rag.app.runtime import choose_variant

    base = {"os": "windows", "arch": "x64", "nvidia": [], "gpus": []}
    nv = {**base, "nvidia": [{"name": "RTX 3090", "memory_mb": 24576, "driver": "581.12"}], "gpus": ["RTX 3090"]}
    assert choose_variant(nv, "auto")[0] == "cuda-13.5"
    assert choose_variant({**nv, "nvidia": [{**nv["nvidia"][0], "driver": "552.44"}]}, "auto")[0] == "cuda-12.4"
    assert choose_variant({**base, "gpus": ["AMD Radeon 760M Graphics"]}, "auto")[0] == "vulkan"
    assert choose_variant({**nv, "os": "linux"}, "auto")[0] == "vulkan"     # no Linux CUDA builds
    assert choose_variant(base, "auto")[0] == "cpu"
    assert choose_variant(base, "cpu")[0] == "cpu"
    # Force CPU on a GPU machine still installs the GPU build (it runs without offload).
    assert choose_variant({**base, "gpus": ["Intel Arc"]}, "cpu")[0] == "vulkan"


def test_download_resumes_and_replaces_atomically(tmp_path, monkeypatch):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from rag.app import runtime

    payload = os.urandom(3 * 1024 * 1024 + 123)
    seen_ranges = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            seen_ranges.append(rng)
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            body = payload[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/model.gguf"
    dest = tmp_path / "model.gguf"
    # An earlier run left the first megabyte behind.
    (tmp_path / "model.gguf.part").write_bytes(payload[: 1024 * 1024])
    events = []
    monkeypatch.setattr(runtime, "expected_sha256", lambda u: __import__("hashlib").sha256(payload).hexdigest())
    runtime.download(url, dest, progress=events.append, label="model")
    srv.shutdown()
    assert dest.read_bytes() == payload and not (tmp_path / "model.gguf.part").exists()
    assert seen_ranges == [f"bytes={1024 * 1024}-"]
    assert events[-1]["stage"] == "verify" and any(e.get("done") == len(payload) for e in events)


def test_unsafe_archive_is_refused(tmp_path):
    import zipfile

    from rag.app.runtime import RuntimeErrorWithHint, _extract

    bad = tmp_path / "evil.zip"
    with zipfile.ZipFile(bad, "w") as zf:
        zf.writestr("../escape.txt", "x")
    with pytest.raises(RuntimeErrorWithHint):
        _extract(bad, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_windows_path_editing_is_idempotent_and_exact():
    from rag.app.cli_install import path_add, path_has, path_remove

    folder = r"C:\Users\m\AppData\Local\RAG Workbench\engine"
    base = r"C:\Windows\system32;C:\Tools;"
    added = path_add(base, folder)
    assert added == r"C:\Windows\system32;C:\Tools;" + folder
    assert path_add(added, folder) == added                       # once only
    assert path_has(added, folder) and not path_has(base, folder)
    assert path_remove(added, folder) == r"C:\Windows\system32;C:\Tools"
    # A neighbouring folder with a longer name is left alone.
    assert path_has(path_remove(added + r";" + folder + "-old", folder), folder + "-old")


def test_cli_status_outside_the_installed_app(svc):
    st = svc.call("cli.status", {})
    assert st["mode"] == "development" and st["can_toggle"] is False
    with pytest.raises(ServiceError) as err:
        svc.call("cli.set_enabled", {"enabled": True})
    assert err.value.code == "cli"


def test_managed_server_prefers_the_cli_port():
    import socket

    from rag.app.runtime import PREFERRED_PORT, _free_port

    try:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", PREFERRED_PORT))
        free = True
    except OSError:
        free = False
    if free:
        assert _free_port() == PREFERRED_PORT
    with socket.socket() as held:
        held.bind(("127.0.0.1", 0))
        taken = held.getsockname()[1]
        assert _free_port(taken) not in (taken, 0)                # falls back to any free port


def test_rag_cli_version_and_one_line_errors(tmp_path):
    out = subprocess.run([sys.executable, "-m", "rag.cli", "--version"], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.startswith("rag ")
    bad = subprocess.run([sys.executable, "-m", "rag.cli", "status", str(tmp_path / "nope")],
                         capture_output=True, text=True)
    assert bad.returncode == 2 and bad.stderr.startswith("error:") and "Traceback" not in bad.stderr
    ws = tmp_path / "ws"
    subprocess.run([sys.executable, "-m", "rag.cli", "init", str(ws)], check=True, capture_output=True)
    cfg = (ws / "config.toml").read_text().replace("127.0.0.1:8081", "127.0.0.1:9")   # nothing listens
    (ws / "config.toml").write_text(cfg)
    down = subprocess.run([sys.executable, "-m", "rag.cli", "search", str(ws), "anything"],
                          capture_output=True, text=True, timeout=120)
    assert down.returncode == 2 and "RAG Workbench" in down.stderr and "Traceback" not in down.stderr


def test_terminal_only_accepts_rag_commands():
    from rag.app.terminal import CommandRejected, split_command

    assert split_command('rag search my.ragpack "leak test limits" -k 3', windows=False) == \
        ["search", "my.ragpack", "leak test limits", "-k", "3"]
    assert split_command('search kb "x"', windows=False) == ["search", "kb", "x"]          # "rag" is optional
    # Windows paths keep their backslashes, quoted or not.
    assert split_command(r'rag build ws D:\KB\Procedures "C:\My Docs\Specs"', windows=True) == \
        ["build", "ws", r"D:\KB\Procedures", r"C:\My Docs\Specs"]
    assert split_command("rag help search", windows=False) == ["search", "--help"]
    assert split_command("rag --version", windows=False) == ["--version"]
    # Shell syntax is just text: it reaches rag as arguments, never a shell.
    assert split_command("rag status ws && rm -rf /", windows=False) == ["status", "ws", "&&", "rm", "-rf", "/"]
    for bad in ("ls -la", "python -c 1", "rag shell", "", "rag", 'rag search "unclosed', "cmd /c dir"):
        with pytest.raises(CommandRejected):
            split_command(bad, windows=False)


def test_terminal_runs_rag_and_streams_output(svc, corpus, tmp_path):
    def run_and_wait(command, cwd=None, timeout=180):
        start = len(svc.events)
        info = svc.call("terminal.run", {"command": command, "cwd": cwd})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            mine = [d for n, d in svc.events[start:] if n == "terminal" and d["run_id"] == info["run_id"]]
            if any(d["type"] == "exit" for d in mine):
                lines = [ln["text"] for d in mine if d["type"] == "output" for ln in d["lines"]]
                return info, lines, next(d for d in mine if d["type"] == "exit")
            time.sleep(0.05)
        raise AssertionError("command did not finish")

    info, lines, done = run_and_wait("rag --version")
    assert done["code"] == 0 and lines[0].startswith("rag ") and info["cwd"] == str(Path.home())
    _, _, done = run_and_wait("init ws --embedder hash-test", cwd=str(tmp_path))
    assert done["code"] == 0 and (tmp_path / "ws" / "rag.db").exists()        # relative to the working folder
    _, lines, done = run_and_wait(f'build ws "{corpus}"', cwd=str(tmp_path))
    assert done["code"] == 0 and any(ln.startswith("Done in") for ln in lines)
    _, lines, done = run_and_wait('search ws "minimum pull strength"', cwd=str(tmp_path))
    assert done["code"] == 0 and lines[0].startswith("[S1]")
    _, lines, done = run_and_wait("status missing-ws", cwd=str(tmp_path))
    assert done["code"] == 2 and any(ln.startswith("error:") for ln in lines)
    with pytest.raises(ServiceError) as err:
        svc.call("terminal.run", {"command": "ls"})
    assert err.value.code == "rejected"

    # A command that hangs (model address that never answers) can be stopped.
    cfg = (tmp_path / "ws" / "config.toml").read_text()
    (tmp_path / "slow").mkdir()
    subprocess.run([sys.executable, "-m", "rag.cli", "init", str(tmp_path / "slow" / "ws")], check=True,
                   capture_output=True)
    slow_cfg = tmp_path / "slow" / "ws" / "config.toml"
    slow_cfg.write_text(slow_cfg.read_text().replace("127.0.0.1:8081", "10.255.255.1:9"))
    assert cfg  # the hash-test workspace is untouched
    start = len(svc.events)
    info = svc.call("terminal.run", {"command": 'search ws "anything"', "cwd": str(tmp_path / "slow")})
    time.sleep(1.5)
    with pytest.raises(ServiceError):
        svc.call("terminal.run", {"command": "--version"})                     # one at a time
    assert svc.call("terminal.stop", {"run_id": info["run_id"]})["stopped"] is True
    deadline = time.monotonic() + 20
    while not any(n == "terminal" and d.get("type") == "exit" for n, d in svc.events[start:]):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    done = next(d for n, d in svc.events[start:] if n == "terminal" and d.get("type") == "exit")
    assert done["stopped"] is True and done["code"] != 0


def test_merge_a_pack_into_a_workspace(svc, corpus, tmp_path):
    a = svc.call("kb.create", {"name": "Line A", "dims": 256, "backend": "hash-test"})
    b = svc.call("kb.create", {"name": "Line B", "dims": 256, "backend": "hash-test"})
    assert wait_job(svc, svc.call("kb.build", {"kb": a["id"], "inputs": [str(corpus / "handoff.txt")]}))["state"] == "done"
    assert wait_job(svc, svc.call("kb.build", {"kb": b["id"], "inputs": [str(corpus / "calibration.csv"),
                                                                         str(corpus / "Pressure Leak.md")]}))["state"] == "done"
    ex = wait_job(svc, svc.call("kb.export", {"kb": b["id"], "out": str(tmp_path / "b.ragpack")}))
    pack = svc.call("kb.open", {"path": ex["result"]["path"]})

    plan = svc.call("kb.merge", {"kb": a["id"], "sources": [pack["path"]], "dry_run": True})
    assert plan["sources"][0]["add"] == 2 and plan["sources"][0]["tables"] == 1
    assert svc.call("kb.summary", {"kb": a["id"]})["counts"]["documents"] == 1

    job = wait_job(svc, svc.call("kb.merge", {"kb": a["id"], "sources": [pack["path"]]}))
    assert job["state"] == "done", job["error"]
    assert job["result"]["sources"][0]["add"] == 2 and job["progress"]["files_done"] == 2
    summary = svc.call("kb.summary", {"kb": a["id"]})
    assert summary["counts"]["documents"] == 3 and summary["counts"]["tables"] == 1
    assert svc.call("kb.search", {"kb": a["id"], "query": "air leak rate"})["results"]

    with pytest.raises(ServiceError) as err:
        svc.call("kb.merge", {"kb": pack["id"], "sources": [a["path"]]})
    assert err.value.code == "readonly"


def test_shammy_knowledge_bases_and_convert(svc, corpus, tmp_path):
    pytest.importorskip("shammy")
    assert svc.call("app.info", {})["shammy"]
    a = svc.call("kb.create", {"name": "On Shammy", "dims": 256, "backend": "hash-test", "vectors": "shammy"})
    assert a["vector_store"] == "shammy"
    assert wait_job(svc, svc.call("kb.build", {"kb": a["id"], "inputs": [str(corpus / "calibration.csv"),
                                                                         str(corpus / "Pressure Leak.md")]}))["state"] == "done"
    before = svc.call("kb.summary", {"kb": a["id"]})
    assert before["counts"]["vectors"] == before["counts"]["chunks"] > 0 and before["counts"]["tables"] == 1
    found = svc.call("kb.search", {"kb": a["id"], "query": "air leak rate"})
    hits = found["results"]
    assert hits and found["engine"] == "native" and "rerank" not in found
    unreachable = svc.call("kb.search", {"kb": a["id"], "query": "air leak rate", "rerank": True,
                                         "rerank_url": "http://127.0.0.1:9"})
    assert unreachable["rerank"]["applied"] is False and unreachable["results"][0]["text"] == hits[0]["text"]

    job = wait_job(svc, svc.call("kb.convert", {"kb": a["id"], "to": "sqlite-vec"}))
    assert job["state"] == "done", job["error"]
    assert job["result"]["vectors"] == before["counts"]["vectors"] and job["result"]["missing"] == 0
    after = svc.call("kb.summary", {"kb": a["id"]})
    assert after["vector_store"] == "sqlite-vec" and after["counts"]["vectors"] == before["counts"]["vectors"]
    assert svc.call("kb.search", {"kb": a["id"], "query": "air leak rate"})["results"][0]["text"] == hits[0]["text"]
    assert wait_job(svc, svc.call("kb.convert", {"kb": a["id"], "to": "shammy"}))["state"] == "done"
    assert wait_job(svc, svc.call("kb.verify", {"kb": a["id"]}))["result"]["ok"]

    ex = wait_job(svc, svc.call("kb.export", {"kb": a["id"], "out": str(tmp_path / "s.ragpack")}))
    pack = svc.call("kb.open", {"path": ex["result"]["path"]})
    assert pack["vector_store"] == "shammy" and svc.call("kb.search", {"kb": pack["id"], "query": "air leak rate"})["results"]
    with pytest.raises(ServiceError) as err:
        svc.call("kb.convert", {"kb": pack["id"], "to": "sqlite-vec"})
    assert err.value.code == "readonly"
    with pytest.raises(ServiceError) as err:
        svc.call("kb.create", {"name": "Odd", "dims": 300, "backend": "hash-test", "vectors": "shammy"})
    assert err.value.code == "bad_params"
