"""Picture descriptions: the describe step against a stand-in vision server, re-reading only the
documents that need it, and the managed llama-server process (with a fake llama-server)."""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from conftest import TEST_EMBEDDING  # noqa: E402
from make_drawings import make_drawings  # noqa: E402


class _Vision(BaseHTTPRequestHandler):
    """An OpenAI-style chat endpoint that 'describes' every picture the same way."""

    seen: list[dict] = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, **body})
        data = json.dumps({"choices": [{"message": {"content": "The image shows a seam sealer glove box with "
                                                               "a red CAUTION label on the oven door."}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def vision(monkeypatch):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Vision)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _Vision.seen = []
    url = f"http://127.0.0.1:{server.server_port}"
    monkeypatch.setenv("RAG_DESCRIBE_URL", url)
    yield url
    server.shutdown()


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return make_drawings(tmp_path_factory.mktemp("vis"))


def test_describe_step(vision, ctx, tmp_path):
    from PIL import Image, ImageDraw

    from rag.core.ingest.figures import describe_image, describe_settings

    assert describe_settings(ctx.config)["url"] == vision + "/v1/chat/completions"
    img = Image.new("RGB", (2400, 1200), "white")
    ImageDraw.Draw(img).rectangle([200, 200, 900, 800], fill="gray")
    text = describe_image(img, ctx, context="step 7.1 Verify the oven calibration")
    assert text == "A seam sealer glove box with a red CAUTION label on the oven door."   # "The image shows" trimmed
    sent = _Vision.seen[-1]
    prompt = sent["messages"][0]["content"][1]["text"]
    assert "illustrating: step 7.1 Verify the oven calibration" in prompt
    assert sent["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,")
    # a redaction box or blank frame isn't sent
    n = len(_Vision.seen)
    assert describe_image(Image.new("RGB", (800, 600), "black"), ctx) is None and len(_Vision.seen) == n
    assert ctx.describe_stats == {"done": 1, "failed": 0, "down": False}


def test_build_describes_pictures_and_rereads_only_when_needed(files, tmp_path, monkeypatch):
    from rag.builder.engine import Builder
    from rag.core.db.store import RagStore
    from rag.core.workspace import Workspace

    monkeypatch.delenv("RAG_DESCRIBE_URL", raising=False)
    src = tmp_path / "src"
    src.mkdir()
    for name in ("pictures", "lookalikes"):          # one with pictures, one without
        (src / files[name].name).write_bytes(files[name].read_bytes())
    ws = Workspace.init(tmp_path / "ws", TEST_EMBEDDING)
    assert Builder(ws).build([src]).failed == 0

    def extractors():
        s = RagStore(ws.rag_db, readonly=True)
        try:
            return {Path(p).name: e for p, e in s.conn.execute("SELECT display_path, extractor FROM documents")}
        finally:
            s.close()

    assert all(e.endswith("@v6") for e in extractors().values())

    # the picture model is down: the build still finishes, nothing is marked described
    monkeypatch.setenv("RAG_DESCRIBE_URL", "http://127.0.0.1:9")
    stats = Builder(ws).build([src])
    assert stats.failed == 0 and stats.unchanged == 1          # only the document with pictures was read again
    ex = extractors()
    assert ex["Yield pictures.pdf"].endswith("@v6") and ex["Not flowcharts.pdf"].endswith("@v6")

    # it's up: the document with pictures is read again and described, the other is left alone
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Vision)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("RAG_DESCRIBE_URL", f"http://127.0.0.1:{server.server_port}")
        stats = Builder(ws).build([src])
        assert stats.unchanged == 1 and stats.failed == 0
        assert extractors()["Yield pictures.pdf"].endswith("@v6+vis")
        s = RagStore(ws.rag_db, readonly=True)
        try:
            described = s.conn.execute("SELECT COUNT(*) FROM chunks WHERE text LIKE '%Description: A seam sealer%'").fetchone()[0]
        finally:
            s.close()
        assert described == 1          # the prose screenshot; the spreadsheet pictures became tables
        # described and current: nothing to do
        assert Builder(ws).build([src]).unchanged == 2
    finally:
        server.shutdown()
    # turning descriptions off doesn't throw them away
    monkeypatch.delenv("RAG_DESCRIBE_URL")
    assert Builder(ws).build([src]).unchanged == 2


FAKE_SERVER = r'''
import sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
args = sys.argv[1:]
port = int(args[args.index("--port") + 1])
if "99" in args and "--fail-on-gpu" in open(__file__).read() and "-ngl" in args and args[args.index("-ngl") + 1] == "99":
    print("ggml: out of memory on the GPU", flush=True)
    sys.exit(3)
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")
print("listening", flush=True)
HTTPServer(("127.0.0.1", port), H).serve_forever()
'''


def test_managed_server_starts_falls_back_and_stops(tmp_path):
    from rag.app.runtime import ManagedServer, _free_port

    script = tmp_path / "fake_llama_server.py"
    script.write_text(FAKE_SERVER + "\n# --fail-on-gpu\n")
    changes = []
    srv = ManagedServer("picture model", tmp_path / "v.pid", lambda: changes.append(srv.state))
    port = _free_port(0)
    gpu = [sys.executable, str(script), "--port", str(port), "-ngl", "99"]
    cpu = [sys.executable, str(script), "--port", str(port), "-ngl", "0"]
    srv.launch(Path(sys.executable), gpu, port, fallback=cpu)
    srv.wait(timeout=30)
    assert srv.state == "ready" and srv.url == f"http://127.0.0.1:{port}"
    assert any("out of memory" in line for line in srv.log) or any("retrying" in line for line in srv.log)
    assert (tmp_path / "v.pid").exists()
    srv.stop()
    assert srv.state == "stopped" and not (tmp_path / "v.pid").exists() and "ready" in changes

    # no fallback: the failure and the end of its log are reported
    srv.launch(Path(sys.executable), gpu, port)
    with pytest.raises(Exception, match="exited"):
        srv.wait(timeout=30)
    assert srv.state == "failed" and "out of memory" in srv.error


def test_vision_runtime_and_service(tmp_path, monkeypatch):
    from rag.app.paths import Settings
    from rag.app.runtime import VisionRuntime
    from rag.app.service import Service

    settings = Settings(tmp_path / "data")
    svc = Service(settings)
    try:
        st = svc.call("vision.status", {})
        assert st["needs_setup"] and not st["enabled"] and st["default_model"]["file"] == "Qwen3VL-4B-Instruct-Q4_K_M.gguf"
        # pretend it was set up: a fake llama-server build and two model files
        exe = tmp_path / "data" / "runtime" / "llama-bfake-cpu" / ("llama-server.exe" if sys.platform == "win32" else "llama-server")
        exe.parent.mkdir(parents=True)
        script = tmp_path / "fake.py"
        script.write_text(FAKE_SERVER)
        exe.write_text(f"#!/bin/sh\nexec {sys.executable} {script} \"$@\"\n")
        exe.chmod(0o755)
        for f in ("m.gguf", "mm.gguf"):
            (tmp_path / f).write_bytes(b"x" * 1024)
        settings.update_runtime(variant="cpu", llama_build="bfake", accel="cpu")
        settings.update_vision(model_path=str(tmp_path / "m.gguf"), mmproj_path=str(tmp_path / "mm.gguf"), port=0)
        if sys.platform == "win32":
            pytest.skip("the fake llama-server is a shell script")
        st = svc.call("vision.configure", {"enabled": True})
        svc.vision.server.wait(timeout=30)
        st = svc.call("vision.status", {})
        assert st["ready"] and st["enabled"]
        import os

        assert os.environ.get("RAG_DESCRIBE_URL") == st["url"]           # builds find it
        assert "--mmproj" in svc.vision.server.log[0] and "--device none" in svc.vision.server.log[0]
        svc.call("vision.configure", {"enabled": False})
        assert not svc.call("vision.status", {})["ready"] and "RAG_DESCRIBE_URL" not in os.environ
        with pytest.raises(Exception):
            svc.call("vision.configure", {"model_path": "x"})
    finally:
        svc.shutdown()
        monkeypatch.delenv("RAG_DESCRIBE_URL", raising=False)
