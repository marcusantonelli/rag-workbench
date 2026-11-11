from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from rag.core.models.embed import EmbedderError, LlamaServerEmbedder, make_embedder


class _FakeLlamaServer(BaseHTTPRequestHandler):
    requests: list = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeLlamaServer.requests.append(body)
        data = []
        for i, text in enumerate(body["input"]):
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.normal(size=4096)
            data.append({"index": i, "embedding": (v / np.linalg.norm(v)).tolist()})
        out = json.dumps({"data": list(reversed(data))}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture()
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeLlamaServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _FakeLlamaServer.requests = []
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _cfg(url, **kw):
    base = {"backend": "llama-server", "model": "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M", "url": url,
            "native_dims": 4096, "dims": 2048, "batch_size": 3, "timeout_s": 5,
            "query_template": "Instruct: {task}\nQuery:{query}", "query_instruction": "Find passages"}
    base.update(kw)
    return base


def test_llama_server_batches_orders_truncates_and_normalizes(fake_server):
    emb = make_embedder(_cfg(fake_server))
    texts = [f"text {i}" for i in range(7)]
    out = emb.embed_documents(texts)
    assert out.shape == (7, 2048)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert [len(r["input"]) for r in _FakeLlamaServer.requests] == [3, 3, 1]
    again = emb.embed_documents(["text 4"])
    assert np.allclose(again[0], out[4], atol=1e-6), "results must be re-ordered by index"


def test_query_prefix_and_eos(fake_server):
    emb = make_embedder(_cfg(fake_server, append_eos=True))
    emb.embed_query("minimum pull?")
    sent = _FakeLlamaServer.requests[-1]["input"][0]
    assert sent == "Instruct: Find passages\nQuery:minimum pull?<|endoftext|>"
    assert emb.info()["query_instruction"] == "Instruct: Find passages\nQuery:{query}"


def test_unreachable_server_gives_start_command():
    emb = LlamaServerEmbedder(_cfg("http://127.0.0.1:9", timeout_s=1))
    with pytest.raises(EmbedderError, match="--embedding --pooling last"):
        emb.embed_documents(["x"])


def test_dims_above_native_rejected():
    with pytest.raises(EmbedderError):
        make_embedder(_cfg("http://x", dims=8192))


def test_query_templates_with_dollar_placeholders():
    from rag.core.models.embed import render_template, to_editable, to_stored

    assert render_template("Instruct: ${task}\nQuery:${query}", task="Find {x}", query="a {b}") == "Instruct: Find {x}\nQuery:a {b}"
    assert render_template("Instruct: {task}\nQuery:{query}", task="T", query="q") == "Instruct: T\nQuery:q"
    assert to_stored("Q: ${query} (manuals)") == "Q: {query} (manuals)" and to_editable("Q: {query}") == "Q: ${query}"
