"""The shared search engine (Rust, behind shammy.Retriever) against the Python reference, and reranking."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

shammy = pytest.importorskip("shammy")

from rag.builder.engine import Builder  # noqa: E402
from rag.cli import main as cli_main  # noqa: E402
from rag.core.db.store import RagStore  # noqa: E402
from rag.core.models.embed import make_embedder  # noqa: E402
from rag.core.retrieve.search import Searcher, fts_query, rerank_settings  # noqa: E402
from rag.core.workspace import Workspace  # noqa: E402

from conftest import TEST_EMBEDDING  # noqa: E402

QUERIES = [
    "slow leak failures after run-in", "curing oven profile", "track-cli holds --reason PressureLeak",
    "what is the air leak rate limit", "GetLotYield", "calibration due dates for the leak tester",
    "who took over the night shift handoff", "E42 error on HTTPServer2", "the of and", "torque 40 Nm",
    "SELECT lot_id FROM yields", "figure showing the seal cross-section", "résumé of the audit findings",
]


@pytest.fixture(scope="module")
def stores(built_ws, tmp_path_factory, corpus):
    sh = Workspace.init(tmp_path_factory.mktemp("shared") / "ws", {**TEST_EMBEDDING, "vectors": {"backend": "shammy"}})
    assert Builder(sh).build([corpus]).failed == 0
    return {"sqlite-vec": built_ws, "shammy": sh}


def _pair(ws):
    store = RagStore(ws.rag_db, readonly=True)
    emb = make_embedder(ws.config["embedding"])
    return store, Searcher(store, emb, ws.config["search"], engine="reference"), Searcher(store, emb, ws.config["search"], engine="native")


def _same_scores(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert abs(x.pop("score") - y.pop("score")) < 2e-5
        assert x == y


@pytest.mark.parametrize("backend", ["sqlite-vec", "shammy"])
def test_engine_matches_the_python_reference(stores, backend):
    store, ref, native = _pair(stores[backend])
    types = [None] + sorted(r[0] for r in store.conn.execute("SELECT DISTINCT file_type FROM documents"))
    compared = with_results = 0
    try:
        for q in QUERIES:
            assert shammy.fts_query(q) == fts_query(q)
            for ft in types:
                for kw in ({}, {"k": 3, "expand": False}, {"k": 15}):
                    a = ref.search(q, file_type=ft, **kw)
                    b = native.search(q, file_type=ft, **kw)
                    assert (a.engine, b.engine) == ("reference", "native")
                    assert a.candidates == b.candidates, (q, ft, kw)
                    assert set(a.ranks) == set(b.ranks)
                    for cid in a.ranks:
                        ra, rb = dict(a.ranks[cid]), dict(b.ranks[cid])
                        assert abs(ra.pop("fused") - rb.pop("fused")) < 2e-5 and ra == rb
                    _same_scores([dict(r) for r in a.results], [dict(r) for r in b.results])
                    _same_scores([dict(t) for t in a.tables], [dict(t) for t in b.tables])
                    assert a.reference_markdown() == b.markdown == b.to_markdown(), (q, ft, kw)
                    compared += 1
                    with_results += bool(a.results)
    finally:
        native.close()
        store.close()
    assert compared > 100 and with_results > compared // 2


def test_retriever_directly_and_errors(stores, tmp_path):
    ws = stores["shammy"]
    emb = make_embedder(ws.config["embedding"])
    with shammy.Retriever(ws.root) as r:
        info = r.info()
        assert info["vector_store"] == "shammy" and info["dims"] == 256 and info["chunks"] == info["vectors"] > 0
        out = r.search("curing oven", emb.embed_query("curing oven"), k=2, markdown=True)
        assert len(out["results"]) == 2 and out["results"][0]["tag"] == "S1" and out["markdown"].startswith("[S1] ")
        assert "rerank" not in out
        with pytest.raises(ValueError, match="values"):
            r.search("x", [0.0] * 300)
        with pytest.raises(ValueError, match="bad search request"):
            r._native.search("{not json", b"\0" * 1024)
    with pytest.raises(OSError, match="rag.db"):
        shammy.Retriever(tmp_path)


class _FakeReranker(BaseHTTPRequestHandler):
    """Scores a passage by how often it contains the word 'calibration', whatever the question is."""

    seen: list[dict] = []
    mode = "ok"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen.append({"path": self.path, **body})
        if type(self).mode == "error":
            data, code = b'{"error": "model not loaded"}', 503
        else:
            results = [{"index": i, "relevance_score": float(d.lower().count("calibration")) - i * 1e-6}
                       for i, d in enumerate(body["documents"])]
            data, code = json.dumps({"model": "fake", "results": results[::-1]}).encode(), 200
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def reranker():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeReranker)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    _FakeReranker.seen, _FakeReranker.mode = [], "ok"
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


@pytest.mark.parametrize("backend", ["sqlite-vec", "shammy"])
def test_reranking(stores, backend, reranker):
    ws = stores[backend]
    store = RagStore(ws.rag_db, readonly=True)
    emb = make_embedder(ws.config["embedding"])
    q = "leak tester seal"
    try:
        plain = Searcher(store, emb, ws.config["search"]).search(q, k=4, expand=False)
        assert plain.rerank is None and "rerank_score" not in plain.results[0]

        cfg = rerank_settings(ws.config.data, True, reranker)
        assert cfg["enabled"] and cfg["candidates"] == 20
        ranked = Searcher(store, emb, ws.config["search"], reranker=cfg).search(q, k=4, expand=False)
        assert ranked.rerank["applied"] and ranked.timings_ms["rerank"] >= 0
        sent = _FakeReranker.seen[-1]
        assert sent["path"] == "/v1/rerank" and sent["query"] == q and 4 < len(sent["documents"]) <= 20
        scores = [r["rerank_score"] for r in ranked.results]
        assert scores == sorted(scores, reverse=True) and scores[0] > 0.9        # the reranker's order
        assert "calibration" in ranked.results[0]["text"].lower()
        assert [r["chunk_id"] for r in ranked.results] != [r["chunk_id"] for r in plain.results]
        assert ranked.candidates == plain.candidates                              # fusion itself is untouched

        # a reranker that fails doesn't take search down: fused order, and the reason is reported
        for url, text in ((reranker, "503"), ("http://127.0.0.1:9", "can't reach")):
            _FakeReranker.mode = "error"
            down = Searcher(store, emb, ws.config["search"], reranker=rerank_settings(ws.config.data, True, url))
            res = down.search(q, k=4, expand=False)
            assert res.rerank["applied"] is False and text in res.rerank["error"]
            assert [r["chunk_id"] for r in res.results] == [r["chunk_id"] for r in plain.results]
    finally:
        store.close()
    assert rerank_settings(ws.config.data) is None and rerank_settings(ws.config.data, False, reranker) is None


def test_cli_rerank_flags(stores, reranker, capsys, monkeypatch):
    root = str(stores["shammy"].root)
    assert cli_main(["search", root, "leak tester seal", "--json", "--rerank", "--rerank-url", reranker]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["rerank"]["applied"] and "rerank_score" in out["results"][0]
    assert cli_main(["search", root, "leak tester seal", "--json"]) == 0
    assert "rerank" not in json.loads(capsys.readouterr().out)
    monkeypatch.setenv("RAG_RERANK_URL", "http://127.0.0.1:9")
    assert cli_main(["search", root, "leak tester seal", "--rerank"]) == 0
    captured = capsys.readouterr()
    assert "not reranked" in captured.err and "[S1]" in captured.out
