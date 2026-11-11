"""Workspaces that keep their vectors in Shammy behave like sqlite-vec ones."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

shammy = pytest.importorskip("shammy")

from rag.builder.engine import Builder  # noqa: E402
from rag.cli import main as cli_main  # noqa: E402
from rag.core.check import check_path, check_workspace  # noqa: E402
from rag.core.db.convert import compact_workspace, convert_workspace  # noqa: E402
from rag.core.db.merge import merge  # noqa: E402
from rag.core.db.pack import export_pack, read_manifest  # noqa: E402
from rag.core.db.store import RagStore  # noqa: E402
from rag.core.db.vectors import VectorStoreError  # noqa: E402
from rag.core.models.embed import make_embedder  # noqa: E402
from rag.core.retrieve.search import Searcher  # noqa: E402
from rag.core.workspace import Workspace, open_any  # noqa: E402

from conftest import TEST_EMBEDDING  # noqa: E402

SHAMMY = {**TEST_EMBEDDING, "vectors": {"backend": "shammy"}}
QUERIES = ["slow leak failures after run-in", "curing oven", "track-cli holds reason PressureLeak",
           "how do I reset the controller", "what does the pump warranty cover"]


def _search(ws, query: str, k: int = 8, file_type: str | None = None):
    store = RagStore(ws.rag_db, readonly=True)
    try:
        s = Searcher(store, make_embedder(ws.config["embedding"]), ws.config["search"])
        qvec = s.embedder.embed_query(query)
        return s._vector(qvec, 10, file_type), s.search(query, k=k, file_type=file_type)
    finally:
        store.close()


def _scores(ws, query: str, ids):
    """Cosine similarity of these chunks to the query, from the stored vectors."""
    store = RagStore(ws.rag_db, readonly=True)
    try:
        q = make_embedder(ws.config["embedding"]).embed_query(query)
        vecs = store.vectors.get(ids)
        return [float(np.dot(vecs[i][:len(q)], q) / np.linalg.norm(vecs[i]) / np.linalg.norm(q)) for i in ids]
    finally:
        store.close()


def _ranked(ws, query: str):
    """Scores of the vector hits, in rank order. Two workspaces agree when these match (chunks
    that tie may swap places, so ids and text are not compared)."""
    return np.array(_scores(ws, query, _search(ws, query)[0]))


def _agree(a, b) -> bool:
    return len(a) == len(b) and bool(np.allclose(a, b, atol=2e-3))


def _texts(ws, ids):
    store = RagStore(ws.rag_db, readonly=True)
    try:
        return [store.conn.execute("SELECT text_hash FROM chunks WHERE chunk_id = ?", (i,)).fetchone()[0] for i in ids]
    finally:
        store.close()


@pytest.fixture(scope="module")
def shammy_ws(tmp_path_factory, corpus):
    ws = Workspace.init(tmp_path_factory.mktemp("sh") / "ws", SHAMMY)
    stats = Builder(ws).build([corpus])
    assert stats.failed == 0, stats.errors
    return ws


def test_build_and_search_match_sqlite_vec(shammy_ws, built_ws):
    assert (shammy_ws.root / "vectors.shammy").is_file()
    a, b = RagStore(shammy_ws.rag_db, readonly=True), RagStore(built_ws.rag_db, readonly=True)
    try:
        assert a.vector_backend == "shammy" and b.vector_backend == "sqlite-vec"
        assert a.counts()["vectors"] == a.counts()["chunks"] == b.counts()["chunks"] > 0
        assert a.conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0] == 0
    finally:
        a.close(), b.close()
    for q in QUERIES:
        for ft in (None, "md"):
            got, full = _search(shammy_ws, q, file_type=ft)
            want, full_want = _search(built_ws, q, file_type=ft)
            # equally good chunks, rank for rank (chunks that tie can swap places)
            assert len(got) == len(want)
            assert np.allclose(_scores(shammy_ws, q, got), _scores(built_ws, q, want), atol=2e-3), (q, ft)
            assert len(full.results) == len(full_want.results) > 0
    assert check_workspace(shammy_ws.root)["ok"]


def test_rebuild_is_incremental_and_reuses_vectors(shammy_ws, corpus):
    stats = Builder(shammy_ws).build([corpus])
    assert stats.embedded == 0 and stats.failed == 0


def test_replace_remove_and_compact(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.md").write_text("# Pumps\n\nThe pump warranty covers seals for two years.\n" * 3)
    (src / "b.md").write_text("# Valves\n\nValves are torqued to 40 Nm.\n" * 3)
    ws = Workspace.init(tmp_path / "ws", SHAMMY)
    Builder(ws).build([src])
    (src / "a.md").write_text("# Pumps\n\nThe pump warranty now covers bearings for five years.\n" * 3)
    Builder(ws).build([src])
    assert Builder(ws).remove(["*b.md"]) == 1
    res = check_workspace(ws.root)
    assert res["ok"] and res["counts"]["vectors"] == res["counts"]["chunks"], res
    _, found = _search(ws, "pump warranty bearings")
    assert "five years" in found.results[0]["text"]
    assert all("Valves" not in r["text"] for r in found.results)
    done = compact_workspace(ws)
    assert done["rows_after"] == res["counts"]["chunks"] < done["rows_before"]
    assert check_workspace(ws.root)["ok"]
    assert "five years" in _search(ws, "pump warranty bearings")[1].results[0]["text"]


def test_export_unpack_and_verify(shammy_ws, tmp_path):
    pack = tmp_path / "kb.ragpack"
    manifest = export_pack(shammy_ws, pack)
    assert manifest["format"] == "ragpack/2" and manifest["vector_store"] == "shammy"
    assert "vectors.shammy" in read_manifest(pack)["files"]
    res = check_path(pack)
    assert res["ok"] and res["counts"]["vector_store"] == "shammy", res
    ws, tmp = open_any(pack)
    try:
        assert _agree(_ranked(ws, QUERIES[0]), _ranked(shammy_ws, QUERIES[0]))
    finally:
        tmp.cleanup()


def test_merge_across_stores(shammy_ws, built_ws, tmp_path):
    for dest_cfg, source in ((SHAMMY, built_ws), (TEST_EMBEDDING, shammy_ws)):
        dest = Workspace.init(tmp_path / f"dest-{dest_cfg is SHAMMY}", dest_cfg)
        merge(dest.root, [source.root])
        res = check_workspace(dest.root)
        assert res["ok"] and res["counts"]["vectors"] == res["counts"]["chunks"] > 0, res
        for q in QUERIES[:2]:
            assert _agree(_ranked(dest, q), _ranked(source, q))


def test_convert_both_ways(built_ws, tmp_path, capsys):
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(built_ws.root, root)
    ws = Workspace.open(root)
    before = [_ranked(ws, q) for q in QUERIES]
    size_before = ws.rag_db.stat().st_size

    assert cli_main(["convert", str(root)]) == 0
    assert "Moved" in capsys.readouterr().out
    res = check_workspace(root)
    assert res["ok"] and res["counts"]["vector_store"] == "shammy" and res["counts"]["vectors"] == res["counts"]["chunks"]
    assert ws.rag_db.stat().st_size < size_before
    assert all(_agree(_ranked(ws, q), b) for q, b in zip(QUERIES, before))
    assert convert_workspace(ws, "shammy")["changed"] is False

    back = convert_workspace(ws, "sqlite-vec")
    assert back["changed"] and back["missing"] == 0 and not (root / "vectors.shammy").exists()
    assert check_workspace(root)["ok"]
    assert all(_agree(_ranked(ws, q), b) for q, b in zip(QUERIES, before))
    assert cli_main(["compact", str(root)]) == 0


def test_an_interrupted_build_is_repaired(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for name in "abc":
        (src / f"{name}.md").write_text(f"# Note {name}\n\nDocument {name} is about topic {name}.\n" * 3)
    ws = Workspace.init(tmp_path / "ws", SHAMMY)
    Builder(ws).build([src])

    # chunks reached rag.db but their vectors never reached the disk (a crash before the commit)
    store = RagStore(ws.rag_db)
    lost = store.conn.execute("SELECT doc_id FROM documents WHERE display_path LIKE '%b.md'").fetchone()[0]
    store.vectors.delete_doc(lost)
    store.close()
    assert not check_workspace(ws.root)["ok"]

    # and vectors whose chunks were rolled back
    ix = shammy.Index.open(ws.root / "vectors.shammy", write=True)
    ix.add([900001, 900002], np.ones((2, 256), dtype=np.float32), doc_ids=777, file_types="md")
    ix.close()

    store = RagStore(ws.rag_db)
    assert store.vectors is not None and store.repaired == {"orphan_vectors": 2, "documents_to_rebuild": 1}
    assert store.conn.execute("SELECT status FROM documents WHERE doc_id = ?", (lost,)).fetchone()[0] == "pending"
    store.close()
    stats = Builder(ws).build([src])
    assert stats.failed == 0
    res = check_workspace(ws.root)
    assert res["ok"] and res["counts"]["vectors"] == res["counts"]["chunks"], res


def test_only_one_writer_and_bad_dims(shammy_ws, tmp_path):
    store = RagStore(shammy_ws.rag_db)
    assert store.vectors.count() > 0
    other = RagStore(shammy_ws.rag_db)
    with pytest.raises(VectorStoreError, match="already open for writing"):
        other.vectors
    other.close()
    store.close()
    with pytest.raises(VectorStoreError, match="power of two"):
        Workspace.init(tmp_path / "odd", {"embedding": {**TEST_EMBEDDING["embedding"], "dims": 300, "native_dims": 300},
                                          "vectors": {"backend": "shammy"}})
    assert cli_main(["init", str(tmp_path / "cli"), "--embedder", "hash-test", "--vectors", "shammy"]) == 0
    assert (tmp_path / "cli" / "vectors.shammy").is_file()


def test_import_from_sqlite(built_ws, tmp_path, capsys):
    from shammy.convert import from_sqlite, main

    out = tmp_path / "from-vec0.shammy"
    res = from_sqlite(built_ws.rag_db, out, id="chunk_id", doc="doc_id", file_type="file_type", model="hash-test")
    store = RagStore(built_ws.rag_db, readonly=True)
    try:
        n = store.counts()["vectors"]
        cid, blob = store.conn.execute("SELECT chunk_id, embedding FROM chunk_vec LIMIT 1").fetchone()
    finally:
        store.close()
    assert res["rows"] == n and res["skipped"] == 0 and res["ids_file"] is None
    ix = shammy.Index.open(out)
    assert len(ix) == n and ix.search(np.frombuffer(blob, dtype=np.float32), k=1)[0].id == cid

    # an ordinary table: 768-value vectors (padded to 1024), JSON text, text ids
    db = tmp_path / "plain.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE items(key TEXT, emb TEXT, kind TEXT)")
    rng = np.random.default_rng(3)
    vecs = rng.standard_normal((40, 768)).astype(np.float32)
    conn.executemany("INSERT INTO items VALUES (?,?,?)",
                     [(f"k{i}", "[" + ",".join(map(str, v)) + "]", "a" if i % 2 else "b") for i, v in enumerate(vecs)])
    conn.commit()
    conn.close()
    out2 = tmp_path / "plain.shammy"
    assert main(["sqlite", str(db), str(out2), "--table", "items", "--id", "key", "--vector", "emb", "--type", "kind"]) == 0
    text = capsys.readouterr().out
    assert "40 rows" in text and "padded to 1024" in text and "ids.csv" in text
    ix = shammy.Index.open(out2)
    hit = ix.search(vecs[17], k=1, file_types=["a"])[0]
    assert hit.id == 18 and (tmp_path / "plain.shammy.ids.csv").read_text().splitlines()[17] == "18,k17"
    assert main(["sqlite", str(db), str(tmp_path / "x.shammy"), "--table", "items; DROP TABLE items"]) == 1


def test_slim_pack(shammy_ws, built_ws, tmp_path):
    from rag.core.db.pack import PackError

    full, slim = tmp_path / "full.ragpack", tmp_path / "slim.ragpack"
    export_pack(shammy_ws, full)
    manifest = export_pack(shammy_ws, slim, slim=True)
    assert manifest["vector_tier"] == "slim"
    import zipfile

    sizes = [zipfile.ZipFile(p).getinfo("vectors.shammy").file_size for p in (full, slim)]
    assert sizes[1] < sizes[0]   # at 4096 dims: 4.6 KB a chunk against 12.8 KB (this tiny file is mostly padding)
    assert check_path(slim)["ok"]
    ws, tmp = open_any(slim)
    try:
        hits = _search(ws, QUERIES[0])[0]
        best = _ranked(shammy_ws, QUERIES[0])
        assert len(hits) == len(best)
        # the same best chunk (by text), found without the float16 copies
        assert _texts(ws, hits[:1]) == _texts(shammy_ws, _search(shammy_ws, QUERIES[0])[0][:1])
    finally:
        tmp.cleanup()
    plan = merge(tmp_path / "dest", [slim], dry_run=True)
    assert not plan["sources"][0]["compatible"] and "slim" in plan["sources"][0]["reason"]
    with pytest.raises(PackError, match="rag convert"):
        export_pack(built_ws, tmp_path / "no.ragpack", slim=True)
