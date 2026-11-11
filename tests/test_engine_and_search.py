from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import pytest

from rag.builder.engine import Builder
from rag.core.db.pack import PackError, export_pack, unpack_pack
from rag.core.db.sqlguard import QueryRejected, run_query
from rag.core.db.store import EmbeddingMismatch, RagStore
from rag.core.db.tables import TablesStore
from rag.core.workspace import Workspace

from conftest import TEST_EMBEDDING


def _consistent(ws: Workspace) -> tuple[int, int, int]:
    s = RagStore(ws.rag_db, readonly=True)
    try:
        q = s.conn.execute
        return (q("SELECT COUNT(*) FROM chunks").fetchone()[0], q("SELECT COUNT(*) FROM chunks_fts").fetchone()[0],
                q("SELECT COUNT(*) FROM chunk_vec").fetchone()[0])
    finally:
        s.close()


def test_built_workspace_is_consistent(built_ws):
    chunks, fts, vec = _consistent(built_ws)
    assert chunks == fts == vec > 30
    t = TablesStore(built_ws.tables_db, readonly=True)
    names = {r["table_name"] for r in t.catalog()}
    t.close()
    assert {"t_run_in_yields_q3_yields", "t_run_in_yields_q3_limits", "t_calibration"} <= names


def test_incremental_update_change_and_prune(tmp_path, corpus):
    src = tmp_path / "docs"
    shutil.copytree(corpus, src)
    ws = Workspace.init(tmp_path / "ws", TEST_EMBEDDING)
    first = Builder(ws).build([src])
    assert first.added == 11 and first.failed == 0

    again = Builder(ws).build([src])
    assert again.unchanged == 11 and again.added == again.updated == 0

    (src / "handoff.txt").write_text("Completely new handoff note about the torque tester.", encoding="utf-8")
    (src / "calibration.csv").unlink()
    (src / "Big Table.md").write_text(
        "# Parts\n\n| part | qty |\n| --- | --- |\n" + "".join(f"| P-{i} | {i} |\n" for i in range(45)),
        encoding="utf-8")
    third = Builder(ws).build([src], prune=True)
    assert (third.updated, third.added, third.removed) == (1, 1, 1)
    chunks, fts, vec = _consistent(ws)
    assert chunks == fts == vec

    store = RagStore(ws.rag_db, readonly=True)
    texts = " ".join(r[0] for r in store.conn.execute("SELECT text FROM chunks"))
    pointer = store.conn.execute("SELECT table_name, text FROM chunks WHERE kind='table' AND table_name IS NOT NULL").fetchone()
    store.close()
    assert "torque" in texts and "hose crimper on station 4" not in texts
    assert pointer is not None and "of 45; the whole table is in tables.db" in pointer["text"]
    t = TablesStore(ws.tables_db, readonly=True)
    names = {r["table_name"] for r in t.catalog()}
    t.close()
    assert "t_calibration" not in names and pointer["table_name"] in names
    res = run_query(ws.tables_db, f'SELECT COUNT(*), SUM(qty) FROM "{pointer["table_name"]}"')
    assert res["rows"] == [[45, sum(range(45))]]


def test_files_read_by_an_older_extractor_are_reextracted(tmp_path, corpus):
    import sqlite3

    ws = Workspace.init(tmp_path / "ws", TEST_EMBEDDING)
    assert Builder(ws).build([corpus]).added == 11
    db = sqlite3.connect(ws.rag_db)
    db.execute("UPDATE documents SET extractor = replace(extractor, '@v', '@old') WHERE file_type = 'pdf'")
    db.commit()
    n_pdf = db.execute("SELECT COUNT(*) FROM documents WHERE file_type = 'pdf'").fetchone()[0]
    db.close()
    again = Builder(ws).build([corpus])
    assert again.updated == n_pdf > 0 and again.unchanged == 11 - n_pdf
    assert Builder(ws).build([corpus]).unchanged == 11
    chunks, fts, vec = _consistent(ws)
    assert chunks == fts == vec


def test_embedding_mismatch_is_refused(tmp_path, corpus):
    ws = Workspace.init(tmp_path / "ws", TEST_EMBEDDING)
    Builder(ws).build([corpus / "handoff.txt"])
    ws.config = ws.config.with_overrides({"embedding": {"dims": 128, "native_dims": 256}})
    with pytest.raises(EmbeddingMismatch):
        Builder(ws).build([corpus / "handoff.txt"])
    stats = Builder(ws).build([corpus / "handoff.txt"], force=True)
    assert stats.added == 1


def test_failed_file_is_isolated(tmp_path, corpus):
    src = tmp_path / "docs"
    src.mkdir()
    shutil.copy(corpus / "handoff.txt", src)
    (src / "broken.xlsx").write_bytes(b"PK\x03\x04garbage that is not a workbook")
    ws = Workspace.init(tmp_path / "ws", TEST_EMBEDDING)
    stats = Builder(ws).build([src])
    assert stats.added == 1 and stats.failed == 1
    assert stats.errors[0]["path"].endswith("broken.xlsx")


def test_export_unpack_and_tamper_detection(tmp_path, built_ws):
    pack = tmp_path / "p.ragpack"
    manifest = export_pack(built_ws, pack, name="t")
    assert manifest["embedding"]["dims"] == 256 and manifest["counts"]["tables"] == 3
    assert any(k.startswith("assets/") for k in manifest["files"])
    dest = tmp_path / "unpacked"
    unpack_pack(pack, dest)
    s = RagStore(dest / "rag.db", readonly=True)
    assert s.counts()["chunks"] == manifest["counts"]["chunks"]
    s.close()

    bad = tmp_path / "bad.ragpack"
    with zipfile.ZipFile(pack) as zin, zipfile.ZipFile(bad, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.toml":
                data += b"\n# tampered\n"
            zout.writestr(item, data)
    with pytest.raises(PackError):
        unpack_pack(bad, tmp_path / "bad_out")


@pytest.mark.parametrize("query,expected_file", [
    ("minimum pull strength aluminum wire", "QP-114 Pull Test.pdf"),
    ("curing oven drifted deviation report", "DR-2291 scan.pdf"),
    ("motor mount stage temperature", "Motor Mount WI.docx"),
    ("GetLotYield", "lots.sql"),
    ("CanRelease minimum run-in yield", "LotReleaseService.cs"),
    ("yield_pct percentage", "runin.py"),
    ("safety order coverall", "safety.html"),
    ("hose crimper alarm jaw", "handoff.txt"),
])
def test_search_finds_expected_file(searcher, query, expected_file):
    res = searcher.search(query, k=3)
    files = [r["source"]["file"] for r in res.results]
    assert any(f.endswith(expected_file) for f in files), files


def test_table_card_surfaces_and_sql_answers(searcher, built_ws):
    res = searcher.search("run-in pass fail by lot and device", k=6)
    assert any(t["table_name"] == "t_run_in_yields_q3_yields" for t in res.tables)
    out = run_query(built_ws.tables_db,
                    "SELECT lot FROM t_run_in_yields_q3_yields WHERE test_fail > 10")
    assert out["rows"] == [["24A-0193"]]
    t = TablesStore(built_ws.tables_db, readonly=True)
    d = t.describe("t_run_in_yields_q3_yields")
    t.close()
    assert "CREATE TABLE" in d["create_table"] and len(d["sample_rows"]) == 3
    assert {c["name"]: c["type"] for c in d["columns"]}["date"] == "date"


def test_markdown_rendering_has_sources(searcher):
    md = searcher.search("gap limits by package", k=2).to_markdown()
    assert md.startswith("[S1] corpus") and "chunk" in md


@pytest.mark.parametrize("sql", [
    "DROP TABLE t_calibration", "SELECT * FROM sqlite_master", "PRAGMA table_info(t_calibration)",
    "ATTACH DATABASE 'x.db' AS x", "SELECT 1; SELECT 2", "SELECT load_extension('x')",
])
def test_sql_guard_rejects(built_ws, sql):
    with pytest.raises(QueryRejected):
        run_query(built_ws.tables_db, sql)


def test_sql_guard_time_limit(built_ws):
    with pytest.raises(QueryRejected, match="stopped"):
        run_query(built_ws.tables_db,
                  "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c",
                  time_limit_ms=200)


def test_eval_runs(tmp_path, searcher):
    from rag.eval import run_eval

    golden = tmp_path / "g.jsonl"
    golden.write_text("\n".join(json.dumps(q) for q in [
        {"id": "a", "type": "doc_table", "question": "minimum pull strength 5 mil aluminum",
         "expect": [{"file": "*QP-114*", "page": 1}]},
        {"id": "b", "type": "scan", "question": "oven zone 3 recalibrate weekly",
         "expect": [{"file": "*DR-2291*"}]},
        {"id": "c", "type": "sheet", "question": "run-in yields pass fail per lot",
         "expect": [{"file": "*Run-in*", "kind": "table_card"}]},
    ]), encoding="utf-8")
    report = run_eval(searcher, golden)
    assert report["overall"]["n"] == 3 and report["overall"]["recall@20"] == 1.0
    assert "retrieved" not in report["questions"][0]

    from rag.eval import report_markdown

    with golden.open("a", encoding="utf-8") as fh:
        fh.write("\n" + json.dumps({"id": "z", "question": "price of a hose crimper", "expect": []}))
    shown = run_eval(searcher, golden, show=3)
    for q in shown["questions"]:
        ranks = [c["rank"] for c in q["retrieved"]]
        assert ranks[:3] == list(range(1, len(ranks[:3]) + 1))
        assert any(c["match"] for c in q["retrieved"]) == bool(q["first_hit_rank"])
        if q["first_hit_rank"] and q["first_hit_rank"] > 3:
            assert q["retrieved"][-1]["rank"] == q["first_hit_rank"] and q["retrieved"][-1]["match"]
    assert len(shown["unanswerable"][0]["retrieved"]) == 3
    md = report_markdown(shown)
    assert "## a: minimum pull strength" in md and "✅" in md and "# Not answerable" in md


@pytest.mark.parametrize("module", ["pysqlite3", "sqlean"])
def test_fallback_sqlite_modules_build_and_search(tmp_path, corpus, module):
    """Run a build, search, SQL guard and export through a drop-in sqlite module."""
    import subprocess
    import sys

    pytest.importorskip({"pysqlite3": "pysqlite3.dbapi2", "sqlean": "sqlean"}[module])
    script = f"""
import json, os, sys
from pathlib import Path
from rag.core.db.connect import sqlite_module
from rag.builder.engine import Builder
from rag.core.workspace import Workspace
from rag.core.db.sqlguard import run_query, QueryRejected
from rag.core.db.pack import export_pack
assert sqlite_module().__name__.startswith({module!r}), sqlite_module().__name__
ws = Workspace.init(Path({str(tmp_path / 'ws')!r}), {TEST_EMBEDDING!r})
stats = Builder(ws).build([Path({str(corpus / 'calibration.csv')!r}), Path({str(corpus / 'handoff.txt')!r})])
assert stats.failed == 0, stats.errors
assert run_query(ws.tables_db, "SELECT COUNT(*) FROM t_calibration")["rows"] == [[3]]
try:
    run_query(ws.tables_db, "DROP TABLE t_calibration"); sys.exit("guard failed")
except QueryRejected:
    pass
export_pack(ws, Path({str(tmp_path / 'p.ragpack')!r}))
print("ok")
"""
    env = {**__import__("os").environ, "RAG_SQLITE_MODULE": module}
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr[-2000:]
