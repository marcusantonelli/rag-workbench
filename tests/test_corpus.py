"""The Halden test corpus: every golden question's answer location exists in the
built pack, and every table question's SQL returns the expected answer."""

from __future__ import annotations

import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from rag.builder.engine import Builder
from rag.core.db.sqlguard import run_query
from rag.core.workspace import Workspace
from rag.eval import _matches

from conftest import TEST_EMBEDDING

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_test_corpus", ROOT / "tools" / "make_test_corpus.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path_factory.mktemp("halden")
    mod.main(out)
    ws = Workspace.init(out / "ws", TEST_EMBEDDING)
    stats = Builder(ws).build([out / "docs"])
    return out, ws, stats, mod.questions()


def test_corpus_builds_cleanly(corpus):
    out, ws, stats, _ = corpus
    assert stats.failed == 0, stats.errors
    assert stats.added == 29          # lock file, bin/, *.Designer.cs and *_files/ excluded
    db = sqlite3.connect(ws.rag_db)
    n_spec = db.execute("SELECT COUNT(*) FROM chunks c JOIN documents d USING(doc_id) "
                        "WHERE d.display_path LIKE '%DS-150%' AND c.kind = 'table' AND c.table_name IS NOT NULL").fetchone()[0]
    assert n_spec == 1                # the 36-row table is one table even though it spans two pages
    t = sqlite3.connect(ws.tables_db)
    assert t.execute("SELECT COUNT(*) FROM t_ds_150_device_parameter_specification_table_1").fetchone()[0] == 36


def test_every_expected_location_exists(corpus):
    out, ws, _, questions = corpus
    db = sqlite3.connect(ws.rag_db)
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute(
        "SELECT c.*, d.display_path FROM chunks c JOIN documents d USING(doc_id)")]
    missing = []
    for q in questions:
        for exp in q["expect"]:
            if not any(_matches(r, exp) for r in rows):
                missing.append((q["id"], exp))
    assert not missing, missing


def test_sql_answers_match(corpus):
    out, ws, _, questions = corpus
    for q in questions:
        if not q["sql"]:
            continue
        res = run_query(ws.tables_db, q["sql"])
        assert res["rows"], q["id"]
        first = res["rows"][0][0]
        if q["id"] in ("spec-1", "csv-4"):
            got = ", ".join(str(r[0]) for r in res["rows"])
            assert got == q["answer"], (q["id"], got, q["answer"])
        else:
            assert str(first) in q["answer"], (q["id"], first, q["answer"])


def test_golden_file_matches_questions(corpus):
    out, _, _, questions = corpus
    lines = [json.loads(l) for l in (out / "questions" / "golden.jsonl").read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    assert [l["id"] for l in lines] == [q["id"] for q in questions]
    assert (out / "questions" / "ANSWERS.md").read_text().count("Answer:") == len(questions)
