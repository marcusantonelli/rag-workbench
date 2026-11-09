"""Workspace markers, `rag verify` and `rag list`."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from rag.builder.engine import Builder
from rag.cli import main as cli_main
from rag.core.check import check_path
from rag.core.db.pack import export_pack, unpack_pack
from rag.core.listing import list_knowledge_bases
from rag.core.workspace import MARKER, NotAWorkspace, Workspace, classify, read_marker

import pytest

from conftest import TEST_EMBEDDING


def test_init_writes_a_marker_and_open_needs_a_real_workspace(tmp_path):
    ws = Workspace.init(tmp_path / "plant", TEST_EMBEDDING, name="Plant docs")
    m = read_marker(ws.root)
    assert m["format"] == "rag-workspace" and m["name"] == "Plant docs" and len(m["id"]) == 36
    assert ws.name == "Plant docs" and classify(ws.root) == "workspace"
    assert Workspace.open(ws.root).name == "Plant docs"

    # A stray rag.db is not enough.
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "rag.db").write_bytes(b"not a database")
    (fake / "config.toml").write_text("", encoding="utf-8")
    assert classify(fake) is None
    with pytest.raises(NotAWorkspace):
        Workspace.open(fake)
    with pytest.raises(NotAWorkspace):
        Workspace.open(tmp_path / "empty-dir")


def test_older_workspaces_get_a_marker_when_opened(tmp_path):
    ws = Workspace.init(tmp_path / "old", TEST_EMBEDDING)
    (ws.root / MARKER).unlink()
    assert classify(ws.root) == "legacy"
    assert check_path(ws.root)["ok"]   # only a warning
    Workspace.open(ws.root)
    assert classify(ws.root) == "workspace" and read_marker(ws.root)["name"] == "old"


def test_unpacked_pack_is_a_named_workspace(tmp_path, built_ws):
    pack = tmp_path / "p.ragpack"
    manifest = export_pack(built_ws, pack, name="Line A")
    dest = tmp_path / "unpacked"
    unpack_pack(pack, dest)
    m = read_marker(dest)
    assert m["name"] == "Line A" and m["from_pack"] == manifest["pack_id"]


def test_verify_finds_problems(tmp_path, corpus, built_ws):
    res = check_path(built_ws.root)
    assert res["ok"] and res["counts"]["chunks"] > 30 and res["counts"]["tables"] == 3, res["problems"]
    pack = tmp_path / "p.ragpack"
    export_pack(built_ws, pack)
    res = check_path(pack)
    assert res["ok"] and res["kind"] == "pack" and "match" in res["checksums"]

    ws = Workspace.init(tmp_path / "w", TEST_EMBEDDING)
    Builder(ws).build([corpus / "calibration.csv", corpus / "handoff.txt"])
    assert check_path(ws.root)["ok"]
    conn = sqlite3.connect(ws.tables_db)
    name = conn.execute("SELECT table_name FROM _catalog").fetchone()[0]
    conn.execute(f'DROP TABLE "{name}"')
    conn.commit()
    conn.close()
    conn = sqlite3.connect(ws.rag_db)
    conn.execute("DELETE FROM chunks_fts WHERE rowid = (SELECT MIN(chunk_id) FROM chunks)")
    conn.commit()
    conn.close()
    res = check_path(ws.root)
    msgs = " | ".join(p["message"] for p in res["problems"])
    assert not res["ok"] and f"table {name} is listed but doesn't exist" in msgs and "keyword index rows" in msgs
    assert cli_main(["verify", str(ws.root)]) == 1
    assert not check_path(tmp_path / "nowhere")["ok"]


def test_list_shows_workspaces_and_packs(tmp_path, corpus, capsys, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("RAG_WORKBENCH_HOME", str(home))
    folder = home / "workspaces"
    a = Workspace.init(folder / "line-a", TEST_EMBEDDING, name="Line A")
    Builder(a).build([corpus / "handoff.txt"])
    Workspace.init(folder / "empty-one", TEST_EMBEDDING)
    (folder / "not-a-workspace").mkdir()
    export_pack(a, folder / "line-a.ragpack", name="Line A pack")

    rows = {r["name"]: r for r in list_knowledge_bases(folder)}
    assert set(rows) == {"Line A", "empty-one", "Line A pack"}
    assert rows["Line A"]["documents"] == 1 and rows["Line A"]["kind"] == "workspace"
    assert rows["Line A pack"]["kind"] == "pack" and rows["empty-one"]["note"] == "empty"

    assert cli_main(["list"]) == 0          # the app's workspaces folder by default
    out = capsys.readouterr().out
    assert "Line A pack" in out and "line-a" in out and "not-a-workspace" not in out
    assert cli_main(["list", str(folder), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)["items"]) == 3
    assert cli_main(["list", str(tmp_path / "missing")]) == 0
    assert "doesn't exist" in capsys.readouterr().out
