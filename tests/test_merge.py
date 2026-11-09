from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest

from rag.builder.engine import Builder
from rag.cli import main as cli_main
from rag.core.db.merge import MergeError, merge
from rag.core.db.pack import export_pack
from rag.core.db.sqlguard import run_query
from rag.core.db.store import RagStore
from rag.core.db.tables import TablesStore
from rag.core.models.embed import make_embedder
from rag.core.retrieve.search import Searcher
from rag.core.workspace import Workspace

from conftest import TEST_EMBEDDING


def _ws(root: Path, *paths: Path, overrides: dict | None = None) -> Workspace:
    ws = Workspace.init(root, overrides or TEST_EMBEDDING)
    stats = Builder(ws).build(list(paths))
    assert stats.failed == 0, stats.errors
    return ws


def _counts(ws: Workspace) -> dict[str, int]:
    s = RagStore(ws.rag_db, readonly=True)
    q = lambda sql: s.conn.execute(sql).fetchone()[0]  # noqa: E731
    try:
        return {"docs": q("SELECT COUNT(*) FROM documents"), "chunks": q("SELECT COUNT(*) FROM chunks"),
                "fts": q("SELECT COUNT(*) FROM chunks_fts"), "vec": q("SELECT COUNT(*) FROM chunk_vec")}
    finally:
        s.close()


def _search(ws: Workspace, query: str, k: int = 5):
    store = RagStore(ws.rag_db, readonly=True)
    try:
        return Searcher(store, make_embedder(ws.config["embedding"]), ws.config["search"]).search(query, k=k)
    finally:
        store.close()


@pytest.fixture()
def two_kbs(tmp_path, corpus):
    """Two knowledge bases with no files in common, one exported as a pack."""
    a = _ws(tmp_path / "a", corpus / "QP-114 Pull Test.pdf", corpus / "Burn-in Yields Q3.xlsx", corpus / "handoff.txt")
    b = _ws(tmp_path / "b", corpus / "Die Attach WI.docx", corpus / "calibration.csv", corpus / "code")
    pack_b = tmp_path / "b.ragpack"
    export_pack(b, pack_b, name="Line B")
    return a, b, pack_b


def test_merge_packs_and_workspaces_into_a_new_workspace(tmp_path, two_kbs):
    a, b, pack_b = two_kbs
    dest = tmp_path / "merged"
    res = merge(dest, [a.root, pack_b])
    assert res["created"]
    sa, sb = res["sources"]
    assert sa["compatible"] and sb["compatible"] and sb["name"] == "Line B"
    assert sa["add"] == _counts(a)["docs"] and sb["add"] == _counts(b)["docs"]

    ws = Workspace.open(dest)
    c = _counts(ws)
    assert c["docs"] == _counts(a)["docs"] + _counts(b)["docs"]
    assert c["chunks"] == c["fts"] == c["vec"] == _counts(a)["chunks"] + _counts(b)["chunks"]

    # Both halves are searchable, by vector and keyword, with the copied vectors.
    files = [r["source"]["file"] for r in _search(ws, "minimum pull strength aluminum wire").results]
    assert any(f.endswith("QP-114 Pull Test.pdf") for f in files)
    files = [r["source"]["file"] for r in _search(ws, "CanRelease minimum burn-in yield").results]
    assert any(f.endswith("LotReleaseService.cs") for f in files)

    # Tables from both, queryable, with their cards pointing at real chunks.
    t = TablesStore(ws.tables_db, readonly=True)
    cat = {r["table_name"]: r for r in t.catalog()}
    t.close()
    assert {"t_burn_in_yields_q3_yields", "t_calibration"} <= set(cat)
    assert run_query(ws.tables_db, "SELECT COUNT(*) FROM t_calibration")["rows"][0][0] > 0
    s = RagStore(ws.rag_db, readonly=True)
    for name, row in cat.items():
        card = s.conn.execute("SELECT kind, table_name, doc_id FROM chunks WHERE chunk_id = ?",
                              (row["card_chunk_id"],)).fetchone()
        assert card["kind"] == "table_card" and card["table_name"] == name and card["doc_id"] == row["doc_id"]
    # Figure crops came along.
    for (asset,) in s.conn.execute("SELECT asset_path FROM chunks WHERE asset_path IS NOT NULL"):
        assert (dest / asset).is_file(), asset
    meta = s.meta()
    s.close()
    assert meta["embed_model"] == "hash-test" and meta["embed_dims"] == "256"


def test_merging_again_skips_everything(tmp_path, two_kbs):
    a, _, pack_b = two_kbs
    dest = tmp_path / "m"
    merge(dest, [a.root, pack_b])
    before = _counts(Workspace.open(dest))
    res = merge(dest, [pack_b, a.root])
    assert all(s["add"] == 0 and s["replace"] == 0 and s["identical"] == s["documents"] for s in res["sources"])
    assert _counts(Workspace.open(dest)) == before


def test_dry_run_changes_nothing(tmp_path, two_kbs):
    a, _, pack_b = two_kbs
    before = _counts(a)
    res = merge(a.root, [pack_b], dry_run=True)
    assert res["dry_run"] and res["sources"][0]["add"] > 0 and res["sources"][0]["chunks"] > 0
    assert _counts(a) == before
    res = merge(tmp_path / "not-yet", [pack_b], dry_run=True)
    assert res["sources"][0]["add"] > 0 and not (tmp_path / "not-yet").exists()


def test_different_embeddings_are_refused(tmp_path, corpus, two_kbs):
    a, *_ = two_kbs
    small = _ws(tmp_path / "small", corpus / "Seal Leak.md",
                overrides={"embedding": {**TEST_EMBEDDING["embedding"], "dims": 128}})
    before = _counts(a)
    res = merge(a.root, [small.root])
    s = res["sources"][0]
    assert not s["compatible"] and "embed_dims" in s["reason"] and s["add"] == 0
    assert _counts(a) == before
    # An empty destination that stores another vector size is refused too.
    empty = Workspace.init(tmp_path / "empty", TEST_EMBEDDING)
    s = merge(empty.root, [small.root])["sources"][0]
    assert not s["compatible"] and "dimensions" in s["reason"]


def test_colliding_table_names_are_renamed_everywhere(tmp_path, corpus):
    for side, value in (("x", "1.5"), ("y", "9.75")):
        d = tmp_path / side
        d.mkdir()
        (d / "limits.csv").write_text(f"step,max_value\nseal,{value}\nbond,{value}\n", encoding="utf-8")
    a = _ws(tmp_path / "wa", tmp_path / "x" / "limits.csv")
    b = _ws(tmp_path / "wb", tmp_path / "y" / "limits.csv")
    res = merge(a.root, [b.root])
    assert res["sources"][0]["renamed_tables"] == [["t_limits", "t_limits_2"]]
    assert run_query(a.tables_db, "SELECT max_value FROM t_limits_2 WHERE step = 'seal'")["rows"] == [[9.75]]
    assert run_query(a.tables_db, "SELECT max_value FROM t_limits WHERE step = 'seal'")["rows"] == [[1.5]]
    s = RagStore(a.rag_db, readonly=True)
    t = TablesStore(a.tables_db, readonly=True)
    card_id = {r["table_name"]: r["card_chunk_id"] for r in t.catalog()}["t_limits_2"]
    card = s.conn.execute("SELECT text, embed_text, table_name FROM chunks WHERE chunk_id = ?", (card_id,)).fetchone()
    assert card["table_name"] == "t_limits_2" and "t_limits_2" in card["text"] and "t_limits_2" in card["embed_text"]
    hits = s.conn.execute("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 't_limits_2'").fetchall()
    assert (card_id,) in [tuple(h) for h in hits]
    s.close()
    t.close()


def test_conflicts_and_copies(tmp_path, corpus):
    src = tmp_path / "docs"
    src.mkdir()
    f = src / "notes.txt"
    f.write_text("Bonder 4 capillary changed every 500k bonds.\n", encoding="utf-8")
    old = _ws(tmp_path / "old", f)
    f.write_text("Bonder 4 capillary changed every 350k bonds after the October review.\n", encoding="utf-8")
    os.utime(f, (f.stat().st_mtime + 60,) * 2)
    new = _ws(tmp_path / "new", f)

    keep = merge(old.root, [new.root], on_conflict="keep", dry_run=True)["sources"][0]
    assert keep["kept"] == 1 and keep["conflicts"][0]["decision"] == "kept"
    res = merge(new.root, [old.root])["sources"][0]          # default: an older incoming file loses
    assert res["kept"] == 1
    res = merge(old.root, [new.root])["sources"][0]          # and a newer one wins
    assert res["replace"] == 1 and _counts(old)["docs"] == 1
    assert "350k" in _search(old, "capillary change interval").results[0]["text"]

    # The same content under another path is skipped unless asked for.
    other = tmp_path / "elsewhere"
    other.mkdir()
    shutil.copy(f, other / "copy of notes.txt")
    dup = _ws(tmp_path / "dup", other / "copy of notes.txt")
    assert merge(old.root, [dup.root])["sources"][0]["copies"] == 1
    assert _counts(old)["docs"] == 1
    assert merge(old.root, [dup.root], keep_copies=True)["sources"][0]["add"] == 1
    assert _counts(old)["docs"] == 2


def test_merge_refuses_itself_and_non_workspaces(tmp_path, two_kbs):
    a, *_ = two_kbs
    with pytest.raises(MergeError):
        merge(a.root, [a.root])
    with pytest.raises(MergeError):
        merge(a.root, [tmp_path / "nothing-here"])


def test_cli_merge(tmp_path, two_kbs, capsys):
    a, _, pack_b = two_kbs
    assert cli_main(["merge", str(tmp_path / "c"), str(a.root), str(pack_b), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "Would merge Line B" in out and "Nothing was changed" in out
    assert cli_main(["merge", str(tmp_path / "c"), str(a.root), str(pack_b)]) == 0
    out = capsys.readouterr().out
    assert "Created workspace" in out and "Merged Line B" in out
    assert cli_main(["merge", str(tmp_path / "c"), str(pack_b)]) == 0
    assert "skipped" in capsys.readouterr().out
