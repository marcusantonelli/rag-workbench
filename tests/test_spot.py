"""Procedures and structure: the spot file, embedding context 2, and procedure-aware search."""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from rag.builder.engine import Builder
from rag.core import spot
from rag.core.chunk.chunker import Chunker, ChunkSettings
from rag.core.db.pack import export_pack, unpack_pack
from rag.core.db.store import RagStore
from rag.core.ingest.figures import confirmed_text
from rag.core.models.embed import make_embedder
from rag.core.retrieve.search import Searcher
from rag.core.structure import heading_number, identity
from rag.core.types import Block, ExtractedDoc
from rag.core.workspace import Workspace

from conftest import TEST_EMBEDDING

SEALER = """# Lid sealer operation

Title: Lid sealer operation | Specification # WI-2210 Revision: C

## 1.0 Purpose

1.1 This document gives the steps for running the lid sealer.

## 7.0 Procedure

### 7.1 Verify oven calibration

- Verify that the calibration sticker on the oven door has not expired.
- Record the calibration date in the log book.

### 7.2 Load the oven

- Place the carrier tray on the oven shelf.
- Close the oven door and press START.
- CAUTION: the door surface is hot.

### 7.3 Seal the lids

- Position the lid on the package and align the edges on the seal ring.
- Press the weld button and hold it until the light turns green.
- If the weld fails, see Section 8.2 and see MF164.

## 8.0 Electrode Replacement

### 8.1 Electrode removal

- Use the Allen wrench to unscrew the center screw and remove the electrode.
- Inspect the white washer and replace it when damaged.

### 8.2 Electrode installation

- Install the new electrode on the arbor and tighten the screw.
- Rotate the electrode to check it turns freely.
"""

MANUAL = """# Sealer user manual

## 1 Overview

The sealer joins lids to packages in a dry nitrogen atmosphere. It has two ovens and a weld station.

## 2 Specifications

### 2.1 Power

The sealer runs on 208 V three-phase power and draws up to 30 A when both ovens heat.

### 2.2 Gas

Nitrogen at 60 psi feeds the glove box; the oxygen sensor reads in ppm.
"""

LIST = """# Resetting the controller

To reset the controller after a fault:

1. Press the STOP button on the front panel.
2. Turn the main switch off and wait ten seconds.
3. Turn the main switch on and wait for the green light.
4. Press RESET and confirm on the screen.

The controller keeps its settings through a reset.
"""


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("procs")
    (d / "WI-2210 sealer.md").write_text(SEALER, encoding="utf-8")
    (d / "manual.md").write_text(MANUAL, encoding="utf-8")
    (d / "reset.md").write_text(LIST, encoding="utf-8")
    (d / "MF164.md").write_text("# MF164 fixture list\n\nDocument # MF164\n\nFixture A fits TO-39, fixture B fits SMD-2.\n",
                                encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def ws(corpus, tmp_path_factory):
    w = Workspace.init(tmp_path_factory.mktemp("spotws") / "ws", TEST_EMBEDDING)
    assert Builder(w).build([corpus]).failed == 0
    return w


def _searcher(w, engine="reference", **cfg):
    store = RagStore(w.rag_db, readonly=True)
    return store, Searcher(store, make_embedder(w.config["embedding"]), {**w.config["search"], **cfg}, engine=engine)


def test_heading_numbers_and_identity():
    assert heading_number("7.0 Procedure/Specification:") == (7,)
    assert heading_number("8.1 Electrode Removal:") == (8, 1)
    assert heading_number("Step 3 - Remove the cover") == (3,)
    assert heading_number("1.5 mm gap") is None
    assert heading_number("Start-up & Login") is None
    i = identity(None, ["Title: | Specification # D02 Revision: A"], "docs/x.pdf")
    assert (i.number, i.revision) == ("D02", "A")
    # a redacted number doesn't swallow the next field
    assert identity("T", ["Specification # Revision: E1"]).number is None
    assert identity(None, ["no title block"], "docs/WI-1234 Seal.pdf").number == "WI-1234"


def test_numbered_headings_nest_by_their_numbers():
    doc = ExtractedDoc(file_type="pdf", title=None, blocks=[
        Block("heading", "7.0 Procedure", level=1), Block("heading", "7.1 Verify", level=2),
        Block("paragraph", "Verify the sticker on the oven door has not expired today."),
        Block("heading", "8.0 Electrode Replacement", level=2),     # the file got the level wrong
        Block("heading", "8.1 Removal", level=3), Block("paragraph", "Remove the screw and lift the electrode out."),
    ])
    chunks = Chunker(ChunkSettings(), lambda t: len(t.split())).chunk(doc, "wi.pdf")
    assert [c.heading_path for c in chunks] == ["7.0 Procedure › 7.1 Verify", "8.0 Electrode Replacement › 8.1 Removal"]


def test_context_two_names_the_document_and_one_is_unchanged():
    doc = ExtractedDoc(file_type="pdf", title="Lid sealer", blocks=[
        Block("paragraph", "Specification # WI-2210 Revision: C"), Block("heading", "Load", level=1),
        Block("paragraph", "Place the carrier tray on the oven shelf and close the door.")])
    count = lambda t: len(t.split())  # noqa: E731
    v1 = Chunker(ChunkSettings(context=1), count).chunk(doc, "a/wi.pdf")
    v2 = Chunker(ChunkSettings(context=2), count).chunk(doc, "a/wi.pdf")
    assert v1[-1].embed_text.startswith("a/wi.pdf › Load\n")
    assert v2[-1].embed_text.startswith("Lid sealer (WI-2210 rev C)\na/wi.pdf › Load\n")
    assert [c.text_hash for c in v1] == [c.text_hash for c in v2]


def test_new_workspaces_use_context_two_and_old_ones_keep_one(corpus, tmp_path):
    new = Workspace.init(tmp_path / "new", TEST_EMBEDDING)
    assert new.config["chunking"]["context"] == 2
    old = Workspace.init(tmp_path / "old", {**TEST_EMBEDDING, "chunking": {"context": 1}})
    Builder(old).build([corpus / "manual.md"])
    store = RagStore(old.rag_db)
    store.conn.execute("DELETE FROM meta WHERE key = 'embed_context'")     # as a 0.4.0 knowledge base
    store.conn.commit()
    store.close()
    old.config.data["chunking"]["context"] = 2
    events = []
    Builder(old, progress=events.append).build([corpus / "manual.md", corpus / "reset.md"])
    store = RagStore(old.rag_db, readonly=True)
    try:
        assert store.get_meta("embed_context") == "1"
        rows = store.conn.execute("SELECT c.embed_text, d.display_path FROM chunks c JOIN documents d USING(doc_id)").fetchall()
        assert rows and all(t.startswith(path) for t, path in rows)      # no document line on top
    finally:
        store.close()
    assert any(e["event"] == "warning" and "context 1" in e.get("message", "") for e in events)


def test_spot_finds_procedures_and_leaves_manuals_alone(ws):
    st = spot.status(ws.root)
    assert st["present"] and st["current"]
    db = sqlite3.connect(spot.spot_path(ws.root))
    try:
        kinds = db.execute("SELECT kind, label, steps FROM procedures ORDER BY proc_id").fetchall()
        assert ("numbered", "7.0", 3) in kinds and ("numbered", "8.0", 2) in kinds
        assert any(k == "list" for k, _, _ in kinds)
        assert len(kinds) == 3                                  # nothing from the manual or MF164
        numbers = dict(db.execute("SELECT number, revision FROM documents WHERE number IS NOT NULL").fetchall())
        assert numbers.get("WI-2210") == "C" and "MF164" in numbers
        links = db.execute("SELECT kind, ref, doc_id IS NOT NULL FROM links ORDER BY rowid").fetchall()
        assert ("section", "8.2", 0) in links and ("document", "MF164", 1) in links
        roles = dict(db.execute("SELECT role, COUNT(*) FROM members GROUP BY role").fetchall())
        assert roles.get("caution") == 1
    finally:
        db.close()


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_a_step_brings_its_procedure(ws, engine):
    if engine == "native":
        pytest.importorskip("shammy")
    store, s = _searcher(ws, engine)
    try:
        r = s.search("remove the electrode with the Allen wrench", k=1).to_dict()["results"][0]
        assert r["procedure"]["label"] == "8.0" and r["procedure"]["whole"]
        assert "### 8.1 Electrode removal" in r["text"] and "### 8.2 Electrode installation" in r["text"]
        assert r["source"]["document"] == "WI-2210" and r["source"]["revision"] == "C"
        r = s.search("press the weld button until the light turns green", k=1).to_dict()["results"][0]
        assert r["procedure"]["label"] == "7.0"
        assert any(x["kind"] == "document" and x["ref"] == "MF164" and x["file"].endswith("/MF164.md") for x in r["related"])
        assert any(x["kind"] == "section" and x["ref"] == "8.2" for x in r["related"])
    finally:
        s.close()
        store.close()


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_a_long_procedure_gives_a_window_of_steps(ws, engine):
    if engine == "native":
        pytest.importorskip("shammy")
    store, s = _searcher(ws, engine, procedure_tokens=60, procedure_window=1)
    try:
        r = s.search("place the carrier tray on the oven shelf", k=1).to_dict()["results"][0]
        p = r["procedure"]
        assert not p["whole"] and p["from"] <= "7.2" <= p["to"] and (p["from"], p["to"]) != ("7.1", "7.3")
        md = s.search("place the carrier tray on the oven shelf", k=1).to_markdown()
        assert "7.0 Procedure (step" in md
    finally:
        s.close()
        store.close()


def test_engines_agree_with_structure(ws):
    pytest.importorskip("shammy")
    store, ref = _searcher(ws, "reference")
    _, nat = _searcher(ws, "native")
    try:
        for q in ("remove the electrode", "press the weld button", "reset the controller", "nitrogen psi",
                  "fixture for TO-39", "oven calibration sticker"):
            for kw in ({}, {"documents": ["WI-2210"]}, {"documents": ["manual"]}, {"documents": ["nothing-like-this"]},
                       {"expand": False}):
                a, b = ref.search(q, k=4, **kw), nat.search(q, k=4, **kw)
                ra, rb = a.to_dict()["results"], b.to_dict()["results"]
                assert [x["chunk_id"] for x in ra] == [x["chunk_id"] for x in rb], (q, kw)
                for x, y in zip(ra, rb):
                    x.pop("score"), y.pop("score")
                    assert x == y, (q, kw)
                assert a.to_markdown() == b.to_markdown(), (q, kw)
    finally:
        ref.close()
        nat.close()
        store.close()


def test_document_filter(ws):
    store, s = _searcher(ws)
    try:
        files = {r["source"]["file"] for r in s.search("oven", k=6, documents=["wi 2210"]).to_dict()["results"]}
        assert {f.split("/")[-1] for f in files} == {"WI-2210 sealer.md"}
        files = {r["source"]["file"] for r in s.search("oven", k=6, documents=["manual"]).to_dict()["results"]}
        assert {f.split("/")[-1] for f in files} == {"manual.md"}
        assert s.search("oven", documents=["nothing-like-this"]).results == []
    finally:
        s.close()
        store.close()


def test_a_stale_or_missing_spot_file_changes_nothing(ws, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(ws.root, copy)
    w = Workspace.open(copy)
    store, s = _searcher(w)
    try:
        with_spot = s.search("remove the electrode", k=3).to_dict()["results"]
        assert with_spot[0].get("procedure")
        db = sqlite3.connect(spot.spot_path(copy))
        db.execute("UPDATE spot_meta SET value = '0:0:0' WHERE key = 'chunks'")
        db.commit()
        db.close()
        assert not spot.status(copy)["current"]
        stale = s.search("remove the electrode", k=3).to_dict()["results"]
        spot.spot_path(copy).unlink()
        none = s.search("remove the electrode", k=3).to_dict()["results"]
        assert stale == none and not any("procedure" in r or "related" in r for r in none)
    finally:
        s.close()
        store.close()


def test_packs_carry_the_structure(ws, tmp_path):
    manifest = export_pack(ws, tmp_path / "p.ragpack")
    assert manifest["structure"]["procedures"] == 3 and spot.SPOT_NAME in manifest["files"]
    out = tmp_path / "unpacked"
    unpack_pack(tmp_path / "p.ragpack", out)
    assert spot.status(out)["current"]


def test_figure_text_keeps_what_the_description_confirms():
    desc = "A yellow label reading CAUTION Hot Surface on the oven door, and a dialog titled EStop."
    assert confirmed_text("CAUTION\n\nHot Sutace.\n\nEStop\n\nEmergencyStopButtonhasbeenpress", desc) == "CAUTION\n\nEStop"
    assert confirmed_text("Oxygen (ppm) 12", "Settings screen: Oxygen (ppm) 12") == "Oxygen (ppm) 12"
