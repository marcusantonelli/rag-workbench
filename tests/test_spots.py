"""Procedures and structure: the spots file, embedding context 2, and procedure-aware search."""

from __future__ import annotations

import shutil
import sqlite3

import pytest

from rag.builder.engine import Builder
from rag.core import spots
from rag.core.chunk.chunker import Chunker, ChunkSettings
from rag.core.db.pack import export_pack, unpack_pack
from rag.core.db.store import RagStore
from rag.core.ingest.figures import confirmed_text
from rag.core.models.embed import make_embedder
from rag.core.retrieve.search import Searcher
from rag.core.spots import SpotsIndex
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
def proc_corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("procs")
    (d / "WI-2210 sealer.md").write_text(SEALER, encoding="utf-8")
    (d / "manual.md").write_text(MANUAL, encoding="utf-8")
    (d / "reset.md").write_text(LIST, encoding="utf-8")
    (d / "MF164.md").write_text("# MF164 fixture list\n\nDocument # MF164\n\nFixture A fits TO-39, fixture B fits SMD-2.\n",
                                encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def ws(proc_corpus, tmp_path_factory):
    w = Workspace.init(tmp_path_factory.mktemp("spotws") / "ws", TEST_EMBEDDING)
    assert Builder(w).build([proc_corpus]).failed == 0
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


def test_new_workspaces_use_context_two_and_old_ones_keep_one(proc_corpus, tmp_path):
    corpus = proc_corpus
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
    st = spots.status(ws.root)
    assert st["present"] and st["current"]
    db = sqlite3.connect(spots.spots_path(ws.root))
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
        db = sqlite3.connect(spots.spots_path(copy))
        db.execute("UPDATE spots_meta SET value = '0:0:0' WHERE key = 'chunks'")
        db.commit()
        db.close()
        assert not spots.status(copy)["current"]
        stale = s.search("remove the electrode", k=3).to_dict()["results"]
        spots.spots_path(copy).unlink()
        none = s.search("remove the electrode", k=3).to_dict()["results"]
        assert stale == none and not any("procedure" in r or "related" in r for r in none)
    finally:
        s.close()
        store.close()


def test_packs_carry_the_structure(ws, tmp_path):
    manifest = export_pack(ws, tmp_path / "p.ragpack")
    assert manifest["structure"]["procedures"] == 3 and spots.SPOTS_NAME in manifest["files"]
    out = tmp_path / "unpacked"
    unpack_pack(tmp_path / "p.ragpack", out)
    assert spots.status(out)["current"]


def test_figure_text_keeps_what_the_description_confirms():
    desc = "A yellow label reading CAUTION Hot Surface on the oven door, and a dialog titled EStop."
    assert confirmed_text("CAUTION\n\nHot Sutace.\n\nEStop\n\nEmergencyStopButtonhasbeenpress", desc) == "CAUTION\n\nEStop"
    assert confirmed_text("Oxygen (ppm) 12", "Settings screen: Oxygen (ppm) 12") == "Oxygen (ppm) 12"


def test_file_names_give_document_numbers_only_when_they_start_with_one():
    number = lambda name: identity(None, [""], name).number  # noqa: E731
    assert number("docs/WI-1234 Seal.pdf") == "WI-1234" and number("QP-114 Pull Test.pdf") == "QP-114"
    # a crawler's hash suffix, or an asset code inside a long name, is not a document number
    assert number("x/whats-new-in-opcenter-execution-semiconductor-8-6-de595425.html") is None
    assert number("Siemens-PLM-Whats-new-in-Camstar-Semiconductor-Suite-V7-SU6-fs-73745-A4_tcm84-46-eba9f644.pdf") is None
    assert number("calibration.csv") is None


def test_other_sources_search_exactly_as_before(built_ws, tmp_path):
    """Spreadsheets, CSV, code, SQL, HTML, text, Markdown and scans: no procedures, and the same
    passages with or without the spots file (only document numbers are added to citations)."""
    assert spots.status(built_ws.root)["procedures"] == 0
    copy = tmp_path / "copy"
    shutil.copytree(built_ws.root, copy)
    w = Workspace.open(copy)
    store, s = _searcher(w)
    queries = ["helium leak rate limit", "burn-in yield by lot", "calibration due date", "GetLotYield",
               "die attach void limits", "gowning order", "SELECT lot_id FROM yields"]
    try:
        with_spot = [s.search(q, k=6).to_dict()["results"] for q in queries]
        spots.spots_path(copy).unlink()
        without = [s.search(q, k=6).to_dict()["results"] for q in queries]
    finally:
        s.close()
        store.close()
    for a, b in zip(with_spot, without):
        assert [(x["chunk_id"], x.get("chunk_ids"), x["text"]) for x in a] == \
               [(x["chunk_id"], x.get("chunk_ids"), x["text"]) for x in b]
        assert not any("procedure" in x or "related" in x for x in a)


# -- long procedures ------------------------------------------------------------------------------
def _long_procedure() -> str:
    verbs = ["Verify", "Press", "Install", "Remove", "Inspect", "Record", "Align", "Tighten", "Clean", "Load"]
    things = ["the seal ring", "the carrier tray", "the oven door", "the weld electrode", "the vacuum gauge",
              "the nitrogen valve", "the fixture clamp", "the lid", "the header", "the traveler"]

    def lines(seed: int, n: int) -> str:
        return "\n".join(f"- {verbs[(seed + i) % 10]} {things[(seed * 3 + i) % 10]} and confirm reading "
                         f"{seed * 31 + i} before the next action of step {seed}." for i in range(n))

    out = ["# Furnace line operation", "", "Title: Furnace line operation | Specification # WI-5000 Revision: D", "",
           "## 7.0 Procedure", ""]
    for i in range(1, 41):
        out.append(f"### 7.{i} Operation step {i}: {verbs[i % 10]} {things[i % 10]}\n")
        if i == 20:        # a step far longer than procedure_tokens, with sub-steps
            for j in range(1, 16):
                out.append(f"#### 7.20.{j} Bake profile segment {j}\n\n{lines(200 + j, 8)}\n")
        elif i == 30:      # a step longer than procedure_tokens, without sub-steps
            out.append(lines(300, 90) + "\n")
        else:
            out.append(lines(i, (3, 6, 10)[i % 3]) + "\n")
    return "\n".join(out)


@pytest.fixture(scope="module")
def long_ws(tmp_path_factory):
    d = tmp_path_factory.mktemp("longproc")
    (d / "WI-5000 Furnace line.md").write_text(_long_procedure(), encoding="utf-8")
    w = Workspace.init(tmp_path_factory.mktemp("longws") / "ws", TEST_EMBEDDING)
    assert Builder(w).build([d]).failed == 0
    return w


def _first_text(w, like: str) -> str:
    store = RagStore(w.rag_db, readonly=True)
    try:
        return store.conn.execute("SELECT text FROM chunks WHERE heading_path LIKE ? ORDER BY ordinal LIMIT 1",
                                  (f"%{like}%",)).fetchone()[0][:160]
    finally:
        store.close()


def test_long_procedures_record_sub_steps(long_ws):
    db = sqlite3.connect(spots.spots_path(long_ws.root))
    try:
        assert db.execute("SELECT steps, tokens > 1500 FROM procedures").fetchone() == (40, 1)
        subs = db.execute("SELECT COUNT(*) FROM substeps s JOIN steps t USING(proc_id, step) WHERE t.label = '7.20'").fetchone()[0]
        assert subs == 15
    finally:
        db.close()


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_a_step_too_long_for_the_limit_keeps_its_place(long_ws, engine):
    if engine == "native":
        pytest.importorskip("shammy")
    store, s = _searcher(long_ws, engine)
    try:
        r = s.search(_first_text(long_ws, "7.20.7 "), k=1).to_dict()["results"][0]
        p = r["procedure"]
        assert p["level"] == "substeps" and p["step"] == "7.20" and p["from"] <= "7.20.7" <= p["to"]
        assert "#### 7.20.7 Bake profile segment 7" in r["text"]
        r = s.search(_first_text(long_ws, "7.30 "), k=1).to_dict()["results"][0]
        assert r["procedure"]["level"] == "chunks" and r["procedure"]["step"] == "7.30"
        assert r["text"].startswith("### 7.30 Operation step 30")
        md = s.search(_first_text(long_ws, "7.30 "), k=1).to_markdown()
        assert "7.0 Procedure (part of step 7.30 of 40)" in md
    finally:
        s.close()
        store.close()


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_one_procedure_leaves_room_and_carries_one_outline(long_ws, engine):
    if engine == "native":
        pytest.importorskip("shammy")
    store, s = _searcher(long_ws, engine, procedure_hits=2)
    try:
        res = s.search("confirm reading before the next action", k=6)
        out = res.to_dict()["results"]
        # only this procedure is in the knowledge base: the capped hits fill the places left
        assert len(out) == 6 and all(r["procedure"]["id"] == out[0]["procedure"]["id"] for r in out)
        assert ["outline" in r for r in out] == [True] + [False] * 5
        outline = out[0]["outline"]
        assert len(outline["steps"]) <= 17 and any(x["shown"] for x in outline["steps"])
        assert res.to_markdown().count("Outline of 7.0 Procedure") == 1
        ids = [c for r in out for c in r.get("chunk_ids", [r["chunk_id"]])]
        assert len(ids) == len(set(ids))                              # nothing returned twice
    finally:
        s.close()
        store.close()


def test_neighbouring_hits_widen_one_window(long_ws):
    store, s = _searcher(long_ws)
    try:
        plan = s._plan(SpotsIndex.open(long_ws.root, store.conn), _chunk_in(store, "7.12 "))
        assert plan["level"] == "steps"
        out = s.search(_first_text(long_ws, "7.12 ") + " " + _first_text(long_ws, "7.14 "), k=6).to_dict()["results"]
        windows = [(r["procedure"]["from"], r["procedure"]["to"]) for r in out if r["procedure"]["level"] == "steps"]
        # no two windows touch: neighbours were merged into one
        spans = sorted((int(a.split(".")[1]), int(b.split(".")[1])) for a, b in windows)
        assert all(nxt[0] > cur[1] + 1 for cur, nxt in zip(spans, spans[1:]))
    finally:
        s.close()
        store.close()


def _chunk_in(store, like: str) -> int:
    return store.conn.execute("SELECT chunk_id FROM chunks WHERE heading_path LIKE ? ORDER BY ordinal LIMIT 1",
                              (f"%{like}%",)).fetchone()[0]


def test_engines_agree_on_long_procedures(long_ws):
    pytest.importorskip("shammy")
    queries = [_first_text(long_ws, x) for x in ("7.3 ", "7.12 ", "7.20.2 ", "7.20.14 ", "7.30 ", "7.40 ")]
    queries += ["confirm reading before the next action", "bake profile segment", "tighten the seal ring"]
    for cfg in ({}, {"procedure_tokens": 300}, {"procedure_tokens": 300, "procedure_window": 3},
                {"procedure_hits": 1}, {"procedure_tokens": 100000}):
        store, ref = _searcher(long_ws, "reference", **cfg)
        _, nat = _searcher(long_ws, "native", **cfg)
        try:
            for q in queries:
                for k in (3, 8):
                    a, b = ref.search(q, k=k), nat.search(q, k=k)
                    ra, rb = a.to_dict()["results"], b.to_dict()["results"]
                    for x, y in zip(ra, rb):
                        x.pop("score"), y.pop("score")
                    assert ra == rb, (q, cfg, k)
                    assert a.to_markdown() == b.to_markdown(), (q, cfg, k)
        finally:
            ref.close()
            nat.close()
            store.close()
