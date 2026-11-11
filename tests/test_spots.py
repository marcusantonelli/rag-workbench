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

PUMP = """# Pump assembly

Title: Pump assembly | Specification # WI-0300 Revision: C

## 1.0 Purpose

1.1 This document gives the steps for assembling the pump.

## 7.0 Procedure

### 7.1 Check the torque driver

- Verify that the calibration sticker on the torque driver has not expired.
- Record the calibration date in the log book.

### 7.2 Mount the housing

- Place the pump housing on the fixture plate.
- Close the clamp and press START.
- CAUTION: the clamp closes with force.

### 7.3 Fit the impeller

- Position the impeller on the shaft and align the keyway.
- Press the seat button and hold it until the light turns green.
- If the impeller does not seat, see Section 8.2 and see FX-100.

## 8.0 Filter Replacement

### 8.1 Filter removal

- Use the Allen wrench to unscrew the cover screw and remove the filter.
- Inspect the O-ring and replace it when damaged.

### 8.2 Filter installation

- Install the new filter in the housing and tighten the screw.
- Rotate the filter to check it sits flat.
"""

MANUAL = """# Pump user manual

## 1 Overview

The pump moves water at up to 40 litres per minute. It has two inlets and a pressure gauge.

## 2 Specifications

### 2.1 Power

The pump runs on 230 V and draws up to 3 A at full speed.

### 2.2 Pressure

The outlet is rated at 6 bar; the gauge reads in bar.
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
    (d / "WI-0300 pump.md").write_text(PUMP, encoding="utf-8")
    (d / "manual.md").write_text(MANUAL, encoding="utf-8")
    (d / "reset.md").write_text(LIST, encoding="utf-8")
    (d / "FX-100.md").write_text("# FX-100 fixture list\n\nDocument # FX-100\n\nFixture A fits H-39, fixture B fits HS-2.\n",
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
    assert heading_number("8.1 Nozzle Removal:") == (8, 1)
    assert heading_number("Step 3 - Remove the cover") == (3,)
    assert heading_number("1.5 mm gap") is None
    assert heading_number("Start-up & Login") is None
    i = identity(None, ["Title: | Specification # DOC-12 Revision: A"], "docs/x.pdf")
    assert (i.number, i.revision) == ("DOC-12", "A")
    # a redacted number doesn't swallow the next field
    assert identity("T", ["Specification # Revision: E1"]).number is None
    assert identity(None, ["no title block"], "docs/WI-1234 Seal.pdf").number == "WI-1234"


def test_numbered_headings_nest_by_their_numbers():
    doc = ExtractedDoc(file_type="pdf", title=None, blocks=[
        Block("heading", "7.0 Procedure", level=1), Block("heading", "7.1 Verify", level=2),
        Block("paragraph", "Verify the sticker on the torque driver has not expired today."),
        Block("heading", "8.0 Nozzle Replacement", level=2),     # the file got the level wrong
        Block("heading", "8.1 Removal", level=3), Block("paragraph", "Remove the screw and lift the nozzle out."),
    ])
    chunks = Chunker(ChunkSettings(), lambda t: len(t.split())).chunk(doc, "wi.pdf")
    assert [c.heading_path for c in chunks] == ["7.0 Procedure › 7.1 Verify", "8.0 Nozzle Replacement › 8.1 Removal"]


def test_context_two_names_the_document_and_one_is_unchanged():
    doc = ExtractedDoc(file_type="pdf", title="Pump assembly", blocks=[
        Block("paragraph", "Specification # WI-0300 Revision: C"), Block("heading", "Load", level=1),
        Block("paragraph", "Place the pump housing on the fixture plate and close the clamp.")])
    count = lambda t: len(t.split())  # noqa: E731
    v1 = Chunker(ChunkSettings(context=1), count).chunk(doc, "a/wi.pdf")
    v2 = Chunker(ChunkSettings(context=2), count).chunk(doc, "a/wi.pdf")
    assert v1[-1].embed_text.startswith("a/wi.pdf › Load\n")
    assert v2[-1].embed_text.startswith("Pump assembly (WI-0300 rev C)\na/wi.pdf › Load\n")
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
        kinds = db.execute("SELECT kind, label, steps FROM procedures WHERE kind != 'section' ORDER BY proc_id").fetchall()
        assert ("numbered", "7.0", 3) in kinds and ("numbered", "8.0", 2) in kinds
        assert any(k == "list" for k, _, _ in kinds)
        assert len(kinds) == 3                                  # no procedure from the manual or FX-100
        # the manual is a section: its chapters are parts, labelled by their titles
        manual = db.execute("SELECT proc_id, title, steps FROM procedures WHERE kind = 'section' AND title = 'Pump user manual'").fetchone()
        assert manual is not None
        parts = [r[0] for r in db.execute("SELECT label FROM steps WHERE proc_id = ? ORDER BY step", (manual[0],))]
        assert parts[-2:] == ["1 Overview", "2 Specifications"]
        subs = [r[0] for r in db.execute("SELECT label FROM substeps WHERE proc_id = ? ORDER BY step, sub", (manual[0],))]
        assert subs[-2:] == ["2.1 Power", "2.2 Pressure"]
        # FX-100 is one chunk: nothing to add, so no section
        numbers = dict(db.execute("SELECT number, revision FROM documents WHERE number IS NOT NULL").fetchall())
        assert numbers.get("WI-0300") == "C" and "FX-100" in numbers
        links = db.execute("SELECT kind, ref, doc_id IS NOT NULL FROM links ORDER BY rowid").fetchall()
        assert ("section", "8.2", 0) in links and ("document", "FX-100", 1) in links
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
        r = s.search("remove the filter with the Allen wrench", k=1).to_dict()["results"][0]
        assert r["procedure"]["label"] == "8.0" and r["procedure"]["whole"]
        assert "### 8.1 Filter removal" in r["text"] and "### 8.2 Filter installation" in r["text"]
        assert r["source"]["document"] == "WI-0300" and r["source"]["revision"] == "C"
        r = s.search("press the seat button until the light turns green", k=1).to_dict()["results"][0]
        assert r["procedure"]["label"] == "7.0"
        assert any(x["kind"] == "document" and x["ref"] == "FX-100" and x["file"].endswith("/FX-100.md") for x in r["related"])
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
        r = s.search("place the pump housing on the fixture plate", k=1).to_dict()["results"][0]
        p = r["procedure"]
        assert not p["whole"] and p["from"] <= "7.2" <= p["to"] and (p["from"], p["to"]) != ("7.1", "7.3")
        md = s.search("place the pump housing on the fixture plate", k=1).to_markdown()
        assert "7.0 Procedure (step" in md
    finally:
        s.close()
        store.close()


def test_engines_agree_with_structure(ws):
    pytest.importorskip("shammy")
    store, ref = _searcher(ws, "reference")
    _, nat = _searcher(ws, "native")
    try:
        for q in ("remove the filter", "press the seat button", "reset the controller", "pressure bar",
                  "fixture for H-39", "torque driver calibration sticker"):
            for kw in ({}, {"documents": ["WI-0300"]}, {"documents": ["manual"]}, {"documents": ["nothing-like-this"]},
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
        files = {r["source"]["file"] for r in s.search("pump", k=6, documents=["wi 0300"]).to_dict()["results"]}
        assert {f.split("/")[-1] for f in files} == {"WI-0300 pump.md"}
        files = {r["source"]["file"] for r in s.search("pump", k=6, documents=["manual"]).to_dict()["results"]}
        assert {f.split("/")[-1] for f in files} == {"manual.md"}
        assert s.search("pump", documents=["nothing-like-this"]).results == []
    finally:
        s.close()
        store.close()


def test_a_stale_or_missing_spot_file_changes_nothing(ws, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(ws.root, copy)
    w = Workspace.open(copy)
    store, s = _searcher(w)
    try:
        with_spot = s.search("remove the filter", k=3).to_dict()["results"]
        assert with_spot[0].get("procedure")
        db = sqlite3.connect(spots.spots_path(copy))
        db.execute("UPDATE spots_meta SET value = '0:0:0' WHERE key = 'chunks'")
        db.commit()
        db.close()
        assert not spots.status(copy)["current"]
        stale = s.search("remove the filter", k=3).to_dict()["results"]
        spots.spots_path(copy).unlink()
        none = s.search("remove the filter", k=3).to_dict()["results"]
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
    desc = "A yellow label reading CAUTION Hot Surface on the pump motor, and a dialog titled Alarm."
    assert confirmed_text("CAUTION\n\nHot Surfcae.\n\nAlarm\n\nMotorOverloadAlarmhasbeenraised", desc) == "CAUTION\n\nAlarm"
    assert confirmed_text("Flow (L/min) 12", "Settings screen: Flow (L/min) 12") == "Flow (L/min) 12"


def test_file_names_give_document_numbers_only_when_they_start_with_one():
    number = lambda name: identity(None, [""], name).number  # noqa: E731
    assert number("docs/WI-1234 Seal.pdf") == "WI-1234" and number("QP-114 Pull Test.pdf") == "QP-114"
    # a crawler's hash suffix, or an asset code inside a long name, is not a document number
    assert number("x/whats-new-in-nimbus-warehouse-8-6-de595425.html") is None
    assert number("Example Corp-PLM-Whats-new-in-Stratus-Pump-Suite-V7-SU6-fs-73745-A4_tcm84-46-eba9f644.pdf") is None
    assert number("calibration.csv") is None


def test_other_sources_search_exactly_as_before(built_ws, tmp_path):
    """Spreadsheets, CSV, code and SQL: no procedures or sections, and the same passages with or
    without the spots file. Documents with headings come back with their section, which always
    holds the passage that matched; with sections off they come back exactly as before (only
    document numbers are added to citations)."""
    assert spots.status(built_ws.root)["procedures"] == 0
    copy = tmp_path / "copy"
    shutil.copytree(built_ws.root, copy)
    w = Workspace.open(copy)
    store, s = _searcher(w)
    _, off = _searcher(w, sections=False)
    queries = ["air leak rate limit", "run-in yield by lot", "calibration due date", "GetLotYield",
               "motor mount gap limits", "safety order", "SELECT lot_id FROM yields"]
    try:
        with_spot = [s.search(q, k=6).to_dict()["results"] for q in queries]
        sections_off = [off.search(q, k=6).to_dict()["results"] for q in queries]
        texts = dict(store.conn.execute("SELECT chunk_id, text FROM chunks").fetchall())
        spots.spots_path(copy).unlink()
        without = [s.search(q, k=6).to_dict()["results"] for q in queries]
    finally:
        s.close()
        off.close()
        store.close()
    key = lambda x: (x["chunk_id"], x.get("chunk_ids"), x["text"])  # noqa: E731
    for a, b, c in zip(with_spot, sections_off, without):
        assert [key(x) for x in b] == [key(x) for x in c]
        assert not any("procedure" in x or "related" in x for x in b)
        for x in a:
            assert texts[x["chunk_id"]] in x["text"]
            if x["source"]["file_type"] not in spots.SECTION_TYPES:
                assert "procedure" not in x
            elif "procedure" in x:
                assert x["procedure"]["kind"] == "section"
    # a spreadsheet, CSV, code or SQL passage reads the same either way (sections taking more or
    # less of the result budget can change which of them make the cut, not what they hold)
    before = {x["chunk_id"]: key(x) for r in without for x in r}
    flat = [x for r in with_spot for x in r if x["source"]["file_type"] not in spots.SECTION_TYPES]
    assert flat and all(key(x) == before[x["chunk_id"]] for x in flat if x["chunk_id"] in before)
    assert any("procedure" in x for r in with_spot for x in r)


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_a_manual_answers_with_its_section(ws, engine):
    if engine == "native":
        pytest.importorskip("shammy")
    store, s = _searcher(ws, engine)
    try:
        res = s.search("outlet rated bar gauge", k=3, documents=["manual"])
        top = res.to_dict()["results"][0]
        assert top["procedure"]["kind"] == "section" and top["procedure"]["whole"]
        assert "6 bar" in top["text"] and "230 V" in top["text"] and "40 litres" in top["text"]
        assert "### 2 Specifications" in top["text"] and "#### 2.2 Pressure" in top["text"]
        assert "Pump user manual (whole section)" in res.to_markdown()
    finally:
        s.close()
        store.close()


def _handbook() -> str:
    """A long manual without numbered headings: chapters with sections and a few sub-sections."""
    topics = ["Unpacking", "Placement", "Electrical supply", "Water connections", "First start", "Daily care",
              "Seasonal storage", "Noise and vibration", "Error codes", "Spare parts", "Disposal", "Warranty"]
    out = ["# Garden pump handbook", "", "This handbook covers the garden pump range.", ""]
    for i, t in enumerate(topics):
        out.append(f"## {t}\n")
        body = " ".join(f"The {t.lower()} guidance item {j} explains detail {i * 40 + j} for owners of the pump."
                        for j in range(30))
        out.append(body + "\n")
        if i == 8:
            for k in range(3):
                out.append(f"### Code E{k + 1}\n\nCode E{k + 1} means fault {k + 1} and is cleared by a restart.\n")
    return "\n".join(out)


@pytest.fixture(scope="module")
def book_ws(tmp_path_factory):
    d = tmp_path_factory.mktemp("book")
    (d / "Garden pump handbook.md").write_text(_handbook(), encoding="utf-8")
    w = Workspace.init(tmp_path_factory.mktemp("bookws") / "ws", TEST_EMBEDDING)
    assert Builder(w).build([d]).failed == 0
    return w


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_a_long_section_gives_parts_an_outline_and_a_way_on(book_ws, engine):
    if engine == "native":
        pytest.importorskip("shammy")
    store, s = _searcher(book_ws, engine)
    try:
        res = s.search("error codes guidance item 5", k=2)
        top = res.to_dict()["results"][0]
        p = top["procedure"]
        assert p["kind"] == "section" and not p["whole"] and p["title"] == "Garden pump handbook"
        assert top.get("outline") and any(x["shown"] for x in top["outline"]["steps"])
        md = res.to_markdown()
        assert "Outline of Garden pump handbook" in md and "Read on: get_procedure_steps(procedure=" in md
        # reading on covers the whole section once
        seen, page = [], s.procedure(p["id"])
        while True:
            seen += page["chunk_ids"]
            if not page.get("next"):
                break
            page = s.procedure(page["next"]["procedure"], from_=page["next"]["from"],
                               after_chunk=page["next"].get("after_chunk"))
        all_ids = [r[0] for r in store.conn.execute("SELECT chunk_id FROM chunks ORDER BY ordinal")]
        assert sorted(seen) == sorted(set(seen)) and set(seen) == set(all_ids)
        assert page["markdown"].endswith("(the section ends here)")
    finally:
        s.close()
        store.close()


def test_engines_agree_on_sections(book_ws, built_ws):
    pytest.importorskip("shammy")
    for w, qs in ((book_ws, ["error codes guidance item 5", "warranty detail 450", "first start", "code E2 restart"]),
                  (built_ws, ["air leak rate limit", "motor mount gap limits", "safety order", "calibration due date"])):
        store, ref = _searcher(w, "reference")
        _, nat = _searcher(w, "native")
        try:
            for q in qs:
                for kw in ({}, {"section_tokens": 300}, {"sections": False}):
                    ref.cfg.update(kw)
                    nat.cfg.update(kw)
                    a, b = ref.search(q, k=4), nat.search(q, k=4)
                    ra, rb = a.to_dict()["results"], b.to_dict()["results"]
                    for x, y in zip(ra, rb):
                        x.pop("score"), y.pop("score")
                    assert ra == rb, (q, kw)
                    assert a.to_markdown() == b.to_markdown(), (q, kw)
                    ref.cfg.update({"section_tokens": 1200, "sections": True})
                    nat.cfg.update({"section_tokens": 1200, "sections": True})
        finally:
            ref.close()
            nat.close()
            store.close()


def test_an_older_spots_file_is_used_and_then_upgraded(ws, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(ws.root, copy)
    db = sqlite3.connect(spots.spots_path(copy))
    db.execute("DELETE FROM members WHERE proc_id IN (SELECT proc_id FROM procedures WHERE kind = 'section')")
    db.execute("DELETE FROM procedures WHERE kind = 'section'")
    db.execute("DELETE FROM spots_meta WHERE key = 'sections'")          # as 0.4.1 wrote it
    db.commit()
    db.close()
    st = spots.status(copy)
    assert st["current"] and st["outdated"]
    w = Workspace.open(copy)
    store, s = _searcher(w)
    try:
        assert s.search("remove the filter", k=3).to_dict()["results"][0]["procedure"]["kind"] == "numbered"
    finally:
        s.close()
        store.close()
    Builder(w).build([])          # any build brings it up to date
    st = spots.status(copy)
    assert st["current"] and not st["outdated"] and st["sections"] >= 2


# -- long procedures ------------------------------------------------------------------------------
def _long_procedure() -> str:
    verbs = ["Verify", "Press", "Install", "Remove", "Inspect", "Record", "Align", "Tighten", "Clean", "Load"]
    things = ["the gasket", "the carrier tray", "the pump cover", "the spray nozzle", "the vacuum gauge",
              "the air valve", "the fixture clamp", "the lid", "the header", "the job sheet"]

    def lines(seed: int, n: int) -> str:
        return "\n".join(f"- {verbs[(seed + i) % 10]} {things[(seed * 3 + i) % 10]} and confirm reading "
                         f"{seed * 31 + i} before the next action of step {seed}." for i in range(n))

    out = ["# Pump line operation", "", "Title: Pump line operation | Specification # WI-5000 Revision: D", "",
           "## 7.0 Procedure", ""]
    for i in range(1, 41):
        out.append(f"### 7.{i} Operation step {i}: {verbs[i % 10]} {things[i % 10]}\n")
        if i == 20:        # a step far longer than procedure_tokens, with sub-steps
            for j in range(1, 16):
                out.append(f"#### 7.20.{j} Test cycle segment {j}\n\n{lines(200 + j, 8)}\n")
        elif i == 30:      # a step longer than procedure_tokens, without sub-steps
            out.append(lines(300, 90) + "\n")
        else:
            out.append(lines(i, (3, 6, 10)[i % 3]) + "\n")
    return "\n".join(out)


@pytest.fixture(scope="module")
def long_ws(tmp_path_factory):
    d = tmp_path_factory.mktemp("longproc")
    (d / "WI-5000 Pump line.md").write_text(_long_procedure(), encoding="utf-8")
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
        assert "#### 7.20.7 Test cycle segment 7" in r["text"]
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
    queries += ["confirm reading before the next action", "test cycle segment", "tighten the gasket"]
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


# -- reading on through a procedure -------------------------------------------------------------
@pytest.mark.parametrize("max_tokens", [4000, 600, 150])
def test_pages_cover_a_procedure_once_and_engines_agree(long_ws, max_tokens):
    store, ref = _searcher(long_ws, "reference")
    engines = [ref]
    if native_engine_available():
        engines.append(_searcher(long_ws, "native")[1])
    try:
        pages = []
        args = {"from_": None, "after_chunk": None}
        while True:
            got = [e.procedure(1, max_tokens=max_tokens, **args) for e in engines]
            assert all(g == got[0] for g in got)
            pages.append(got[0])
            if not got[0]["next"]:
                break
            args = {"from_": got[0]["next"]["from"], "after_chunk": got[0]["next"].get("after_chunk")}
        ids = [c for p in pages for c in p["chunk_ids"]]
        db = sqlite3.connect(spots.spots_path(long_ws.root))
        try:
            members = [r[0] for r in db.execute("SELECT chunk_id FROM members WHERE proc_id = 1 ORDER BY step, ordinal")]
        finally:
            db.close()
        assert ids == members                       # every chunk once, in reading order
        assert pages[-1]["markdown"].endswith("(the procedure ends here)")
        assert all(p["markdown"].startswith("[P1] ") for p in pages)
    finally:
        for e in engines:
            e.close()
        store.close()


def native_engine_available() -> bool:
    try:
        import shammy  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.parametrize("engine", ["reference", "native"])
def test_a_partial_result_says_how_to_read_on(long_ws, engine):
    if engine == "native":
        pytest.importorskip("shammy")
    store, s = _searcher(long_ws, engine)
    try:
        res = s.search(_first_text(long_ws, "7.12 "), k=3)
        r = res.to_dict()["results"][0]
        more = r["more"]
        assert more["procedure"] == r["procedure"]["id"] and more["from"]
        assert f'Read on: get_procedure_steps(procedure={more["procedure"]}, from="{more["from"]}"' in res.to_markdown()
        # following it continues right after what the result showed
        page = s.procedure(more["procedure"], from_=more["from"], after_chunk=more.get("after_chunk"))
        assert not set(page["chunk_ids"]) & set(r.get("chunk_ids", [r["chunk_id"]]))
        # a whole procedure has nothing more
        store2, s2 = _searcher(long_ws, engine, procedure_tokens=100000)
        try:
            assert all("more" not in x for x in s2.search(_first_text(long_ws, "7.12 "), k=3).to_dict()["results"])
        finally:
            s2.close()
            store2.close()
    finally:
        s.close()
        store.close()


def test_reading_on_needs_the_spots_file(long_ws, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(long_ws.root, copy)
    spots.spots_path(copy).unlink()
    store, s = _searcher(Workspace.open(copy))
    try:
        with pytest.raises(ValueError, match="spots"):
            s.procedure(1)
    finally:
        s.close()
        store.close()
