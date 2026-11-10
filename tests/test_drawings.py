"""Flowcharts drawn as vector shapes, and pictures of tables, in PDFs and DOCX files."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_drawings import SHEET, long_sheet, make_drawings  # noqa: E402

from rag.core.ingest.router import extract  # noqa: E402


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return make_drawings(tmp_path_factory.mktemp("drawings"))


def _blocks(path, ctx):
    doc, _ = extract(path, path.suffix[1:], ctx, cache_dir=None, use_cache=False)
    return doc.blocks


def _steps(text: str) -> dict[str, str]:
    """{label: what follows the arrow or dash} with step numbers replaced by labels."""
    lines = [ln for ln in text.split("\n") if ln[:1].isdigit()]
    names = {ln.split(". ", 1)[0]: ln.split(". ", 1)[1].split(" → ")[0].split(" — ")[0] for ln in lines}
    out = {}
    for ln in lines:
        body = ln.split(". ", 1)[1]
        label = names[ln.split(". ", 1)[0]]
        rest = body[len(label):]
        for num in sorted(names, key=len, reverse=True):
            rest = rest.replace(f" {num}", f" <{names[num]}>")
        out[label] = rest.strip()
    return out


def test_flowchart_becomes_steps_and_connections(files, ctx):
    blocks = _blocks(files["flowchart"], ctx)
    figs = [b for b in blocks if b.kind == "figure"]
    assert len(figs) == 2 and all(b.meta["diagram"] for b in figs)
    first = figs[0]
    assert first.text.startswith("**Flowchart** (page 1): Figure 4. Lot disposition after a fine leak failure")
    assert first.meta["diagram"] == {"steps": 8, "connections": 8}
    assert (ctx.assets_root.parent / first.meta["asset_path"]).is_file()
    s = _steps(first.text)
    assert s["Start: lot fails fine leak"] == "→ <Place lot on hold (mes-cli holds --reason SealLeak)>"
    assert s["Leak rate < 5E-8 atm·cc/s?"] == "→ No: <Submit to MRB for review>; Yes: <Retest with helium bomb>"
    assert s["Scrap lot"] == "→ <End>" and s["Release lot to stock"] == "→ <End>" and s["End"] == ""
    # the labels left the text flow; the prose around the drawing is intact and in order
    prose = [b.text for b in blocks if b.kind == "paragraph" and b.page == 1]
    assert prose == ["Every lot that fails fine leak after burn-in follows the process below before it can be released "
                     "or scrapped. The MRB decides within two shifts.",
                     "Lots released after a retest keep their original date code."]
    assert not any("Submit to MRB" in b.text or "Figure 4" in b.text for b in blocks if b.kind != "figure")
    kinds = [b.kind for b in blocks if b.page == 1]
    assert kinds == ["heading", "paragraph", "figure", "paragraph"]


def test_visio_style_flowchart(files, ctx):
    """Connectors in pieces, open arrowheads, lanes, a branch, a two-way arrow, a plain link, a page frame."""
    blocks = _blocks(files["flowchart"], ctx)
    fig = [b for b in blocks if b.kind == "figure"][1]
    assert fig.text.startswith("**Flowchart** (page 2): Figure 5. Gross leak check")
    s = _steps(fig.text)
    assert s["Pull 5-unit sample [Operator]"] == "→ <Run gross leak test [Operator]>"
    assert s["Bubbles seen? [Operator]"] == "→ Not OK: <Log failure in MES [QA Engineer]>; OK: <Continue to marking [Operator]>"
    log = s["Log failure in MES [QA Engineer]"]
    assert all(t in log for t in ("<QA on-call phone [QA Engineer]>", "<Notify QA engineer [QA Engineer]>",
                                  "<Place lot on hold [QA Engineer]>"))
    assert s["QA on-call phone [QA Engineer]"] == "→ <Log failure in MES [QA Engineer]>"       # two-way
    assert s["Notify QA engineer [QA Engineer]"] == "— <See WI-0042 [Operator]>"               # no arrowhead
    assert "Notes:" not in fig.text
    prose = [b.text for b in blocks if b.kind == "paragraph" and b.page == 2]
    assert prose[0].startswith("Operators run the gross leak check") and prose[-1].startswith("Record every check")
    assert not any("Operator" == p or "Not OK" in p for p in prose)


def test_lookalikes_are_not_flowcharts(files, ctx, cfg):
    blocks = _blocks(files["lookalikes"], ctx)
    assert not any(b.kind == "figure" for b in blocks)
    table = next(b for b in blocks if b.kind == "table")
    assert table.meta["headers"] == ["Oven", "Max temp", "Slots"] and table.meta["rows"][-1] == ["OV-3", "175 C", "32"]
    assert any("Safety first" in b.text for b in blocks)
    cfg.data["pdf"]["diagrams"] = False                 # the switch: flowcharts read as plain text again
    assert not any(b.kind == "figure" for b in _blocks(files["flowchart"], ctx))


def test_pictures_of_spreadsheets_become_tables(files, ctx):
    blocks = _blocks(files["pictures"], ctx)
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 2
    small, big = tables
    assert [small.meta["headers"]] + small.meta["rows"] == SHEET          # every cell, the lone "0" included
    assert small.meta["caption"] == "Table 2. Burn-in yields by lot (read from the picture with OCR)"
    assert small.meta["from_image"] and (ctx.assets_root.parent / small.meta["asset_path"]).is_file()
    # a full-page Excel screenshot: its column letters and row numbers are dropped, single digits kept
    assert [big.meta["headers"]] + big.meta["rows"] == long_sheet()
    assert big.page == 2 and (ctx.assets_root.parent / big.meta["asset_path"]).is_file()
    # the picture sits where it was on the page, and a screenshot of prose stays a figure
    kinds = [b.kind for b in blocks if b.page == 1]
    assert kinds == ["heading", "paragraph", "table", "figure", "paragraph"]
    fig = next(b for b in blocks if b.kind == "figure")
    assert "Seal the lid within four hours" in fig.text and "|" not in fig.text
    assert not any("Table 2." in b.text for b in blocks if b.kind == "paragraph")


def test_picture_table_in_docx_and_switch(files, ctx, cfg):
    blocks = _blocks(files["docx"], ctx)
    table = next(b for b in blocks if b.kind == "table")
    assert [table.meta["headers"]] + table.meta["rows"] == SHEET
    cfg.data["figures"]["table_ocr"] = False
    blocks = _blocks(files["docx"], ctx)
    assert not any(b.kind == "table" for b in blocks) and any(b.kind == "figure" and "L24-0819" in b.text for b in blocks)


def test_built_and_searchable(files, tmp_path):
    from conftest import TEST_EMBEDDING
    from rag.builder.engine import Builder
    from rag.core.check import check_workspace
    from rag.core.db.sqlguard import run_query
    from rag.core.db.store import RagStore
    from rag.core.db.tables import TablesStore
    from rag.core.workspace import Workspace

    ws = Workspace.init(tmp_path / "ws", TEST_EMBEDDING)
    stats = Builder(ws).build([files["flowchart"], files["pictures"]])
    assert stats.failed == 0, stats.errors
    assert check_workspace(ws.root)["ok"]
    store = RagStore(ws.rag_db, readonly=True)
    try:
        flow = store.conn.execute("SELECT text, asset_path FROM chunks WHERE kind = 'figure' AND text LIKE '%Flowchart%'").fetchall()
        assert len(flow) == 2 and all(r[1] for r in flow)
        pic = store.conn.execute("SELECT asset_path, ocr_conf FROM chunks WHERE kind = 'table' AND text LIKE '%L24-0819%'").fetchone()
        assert pic[0] and pic[1] > 0.5                      # the table chunk carries its picture
    finally:
        store.close()
    t = TablesStore(ws.tables_db, readonly=True)
    try:
        name = next(r["table_name"] for r in t.catalog() if r["n_rows"] == 40)   # the 40-row sheet is queryable
    finally:
        t.close()
    res = run_query(ws.tables_db, f'SELECT SUM(CAST("Fails" AS INTEGER)) FROM "{name}"', row_limit=5, time_limit_ms=2000)
    assert res["rows"][0][0] == sum(int(r[3]) for r in long_sheet()[1:])
