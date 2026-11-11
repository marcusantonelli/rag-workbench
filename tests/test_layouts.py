"""General page layouts read from PDFs: block-arrow flowcharts, boxed panels and titled tables."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_drawings import panels_pdf  # noqa: E402

from rag.core.ingest.router import extract  # noqa: E402


@pytest.fixture(scope="module")
def panels(tmp_path_factory):
    return panels_pdf(tmp_path_factory.mktemp("panels") / "Line setup.pdf")


@pytest.fixture()
def blocks(panels, ctx):
    doc, _ = extract(panels, "pdf", ctx, cache_dir=None, use_cache=False)
    return doc.blocks


def test_block_arrows_join_the_flowchart(blocks):
    figs = [b for b in blocks if b.kind == "figure" and b.meta.get("diagram")]
    assert len(figs) == 1
    assert figs[0].meta["diagram"] == {"steps": 4, "connections": 3}
    text = figs[0].text
    assert "Load the recipe → " in text and "Inspect the first article → " in text


def test_a_one_column_box_is_text_not_a_table(blocks):
    tables = [b for b in blocks if b.kind == "table"]
    assert not any("Fixture clean" in b.text for b in tables)
    body = "\n".join(b.text for b in blocks if b.kind in ("paragraph", "heading"))
    assert "Fixture clean and dry" in body and "Spare seals at the station" in body


def test_a_table_with_a_title_row(blocks):
    tables = [b for b in blocks if b.kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t.meta["headers"] == ["Symptom", "Likely cause", "What to do"]
    assert t.meta["caption"] == "Troubleshooting"
    assert [r[0] for r in t.meta["rows"]] == ["Seal leaks", "Torque alarm", "Label misprint"]   # no blank "4." row
    i = blocks.index(t)
    assert blocks[i - 1].kind == "heading" and blocks[i - 1].text == "Troubleshooting"


# -- Word ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def word(tmp_path_factory):
    from make_drawings import word_shapes_docx

    return word_shapes_docx(tmp_path_factory.mktemp("word") / "Line setup.docx")


@pytest.fixture()
def wblocks(word, ctx):
    doc, _ = extract(word, "docx", ctx, cache_dir=None, use_cache=False)
    return doc.blocks


def test_word_shapes_make_a_flowchart(wblocks):
    figs = [b for b in wblocks if b.kind == "figure" and b.meta.get("diagram")]
    assert len(figs) == 2
    first = figs[0].text
    assert first.startswith("**Flowchart**")
    assert "1. Load the recipe → 2" in first and "2. Run a first article → 3" in first
    assert "3. Article good? → Yes: 4" in first and "4. Release the line" in first
    assert figs[0].meta["asset_path"] is None
    second = figs[1].text          # separate floating shapes, each anchored to its own paragraph
    assert "1. Stop the pump → 2" in second and "2. Close the valve → 3" in second and "3. Log the stop" in second
    assert "Night shift" not in second and "Night shift: skip the log" in [b.text for b in wblocks]
    # the shape text isn't repeated as paragraphs, and the VML copy is never read
    body = [b.text for b in wblocks if b.kind == "paragraph"]
    assert not any("Load the recipe" in t or "Stop the pump" in t or "FALLBACK" in t for t in body)
    i = next(n for n, b in enumerate(wblocks) if b.text.startswith("Shut the line down"))
    assert wblocks[i + 1] is figs[1]


def test_word_text_boxes_are_read(wblocks):
    body = [b.text for b in wblocks if b.kind == "paragraph"]
    assert "Keep the lid closed\nwhile the pump runs." in body
    assert "Call the shift lead before restarting." in body
    i = body.index("The pump starts on its own after a recipe change.")
    assert body[i + 1] == "Keep the lid closed\nwhile the pump runs."


def test_word_tables_follow_the_same_rules(wblocks):
    tables = [b for b in wblocks if b.kind == "table"]
    assert len(tables) == 1
    t = tables[0]
    assert t.meta["caption"] == "Troubleshooting" and t.meta["headers"] == ["Symptom", "Likely cause", "What to do"]
    assert [r[0] for r in t.meta["rows"]] == ["Seal leaks", "Torque alarm", "Label misprint"]
    i = wblocks.index(t)
    assert wblocks[i - 1].kind == "heading" and wblocks[i - 1].text == "Troubleshooting" and wblocks[i - 1].level == 2
    panel = [b for b in wblocks if b.kind == "paragraph" and b.text.startswith("Set-up Checklist")]
    assert panel and panel[0].text.split("\n") == ["Set-up Checklist", "Fixture clean and dry", "Torque driver calibrated",
                                                  "Two operators signed in"]
