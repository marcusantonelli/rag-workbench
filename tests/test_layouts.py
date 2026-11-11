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
