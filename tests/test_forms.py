"""Forms: fillable PDF fields, label/value grids, Word content controls, checkboxes, tracked changes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures"))

from make_forms import make_forms  # noqa: E402

from rag.core.ingest.forms import checkbox_symbol, form_lines  # noqa: E402
from rag.core.ingest.router import extract  # noqa: E402


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    return make_forms(tmp_path_factory.mktemp("forms"))


def _text(path, ctx):
    doc, _ = extract(path, path.suffix[1:], ctx, cache_dir=None, use_cache=False)
    return doc.blocks, "\n".join(b.text for b in doc.blocks)


def test_fillable_pdf_values_are_read(files, ctx):
    blocks, text = _text(files["fillable"], ctx)
    assert "Lot number: L26-3007 Date: 2026-11-03" in text
    assert "Gasket inspected ☒ Rework required ☐" in text
    assert "Disposition: ○ Release ◉ MRB" in text
    assert "Oven: OV-2" in text and "Comments: Leak at corner 3, retest after bake" in text
    # values typed into fields inside a ruled table land in their cells
    table = next(b for b in blocks if b.kind == "table")
    assert table.meta["headers"] == ["Unit", "Slow leak (atm cc/s)", "Fast leak"]
    assert table.meta["rows"] == [["Unit 1", "2.1E-9", "☐"], ["Unit 2", "6.0E-8", "☒"]]


def test_fields_switch(files, ctx, cfg):
    cfg.data["pdf"]["form_fields"] = False
    _, text = _text(files["fillable"], ctx)
    assert "L26-3007" not in text and "Lot number:" in text


def test_printed_grid_form_reads_as_label_value_lines(files, ctx):
    blocks, _ = _text(files["grid"], ctx)
    form = next(b for b in blocks if b.meta.get("form"))
    assert form.kind == "paragraph"
    assert form.text.split("\n") == ["Lot number: L26-3011", "Date: 2026-11-04", "Part number: PX-7250",
                                     "Quantity: 48", "Operator: J. Ruiz", "Shift: 2"]
    assert not any(b.kind == "table" for b in blocks)


def test_word_form(files, ctx):
    blocks, text = _text(files["word"], ctx)
    lines = [b.text for b in blocks]
    assert "Operator: M. Okafor" in lines                                   # inline content control
    assert "☒ Oven door seal checked" in lines                             # content-control checkbox
    assert "☑ Calibration sticker current  ☐ Rework needed" in lines        # Wingdings symbol and font
    assert "☒ Air flow verified" in lines                             # legacy form checkbox
    assert "Bake time: 4 hours" in lines and "2 hours" not in text          # tracked insertion / deletion
    assert "Released by QA after the bake log was reviewed." in lines       # block-level content control
    form = next(b for b in blocks if b.meta.get("form"))
    assert form.text.split("\n") == ["Lot number: L26-3020", "Oven: OV-3", "Start: 08:15", "End: 12:15"]


def test_form_lines_rules():
    assert form_lines([["Lot", "Yield"], ["L1", "97.0"], ["L2", "91.5"]]) is None            # a data table
    assert form_lines([["Lot", "Notes"], ["L1", "Note: retest"], ["L2", ""], ["L3", ""], ["L4", ""]]) is None
    assert form_lines([["Start time:", "08:15", "Due:", "12:00"], ["Reviewer: A. Chen", "", "", ""]]) == [
        "Start time: 08:15", "Due: 12:00", "Reviewer: A. Chen"]
    assert checkbox_symbol("", "Wingdings") == "☑" and checkbox_symbol("þ", "Wingdings-Regular") == "☑"
    assert checkbox_symbol("R", "Wingdings2") == "☑" and checkbox_symbol("o", "Arial") is None
