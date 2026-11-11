from __future__ import annotations

from rag.core.ingest.router import detect_type, extract


def _run(path, ctx):
    doc, _ = extract(path, detect_type(path), ctx, cache_dir=None)
    return doc


def test_text_pdf_headings_table_figure_and_footer(corpus, ctx):
    doc = _run(corpus / "QP-114 Pull Test.pdf", ctx)
    headings = [(b.level, b.text) for b in doc.blocks if b.kind == "heading"]
    assert (1, "1 Purpose") in headings and (1, "5 Records") in headings
    tables = [b for b in doc.blocks if b.kind == "table"]
    assert len(tables) == 1
    assert tables[0].meta["headers"] == ["Hose diameter (mm)", "Material", "Minimum pull (gf)"]
    assert tables[0].meta["caption"].startswith("Table 1")
    figures = [b for b in doc.blocks if b.kind == "figure"]
    assert len(figures) == 1 and "F-22" in figures[0].text and figures[0].page == 2
    assert figures[0].meta["asset_path"].startswith("assets/")
    all_text = " ".join(b.text for b in doc.blocks)
    assert "Controlled Document" not in all_text, "repeated footer should be stripped"
    assert doc.title == "Hose Clamp Pull Test Procedure"


def test_scanned_pdf_is_ocrd(corpus, ctx):
    doc = _run(corpus / "DR-2291 scan.pdf", ctx)
    text = " ".join(b.text for b in doc.blocks)
    assert "DR-2291" in text and "oven" in text.lower()
    assert any("OCR" in w for w in doc.warnings)
    assert all(b.meta.get("ocr_conf") for b in doc.blocks)


def test_docx_structure(corpus, ctx):
    doc = _run(corpus / "Motor Mount WI.docx", ctx)
    levels = {b.text: b.level for b in doc.blocks if b.kind == "heading"}
    assert levels["Motor Mount Work Instruction"] == 1 and levels["Scope"] == 2 and levels["Gap Limits"] == 3
    table = next(b for b in doc.blocks if b.kind == "table")
    assert table.meta["rows"][0] == ["H-254", "10", "X-ray"]
    assert any(b.kind == "figure" and "F-22" in b.text for b in doc.blocks)
    assert any(b.text.startswith("- Set the stage") for b in doc.blocks)


def test_xlsx_regions_merged_headers_and_notes(corpus, ctx):
    doc = _run(corpus / "Run-in Yields Q3.xlsx", ctx)
    yields = next(t for t in doc.tables if t.sheet == "Yields")
    assert yields.headers == ["Lot", "Device", "Test / Pass", "Test / Fail", "Date"]
    assert yields.row_numbers[0] == 5 and len(yields.rows) == 4
    assert any("Q3 run-in yields by lot" in b.text for b in doc.blocks)
    assert any(t.sheet == "Limits" for t in doc.tables)


def test_csv(corpus, ctx):
    doc = _run(corpus / "calibration.csv", ctx)
    assert doc.tables[0].headers[0] == "Tool" and len(doc.tables[0].rows) == 3


def test_sql_keeps_procedure_whole_and_collapses_inserts(corpus, ctx):
    doc = _run(corpus / "code" / "lots.sql", ctx)
    proc = next(b for b in doc.blocks if b.symbol == "dbo.GetLotYield")
    assert "END" in proc.text and "GROUP BY l.LotId" in proc.text
    assert "dbo.TestResults" in proc.meta["tables"]
    bulk = [b for b in doc.blocks if b.meta.get("bulk")]
    assert bulk and bulk[0].meta["bulk"] == 30
    assert len(doc.blocks) == 4


def test_csharp_symbols(corpus, ctx):
    doc = _run(corpus / "code" / "LotReleaseService.cs", ctx)
    symbols = {b.symbol for b in doc.blocks}
    assert "Acme.Track.Release.LotReleaseService.CanRelease" in symbols
    assert "Acme.Track.Release.HoldReason" in symbols
    can = next(b for b in doc.blocks if b.symbol.endswith("CanRelease"))
    assert "<summary>Returns true" in can.text
    header = next(b for b in doc.blocks if b.symbol == "Acme.Track.Release.LotReleaseService")
    assert "MinimumYield" in header.text and "members:" in header.text
    # Interface members without bodies live only in the interface header.
    assert "Acme.Track.Release.IYieldRepository.GetLotYield" not in symbols


def test_python_symbols(corpus, ctx):
    doc = _run(corpus / "code" / "runin.py", ctx)
    symbols = [b.symbol for b in doc.blocks]
    assert symbols[0] == "runin" and "runin.RunInRecord.yield_pct" in symbols
    assert "runin.load_records" in symbols
    assert "DEFAULT_HOURS" in doc.blocks[0].text


def test_html_drops_boilerplate(corpus, ctx):
    doc = _run(corpus / "safety.html", ctx)
    text = " ".join(b.text for b in doc.blocks)
    assert "tracking" not in text and "Home" not in text and "Copyright" not in text
    assert doc.title == "Workshop Safety"
    lst = next(b for b in doc.blocks if b.text.startswith("- Hood"))
    assert lst.text.count("- ") == 4 and "Coverall" in lst.text
    assert any(b.kind == "table" and b.meta["caption"] == "Garment change frequency" for b in doc.blocks)


def test_markdown(corpus, ctx):
    doc = _run(corpus / "Pressure Leak.md", ctx)
    assert doc.title == "Pressure Leak Troubleshooting"
    assert [b.kind for b in doc.blocks].count("table") == 1
    code = next(b for b in doc.blocks if b.kind == "code")
    assert code.lang == "bash" and code.line_start == 19


def test_txt_cp1252(corpus, ctx):
    doc = _run(corpus / "handoff.txt", ctx)
    assert len(doc.blocks) == 3 and doc.blocks[1].line_start == 3


def test_detect_type_rejects_mismatched_signature(tmp_path):
    fake = tmp_path / "fake.docx"
    fake.write_bytes(b"not a zip")
    assert detect_type(fake) is None
    legacy = tmp_path / "old.docx"
    legacy.write_bytes(b"\xd0\xcf\x11\xe0" + b"0" * 20)
    assert detect_type(legacy) == "doc"
