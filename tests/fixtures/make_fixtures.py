"""Generate a small corpus with every supported file type.

    python tests/fixtures/make_fixtures.py <out_dir>

The content is fictional but shaped like real plant documents: a procedure
PDF with headings, a ruled table, a figure and repeating footers; a scanned
PDF; a Word work instruction; spreadsheets with merged headers and notes; SQL
with a stored procedure and bulk inserts; C# and Python source.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for name in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def text_image(lines: list[str], size=(1000, 300), font_size=40, border=True) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    if border:
        d.rectangle([5, 5, size[0] - 6, size[1] - 6], outline="black", width=4)
    y = 30
    for line in lines:
        d.text((40, y), line, fill="black", font=_font(font_size))
        y += int(font_size * 1.5)
    return img


def make_text_pdf(path: Path, figure_png: Path) -> None:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(72, 30, "Acme Components - Controlled Document QP-114")
        canvas.drawRightString(540, 30, f"Page {doc.page}")
        canvas.restoreState()

    story = [
        Paragraph("Wire Bond Pull Test Procedure", styles["Title"]),
        Paragraph("1 Purpose", styles["Heading1"]),
        Paragraph(
            "This procedure defines how operators perform destructive wire bond pull testing on "
            "hermetic power devices. Pull testing verifies bond strength before lot release and "
            "must be performed on every assembly lot.", styles["BodyText"]),
        Paragraph("2 Equipment", styles["Heading1"]),
        Paragraph(
            "Use the Dage 4000 bond tester with a calibrated 100 gram cartridge. Calibration must be "
            "current within 12 months. Record the tester serial number on the traveler.", styles["BodyText"]),
        Paragraph("3 Minimum Pull Strength", styles["Heading1"]),
        Paragraph("Table 1 Minimum pull strength by wire diameter", styles["BodyText"]),
    ]
    data = [["Wire diameter (mil)", "Material", "Minimum pull (gf)"],
            ["1.0", "Gold", "3.0"], ["1.25", "Gold", "4.0"], ["5.0", "Aluminum", "35.0"],
            ["10.0", "Aluminum", "75.0"], ["15.0", "Aluminum", "150.0"]]
    t = Table(data)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    story += [t, Spacer(1, 12)]
    story += [
        Paragraph(
            "A pull value below the minimum is a reject. Record every failure mode code on the data "
            "sheet: lift at first bond, lift at second bond, wire break at heel, or wire break mid span.",
            styles["BodyText"]),
        PageBreak(),
        Paragraph("4 Fixture Setup", styles["Heading1"]),
        Paragraph("Mount the device in fixture F-22 as shown below before testing.", styles["BodyText"]),
        RLImage(str(figure_png), width=360, height=108),
        Paragraph("Figure 1 Fixture F-22 clamp positions", styles["BodyText"]),
        Paragraph("5 Records", styles["Heading1"]),
        Paragraph(
            "Store the completed data sheet with the lot traveler. Records are retained for ten years "
            "per the quality manual.", styles["BodyText"]),
        PageBreak(),
        Paragraph("6 Revision History", styles["Heading1"]),
        Paragraph("Revision C added the aluminum 15 mil limit and the failure mode codes.", styles["BodyText"]),
    ]
    SimpleDocTemplate(str(path), pagesize=letter).build(story, onFirstPage=footer, onLaterPages=footer)


def make_scanned_pdf(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    page = Image.new("RGB", (1700, 2200), "white")
    d = ImageDraw.Draw(page)
    d.text((150, 150), "Deviation Report DR-2291", fill="black", font=_font(64))
    body = [
        "Lot 24A-0193 failed seal leak test after burn-in.",
        "Root cause: lid seal furnace profile drifted 8 degrees low.",
        "Corrective action: recalibrate furnace zone 3 weekly.",
        "Approved by quality engineering on March 4.",
    ]
    y = 320
    for line in body:
        d.text((150, y), line, fill="black", font=_font(40))
        y += 70
    img_path = path.with_suffix(".png")
    page.save(img_path)
    c = canvas.Canvas(str(path), pagesize=letter)
    w, h = letter
    c.drawImage(str(img_path), 0, 0, width=w, height=h)
    c.showPage()
    c.save()
    img_path.unlink()


def make_docx(path: Path, figure_png: Path) -> None:
    import docx
    from docx.shared import Inches

    d = docx.Document()
    d.core_properties.title = "Die Attach Work Instruction"
    d.add_heading("Die Attach Work Instruction", 0)
    d.add_heading("Scope", 1)
    d.add_paragraph("Applies to eutectic die attach on the Palomar 3800 for all TO-254 packages.")
    d.add_heading("Steps", 1)
    for step in ["Verify the preform lot against the traveler.",
                 "Set the stage temperature to 340 C and wait for stability.",
                 "Scrub the die for 3 seconds with 50 gram force."]:
        d.add_paragraph(step, style="List Number")
    d.add_heading("Void Limits", 2)
    table = d.add_table(rows=1, cols=3)
    table.style = "Table Grid"
    for cell, text in zip(table.rows[0].cells, ["Package", "Max void (%)", "Inspection"]):
        cell.text = text
    for row in [["TO-254", "10", "X-ray"], ["TO-257", "15", "X-ray"], ["SMD-0.5", "20", "SAM"]]:
        cells = table.add_row().cells
        for cell, text in zip(cells, row):
            cell.text = text
    d.add_paragraph("Reference image of the stage setup:")
    d.add_picture(str(figure_png), width=Inches(4))
    d.save(path)


def make_xlsx(path: Path) -> None:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Yields"
    ws["A1"] = "Q3 burn-in yields by lot"            # note line
    ws.append([])
    ws.append(["Lot", "Device", "Test", None, "Date"])
    ws.append([None, None, "Pass", "Fail", None])
    ws.merge_cells("C3:D3")
    rows = [["24A-0190", "IRHM7360", 498, 2, date(2025, 7, 3)],
            ["24A-0191", "IRHM7360", 495, 5, date(2025, 7, 9)],
            ["24A-0193", "IRHN57250", 470, 30, date(2025, 8, 2)],
            ["24A-0197", "IRHN57250", 499, 1, date(2025, 8, 21)]]
    for r in rows:
        ws.append(r)
    ws.append([])
    ws.append(["Owner", "Reliability engineering"])
    ws2 = wb.create_sheet("Limits")
    ws2.append(["Test", "Limit", "Unit"])
    ws2.append(["Seal leak", 5e-8, "atm cc/s"])
    ws2.append(["Burn-in hours", 160, "h"])
    wb.save(path)


def make_csv(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Tool", "Last calibration", "Interval (months)", "Owner"])
        w.writerow(["Dage 4000", "2025-03-14", "12", "Metrology"])
        w.writerow(["Palomar 3800", "2025-01-20", "6", "Assembly"])
        w.writerow(["Seal furnace 2", "2024-11-02", "3", "Process"])


SQL = """-- Lot genealogy schema
CREATE TABLE dbo.Lots (
    LotId     VARCHAR(20) PRIMARY KEY,
    DeviceId  VARCHAR(20) NOT NULL,
    StartDate DATE
);
GO

/* Returns the yield of one lot; used by the release report. */
CREATE PROCEDURE dbo.GetLotYield
    @LotId VARCHAR(20)
AS
BEGIN
    SET NOCOUNT ON;
    SELECT l.LotId, SUM(t.Pass) AS Pass, SUM(t.Fail) AS Fail,
           CASE WHEN SUM(t.Pass) + SUM(t.Fail) = 0 THEN NULL
                ELSE 1.0 * SUM(t.Pass) / (SUM(t.Pass) + SUM(t.Fail)) END AS Yield
    FROM dbo.Lots l
    JOIN dbo.TestResults t ON t.LotId = l.LotId
    WHERE l.LotId = @LotId
    GROUP BY l.LotId;
END
GO
""" + "".join(
    f"INSERT INTO dbo.Lots (LotId, DeviceId, StartDate) VALUES ('24A-{i:04d}', 'IRHM7360', '2025-07-01');\n"
    for i in range(30)
) + """
SELECT DeviceId, COUNT(*) FROM dbo.Lots GROUP BY DeviceId;
"""

CS = '''using System;
using System.Collections.Generic;

namespace Acme.Mes.Release
{
    /// <summary>Decides whether a lot can be released.</summary>
    public class LotReleaseService
    {
        private readonly IYieldRepository _yields;
        public double MinimumYield { get; set; } = 0.95;

        public LotReleaseService(IYieldRepository yields)
        {
            _yields = yields;
        }

        /// <summary>Returns true when the lot meets the minimum burn-in yield.</summary>
        /// <param name="lotId">Lot identifier, e.g. 24A-0193.</param>
        public bool CanRelease(string lotId)
        {
            var yield = _yields.GetLotYield(lotId);
            if (yield is null)
            {
                throw new InvalidOperationException($"No yield data for {lotId}");
            }
            return yield.Value >= MinimumYield;
        }
    }

    public enum HoldReason
    {
        None,
        LowYield,
        SealLeak,
    }

    public interface IYieldRepository
    {
        double? GetLotYield(string lotId);
    }
}
'''

PY = '''"""Burn-in data loader for the reliability dashboard."""

import csv
from dataclasses import dataclass

DEFAULT_HOURS = 160


@dataclass
class BurnInRecord:
    """One lot's burn-in result."""

    lot_id: str
    passed: int
    failed: int

    def yield_pct(self) -> float:
        """Pass yield as a percentage."""
        total = self.passed + self.failed
        return 100.0 * self.passed / total if total else 0.0


def load_records(path):
    """Read BurnInRecord rows from a CSV export."""
    with open(path, newline="") as fh:
        return [BurnInRecord(r["lot"], int(r["pass"]), int(r["fail"])) for r in csv.DictReader(fh)]
'''

MD = """---
title: Seal Leak Troubleshooting
---

# Seal Leak Troubleshooting

Fine leak failures after burn-in usually trace back to the lid seal furnace.

## Checks

1. Confirm the furnace zone 3 thermocouple reading.
2. Inspect lid plating for contamination.

| Symptom | Likely cause |
| --- | --- |
| Gross leak | Cracked lid |
| Fine leak | Low seal temperature |

```bash
mes-cli holds --reason SealLeak --since 2025-07-01
```
"""

HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Cleanroom Gowning</title>
<script>var tracking = 1;</script><style>body{}</style></head>
<body><nav><a href="/">Home</a> | <a href="/qa">Quality</a></nav>
<main><h1>Cleanroom Gowning</h1>
<p>All personnel must gown in the order below before entering the Class 1000 assembly area.</p>
<h2>Gowning order</h2>
<ul><li>Hood</li><li>Coverall<ul><li>Zip fully</li></ul></li><li>Boots</li></ul>
<table><caption>Garment change frequency</caption>
<tr><th>Garment</th><th>Change</th></tr>
<tr><td>Coverall</td><td>Weekly</td></tr><tr><td>Gloves</td><td>Every entry</td></tr></table>
</main><footer>Copyright Acme</footer></body></html>
"""

TXT = """Shift handoff notes - Assembly line 2

The wire bonder on station 4 threw a bond placement alarm twice during second shift.
Maintenance replaced the capillary and the alarm cleared.

Seal furnace 2 is scheduled for calibration on Thursday; route lots to furnace 1 until then.
"""


def make_all(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    fig = out / "_figure.png"
    text_image(["FIXTURE F-22", "CLAMP A  CLAMP B"], font_size=56).save(fig)
    make_text_pdf(out / "QP-114 Pull Test.pdf", fig)
    make_scanned_pdf(out / "DR-2291 scan.pdf")
    make_docx(out / "Die Attach WI.docx", fig)
    make_xlsx(out / "Burn-in Yields Q3.xlsx")
    make_csv(out / "calibration.csv")
    (out / "code").mkdir(exist_ok=True)
    (out / "code" / "lots.sql").write_text(SQL, encoding="utf-8")
    (out / "code" / "LotReleaseService.cs").write_text(CS, encoding="utf-8")
    (out / "code" / "burnin.py").write_text(PY, encoding="utf-8")
    (out / "code" / "bin").mkdir(exist_ok=True)
    (out / "code" / "bin" / "ignored.cs").write_text("class Ignored {}", encoding="utf-8")
    (out / "Seal Leak.md").write_text(MD, encoding="utf-8")
    (out / "gowning.html").write_text(HTML, encoding="utf-8")
    (out / "handoff.txt").write_text(TXT, encoding="cp1252")
    (out / "~$Die Attach WI.docx").write_bytes(b"lock")
    fig.unlink()
    return out


if __name__ == "__main__":
    print(make_all(Path(sys.argv[1] if len(sys.argv) > 1 else "corpus")))
