"""Generate the Halden Microsystems test corpus, its golden questions and answer key.

    python tools/make_test_corpus.py OUT_DIR

Halden Microsystems is fictional: a back-end (assembly and test) plant for
hermetic power devices, running an MES. Every document type the Builder reads
is represented, with the messiness real document sets have: a superseded
revision next to the current one, a byte-identical copy, a two-column
newsletter, a large spec table, a table without ruling lines, scanned and
half-scanned PDFs, merged spreadsheet headers, a hidden sheet, a cp1252 log,
semicolon CSV, generated code that should be excluded, and a saved web page
with its asset folder.

Outputs:
  OUT_DIR/docs/                   build this:  rag build WS OUT_DIR/docs
  OUT_DIR/questions/golden.jsonl  rag eval WS OUT_DIR/questions/golden.jsonl
  OUT_DIR/questions/ANSWERS.md    expected answers, sources, SQL for table questions
"""

from __future__ import annotations

import csv
import io
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

R = random.Random(20251107)
COMPANY = "Halden Microsystems"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def font(size: int, bold: bool = False):
    names = (["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "arialbd.ttf"]
             if bold else ["DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "arial.ttf"])
    for n in names:
        try:
            return ImageFont.truetype(n, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def label_figure(labels: list[str], size=(1100, 420)) -> Image.Image:
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    w = (size[0] - 80) // len(labels)
    for i, lab in enumerate(labels):
        x0 = 40 + i * w
        d.rectangle([x0 + 10, 90, x0 + w - 10, size[1] - 90], outline="black", width=5)
        d.text((x0 + 30, size[1] // 2 - 30), lab, fill="black", font=font(40, True))
        if i:
            d.line([(x0 - 10, size[1] // 2), (x0 + 10, size[1] // 2)], fill="black", width=5)
    return img


def scanned_page(lines: list[tuple[str, int, bool]], rotate: float = 0.7, seed: int = 1) -> Image.Image:
    """Render text like a scanned sheet: slight skew, speckle, soft focus."""
    rnd = random.Random(seed)
    img = Image.new("L", (1700, 2200), 250)
    d = ImageDraw.Draw(img)
    y = 170
    for text, size, bold in lines:
        if text:
            d.text((150, y), text, fill=25, font=font(size, bold))
        y += int(size * 1.75) if text else 40
    for _ in range(2500):
        x, yy = rnd.randrange(1700), rnd.randrange(2200)
        d.point((x, yy), fill=rnd.randrange(80, 200))
    img = img.rotate(rotate, resample=Image.BICUBIC, fillcolor=250, expand=False)
    return img.filter(ImageFilter.GaussianBlur(0.6)).convert("RGB")


def styles():
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet

    s = getSampleStyleSheet()
    s.add(ParagraphStyle("Small", parent=s["BodyText"], fontSize=8, leading=10))
    s.add(ParagraphStyle("Caption", parent=s["BodyText"], fontSize=9, leading=11, fontName="Helvetica-Oblique"))
    return s


def footer_fn(text: str):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.drawString(54, 28, text)
        canvas.drawRightString(558, 28, f"Page {doc.page}")
        canvas.restoreState()
    return draw


def grid_table(data, col_widths=None, font_size=9):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import Paragraph, Table, TableStyle

    cell = ParagraphStyle("cell", fontName="Helvetica", fontSize=font_size, leading=font_size + 2)
    data = [data[0]] + [[Paragraph(str(v), cell) if len(str(v)) > 30 else v for v in row] for row in data[1:]]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dddddd")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


# ---------------------------------------------------------------------------
# PDFs
# ---------------------------------------------------------------------------
BOMB_TABLE = [
    ["Cavity volume (cc)", "Bomb pressure (psig)", "Bomb time (h)", "Max dwell (h)", "Reject limit (atm-cc/s He)"],
    ["<= 0.01", "60", "2", "1", "5.0E-8"],
    ["> 0.01 to 0.05", "60", "4", "1", "5.0E-8"],
    ["> 0.05 to 0.40", "30", "4", "1", "1.0E-7"],
    ["> 0.40", "30", "8", "1", "5.0E-7"],
]


def make_sop(path: Path, rev: str) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    c = rev == "C"
    s = styles()
    packages = "TO-254, TO-257, SMD-0.5 and SMD-1" if c else "TO-254, TO-257 and SMD-0.5"
    observe = "30 seconds" if c else "60 seconds"
    hold = "HL-SEAL" if c else "HL-LEAK"
    fig = path.parent / "_fig_leak.png"
    label_figure(["BOMB CHAMBER BC-2", "LEAK DETECTOR HLD-7"]).save(fig)
    P = lambda t, st="BodyText": Paragraph(t, s[st])  # noqa: E731
    story = [
        P(f"SOP-0412 Hermetic Seal Leak Test", "Title"),
        P(f"Revision {rev} &nbsp;&nbsp; Effective {'2025-09-02' if c else '2024-05-21'} &nbsp;&nbsp; Owner: Reliability Engineering"),
        P("1 Purpose", "Heading1"),
        P("This procedure defines fine and gross leak testing of hermetically sealed packages after lid seal "
          "and after burn-in. It ensures that every shipped device meets the seal requirements of the "
          "customer specification and of MIL-STD-883 Test Method 1014."),
        P("2 Scope", "Heading1"),
        P(f"Applies to all hermetic packages assembled at the Fairbury site: {packages}. Plastic packages are "
          "out of scope."),
        P("3 References", "Heading1"),
        P("MIL-STD-883 Test Method 1014 (Seal); WI-0217 Die Attach; SOP-0388 Lid Seal; MES Hold Codes (KB)."),
        PageBreak(),
        P("4 Equipment", "Heading1"),
        P("Helium leak detector HLD-7 (calibrated annually), bomb chamber BC-2, gross leak bath GL-1 filled "
          "with FC-40 fluorocarbon, black-background viewing station with 10X magnifier."),
        P("5 Fine Leak Test", "Heading1"),
        P("Bomb the parts in helium according to Table 2, then measure within the maximum dwell time. Parts "
          "measuring above the reject limit fail fine leak."),
        P("Table 2 Fine leak bombing conditions by cavity volume", "Caption"),
        grid_table(BOMB_TABLE, [1.4 * inch, 1.3 * inch, 1.0 * inch, 1.0 * inch, 1.8 * inch], 8),
        Spacer(1, 10),
        P("If the measurement cannot start within the maximum dwell time, re-bomb the parts. Do not re-bomb "
          "more than twice."),
        PageBreak(),
        P("Mount parts in the bomb chamber and connect the detector as shown in Figure 1."),
        RLImage(str(fig), width=5.5 * inch, height=2.1 * inch),
        P("Figure 1 Helium fine leak test setup", "Caption"),
        P("6 Gross Leak Test", "Heading1"),
        P(f"Immerse the parts in FC-40 at 125 &deg;C &plusmn; 5 &deg;C and observe them for {observe} against "
          "the black background. A continuous stream of bubbles from one location, or two or more large "
          "bubbles, is a gross leak reject."),
        PageBreak(),
        P("7 Disposition", "Heading1"),
        P(f"Tag rejects with a red HOLD label and place the lot on hold in the MES with hold code {hold}. One "
          "retest is allowed within 4 hours if the first failure is suspected to be surface helium; a second "
          "failure is final."),
        P("8 Records", "Heading1"),
        P("Leak test data are recorded in the MES against the lot. Paper data sheets, if used, are attached "
          "to the traveler. Records are retained for ten years."),
    ]
    if c:
        story += [PageBreak(), P("9 Revision History", "Heading1"), grid_table([
            ["Rev", "Date", "Change"],
            ["A", "2023-02-10", "Initial release"],
            ["B", "2024-05-21", "Added SMD-0.5; added retest rule"],
            ["C", "2025-09-02", "Added SMD-1; gross leak observation reduced from 60 s to 30 s; hold code HL-LEAK renamed HL-SEAL"],
        ], [0.5 * inch, 1.0 * inch, 5.0 * inch], 8)]
    doc = SimpleDocTemplate(str(path), pagesize=letter, title=f"SOP-0412 Hermetic Seal Leak Test Rev {rev}")
    foot = footer_fn(f"{COMPANY} - Controlled Copy - SOP-0412 Rev {rev}")
    doc.build(story, onFirstPage=foot, onLaterPages=foot)
    fig.unlink()


def make_newsletter(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer

    s = styles()
    W, H = letter
    gap, margin = 18, 54
    colw = (W - 2 * margin - gap) / 2
    frames = [Frame(margin, margin, colw, H - 2 * margin - 40, id="left"),
              Frame(margin + colw + gap, margin, colw, H - 2 * margin - 40, id="right")]

    def header(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawString(margin, H - 60, "Reliability Newsletter - Q3 2025")
        canvas.setFont("Helvetica", 8)
        canvas.drawString(margin, 28, f"{COMPANY} internal newsletter")
        canvas.restoreState()

    doc = BaseDocTemplate(str(path), pagesize=letter, title="Reliability Newsletter Q3 2025")
    doc.addPageTemplates([PageTemplate(id="two", frames=frames, onPage=header)])
    P = lambda t, st="BodyText": Paragraph(t, s[st])  # noqa: E731
    filler = ("Teams across assembly and test contributed to the result, and the reliability group thanks "
              "everyone who supported the qualification runs and the extra weekend shifts. ")
    story = [
        P("Burn-in capacity expands", "Heading2"),
        P("Two new burn-in ovens, BI-05 and BI-06, were commissioned in August. Together they raise burn-in "
          "capacity by 40 percent and remove the queue that held lots for up to three days in July. " + filler),
        P("Both ovens run the standard 160-hour profile at 125 &deg;C and are connected to the MES so that "
          "start and end times are captured automatically. " + filler * 2),
        P("Customer return CR-2207 closed", "Heading2"),
        P("The root cause of customer return CR-2207 was die attach voiding above 25 percent on lot "
          "HM25-1187, traced to preform lot PF-3381 from the supplier. Failure analysis report FA-25-031 has "
          "the details. Incoming inspection now X-rays one preform coupon per lot. " + filler),
        P("Safety", "Heading2"),
        P("The site reached 212 days without a lost-time incident on September 30. Remember to wear "
          "cryogenic gloves when refilling the leak detector cold trap. " + filler),
        P("Metrology welcomes a new lead", "Heading2"),
        P("Dana Whitfield joined as metrology lead in September. Dana owns the calibration register and the "
          "measurement system analysis program. " + filler * 2),
        P("Wire bond pull data", "Heading2"),
        P("Average pull strength on 15 mil aluminum wire held at 182 grams-force in Q3, well above the "
          "150 gram-force minimum. " + filler * 3),
    ]
    doc.build(story)


SPEC_ROWS: list[list] = []


def make_spec(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    s = styles()
    pkgs = ["TO-254", "TO-257", "SMD-0.5", "SMD-1"]
    for i in range(36):
        bv = [60, 100, 150, 200, 250][i % 5]
        rds = round(8 + bv * 0.42 + R.uniform(-3, 3), 1)
        idm = round(max(6, 70 - bv * 0.22 + R.uniform(-4, 4)), 1)
        pkg = pkgs[(i * 7) % 4]
        rad = [100, 300, 1000][i % 3]
        SPEC_ROWS.append([f"HM-R{bv:03d}{chr(65 + i % 26)}{i:02d}", bv, rds, idm, pkg, rad])
    data = [["Part number", "BVDSS (V)", "RDS(on) max (mOhm)", "ID (A)", "Package", "TID rating (krad)"]]
    data += [[r[0], str(r[1]), f"{r[2]:.1f}", f"{r[3]:.1f}", r[4], str(r[5])] for r in SPEC_ROWS]
    doc = SimpleDocTemplate(str(path), pagesize=letter, title="DS-150 Device Parameter Specification",
                            topMargin=40, bottomMargin=40)
    doc.build([
        Paragraph("DS-150 Device Parameter Specification", s["Title"]),
        Paragraph("Electrical limits for all qualified N-channel power MOSFETs at 25 &deg;C. TID rating is the "
                  "total ionizing dose qualification level.", s["Small"]),
        Spacer(1, 6),
        grid_table(data, [1.3 * inch, 0.8 * inch, 1.3 * inch, 0.7 * inch, 0.8 * inch, 1.1 * inch], 7),
    ], onFirstPage=footer_fn(f"{COMPANY} - DS-150 Rev F"))


SUPPLIERS = [
    ("Kestrel Wire Co.", "Gold bond wire 1.0 mil", "ISO 9001", "2026-03-31"),
    ("Brightline Alloys", "Aluminum bond wire 15 mil", "AS9100", "2026-01-15"),
    ("Morrow Ceramics", "TO-254 ceramic seal rings", "AS9100", "2025-12-01"),
    ("Pellam Metals", "AuSn 80/20 preforms", "ISO 9001", "2026-06-30"),
    ("Corvid Lids", "Kovar lids, gold plated", "AS9100", "2026-02-28"),
    ("Tallis Gas", "Helium 99.999 percent", "ISO 9001", "2025-11-30"),
]


def make_supplier_list(path: Path) -> None:
    """Columns aligned with spaces and no ruling lines: no table to detect."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setTitle("Approved Supplier List")
    c.setFont("Helvetica-Bold", 15)
    c.drawString(54, 730, "Approved Supplier List (ASL) - October 2025")
    c.setFont("Helvetica", 9)
    c.drawString(54, 712, "Suppliers approved for production material. Purchasing may not order production "
                          "material from any supplier not on this list.")
    xs = [54, 190, 360, 460]
    c.setFont("Helvetica-Bold", 9)
    for x, h in zip(xs, ["Supplier", "Material", "Certification", "Approval expires"]):
        c.drawString(x, 680, h)
    c.setFont("Helvetica", 9)
    y = 662
    for row in SUPPLIERS:
        for x, v in zip(xs, row):
            c.drawString(x, y, v)
        y -= 16
    c.drawString(54, y - 20, "Questions about supplier status go to Supplier Quality (ext. 4471).")
    c.setFont("Helvetica", 8)
    c.drawString(54, 28, f"{COMPANY} - Purchasing")
    c.save()


def make_fa_report(path: Path) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate

    s = styles()
    P = lambda t, st="BodyText": Paragraph(t, s[st])  # noqa: E731
    scan = path.parent / "_fa_scan.jpg"
    scanned_page([
        ("MRB DISPOSITION - FA-25-031", 54, True), ("", 0, False),
        ("Lot: HM25-1187   Device: HM-R150C02   Qty: 1,240 units", 38, False),
        ("Finding: die attach voids 31% (limit 10%)", 38, False),
        ("Disposition: SCRAP entire lot", 44, True),
        ("Customer notified: yes, return CR-2207", 38, False), ("", 0, False),
        ("Approved by: R. Okafor, Quality Manager", 38, False),
        ("Date: 2025-08-19", 38, False),
    ], rotate=-0.6, seed=7).save(scan, quality=70)
    doc = SimpleDocTemplate(str(path), pagesize=letter, title="FA-25-031 Failure Analysis Report")
    doc.build([
        P("FA-25-031 Failure Analysis Report", "Title"),
        P("Customer return CR-2207, lot HM25-1187", "Heading2"),
        P("1 Summary", "Heading1"),
        P("Four units returned by the customer failed thermal impedance screening. X-ray inspection of the "
          "returned units and of 20 retained units showed die attach voiding of 18 to 31 percent against a "
          "limit of 10 percent for TO-254. Die shear on five units averaged 3.1 kgf against a minimum of "
          "5.0 kgf."),
        P("2 Root cause", "Heading1"),
        P("The AuSn preform lot PF-3381 had oxidized surfaces. Wetting was incomplete at the standard stage "
          "temperature, which left voids under the die. The preform supplier confirmed a storage excursion."),
        PageBreak(),
        P("3 Corrective actions", "Heading1"),
        P("Incoming inspection X-rays one preform coupon per preform lot. Preforms are stored in nitrogen "
          "cabinets. WI-0217 now requires a void check on the first three units of every lot."),
        P("4 Containment", "Heading1"),
        P("All WIP built with preform lot PF-3381 was placed on hold with code HL-VOID. Lots HM25-1190 and "
          "HM25-1192 passed X-ray screening and were released."),
        PageBreak(),
        RLImage(str(scan), width=6.2 * inch, height=6.2 * inch * 2200 / 1700),
    ], onFirstPage=footer_fn(f"{COMPANY} - FA-25-031"), onLaterPages=footer_fn(f"{COMPANY} - FA-25-031"))
    scan.unlink()


def make_scan_pdf(path: Path, pages: list[Image.Image], title: str) -> None:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    c = canvas.Canvas(str(path), pagesize=letter)
    c.setTitle(title)
    for img in pages:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        buf.seek(0)
        c.drawImage(ImageReader(buf), 0, 0, width=letter[0], height=letter[1])
        c.showPage()
    c.save()


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------
def make_wi(path: Path) -> None:
    import docx
    from docx.shared import Inches

    d = docx.Document()
    d.core_properties.title = "WI-0217 Die Attach"
    d.add_heading("WI-0217 Die Attach", 0)
    d.add_paragraph("Revision D. Applies to eutectic AuSn die attach on the Palomar 3800 die bonder.")
    d.add_heading("Materials", 1)
    d.add_paragraph("AuSn 80/20 preforms from an approved supplier, stored in a nitrogen cabinet. "
                    "Preforms out of the cabinet for more than 8 hours must be scrapped.")
    d.add_heading("Procedure", 1)
    for step in [
        "Verify the preform lot number against the traveler and scan it into the MES.",
        "Set the stage temperature to 335 °C and wait for the reading to stabilize within ±3 °C.",
        "Set nitrogen forming gas flow to 20 L/min over the stage.",
        "Scrub the die for 3 seconds with 50 gf bond force.",
        "X-ray the first three units of every lot before continuing (added after FA-25-031).",
    ]:
        d.add_paragraph(step, style="List Number")
    d.add_heading("Void limits", 2)
    t = d.add_table(rows=2, cols=3)
    t.style = "Table Grid"
    t.cell(0, 0).text = "Package"
    t.cell(0, 1).merge(t.cell(0, 2)).text = "Maximum void area (%)"
    t.cell(1, 0).text = ""
    t.cell(1, 1).text = "X-ray"
    t.cell(1, 2).text = "SAM"
    for row in [["TO-254", "10", "15"], ["TO-257", "10", "15"], ["SMD-0.5", "15", "20"], ["SMD-1", "15", "20"]]:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = v
    d.add_paragraph("Stage setup reference:")
    img = path.parent / "_stage.png"
    label_figure(["PALOMAR 3800 STAGE", "N2 NOZZLE"], size=(1000, 380)).save(img)
    d.add_picture(str(img), width=Inches(4.5))
    img.unlink()
    d.save(path)


def make_mrb_minutes(path: Path) -> None:
    import docx

    d = docx.Document()
    d.core_properties.title = "MRB Minutes 2025-10-14"
    d.add_heading("Material Review Board - Minutes", 0)
    d.add_paragraph("Date: 2025-10-14. Attendees: R. Okafor (chair), J. Marsh, D. Whitfield, P. Lindqvist.")
    d.add_heading("Decisions", 1)
    d.add_paragraph("Lot HM25-1201: cosmetic scratch on lid marking. Disposition USE AS IS under customer "
                    "waiver W-552.")
    d.add_paragraph("Lot HM25-1215: illegible laser marking on 64 units. Disposition REWORK: re-mark and "
                    "re-inspect.")
    d.add_paragraph("Lot HM25-1222: fine leak failures on 3 of 400 units after burn-in. Disposition: "
                    "screen 100 percent and scrap failures.")
    d.add_heading("Action items", 1)
    t = d.add_table(rows=1, cols=3)
    t.style = "Table Grid"
    for c, v in zip(t.rows[0].cells, ["Action", "Owner", "Due"]):
        c.text = v
    for row in [["Retrain operators on SOP-0412 Rev C", "D. Whitfield", "2025-11-15"],
                ["Close deviation DV-2025-044 or extend it", "J. Marsh", "2025-10-31"],
                ["Add HL-SEAL to the hold code report", "P. Lindqvist", "2025-10-24"]]:
        cells = t.add_row().cells
        for c, v in zip(cells, row):
            c.text = v
    d.save(path)


# ---------------------------------------------------------------------------
# text, markdown, html
# ---------------------------------------------------------------------------
SHIFT_LOG = """Shift log - Fairbury assembly - 2025-10-20 - night shift

22:10  Started burn-in load BI-03 with lots HM25-1230 and HM25-1231 (160 h at 125°C).
23:45  Wire bonder WB-04 alarm "bond placement error" twice on lot HM25-1228.
02:15  Maintenance replaced the WB-04 capillary; alarm cleared after test bonds passed.
05:40  Lid seal furnace F-3 alarm "zone 3 over-temp" – reset, zone 3 reading back to 320°C.
       Note: F-3 is running at 320°C under deviation DV-2025-044.
06:30  Handoff to day shift. Open item: WB-04 pull test data for HM25-1228 still pending.
"""

LINE_STATUS = """Line 2 status board export
Updated 2025-10-21 07:00

Die attach: running (Palomar 3800 #2)
Wire bond: WB-04 back in production after capillary change
Lid seal: F-3 on deviation DV-2025-044 (320°C), F-2 down for PM until 2025-10-23
Burn-in: BI-05 and BI-06 available; BI-03 loaded
"""

WIREBOND_MD = """---
title: Troubleshooting - Wire Bond
owner: Process Engineering
---

# Troubleshooting - Wire Bond

Use this page when pull test results or bonder alarms point to a wire bond problem.
Check the bonder's alarm history in the MES before changing anything.

## Symptoms and likely causes

| Symptom | Likely cause | First action |
| --- | --- | --- |
| Heel cracks | Loop too low or excessive bond force | Check loop height and force settings |
| Lift at first bond | Contaminated bond pad | Plasma clean the lot and re-test |
| Low pull strength on all wires | Worn capillary or wedge | Replace the tool and re-qualify |
| Bond placement error alarm | Vision misalignment | Re-teach the pattern recognition |

## Pull test minimums

Aluminum 15 mil wire needs at least 150 gf; gold 1.0 mil wire needs at least 3.0 gf.

## Query recent pull failures

```sql
SELECT lot_id, wire_id, pull_gf FROM pull_tests WHERE pull_gf < 150 AND test_date > '2025-10-01';
```
"""

HOLDCODES_MD = """# MES Hold Codes

Every hold placed in the MES needs a reason code. Codes marked QE require
Quality Engineering approval to release.

| Code | Meaning | Release needs |
| --- | --- | --- |
| HL-SEAL | Fine or gross leak failure (SOP-0412 Rev C) | QE |
| HL-VOID | Die attach voiding above limit | QE |
| HL-REL | Reliability screen failure (burn-in, temp cycle) | QE |
| HL-ESD | ESD control violation in the area | Supervisor |
| HL-MARK | Marking defect | Supervisor |
| HL-DOC | Missing or wrong traveler data | Supervisor |

HL-LEAK was the code for leak failures before SOP-0412 Rev C; it is retired.
"""

ESD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>ESD Control Program</title>
<script>window.analytics = {site: "intranet"};</script>
<style>body { font-family: sans-serif }</style></head>
<body>
<nav><a href="/">Intranet home</a> | <a href="/quality">Quality</a> | <a href="/hr">HR</a></nav>
<main>
<h1>ESD Control Program</h1>
<p>Electrostatic discharge damages power MOSFET gates long before anyone feels a spark. Everyone who
enters an ESD protected area (EPA) follows these rules.</p>
<h2>Personnel grounding</h2>
<ul>
  <li>Test your wrist strap at the entry tester every time you enter.
    <ul><li>Pass range: 750 kilohm to 35 megohm.</li><li>A fail means you do not enter.</li></ul>
  </li>
  <li>Wear heel grounders on both feet in areas without wrist strap stations.</li>
  <li>Wear the ESD smock fully buttoned.</li>
</ul>
<h2>Equipment checks</h2>
<p>Ionizers are verified weekly by the area owner; offset voltage must stay within plus or minus 35 V.
Work surfaces are checked monthly for resistance to ground.</p>
<table>
  <caption>EPA zones</caption>
  <tr><th>Zone</th><th>Area</th><th>Owner</th></tr>
  <tr><td>EPA-1</td><td>Die attach and wire bond</td><td>Assembly</td></tr>
  <tr><td>EPA-2</td><td>Lid seal and leak test</td><td>Assembly</td></tr>
  <tr><td>EPA-3</td><td>Electrical test and burn-in</td><td>Test</td></tr>
</table>
<p>Violations are logged in the MES with hold code HL-ESD on any lot in the area.</p>
</main>
<footer>Copyright Halden Microsystems. Internal use only.</footer>
</body></html>
"""

GOWNING_HTM = """<html><head><title>Cleanroom Gowning</title>
<link rel="stylesheet" href="Cleanroom Gowning_files/style.css"></head>
<body><div id="menu"><a href="/">Home</a></div>
<h1>Cleanroom Gowning</h1>
<p>The assembly cleanroom is ISO Class 7. Gown in this order before entering.</p>
<ol><li>Hair cover and beard cover</li><li>Hood</li><li>Coverall, zipped to the neck</li>
<li>Boot covers</li><li>Gloves, cuffs over the coverall sleeves</li></ol>
<p>Coveralls are changed weekly and gloves at every entry. A torn glove is replaced immediately.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# data files
# ---------------------------------------------------------------------------
CAL_ROWS: list[dict] = []
BURNIN_ROWS: list[dict] = []
YIELD_ROWS: list[dict] = []
EQUIP_ROWS: list[dict] = []


def make_calibration_csv(path: Path) -> None:
    kinds = [("HLD", "Helium leak detector", 12), ("BPT", "Bond pull tester", 12), ("DST", "Die shear tester", 12),
             ("TC", "Thermocouple reader", 6), ("XR", "X-ray inspection", 12), ("MM", "Multimeter", 12),
             ("TM", "Torque meter", 6), ("ESD", "Wrist strap tester", 6)]
    owners = ["Metrology", "Assembly", "Test", "Reliability"]
    for i in range(28):
        code, desc, interval = kinds[i % len(kinds)]
        last = date(2024, 9, 1) + timedelta(days=R.randrange(0, 400))
        due = last + timedelta(days=round(interval * 30.4))
        CAL_ROWS.append({"id": f"{code}-{i + 1:02d}" if code != "HLD" else f"HLD-{i + 3}",
                         "desc": desc, "last": last, "interval": interval, "due": due,
                         "owner": owners[i % 4]})
    CAL_ROWS[0].update({"id": "HLD-7", "last": date(2025, 6, 12), "interval": 12, "due": date(2026, 6, 12),
                        "owner": "Metrology"})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["Instrument ID", "Description", "Last calibration", "Interval (months)", "Due date", "Owner"])
        for r in CAL_ROWS:
            w.writerow([r["id"], r["desc"], r["last"].isoformat(), r["interval"], r["due"].isoformat(), r["owner"]])


def make_burnin_csv(path: Path) -> None:
    parts = ["HM-R150C02", "HM-R200D03", "HM-R100B01", "HM-R250E04"]
    ovens = ["BI-01", "BI-02", "BI-03", "BI-04", "BI-05", "BI-06"]
    lot = 1100
    day = date(2025, 7, 1)
    while day <= date(2025, 9, 30):
        for _ in range(R.choice([3, 4, 5])):
            lot += 1
            oven = R.choice(ovens[:4] if day < date(2025, 8, 11) else ovens)
            qty = R.choice([200, 250, 300, 400])
            rate = R.uniform(0.002, 0.02) if lot != 1187 else 0.11
            fail = int(round(qty * rate))
            BURNIN_ROWS.append({"lot": f"HM25-{lot}", "part": R.choice(parts), "oven": oven,
                                "start": day, "hours": R.choice([160, 160, 160, 168, 240]),
                                "qty_in": qty, "qty_pass": qty - fail, "qty_fail": fail})
        day += timedelta(days=1)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["lot_id", "part_number", "oven", "start_date", "hours", "qty_in", "qty_pass", "qty_fail"])
        for r in BURNIN_ROWS:
            w.writerow([r["lot"], r["part"], r["oven"], r["start"].isoformat(), r["hours"], r["qty_in"],
                        r["qty_pass"], r["qty_fail"]])


MONTHS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep"]


def make_yield_xlsx(path: Path) -> None:
    import openpyxl
    from openpyxl.styles import Alignment, Font

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Monthly"
    ws["A1"] = "FY25 assembly and test yield (fiscal year Oct 2024 - Sep 2025)"
    ws["A1"].font = Font(bold=True, size=13)
    ws.append([])
    ws.append(["Month", "Assembly", None, "Test", None, "Shipped"])
    ws.append([None, "In", "Out", "In", "Out", None])
    ws.merge_cells("B3:C3")
    ws.merge_cells("D3:E3")
    ws["B3"].alignment = ws["D3"].alignment = Alignment(horizontal="center")
    for m in MONTHS:
        a_in = R.randrange(38000, 46000)
        a_out = int(a_in * R.uniform(0.955, 0.985))
        t_in = a_out
        t_out = int(t_in * (R.uniform(0.90, 0.96) if m != "Aug" else 0.861))
        YIELD_ROWS.append({"month": m, "a_in": a_in, "a_out": a_out, "t_in": t_in, "t_out": t_out, "ship": t_out})
        ws.append([m, a_in, a_out, t_in, t_out, t_out])
    best = max(YIELD_ROWS, key=lambda r: r["t_out"] / r["t_in"])
    ws["H3"], ws["I3"] = "Metric", "Value"
    ws["H4"], ws["I4"] = "FY25 test yield target", "92%"
    ws["H5"], ws["I5"] = "Best test yield month", best["month"]
    ws["H6"], ws["I6"] = "Owner", "Yield engineering"
    t = wb.create_sheet("Targets")
    t.append(["Metric", "FY25 target", "FY26 target"])
    t.append(["Assembly yield (%)", 97.0, 97.5])
    t.append(["Test yield (%)", 92.0, 93.0])
    t.append(["On-time delivery (%)", 95.0, 96.0])
    h = wb.create_sheet("Scratch")
    h.append(["do not use", "draft numbers", 123])
    h.sheet_state = "hidden"
    wb.save(path)


def make_equipment_xlsx(path: Path) -> None:
    import openpyxl

    areas = {"Die attach": ["Palomar 3800 die bonder", "Nitrogen cabinet"],
             "Wire bond": ["K&S wedge bonder", "Bond pull tester"],
             "Seal": ["Lid seal furnace", "Helium leak detector", "Gross leak bath"],
             "Test": ["Parametric tester", "Burn-in oven"]}
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Assets"
    ws.append(["Asset ID", "Description", "Area", "PM interval (days)", "Last PM"])
    n = 0
    for area, kinds in areas.items():
        for k in range(10):
            n += 1
            desc = kinds[k % len(kinds)]
            interval = R.choice([30, 60, 90, 180])
            last = date(2025, 6, 1) + timedelta(days=R.randrange(0, 140))
            EQUIP_ROWS.append({"id": f"EQ-{n:03d}", "desc": desc, "area": area, "interval": interval, "last": last})
            ws.append([f"EQ-{n:03d}", desc, area, interval, last])
    wb.save(path)


# ---------------------------------------------------------------------------
# code
# ---------------------------------------------------------------------------
HOLD_SERVICE_CS = '''using System;
using System.Collections.Generic;
using System.Linq;

namespace Halden.Mes.Holds
{
    /// <summary>Places and releases lot holds in the MES.</summary>
    public sealed class HoldService
    {
        /// <summary>Hold codes whose release needs Quality Engineering approval.</summary>
        public static readonly IReadOnlySet<string> QeApprovalCodes =
            new HashSet<string> { "HL-SEAL", "HL-VOID", "HL-REL" };

        public const int MaxCommentLength = 500;

        private readonly IHoldRepository _repository;
        private readonly IClock _clock;

        public HoldService(IHoldRepository repository, IClock clock)
        {
            _repository = repository;
            _clock = clock;
        }

        /// <summary>Places a hold on a lot.</summary>
        /// <param name="lotId">Lot to hold, e.g. HM25-1187.</param>
        /// <param name="code">Hold reason code; must be an active code.</param>
        /// <exception cref="ArgumentException">Unknown or retired code, or comment too long.</exception>
        public Hold PlaceHold(string lotId, string code, string comment, string user)
        {
            if (!HoldReasons.IsActive(code))
                throw new ArgumentException($"Hold code {code} is unknown or retired.", nameof(code));
            if (comment.Length > MaxCommentLength)
                throw new ArgumentException($"Comment is limited to {MaxCommentLength} characters.", nameof(comment));
            var hold = new Hold(lotId, code, comment, user, _clock.UtcNow);
            _repository.Add(hold);
            return hold;
        }

        /// <summary>Releases a hold. QE codes need an approver from the QE group.</summary>
        public void ReleaseHold(Guid holdId, string user, string? qeApprover = null)
        {
            var hold = _repository.Get(holdId) ?? throw new KeyNotFoundException($"No hold {holdId}");
            if (RequiresQeApproval(hold.Code) && string.IsNullOrWhiteSpace(qeApprover))
                throw new InvalidOperationException($"Releasing {hold.Code} requires QE approval.");
            _repository.Release(holdId, user, qeApprover, _clock.UtcNow);
        }

        public static bool RequiresQeApproval(string code) => QeApprovalCodes.Contains(code);
    }
}
'''

REPO_CS = '''using System;
using System.Collections.Generic;

namespace Halden.Mes.Holds
{
    public interface IHoldRepository
    {
        void Add(Hold hold);
        Hold? Get(Guid holdId);
        IEnumerable<Hold> OpenHoldsForLot(string lotId);
        void Release(Guid holdId, string user, string? qeApprover, DateTime releasedUtc);
    }

    public interface IClock
    {
        DateTime UtcNow { get; }
    }
}
'''

REASONS_CS = '''using System;
using System.Collections.Generic;

namespace Halden.Mes.Holds
{
    public sealed record Hold(string LotId, string Code, string Comment, string PlacedBy, DateTime PlacedUtc)
    {
        public Guid Id { get; init; } = Guid.NewGuid();
    }

    public static class HoldReasons
    {
        private static readonly HashSet<string> Active = new()
        {
            "HL-SEAL", "HL-VOID", "HL-REL", "HL-ESD", "HL-MARK", "HL-DOC",
        };

        /// <summary>HL-LEAK was retired when SOP-0412 Rev C introduced HL-SEAL.</summary>
        private static readonly HashSet<string> Retired = new() { "HL-LEAK" };

        public static bool IsActive(string code) => Active.Contains(code) && !Retired.Contains(code);
    }
}
'''

DESIGNER_CS = '''namespace Halden.Mes.Holds.Forms
{
    partial class MainForm
    {
        private void InitializeComponent() { this.SuspendLayout(); this.ResumeLayout(false); }
    }
}
'''

YIELD_PY = '''"""Yield calculations for the weekly yield report."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import prod


@dataclass(frozen=True)
class StepYield:
    step: str
    units_in: int
    units_out: int

    @property
    def yield_fraction(self) -> float:
        return self.units_out / self.units_in if self.units_in else 0.0


def rolled_throughput_yield(steps: list[StepYield]) -> float:
    """Rolled throughput yield: the product of every step's yield (not the average)."""
    return prod(s.yield_fraction for s in steps)


def first_pass_yield(units_in: int, units_passed_first_time: int) -> float:
    """Share of units that pass a step without rework."""
    return units_passed_first_time / units_in if units_in else 0.0


@lru_cache(maxsize=128)
def target_for(metric: str) -> float:
    """FY26 targets; test yield is 93 percent."""
    return {"assembly": 0.975, "test": 0.93, "otd": 0.96}[metric]
'''

SPC_PY = '''"""Western Electric rules for the SPC charts on burn-in failure rates."""

from __future__ import annotations


def rule_1(points: list[float], mean: float, sigma: float) -> list[int]:
    """One point beyond 3 sigma."""
    return [i for i, p in enumerate(points) if abs(p - mean) > 3 * sigma]


def rule_2(points: list[float], mean: float, sigma: float) -> list[int]:
    """Two of three consecutive points beyond 2 sigma on the same side."""
    hits = []
    for i in range(2, len(points)):
        window = points[i - 2:i + 1]
        for side in (1, -1):
            if sum(1 for p in window if side * (p - mean) > 2 * sigma) >= 2:
                hits.append(i)
                break
    return hits


def rule_4(points: list[float], mean: float) -> list[int]:
    """Eight consecutive points on the same side of the center line."""
    hits = []
    for i in range(7, len(points)):
        window = points[i - 7:i + 1]
        if all(p > mean for p in window) or all(p < mean for p in window):
            hits.append(i)
    return hits
'''

SCHEMA_SQL = """-- MES lot and hold schema (SQL Server)
CREATE TABLE dbo.Lots (
    LotId       VARCHAR(20)  NOT NULL PRIMARY KEY,
    PartNumber  VARCHAR(20)  NOT NULL,
    Quantity    INT          NOT NULL,
    StartedUtc  DATETIME2    NOT NULL
);
GO

CREATE TABLE dbo.Holds (
    HoldId      UNIQUEIDENTIFIER NOT NULL PRIMARY KEY DEFAULT NEWID(),
    LotId       VARCHAR(20)  NOT NULL REFERENCES dbo.Lots(LotId),
    Code        VARCHAR(10)  NOT NULL REFERENCES dbo.HoldReasons(Code),
    Comment     NVARCHAR(500) NULL,
    PlacedBy    VARCHAR(50)  NOT NULL,
    PlacedUtc   DATETIME2    NOT NULL,
    ReleasedUtc DATETIME2    NULL,
    QeApprover  VARCHAR(50)  NULL
);
GO

/* Open holds with their age in hours; used by the hold dashboard. */
CREATE VIEW dbo.vw_OpenHolds AS
SELECT h.HoldId, h.LotId, l.PartNumber, h.Code,
       DATEDIFF(HOUR, h.PlacedUtc, SYSUTCDATETIME()) AS AgeHours
FROM dbo.Holds h
JOIN dbo.Lots l ON l.LotId = h.LotId
WHERE h.ReleasedUtc IS NULL;
GO

-- Places a hold; rejects retired codes such as HL-LEAK.
CREATE PROCEDURE dbo.usp_PlaceHold
    @LotId VARCHAR(20), @Code VARCHAR(10), @Comment NVARCHAR(500), @User VARCHAR(50)
AS
BEGIN
    SET NOCOUNT ON;
    BEGIN TRY
        IF NOT EXISTS (SELECT 1 FROM dbo.HoldReasons WHERE Code = @Code AND IsActive = 1)
        BEGIN
            RAISERROR('Hold code %s is unknown or retired', 16, 1, @Code);
        END
        INSERT INTO dbo.Holds (LotId, Code, Comment, PlacedBy, PlacedUtc)
        VALUES (@LotId, @Code, @Comment, @User, SYSUTCDATETIME());
    END TRY
    BEGIN CATCH
        THROW;
    END CATCH
END
GO
"""

WEEKLY_SQL = """-- Weekly test yield by part number, last 12 weeks
WITH weekly AS (
    SELECT PartNumber,
           DATEPART(ISO_WEEK, TestedUtc) AS Week,
           SUM(UnitsIn)  AS UnitsIn,
           SUM(UnitsOut) AS UnitsOut
    FROM dbo.TestResults
    WHERE TestedUtc >= DATEADD(WEEK, -12, SYSUTCDATETIME())
    GROUP BY PartNumber, DATEPART(ISO_WEEK, TestedUtc)
)
SELECT PartNumber, Week, UnitsIn, UnitsOut,
       CAST(100.0 * UnitsOut / NULLIF(UnitsIn, 0) AS DECIMAL(5, 2)) AS YieldPct
FROM weekly
ORDER BY PartNumber, Week;
"""


def seed_sql() -> str:
    rows = [("HL-SEAL", "Leak failure", 1), ("HL-VOID", "Die attach voids", 1), ("HL-REL", "Reliability failure", 1),
            ("HL-ESD", "ESD violation", 1), ("HL-MARK", "Marking defect", 1), ("HL-DOC", "Traveler data", 1),
            ("HL-LEAK", "Leak failure (retired)", 0)]
    rows += [(f"HL-X{i:02d}", f"Legacy reason {i}", 0) for i in range(30)]
    return "-- Reference data for dbo.HoldReasons\n" + "".join(
        f"INSERT INTO dbo.HoldReasons (Code, Description, IsActive) VALUES ('{c}', '{d}', {a});\n" for c, d, a in rows)


# ---------------------------------------------------------------------------
# questions
# ---------------------------------------------------------------------------
def questions() -> list[dict]:
    aug_bi03 = sum(r["qty_fail"] for r in BURNIN_ROWS if r["oven"] == "BI-03" and r["start"].month == 8)
    worst = max(BURNIN_ROWS, key=lambda r: r["qty_fail"] / r["qty_in"])
    asof = date(2025, 11, 1)
    overdue = sorted(r["id"] for r in CAL_ROWS if r["due"] < asof)
    low = min(YIELD_ROWS, key=lambda r: r["t_out"] / r["t_in"])
    seal_assets = sum(1 for r in EQUIP_ROWS if r["area"] == "Seal")
    to257_200 = sorted(r[0] for r in SPEC_ROWS if r[4] == "TO-257" and r[1] == 200)
    lowest_rds = min((r for r in SPEC_ROWS if r[1] == 150), key=lambda r: r[2])
    new_ovens = sum(r["qty_in"] for r in BURNIN_ROWS if r["oven"] in ("BI-05", "BI-06"))

    def q(id_, type_, question, answer, expect, source, sql=None, note=None):
        return {"id": id_, "type": type_, "question": question, "answer": answer, "expect": expect,
                "source": source, "sql": sql, "note": note}

    sop = "*SOP-0412 Hermetic Seal Leak Test Rev C*"
    return [
        # text PDF, current revision
        q("sop-1", "pdf_text", "How long are parts observed in the gross leak bath?",
          "30 seconds (SOP-0412 Rev C). Rev B said 60 seconds; Rev C is current.",
          [{"file": sop, "contains": "observe them for 30 seconds"}], "SOP-0412 Rev C, section 6",
          note="Revision trap: the archived Rev B says 60 seconds."),
        q("sop-2", "pdf_text", "Which packages does the seal leak test procedure cover?",
          "TO-254, TO-257, SMD-0.5 and SMD-1.", [{"file": sop, "contains": "SMD-1"}], "SOP-0412 Rev C, section 2"),
        q("sop-3", "pdf_text", "What MES hold code is used for parts that fail leak test?",
          "HL-SEAL (it replaced HL-LEAK in Rev C).", [{"file": sop, "contains": "HL-SEAL"},
                                                    {"file": "*Hold Codes*", "contains": "HL-SEAL"}],
          "SOP-0412 Rev C section 7; MES Hold Codes"),
        q("sop-4", "pdf_table", "What bomb pressure and time are used for a 0.03 cc cavity?",
          "60 psig for 4 hours (cavity > 0.01 to 0.05 cc), max dwell 1 h, reject above 5.0E-8 atm-cc/s.",
          [{"file": sop, "contains": "0.05"}], "SOP-0412 Rev C, Table 2"),
        q("sop-5", "pdf_text", "How long are leak test records kept?", "Ten years.",
          [{"file": sop, "contains": "ten years"}], "SOP-0412 Rev C, section 8"),
        q("sop-6", "pdf_table", "What changed in revision C of SOP-0412?",
          "Added SMD-1; gross leak observation reduced from 60 s to 30 s; hold code HL-LEAK renamed HL-SEAL.",
          [{"file": sop, "contains": "renamed HL-SEAL"}], "SOP-0412 Rev C, revision history"),
        q("fig-1", "figure", "What equipment is shown in the helium fine leak test setup figure?",
          "Bomb chamber BC-2 and leak detector HLD-7.", [{"file": sop, "kind": "figure"}],
          "SOP-0412 Rev C, Figure 1 (text read by OCR)"),
        # two-column newsletter
        q("news-1", "pdf_two_column", "Which burn-in ovens were added in Q3 and how much capacity did they add?",
          "BI-05 and BI-06, commissioned in August; +40 percent burn-in capacity.",
          [{"file": "*Newsletter*", "contains": "BI-06"}], "Reliability Newsletter Q3 2025"),
        q("news-2", "pdf_two_column", "How many days without a lost-time incident did the site reach?",
          "212 days (on September 30).", [{"file": "*Newsletter*", "contains": "212 days"}],
          "Reliability Newsletter Q3 2025"),
        q("news-3", "pdf_two_column", "Who is the new metrology lead?", "Dana Whitfield (joined in September).",
          [{"file": "*Newsletter*", "contains": "Dana Whitfield"}], "Reliability Newsletter Q3 2025"),
        # large spec table -> tables.db
        q("spec-1", "pdf_big_table", "Which TO-257 parts are rated for 200 V BVDSS?", ", ".join(to257_200),
          [{"file": "*DS-150*"}], "DS-150 (large table, also in tables.db)",
          sql="SELECT part_number FROM t_ds_150_device_parameter_specification_table_1 "
              "WHERE package = 'TO-257' AND bvdss_v = 200 ORDER BY part_number"),
        q("spec-2", "pdf_big_table", "Which 150 V part has the lowest RDS(on)?",
          f"{lowest_rds[0]} at {lowest_rds[2]:.1f} mOhm.", [{"file": "*DS-150*"}], "DS-150",
          sql="SELECT part_number, rds_on_max_mohm FROM t_ds_150_device_parameter_specification_table_1 "
              "WHERE bvdss_v = 150 ORDER BY rds_on_max_mohm LIMIT 1"),
        # table without ruling lines
        q("asl-1", "pdf_unruled_table", "Who supplies the gold bond wire and when does their approval expire?",
          "Kestrel Wire Co.; approval expires 2026-03-31.", [{"file": "*Supplier*", "contains": "Kestrel"}],
          "Approved Supplier List"),
        q("asl-2", "pdf_unruled_table", "Which supplier's approval expires first?",
          "Tallis Gas (helium), 2025-11-30.", [{"file": "*Supplier*", "contains": "Tallis"}],
          "Approved Supplier List"),
        # half-scanned FA report
        q("fa-1", "pdf_mixed", "What was the root cause of customer return CR-2207?",
          "Die attach voiding (18-31 percent vs 10 percent limit) on lot HM25-1187, caused by oxidized AuSn "
          "preform lot PF-3381.", [{"file": "*FA-25-031*"}, {"file": "*Newsletter*", "contains": "CR-2207"}],
          "FA-25-031; newsletter"),
        q("fa-2", "pdf_scanned_page", "What was the disposition of lot HM25-1187 and who approved it?",
          "Scrap the entire lot (1,240 units); approved by R. Okafor, Quality Manager, 2025-08-19.",
          [{"file": "*FA-25-031*", "page": 3}], "FA-25-031 page 3 (scanned, OCR)"),
        q("fa-3", "pdf_text", "Which lots were held because of preform lot PF-3381 and later released?",
          "HM25-1190 and HM25-1192 passed X-ray and were released (hold code HL-VOID).",
          [{"file": "*FA-25-031*", "contains": "HM25-1190"}], "FA-25-031 section 4"),
        # scans
        q("dv-1", "pdf_scanned", "Why is furnace F-3 running at 320 °C?",
          "Deviation DV-2025-044: thermocouple TC-3 replacement is on backorder, so F-3 runs at 320 °C instead "
          "of 330 °C from 2025-10-01 to 2025-10-31, with 5 extra fine-leak samples per lot.",
          [{"file": "*DV-2025-044*"}, {"file": "*shift_log*", "contains": "DV-2025-044"}],
          "DV-2025-044 (scanned); shift log"),
        q("dv-2", "pdf_scanned", "Who approved deviation DV-2025-044?", "J. Marsh, Quality Engineering.",
          [{"file": "*DV-2025-044*", "page": 2}], "DV-2025-044 page 2 (scanned)"),
        q("cal-1", "pdf_scanned", "When is the helium leak detector HLD-7 due for calibration?",
          "2026-06-12 (calibrated 2025-06-12, certificate CC-88213).",
          [{"file": "*CC-88213*"}, {"file": "*calibration_register*", "kind": "table_card"}],
          "Calibration certificate CC-88213 (scanned); calibration register"),
        # Word
        q("wi-1", "docx", "What stage temperature is used for die attach?", "335 °C, stable within ±3 °C.",
          [{"file": "*WI-0217*", "contains": "335"}], "WI-0217 Die Attach, step 2"),
        q("wi-2", "docx_table", "What is the maximum void area for SMD-1 by SAM?", "20 percent.",
          [{"file": "*WI-0217*", "contains": "SAM"}], "WI-0217 void limits table (merged header)"),
        q("wi-3", "docx", "How long can AuSn preforms be out of the nitrogen cabinet?",
          "8 hours; after that they must be scrapped.", [{"file": "*WI-0217*", "contains": "8 hours"}],
          "WI-0217 Materials"),
        q("mrb-1", "docx", "What did the MRB decide for lot HM25-1201?",
          "Use as is under customer waiver W-552 (cosmetic lid marking scratch).",
          [{"file": "*MRB Minutes*", "contains": "HM25-1201"}], "MRB minutes 2025-10-14"),
        q("mrb-2", "docx_table", "Who owns retraining on SOP-0412 Rev C and when is it due?",
          "D. Whitfield, due 2025-11-15.", [{"file": "*MRB Minutes*", "contains": "Retrain"}],
          "MRB minutes 2025-10-14, action items"),
        # text, markdown, html
        q("log-1", "txt", "What happened to wire bonder WB-04 on the night shift of 2025-10-20?",
          "Bond placement error alarm twice on lot HM25-1228; maintenance replaced the capillary at 02:15 and the "
          "alarm cleared.", [{"file": "*shift_log*", "contains": "WB-04"}], "Shift log 2025-10-20"),
        q("log-2", "txt", "When does furnace F-2 come back from PM?", "2025-10-23.",
          [{"file": "*line2_status*", "contains": "F-2"}], "Line 2 status"),
        q("md-1", "markdown", "What usually causes heel cracks in wire bonds?",
          "Loop too low or excessive bond force; check loop height and force settings.",
          [{"file": "*Wire Bond*", "contains": "Heel cracks"}], "Troubleshooting - Wire Bond"),
        q("md-2", "markdown", "Which hold codes need QE approval to release?", "HL-SEAL, HL-VOID and HL-REL.",
          [{"file": "*Hold Codes*", "contains": "HL-REL"}, {"file": "*HoldService*", "contains": "QeApprovalCodes"}],
          "MES Hold Codes; HoldService.cs"),
        q("html-1", "html", "What resistance range must a wrist strap test pass?", "750 kilohm to 35 megohm.",
          [{"file": "*ESD Program*", "contains": "750 kilohm"}], "ESD Control Program (intranet)"),
        q("html-2", "html", "How often are coveralls changed in the cleanroom?",
          "Weekly (gloves at every entry).", [{"file": "*Cleanroom Gowning*", "contains": "weekly"}],
          "Cleanroom Gowning (.htm)"),
        # spreadsheets and CSV -> SQL
        q("csv-1", "csv_sql", "How many units failed burn-in in oven BI-03 in August 2025?", str(aug_bi03),
          [{"file": "*burnin_results*", "kind": "table_card"}], "burnin_results_2025Q3.csv",
          sql="SELECT SUM(qty_fail) FROM t_burnin_results_2025q3 WHERE oven = 'BI-03' "
              "AND start_date BETWEEN '2025-08-01' AND '2025-08-31'"),
        q("csv-2", "csv_sql", "Which lot had the highest burn-in failure rate in Q3?",
          f"{worst['lot']} ({worst['qty_fail']} of {worst['qty_in']} failed).",
          [{"file": "*burnin_results*", "kind": "table_card"}], "burnin_results_2025Q3.csv",
          sql="SELECT lot_id, qty_fail, qty_in, 1.0 * qty_fail / qty_in AS rate FROM t_burnin_results_2025q3 "
              "ORDER BY rate DESC LIMIT 1"),
        q("csv-3", "csv_sql", "How many units went through the new ovens BI-05 and BI-06 in Q3?", str(new_ovens),
          [{"file": "*burnin_results*", "kind": "table_card"}], "burnin_results_2025Q3.csv",
          sql="SELECT SUM(qty_in) FROM t_burnin_results_2025q3 WHERE oven IN ('BI-05', 'BI-06')"),
        q("csv-4", "csv_semicolon", "Which instruments were overdue for calibration on 2025-11-01?",
          ", ".join(overdue) or "None.", [{"file": "*calibration_register*", "kind": "table_card"}],
          "calibration_register.csv (semicolon-delimited)",
          sql="SELECT instrument_id, due_date FROM t_calibration_register WHERE due_date < '2025-11-01' "
              "ORDER BY instrument_id"),
        q("xlsx-1", "xlsx_sql", "Which month of FY25 had the lowest test yield?",
          f"{low['month']} ({100 * low['t_out'] / low['t_in']:.1f} percent).",
          [{"file": "*Yield Tracker*", "kind": "table_card"}], "Yield Tracker FY25.xlsx, Monthly (merged headers)",
          sql="SELECT month, 1.0 * test_out / test_in AS yield FROM t_yield_tracker_fy25_monthly_1 "
              "ORDER BY yield LIMIT 1"),
        q("xlsx-2", "xlsx", "What is the FY26 test yield target?", "93 percent.",
          [{"file": "*Yield Tracker*", "sheet": "Targets"}, {"file": "*yield_report*", "contains": "0.93"}],
          "Yield Tracker FY25.xlsx, Targets; yield_report.py"),
        q("xlsx-3", "xlsx_sql", "How many assets are in the Seal area?", str(seal_assets),
          [{"file": "*Equipment List*", "kind": "table_card"}], "Equipment List.xlsx",
          sql="SELECT COUNT(*) FROM t_equipment_list_assets WHERE area = 'Seal'"),
        # code
        q("cs-1", "csharp", "Which method rejects retired hold codes when placing a hold?",
          "HoldService.PlaceHold (via HoldReasons.IsActive); in SQL, dbo.usp_PlaceHold.",
          [{"file": "*HoldService.cs", "contains": "PlaceHold"}, {"file": "*mes_schema.sql", "contains": "usp_PlaceHold"}],
          "HoldService.cs; mes_schema.sql"),
        q("cs-2", "csharp", "What is the maximum hold comment length?", "500 characters.",
          [{"file": "*HoldService.cs", "contains": "MaxCommentLength"}, {"file": "*mes_schema.sql", "contains": "NVARCHAR(500)"}],
          "HoldService.cs; mes_schema.sql"),
        q("py-1", "python", "How is rolled throughput yield calculated in the yield tools?",
          "As the product of every step's yield, not the average.",
          [{"file": "*yield_report.py", "contains": "rolled_throughput_yield"}], "yield_report.py"),
        q("py-2", "python", "What does Western Electric rule 2 check?",
          "Two of three consecutive points beyond 2 sigma on the same side.",
          [{"file": "*spc.py", "contains": "rule_2"}], "spc.py"),
        q("sql-1", "sql", "Which view lists open holds and how old they are?", "dbo.vw_OpenHolds (AgeHours).",
          [{"file": "*mes_schema.sql", "contains": "vw_OpenHolds"}], "mes_schema.sql"),
        q("sql-2", "sql", "How is weekly test yield computed in the report query?",
          "Per part number and ISO week over the last 12 weeks: 100 * UnitsOut / UnitsIn from dbo.TestResults.",
          [{"file": "*weekly_yield.sql"}], "reports/weekly_yield.sql"),
        # unanswerable
        q("none-1", "not_found", "What is the cafeteria menu on Friday?", "Not in the documents.", [], "-"),
        q("none-2", "not_found", "What is the phone number of Kestrel Wire Co.?", "Not in the documents.", [], "-"),
        q("none-3", "not_found", "What was the test yield in FY23?", "Not in the documents (only FY25 data).", [], "-"),
    ]


# ---------------------------------------------------------------------------
def main(out: Path) -> None:
    docs = out / "docs"
    for sub in ("procedures/archive", "newsletters", "specs", "suppliers", "quality/scans", "work-instructions",
                "meetings", "logs", "kb", "intranet/Cleanroom Gowning_files", "data",
                "code/mes-holds/bin/Debug", "code/mes-holds/Forms", "code/yield_tools", "code/sql/reports"):
        (docs / sub).mkdir(parents=True, exist_ok=True)
    make_sop(docs / "procedures" / "SOP-0412 Hermetic Seal Leak Test Rev C.pdf", "C")
    make_sop(docs / "procedures" / "archive" / "SOP-0412 Hermetic Seal Leak Test Rev B.pdf", "B")
    (docs / "procedures" / "SOP-0412 Hermetic Seal Leak Test Rev C (copy).pdf").write_bytes(
        (docs / "procedures" / "SOP-0412 Hermetic Seal Leak Test Rev C.pdf").read_bytes())
    make_newsletter(docs / "newsletters" / "Reliability Newsletter 2025 Q3.pdf")
    make_spec(docs / "specs" / "DS-150 Device Parameter Specification.pdf")
    make_supplier_list(docs / "suppliers" / "Approved Supplier List.pdf")
    make_fa_report(docs / "quality" / "FA-25-031 Failure Analysis Report.pdf")
    make_scan_pdf(docs / "quality" / "scans" / "DV-2025-044 Deviation (scan).pdf", [
        scanned_page([("DEVIATION DV-2025-044", 56, True), ("", 0, False),
                      ("Process: Lid seal furnace F-3", 38, False),
                      ("Requirement: seal profile peak 330 C (SOP-0388)", 38, False),
                      ("Deviation: run F-3 at 320 C peak", 38, False),
                      ("Period: 2025-10-01 to 2025-10-31", 38, False), ("", 0, False),
                      ("Reason: replacement thermocouple TC-3", 38, False),
                      ("is on backorder; zone 3 reads high.", 38, False), ("", 0, False),
                      ("Risk control: fine leak test 5 extra", 38, False),
                      ("samples per lot sealed on F-3.", 38, False)], rotate=0.8, seed=3),
        scanned_page([("DV-2025-044 page 2", 44, True), ("", 0, False),
                      ("Approvals", 44, True),
                      ("Requested by: T. Ferris, Process Engineering", 38, False),
                      ("Approved by: J. Marsh, Quality Engineering", 38, False),
                      ("Date approved: 2025-09-29", 38, False), ("", 0, False),
                      ("Expires: 2025-10-31 unless extended by MRB", 38, False)], rotate=-0.5, seed=4),
    ], "DV-2025-044 Deviation")
    make_scan_pdf(docs / "quality" / "scans" / "Calibration Certificate CC-88213 (scan).pdf", [
        scanned_page([("CERTIFICATE OF CALIBRATION", 56, True), ("Certificate No. CC-88213", 40, False),
                      ("", 0, False), ("Instrument: Helium leak detector", 38, False),
                      ("Asset ID: HLD-7", 38, False), ("Calibration date: 2025-06-12", 38, False),
                      ("Next due: 2026-06-12", 38, False),
                      ("Standard leak: 2.1E-8 atm-cc/s helium", 38, False),
                      ("As found: in tolerance", 38, False), ("As left: in tolerance", 38, False), ("", 0, False),
                      ("Technician: M. Osei, Vacutek Services", 38, False)], rotate=0.4, seed=5)],
        "Calibration Certificate CC-88213")
    make_wi(docs / "work-instructions" / "WI-0217 Die Attach.docx")
    make_mrb_minutes(docs / "meetings" / "MRB Minutes 2025-10-14.docx")
    (docs / "logs" / "shift_log_2025-10-20.txt").write_text(SHIFT_LOG, encoding="cp1252")
    (docs / "logs" / "line2_status.txt").write_text(LINE_STATUS, encoding="utf-8")
    (docs / "kb" / "Troubleshooting - Wire Bond.md").write_text(WIREBOND_MD, encoding="utf-8")
    (docs / "kb" / "MES Hold Codes.md").write_text(HOLDCODES_MD, encoding="utf-8")
    (docs / "intranet" / "ESD Program.html").write_text(ESD_HTML, encoding="utf-8")
    (docs / "intranet" / "Cleanroom Gowning.htm").write_text(GOWNING_HTM, encoding="utf-8")
    (docs / "intranet" / "Cleanroom Gowning_files" / "style.css").write_text("body{margin:0}", encoding="utf-8")
    (docs / "intranet" / "Cleanroom Gowning_files" / "menu.html").write_text("<a href='/'>Home</a>", encoding="utf-8")
    make_calibration_csv(docs / "data" / "calibration_register.csv")
    make_burnin_csv(docs / "data" / "burnin_results_2025Q3.csv")
    make_yield_xlsx(docs / "data" / "Yield Tracker FY25.xlsx")
    make_equipment_xlsx(docs / "data" / "Equipment List.xlsx")
    (docs / "data" / "~$Yield Tracker FY25.xlsx").write_bytes(b"lock file")
    (docs / "code" / "mes-holds" / "HoldService.cs").write_text(HOLD_SERVICE_CS, encoding="utf-8")
    (docs / "code" / "mes-holds" / "IHoldRepository.cs").write_text(REPO_CS, encoding="utf-8")
    (docs / "code" / "mes-holds" / "HoldReasons.cs").write_text(REASONS_CS, encoding="utf-8")
    (docs / "code" / "mes-holds" / "Forms" / "MainForm.Designer.cs").write_text(DESIGNER_CS, encoding="utf-8")
    (docs / "code" / "mes-holds" / "bin" / "Debug" / "Generated.cs").write_text("class Generated {}", encoding="utf-8")
    (docs / "code" / "yield_tools" / "yield_report.py").write_text(YIELD_PY, encoding="utf-8")
    (docs / "code" / "yield_tools" / "spc.py").write_text(SPC_PY, encoding="utf-8")
    (docs / "code" / "sql" / "mes_schema.sql").write_text(SCHEMA_SQL, encoding="utf-8")
    (docs / "code" / "sql" / "seed_hold_reasons.sql").write_text(seed_sql(), encoding="utf-8")
    (docs / "code" / "sql" / "reports" / "weekly_yield.sql").write_text(WEEKLY_SQL, encoding="utf-8")

    qs = questions()
    qdir = out / "questions"
    qdir.mkdir(exist_ok=True)
    with open(qdir / "golden.jsonl", "w", encoding="utf-8") as fh:
        fh.write("# Halden Microsystems test corpus. expect: [] = not answerable from the documents.\n")
        for x in qs:
            fh.write(json.dumps({k: x[k] for k in ("id", "type", "question", "expect")}) + "\n")
    lines = ["# Halden Microsystems test corpus: questions and expected answers", "",
             "All companies, people, parts and numbers here are fictional.", "",
             "Build `docs/`, then run `rag eval <workspace> questions/golden.jsonl` for retrieval scores. "
             "Ask the same questions through your agent and compare with the answers below. Questions marked "
             "SQL need the agent to query `tables.db`; the SQL shown returns the expected answer.", ""]
    current = None
    for x in qs:
        if x["type"] != current:
            current = x["type"]
            lines += [f"## {current}", ""]
        lines.append(f"**{x['id']}. {x['question']}**  ")
        lines.append(f"Answer: {x['answer']}  ")
        lines.append(f"Source: {x['source']}" + ("  " if x["sql"] or x["note"] else ""))
        if x["note"]:
            lines.append(f"Note: {x['note']}" + ("  " if x["sql"] else ""))
        if x["sql"]:
            lines.append(f"SQL: `{x['sql']}`")
        lines.append("")
    (qdir / "ANSWERS.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "README.md").write_text(
        "# Halden Microsystems test corpus\n\nFictional documents for testing the RAG Builder end to end.\n\n"
        "```\nrag init halden-ws\nrag build halden-ws docs\nrag eval halden-ws questions/golden.jsonl\n```\n\n"
        "`questions/ANSWERS.md` has the expected answers. Excluded on purpose: `~$` lock files, "
        "`bin/`, `*.Designer.cs`, and the `Cleanroom Gowning_files/` asset folder.\n", encoding="utf-8")
    print(f"wrote {sum(1 for _ in docs.rglob('*') if _.is_file())} files and {len(qs)} questions to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "halden-corpus"))
