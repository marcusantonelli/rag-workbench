"""PDFs and DOCX files with drawings: flowcharts as vector shapes (as Visio, PowerPoint and
draw.io export them), look-alikes that are not flowcharts, and pictures of spreadsheets."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


# -- pictures ----------------------------------------------------------------------------------
SHEET = [["Lot", "Date", "Units in", "Units out", "Yield %", "Status"],
         ["L24-0811", "2026-03-02", "1,200", "1,164", "97.0", "Released"],
         ["L24-0812", "2026-03-02", "1,200", "1,098", "91.5", "MRB"],
         ["L24-0815", "2026-03-04", "800", "792", "99.0", "Released"],
         ["L24-0819", "2026-03-05", "1,000", "0", "0.0", "Scrapped"],
         ["L24-0821", "2026-03-07", "1,200", "1,188", "99.0", "Released"],
         ["L24-0830", "2026-03-09", "600", "571", "95.2", "On hold"],
         ["Total", "", "6,000", "4,813", "80.2", ""]]


def long_sheet(n: int = 40) -> list[list[str]]:
    rows = [["Lot", "Oven", "Hours", "Fails"]]
    for i in range(n):
        rows.append([f"B26-{1000 + i}", f"OV-{1 + i % 4}", str(96 + 24 * (i % 3)), str((i * 7) % 5)])
    return rows


def sheet_image(path: Path, rows: list[list[str]] = SHEET, *, excel: bool = False, scale: float = 1.0) -> Path:
    """A spreadsheet screenshot: header fill, light grid, numbers right-aligned; optionally
    Excel's column letters and row numbers around it."""
    f = _font(int(15 * scale))
    ncol = len(rows[0])
    widths = [int(110 * scale)] * ncol
    rh = int(26 * scale)
    left, top = (int(36 * scale), rh) if excel else (0, 0)
    width, height = left + sum(widths) + 1, top + rh * len(rows) + 1
    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)
    if excel:
        d.rectangle([0, 0, width, top], fill=(230, 230, 230))
        d.rectangle([0, 0, left, height], fill=(230, 230, 230))
        x = left
        for i, w in enumerate(widths):
            d.text((x + w // 2 - 5, 5 * scale), "ABCDEFGHIJ"[i], font=f, fill=(80, 80, 80))
            x += w
        for r in range(len(rows)):
            d.text((8 * scale, top + r * rh + 5 * scale), str(r + 1), font=f, fill=(80, 80, 80))
    d.rectangle([left, top, width, top + rh], fill=(214, 228, 245))
    for r, row in enumerate(rows):
        x = left
        for c, val in enumerate(row):
            numeric = r > 0 and val.replace(",", "").replace(".", "").isdigit()
            tw = d.textlength(val, font=f)
            d.text((x + widths[c] - tw - 6 * scale if numeric else x + 6 * scale, top + r * rh + 5 * scale), val,
                   font=f, fill="black")
            x += widths[c]
    for r in range(len(rows) + 1):
        d.line([left, top + r * rh, width, top + r * rh], fill=(200, 200, 200))
    x = left
    for w in widths + [0]:
        d.line([x, top, x, height], fill=(200, 200, 200))
        x += w
    img.save(path)
    return path


def prose_image(path: Path) -> Path:
    """A screenshot of a paragraph: text, not a table."""
    lines = ["Seal the lid within four hours of die attach. If the window is",
             "missed, bake the parts at 150 C for 24 hours before sealing and",
             "record the bake on the traveler. Parts that wait longer than",
             "seven days are scrapped and the lot is reviewed by the MRB."]
    f = _font(18)
    img = Image.new("RGB", (640, 40 + 30 * len(lines)), "white")
    d = ImageDraw.Draw(img)
    for i, t in enumerate(lines):
        d.text((16, 16 + 30 * i), t, font=f, fill="black")
    img.save(path)
    return path


# -- vector drawings -----------------------------------------------------------------------------
def _canvas(path: Path):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    return canvas.Canvas(str(path), pagesize=letter), letter[1]


def _arrow(c, H, pts, head="filled", pieces=False):
    """A connector through pts (top-down coordinates). pieces: one line per segment, as Visio
    exports elbows. head: filled triangle, open V (two strokes), "both" or None."""
    if pieces:
        for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
            c.line(x1, H - y1, x2, H - y2)
    else:
        p = c.beginPath()
        p.moveTo(pts[0][0], H - pts[0][1])
        for x, y in pts[1:]:
            p.lineTo(x, H - y)
        c.drawPath(p, stroke=1, fill=0)

    def tip(a_pt, b_pt, style):
        (x1, y1), (x2, y2) = a_pt, b_pt
        a = math.atan2(y2 - y1, x2 - x1)
        L, w = 8, 4
        bx, by = x2 - L * math.cos(a), y2 - L * math.sin(a)
        p1 = (bx + w * math.sin(a), by - w * math.cos(a))
        p2 = (bx - w * math.sin(a), by + w * math.cos(a))
        if style == "open":
            c.line(p1[0], H - p1[1], x2, H - y2)
            c.line(p2[0], H - p2[1], x2, H - y2)
        else:
            q = c.beginPath()
            q.moveTo(x2, H - y2)
            q.lineTo(p1[0], H - p1[1])
            q.lineTo(p2[0], H - p2[1])
            q.close()
            c.drawPath(q, stroke=0, fill=1)

    if head in ("filled", "open"):
        tip(pts[-2], pts[-1], head)
    elif head == "both":
        tip(pts[-2], pts[-1], "filled")
        tip(pts[1], pts[0], "filled")


def _shape(c, H, x, y, w, h, lines, kind="rect", size=9):
    if kind == "rect":
        c.rect(x, H - y - h, w, h)
    elif kind == "round":
        c.roundRect(x, H - y - h, w, h, 12)
    elif kind == "diamond":
        p = c.beginPath()
        p.moveTo(x + w / 2, H - y)
        p.lineTo(x + w, H - y - h / 2)
        p.lineTo(x + w / 2, H - y - h)
        p.lineTo(x, H - y - h / 2)
        p.close()
        c.drawPath(p, stroke=1, fill=0)
    elif kind == "filled":
        c.setFillColorRGB(0.85, 0.92, 1.0)
        c.rect(x, H - y - h, w, h, stroke=1, fill=1)
        c.setFillColorRGB(0, 0, 0)
    c.setFont("Helvetica", size)
    n = len(lines)
    for i, t in enumerate(lines):
        c.drawCentredString(x + w / 2, H - (y + h / 2 + (i - (n - 1) / 2) * (size + 2)) - 3, t)


def _text(c, H, x, y, s, size=10, font="Helvetica"):
    c.setFont(font, size)
    c.drawString(x, H - y, s)


def flowchart_pdf(path: Path) -> Path:
    """Page 1: the lot-disposition flowchart (filled arrowheads, an elbow polyline, Yes/No).
    Page 2: a Visio-style one: page frame with body text, swimlanes, connectors in pieces,
    open arrowheads, a branch, a two-word label, a plain link and a two-way arrow."""
    c, H = _canvas(path)
    _text(c, H, 72, 72, "3.2 Lot disposition", 14, "Helvetica-Bold")
    _text(c, H, 72, 95, "Every lot that fails fine leak after burn-in follows the process below before")
    _text(c, H, 72, 108, "it can be released or scrapped. The MRB decides within two shifts.")
    cx, top = 306, 150
    _shape(c, H, cx - 60, top, 120, 30, ["Start: lot fails", "fine leak"], "round")
    _shape(c, H, cx - 70, top + 60, 140, 34, ["Place lot on hold", "(mes-cli holds --reason SealLeak)"])
    _shape(c, H, cx - 60, top + 125, 120, 60, ["Leak rate", "< 5E-8 atm·cc/s?"], "diamond")
    _shape(c, H, cx + 120, top + 136, 120, 38, ["Retest with", "helium bomb"])
    _shape(c, H, cx - 240, top + 136, 120, 38, ["Submit to MRB", "for review"])
    _shape(c, H, cx - 240, top + 230, 120, 34, ["Scrap lot"])
    _shape(c, H, cx + 120, top + 230, 120, 34, ["Release lot", "to stock"])
    _shape(c, H, cx - 60, top + 320, 120, 30, ["End"], "round")
    _arrow(c, H, [(cx, top + 30), (cx, top + 60)])
    _arrow(c, H, [(cx, top + 94), (cx, top + 125)])
    _arrow(c, H, [(cx + 60, top + 155), (cx + 120, top + 155)])
    _arrow(c, H, [(cx - 60, top + 155), (cx - 120, top + 155)])
    _arrow(c, H, [(cx - 180, top + 174), (cx - 180, top + 230)])
    _arrow(c, H, [(cx + 180, top + 174), (cx + 180, top + 230)])
    _arrow(c, H, [(cx - 180, top + 264), (cx - 180, top + 335), (cx - 60, top + 335)])
    _arrow(c, H, [(cx + 180, top + 264), (cx + 180, top + 335), (cx + 60, top + 335)])
    _text(c, H, cx + 75, top + 150, "Yes", 8)
    _text(c, H, cx - 100, top + 150, "No", 8)
    c.setFont("Helvetica-Oblique", 9)
    c.drawCentredString(cx, H - (top + 375), "Figure 4. Lot disposition after a fine leak failure")
    _text(c, H, 72, top + 410, "Lots released after a retest keep their original date code.")
    c.showPage()

    # page 2
    c.rect(36, 36, 612 - 72, H - 72)                                    # page frame
    _text(c, H, 60, 70, "3.3 Gross leak check", 14, "Helvetica-Bold")
    _text(c, H, 60, 92, "Operators run the gross leak check on a sample from every sealed lot; QA owns")
    _text(c, H, 60, 105, "the follow-up when bubbles are seen.")
    lane_top = 130
    c.rect(60, H - (lane_top + 330), 300, 330)                          # lane: Operator
    c.rect(360, H - (lane_top + 330), 190, 330)                         # lane: QA Engineer
    _text(c, H, 66, lane_top + 14, "Operator", 9, "Helvetica-Bold")
    _text(c, H, 366, lane_top + 14, "QA Engineer", 9, "Helvetica-Bold")
    _shape(c, H, 140, lane_top + 30, 120, 30, ["Pull 5-unit sample"], "filled")
    _shape(c, H, 140, lane_top + 90, 120, 30, ["Run gross leak test"], "filled")
    _shape(c, H, 140, lane_top + 150, 120, 56, ["Bubbles", "seen?"], "diamond")
    _shape(c, H, 140, lane_top + 250, 120, 30, ["Continue to marking"], "filled")
    _shape(c, H, 395, lane_top + 160, 120, 34, ["Log failure", "in MES"], "filled")
    _shape(c, H, 395, lane_top + 235, 120, 30, ["Notify QA engineer"], "filled")
    _shape(c, H, 395, lane_top + 290, 120, 30, ["Place lot on hold"], "filled")
    _shape(c, H, 395, lane_top + 40, 120, 30, ["QA on-call phone"], "filled")
    _shape(c, H, 230, lane_top + 300, 110, 22, ["See WI-0042"])
    _arrow(c, H, [(200, lane_top + 60), (200, lane_top + 90)], "open", pieces=True)
    _arrow(c, H, [(200, lane_top + 120), (200, lane_top + 150)], "open", pieces=True)
    _arrow(c, H, [(200, lane_top + 206), (200, lane_top + 250)], "open", pieces=True)       # no -> continue
    _arrow(c, H, [(260, lane_top + 178), (330, lane_top + 178), (330, lane_top + 177), (395, lane_top + 177)],
           "open", pieces=True)                                                               # yes -> log
    # log -> branch to notify and hold (a T: one stem, two arms)
    c.line(455, H - (lane_top + 194), 455, H - (lane_top + 215))
    c.line(375, H - (lane_top + 215), 455, H - (lane_top + 215))
    c.line(375, H - (lane_top + 215), 375, H - (lane_top + 305))
    _arrow(c, H, [(375, lane_top + 305), (395, lane_top + 305)], "filled", pieces=True)
    _arrow(c, H, [(455, lane_top + 215), (455, lane_top + 235)], "filled", pieces=True)
    _arrow(c, H, [(455, lane_top + 70), (455, lane_top + 160)], "both")                       # phone <-> log
    c.setDash(3, 2)
    c.line(285, H - (lane_top + 300), 285, H - (lane_top + 280))                               # note link (no head)
    c.line(285, H - (lane_top + 280), 395, H - (lane_top + 250))
    c.setDash()
    _text(c, H, 272, lane_top + 172, "Not OK", 8)
    _text(c, H, 205, lane_top + 230, "OK", 8)
    _text(c, H, 60, lane_top + 360, "Figure 5. Gross leak check", 9, "Helvetica-Oblique")
    _text(c, H, 60, lane_top + 390, "Record every check on the lot traveler, including passes.")
    c.showPage()
    c.save()
    return path


def lookalikes_pdf(path: Path) -> Path:
    """Drawings that are not flowcharts: a table made of touching boxes with rules, a bar chart,
    and three text boxes with no connectors."""
    c, H = _canvas(path)
    _text(c, H, 72, 72, "Burn-in ovens", 14, "Helvetica-Bold")
    cols, rows = [72, 172, 272, 372], [100, 124, 148, 172]
    data = [["Oven", "Max temp", "Slots"], ["OV-1", "150 C", "48"], ["OV-2", "150 C", "48"], ["OV-3", "175 C", "32"]]
    for r, row in enumerate(data):
        for k, val in enumerate(row):
            c.rect(cols[k], H - rows[r] - 24, 100, 24)
            _text(c, H, cols[k] + 6, rows[r] + 16, val, 9)
    # bar chart
    _text(c, H, 72, 250, "Fails by week", 11, "Helvetica-Bold")
    c.line(90, H - 400, 400, H - 400)
    c.line(90, H - 400, 90, H - 270)
    for i, v in enumerate([40, 75, 30, 110, 55]):
        c.rect(110 + i * 55, H - 400, 30, v, stroke=0, fill=1)
        _text(c, H, 112 + i * 55, 415, f"W{i + 10}", 8)
    for v in (0, 50, 100):
        _text(c, H, 70, 403 - v, str(v), 8)
    # boxes, no connectors
    for i, t in enumerate(["Safety first", "Quality always", "Ship on time"]):
        _shape(c, H, 72 + i * 160, 470, 140, 40, [t])
    c.showPage()
    c.save()
    return path


def picture_pdf(path: Path, pictures: dict[str, Path]) -> Path:
    """Page 1: a spreadsheet picture with a caption below it, then a prose screenshot.
    Page 2: an Excel screenshot (row numbers and column letters) of a 40-row sheet."""
    c, H = _canvas(path)
    _text(c, H, 72, 72, "4 Burn-in results", 14, "Helvetica-Bold")
    _text(c, H, 72, 95, "The March lots are summarised below.")
    c.drawImage(str(pictures["sheet"]), 72, H - 110 - 160, width=440, height=160)
    _text(c, H, 72, 290, "Table 2. Burn-in yields by lot", 9, "Helvetica-Oblique")
    c.drawImage(str(pictures["prose"]), 72, H - 320 - 100, width=440, height=100)
    _text(c, H, 72, 440, "Lot L24-0819 was scrapped after the oven alarm.")
    c.showPage()
    c.drawImage(str(pictures["long"]), 72, 72, width=420, height=H - 160)
    c.showPage()
    c.save()
    return path


def picture_docx(path: Path, picture: Path) -> Path:
    from docx import Document
    from docx.shared import Inches

    d = Document()
    d.add_heading("Weekly yield", 1)
    d.add_paragraph("Yields for the March lots:")
    d.add_picture(str(picture), width=Inches(6))
    d.add_paragraph("Questions go to the yield engineer.")
    d.save(path)
    return path


def make_drawings(root: Path) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    pics = {"sheet": sheet_image(root / "sheet.png", scale=1.5), "prose": prose_image(root / "prose.png"),
            "long": sheet_image(root / "long.png", long_sheet(), excel=True, scale=1.5)}
    return {
        "flowchart": flowchart_pdf(root / "Leak Flowcharts.pdf"),
        "lookalikes": lookalikes_pdf(root / "Not flowcharts.pdf"),
        "pictures": picture_pdf(root / "Yield pictures.pdf", pics),
        "docx": picture_docx(root / "Weekly yield.docx", sheet_image(root / "sheet_small.png")),
    }
