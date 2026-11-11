"""Forms: filled-in PDF fields, label/value grids, and checkbox symbols.

* A fillable PDF keeps what was typed in its fields (AcroForm widgets), not in the page text, so
  a plain text read loses every value. ``inject_pdf_fields`` adds each field's value to the page
  as characters inside the field's box, so the rest of the reader (lines, tables, procedure
  tables) sees it exactly as if it were printed there: "Lot number: L26-3007", "☒ Seal ring
  inspected", "○ Release ◉ MRB".
* Checkbox and tick symbols set in Wingdings come out of a PDF or a Word file as stray letters
  ("þ", "o", "x") or private-use characters; ``checkbox_symbol`` turns them into ☐ ☑ ☒ ✓ ✗.
* A form laid out as a grid ("Lot number:" | "L26-3007" | "Date:" | "2026-11-03") is not a data
  table: its first row is not column names. ``form_lines`` turns such a grid into
  "Label: value" lines.
"""

from __future__ import annotations

import re
from typing import Any

CHECKED, UNCHECKED = "☒", "☐"
RADIO_ON, RADIO_OFF = "◉", "○"

# Wingdings and Wingdings 2 code points (as the plain character, or in the private-use area at
# U+F000 + code) for boxes and ticks.
_WINGDINGS = {0x6F: "☐", 0x71: "☐", 0xA8: "☐", 0xFE: "☑", 0xFD: "☒", 0x78: "☒", 0xFC: "✓", 0xFB: "✗"}
_WINGDINGS2 = {0xA3: "☐", 0x52: "☑", 0x53: "☒", 0x54: "☒", 0x50: "✓", 0x4F: "✗"}


def checkbox_symbol(ch: str, font: str | None) -> str | None:
    """The box or tick a Wingdings character stands for, or None if it isn't one."""
    if not ch or len(ch) != 1 or not font:
        return None
    f = font.lower()
    table = _WINGDINGS2 if "wingdings2" in f.replace(" ", "").replace("-", "") else _WINGDINGS if "wingdings" in f else None
    if table is None:
        return None
    code = ord(ch)
    if 0xF000 <= code <= 0xF0FF:
        code -= 0xF000
    return table.get(code)


# -- fillable PDF fields -------------------------------------------------------------------------
def _text(value: Any) -> str:
    """A PDF string or name as text."""
    from pdfminer.psparser import PSLiteral
    from pdfminer.utils import decode_text

    if value is None:
        return ""
    if isinstance(value, PSLiteral):
        name = value.name
        return name.decode("latin-1") if isinstance(name, bytes) else str(name)
    if isinstance(value, bytes):
        return decode_text(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(t for t in (_text(v) for v in value) if t)
    return str(value)


def _field(data: dict) -> tuple[str, str, str, int]:
    """(type, name, value, flags) of a widget, looking up the parent field for what it doesn't hold."""
    from pdfminer.pdftypes import resolve1

    ft, name, value, flags = data.get("FT"), data.get("T"), data.get("V"), data.get("Ff")
    parent = resolve1(data.get("Parent"))
    depth = 0
    while isinstance(parent, dict) and depth < 8:
        ft = ft if ft is not None else parent.get("FT")
        name = name if name is not None else parent.get("T")
        value = value if value is not None else parent.get("V")
        flags = flags if flags is not None else parent.get("Ff")
        parent = resolve1(parent.get("Parent"))
        depth += 1
    try:
        flags = int(resolve1(flags) or 0)
    except (TypeError, ValueError):
        flags = 0
    return _text(resolve1(ft)), _text(resolve1(name)), _text(resolve1(value)), flags


def pdf_fields(page) -> list[dict[str, Any]]:
    """The filled-in form fields on a pdfplumber page: [{bbox, kind, name, text}], where text is
    what to show in the field's place ("" for an empty text field)."""
    from pdfminer.pdftypes import resolve1

    out: list[dict[str, Any]] = []
    try:
        annots = page.annots
    except Exception:  # noqa: BLE001 - a damaged annotation list costs the fields, not the page
        return out
    for a in annots:
        data = a.get("data") or {}
        if _text(resolve1(data.get("Subtype"))) != "Widget":
            continue
        ft, name, value, flags = _field(data)
        bbox = (float(a["x0"]), float(a["top"]), float(a["x1"]), float(a["bottom"]))
        if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2:
            continue
        if ft == "Btn":
            if flags & (1 << 16):                     # a push button: nothing to read
                continue
            state = _text(resolve1(data.get("AS"))) or value
            on = bool(state) and state != "Off"
            radio = bool(flags & (1 << 15))
            text = (RADIO_ON if on else RADIO_OFF) if radio else (CHECKED if on else UNCHECKED)
            out.append({"bbox": bbox, "kind": "radio" if radio else "checkbox", "name": name, "text": text})
        elif ft in ("Tx", "Ch"):
            text = re.sub(r"\s+", " ", value).strip()
            if ft == "Tx" and flags & (1 << 13):      # a password field
                text = ""
            out.append({"bbox": bbox, "kind": "text" if ft == "Tx" else "choice", "name": name, "text": text})
    return out


def inject_pdf_fields(page) -> int:
    """Add each filled-in field's value to the page as characters in the field's box (skipping
    fields whose box already holds printed text, as in a flattened copy). Returns how many."""
    fields = [f for f in pdf_fields(page) if f["text"]]
    if not fields:
        return 0
    chars = page.objects.setdefault("char", [])
    existing = list(chars)
    height = float(page.height)
    added = 0
    for f in fields:
        x0, top, x1, bottom = f["bbox"]
        inside = [c for c in existing if x0 - 1 <= (c["x0"] + c["x1"]) / 2 <= x1 + 1 and top - 1 <= (c["top"] + c["bottom"]) / 2 <= bottom + 1]
        if inside:
            continue
        # the size of nearby printed text on the same line, so the value joins that line
        near = [c for c in existing if abs((c["top"] + c["bottom"]) / 2 - (top + bottom) / 2) <= (bottom - top)]
        size = float(near[0]["size"]) if near else max(6.0, min(10.0, (bottom - top) * 0.7))
        mid = (top + bottom) / 2
        ctop, cbottom = mid - size / 2, mid + size / 2
        x = x0 + (0.0 if f["kind"] in ("checkbox", "radio") else 2.0)
        for ch in f["text"]:
            w = size * (0.55 if ch != " " else 0.28)
            chars.append({
                "matrix": (1.0, 0.0, 0.0, 1.0, x, height - cbottom), "fontname": "FormField", "adv": w,
                "upright": True, "x0": x, "y0": height - cbottom, "x1": x + w, "y1": height - ctop,
                "width": w, "height": size, "size": size, "mcid": None, "tag": None, "object_type": "char",
                "page_number": page.page_number, "ncs": "DeviceGray", "text": ch, "stroking_color": (0,),
                "non_stroking_color": (0,), "top": ctop, "bottom": cbottom,
                "doctop": ctop + float(getattr(page, "initial_doctop", 0) or 0),
            })
            x += w
        added += 1
    if added:
        chars.sort(key=lambda c: (round(c["doctop"], 1), c["x0"]))
    return added


def normalize_symbol_chars(page) -> None:
    """Turn Wingdings boxes and ticks on a pdfplumber page into ☐ ☑ ☒ ✓ ✗ in place."""
    for c in page.objects.get("char", []):
        sym = checkbox_symbol(c.get("text", ""), c.get("fontname"))
        if sym:
            c["text"] = sym


# -- label/value grids ---------------------------------------------------------------------------
_LABEL = re.compile(r"^(?P<label>[^:\n]{1,48}?)\s*:\s*(?P<value>.*)$", re.S)
_TIME = re.compile(r"^\d{1,2}:\d{2}")


def _label(cell: str) -> tuple[str, str] | None:
    """("Lot number", "L26-3007") for "Lot number: L26-3007", ("Date", "") for "Date:"."""
    if _TIME.match(cell):
        return None
    m = _LABEL.match(cell.strip())
    if not m or not re.search(r"[A-Za-z]", m.group("label")):
        return None
    return m.group("label").strip(), m.group("value").strip()


def form_lines(rows: list[list[str]]) -> list[str] | None:
    """A grid of "Label:" cells with values beside them (or inside them) as "Label: value"
    lines, row by row; None when the grid isn't a form (most cells must be labelled)."""
    cells = [c.strip() for r in rows for c in r if c and c.strip()]
    if len(cells) < 3:
        return None
    labelled = [c for c in cells if _label(c)]
    if len(labelled) < 3 or len(labelled) < 0.4 * len(cells):
        return None
    # a data table with one "Notes: ..." column isn't a form: the labels must be spread out
    if len(rows) > 3 and sum(1 for r in rows if any(_label(c or "") for c in r)) < 0.6 * len(rows):
        return None
    lines: list[str] = []
    for row in rows:
        row = [(c or "").strip() for c in row]
        # merged cells repeat their text across the columns they span
        dedup = [c for k, c in enumerate(row) if c and (k == 0 or c != row[k - 1])]
        k = 0
        while k < len(dedup):
            cell = dedup[k]
            found = _label(cell)
            if found and not found[1] and k + 1 < len(dedup) and not _label(dedup[k + 1]):
                lines.append(f"{found[0]}: {dedup[k + 1]}")
                k += 2
                continue
            if found:
                lines.append(f"{found[0]}: {found[1]}".rstrip())
            else:
                lines.append(cell)
            k += 1
    return lines
