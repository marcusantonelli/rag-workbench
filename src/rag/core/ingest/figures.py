"""Figures: save the crop, OCR its text, optionally describe it via a vision endpoint."""

from __future__ import annotations

import base64
import hashlib
import re
import io
import json
import urllib.request
from pathlib import Path

from PIL import Image

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block


def image_fingerprint(img: Image.Image) -> str:
    """Perceptual-ish fingerprint so repeated logos can be detected."""
    small = img.convert("L").resize((16, 16))
    return hashlib.md5(small.tobytes()).hexdigest()


def describe_image(img: Image.Image, ctx: ExtractContext) -> str | None:
    """Ask an OpenAI-compatible vision endpoint (e.g. the inference server) to describe a figure."""
    cfg = ctx.config["figures"]
    url = cfg.get("describe_url")
    if not url:
        return None
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    payload = {
        "model": cfg.get("describe_model") or "default",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this figure from a technical document in 2-4 "
                                         "sentences. Include any labels, values or part names you can read. "
                                         "If it is a table or spreadsheet, transcribe it as a Markdown table "
                                         "instead. If it is a flowchart, list its steps in order with where "
                                         "each one leads, including branch labels such as Yes and No."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
            ],
        }],
        "temperature": 0,
        "max_tokens": 300,
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=float(cfg.get("describe_timeout_s", 60))) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip() or None
    except Exception:  # descriptions are optional; never fail a build over one
        return None


_WORDLIKE = re.compile(r"[A-Za-z]{3,}")


def readable(text: str, conf: float | None) -> bool:
    """Is OCR text from a photograph worth keeping? Sure reads with real words are; low-confidence
    scraps of a few characters are not."""
    if conf is not None and conf < 0.6:
        return False
    solid = re.sub(r"[^A-Za-z0-9]", "", text)
    return len(solid) >= 6 or (len(solid) >= 2 and (conf or 0) >= 0.85)


def make_figure_block(
    img: Image.Image,
    ctx: ExtractContext,
    *,
    page: int | None,
    index: int,
    caption: str | None,
    context: str | None = None,
    labels: list[str] | None = None,
) -> Block:
    """Save a figure crop and build its block (caption + OCR text + description). A picture of a
    table or spreadsheet comes back as a table block instead, with the picture attached.

    ``context`` says what the picture goes with (the step of a procedure it illustrates) and
    ``labels`` are callouts printed on or beside it; both make a photograph findable, since a
    photo has no text of its own."""
    name = f"p{page}-{index}.png" if page else f"img-{index}.png"
    path: Path = ctx.asset_dir / name
    img.convert("RGB").save(path, format="PNG", optimize=True)
    ocr, conf = "", None
    if ctx.config["pdf"].get("figure_ocr", True):
        from rag.core.ingest.ocr import ocr_figure

        found = ocr_figure(img, tables=bool(ctx.config["figures"].get("table_ocr", True)))
        ocr, conf = found.text, found.conf
        if found.table is not None:
            return _table_block(found.table, ctx, path, img, page=page, caption=caption)
    description = describe_image(img, ctx)
    parts = [f"**Figure**{f' (page {page})' if page else ''}" + (f": {caption}" if caption else "")]
    if context:
        parts.append("Shown with: " + context)
    if labels:
        parts.append("Labels: " + "; ".join(labels))
    if ocr and not readable(ocr, conf):
        ocr = ""          # glare, engraving and screen blur read as noise; noise only hurts search
    if ocr and labels and re.sub(r"\W", "", ocr).lower() in re.sub(r"\W", "", " ".join(labels)).lower():
        ocr = ""          # the same words as the labels
    if ocr:
        parts.append("Text in figure: " + ocr.replace("\n\n", " / "))
    if description:
        parts.append("Description: " + description)
    return Block(
        kind="figure",
        text="\n\n".join(parts),
        page=page,
        meta={
            "asset_path": ctx.asset_rel(path),
            "fingerprint": image_fingerprint(img),
            "ocr_conf": conf,
            "described": bool(description),
            # nothing but the picture: no caption, labels, readable text or description
            "bare": not (caption or labels or ocr or description),
        },
    )


def _table_block(table, ctx: ExtractContext, path: Path, img: Image.Image, *, page: int | None,
                 caption: str | None) -> Block:
    """A picture of a table, as a table block: Markdown text, headers and rows for tables.db,
    and the picture attached so the values can be checked against it."""
    from rag.core.util import markdown_table

    name = caption or table.title or (f"Table in a picture on page {page}" if page else "Table in a picture")
    if caption and table.title and table.title not in caption:
        name = f"{caption}: {table.title}"
    label = f"{name} (read from the picture with OCR)"
    return Block(
        kind="table",
        text=f"**{label}**\n\n" + markdown_table(table.headers, table.rows),
        page=page,
        meta={"headers": table.headers, "rows": table.rows, "caption": label, "from_image": True,
              "asset_path": ctx.asset_rel(path), "fingerprint": image_fingerprint(img),
              "ocr_conf": round(table.conf, 3)},
    )


def make_diagram_block(img: Image.Image, diagram, ctx: ExtractContext, *, page: int | None, index: int,
                       caption: str | None) -> Block:
    """A flowchart rebuilt from the PDF's shapes: numbered steps with where each leads, plus a
    picture of it. The optional vision description is added when one is configured."""
    name = f"p{page}-{index}.png" if page else f"img-{index}.png"
    path: Path = ctx.asset_dir / name
    img.convert("RGB").save(path, format="PNG", optimize=True)
    kind = "Flowchart" if diagram.flowchart else "Diagram"
    title = f"**{kind}**{f' (page {page})' if page else ''}" + (f": {caption}" if caption else "")
    text = diagram.to_text(title)
    description = describe_image(img, ctx)
    if description:
        text += "\n\nDescription: " + description
    return Block(
        kind="figure",
        text=text,
        page=page,
        meta={"asset_path": ctx.asset_rel(path), "fingerprint": image_fingerprint(img), "ocr_conf": None,
              "described": bool(description), "diagram": {"steps": len(diagram.nodes), "connections": len(diagram.edges)}},
    )
