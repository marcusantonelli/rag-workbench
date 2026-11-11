"""Figures: save the crop, OCR its text, optionally describe it via a vision endpoint."""

from __future__ import annotations

import base64
import hashlib
import re
import unicodedata
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

from rag.core.ingest.context import ExtractContext
from rag.core.types import Block


def image_fingerprint(img: Image.Image) -> str:
    """Perceptual-ish fingerprint so repeated logos can be detected."""
    small = img.convert("L").resize((16, 16))
    return hashlib.md5(small.tobytes()).hexdigest()


def describe_settings(config) -> dict | None:
    """Where picture descriptions come from, or None when they are off: the workspace's
    [figures] describe_url, else RAG_DESCRIBE_URL (set by the desktop app while its picture
    model runs, or by you for a llama-server of your own)."""
    import os

    cfg = config["figures"]
    url = (cfg.get("describe_url") or os.environ.get("RAG_DESCRIBE_URL") or "").strip().rstrip("/")
    if not url:
        return None
    if "/v1/" not in url and not url.endswith("/completions"):
        url += "/v1/chat/completions"          # a bare server address
    return {"url": url, "model": cfg.get("describe_model") or os.environ.get("RAG_DESCRIBE_MODEL") or "default",
            "timeout": float(cfg.get("describe_timeout_s", 120))}


PROMPT = (
    "This picture is from a technical document such as a work instruction, procedure or "
    "specification{where}. In 1-3 sentences, say what it shows so that someone searching for it "
    "would find it: name the equipment, parts, controls, screens and indicators, and copy any "
    "labels, buttons, field names and values you can read exactly. If it is a screenshot, give "
    "the window or screen title and the settings with their values. If it is a table, transcribe "
    "it as a Markdown table instead. Describe only what is visible; don't guess part numbers or "
    "values you can't read. No preamble."
)


def _blank(img: Image.Image) -> bool:
    """A redaction box, an empty frame or a solid fill: nothing for a model to describe."""
    from PIL import ImageStat

    return max(ImageStat.Stat(img.convert("L").resize((64, 64))).stddev) < 6


def describe_image(img: Image.Image, ctx: ExtractContext, *, context: str | None = None,
                   caption: str | None = None) -> str | None:
    """Ask the vision model (any OpenAI-compatible chat endpoint) to describe a picture. Optional:
    a failure never fails the build, but it is counted, so the document is read again next time."""
    settings = describe_settings(ctx.config)
    if settings is None or _blank(img):
        return None
    stats = ctx.describe_stats
    if stats.get("down"):
        stats["failed"] += 1          # the model stopped answering earlier in this document
        return None
    im = img.convert("RGB")
    if max(im.size) > 1280:           # more pixels cost time and memory, not accuracy
        f = 1280 / max(im.size)
        im = im.resize((max(1, int(im.width * f)), max(1, int(im.height * f))), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    where = ""
    if context:
        where = f", illustrating: {context}"
    elif caption:
        where = f", captioned: {caption}"
    payload = {
        "model": settings["model"],
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": PROMPT.format(where=where)},
            ],
        }],
        "temperature": 0,
        "max_tokens": 320,
    }
    req = urllib.request.Request(settings["url"], data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=settings["timeout"]) as resp:
            data = json.loads(resp.read())
        text = (data["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:  # noqa: BLE001 - descriptions are optional; never fail a build over one
        stats["failed"] += 1
        if isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError, OSError)):
            stats["down"] = True      # don't wait out a timeout for every picture
        return None
    stats["done"] += 1
    ctx.progress("describe", {"done": stats["done"] + stats["failed"]})
    text = re.sub(r"^(this|the) (picture|image|photo|figure) (shows|is|depicts)\s*", "", text, flags=re.I).strip()
    return (text[:1].upper() + text[1:]) if text else None


_WORDLIKE = re.compile(r"[A-Za-z]{3,}")


def readable(text: str, conf: float | None) -> bool:
    """Is OCR text from a photograph worth keeping? Sure reads with real words are; low-confidence
    scraps of a few characters are not."""
    if conf is not None and conf < 0.6:
        return False
    solid = re.sub(r"[^A-Za-z0-9]", "", text)
    return len(solid) >= 6 or (len(solid) >= 2 and (conf or 0) >= 0.85)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())


def confirmed_text(ocr: str, description: str) -> str:
    """The lines of a picture's OCR text that its description confirms. OCR reads glyph by glyph
    and invents words on glare, engraving and small screen text ("Hot Surfcae", "MotorOverlaod"); the
    vision model reads the picture as a whole and copies its labels. A line is kept when most of
    its words are also in the description, so what remains is text both readers saw."""
    said = set(_words(description))
    keep: list[str] = []
    for para in ocr.split("\n\n"):
        for line in para.split("\n"):
            words = [w for w in _words(line) if len(w) >= 2 or w.isdigit()]
            if not words:
                continue
            if sum(1 for w in words if w in said) >= max(1, -(-len(words) * 3 // 5)):
                keep.append(line.strip())
    return "\n\n".join(keep)


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
    description = describe_image(img, ctx, context=context, caption=caption)
    parts = [f"**Figure**{f' (page {page})' if page else ''}" + (f": {caption}" if caption else "")]
    if context:
        parts.append("Shown with: " + context)
    if labels:
        parts.append("Labels: " + "; ".join(labels))
    if ocr and not readable(ocr, conf):
        ocr = ""          # glare, engraving and screen blur read as noise; noise only hurts search
    if ocr and labels and re.sub(r"\W", "", ocr).lower() in re.sub(r"\W", "", " ".join(labels)).lower():
        ocr = ""          # the same words as the labels
    if ocr and description:
        ocr = confirmed_text(ocr, description)
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


def make_diagram_block(img: Image.Image | None, diagram, ctx: ExtractContext, *, page: int | None, index: int,
                       caption: str | None) -> Block:
    """A flowchart rebuilt from the PDF's shapes: numbered steps with where each leads, plus a
    picture of it. The optional vision description is added when one is configured."""
    asset = fingerprint = None
    if img is not None:      # Word's own shapes have no picture to keep
        name = f"p{page}-{index}.png" if page else f"img-{index}.png"
        path: Path = ctx.asset_dir / name
        img.convert("RGB").save(path, format="PNG", optimize=True)
        asset, fingerprint = ctx.asset_rel(path), image_fingerprint(img)
    kind = "Flowchart" if diagram.flowchart else "Diagram"
    title = f"**{kind}**{f' (page {page})' if page else ''}" + (f": {caption}" if caption else "")
    text = diagram.to_text(title)
    description = None      # the steps rebuilt from the drawing are exact; a model's summary adds nothing
    return Block(
        kind="figure",
        text=text,
        page=page,
        meta={"asset_path": asset, "fingerprint": fingerprint, "ocr_conf": None,
              "described": bool(description), "diagram": {"steps": len(diagram.nodes), "connections": len(diagram.edges)}},
    )
