"""Figures: save the crop, OCR its text, optionally describe it via a vision endpoint."""

from __future__ import annotations

import base64
import hashlib
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
                                         "sentences. Include any labels, values or part names you can read."},
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


def make_figure_block(
    img: Image.Image,
    ctx: ExtractContext,
    *,
    page: int | None,
    index: int,
    caption: str | None,
) -> Block:
    """Save a figure crop and build its block (caption + OCR text + description)."""
    name = f"p{page}-{index}.png" if page else f"img-{index}.png"
    path: Path = ctx.asset_dir / name
    img.convert("RGB").save(path, format="PNG", optimize=True)
    ocr, conf = "", None
    if ctx.config["pdf"].get("figure_ocr", True):
        from rag.core.ingest.ocr import ocr_text

        ocr, conf = ocr_text(img)
    description = describe_image(img, ctx)
    parts = [f"**Figure**{f' (page {page})' if page else ''}" + (f": {caption}" if caption else "")]
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
        },
    )
