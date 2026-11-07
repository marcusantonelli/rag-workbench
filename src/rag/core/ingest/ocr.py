"""OCR with RapidOCR (PP-OCR models on ONNX Runtime; models ship in the wheel)."""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass

import numpy as np
from PIL import Image

_engine = None
_lock = threading.Lock()


@dataclass
class OcrLine:
    text: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def height(self) -> float:
        return self.y1 - self.y0


@dataclass
class OcrParagraph:
    text: str
    conf: float
    y0: float
    is_heading: bool = False


def _get_engine():
    global _engine
    with _lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        return _engine


def ocr_lines(image: Image.Image) -> list[OcrLine]:
    """Recognize text and merge fragments that sit on the same visual line."""
    arr = np.array(image.convert("RGB"))
    result, _ = _get_engine()(arr)
    if not result:
        return []
    frags: list[OcrLine] = []
    for box, text, score in result:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        frags.append(OcrLine(str(text), float(score), min(xs), min(ys), max(xs), max(ys)))
    frags.sort(key=lambda f: ((f.y0 + f.y1) / 2, f.x0))
    lines: list[list[OcrLine]] = []
    for frag in frags:
        mid = (frag.y0 + frag.y1) / 2
        if lines:
            last = lines[-1]
            ly0 = min(f.y0 for f in last)
            ly1 = max(f.y1 for f in last)
            if ly0 <= mid <= ly1:
                last.append(frag)
                continue
        lines.append([frag])
    merged: list[OcrLine] = []
    for group in lines:
        group.sort(key=lambda f: f.x0)
        merged.append(
            OcrLine(
                text=" ".join(f.text for f in group),
                conf=sum(f.conf * len(f.text) for f in group) / max(1, sum(len(f.text) for f in group)),
                x0=min(f.x0 for f in group),
                y0=min(f.y0 for f in group),
                x1=max(f.x1 for f in group),
                y1=max(f.y1 for f in group),
            )
        )
    return merged


def lines_to_paragraphs(lines: list[OcrLine]) -> list[OcrParagraph]:
    """Group lines into paragraphs by vertical gaps; flag tall short lines as headings."""
    if not lines:
        return []
    heights = [ln.height for ln in lines if ln.height > 0]
    median_h = statistics.median(heights) if heights else 10.0
    paras: list[OcrParagraph] = []
    buf: list[OcrLine] = []

    def flush() -> None:
        if buf:
            text = ""
            for ln in buf:
                if text.endswith("-") and ln.text[:1].islower():
                    text = text[:-1] + ln.text
                else:
                    text = f"{text} {ln.text}".strip()
            conf = sum(ln.conf for ln in buf) / len(buf)
            paras.append(OcrParagraph(text=text, conf=conf, y0=buf[0].y0))
            buf.clear()

    prev: OcrLine | None = None
    for ln in lines:
        heading = ln.height >= 1.4 * median_h and len(ln.text) <= 100 and not ln.text.endswith(".")
        if heading:
            flush()
            paras.append(OcrParagraph(text=ln.text, conf=ln.conf, y0=ln.y0, is_heading=True))
            prev = ln
            continue
        if prev is not None and buf:
            gap = ln.y0 - prev.y1
            if gap > 0.9 * median_h or ln.text[:1] in "•▪-*" :
                flush()
        buf.append(ln)
        prev = ln
    flush()
    return paras


def ocr_text(image: Image.Image) -> tuple[str, float]:
    lines = ocr_lines(image)
    if not lines:
        return "", 0.0
    paras = lines_to_paragraphs(lines)
    text = "\n\n".join(p.text for p in paras)
    conf = sum(ln.conf for ln in lines) / len(lines)
    return text, conf
