"""Topic-shift splitting for long sections, using the embedding model already loaded.

Each sentence is embedded; cuts go where the cosine distance between
neighbouring sentences falls in the section's top percentile. Only sections
longer than ``long_section_tokens`` with no sub-headings come here.
"""

from __future__ import annotations

import re

import numpy as np

_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n{2,}")


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_RE.split(text)]
    return [p for p in parts if p]


def topic_segments(sentences: list[str], embedder, *, percentile: float, min_sentences: int = 3) -> list[list[str]]:
    if embedder is None or len(sentences) < 2 * min_sentences:
        return [sentences]
    vecs = embedder.embed_documents(sentences)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    dist = 1.0 - sims
    threshold = float(np.percentile(dist, percentile))
    segments: list[list[str]] = []
    current: list[str] = [sentences[0]]
    for i, d in enumerate(dist, start=1):
        if d >= threshold and len(current) >= min_sentences and len(sentences) - i >= min_sentences:
            segments.append(current)
            current = []
        current.append(sentences[i])
    segments.append(current)
    return segments
