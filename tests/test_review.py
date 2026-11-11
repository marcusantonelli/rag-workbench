"""Chunks to review: a fixed threshold until the knowledge base has enough OCR'd chunks, then
its own spread."""

from __future__ import annotations

import sqlite3

from rag.core.review import MIN_SAMPLE, review


def _db(confs: list[float]) -> sqlite3.Connection:
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE documents (doc_id INTEGER PRIMARY KEY, display_path TEXT)")
    c.execute("CREATE TABLE chunks (chunk_id INTEGER PRIMARY KEY, doc_id INT, kind TEXT, text TEXT, ocr_conf REAL, "
              "page_start INT, heading_path TEXT)")
    c.execute("INSERT INTO documents VALUES (1, 'scan.pdf')")
    for i, v in enumerate(confs):
        c.execute("INSERT INTO chunks VALUES (?, 1, 'prose', ?, ?, 1, NULL)", (i + 1, f"text {i}", v))
    # a photo with no text in it isn't something to review, whatever its confidence
    c.execute("INSERT INTO chunks VALUES (999, 1, 'figure', '**Figure** Description: a pump', 0.0, 2, NULL)")
    return c


def test_threshold_while_the_sample_is_small():
    r = review(_db([0.95, 0.9, 0.7, 0.4]))
    assert r["method"] == "threshold" and r["count"] == 2
    assert [c["ocr_conf"] for c in r["chunks"]] == [0.4, 0.7]


def test_spread_once_there_is_enough():
    confs = [0.95] * (MIN_SAMPLE - 2) + [0.93, 0.80]
    r = review(_db(confs))
    assert r["method"] == "spread" and r["threshold"] > 0.8
    assert [c["ocr_conf"] for c in r["chunks"]] == [0.8]       # 0.80 is unusual here, though above 0.75
