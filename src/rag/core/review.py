"""Chunks worth a person's look: text read with OCR at low confidence (scanned pages, tables read
from pictures, text in figures).

Until there are enough OCR'd chunks to judge what is normal for this knowledge base (fewer than
``MIN_SAMPLE``), a fixed threshold decides. After that a chunk is flagged when its confidence is
unusually low for this knowledge base, more than ``spread`` standard deviations below the mean,
and always when it is below the floor. 0.4.3 only lists them; editing a chunk is planned for 0.5.
"""

from __future__ import annotations

import statistics
from typing import Any

MIN_SAMPLE = 30          # OCR'd chunks needed before the knowledge base's own spread is used
THRESHOLD = 0.75         # until then: below this is flagged
FLOOR = 0.5              # always flagged
SPREAD = 2.0             # standard deviations below the mean


def review(conn, *, limit: int = 200, spread: float = SPREAD) -> dict[str, Any]:
    """{"method", "threshold", "mean", "std", "sample", "count", "chunks": [...]} for a rag.db."""
    rows = conn.execute(
        "SELECT c.chunk_id, c.kind, c.text, c.ocr_conf, c.page_start, c.heading_path, d.display_path, d.doc_id "
        "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
        "WHERE c.ocr_conf IS NOT NULL AND (c.kind != 'figure' OR c.text LIKE '%Text in figure:%')").fetchall()
    confs = [float(r[3]) for r in rows]
    out: dict[str, Any] = {"sample": len(confs), "mean": None, "std": None}
    if len(confs) >= MIN_SAMPLE:
        mean, std = statistics.fmean(confs), statistics.pstdev(confs)
        cut = max(FLOOR, mean - spread * std)
        out.update(method="spread", mean=round(mean, 3), std=round(std, 3), threshold=round(cut, 3))
    else:
        cut = THRESHOLD
        if confs:
            out.update(mean=round(statistics.fmean(confs), 3))
        out.update(method="threshold", threshold=cut)
    flagged = sorted((r for r in rows if float(r[3]) < cut), key=lambda r: float(r[3]))
    out["count"] = len(flagged)
    out["chunks"] = [{"chunk_id": r[0], "kind": r[1], "ocr_conf": round(float(r[3]), 3), "page": r[4],
                      "heading": r[5], "file": r[6], "doc_id": r[7],
                      "text": r[2] if len(r[2]) <= 600 else r[2][:600] + " …"} for r in flagged[:limit]]
    return out
