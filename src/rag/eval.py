"""Retrieval evaluation against a golden set labelled by location, not chunk id.

golden.jsonl, one question per line::

    {"id": "pull-1", "question": "What is the minimum pull for 5 mil aluminum wire?",
     "type": "doc_table", "expect": [{"file": "*QP-114*", "page": 1}]}

An expectation matches a retrieved chunk when every field it gives matches:
``file`` (glob on the display path, case-insensitive), ``page`` (inside the
chunk's page range), ``sheet``, ``line`` (inside the line range), ``kind``
(chunk kind, e.g. "table_card"), ``contains`` (substring of the chunk text).
Questions with ``"expect": []`` are unanswerable from the corpus; they are
listed but not scored here (the agent's "not found" is judged end to end).
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Any


def _matches(row: dict[str, Any], exp: dict[str, Any]) -> bool:
    if "file" in exp and not fnmatch.fnmatch(row["display_path"].lower(), str(exp["file"]).lower()):
        return False
    if "page" in exp:
        ps, pe = row.get("page_start"), row.get("page_end") or row.get("page_start")
        if ps is None or not (ps <= int(exp["page"]) <= pe):
            return False
    if "sheet" in exp and (row.get("sheet") or "").lower() != str(exp["sheet"]).lower():
        return False
    if "line" in exp:
        ls, le = row.get("line_start"), row.get("line_end") or row.get("line_start")
        if ls is None or not (ls <= int(exp["line"]) <= le):
            return False
    if "kind" in exp and row["kind"] != exp["kind"]:
        return False
    if "contains" in exp and str(exp["contains"]).lower() not in row["text"].lower():
        return False
    return True


def run_eval(searcher, golden_path: Path, *, k_values=(5, 20)) -> dict[str, Any]:
    questions = [json.loads(line) for line in Path(golden_path).read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith("#")]
    # Unanswerable questions ("expect": []) can only be judged end to end, by whether
    # the agent says "not found"; they are listed but kept out of retrieval recall.
    unanswerable = [q for q in questions if not q.get("expect")]
    questions = [q for q in questions if q.get("expect")]
    per_q = []
    for q in questions:
        res = searcher.search(q["question"], expand=False)
        order = res.candidates[: max(k_values + (10,))]
        rows = searcher._load(order)
        first_hit = None
        for rank, cid in enumerate(order, start=1):
            if any(_matches(rows[cid], e) for e in q["expect"]):
                first_hit = rank
                break
        per_q.append({
            "id": q.get("id"), "type": q.get("type", "unspecified"), "question": q["question"],
            "first_hit_rank": first_hit,
            **{f"hit@{k}": bool(first_hit and first_hit <= k) for k in k_values},
            "rr@10": (1.0 / first_hit) if first_hit and first_hit <= 10 else 0.0,
            "embed_ms": res.timings_ms.get("embed"), "search_ms": res.timings_ms.get("search"),
        })

    def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items) or 1
        out = {"n": len(items)}
        for k in k_values:
            out[f"recall@{k}"] = round(sum(i[f"hit@{k}"] for i in items) / n, 3)
        out["mrr@10"] = round(sum(i["rr@10"] for i in items) / n, 3)
        return out

    by_type: dict[str, list] = {}
    for item in per_q:
        by_type.setdefault(item["type"], []).append(item)
    lat = sorted(i["search_ms"] or 0 for i in per_q)
    emb = sorted(i["embed_ms"] or 0 for i in per_q)

    def pct(xs: list[float], p: float) -> float:
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))] if xs else 0.0

    return {
        "overall": summarize(per_q),
        "by_type": {t: summarize(v) for t, v in sorted(by_type.items())},
        "latency_ms": {"embed_p50": pct(emb, 0.5), "embed_p95": pct(emb, 0.95),
                       "search_p50": pct(lat, 0.5), "search_p95": pct(lat, 0.95)},
        "questions": per_q,
        "unanswerable": [{"id": q.get("id"), "question": q["question"]} for q in unanswerable],
    }
