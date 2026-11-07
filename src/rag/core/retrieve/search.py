"""Test retrieval over a pack: FTS5 + sqlite-vec, fused with Reciprocal Rank Fusion.

This is the reference behaviour for the future C# retrieval server: same
query parsing, same FTS query construction, same RRF, same result shape.
There is no reranker in v1; ``rerank`` is the hook where v2 plugs one in.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from rag.core.chunk.identifiers import query_identifiers, split_identifier
from rag.core.db.store import RagStore, vec_bytes

_STOP = set("""a an and are as at be by can do does for from has have how i if in is it of on or our
should that the their there these this to was we what when where which who why will with you your""".split())

RerankFn = Callable[[str, list[dict[str, Any]]], list[float]]


def fts_query(query: str) -> str:
    """Build an FTS5 MATCH expression: identifiers as exact phrases, words OR-ed."""
    terms: list[str] = []
    for ident in query_identifiers(query):
        terms.append('"' + ident.replace('"', '""') + '"')
        parts = split_identifier(ident)
        if len(parts) > 1 and all(p.isalnum() for p in parts):
            terms.append('"' + " ".join(parts) + '"')
    for word in re.findall(r"[A-Za-z0-9]+", query):
        w = word.lower()
        if len(w) < 2 or w in _STOP:
            continue
        terms.append('"' + w + '"')
    seen: list[str] = []
    for t in terms:
        if t not in seen:
            seen.append(t)
    return " OR ".join(seen)


@dataclass
class SearchResult:
    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: list[int] = field(default_factory=list)   # fused order before cutoff (for eval)

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "results": self.results, "tables": self.tables, "timings_ms": self.timings_ms}

    def to_markdown(self) -> str:
        """Compact form for an agent: a source line per result, then the passage."""
        out = []
        for r in self.results:
            s = r["source"]
            loc = []
            if s.get("page"):
                loc.append(f"p.{s['page']}" + (f"-{s['page_end']}" if s.get("page_end") not in (None, s["page"]) else ""))
            if s.get("sheet"):
                loc.append(f"sheet {s['sheet']}")
            if s.get("range"):
                loc.append(s["range"])
            if s.get("lines"):
                loc.append(f"lines {s['lines']}")
            if s.get("heading_path"):
                loc.append(s["heading_path"])
            if s.get("symbol"):
                loc.append(s["symbol"])
            out.append(f"[{r['tag']}] {s['file']}" + (" · " + " · ".join(loc) if loc else "") + f"  (chunk {r['chunk_id']})")
            out.append(r["text"])
            out.append("")
        if self.tables:
            out.append("Tables that may answer this (use describe_table / query_table):")
            for t in self.tables:
                out.append(f"- {t['table_name']}: {t['summary']}")
        return "\n".join(out).strip()


class Searcher:
    def __init__(self, store: RagStore, embedder, cfg: dict[str, Any], *, rerank: RerankFn | None = None):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self.rerank = rerank

    def _vector(self, qvec: np.ndarray, k: int, file_type: str | None) -> list[int]:
        sql = "SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ? AND k = ?"
        args: list[Any] = [vec_bytes(qvec), k]
        if file_type:
            sql += " AND file_type = ?"
            args.append(file_type)
        return [r[0] for r in self.store.conn.execute(sql, args).fetchall()]

    def _keyword(self, query: str, k: int, file_type: str | None) -> list[int]:
        expr = fts_query(query)
        if not expr:
            return []
        rows = self.store.conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?", (expr, k * 2)
        ).fetchall()
        ids = [r[0] for r in rows]
        if file_type and ids:
            marks = ",".join("?" * len(ids))
            keep = {r[0] for r in self.store.conn.execute(
                f"SELECT c.chunk_id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
                f"WHERE c.chunk_id IN ({marks}) AND d.file_type = ?", (*ids, file_type))}
            ids = [i for i in ids if i in keep]
        return ids[:k]

    def _load(self, ids: list[int]) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        rows = self.store.conn.execute(
            f"SELECT c.*, d.display_path, d.file_type FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            f"WHERE c.chunk_id IN ({marks})", ids).fetchall()
        return {r["chunk_id"]: dict(r) for r in rows}

    def neighbors(self, chunk_id: int, n: int = 1) -> list[dict[str, Any]]:
        """The chunk plus up to n prose chunks on each side from the same section."""
        row = self.store.conn.execute(
            "SELECT doc_id, ordinal, heading_path FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if row is None:
            return []
        rows = self.store.conn.execute(
            "SELECT c.*, d.display_path, d.file_type FROM chunks c JOIN documents d ON d.doc_id = c.doc_id "
            "WHERE c.doc_id = ? AND c.ordinal BETWEEN ? AND ? AND c.kind = 'prose' "
            "AND c.heading_path IS ? ORDER BY c.ordinal",
            (row["doc_id"], row["ordinal"] - n, row["ordinal"] + n, row["heading_path"])).fetchall()
        # Keep only the contiguous run around the hit.
        ordinals = [r["ordinal"] for r in rows]
        if row["ordinal"] not in ordinals:
            return []
        i = ordinals.index(row["ordinal"])
        lo, hi = i, i
        while lo > 0 and ordinals[lo - 1] == ordinals[lo] - 1:
            lo -= 1
        while hi < len(rows) - 1 and ordinals[hi + 1] == ordinals[hi] + 1:
            hi += 1
        return [dict(r) for r in rows[lo:hi + 1]]

    def search(self, query: str, *, k: int | None = None, file_type: str | None = None,
               expand: bool = True) -> SearchResult:
        s = self.cfg
        k = int(k or s["keep"])
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        timings["embed"] = round((time.perf_counter() - t0) * 1000, 1)

        t1 = time.perf_counter()
        vec_ids = self._vector(qvec, int(s["vec_k"]), file_type)
        kw_ids = self._keyword(query, int(s["fts_k"]), file_type)
        rrf_k = float(s["rrf_k"])
        scores: dict[int, float] = {}
        for ranked in (vec_ids, kw_ids):
            for rank, cid in enumerate(ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        fused = sorted(scores, key=lambda c: scores[c], reverse=True)
        rows = self._load(fused)
        # Collapse identical text (boilerplate repeated across files).
        seen_hash: set[str] = set()
        ordered: list[int] = []
        for cid in fused:
            h = rows[cid]["text_hash"]
            if h in seen_hash:
                continue
            seen_hash.add(h)
            ordered.append(cid)
        timings["search"] = round((time.perf_counter() - t1) * 1000, 1)

        candidates = ordered[:20]
        if self.rerank is not None:  # v2 hook
            t2 = time.perf_counter()
            rr = self.rerank(query, [rows[c] for c in candidates])
            candidates = [c for _, c in sorted(zip(rr, candidates), key=lambda x: x[0], reverse=True)]
            timings["rerank"] = round((time.perf_counter() - t2) * 1000, 1)

        result = SearchResult(query=query, candidates=ordered, timings_ms=timings)
        budget = int(s.get("max_result_tokens", 6000))
        used = 0
        tag = 0
        included: set[int] = set()
        for cid in candidates:
            if len(result.results) >= k:
                break
            if cid in included:
                continue
            row = rows[cid]
            if row["kind"] == "table_card":
                result.tables.append({
                    "table_name": row["table_name"], "chunk_id": cid, "score": round(scores[cid], 5),
                    "summary": row["text"].split("\n", 1)[0], "card": row["text"],
                })
                continue
            text = row["text"]
            span = [cid]
            if expand and row["kind"] == "prose" and int(s.get("neighbors", 0)) > 0:
                around = [r for r in self.neighbors(cid, int(s["neighbors"])) if r["chunk_id"] not in included]
                if len(around) > 1:
                    text = "\n\n".join(r["text"] for r in around)
                    span = [r["chunk_id"] for r in around]
            tokens = int(row["n_tokens"] or 0) * (3 if text != row["text"] else 1)
            if used and used + tokens > budget:
                break
            used += tokens
            included.update(span)
            tag += 1
            item = self._format(row, cid, text, f"S{tag}", scores[cid])
            if len(span) > 1:
                item["chunk_ids"] = span
            result.results.append(item)
        return result

    @staticmethod
    def _format(row: dict[str, Any], cid: int, text: str, tag: str, score: float) -> dict[str, Any]:
        lines = None
        if row.get("line_start"):
            lines = f"{row['line_start']}-{row['line_end']}" if row.get("line_end") else str(row["line_start"])
        return {
            "tag": tag,
            "chunk_id": cid,
            "kind": row["kind"],
            "score": round(score, 5),
            "text": text,
            "source": {
                "file": row["display_path"],
                "file_type": row["file_type"],
                "page": row.get("page_start"),
                "page_end": row.get("page_end"),
                "sheet": row.get("sheet"),
                "range": row.get("cell_range"),
                "lines": lines,
                "heading_path": row.get("heading_path"),
                "symbol": row.get("symbol"),
                "table_name": row.get("table_name"),
                "asset_path": row.get("asset_path"),
                "ocr_conf": row.get("ocr_conf"),
            },
        }
