"""Retrieval over a workspace or pack: FTS5 + the vector store, fused with Reciprocal Rank Fusion,
optionally reranked, then expanded and cited.

The search itself runs in the shared engine (``shammy.Retriever``, Rust), the same code the
retrieval server calls through C#, so the Workbench shows exactly what the agent gets. This
module embeds the question, calls the engine and wraps the answer.

The original Python implementation is kept below as the reference (``engine="reference"``): a
test holds the two to the same output, and it is what runs when the shammy module isn't
installed. Reranking is only in the shared engine.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
import os
from typing import Any, Callable

import numpy as np

from rag.core.chunk.identifiers import query_identifiers, split_identifier
from rag.core.db.store import RagStore
from rag.core.spot import SpotIndex, norm_number

_STOP = set("""a an and are as at be by can do does for from has have how i if in is it of on or our
should that the their there these this to was we what when where which who why will with you your""".split())

RerankFn = Callable[[str, list[dict[str, Any]]], list[float]]


def rerank_settings(config: dict[str, Any] | None, enabled: bool | None = None, url: str | None = None) -> dict[str, Any] | None:
    """The reranker request for a search, from a workspace's ``[rerank]`` settings. ``enabled`` and
    ``url`` override what the config says; RAG_RERANK_URL overrides the config's url. None when off."""
    r = dict((config or {}).get("rerank") or {})
    on = bool(r.get("enabled")) if enabled is None else bool(enabled)
    if not on:
        return None
    r["enabled"] = True
    r["url"] = url or os.environ.get("RAG_RERANK_URL") or r.get("url") or ""
    return r


def native_engine():
    """The shared search engine module, or None when it isn't installed."""
    try:
        import shammy
    except ImportError:
        return None
    return shammy if hasattr(shammy, "Retriever") else None


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


def procedure_note(p: dict[str, Any]) -> str:
    """"8.0 Electrode Replacement: (steps 8.1-8.2 of 4)" for a result's source line."""
    name = p.get("title") or p.get("label") or "procedure"
    n = p.get("steps") or 0
    if p.get("whole"):
        span = f"whole, {n} step" + ("" if n == 1 else "s")
    elif p.get("from") == p.get("to"):
        span = f"step {p.get('from') or '?'} of {n}"
    else:
        span = f"steps {p.get('from') or '?'}-{p.get('to') or '?'} of {n}"
    return f"{name} ({span})"


def related_note(x: dict[str, Any]) -> str:
    if x.get("kind") == "section":
        return "section " + (x.get("title") or x.get("ref") or "")
    return (x.get("ref") or "") + (f" ({x['file']})" if x.get("file") else "")


@dataclass
class SearchResult:
    query: str
    results: list[dict[str, Any]] = field(default_factory=list)
    tables: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    candidates: list[int] = field(default_factory=list)   # fused order before cutoff (for eval)
    ranks: dict[int, dict[str, Any]] = field(default_factory=dict)   # chunk_id -> vector/keyword rank, fused score
    rerank: dict[str, Any] | None = None   # {"applied": bool, "error": str?} when reranking was asked for
    markdown: str | None = None            # the agent's compact form, as the shared engine wrote it
    engine: str = "reference"

    def to_dict(self) -> dict[str, Any]:
        out = {"query": self.query, "results": self.results, "tables": self.tables, "timings_ms": self.timings_ms}
        if self.rerank is not None:
            out["rerank"] = self.rerank
        return out

    def to_markdown(self) -> str:
        """Compact form for an agent: a source line per result, then the passage."""
        return self.markdown if self.markdown is not None else self.reference_markdown()

    def reference_markdown(self) -> str:
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
            if s.get("document"):
                loc.insert(0, s["document"] + (f" rev {s['revision']}" if s.get("revision") else ""))
            if r.get("procedure"):
                loc.append(procedure_note(r["procedure"]))
            out.append(f"[{r['tag']}] {s['file']}" + (" · " + " · ".join(loc) if loc else "") + f"  (chunk {r['chunk_id']})")
            out.append(r["text"])
            if r.get("related"):
                out.append("Related: " + "; ".join(related_note(x) for x in r["related"]))
            out.append("")
        if self.tables:
            out.append("Tables that may answer this (use describe_table / query_table):")
            for t in self.tables:
                out.append(f"- {t['table_name']}: {t['summary']}")
        return "\n".join(out).strip()


class Searcher:
    def __init__(self, store: RagStore, embedder, cfg: dict[str, Any], *, rerank: RerankFn | None = None,
                 reranker: dict[str, Any] | None = None, engine: str = "auto"):
        """``reranker`` is a rerank request (see ``rerank_settings``). ``engine`` is "auto" (the
        shared engine when installed), "native" (require it) or "reference" (the Python version;
        ``rerank`` is its callable hook)."""
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self.rerank = rerank
        self.reranker = reranker
        self.engine = engine
        self._retriever = None

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()
            self._retriever = None

    def _native(self):
        if self.engine == "reference" or self.rerank is not None:
            return None
        if self._retriever is None:
            module = native_engine()
            if module is None:
                if self.engine == "native":
                    raise RuntimeError("the shared search engine (the shammy module) isn't installed")
                return None
            if not self.store.readonly and self.store._vectors is not None:
                self.store.vectors.flush()   # the engine reads what is on disk
            self._retriever = module.Retriever(self.store.path.parent)
        return self._retriever

    def _vector(self, qvec: np.ndarray, k: int, file_type: str | None) -> list[int]:
        return self.store.vectors.search(qvec, k, file_type)

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
               expand: bool = True, documents: list[str] | None = None) -> SearchResult:
        """``documents``: only these documents, by document number ("D02") or part of the path."""
        s = self.cfg
        k = int(k or s["keep"])
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        qvec = self.embedder.embed_query(query)
        timings["embed"] = round((time.perf_counter() - t0) * 1000, 1)

        retriever = self._native()
        if retriever is not None:
            out = retriever.search(
                query, qvec, k=k, file_type=file_type or None, expand=expand, vec_k=int(s["vec_k"]),
                fts_k=int(s["fts_k"]), rrf_k=float(s["rrf_k"]), neighbors=int(s.get("neighbors", 0)),
                max_result_tokens=int(s.get("max_result_tokens", 6000)), markdown=True, rerank=self.reranker,
                procedures=bool(s.get("procedures", True)), procedure_tokens=int(s.get("procedure_tokens", 1500)),
                procedure_window=int(s.get("procedure_window", 1)), documents=list(documents or []))
            return SearchResult(
                query=query, results=out["results"], tables=out["tables"], timings_ms={**timings, **out["timings_ms"]},
                candidates=out["candidates"], ranks={int(c): r for c, r in out["ranks"].items()},
                rerank=out.get("rerank"), markdown=out.get("markdown"), engine="native")
        if self.reranker:
            return self._reference_search(query, qvec, k, file_type, expand, timings, note={
                "applied": False, "error": "reranking needs the shared search engine (the shammy module), which isn't installed"},
                documents=documents)
        return self._reference_search(query, qvec, k, file_type, expand, timings, documents=documents)

    def _reference_search(self, query: str, qvec, k: int, file_type: str | None, expand: bool,
                          timings: dict[str, float], note: dict[str, Any] | None = None,
                          documents: list[str] | None = None) -> SearchResult:
        s = self.cfg
        spot = SpotIndex.open(self.store.path.parent, self.store.conn)
        try:
            return self._reference_core(query, qvec, k, file_type, expand, timings, note, documents, spot)
        finally:
            if spot is not None:
                spot.close()

    def _documents(self, spot: SpotIndex | None, terms: list[str] | None) -> set[int] | None:
        """The doc_ids a document filter keeps (None: no filter)."""
        terms = [t.strip() for t in (terms or []) if t and t.strip()]
        if not terms:
            return None
        keep: set[int] = set()
        lowered = [t.lower() for t in terms]
        for doc_id, path in self.store.conn.execute("SELECT doc_id, display_path FROM documents ORDER BY doc_id"):
            if any(t in (path or "").lower() for t in lowered):
                keep.add(doc_id)
        if spot is not None:
            wanted = {norm_number(t) for t in terms}
            keep.update(doc_id for doc_id, number in spot.numbers() if norm_number(number) in wanted)
        return keep

    def _procedure(self, spot: SpotIndex, cid: int, included: set[int]):
        """The hit's procedure, whole when it fits procedure_tokens, else its step with up to
        procedure_window steps on each side that fit. None when the hit isn't in a procedure or its
        step alone is too long (then it is expanded the usual way)."""
        s = self.cfg
        found = spot.member(cid)
        if found is None:
            return None
        pid, step = found
        proc = spot.procedure(pid)
        steps = spot.steps(pid)
        idx = next((i for i, st in enumerate(steps) if st["step"] == step), None)
        if proc is None or idx is None:
            return None
        cap = int(s.get("procedure_tokens", 1500))
        if proc["tokens"] <= cap:
            lo, hi, whole = 0, len(steps) - 1, True
        else:
            if steps[idx]["tokens"] > cap:
                return None
            lo = hi = idx
            total = steps[idx]["tokens"]
            whole = False
            for _ in range(max(0, int(s.get("procedure_window", 1)))):
                grew = False
                if hi + 1 < len(steps) and total + steps[hi + 1]["tokens"] <= cap:
                    hi += 1
                    total += steps[hi]["tokens"]
                    grew = True
                if lo - 1 >= 0 and total + steps[lo - 1]["tokens"] <= cap:
                    lo -= 1
                    total += steps[lo]["tokens"]
                    grew = True
                if not grew:
                    break
        members = [(c, st) for c, st in spot.members(pid, steps[lo]["step"], steps[hi]["step"]) if c not in included]
        if cid not in {c for c, _ in members}:
            return None
        rows = self._load([c for c, _ in members])
        members = [(c, st) for c, st in members if c in rows]
        titles = {st["step"]: st["title"] or st["label"] for st in steps}
        blocks: list[str] = []
        for st in [x["step"] for x in steps[lo:hi + 1]]:
            texts = [rows[c]["text"] for c, m in members if m == st]
            if not texts:
                continue
            body = "\n\n".join(texts)
            header = titles.get(st) if proc["kind"] == "numbered" else None
            blocks.append(f"### {header}\n\n{body}" if header else body)
        info: dict[str, Any] = {"id": pid, "kind": proc["kind"], "title": proc["title"], "label": proc["label"],
                                "steps": proc["steps"], "from": steps[lo]["label"], "to": steps[hi]["label"], "whole": whole}
        return [c for c, _ in members], rows, "\n\n".join(blocks), info, (pid, steps[lo]["step"], steps[hi]["step"])

    def _related(self, spot: SpotIndex, span: list[int], window: tuple[int, int, int] | None) -> list[dict[str, Any]]:
        """References made in a result's chunks ("see Section 8.4", "see MF164"), except to steps
        the result already holds."""
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for c in span:
            for link in spot.links(c):
                key = (link["kind"], link["ref"])
                if key in seen:
                    continue
                if link["kind"] == "section":
                    if window and link["proc_id"] == window[0] and window[1] <= link["step"] <= window[2]:
                        continue
                    info = spot.step_info(link["proc_id"], link["step"])
                    item = {"kind": "section", "ref": link["ref"], "title": (info[1] or info[0]) if info else None}
                else:
                    path = None
                    if link["doc_id"] is not None:
                        row = self.store.conn.execute("SELECT display_path FROM documents WHERE doc_id = ?",
                                                      (link["doc_id"],)).fetchone()
                        path = row[0] if row else None
                    item = {"kind": "document", "ref": link["ref"], "file": path}
                seen.add(key)
                out.append(item)
        return out

    def _reference_core(self, query: str, qvec, k: int, file_type: str | None, expand: bool,
                        timings: dict[str, float], note: dict[str, Any] | None, documents: list[str] | None,
                        spot: SpotIndex | None) -> SearchResult:
        s = self.cfg
        allowed = self._documents(spot, documents)
        widen = 4 if allowed is not None else 1

        t1 = time.perf_counter()
        vec_ids = self._vector(qvec, int(s["vec_k"]) * widen, file_type)
        kw_ids = self._keyword(query, int(s["fts_k"]) * widen, file_type)
        rrf_k = float(s["rrf_k"])
        scores: dict[int, float] = {}
        for ranked in (vec_ids, kw_ids):
            for rank, cid in enumerate(ranked, start=1):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
        fused = sorted(scores, key=lambda c: scores[c], reverse=True)
        rows = self._load(fused)
        fused = [c for c in fused if c in rows and (allowed is None or rows[c]["doc_id"] in allowed)]
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

        vec_rank = {cid: r for r, cid in enumerate(vec_ids, start=1)}
        kw_rank = {cid: r for r, cid in enumerate(kw_ids, start=1)}
        result = SearchResult(query=query, candidates=ordered, timings_ms=timings, rerank=note, ranks={
            cid: {"vector": vec_rank.get(cid), "keyword": kw_rank.get(cid), "fused": round(scores[cid], 5),
                  "rank": i}
            for i, cid in enumerate(ordered, start=1)})
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
            proc_info = None
            window = None
            found = (self._procedure(spot, cid, included)
                     if expand and spot is not None and s.get("procedures", True) else None)
            if found is not None:
                span, span_rows, text, proc_info, window = found
                tokens = sum(int(span_rows[c]["n_tokens"] or 0) for c in span)
            else:
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
            if spot is not None:
                number, revision = spot.document(row["doc_id"])
                if number:
                    item["source"]["document"] = number
                    if revision:
                        item["source"]["revision"] = revision
                if proc_info is not None:
                    item["procedure"] = proc_info
                related = self._related(spot, span, window)
                if related:
                    item["related"] = related
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
