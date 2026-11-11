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
from rag.core.spots import SpotsIndex, norm_number

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
    a, b = p.get("from") or "?", p.get("to") or "?"
    level = p.get("level")
    if p.get("whole"):
        span = f"whole, {n} step" + ("" if n == 1 else "s")
    elif level == "substeps":
        span = f"step {p.get('step') or '?'} of {n}, " + (f"sub-step {a}" if a == b else f"sub-steps {a}-{b}")
    elif level == "chunks":
        span = f"part of step {a} of {n}" if a == p.get("step") else f"part of step {p.get('step') or '?'} of {n}, sub-step {a}"
    elif p.get("from") == p.get("to"):
        span = f"step {a} of {n}"
    else:
        span = f"steps {a}-{b} of {n}"
    return f"{name} ({span})"


def outline_note(p: dict[str, Any], o: dict[str, Any]) -> str:
    """The procedure's steps, the ones shown in bold, so a reader knows where this passage sits."""
    items = [f"**{x['title']}**" if x.get("shown") else x["title"] for x in o.get("steps", [])]
    if o.get("before"):
        items.insert(0, "…")
    if o.get("after"):
        items.append("…")
    return f"Outline of {p.get('title') or p.get('label') or 'the procedure'}: " + "; ".join(items)


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
            if r.get("outline"):
                out.append(outline_note(r["procedure"], r["outline"]))
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
                procedure_window=int(s.get("procedure_window", 1)), procedure_hits=int(s.get("procedure_hits", 3)),
                documents=list(documents or []))
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
        spots = SpotsIndex.open(self.store.path.parent, self.store.conn)
        try:
            return self._reference_core(query, qvec, k, file_type, expand, timings, note, documents, spots)
        finally:
            if spots is not None:
                spots.close()

    def _documents(self, spots: SpotsIndex | None, terms: list[str] | None) -> set[int] | None:
        """The doc_ids a document filter keeps (None: no filter)."""
        terms = [t.strip() for t in (terms or []) if t and t.strip()]
        if not terms:
            return None
        keep: set[int] = set()
        lowered = [t.lower() for t in terms]
        for doc_id, path in self.store.conn.execute("SELECT doc_id, display_path FROM documents ORDER BY doc_id"):
            if any(t in (path or "").lower() for t in lowered):
                keep.add(doc_id)
        if spots is not None:
            wanted = {norm_number(t) for t in terms}
            keep.update(doc_id for doc_id, number in spots.numbers() if norm_number(number) in wanted)
        return keep

    @staticmethod
    def _grow(tokens: list[int], i: int, cap: int, window: int) -> tuple[int, int]:
        """From item i, add up to ``window`` items on each side (after first, then before) while
        the total stays within cap."""
        lo = hi = i
        total = tokens[i]
        for _ in range(max(0, window)):
            grew = False
            if hi + 1 < len(tokens) and total + tokens[hi + 1] <= cap:
                hi += 1
                total += tokens[hi]
                grew = True
            if lo - 1 >= 0 and total + tokens[lo - 1] <= cap:
                lo -= 1
                total += tokens[lo]
                grew = True
            if not grew:
                break
        return lo, hi

    def _plan(self, spots: SpotsIndex, cid: int) -> dict[str, Any] | None:
        """What to return for a hit inside a procedure, at the largest size that fits
        procedure_tokens: the whole procedure; the hit's step and the steps around it; the hit's
        sub-step and those around it; or the chunks around the hit inside its step."""
        s = self.cfg
        found = spots.member(cid)
        if found is None:
            return None
        pid, step, sub = found
        proc = spots.procedure(pid)
        steps = spots.steps(pid)
        idx = next((i for i, st in enumerate(steps) if st["step"] == step), None)
        if proc is None or idx is None:
            return None
        cap = int(s.get("procedure_tokens", 1500))
        win = int(s.get("procedure_window", 1))
        plan: dict[str, Any] = {"pid": pid, "proc": proc, "steps": steps, "idx": idx}
        if proc["tokens"] <= cap:
            return {**plan, "level": "procedure", "lo": 0, "hi": len(steps) - 1}
        if steps[idx]["tokens"] <= cap:
            lo, hi = self._grow([st["tokens"] for st in steps], idx, cap, win)
            return {**plan, "level": "steps", "lo": lo, "hi": hi}
        subs = spots.substeps(pid, step)
        j = next((i for i, x in enumerate(subs) if x["sub"] == sub), None)
        if len(subs) >= 2 and j is not None and subs[j]["tokens"] <= cap:
            lo, hi = self._grow([x["tokens"] for x in subs], j, cap, win)
            return {**plan, "level": "substeps", "subs": subs, "lo": lo, "hi": hi}
        # one step (or sub-step) longer than procedure_tokens: the chunks around the hit inside it
        inside = [m[0] for m in spots.members(pid, step, step) if len(subs) < 2 or m[2] == sub]
        rows = self._load(inside)
        inside = [c for c in inside if c in rows]
        k = inside.index(cid) if cid in inside else None
        if k is None:
            return None
        lo, hi = self._grow([int(rows[c]["n_tokens"] or 0) for c in inside], k, cap, win)
        at = subs[j]["label"] if len(subs) >= 2 and j is not None and sub > 0 else steps[idx]["label"]
        return {**plan, "level": "chunks", "chunks": inside[lo:hi + 1], "at": at}

    def _render(self, spots: SpotsIndex, plan: dict[str, Any], exclude: set[int]):
        """The text, chunks, procedure note and outline for a plan, leaving out chunks already
        returned. None when nothing is left."""
        pid, proc, steps, level = plan["pid"], plan["proc"], plan["steps"], plan["level"]
        if level in ("procedure", "steps"):
            members = spots.members(pid, steps[plan["lo"]]["step"], steps[plan["hi"]]["step"])
        else:
            step = steps[plan["idx"]]["step"]
            members = spots.members(pid, step, step)
            if level == "substeps":
                a, b = plan["subs"][plan["lo"]]["sub"], plan["subs"][plan["hi"]]["sub"]
                members = [m for m in members if a <= m[2] <= b]
            else:
                wanted = set(plan["chunks"])
                members = [m for m in members if m[0] in wanted]
        members = [m for m in members if m[0] not in exclude]
        rows = self._load([m[0] for m in members])
        members = [m for m in members if m[0] in rows]
        if not members:
            return None
        numbered = proc["kind"] == "numbered"
        step_titles = {st["step"]: st["title"] or st["label"] for st in steps}
        sub_titles: dict[int, dict[int, str | None]] = {}
        blocks: list[str] = []
        last_step = None
        i = 0
        while i < len(members):
            step, sub = members[i][1], members[i][2]
            group = []
            while i < len(members) and (members[i][1], members[i][2]) == (step, sub):
                group.append(members[i][0])
                i += 1
            parts: list[str] = []
            if numbered and step != last_step and step_titles.get(step):
                parts.append(f"### {step_titles[step]}")
            if numbered and sub > 0:
                if step not in sub_titles:
                    sub_titles[step] = {x["sub"]: x["title"] or x["label"] for x in spots.substeps(pid, step)}
                if sub_titles[step].get(sub):
                    parts.append(f"#### {sub_titles[step][sub]}")
            parts.append("\n\n".join(rows[c]["text"] for c in group))
            blocks.append("\n\n".join(parts))
            last_step = step
        info: dict[str, Any] = {"id": pid, "kind": proc["kind"], "title": proc["title"], "label": proc["label"],
                                "steps": proc["steps"], "level": level, "whole": level == "procedure"}
        if level in ("procedure", "steps"):
            info["from"], info["to"] = steps[plan["lo"]]["label"], steps[plan["hi"]]["label"]
            shown = set(range(plan["lo"], plan["hi"] + 1))
            window = (pid, steps[plan["lo"]]["step"], steps[plan["hi"]]["step"])
        else:
            info["step"] = steps[plan["idx"]]["label"]
            if level == "substeps":
                info["from"], info["to"] = plan["subs"][plan["lo"]]["label"], plan["subs"][plan["hi"]]["label"]
            else:
                info["from"] = info["to"] = plan["at"]
            shown = {plan["idx"]}
            window = (pid, steps[plan["idx"]]["step"], steps[plan["idx"]]["step"])
        outline = None
        if numbered and level != "procedure":
            n = len(steps)
            a, b = min(shown), max(shown)
            s0, e0 = (0, n - 1) if n <= 20 else (max(0, a - 8), min(n - 1, b + 8))
            outline = {"steps": [{"label": steps[x]["label"], "title": (steps[x]["title"] or steps[x]["label"] or "")[:50],
                                  "shown": x in shown} for x in range(s0, e0 + 1)],
                       "before": s0 > 0, "after": e0 < n - 1}
        span = [m[0] for m in members]
        tokens = sum(int(rows[c]["n_tokens"] or 0) for c in span)
        return span, "\n\n".join(blocks), info, outline, window, tokens

    def _related(self, spots: SpotsIndex, span: list[int], window: tuple[int, int, int] | None) -> list[dict[str, Any]]:
        """References made in a result's chunks ("see Section 8.4", "see MF164"), except to steps
        the result already holds."""
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for c in span:
            for link in spots.links(c):
                key = (link["kind"], link["ref"])
                if key in seen:
                    continue
                if link["kind"] == "section":
                    if window and link["proc_id"] == window[0] and window[1] <= link["step"] <= window[2]:
                        continue
                    info = spots.step_info(link["proc_id"], link["step"])
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
                        spots: SpotsIndex | None) -> SearchResult:
        s = self.cfg
        allowed = self._documents(spots, documents)
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
        procs: list[dict[str, Any]] = []        # procedure results so far, for merging
        procs_seen: dict[int, int] = {}         # results per procedure
        # hits past a procedure's procedure_hits wait at the end of the queue: they only fill
        # places nothing else took
        queue = list(candidates)
        deferred: set[int] = set()
        at = 0
        while at < len(queue):
            cid = queue[at]
            at += 1
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
            proc_info = outline = window = None
            plan = (self._plan(spots, cid)
                    if expand and spots is not None and s.get("procedures", True) else None)
            if plan is not None:
                if plan["level"] == "steps":
                    merged = self._merge(spots, plan, procs, result, included, budget, used)
                    if merged is not None:
                        used = merged
                        continue
                if cid not in deferred and procs_seen.get(plan["pid"], 0) >= int(s.get("procedure_hits", 3)):
                    deferred.add(cid)         # enough from this procedure for now; others first
                    queue.append(cid)
                    continue
            rendered = self._render(spots, plan, included) if plan is not None else None
            if rendered is not None:
                span, text, proc_info, outline, window, tokens = rendered
            else:
                plan = None
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
            if spots is not None:
                number, revision = spots.document(row["doc_id"])
                if number:
                    item["source"]["document"] = number
                    if revision:
                        item["source"]["revision"] = revision
                if proc_info is not None:
                    item["procedure"] = proc_info
                    if outline is not None and not procs_seen.get(plan["pid"]):
                        item["outline"] = outline      # once per procedure: the first result
                related = self._related(spots, span, window)
                if related:
                    item["related"] = related
            if plan is not None:
                procs_seen[plan["pid"]] = procs_seen.get(plan["pid"], 0) + 1
                procs.append({"index": len(result.results), "plan": plan, "span": span, "tokens": tokens})
            result.results.append(item)
        return result

    def _merge(self, spots: SpotsIndex, plan: dict[str, Any], procs: list[dict[str, Any]], result: SearchResult,
               included: set[int], budget: int, used: int) -> int | None:
        """A hit in the step just before or after a window of steps already returned widens that
        window instead of coming back as a separate piece (up to twice procedure_tokens). Returns
        the tokens used after merging, or None when it wasn't merged."""
        cap = int(self.cfg.get("procedure_tokens", 1500))
        idx = plan["idx"]
        for prev in procs:
            p = prev["plan"]
            if p["pid"] != plan["pid"] or p["level"] != "steps" or not (p["lo"] - 1 <= idx <= p["hi"] + 1):
                continue
            lo, hi = min(p["lo"], idx), max(p["hi"], idx)
            if sum(st["tokens"] for st in p["steps"][lo:hi + 1]) > 2 * cap:
                return None
            wider = {**p, "lo": lo, "hi": hi}
            rendered = self._render(spots, wider, included - set(prev["span"]))
            if rendered is None:
                return None
            span, text, info, outline, window, tokens = rendered
            if used + tokens - prev["tokens"] > budget:
                return None
            item = result.results[prev["index"]]
            item["text"] = text
            if len(span) > 1:
                item["chunk_ids"] = span
            item["procedure"] = info
            if outline is not None and "outline" in item:
                item["outline"] = outline
            related = self._related(spots, span, window)
            if related:
                item["related"] = related
            else:
                item.pop("related", None)
            included.update(span)
            grown = used + tokens - prev["tokens"]
            prev.update({"plan": wider, "span": span, "tokens": tokens})
            return grown
        return None

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
