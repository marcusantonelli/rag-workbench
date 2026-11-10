#!/usr/bin/env python3
"""Shammy milestone 0: does the three-stage funnel hold recall on real embeddings?

Reads the vectors already stored in a workspace or .ragpack (nothing is re-embedded and nothing
is written to it), encodes them the way the Shammy plan describes, and simulates the search:

    stage 1  1-bit sketch of the first D/4 (or D/2) rotated dimensions, over every chunk
    stage 2  1 bit for all D dimensions + the RaBitQ estimate, over the stage-1 survivors
    stage 3  8-bit (or 6-bit) codes over the stage-2 survivors
    finish   exact rescoring of the best 50 (the lossless tier)

and compares what comes out with an exact float32 search. It answers the open design questions:
centroid or no centroid, block-wise or full rotation, sketch width, 8-bit or 6-bit, and how many
candidates each stage must keep.

    python m0_funnel.py siemens-kb.ragpack
    python m0_funnel.py my-workspace --golden golden.jsonl --embed-url http://127.0.0.1:8081

Needs: Python 3.10+, numpy, sqlite-vec (pip install numpy sqlite-vec). No GPU, no model, unless
--golden is used (then the embedding model must be running, e.g. in RAG Workbench).

Queries: by default 200 stored chunks are held out and used as queries (no model needed). That
measures the codes fairly but chunk-to-chunk is easier than question-to-chunk, so confirm with
--golden when you can.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

K = 20                      # results the agent gets
FINISH = 50                 # candidates rescored exactly at the end
STAGE1 = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
STAGE2 = [64, 128, 256, 512]
TARGET = 0.99


# ---------------------------------------------------------------- loading

def open_db(path: Path) -> tuple[sqlite3.Connection, tempfile.TemporaryDirectory | None]:
    tmp = None
    if path.is_file() and zipfile.is_zipfile(path):
        tmp = tempfile.TemporaryDirectory(prefix="shammy_m0_")
        with zipfile.ZipFile(path) as z:
            z.extract("rag.db", tmp.name)
        db = Path(tmp.name) / "rag.db"
    elif (path / "rag.db").exists():
        db = path / "rag.db"
    else:
        sys.exit(f"{path} is not a .ragpack file or a workspace folder")
    try:                                   # rag-builder knows how to load sqlite-vec everywhere
        from rag.core.db.connect import connect
        return connect(db, readonly=True, load_vec=True), tmp
    except ImportError:
        pass
    try:
        import sqlite_vec
    except ImportError:
        sys.exit("sqlite-vec is missing: pip install sqlite-vec")
    conn = sqlite3.connect(f"{db.resolve().as_uri()}?mode=ro", uri=True)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
    except AttributeError:
        sys.exit("This Python's sqlite3 can't load extensions. Run with the Python that has "
                 "rag-builder installed, or: pip install sqlean.py")
    return conn, tmp


def load_vectors(conn: sqlite3.Connection, max_rows: int, seed: int) -> tuple[np.ndarray, dict]:
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    n = conn.execute("SELECT COUNT(*) FROM chunk_vec").fetchone()[0]
    if n == 0:
        sys.exit("This knowledge base has no vectors.")
    ids = [r[0] for r in conn.execute("SELECT chunk_id FROM chunk_vec")]
    if n > max_rows:
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=max_rows, replace=False).tolist())
    first = conn.execute("SELECT embedding FROM chunk_vec WHERE chunk_id = ?", (ids[0],)).fetchone()[0]
    dims = len(first) // 4
    out = np.empty((len(ids), dims), dtype=np.float32)
    wanted = set(ids) if n > max_rows else None
    i = 0
    for cid, blob in conn.execute("SELECT chunk_id, embedding FROM chunk_vec"):
        if wanted is not None and cid not in wanted:
            continue
        out[i] = np.frombuffer(blob, dtype=np.float32)
        i += 1
    out = out[:i]
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return out, {"total_rows": n, "dims": dims, "model": meta.get("embed_model"),
                 "backend": meta.get("embed_backend"), "query_instruction": meta.get("query_instruction")}


def embed_golden(golden: Path, url: str, info: dict) -> np.ndarray:
    questions = []
    for line in golden.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            q = json.loads(line).get("question")
            if q:
                questions.append(q)
    if not questions:
        sys.exit(f"No questions in {golden}")
    template = info.get("query_instruction") or "{query}"
    texts = [template.replace("{query}", q) for q in questions]
    out = []
    for i in range(0, len(texts), 8):
        req = urllib.request.Request(f"{url.rstrip('/')}/v1/embeddings",
                                     data=json.dumps({"input": texts[i:i + 8]}).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())["data"]
        except OSError as exc:
            sys.exit(f"Can't reach the embedding model at {url}: {exc}")
        out.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
    q = np.asarray(out, dtype=np.float32)[:, :info["dims"]]
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    return q


# ---------------------------------------------------------------- encoding

def blocks_for(dims: int, mode: str) -> list[tuple[int, int]]:
    """Rotation blocks. 'block' keeps the Matryoshka prefixes D/4 and D/2 intact."""
    if mode == "none":
        return []
    if mode == "full":
        return [(0, dims)]
    return [(0, dims // 4), (dims // 4, dims // 2), (dims // 2, dims)]


def fwht(x: np.ndarray) -> np.ndarray:
    """Orthonormal Walsh-Hadamard transform along the last axis (length must be a power of two)."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    h = 1
    while h < n:
        x = x.reshape(*lead, n // (2 * h), 2, h)
        x = np.stack((x[..., 0, :] + x[..., 1, :], x[..., 0, :] - x[..., 1, :]), axis=-2)
        h *= 2
    return x.reshape(*lead, n) / np.sqrt(n)


class Rotation:
    """Three rounds of (random signs, Walsh-Hadamard) per block; a dense random rotation for a
    block whose length isn't a power of two."""

    def __init__(self, dims: int, mode: str, seed: int):
        rng = np.random.default_rng(seed)
        self.parts = []
        for a, b in blocks_for(dims, mode):
            n = b - a
            if n & (n - 1) == 0:
                self.parts.append((a, b, "h", rng.choice(np.float32([-1, 1]), size=(3, n))))
            else:
                q, _ = np.linalg.qr(rng.standard_normal((n, n)))
                self.parts.append((a, b, "d", q.astype(np.float32)))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if not self.parts:
            return x
        out = np.empty_like(x)
        for a, b, kind, m in self.parts:
            v = x[:, a:b]
            if kind == "h":
                for signs in m:
                    v = fwht(v * signs)
            else:
                v = v @ m
            out[:, a:b] = v
        return out


def quantize(v: np.ndarray, bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Extended-RaBitQ-style scalar code: per row, try several clipping scales, keep the code best
    aligned with the vector. Returns codes (float32 holding small ints) and the per-row factor
    that turns code.q into an estimate of v.q."""
    levels = 2 ** (bits - 1) - 1
    peak = np.abs(v).max(axis=1, keepdims=True)
    norm = np.linalg.norm(v, axis=1)
    best_cos = np.full(v.shape[0], -1.0, dtype=np.float32)
    best = np.zeros_like(v)
    for f in (1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.25, 0.18):
        code = np.clip(np.rint(v / (peak * f + 1e-12) * levels), -levels, levels)
        cos = np.einsum("ij,ij->i", code, v) / (np.linalg.norm(code, axis=1) * norm + 1e-12)
        better = cos > best_cos
        best[better] = code[better]
        best_cos[better] = cos[better]
    factor = (norm ** 2) / (np.einsum("ij,ij->i", best, v) + 1e-12)
    return best, factor.astype(np.float32)


# ---------------------------------------------------------------- scoring

def score_all(x: np.ndarray, q: np.ndarray, rot: Rotation, centroid: np.ndarray | None,
              qbits: tuple[int, ...], chunk: int = 8192) -> dict[str, np.ndarray]:
    """Score matrices (queries x rows) for every stage, built chunk by chunk."""
    n, d = x.shape
    nq = q.shape[0]
    quarter, half = d // 4, d // 2
    qr = rot(q)                    # for dot-product estimates the query stays uncentred (centring it
    #                                only shifts all of a query's scores equally); for XOR+popcount the
    #                                query's bits must come from the same centred space as the rows'
    qs = np.where((qr if centroid is None else rot(q - centroid)) >= 0, 1.0, -1.0).astype(np.float32)
    names = ["ham_q", "ham_h", "ham_f", "asym_q", "asym_h", "asym_f"] + [f"q{b}" for b in qbits]
    out = {k: np.empty((nq, n), dtype=np.float32) for k in names}
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        v = x[s:e] if centroid is None else x[s:e] - centroid
        v = rot(v)
        bits = np.where(v >= 0, 1.0, -1.0).astype(np.float32)
        parts = ((0, quarter), (quarter, half), (half, d))
        ham = [bits[:, a:b] @ qs[:, a:b].T for a, b in parts]          # = width - 2 x Hamming distance
        asym = [bits[:, a:b] @ qr[:, a:b].T for a, b in parts]
        l1 = [np.abs(v[:, a:b]).sum(axis=1) for a, b in parts]
        l2 = [np.einsum("ij,ij->i", v[:, a:b], v[:, a:b]) for a, b in parts]
        for i, tag in enumerate(("q", "h", "f")):
            out[f"ham_{tag}"][:, s:e] = sum(ham[:i + 1]).T
            # RaBitQ estimate of the dot product over these dimensions: |v|^2 (b.q) / |v|_1
            scale = sum(l2[:i + 1]) / (sum(l1[:i + 1]) + 1e-12)
            out[f"asym_{tag}"][:, s:e] = (sum(asym[:i + 1]) * scale[:, None]).T
        for b in qbits:
            code, factor = quantize(v, b)
            out[f"q{b}"][:, s:e] = ((code @ qr.T) * factor[:, None]).T
    return out


def top(scores: np.ndarray, c: int) -> np.ndarray:
    c = min(c, scores.shape[1])
    if c == scores.shape[1]:
        return np.tile(np.arange(c), (scores.shape[0], 1))
    return np.argpartition(-scores, c - 1, axis=1)[:, :c]


def narrow(cand: np.ndarray, scores: np.ndarray, c: int) -> np.ndarray:
    """Keep the best c of each query's candidates according to another score matrix."""
    s = np.take_along_axis(scores, cand, axis=1)
    return np.take_along_axis(cand, top(s, c), axis=1)


def recall(cand: np.ndarray, truth: np.ndarray) -> float:
    hits = sum(len(np.intersect1d(c, t, assume_unique=False)) for c, t in zip(cand, truth))
    return hits / truth.size


def score_ratio(cand: np.ndarray, truth: np.ndarray, exact: np.ndarray) -> float:
    """Mean exact score of what was returned / mean exact score of the true top-k (1.0 = no loss)."""
    got = np.sort(np.take_along_axis(exact, cand, axis=1), axis=1)
    want = np.sort(np.take_along_axis(exact, truth, axis=1), axis=1)
    return float(got.mean() / want.mean())


def evaluate(sc: dict[str, np.ndarray], exact: np.ndarray, truth: np.ndarray, n: int,
             qbits: tuple[int, ...]) -> list[dict]:
    rows = []
    for sketch, tag in (("D/4", "q"), ("D/2", "h")):
        for mode in ("ham", "asym"):
            s1 = sc[f"{mode}_{tag}"]
            for c1 in [c for c in STAGE1 if c < n]:
                cand1 = top(s1, c1)
                r1 = recall(cand1, truth)
                for c2 in [c for c in STAGE2 if c < c1]:
                    cand2 = narrow(cand1, sc["asym_f"], c2)
                    r2 = recall(cand2, truth)
                    for b in qbits:
                        final = narrow(cand2, sc[f"q{b}"], K)
                        fin = narrow(narrow(cand2, sc[f"q{b}"], min(FINISH, c2)), exact, K)
                        rows.append({"sketch": sketch, "stage1": mode, "c1": c1, "c2": c2, "qbits": b,
                                     "per_million": round(c1 / n * 1e6),
                                     "recall_stage1": r1, "recall_stage2": r2,
                                     "recall_final": recall(final, truth),
                                     "recall_exact_finish": recall(fin, truth),
                                     "score_ratio": score_ratio(fin, truth, exact)})
    return rows


# ---------------------------------------------------------------- report

def cost(r: dict, dims: int) -> float:
    """Bytes read per query per chunk in the file (lower is cheaper): what to minimise."""
    sketch = dims // 32 if r["sketch"] == "D/4" else dims // 16
    return sketch + (r["c1"] / 1e6) * (dims // 8) + (r["c2"] / 1e6) * dims * r["qbits"] / 8


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="workspace folder or .ragpack file with real embeddings")
    ap.add_argument("--queries", type=int, default=200, help="held-out chunks used as queries (default 200)")
    ap.add_argument("--golden", help="golden.jsonl: use its questions as queries (needs --embed-url)")
    ap.add_argument("--embed-url", default="http://127.0.0.1:8081", help="llama-server for --golden")
    ap.add_argument("--max-rows", type=int, default=150_000, help="sample this many chunks at most")
    ap.add_argument("--seed", type=int, default=20261109)
    ap.add_argument("--out", default="m0_results.json")
    args = ap.parse_args()

    t0 = time.time()
    conn, tmp = open_db(Path(args.source))
    x, info = load_vectors(conn, args.max_rows + (0 if args.golden else args.queries), args.seed)
    conn.close()
    dims = info["dims"]
    if dims % 4:
        sys.exit(f"{dims} dimensions can't be split into quarter/half prefixes")
    rng = np.random.default_rng(args.seed)
    if args.golden:
        q = embed_golden(Path(args.golden), args.embed_url, info)
        qsource = f"{len(q)} questions from {args.golden}"
    else:
        nq = min(args.queries, len(x) // 10)
        pick = rng.choice(len(x), size=nq, replace=False)
        q = x[pick].copy()
        x = np.delete(x, pick, axis=0)
        qsource = f"{nq} held-out chunks"
    n = len(x)
    print(f"{args.source}: {info['total_rows']:,} chunks stored, using {n:,} x {dims} dims, "
          f"model {info['model']} ({info['backend']})")
    print(f"Queries: {qsource}. Loaded in {time.time() - t0:.0f} s.")
    if info["backend"] == "hash-test":
        print("WARNING: these are hash-test vectors, not real embeddings. The result means nothing.")
    if n < 20_000:
        print(f"NOTE: only {n:,} chunks. Read the 'per 1M' column: keeping 256 of {n:,} is like keeping "
              f"{round(256 / n * 1e6):,} of a million, so small candidate counts are the informative ones.")
    if n <= K * 4:
        sys.exit("Too few chunks to test.")

    exact = q @ x.T
    truth = top(exact, K)
    gb = (6 + 2) * q.shape[0] * n * 4 / 1e9
    if gb > 4:
        print(f"NOTE: this needs about {gb:.1f} GB of RAM; lower --max-rows or --queries if it swaps.")

    qbits = (8, 6)
    results, summary = [], []
    for rotation in ("block", "full", "none"):
        rot = Rotation(dims, rotation, args.seed)
        for cname in ("none", "mean"):
            t = time.time()
            centroid = x.mean(axis=0) if cname == "mean" else None
            sc = score_all(x, q, rot, centroid, qbits)
            rows = evaluate(sc, exact, truth, n, qbits)
            for r in rows:
                r.update(rotation=rotation, centroid=cname)
            results.extend(rows)
            direct = {name: recall(top(sc[key], K), truth) for name, key in
                      (("1-bit D/4", "asym_q"), ("1-bit D/2", "asym_h"), ("1-bit D", "asym_f"),
                       ("8-bit D", "q8"), ("6-bit D", "q6"))}
            summary.append({"rotation": rotation, "centroid": cname, **direct})
            print(f"  rotation={rotation:5} centroid={cname:4}  done in {time.time() - t:.0f} s")
            del sc

    print(f"\nRecall@{K} of each code used alone (no funnel, no rescoring): how good is the code itself?")
    cols = ["1-bit D/4", "1-bit D/2", "1-bit D", "6-bit D", "8-bit D"]
    print(f"  {'rotation':9}{'centroid':9}" + "".join(f"{c:>11}" for c in cols))
    for s in summary:
        print(f"  {s['rotation']:9}{s['centroid']:9}" + "".join(f"{s[c]:>11.3f}" for c in cols))

    design = [r for r in results if r["rotation"] == "block" and r["centroid"] == "mean"
              and r["sketch"] == "D/4" and r["stage1"] == "ham" and r["qbits"] == 8]
    print(f"\nThe funnel as planned (block rotation, mean centroid, D/4 sketch, XOR+popcount, 8-bit):")
    print(f"  {'stage 1':>8}{'per 1M':>9}{'stage 2':>9}{'after s1':>10}{'after s2':>10}{'final':>8}"
          f"{'+exact':>8}{'score':>8}")
    for r in sorted(design, key=lambda r: (r["c1"], r["c2"])):
        if r["c2"] in (64, 128, 256) and r["c1"] <= 4096:
            print(f"  {r['c1']:>8}{r['per_million']:>9,}{r['c2']:>9}{r['recall_stage1']:>10.3f}"
                  f"{r['recall_stage2']:>10.3f}{r['recall_final']:>8.3f}{r['recall_exact_finish']:>8.3f}"
                  f"{r['score_ratio']:>8.4f}")
    print("  (after s1/s2: share of the true top 20 still among the candidates; +exact: with the "
          "lossless finish;\n   score: exact score of what was returned / of the true top 20)")

    ok = [r for r in results if r["recall_exact_finish"] >= TARGET]
    print(f"\nCheapest settings that reach recall@{K} >= {TARGET} with the exact finish:")
    if not ok:
        best = max(results, key=lambda r: r["recall_exact_finish"])
        print(f"  none. Best was {best['recall_exact_finish']:.3f} with {best}")
        verdict = "fail"
    else:
        seen = set()
        for r in sorted(ok, key=lambda r: cost(r, dims)):
            key = (r["rotation"], r["centroid"], r["sketch"], r["stage1"])
            if key in seen:
                continue
            seen.add(key)
            if len(seen) > 8:
                break
            print(f"  rotation={r['rotation']:5} centroid={r['centroid']:4} sketch={r['sketch']} "
                  f"stage1={r['stage1']:4} keep {r['c1']:>5} ({r['per_million']:,} per 1M) -> {r['c2']:>3}, "
                  f"{r['qbits']}-bit: {r['recall_exact_finish']:.3f}")
        planned = [r for r in design if r["recall_exact_finish"] >= TARGET]
        share = min((r["c1"] / n for r in planned), default=1.0)
        verdict = "pass" if share <= 0.005 else "pass-more" if share <= 0.02 else "pass-with-changes"

    msg = {"pass": "PASS: the design as planned reaches the target keeping 0.5% or less in stage 1. "
                   "Go to milestone 1.",
           "pass-more": "PASS: the planned design reaches the target if stage 1 keeps up to 2% of the chunks "
                        "(20,000 per million: still cheap). Go to milestone 1 with that default.",
           "pass-with-changes": "PASS WITH CHANGES: the target is reachable, but not with the planned "
                                "settings at an affordable stage-1 count. Use the cheapest line above.",
           "fail": "FAIL: no setting reached the target. The sketch needs rethinking before any Rust."}
    print(f"\n{msg[verdict]}")
    Path(args.out).write_text(json.dumps({"source": str(args.source), "rows": n, "dims": dims, "info": info,
                                          "queries": qsource, "k": K, "target": TARGET, "verdict": verdict,
                                          "codes_alone": summary, "funnel": results}, indent=1), encoding="utf-8")
    print(f"Full results: {args.out} (send this file back). Total {time.time() - t0:.0f} s.")
    if tmp:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
