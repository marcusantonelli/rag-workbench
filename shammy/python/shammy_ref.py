"""Reference implementation of the Shammy encoding and a read-only file parser, in NumPy.

This is the specification in executable form (spec/FORMAT.md is the prose). The Rust engine is
tested against it: codes must match bit for bit, scores to float precision.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

MAGIC = b"SHAMMY\x00\x01"
FOOTER_MAGIC = b"SHFOOT\x00\x01"
DEFAULT_SEED = 0x5348414D4D593031
ROUNDS = 3
Q8_SCALES = (1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.25, 0.18)
SECTIONS = {1: "ids", 2: "doc", 3: "ftype", 4: "sketch", 5: "bits", 6: "corr", 7: "q8", 8: "q8corr", 9: "f16",
            10: "tomb"}
_M = (1 << 64) - 1


def splitmix64(seed: int):
    state = seed & _M
    while True:
        state = (state + 0x9E3779B97F4A7C15) & _M
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M
        yield z ^ (z >> 31)


def blocks(dims: int) -> list[tuple[int, int]]:
    return [(0, dims // 4), (dims // 4, dims // 2), (dims // 2, dims)]


def rotation_signs(dims: int, seed: int = DEFAULT_SEED) -> np.ndarray:
    """signs[round, dimension]: drawn block by block, round by round, 64 dimensions per word."""
    rng = splitmix64(seed)
    signs = np.ones((ROUNDS, dims), dtype=np.float32)
    for a, b in blocks(dims):
        for r in range(ROUNDS):
            word = 0
            for i in range(b - a):
                if i % 64 == 0:
                    word = next(rng)
                if (word >> (i % 64)) & 1:
                    signs[r, a + i] = -1.0
    return signs


def fwht(x: np.ndarray) -> np.ndarray:
    """Orthonormal Walsh-Hadamard transform along the last axis, in float32 like the engine."""
    n = x.shape[-1]
    lead = x.shape[:-1]
    h = 1
    while h < n:
        x = x.reshape(*lead, n // (2 * h), 2, h)
        x = np.stack((x[..., 0, :] + x[..., 1, :], x[..., 0, :] - x[..., 1, :]), axis=-2)
        h *= 2
    return x.reshape(*lead, n) * np.float32(1.0 / np.sqrt(np.float64(n)))


def rotate(x: np.ndarray, seed: int = DEFAULT_SEED) -> np.ndarray:
    x = np.ascontiguousarray(x, dtype=np.float32)
    signs = rotation_signs(x.shape[-1], seed)
    out = np.empty_like(x)
    for a, b in blocks(x.shape[-1]):
        v = x[..., a:b]
        for r in range(ROUNDS):
            v = fwht(v * signs[r, a:b])
        out[..., a:b] = v
    return out


def normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = np.sqrt((x.astype(np.float64) ** 2).sum(axis=-1, keepdims=True))
    inv = np.where(norm > 0, 1.0 / np.where(norm > 0, norm, 1.0), 1.0).astype(np.float32)
    return x * inv


def encode(vectors: np.ndarray, sketch_dims: int | None = None, seed: int = DEFAULT_SEED) -> dict[str, np.ndarray]:
    """Encode rows exactly as the writer does. Returns the section contents as arrays."""
    unit = normalize(np.atleast_2d(vectors))
    n, dims = unit.shape
    sketch_dims = sketch_dims or dims // 4
    v = rotate(unit, seed)
    v64 = v.astype(np.float64)
    packed = np.packbits(v < 0, axis=1, bitorder="little")

    def scale(part: np.ndarray) -> np.ndarray:
        l1 = np.abs(part).sum(axis=1)
        return np.where(l1 > 0, (part ** 2).sum(axis=1) / np.where(l1 > 0, l1, 1), 0).astype(np.float32)

    corr = np.stack((scale(v64[:, :sketch_dims]), scale(v64)), axis=1)
    peak = np.abs(v).max(axis=1, keepdims=True)
    norm2 = (v64 ** 2).sum(axis=1)
    best = np.zeros((n, dims), dtype=np.int8)
    best_cos = np.full(n, -1.0)
    best_dot = np.zeros(n)
    live = peak[:, 0] > 0
    for f in Q8_SCALES:
        with np.errstate(divide="ignore", invalid="ignore"):
            code = np.clip(np.rint(v / (peak * np.float32(f)) * np.float32(127.0)), -127, 127)
        code = np.where(live[:, None], code, 0).astype(np.float64)
        dot = (code * v64).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = dot / (np.sqrt((code ** 2).sum(axis=1)) * np.sqrt(norm2))
        better = live & (cos > best_cos)
        best[better] = code[better].astype(np.int8)
        best_cos[better] = cos[better]
        best_dot[better] = dot[better]
    factor = np.where(best_dot > 0, norm2 / np.where(best_dot > 0, best_dot, 1), 0).astype(np.float32)
    return {"unit": unit, "rotated": v, "sketch": packed[:, :sketch_dims // 8], "bits": packed[:, sketch_dims // 8:],
            "corr": corr, "q8": best, "q8corr": np.stack((factor, np.zeros_like(factor)), axis=1),
            "f16": unit.astype(np.float16)}


def read(path: str | Path) -> dict:
    """Parse a .shammy file: header fields, footer fields and each segment's sections as arrays."""
    data = Path(path).read_bytes()
    if data[:8] != MAGIC:
        raise ValueError("not a Shammy file")
    major, minor, dims, sketch_dims, flags, seed = struct.unpack_from("<HHIIIQ", data, 8)
    out = {"major": major, "minor": minor, "dims": dims, "sketch_dims": sketch_dims, "flags": flags,
           "rotation_seed": seed, "uuid": data[32:48].hex()}
    end = len(data)
    while end >= 96:
        if data[end - 8:end] == FOOTER_MAGIC:
            off, length, _checksum = struct.unpack_from("<QQQ", data, end - 32)
            if off + length == end - 32:
                break
        end -= 1
    else:
        raise ValueError("no complete commit")
    f = memoryview(data)[off:off + length]
    version, gen, rows, live, nseg, ihash = struct.unpack_from("<IQQQIQ", f, 0)
    at = 40

    def string() -> str:
        nonlocal at
        (n,) = struct.unpack_from("<H", f, at)
        s = bytes(f[at + 2:at + 2 + n]).decode()
        at += 2 + n
        return s

    model = string()
    (ntypes,) = struct.unpack_from("<H", f, at)
    at += 2
    types = [string() for _ in range(ntypes)]
    (nsec,) = struct.unpack_from("<I", f, at)
    at += 4
    out.update(footer_version=version, generation=gen, rows=rows, live_rows=live, model=model,
               instruction_hash=ihash, file_types=types, committed_bytes=end, segments=[{} for _ in range(nseg)])
    dtypes = {"ids": ("<u8", 1), "doc": ("<u4", 1), "ftype": ("u1", 1), "sketch": ("u1", sketch_dims // 8),
              "bits": ("u1", (dims - sketch_dims) // 8), "corr": ("<f4", 2), "q8": ("i1", dims),
              "q8corr": ("<f4", 2), "f16": ("<f2", dims)}
    for _ in range(nsec):
        seg, kind, _codec, offset, length, srows, _sum, _blocks = struct.unpack_from("<IHHQQQQQ", f, at)
        at += 48
        name = SECTIONS.get(kind)
        if name is None:
            continue
        raw = data[offset:offset + length]
        if name == "tomb":
            arr = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:srows].astype(bool)
        else:
            dt, width = dtypes[name]
            arr = np.frombuffer(raw, dtype=dt)
            arr = arr.reshape(srows, width) if width > 1 else arr
        out["segments"][seg][name] = arr          # a later tombstone bitmap replaces an earlier one
    return out


def search(file: dict, query: np.ndarray, k: int = 20, stage1: int = 0, stage2: int = 256, finish: int = 50):
    """The engine's search, written the slow obvious way. Returns [(id, score)] best first."""
    dims, sd = file["dims"], file["sketch_dims"]
    unit = normalize(np.asarray(query, dtype=np.float32))
    qr = rotate(unit, file["rotation_seed"])
    qbits = np.packbits(qr < 0, bitorder="little")
    stage1 = max(stage1 or max(file["live_rows"] // 100, 4096), k)
    stage2, finish = max(stage2, k), max(finish, k)
    cands = []
    for s, seg in enumerate(file["segments"]):
        ham = np.unpackbits(seg["sketch"] ^ qbits[:sd // 8], axis=1).sum(axis=1)
        dead = seg.get("tomb", np.zeros(len(ham), dtype=bool))
        cands += [(-float(h), s, r) for r, h in enumerate(ham) if not dead[r]]
    order = lambda c: (-c[0], c[1], c[2])  # noqa: E731
    cands = sorted(cands, key=order)[:stage1]

    def rescored(fn, keep):
        return sorted(((fn(file["segments"][s], r), s, r) for _, s, r in cands), key=order)[:keep]

    q32 = qr.astype(np.float32)

    def lanes(terms: np.ndarray) -> float:
        """Sum float32 terms in 16 lanes (dimension i in lane i % 16), then the lanes in float64."""
        acc = np.zeros(16, dtype=np.float32)
        for group in terms.astype(np.float32).reshape(-1, 16):
            acc += group
        total = 0.0
        for x in acc:
            total += float(x)
        return total

    def one_bit(seg, r):
        minus = np.unpackbits(np.concatenate((seg["sketch"][r], seg["bits"][r])), bitorder="little").astype(bool)
        return float(np.float32(lanes(np.where(minus, -q32, q32)) * np.float64(seg["corr"][r, 1])))

    cands = rescored(one_bit, stage2)
    lossless = bool(file["flags"] & 2)
    cands = rescored(lambda seg, r: float(np.float32(lanes(seg["q8"][r].astype(np.float32) * q32)
                                                       * np.float64(seg["q8corr"][r, 0]))), finish if lossless else k)
    if lossless:
        u64 = unit.astype(np.float64)
        cands = rescored(lambda seg, r: float(np.float32((seg["f16"][r].astype(np.float64) * u64).sum())), k)
    hits = [(int(file["segments"][s]["ids"][r]), score) for score, s, r in cands]
    return sorted(hits, key=lambda h: (-h[1], h[0]))
