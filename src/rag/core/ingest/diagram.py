"""Flowcharts and other box-and-arrow diagrams drawn as vector shapes in a PDF.

PDF exports from Visio, PowerPoint, draw.io and Word keep a diagram's labels as real text and
its boxes and arrows as paths. Read as plain text, the labels of boxes that sit side by side
run together into one line and the arrows are lost. Here the drawing is rebuilt instead:

* a box (rectangle, rounded rectangle, ellipse, diamond, any closed or filled shape) that
  holds text becomes a step; a shape around other steps with a short title of its own is a
  lane or group;
* lines and polylines are joined into wires (pieces that meet end to end, or a branch that
  starts on another wire), and a wire connects the steps its free ends touch, pointing at the
  ends that carry an arrowhead (a small filled shape);
* block arrows (a filled polygon with a shaft and a head, straight or bent, as Word, PowerPoint
  and Visio draw them) connect the step at their tail to the step at their tip;
* short text lying along a wire is its label ("Yes", "No"); other loose text inside the
  drawing is kept as notes.

A group of at least three steps joined by at least two connections is a diagram. Boxes that
touch each other (the cells of a table) and lines that run along box edges (table rules) are
not, so tables drawn with rectangles are left to the table reader.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass, field

Box = tuple[float, float, float, float]   # x0, top, x1, bottom (pdfplumber's top-down coordinates)
Pt = tuple[float, float]

MIN_SHAPE = 12.0       # smaller shapes are arrowheads, bullets or tick marks
HEAD_MAX = 16.0        # largest side of an arrowhead
JOIN = 1.6             # wire pieces whose ends are this close are one wire
ATTACH = 14.0          # a wire end this close to a box connects to it (room for an arrowhead)
HEAD_NEAR = 13.0       # an arrowhead this close to a wire end belongs to it
LABEL_NEAR = 14.0      # loose text this close to a wire is its label
MAX_OBJECTS = 4000     # pages with more drawing objects than this (maps, CAD, dense charts) are skipped


@dataclass
class Node:
    id: int
    bbox: Box
    label: str = ""
    kind: str = "box"            # box | decision | terminal
    lane: str | None = None
    words: list[int] = field(default_factory=list)


@dataclass
class Edge:
    src: int
    dst: int
    label: str = ""
    directed: bool = True
    both: bool = False


@dataclass
class Diagram:
    bbox: Box
    nodes: list[Node]
    edges: list[Edge]
    notes: list[str]
    word_ids: set[int]
    order: list[int] = field(default_factory=list)   # node ids in reading order
    node_notes: dict[int, list[str]] = field(default_factory=dict)   # note boxes beside a step

    @property
    def flowchart(self) -> bool:
        return sum(e.directed for e in self.edges) >= 2

    def to_text(self, title: str) -> str:
        """Steps in flow order, each with where it leads; then notes."""
        number = {nid: i + 1 for i, nid in enumerate(self.order)}
        by_id = {n.id: n for n in self.nodes}
        leads: dict[int, list[tuple[str, int]]] = defaultdict(list)     # arrows out of a step
        links: dict[int, list[tuple[str, int]]] = defaultdict(list)     # plain lines (no arrowhead)
        for e in self.edges:
            if e.directed:
                leads[e.src].append((e.label, e.dst))
                if e.both:
                    leads[e.dst].append((e.label, e.src))
            else:   # listed once, under the step that comes first
                a, b = sorted((e.src, e.dst), key=lambda n: number[n])
                links[a].append((e.label, b))

        def refs(items: list[tuple[str, int]]) -> str:
            return "; ".join(f"{text}: {number[n]}" if text else str(number[n])
                             for text, n in sorted(items, key=lambda t: number[t[1]]))

        has_in = {e.dst for e in self.edges} | {e.src for e in self.edges if e.both or not e.directed}
        has_out = {e.src for e in self.edges} | {e.dst for e in self.edges if e.both or not e.directed}
        lines = [title, ""]
        for nid in self.order:
            n = by_id[nid]
            label = n.label or "(unlabelled)"
            if n.kind == "offpage":      # a connector symbol that carries the flow to another page
                nxt, prv = getattr(self, "next_page", None), getattr(self, "prev_page", None)
                label = ((f"(continues on page {nxt})" if nxt else "(continues on another page)")
                         if nid in has_in and nid not in has_out else
                         (f"(continued from page {prv})" if prv else "(continued from another page)")
                         if nid in has_out and nid not in has_in else "(connector)")
            if n.kind == "decision" and not label.rstrip().endswith("?"):
                label += " (decision)"
            if n.lane:
                label += f" [{n.lane}]"
            line = f"{number[nid]}. {label}"
            if leads.get(nid):
                line += " → " + refs(leads[nid])
            if links.get(nid):
                line += " — " + refs(links[nid])
            lines.append(line)
            for note in self.node_notes.get(nid, []):
                lines.append(f"   Note: {note}")
        if self.notes:
            lines += ["", "Notes: " + " / ".join(self.notes)]
        return "\n".join(lines)


# -- geometry ------------------------------------------------------------------------------
def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _contains_pt(b: Box, p: Pt, pad: float = 0.0) -> bool:
    return b[0] - pad <= p[0] <= b[2] + pad and b[1] - pad <= p[1] <= b[3] + pad


def _contains_box(outer: Box, inner: Box, pad: float = 1.0) -> bool:
    return (outer[0] - pad <= inner[0] and outer[1] - pad <= inner[1]
            and inner[2] <= outer[2] + pad and inner[3] <= outer[3] + pad)


def _box_dist(b: Box, p: Pt) -> float:
    dx = max(b[0] - p[0], 0.0, p[0] - b[2])
    dy = max(b[1] - p[1], 0.0, p[1] - b[3])
    return math.hypot(dx, dy)


def _gap(a: Box, b: Box) -> float:
    dx = max(a[0] - b[2], 0.0, b[0] - a[2])
    dy = max(a[1] - b[3], 0.0, b[1] - a[3])
    return math.hypot(dx, dy)


def _seg_dist(p: Pt, a: Pt, b: Pt) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == dy == 0:
        return math.hypot(p[0] - ax, p[1] - ay)
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _poly_dist(p: Pt, pts: list[Pt]) -> float:
    if len(pts) == 1:
        return math.hypot(p[0] - pts[0][0], p[1] - pts[0][1])
    return min(_seg_dist(p, a, b) for a, b in zip(pts, pts[1:]))


def _length(pts: list[Pt]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))


def _bbox_of(pts: list[Pt]) -> Box:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _points_along(pts: list[Pt], fractions: tuple[float, ...]) -> list[Pt]:
    total = _length(pts)
    out = []
    for f in fractions:
        want, run = f * total, 0.0
        for a, b in zip(pts, pts[1:]):
            seg = math.hypot(b[0] - a[0], b[1] - a[1])
            if run + seg >= want and seg > 0:
                t = (want - run) / seg
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))
                break
            run += seg
    return out


def _pts(obj) -> list[Pt]:
    raw = obj.get("pts") or []
    pts = [(float(x), float(y)) for x, y in raw]
    # pdfplumber gives pts top-down in recent versions; older ones were bottom-up
    top, bottom = float(obj["top"]), float(obj["bottom"])
    if pts and not all(top - 1 <= y <= bottom + 1 for _, y in pts):
        height = float(obj.get("page_height") or (obj["top"] + obj["bottom"]))
        pts = [(x, height - y) for x, y in pts]
    return pts


def _is_diamond(pts: list[Pt], b: Box) -> bool:
    corners = {(round(p[0]), round(p[1])) for p in pts}
    if not 4 <= len(corners) <= 5:
        return False
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    mids = [(cx, b[1]), (b[2], cy), (cx, b[3]), (b[0], cy)]
    return all(any(math.hypot(p[0] - m[0], p[1] - m[1]) <= 2.5 for p in pts) for m in mids)


def _closed_like(pts: list[Pt], b: Box) -> bool:
    """A path that goes around its box (a shape) rather than from one place to another (a wire)."""
    if len(pts) < 3:
        return False
    if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) <= 2.0:
        return True
    w, h = b[2] - b[0], b[3] - b[1]
    sides = (any(abs(p[0] - b[0]) <= 2 for p in pts), any(abs(p[0] - b[2]) <= 2 for p in pts),
             any(abs(p[1] - b[1]) <= 2 for p in pts), any(abs(p[1] - b[3]) <= 2 for p in pts))
    gap = math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
    # a rounded rectangle: many points, all four sides, and the closing side left implicit
    return len(pts) >= 7 and all(sides) and gap <= 0.3 * 2 * (w + h)


# -- words ---------------------------------------------------------------------------------
def _block_arrow(pts: list[Pt], b: Box) -> Pt | None:
    """The tip of a block arrow, or None. A block arrow drawn upright has horizontal and vertical
    sides except the two that form its head, which meet at the tip: so exactly one corner has a
    slanted side on both hands."""
    if len(pts) < 7 or len(pts) > 14 or max(b[2] - b[0], b[3] - b[1]) < 14.0:
        return None
    ring = pts[:-1] if math.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1]) < 0.5 else list(pts)
    n = len(ring)
    if n < 7:
        return None

    def slanted(a: Pt, c: Pt) -> bool:
        return abs(a[0] - c[0]) > 1.0 and abs(a[1] - c[1]) > 1.0

    sides = [slanted(ring[i], ring[(i + 1) % n]) for i in range(n)]
    if sum(sides) != 2:
        return None
    tips = [ring[i] for i in range(n) if sides[i - 1] and sides[i]]
    return tips[0] if len(tips) == 1 else None


def _centre(w: dict) -> Pt:
    return (float(w["x0"]) + float(w["x1"])) / 2, (float(w["top"]) + float(w["bottom"])) / 2


def _join_words(words: list[dict]) -> str:
    """Words in reading order: lines top to bottom, words left to right, hyphenated breaks joined."""
    rows: list[list[dict]] = []
    for w in sorted(words, key=lambda w: (float(w["top"]), float(w["x0"]))):
        if rows and abs(float(rows[-1][0]["top"]) - float(w["top"])) <= 2.5:
            rows[-1].append(w)
        else:
            rows.append([w])
    text = ""
    for row in rows:
        line = " ".join(w["text"] for w in sorted(row, key=lambda w: float(w["x0"])))
        if text.endswith("-") and line[:1].islower():
            text = text[:-1] + line
        else:
            text = f"{text} {line}".strip()
    return re.sub(r"\s+", " ", text)


def _phrases(words: list[tuple[int, dict]]) -> list[list[tuple[int, dict]]]:
    """Loose words grouped into short phrases: same line and close together, or stacked tightly."""
    items = sorted(words, key=lambda iw: (float(iw[1]["top"]), float(iw[1]["x0"])))
    groups: list[list[tuple[int, dict]]] = []
    for iw in items:
        w = iw[1]
        size = float(w.get("size") or (float(w["bottom"]) - float(w["top"])) or 8.0)
        placed = False
        for g in groups:
            last = g[-1][1]
            same_line = abs(float(last["top"]) - float(w["top"])) <= 2.5 and 0 <= float(w["x0"]) - float(last["x1"]) <= 1.2 * size
            below = (0 <= float(w["top"]) - float(last["bottom"]) <= 0.6 * size
                     and abs((float(w["x0"]) + float(w["x1"])) / 2 - (float(last["x0"]) + float(last["x1"])) / 2) <= 4 * size)
            if same_line or below:
                g.append(iw)
                placed = True
                break
        if not placed:
            groups.append([iw])
    return groups


# -- the reader ----------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def find_diagrams(page, words: list[dict]) -> list[Diagram]:
    """The diagrams on a pdfplumber page. ``words`` is the page's extract_words() output; each
    Diagram lists the indexes of the words it took (labels, edge labels, notes, lane titles)."""
    rects, curves, lines = list(page.rects), list(page.curves), list(page.lines)
    if len(rects) + len(curves) + len(lines) > MAX_OBJECTS or not (rects or curves):
        return []

    shapes: list[tuple[Box, str]] = []          # (bbox, kind)
    plain: set[tuple[int, int, int, int]] = set()   # filled with no outline of their own
    heads: list[Pt] = []
    wires: list[list[Pt]] = []
    arrows: list[tuple[Pt, list[Pt]]] = []      # block arrows: (tip, outline)

    def _key(b: Box) -> tuple[int, int, int, int]:
        return tuple(int(round(v)) for v in b)   # type: ignore[return-value]

    def _white(obj) -> bool:
        c = obj.get("non_stroking_color")
        vals = list(c) if isinstance(c, (list, tuple)) else [c]
        try:
            return all(float(v) >= 0.95 for v in vals) and len(vals) in (1, 3)
        except (TypeError, ValueError):
            return False

    def _black(obj) -> bool:
        c = obj.get("non_stroking_color")
        vals = list(c) if isinstance(c, (list, tuple)) else [c]
        try:
            return all(float(v) <= 0.05 for v in vals) and len(vals) in (1, 3)
        except (TypeError, ValueError):
            return False

    # shapes drawn as a fill plus a separate outline share a box
    outlined = {_key((float(o["x0"]), float(o["top"]), float(o["x1"]), float(o["bottom"])))
                for o in rects + curves if o.get("stroke")}

    for r in rects:
        b = (float(r["x0"]), float(r["top"]), float(r["x1"]), float(r["bottom"]))
        w, h = b[2] - b[0], b[3] - b[1]
        if min(w, h) <= 2.0 and max(w, h) >= 8.0:          # a line drawn as a thin rectangle
            if w >= h:
                wires.append([(b[0], (b[1] + b[3]) / 2), (b[2], (b[1] + b[3]) / 2)])
            else:
                wires.append([((b[0] + b[2]) / 2, b[1]), ((b[0] + b[2]) / 2, b[3])])
        elif r.get("fill") and not r.get("stroke") and (_black(r) or _white(r) or (h <= 20.0 and _key(b) not in outlined)):
            continue        # a redaction bar, a white mask, or the background painted behind a line of text
        elif max(w, h) <= HEAD_MAX and r.get("fill"):
            if not _white(r):
                heads.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
        elif min(w, h) >= MIN_SHAPE:
            shapes.append((b, "box"))
            if r.get("fill") and not r.get("stroke") and _key(b) not in outlined:
                plain.add(_key(b))
    for c in curves:
        b = (float(c["x0"]), float(c["top"]), float(c["x1"]), float(c["bottom"]))
        w, h = b[2] - b[0], b[3] - b[1]
        pts = _pts(c)
        if max(w, h) <= HEAD_MAX:
            if (c.get("fill") and not _white(c)) or (not c.get("fill") and len(pts) >= 3):
                heads.append(((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
            continue
        if c.get("fill") and not c.get("stroke") and (_black(c) or _white(c) or min(w, h) < 2.0):
            continue        # a redaction bar, a white mask behind a label, or a hairline
        if c.get("fill") and len(pts) <= 5 and not c.get("stroke") and h <= 20.0 and _key(b) not in outlined:
            continue        # a rectangle drawn as a path: the background behind a line of text
        tip = _block_arrow(pts, b) if c.get("fill") else None
        if tip is not None:
            arrows.append((tip, pts))
            continue
        if min(w, h) >= MIN_SHAPE and (c.get("fill") or _closed_like(pts, b)):
            kind = "decision" if _is_diamond(pts, b) else ("terminal" if len(pts) >= 7 else "box")
            shapes.append((b, kind))
            if c.get("fill") and not c.get("stroke") and _key(b) not in outlined:
                plain.add(_key(b))
        elif c.get("stroke", True) and len(pts) >= 2:
            wires.append(pts)
    for ln in lines:
        pts = _pts(ln)
        if len(pts) >= 2 and _length(pts) >= 2.0:
            wires.append(pts)
    # one entry per box (fill and outline are the same shape); the more specific kind wins
    rank = {"box": 0, "terminal": 1, "decision": 2}
    unique: dict[tuple[int, int, int, int], tuple[Box, str]] = {}
    for b, kind in shapes:
        k = _key(b)
        if k not in unique or rank[kind] > rank[unique[k][1]]:
            unique[k] = (b, kind)
    shapes = list(unique.values())
    if len(shapes) < 3 or not (wires or arrows):
        return []

    # Open arrowheads: two short strokes meeting at a point (a "V") mark the tip of an arrow.
    short = [i for i, w in enumerate(wires) if len(w) == 2 and _length(w) <= HEAD_MAX]
    head_strokes: set[int] = set()
    for a_i, a in enumerate(short):
        for b in short[a_i + 1:]:
            pa, pb = wires[a], wires[b]
            for p in pa:
                for q in pb:
                    if math.hypot(p[0] - q[0], p[1] - q[1]) <= JOIN:
                        oa = pa[1] if p is pa[0] else pa[0]
                        ob = pb[1] if q is pb[0] else pb[0]
                        ang = abs(math.atan2(oa[1] - p[1], oa[0] - p[0]) - math.atan2(ob[1] - q[1], ob[0] - q[0]))
                        ang = min(ang, 2 * math.pi - ang)
                        if math.radians(20) <= ang <= math.radians(120):
                            heads.append(((p[0] + q[0]) / 2, (p[1] + q[1]) / 2))
                            head_strokes.update((a, b))
    wires = [w for i, w in enumerate(wires) if i not in head_strokes]
    if not (wires or arrows):
        return []

    # Every word belongs to the smallest shape around it.
    owner: dict[int, int] = {}
    for i, w in enumerate(words):
        c = _centre(w)
        inside = [k for k, (b, _) in enumerate(shapes) if _contains_pt(b, c, pad=-0.5)]
        if inside:
            owner[i] = min(inside, key=lambda k: _area(shapes[k][0]))
    owned: dict[int, list[int]] = defaultdict(list)
    for i, k in owner.items():
        owned[k].append(i)

    labelled = [k for k in range(len(shapes)) if owned.get(k)]
    # A shape around other labelled shapes is a group or lane (or a page frame), not a step.
    # A single small or near-identical shape inside a step (a highlight, a doubled outline) is
    # part of that step: its text goes to the step.
    containers: set[int] = set()
    absorbed: set[int] = set()
    wire_ends = [p for w in wires for p in (w[0], w[-1])]
    for k in sorted(labelled, key=lambda k: _area(shapes[k][0])):
        inner = [j for j in labelled if j != k and j not in absorbed and _contains_box(shapes[k][0], shapes[j][0])
                 and _area(shapes[k][0]) > _area(shapes[j][0])]
        # a fill with no outline inside another shape, which no connector touches, is a
        # highlight behind that shape's text
        for j in list(inner):
            if _key(shapes[j][0]) in plain and not any(_box_dist(shapes[j][0], p) <= 3.0 for p in wire_ends):
                owned[k].extend(owned[j])
                for i in owned[j]:
                    owner[i] = k
                absorbed.add(j)
                inner.remove(j)
        if not inner:
            continue
        ratio = _area(shapes[inner[0]][0]) / max(_area(shapes[k][0]), 1.0)
        small = shapes[inner[0]][0][3] - shapes[inner[0]][0][1] <= 22.0 and ratio < 0.25
        if len(inner) == 1 and inner[0] not in containers and (ratio >= 0.5 or small):
            owned[k].extend(owned[inner[0]])
            for i in owned[inner[0]]:
                owner[i] = k
            absorbed.add(inner[0])
        else:
            containers.add(k)
    nodes = [Node(id=n, bbox=shapes[k][0], kind=shapes[k][1], label=_join_words([words[i] for i in owned[k]]),
                  words=sorted(owned[k])) for n, k in enumerate(k for k in labelled if k not in containers and k not in absorbed)]
    if len(nodes) < 3:
        return []
    # Small shapes with no text that a connector runs into are connector symbols (the flow
    # continues on another page). They count as steps only if a wire reaches them.
    n_real = len(nodes)
    for k, (b, _) in enumerate(shapes):
        if owned.get(k) or max(b[2] - b[0], b[3] - b[1]) > 90 or any(_contains_box(b, n.bbox) for n in nodes[:n_real]) \
                or any(_contains_box(n.bbox, b) for n in nodes[:n_real]):
            continue
        nodes.append(Node(id=len(nodes), bbox=b, kind="offpage"))

    # Wires: pieces that meet end to end, or a branch starting on another piece, are one wire.
    uf = _UnionFind(len(wires))
    ends = [(w[0], w[-1]) for w in wires]
    near_node = lambda p, d=1.0: any(_box_dist(n.bbox, p) <= d for n in nodes)  # noqa: E731
    joined_ends: set[tuple[int, int]] = set()      # (wire, 0 or 1): this end meets another wire

    def meets(a: int, b: int) -> bool:
        """Does an end of wire a meet wire b: end to end, on its length, or through an arrowhead
        that points into b (arrows merging into a shared line)?"""
        hit = False
        for e, p in enumerate(ends[a]):
            if near_node(p):
                continue
            if min(math.hypot(p[0] - q[0], p[1] - q[1]) for q in ends[b]) <= JOIN or _poly_dist(p, wires[b]) <= JOIN:
                joined_ends.add((a, e))
                hit = True
                continue
            for h in heads:
                if math.hypot(h[0] - p[0], h[1] - p[1]) <= HEAD_NEAR and _poly_dist(h, wires[b]) <= 8.0 and not near_node(h, 6.0):
                    joined_ends.add((a, e))
                    hit = True
                    break
        return hit

    for a in range(len(wires)):
        for b in range(len(wires)):
            if a != b and meets(a, b):
                uf.union(a, b)
                for e, q in enumerate(ends[b]):    # an end-to-end meeting uses up both ends
                    if not near_node(q) and min(math.hypot(q[0] - p[0], q[1] - p[1]) for p in ends[a]) <= JOIN:
                        joined_ends.add((b, e))
    nets: dict[int, list[int]] = defaultdict(list)
    for i in range(len(wires)):
        nets[uf.find(i)].append(i)

    def runs_along_a_box(pts: list[Pt]) -> bool:
        """A table rule or a box outline: the middle of the wire lies on or in a box."""
        if _length(pts) < 12.0:
            return False      # a short link between two close shapes is near both by nature
        inner = _points_along(pts, (0.3, 0.5, 0.7))
        return any(any(_box_dist(n.bbox, p) <= 1.5 for n in nodes) for p in inner)

    edges: list[Edge] = []
    net_paths: list[tuple[list[list[Pt]], list[Edge]]] = []
    for members in nets.values():
        paths = [wires[i] for i in members]
        if all(runs_along_a_box(p) for p in paths):
            continue
        # free ends: ends not joined to another piece of the same wire
        free: list[tuple[Pt, list[Pt]]] = []
        for i in members:
            for e, p in enumerate(ends[i]):
                if (i, e) not in joined_ends:
                    free.append((p, wires[i]))
        touched: dict[int, bool] = {}       # node id -> has an arrowhead at that end
        branch: dict[int, list[Pt]] = {}
        for p, path in free:
            other = path[-1] if p is path[0] else path[0]
            # an arrowhead belongs to the end it is nearer (a short wire has both ends in reach),
            # and the shape it points at is measured from the arrowhead, which lies past the end
            mine = [h for h in heads if math.hypot(h[0] - p[0], h[1] - p[1]) <= HEAD_NEAR
                    and math.hypot(h[0] - p[0], h[1] - p[1]) <= math.hypot(h[0] - other[0], h[1] - other[1])]
            head = bool(mine)
            at = min(mine, key=lambda h: math.hypot(h[0] - p[0], h[1] - p[1])) if mine else p
            best = min(nodes, key=lambda n: min(_box_dist(n.bbox, at), _box_dist(n.bbox, p) + (4.0 if mine else 0.0)))
            if min(_box_dist(best.bbox, p), _box_dist(best.bbox, at)) > ATTACH:
                continue
            touched[best.id] = touched.get(best.id, False) or head
            branch[best.id] = path
        if len(touched) < 2:
            continue
        ids = sorted(touched, key=lambda n: (nodes[n].bbox[1], nodes[n].bbox[0]))
        targets = [n for n in ids if touched[n]]
        sources = [n for n in ids if not touched[n]]
        made: list[Edge] = []
        if targets and sources:
            made = [Edge(s, t) for s in sources for t in targets]
        elif targets and len(targets) == 2:                     # arrowheads at both ends
            made = [Edge(targets[0], targets[1], both=True)]
        elif len(ids) == 2:                                     # no arrowheads: a plain link
            made = [Edge(ids[0], ids[1], directed=False)]
        for e in made:
            e._paths = paths                                             # type: ignore[attr-defined]
            e._branch = branch.get(e.dst) or branch.get(e.src)          # type: ignore[attr-defined]
        edges.extend(made)
        net_paths.append((paths, made))

    # Block arrows: from the step nearest the tail to the step the tip points at.
    for tip, outline in arrows:
        target = min(nodes, key=lambda n: _box_dist(n.bbox, tip))
        if _box_dist(target.bbox, tip) > ATTACH:
            continue
        others = [n for n in nodes if n.id != target.id]
        if not others:
            continue
        reach = lambda n: min(_box_dist(n.bbox, q) for q in outline)  # noqa: E731
        source = min(others, key=reach)
        if reach(source) > ATTACH:
            continue
        e = Edge(source.id, target.id)
        e._paths = [outline]                                            # type: ignore[attr-defined]
        e._branch = outline                                             # type: ignore[attr-defined]
        edges.append(e)
        net_paths.append(([outline], [e]))

    # Diagrams: groups of steps joined by connections.
    uf_nodes = _UnionFind(len(nodes))
    for e in edges:
        uf_nodes.union(e.src, e.dst)
    groups: dict[int, list[int]] = defaultdict(list)
    for n in nodes:
        groups[uf_nodes.find(n.id)].append(n.id)
    comps = []
    for members in groups.values():
        es = [e for e in edges if e.src in members]
        if len(members) >= 3 and len(es) >= 2:
            comps.append(set(members))
    if not comps:
        return []
    # parts of one drawing that aren't connected to each other (or sit close) are one diagram
    merged: list[set[int]] = []
    for comp in comps:
        box = _union([nodes[i].bbox for i in comp])
        for m in merged:
            if _gap(box, _union([nodes[i].bbox for i in m])) <= 30:
                m |= comp
                break
        else:
            merged.append(set(comp))

    out: list[Diagram] = []
    for comp in merged:
        comp_nodes = [nodes[i] for i in sorted(comp)]
        # boxes that touch are table cells, not steps
        touching = sum(1 for a in comp_nodes if any(b is not a and _gap(a.bbox, b.bbox) < 2.0 for b in comp_nodes))
        if touching > 0.3 * len(comp_nodes):
            continue
        comp_edges = [e for e in edges if e.src in comp]
        wire_boxes = [_bbox_of(p) for paths, made in net_paths if any(e in comp_edges for e in made) for p in paths]
        region = _union([n.bbox for n in comp_nodes] + wire_boxes)
        # labelled boxes inside the drawing that nothing connects to are notes: each goes with the
        # step it sits beside
        note_nodes: list[Node] = []
        node_notes: dict[int, list[str]] = defaultdict(list)
        for n in nodes:
            if n.id in comp or n.kind == "offpage" or any(n.id in m for m in merged if m is not comp):
                continue
            w = max(0.0, min(n.bbox[2], region[2] + 2) - max(n.bbox[0], region[0] - 2))
            h = max(0.0, min(n.bbox[3], region[3] + 2) - max(n.bbox[1], region[1] - 2))
            if w * h >= 0.5 * _area(n.bbox) and n.label:
                note_nodes.append(n)
        for n in sorted(note_nodes, key=lambda n: (n.bbox[1], n.bbox[0])):
            # a note sits to the right of the step it explains, level with it
            beside = [m for m in comp_nodes if m.kind != "offpage" and m.bbox[2] <= n.bbox[0] + 2
                      and min(m.bbox[3], n.bbox[3]) - max(m.bbox[1], n.bbox[1]) > 0]
            near = min(beside or comp_nodes, key=lambda m: _gap(m.bbox, n.bbox))
            node_notes[near.id].append(n.label)
            region = _union([region, n.bbox])
        # lanes: shapes around some of these steps whose own text is a short title (the
        # innermost lane wins); a frame around more than this drawing is left alone. Text in a
        # lane that lies along a connector is that connector's label, not part of the title.
        all_paths = [p for paths, _ in net_paths for p in paths]
        lane_words: set[int] = set()
        for k in sorted(containers, key=lambda k: -_area(shapes[k][0])):
            b = shapes[k][0]
            title_words = [i for i in owned.get(k, [])
                           if not any(_poly_dist(_centre(words[i]), p) <= LABEL_NEAR for p in all_paths)]
            inside = [n for n in comp_nodes if _contains_box(b, n.bbox)]
            if inside and len(title_words) <= 10 and not any(
                    _contains_box(b, n.bbox) for n in nodes if n.id not in comp):
                title = _join_words([words[i] for i in title_words])
                if title:
                    for n in inside:
                        n.lane = title
                    region = _union([region, b])
                    lane_words.update(title_words)
        word_ids = {i for n in comp_nodes + note_nodes for i in n.words} | lane_words
        # loose text inside the drawing: labels of the wires it lies along, or notes
        # (text owned by a step is not loose; text owned by a frame or other container may be)
        loose = [(i, w) for i, w in enumerate(words)
                 if i not in word_ids and _contains_pt(region, _centre(w), pad=4)
                 and (owner.get(i) is None or owner[i] in containers)]
        notes: list[str] = []
        for phrase in _phrases(loose):
            ids = [i for i, _ in phrase]
            text = _join_words([w for _, w in phrase])
            box = _union([(float(w["x0"]), float(w["top"]), float(w["x1"]), float(w["bottom"])) for _, w in phrase])
            c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
            # the nearest connector; where one splits, the branch nearest the text
            best, rank = None, (LABEL_NEAR, math.inf)
            for e in comp_edges:
                paths = getattr(e, "_paths", None)
                if not paths:
                    continue
                d = min(min(_poly_dist(c, p), _box_dist(box, p[0]), _box_dist(box, p[-1])) for p in paths)
                r = (d, _poly_dist(c, e._branch) if getattr(e, "_branch", None) else math.inf)
                if d <= LABEL_NEAR and r < rank:
                    best, rank = e, r
            if not any(ch.isalnum() for ch in text):
                word_ids.update(ids)      # stray punctuation (a redacted word's leftovers)
                continue
            if best is not None and len(text) <= 40:
                best.label = f"{best.label} {text}".strip() if best.label else text
            else:
                notes.append(text)
            word_ids.update(ids)
        d = Diagram(bbox=region, nodes=comp_nodes, edges=comp_edges, notes=notes, word_ids=word_ids,
                    node_notes=dict(node_notes))
        d.order = _flow_order(d)
        out.append(d)
    for d in out:
        for e in d.edges:
            e.__dict__.pop("_paths", None)
            e.__dict__.pop("_branch", None)
    return out


def _flow_order(d: Diagram) -> list[int]:
    """Steps in the order the flow visits them: from the starts (nothing points at them), breadth
    first, ties broken top to bottom and left to right; anything not reached follows by position."""
    pos = {n.id: (round(n.bbox[1] / 4), n.bbox[0]) for n in d.nodes}
    nxt: dict[int, list[int]] = defaultdict(list)
    indeg: dict[int, int] = {n.id: 0 for n in d.nodes}
    for e in d.edges:
        nxt[e.src].append(e.dst)
        if e.directed:
            indeg[e.dst] += 1
        if not e.directed or e.both:
            nxt[e.dst].append(e.src)
    starts = sorted((n for n in indeg if indeg[n] == 0), key=pos.get)
    order: list[int] = []
    seen: set[int] = set()
    for root in starts + sorted(indeg, key=pos.get):
        if root in seen:
            continue
        queue = deque([root])
        seen.add(root)
        while queue:
            n = queue.popleft()
            order.append(n)
            for m in sorted(nxt[n], key=pos.get):
                if m not in seen:
                    seen.add(m)
                    queue.append(m)
    return order
