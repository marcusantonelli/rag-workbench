"""Shapes in Word documents: text boxes, panels and flowcharts drawn with Word's own shapes.

Word keeps these in DrawingML (``wps`` shapes, ``wpg`` groups, ``wpc`` drawing canvases), with a
VML copy for old readers inside ``mc:Fallback``; only the DrawingML is read, so nothing is read
twice. Documents that only have VML (written before Word 2010) give their text-box text.

A drawing whose shapes are joined by connectors (lines or arrows, glued to the shapes or ending on
them) becomes a diagram with the same step-by-step text as a flowchart rebuilt from a PDF's
vector shapes. Shapes that aren't joined give their text as paragraphs, in reading order, so the
words in a side panel or a call-out box are no longer lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "wps": "http://schemas.microsoft.com/office/word/2010/wordprocessingShape",
    "wpg": "http://schemas.microsoft.com/office/word/2010/wordprocessingGroup",
    "wpc": "http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas",
    "mc": "http://schemas.openxmlformats.org/markup-compatibility/2006",
    "v": "urn:schemas-microsoft-com:vml",
}
EMU_PT = 12700                  # EMU per point
ATTACH = 14 * EMU_PT            # a connector end this close to a shape is joined to it
LABEL_NEAR = 30 * EMU_PT        # a short loose text this close to a connector is its label

_CONNECTORS = {"line", "straightConnector1", "bentConnector2", "bentConnector3", "bentConnector4",
               "bentConnector5", "curvedConnector2", "curvedConnector3", "curvedConnector4", "curvedConnector5"}
_BLOCK_ARROWS = {"rightArrow": (1, 0), "leftArrow": (-1, 0), "downArrow": (0, 1), "upArrow": (0, -1),
                 "notchedRightArrow": (1, 0), "stripedRightArrow": (1, 0), "chevron": (1, 0)}
_DECISION = {"flowChartDecision", "diamond"}
_TERMINAL = {"flowChartTerminator", "ellipse", "flowChartAlternateProcess", "roundRect"}
_OFFPAGE = {"flowChartOffpageConnector", "flowChartConnector"}

Box = tuple[float, float, float, float]


def _q(tag: str) -> str:
    pre, name = tag.split(":")
    return f"{{{NS[pre]}}}{name}"


def _local(el) -> str:
    return el.tag.split("}")[-1] if isinstance(el.tag, str) else ""


@dataclass
class _Shape:
    id: str | None
    box: Box
    prst: str = ""
    text: str = ""
    connector: bool = False
    start: tuple[float, float] | None = None
    end: tuple[float, float] | None = None
    head: bool = False                 # arrowhead at the start
    tail: bool = False                 # arrowhead at the end
    st_id: str | None = None           # glued to these shapes
    end_id: str | None = None
    plain: bool = False                # no fill and no outline: loose text


@dataclass
class Drawing:
    """One drawing's shapes, in one coordinate space (EMU)."""
    shapes: list[_Shape] = field(default_factory=list)


# -- reading the XML ------------------------------------------------------------------------
class _Space:
    """Maps a group's child coordinates to its parent's."""

    def __init__(self, xfrm=None, parent: "_Space | None" = None):
        self.parent = parent
        self.off = self.ext = self.choff = self.chext = None
        if xfrm is not None:
            self.off = _pair(xfrm.find(_q("a:off")), "x", "y")
            self.ext = _pair(xfrm.find(_q("a:ext")), "cx", "cy")
            self.choff = _pair(xfrm.find(_q("a:chOff")), "x", "y")
            self.chext = _pair(xfrm.find(_q("a:chExt")), "cx", "cy")

    def map(self, x: float, y: float) -> tuple[float, float]:
        if self.off and self.ext and self.choff and self.chext and self.chext[0] and self.chext[1]:
            x = self.off[0] + (x - self.choff[0]) * self.ext[0] / self.chext[0]
            y = self.off[1] + (y - self.choff[1]) * self.ext[1] / self.chext[1]
        return self.parent.map(x, y) if self.parent else (x, y)

    def scale(self) -> tuple[float, float]:
        sx = sy = 1.0
        s: _Space | None = self
        while s is not None:
            if s.ext and s.chext and s.chext[0] and s.chext[1]:
                sx *= s.ext[0] / s.chext[0]
                sy *= s.ext[1] / s.chext[1]
            s = s.parent
        return sx, sy


def _pair(el, a: str, b: str) -> tuple[float, float] | None:
    if el is None:
        return None
    try:
        return float(el.get(a) or 0), float(el.get(b) or 0)
    except ValueError:
        return None


def _shape_text(sp, para_text: Callable) -> str:
    content = sp.find(f"{_q('wps:txbx')}/{_q('w:txbxContent')}")
    if content is None:
        return ""
    lines = [para_text(p).strip() for p in content.iter(_q("w:p"))]
    return "\n".join(t for t in lines if t)


def _arrow(ln, end: str) -> bool:
    el = ln.find(_q(f"a:{end}")) if ln is not None else None
    return el is not None and (el.get("type") or "none") != "none"


def _read_wsp(sp, space: _Space, origin: tuple[float, float], para_text: Callable, sid: str | None) -> _Shape | None:
    sppr = sp.find(_q("wps:spPr"))
    xfrm = sppr.find(_q("a:xfrm")) if sppr is not None else None
    off = _pair(xfrm.find(_q("a:off")), "x", "y") if xfrm is not None else None
    ext = _pair(xfrm.find(_q("a:ext")), "cx", "cy") if xfrm is not None else None
    if off is None or ext is None:
        return None
    x0, y0 = space.map(*off)
    sx, sy = space.scale()
    w, h = ext[0] * sx, ext[1] * sy
    x0, y0 = x0 + origin[0], y0 + origin[1]
    geom = sppr.find(_q("a:prstGeom"))
    prst = geom.get("prst", "") if geom is not None else ""
    ln = sppr.find(_q("a:ln"))
    cnv = sp.find(_q("wps:cNvCnPr"))
    connector = cnv is not None or prst in _CONNECTORS
    if sid is None:
        pr = sp.find(_q("wps:cNvPr"))
        sid = pr.get("id") if pr is not None else None
    shape = _Shape(id=sid, box=(x0, y0, x0 + w, y0 + h), prst=prst, text=_shape_text(sp, para_text), connector=connector)
    if connector:
        flip_h, flip_v = xfrm.get("flipH") in ("1", "true"), xfrm.get("flipV") in ("1", "true")
        shape.start = (x0 + (w if flip_h else 0), y0 + (h if flip_v else 0))
        shape.end = (x0 + (0 if flip_h else w), y0 + (0 if flip_v else h))
        shape.head, shape.tail = _arrow(ln, "headEnd"), _arrow(ln, "tailEnd")
        if cnv is not None:
            st, en = cnv.find(_q("a:stCxn")), cnv.find(_q("a:endCxn"))
            shape.st_id = st.get("id") if st is not None else None
            shape.end_id = en.get("id") if en is not None else None
    else:
        no_fill = sppr.find(_q("a:noFill")) is not None
        no_line = ln is not None and ln.find(_q("a:noFill")) is not None
        shape.plain = no_fill and (no_line or ln is None) and prst in ("", "rect")
    return shape


def _walk_group(el, space: _Space, origin, para_text, out: list[_Shape]) -> None:
    for child in el:
        name = _local(child)
        if name == "wsp":
            s = _read_wsp(child, space, origin, para_text, None)
            if s is not None:
                out.append(s)
        elif name == "grpSp":
            gpr = child.find(_q("wpg:grpSpPr"))
            xfrm = gpr.find(_q("a:xfrm")) if gpr is not None else None
            _walk_group(child, _Space(xfrm, space), origin, para_text, out)


def _anchor_origin(anchor, index: int) -> tuple[float, float]:
    """Where an anchored drawing sits. Offsets relative to the paragraph are placed one line per
    paragraph below the ones before (Word doesn't store paragraph positions)."""
    x = y = 0.0
    if _local(anchor) == "anchor":
        ph, pv = anchor.find(_q("wp:positionH")), anchor.find(_q("wp:positionV"))
        for pos, axis in ((ph, 0), (pv, 1)):
            if pos is None:
                continue
            off = pos.find(_q("wp:posOffset"))
            val = float(off.text) if off is not None and off.text and off.text.lstrip("-").isdigit() else 0.0
            if axis == 0:
                x = val
            else:
                y = val + (index * 14 * EMU_PT if pos.get("relativeFrom") in ("paragraph", "line") else 0)
    else:
        y = index * 14 * EMU_PT
    return x, y


def read_drawings(p, index: int, para_text: Callable) -> list[Drawing]:
    """The drawings anchored in a paragraph (DrawingML only; VML copies in mc:Fallback skipped)."""
    out: list[Drawing] = []
    for drawing in _iter_no_fallback(p, "drawing"):
        for anchor in drawing:
            if _local(anchor) not in ("anchor", "inline"):
                continue
            origin = _anchor_origin(anchor, index)
            pr = anchor.find(_q("wp:docPr"))
            doc_id = pr.get("id") if pr is not None else None
            data = anchor.find(f"{_q('a:graphic')}/{_q('a:graphicData')}")
            if data is None:
                continue
            d = Drawing()
            for el in data:
                name = _local(el)
                if name == "wsp":
                    s = _read_wsp(el, _Space(), origin, para_text, doc_id)
                    if s is not None:
                        d.shapes.append(s)
                elif name in ("wgp", "wpc"):
                    gpr = el.find(_q("wpg:grpSpPr"))
                    xfrm = gpr.find(_q("a:xfrm")) if gpr is not None else None
                    _walk_group(el, _Space(xfrm), origin, para_text, d.shapes)
            if d.shapes:
                out.append(d)
    return out


def vml_text(p, para_text: Callable) -> list[str]:
    """Text in VML text boxes of a paragraph with no DrawingML version (older documents)."""
    out: list[str] = []
    for pict in _iter_no_fallback(p, "pict"):
        for box in pict.iter(_q("w:txbxContent")):
            text = "\n".join(t for t in (para_text(q).strip() for q in box.iter(_q("w:p"))) if t)
            if text:
                out.append(text)
    return out


def _iter_no_fallback(el, name: str):
    """Descendants named ``name`` (w: namespace), not inside mc:Fallback and not inside a text box
    of another drawing (those are read with their own drawing)."""
    for child in el:
        tag = _local(child)
        if tag == "Fallback" or tag == "txbxContent":
            continue
        if tag == name and child.tag == _q(f"w:{name}"):
            yield child
            continue
        yield from _iter_no_fallback(child, name)


# -- making sense of the shapes ----------------------------------------------------------------
def _dist(b: Box, p: tuple[float, float]) -> float:
    dx = max(b[0] - p[0], 0, p[0] - b[2])
    dy = max(b[1] - p[1], 0, p[1] - b[3])
    return math.hypot(dx, dy)


def _seg_dist(p, a, b) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    vx, vy = bx - ax, by - ay
    t = 0.0 if vx == vy == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)))
    return math.hypot(px - (ax + t * vx), py - (ay + t * vy))


def to_diagram(shapes: list[_Shape]):
    """(diagram, the shapes it uses) when two or more shapes with text are joined by connectors;
    else None. Shapes no connector touches are left out (a note beside the flowchart is text)."""
    from rag.core.ingest.diagram import Diagram, Edge, Node, _flow_order

    labels = [s for s in shapes if s.plain and s.text and len(s.text.split()) <= 3]
    boxes = [s for s in shapes if not s.connector and s.prst not in _BLOCK_ARROWS and s not in labels
             and (s.text or s.prst in _OFFPAGE)]
    if len(boxes) < 2:
        return None
    nodes: list[Node] = []
    for i, s in enumerate(boxes):
        kind = ("decision" if s.prst in _DECISION else "terminal" if s.prst in _TERMINAL
                else "offpage" if s.prst in _OFFPAGE and not s.text else "box")
        nodes.append(Node(id=i, bbox=s.box, label=" ".join(s.text.split()), kind=kind))
    by_id = {s.id: i for i, s in enumerate(boxes) if s.id is not None}

    def nearest(p) -> int | None:
        best = min(range(len(boxes)), key=lambda i: _dist(boxes[i].box, p))
        return best if _dist(boxes[best].box, p) <= ATTACH else None

    edges: list[Edge] = []
    lines: list[tuple[Edge, tuple, tuple]] = []
    for s in shapes:
        if s.connector and s.start and s.end:
            a = by_id.get(s.st_id) if s.st_id else None
            b = by_id.get(s.end_id) if s.end_id else None
            a = a if a is not None else nearest(s.start)
            b = b if b is not None else nearest(s.end)
            if a is None or b is None or a == b:
                continue
            if s.tail or not s.head:
                e = Edge(src=a, dst=b, directed=s.tail or s.head, both=s.tail and s.head)
            else:
                e = Edge(src=b, dst=a)                    # arrowhead at the start only
            edges.append(e)
            lines.append((e, s.start, s.end))
        elif s.prst in _BLOCK_ARROWS:                    # a wide arrow shape between two boxes
            dx, dy = _BLOCK_ARROWS[s.prst]
            x0, y0, x1, y1 = s.box
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            tail = (cx - dx * (x1 - x0) / 2, cy - dy * (y1 - y0) / 2)
            tip = (cx + dx * (x1 - x0) / 2, cy + dy * (y1 - y0) / 2)
            a, b = nearest(tail), nearest(tip)
            if a is not None and b is not None and a != b:
                e = Edge(src=a, dst=b, label=" ".join(s.text.split()))
                edges.append(e)
                lines.append((e, tail, tip))
    if not edges:
        return None
    for lab in labels:                                   # "Yes" / "No" beside a connector
        c = ((lab.box[0] + lab.box[2]) / 2, (lab.box[1] + lab.box[3]) / 2)
        best = min(lines, key=lambda t: _seg_dist(c, t[1], t[2]), default=None)
        if best is not None and _seg_dist(c, best[1], best[2]) <= LABEL_NEAR and not best[0].label:
            best[0].label = lab.text.strip()
    joined = sorted({e.src for e in edges} | {e.dst for e in edges})
    renumber = {old: new for new, old in enumerate(joined)}
    nodes = [Node(id=renumber[n.id], bbox=n.bbox, label=n.label, kind=n.kind) for n in nodes if n.id in renumber]
    for e in edges:
        e.src, e.dst = renumber[e.src], renumber[e.dst]
    used = [boxes[i] for i in joined] + [s for s in shapes if s.connector or s.prst in _BLOCK_ARROWS]
    used += [lab for lab in labels if any(e.label == lab.text.strip() for e in edges)]
    bbox = (min(s.box[0] for s in used), min(s.box[1] for s in used),
            max(s.box[2] for s in used), max(s.box[3] for s in used))
    d = Diagram(bbox=bbox, nodes=nodes, edges=edges, notes=[], word_ids=set())
    d.order = _flow_order(d)
    return d, used


def loose_text(shapes: list[_Shape]) -> list[str]:
    """The text of shapes that aren't a diagram, top to bottom and left to right."""
    with_text = [s for s in shapes if s.text and not s.connector]
    with_text.sort(key=lambda s: (round(s.box[1] / (6 * EMU_PT)), s.box[0]))
    return [s.text for s in with_text]
