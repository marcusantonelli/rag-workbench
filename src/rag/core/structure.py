"""Document structure shared by the chunker and the spot file: step numbers in headings and
lists, and a document's identity (title, document number, revision).

Kept deliberately conservative: a pattern only counts when it is unambiguous, because a wrong
"procedure" is worse than none (a manual or a web page is read as plain sections).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

# "7.2 Oven Operation", "8.0 Electrode Replacement:", "Step 3 - Remove the cover", "Section 4.1 Scope"
_HEADING_NUMBER = re.compile(
    r"^\s*(?:(?i:step|section|chapter)\s+)?(\d{1,2}(?:\.\d{1,3}){0,4})(?:[.):]\s*|\s+[-–]?\s*)(?=\S)")
_MEASURE = re.compile(r"^\s*\d+(?:\.\d+)?\s*(?:mm|cm|m|in|°|%|v|a|w|kg|g|lb|psi|bar|s|ms|h|hz|khz|mhz|ghz)\b", re.I)


def heading_number(title: str) -> tuple[int, ...] | None:
    """(7, 2) for "7.2 Oven Operation", (8,) for "8.0 Electrode Replacement:", None when the
    heading isn't numbered (or starts with a measurement such as "1.5 mm gap")."""
    if not title or _MEASURE.match(title):
        return None
    m = _HEADING_NUMBER.match(title)
    if not m:
        return None
    parts = [int(p) for p in m.group(1).split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def is_prefix(a: tuple[int, ...], b: tuple[int, ...]) -> bool:
    """Is a a proper prefix of b (7 of 7.2)?"""
    return len(a) < len(b) and b[:len(a)] == a


def label(number: tuple[int, ...]) -> str:
    return ".".join(str(p) for p in number) if len(number) > 1 else f"{number[0]}.0"


# numbered list lines inside a section: "1. Open the valve", "2) Wait", "Step 3: Record"
LIST_STEP = re.compile(r"^\s*(?:[-•*]\s*)?(?:(?i:step)\s*)?(\d{1,2})[.):]\s+\S", re.M)

# warnings and notes that belong with the step they sit in
CAUTION = re.compile(r"\b(?:CAUTION|WARNING|DANGER)\b|^\s*(?:Caution|Warning|Danger)\s*:", re.M)
NOTE = re.compile(r"^\s*(?:\*\*)?(?:Note|NOTE|Important|IMPORTANT)\b\s*:?", re.M)

# words that make a numbered group a procedure rather than chapters of a manual
_PROCEDURAL_TITLE = re.compile(
    r"(?i)\b(procedure|instruction|operation|operating|steps?|replacement|removal|installation|install|"
    r"set-?up|start-?up|shut-?down|how to|recovery|calibration|cleaning|maintenance|inspection|"
    r"troubleshoot\w*|assembly|disassembly|loading|unloading|changeover|checklist)\b")
_IMPERATIVE = re.compile(
    r"(?im)^\s*(?:[-•*]\s*|\d{1,2}[.)]\s*)?(?:then,?\s+|first,?\s+)?(verify|ensure|check|confirm|press|push|"
    r"click|select|enter|type|open|close|turn|switch|set|place|position|align|insert|install|remove|"
    r"replace|connect|disconnect|load|unload|record|wait|apply|clean|wipe|inspect|measure|adjust|tighten|"
    r"loosen|use|attach|detach|lift|lower|mount|start|stop|run|log|disable|enable|power|move|hold|release|"
    r"rotate|slide|pull|scan|label|fill|drain|seal|weld|bake|cure|cut|mark|sign|notify|contact|go|repeat|"
    r"return|navigate|choose|save|print|reset|restart|calibrate|configure)\b")


def procedural_title(title: str | None) -> bool:
    return bool(title and _PROCEDURAL_TITLE.search(title))


def imperative_share(text: str) -> float:
    """The share of a text's lines that read as instructions ("Press …", "- Verify …")."""
    lines = [ln for ln in text.splitlines() if len(ln.split()) >= 2]
    if not lines:
        return 0.0
    return sum(1 for ln in lines if _IMPERATIVE.match(ln)) / len(lines)


# -- document identity ----------------------------------------------------------------------
@dataclass
class Identity:
    title: str | None = None
    number: str | None = None
    revision: str | None = None

    def line(self) -> str | None:
        """"Oven sealer operation (D02 rev A)"; None when nothing is known."""
        ident = ""
        if self.number:
            ident = self.number + (f" rev {self.revision}" if self.revision else "")
        elif self.revision:
            ident = f"rev {self.revision}"
        if self.title and ident:
            return f"{self.title} ({ident})"
        return self.title or ident or None


_NUMBER = re.compile(
    r"(?i:\b(?:specification|spec|document|doc|procedure|work\s+instruction|instruction|sop|drawing|dwg|form|part)"
    r"\s*(?:#|no\.?|number|num\.?|id)\s*[:.]?\s*)"
    r"((?=[A-Z0-9._/-]*\d)[A-Z0-9][A-Z0-9._/-]{1,30})(?![a-z])")
_REVISION = re.compile(r"(?i:\brev(?:ision)?\.?\s*(?:level\s*)?[:#]?\s*)([A-Z]{1,2}\d{0,2}|\d{1,3}(?:\.\d{1,2})?)(?![A-Za-z0-9])")
_TITLE = re.compile(r"(?im)^\s*(?:\*\*)?title\s*[:|]\s*(?:\*\*)?\s*([^|\n]{3,120}?)\s*(?:\||$)")
_FILE_NUMBER = re.compile(r"\b([A-Z]{1,5}[-_]?\d{2,7}(?:[-_][A-Z0-9]{1,6})*)\b")


def identity(title: str | None, texts: list[str], display_path: str = "") -> Identity:
    """Title, document number and revision from a document's title and its first passages
    (a title block such as "Title: … | Specification # D02 Revision: A"), with the file name's
    document number as the fallback."""
    head = "\n".join(t for t in texts if t)[:4000]
    out = Identity(title=(title or "").strip() or None)
    if out.title is None:
        m = _TITLE.search(head)
        if m and re.search(r"[A-Za-z]{3}", m.group(1)) and not re.match(r"(?i)\s*(specification|revision)\b", m.group(1)):
            out.title = m.group(1).strip()
    m = _NUMBER.search(head)
    if m:
        out.number = m.group(1).rstrip("._-/")
    m = _REVISION.search(head)
    if m:
        out.revision = m.group(1)
    if out.number is None and display_path:
        m = _FILE_NUMBER.search(PurePosixPath(display_path.replace("\\", "/")).stem.upper())
        if m:
            out.number = m.group(1)
    if out.title and out.title.lower() == (out.number or "").lower():
        out.title = None
    return out


# -- references ------------------------------------------------------------------------------
SECTION_REF = re.compile(r"(?i)\b(?:see|per|refer\s+to|go\s+to|continue\s+(?:to|with)|proceed\s+to)\s+(?:section\s+|step\s+)?"
                         r"(\d{1,2}(?:\.\d{1,3}){1,4})\b")
DOC_REF = re.compile(r"(?i:\b(?:see|per|refer\s+to|in|according\s+to|reference)\s+)"
                     r"(?:(?i:document|spec(?:ification)?|procedure|wi|sop|form)\s*#?\s*)?"
                     r"((?=[A-Z0-9-]*\d)[A-Z]{1,5}[- ]?\d{2,7}(?:-[A-Z0-9]{1,6})*)\b")
