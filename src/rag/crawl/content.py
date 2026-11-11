"""Media-type detection, quick text extraction, relevance and language checks."""

from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import unquote, urlsplit

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
_EXT_TYPES = {
    ".pdf": "pdf", ".docx": "docx", ".doc": "doc", ".txt": "txt", ".text": "txt", ".log": "txt",
    ".md": "md", ".markdown": "md", ".html": "html", ".htm": "html", ".shtml": "html", ".xhtml": "html",
    ".aspx": "html", ".php": "html", ".jsp": "html", ".csv": "csv", ".xlsx": "xlsx", ".xlsm": "xlsx",
    ".sql": "sql", ".cs": "cs", ".py": "py",
}
_CT_TYPES = {
    "application/pdf": "pdf",
    "application/x-pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel.sheet.macroenabled.12": "xlsx",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/csv": "csv",
    "text/markdown": "md",
    "text/x-markdown": "md",
    "application/sql": "sql",
    "text/x-python": "py",
    "text/x-csharp": "cs",
}
_SKIP_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp", ".ico", ".bmp", ".tif", ".tiff", ".mp4", ".mov",
    ".avi", ".webm", ".mp3", ".wav", ".zip", ".7z", ".rar", ".gz", ".tgz", ".exe", ".msi", ".dmg", ".iso",
    ".css", ".js", ".json", ".woff", ".woff2", ".ttf", ".eot", ".pptx", ".ppt", ".xls", ".rss", ".ics",
}


def url_ext(url: str) -> str:
    path = unquote(urlsplit(url).path).lower()
    dot = path.rfind(".")
    slash = path.rfind("/")
    return path[dot:] if dot > slash else ""


def type_from_url(url: str) -> str | None:
    return _EXT_TYPES.get(url_ext(url))


def obviously_unsupported(url: str) -> bool:
    return url_ext(url) in _SKIP_EXTS


def type_from_headers(url: str, content_type: str | None) -> str | None:
    """Best guess before reading the body; None means unsupported."""
    ct = (content_type or "").split(";")[0].strip().lower()
    by_ext = type_from_url(url)
    if ct in _CT_TYPES:
        t = _CT_TYPES[ct]
        # Servers often send code and markdown as text/plain or octet-stream; the
        # extension decides then. An HTML content type on a .pdf URL is an error page.
        if t == "html" and by_ext in ("pdf", "docx", "doc", "xlsx", "csv"):
            return "html"
        return t
    if ct.startswith("text/"):
        if by_ext in ("txt", "md", "csv", "sql", "cs", "py", "html"):
            return by_ext
        if ct == "text/plain":
            return "txt"
        return None
    if ct in ("", "application/octet-stream", "binary/octet-stream", "application/download",
              "application/force-download", "application/x-download"):
        return by_ext or ("html" if ct == "" else None)
    if ct.startswith(("image/", "video/", "audio/", "font/")):
        return None
    return by_ext if by_ext in ("sql", "cs", "py", "md", "csv") else None


def sniff_type(data: bytes, guess: str | None) -> str | None:
    """Confirm the type from the bytes; None means unsupported or mismatched."""
    head = data[:2048].lstrip()
    if data[:5] == b"%PDF-" or b"%PDF-" in data[:1024]:
        return "pdf"
    if data[:4] == b"PK\x03\x04":
        try:
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
        except zipfile.BadZipFile:
            return None
        if "word/document.xml" in names:
            return "docx"
        if "xl/workbook.xml" in names:
            return "xlsx"
        return None
    if data[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
        return "doc" if guess == "doc" else None
    if b"\x00" in data[:4096]:
        return None
    low = head[:512].lower()
    if guess in ("html", None, "txt") and (low.startswith(b"<!doctype html") or low.startswith(b"<html")
                                          or b"<html" in low[:300] or b"<body" in low):
        return "html"
    if guess == "html":
        return "html" if b"<" in low else "txt"
    return guess or "txt"


# ---------------------------------------------------------------------------
# Text for relevance
# ---------------------------------------------------------------------------
@dataclass
class Parsed:
    text: str = ""
    title: str = ""
    words: int = 0
    links: list[tuple[str, str]] = field(default_factory=list)   # (href, anchor text)
    canonical: str | None = None
    noindex: bool = False
    nofollow: bool = False
    pages: int | None = None
    broken: bool = False
    note: str = ""


_DROP_TAGS = ["script", "style", "noscript", "nav", "footer", "aside", "form", "iframe", "svg", "template", "header"]


def parse_html(data: bytes, max_chars: int = 200_000) -> Parsed:
    from bs4 import BeautifulSoup, UnicodeDammit

    markup = UnicodeDammit(data, is_html=True).unicode_markup or ""
    try:
        soup = BeautifulSoup(markup, "lxml")
    except Exception:  # noqa: BLE001 - lxml not installed
        soup = BeautifulSoup(markup, "html.parser")
    p = Parsed()
    if soup.title and soup.title.string:
        p.title = " ".join(soup.title.string.split())
    for meta in soup.find_all("meta", attrs={"name": re.compile("^robots$", re.I)}):
        content = (meta.get("content") or "").lower()
        p.noindex |= "noindex" in content or "none" in content
        p.nofollow |= "nofollow" in content or "none" in content
    canon = soup.find("link", rel=lambda v: v and "canonical" in (v if isinstance(v, list) else [v]))
    if canon and canon.get("href"):
        p.canonical = canon["href"].strip()
    for a in soup.find_all("a", href=True):
        rel = a.get("rel") or []
        if "nofollow" in rel:
            continue
        p.links.append((a["href"].strip(), " ".join(a.get_text(" ").split())[:200]))
    for tag in soup.find_all(_DROP_TAGS):
        tag.decompose()
    root = soup.find("main") or soup.find("article") or soup.body or soup
    p.text = " ".join(root.get_text(" ").split())[:max_chars]
    p.words = len(p.text.split())
    return p


def parse_file(data: bytes, ftype: str, max_pages: int = 8, max_chars: int = 200_000) -> Parsed:
    p = Parsed()
    try:
        if ftype == "pdf":
            import pypdfium2 as pdfium

            pdf = pdfium.PdfDocument(data)
            try:
                p.pages = len(pdf)
                parts = []
                meta_title = (pdf.get_metadata_dict() or {}).get("Title") or ""
                p.title = meta_title.strip()
                idx = list(range(min(p.pages, max_pages)))
                if p.pages > max_pages:
                    idx.append(p.pages - 1)
                for i in idx:
                    page = pdf[i]
                    tp = page.get_textpage()
                    parts.append(tp.get_text_range())
                    tp.close()
                    page.close()
            finally:
                pdf.close()
            p.text = " ".join(" ".join(parts).split())[:max_chars]
        elif ftype == "docx":
            import docx

            d = docx.Document(io.BytesIO(data))
            parts = [para.text for para in d.paragraphs]
            for t in d.tables[:20]:
                for row in t.rows:
                    parts.append(" ".join(c.text for c in row.cells))
            p.title = (d.core_properties.title or "").strip()
            p.text = " ".join(" ".join(parts).split())[:max_chars]
        elif ftype == "xlsx":
            import openpyxl

            wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
            parts: list[str] = []
            for ws in wb.worksheets[:5]:
                parts.append(ws.title)
                for n, row in enumerate(ws.iter_rows(values_only=True)):
                    parts.extend(str(v) for v in row if isinstance(v, str))
                    if n > 500:
                        break
            wb.close()
            p.text = " ".join(" ".join(parts).split())[:max_chars]
        elif ftype == "doc":
            runs = re.findall(rb"[\x20-\x7e]{4,}", data[:4_000_000])
            p.text = " ".join(r.decode("ascii", "ignore") for r in runs)[:max_chars]
        elif ftype == "html":
            return parse_html(data, max_chars)
        else:
            from rag.core.util import decode_bytes

            p.text = decode_bytes(data[:2_000_000])[0][:max_chars]
    except Exception as exc:  # noqa: BLE001 - unreadable file
        p.broken = True
        p.note = f"{type(exc).__name__}: {exc}"[:200]
    p.words = len(p.text.split())
    return p


# ---------------------------------------------------------------------------
# Relevance and language
# ---------------------------------------------------------------------------
def _term_re(term: str) -> re.Pattern[str]:
    parts = [re.escape(w) for w in term.lower().split()]
    return re.compile(r"(?<![a-z0-9])" + r"[\s\-_/+.]*".join(parts) + r"(?![a-z0-9])")


class Relevance:
    """terms: the subject (needed min_hits times); also_required: a second subject that
    must also appear (e.g. an edition next to a product name); related: score only."""

    def __init__(self, terms: list[str], related: list[str], min_hits: int,
                 also_required: list[str] | None = None, also_min_hits: int = 1):
        self.terms = [(t, _term_re(t)) for t in terms]
        self.related = [(t, _term_re(t)) for t in related]
        self.also = [(t, _term_re(t)) for t in (also_required or [])]
        self.min_hits = int(min_hits)
        self.also_min = int(also_min_hits)

    @staticmethod
    def _norm(text: str) -> str:
        return unquote(text).lower()

    def hits(self, text: str) -> tuple[int, int]:
        low = self._norm(text)
        strong = sum(len(r.findall(low)) for _, r in self.terms)
        weak = sum(len(r.findall(low)) for _, r in self.related)
        return strong, weak

    def also_hits(self, text: str) -> int:
        low = self._norm(text)
        return sum(len(r.findall(low)) for _, r in self.also)

    def link_score(self, url: str, anchor: str = "") -> float:
        parts = urlsplit(url)
        text = f"{parts.path} {parts.query} {anchor}"
        strong, weak = self.hits(text)
        score = 10.0 * min(strong, 3) + 2.0 * min(weak, 5)
        if self.also and strong:
            score += 6.0 * min(self.also_hits(text), 2)
        return score

    def content_ok(self, title: str, text: str) -> tuple[bool, float]:
        s_title, w_title = self.hits(title)
        s_text, w_text = self.hits(text)
        strong = 2 * s_title + s_text
        score = strong + 0.25 * (w_title + w_text)
        ok = strong >= self.min_hits
        if self.also:
            also = 2 * self.also_hits(title) + self.also_hits(text)
            ok = ok and also >= self.also_min
            score += 0.5 * min(also, 20)
        return ok, round(score, 2)


_EN_STOP = set("""the of and to in is that for it as with was on be by this are or from at an which
have not can has but its also these their will more other used such when into than may been one
all were they our you your we if there use using can how what should each only both between""".split())


# Common words of other Latin-script languages that are not English words.
_FOREIGN_STOP = set("""der das und ist nicht mit für auf dem den des ein eine einer von zu im sich auch werden
wird oder bei über unter le la les et une pour dans qui sur par pas avec sont du au aux ce cette être
el los las del que una por con para como más pero sus está al il lo gli della delle che è nel alla
degli os da do das dos em uma não são pelo pela het een van zijn voor niet ook aan wordt och att för
är inte på som med til af og ikke af""".split())


def looks_english(text: str) -> bool:
    """False only when the text is clearly another language.

    Terse English (datasheets, spec tables, code-like text) has few function words
    of any language, so it passes; a page needs more foreign than English common
    words, or mostly non-Latin script, to be rejected.
    """
    words = re.findall(r"[a-zA-ZÀ-ɏͰ-ϿЀ-ӿ぀-ヿ一-鿿]+", text[:20000])
    if len(words) < 40:
        return True   # too little text to judge
    latin = [w for w in words if all(ord(c) < 0x250 for c in w)]
    if len(latin) < 0.6 * len(words):
        return False
    lows = [w.lower() for w in latin]
    english = sum(1 for w in lows if w in _EN_STOP)
    foreign = sum(1 for w in lows if w in _FOREIGN_STOP)
    return not (foreign >= 8 and foreign > 1.5 * english)


_LANGS = set("""ar bg bs ca cs cy da de el es et eu fa fi fr ga gl he hi hr hu hy id is it ja ka kk ko
lt lv mk ms mt nb nl nn no pl pt ro ru sk sl sq sr sv sw ta th tr uk ur uz vi zh""".split())
_LOCALE_SEG = re.compile(r"^([a-z]{2})(?:[-_]([a-z]{2}|hans|hant))?$", re.I)


def url_language(url: str) -> str | None:
    """A language code found in the URL's locale path segment or query, if any."""
    parts = urlsplit(url)
    for seg in [s for s in parts.path.split("/") if s][:4]:
        m = _LOCALE_SEG.match(seg)
        if m and (m.group(1).lower() in _LANGS or m.group(1).lower() == "en"):
            return m.group(1).lower()
    q = re.search(r"(?:^|&)(?:lang|locale|language|hl)=([a-z]{2})", parts.query, re.I)
    return q.group(1).lower() if q else None


def summarize_types(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for r in rows:
        t = out.setdefault(r["type"], {"files": 0, "bytes": 0})
        t["files"] += 1
        t["bytes"] += r["bytes"]
    return out
