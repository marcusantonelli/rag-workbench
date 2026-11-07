"""rag-crawl against a fake web: two local HTTP servers with every situation the
real crawl meets (robots, sitemaps, redirects, login walls, rate limits,
duplicates, other languages, off-topic pages, external links)."""

from __future__ import annotations

import gzip
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from rag.crawl.config import load_config
from rag.crawl.crawler import Crawler

WORDS = ("Opcenter Execution manages work in process, genealogy and holds on the shop floor. "
         "The Camstar heritage brings modeling of products, specifications and workflows. ")


def article(title: str, body: str, links: list[tuple[str, str]] = (), n: int = 12) -> bytes:
    a = "".join(f'<li><a href="{h}">{t}</a></li>' for h, t in links)
    paras = "".join(f"<p>{body} Paragraph {i} explains how the system is used by the team.</p>" for i in range(n))
    return (f"<!doctype html><html><head><title>{title}</title></head><body>"
            f"<nav><a href='/'>Home</a></nav><main><h1>{title}</h1>{paras}<ul>{a}</ul></main>"
            f"<footer>Copyright</footer></body></html>").encode()


def pdf_bytes(lines: list[str]) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter, pageCompression=0)
    y = 740
    for line in lines:
        c.drawString(60, y, line)
        y -= 16
        if y < 60:
            c.showPage()
            y = 740
    c.save()
    return buf.getvalue()


def scanned_pdf() -> bytes:
    from PIL import Image, ImageDraw
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas

    img = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(img)
    for i in range(30):
        d.line([(20, 30 + i * 25), (580, 30 + i * 25)], fill=(i * 7 % 255, 0, 0), width=2)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    c.drawImage(ImageReader(img), 0, 0, width=612, height=792)
    c.save()
    return buf.getvalue()


def docx_bytes() -> bytes:
    import docx

    d = docx.Document()
    d.add_heading("Opcenter work instruction", 1)
    for i in range(8):
        d.add_paragraph(WORDS + f"Step {i} of the Camstar container move procedure.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def xlsx_bytes() -> bytes:
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Opcenter object", "Camstar name", "Notes"])
    for i in range(40):
        ws.append([f"Opcenter spec {i}", f"Camstar spec {i}", "used in the MES model"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


FRENCH = ("Opcenter Execution est le système de gestion de la production de Siemens. Les opérateurs de "
          "l'atelier utilisent Opcenter pour suivre les lots, les arrêts et la généalogie des produits "
          "dans les usines du groupe. ")


class Site:
    """A tiny programmable web server that records every request."""

    def __init__(self, host: str, port: int = 0):
        self.routes: dict[str, object] = {}
        self.log: list[tuple[str, float, dict]] = []
        self.hits: dict[str, int] = {}
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                site.log.append((self.path, time.monotonic(), dict(self.headers)))
                site.hits[self.path] = site.hits.get(self.path, 0) + 1
                route = site.routes.get(self.path)
                if callable(route):
                    route = route(site.hits[self.path])
                if route is None:
                    route = (404, {"Content-Type": "text/html"}, b"<html><body>not found</body></html>")
                status, headers, body = route
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                if "Content-Length" not in headers:
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.port = self.server.server_address[1]
        self.base = f"http://{host}:{self.port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def html(self, path: str, body: bytes) -> None:
        self.routes[path] = (200, {"Content-Type": "text/html; charset=utf-8"}, body)

    def file(self, path: str, ctype: str, body: bytes, **headers) -> None:
        self.routes[path] = (200, {"Content-Type": ctype, **headers}, body)

    def paths(self) -> list[str]:
        return [p for p, _, _ in self.log]

    def close(self) -> None:
        self.server.shutdown()


@pytest.fixture()
def web():
    a = Site("127.0.0.1")
    b = Site("localhost")
    A, B = a.base, b.base
    a.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"},
                               f"User-agent: *\nDisallow: /private/\nSitemap: {A}/sitemap_index.xml\n".encode())
    b.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"}, b"User-agent: *\nAllow: /\n")
    a.file("/sitemap_index.xml", "application/xml",
           f'<?xml version="1.0"?><sitemapindex><sitemap><loc>{A}/sitemap-posts.xml.gz</loc></sitemap></sitemapindex>'.encode())
    a.file("/sitemap-posts.xml.gz", "application/gzip", gzip.compress(
        (f'<?xml version="1.0"?><urlset><url><loc>{A}/blog/opcenter-sitemap-only-post/</loc></url>'
         f'<url><loc>{A}/blog/unrelated-post/</loc></url>'
         f'<url><loc>{A}/de-DE/opcenter-uebersicht/</loc></url></urlset>').encode()))

    links = [
        ("/blog/opcenter-execution-overview/", "Overview"),
        ("/blog/opcenter-execution-overview/?utm_source=newsletter", "Overview again"),
        ("/blog/?p=12", "Short link"),
        ("/blog/copy-of-overview/", "Copy"),
        ("/blog/cooking-pasta/", "Pasta recipes"),
        ("/tag/opcenter/", "Tag"),
        ("/private/opcenter-secret.pdf", "Secret"),
        ("/support/opcenter-manual.pdf", "Manual"),
        ("/downloads/opcenter-factsheet.pdf", "Fact sheet"),
        ("/downloads/opcenter-factsheet-copy.pdf", "Fact sheet copy"),
        ("/downloads/scanned-opcenter.pdf", "Scanned"),
        ("/downloads/brochure.pdf", "Brochure"),
        ("/downloads/opcenter-huge.pdf", "Huge"),
        ("/downloads/opcenter-wi.docx", "Work instruction"),
        ("/downloads/opcenter-config.xlsx", "Config"),
        ("/downloads/opcenter-hold-codes.csv", "Hold codes"),
        ("/code/OpcenterClient.cs", "Client"),
        ("/code/camstar_export.py", "Export script"),
        ("/code/opcenter_views.sql", "Views"),
        ("/images/opcenter-diagram.png", "Diagram"),
        ("/downloads/opcenter-slides.pptx", "Slides"),
        ("/flaky/opcenter-rate-limited.pdf", "Rate limited"),
        ("/fr/opcenter-presentation/", "Français"),
        ("/blog/opcenter-francais/", "Article"),
        (f"{B}/opcenter-partner-article/", "Partner article"),
        (f"{B}/random/", "Random external"),
        ("https://www.scribd.com/document/1/opcenter-guide", "Scribd copy"),
        ("mailto:someone@example.com", "Mail"),
    ]
    a.html("/", article("Opcenter home", WORDS, links))
    overview = article("Opcenter Execution overview", WORDS, [("/blog/opcenter-quality-notes/", "Quality notes")])
    a.html("/blog/opcenter-execution-overview/", overview)
    a.routes["/blog/?p=12"] = (301, {"Location": f"{A}/blog/opcenter-execution-overview/"}, b"")
    a.html("/blog/copy-of-overview/", overview.replace(b"<nav><a href='/'>Home</a></nav>", b"<nav>Other nav</nav>"))
    a.html("/blog/cooking-pasta/", article(
        "Pasta", "Boil water, add salt and cook the pasta for ten minutes before you drain it. ",
        [("/blog/deep-unrelated/", "More"), ("/docs/camstar-guide.pdf", "Guide")]))
    a.html("/blog/deep-unrelated/", article("Deep", "Nothing to see here at all, just words. "))
    a.file("/docs/camstar-guide.pdf", "application/pdf",
           pdf_bytes([f"Camstar guide line {i}: Opcenter modeling of specs and workflows." for i in range(60)]))
    a.html("/tag/opcenter/", article("Tag opcenter", WORDS, [("/blog/opcenter-quality-notes/", "Quality")], n=1))
    a.html("/blog/opcenter-quality-notes/", article("Opcenter Quality notes", WORDS))
    a.html("/blog/opcenter-sitemap-only-post/", article("Opcenter from sitemap", WORDS))
    a.html("/blog/unrelated-post/", article("Unrelated", "Gardening in spring. "))
    a.file("/private/opcenter-secret.pdf", "application/pdf", pdf_bytes(["secret opcenter"] * 60))
    a.routes["/support/opcenter-manual.pdf"] = (302, {"Location": f"{A}/login?next=/support/opcenter-manual.pdf"}, b"")
    factsheet = pdf_bytes([f"Opcenter Execution fact sheet line {i}: MES capabilities for discrete and process "
                           f"industries with Camstar heritage." for i in range(80)])
    a.file("/downloads/opcenter-factsheet.pdf", "application/pdf", factsheet)
    a.file("/downloads/opcenter-factsheet-copy.pdf", "application/octet-stream", factsheet)
    a.file("/downloads/scanned-opcenter.pdf", "application/pdf", scanned_pdf())
    a.file("/downloads/brochure.pdf", "application/pdf",
           pdf_bytes([f"Garden furniture brochure line {i}, teak and steel." for i in range(80)]))
    a.file("/downloads/opcenter-huge.pdf", "application/pdf", b"%PDF-1.4\n" + b"0" * 300_000)
    a.file("/downloads/opcenter-wi.docx",
           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", docx_bytes())
    a.file("/downloads/opcenter-config.xlsx",
           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", xlsx_bytes())
    a.file("/downloads/opcenter-hold-codes.csv", "text/csv",
           ("code,description\n" + "".join(f"H{i},Opcenter hold reason {i} from Camstar\n" for i in range(60))).encode())
    a.file("/code/OpcenterClient.cs", "text/plain", (
        "// Client for the Opcenter Execution web services (Camstar API)\n"
        + "".join(f"public class OpcenterThing{i} {{ public string Camstar{i} => \"x\"; }}\n" for i in range(60))).encode())
    a.file("/code/camstar_export.py", "application/octet-stream", (
        '"""Export Camstar containers into Opcenter."""\n'
        + "".join(f"def export_{i}():\n    return 'opcenter camstar {i}'\n" for i in range(60))).encode())
    a.file("/code/opcenter_views.sql", "application/sql", (
        "-- Opcenter reporting views over the Camstar ODS\n"
        + "".join(f"CREATE VIEW v{i} AS SELECT 'opcenter' AS a;\n" for i in range(60))).encode())
    a.file("/images/opcenter-diagram.png", "image/png", b"\x89PNG....")
    a.file("/downloads/opcenter-slides.pptx", "application/vnd.ms-powerpoint", b"PK\x03\x04....")
    flaky_pdf = pdf_bytes([f"Opcenter rate limited document line {i}, Camstar too." for i in range(60)])
    a.routes["/flaky/opcenter-rate-limited.pdf"] = lambda n: (
        (429, {"Retry-After": "1", "Content-Type": "text/plain"}, b"slow down") if n == 1
        else (200, {"Content-Type": "application/pdf"}, flaky_pdf))
    a.html("/fr/opcenter-presentation/", article("Opcenter FR", FRENCH))
    a.html("/blog/opcenter-francais/", article("Opcenter en France", FRENCH * 3))
    b.html("/opcenter-partner-article/", article("Partner on Opcenter", WORDS,
                                                 [(f"{B}/opcenter-partner-2/", "Second"),
                                                  (f"{A}/blog/opcenter-quality-notes/", "Back home")]))
    b.html("/opcenter-partner-2/", article("Partner 2 Opcenter", WORDS))
    b.html("/random/", article("Random", "Random words. "))
    yield a, b
    a.close()
    b.close()


def make_cfg(a, **crawl):
    over = {
        "topic": {"name": "test", "terms": ["opcenter", "camstar"], "related": ["mes"], "min_term_hits": 2},
        "crawl": {"seeds": [a.base + "/"], "allowed_domains": ["127.0.0.1"], "delay_s": 0.05, "workers": 3,
                  "timeout_s": 5, "max_file_mb": 0.2, "min_file_kb": 0.5, "html_min_words": 100, **crawl},
    }
    return load_config(None, over)


def run(cfg, out, budget=50_000_000, hard=None, **kw):
    c = Crawler(cfg, out, budget=budget, hard_cap=hard or budget, log=lambda m: None, **kw)
    try:
        return c.run(), c
    finally:
        c.close()


def saved_paths(out: Path) -> set[str]:
    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    from urllib.parse import urlsplit
    return {urlsplit(r["url"]).path for r in rows}


def test_full_crawl_behaviour(web, tmp_path):
    a, b = web
    summary, _ = run(make_cfg(a), tmp_path / "out")
    saved = saved_paths(tmp_path / "out")
    expected = {
        "/", "/blog/opcenter-execution-overview/", "/blog/opcenter-quality-notes/",
        "/blog/opcenter-sitemap-only-post/", "/docs/camstar-guide.pdf",
        "/downloads/opcenter-factsheet.pdf", "/downloads/scanned-opcenter.pdf",
        "/downloads/opcenter-wi.docx", "/downloads/opcenter-config.xlsx", "/downloads/opcenter-hold-codes.csv",
        "/code/OpcenterClient.cs", "/code/camstar_export.py", "/code/opcenter_views.sql",
        "/flaky/opcenter-rate-limited.pdf", "/opcenter-partner-article/",
    }
    assert saved == expected, (saved - expected, expected - saved)
    assert "frontier exhausted" in summary.stop_reason

    pa, pb = a.paths(), b.paths()
    # robots, login walls, languages, wrong types and blocked domains are never requested
    for never in ("/private/opcenter-secret.pdf", "/login?next=%2Fsupport%2Fopcenter-manual.pdf",
                  "/images/opcenter-diagram.png", "/downloads/opcenter-slides.pptx",
                  "/fr/opcenter-presentation/", "/de-DE/opcenter-uebersicht/", "/blog/unrelated-post/"):
        assert never not in pa, never
    assert not any(p.startswith("/login") for p in pa)
    # focus: one hop of off-topic tunnelling, not two
    assert "/blog/cooking-pasta/" in pa and "/blog/deep-unrelated/" not in pa
    # external: one hop only, and only for links that name a term
    assert "/opcenter-partner-article/" in pb and "/opcenter-partner-2/" not in pb and "/random/" not in pb
    # each URL fetched once, except the retried 429 and the tracking-parameter variant collapsed
    assert a.hits["/flaky/opcenter-rate-limited.pdf"] == 2
    assert a.hits.get("/blog/opcenter-execution-overview/?utm_source=newsletter") is None
    assert a.hits["/blog/opcenter-execution-overview/"] == 1
    # politeness: requests to one host never overlap and keep the delay
    times = sorted(t for p, t, _ in a.log if not p.startswith(("/robots", "/sitemap")))
    gaps = [t2 - t1 for t1, t2 in zip(times, times[1:])]
    assert min(gaps) >= 0.045, min(gaps)
    assert all(h.get("User-Agent", "").startswith("rag-crawl/") for _, _, h in a.log)

    statuses = {r[0]: r[1] for r in summary.skip_reasons}
    assert statuses.get("robots") == 1 and statuses.get("login_required") == 1
    assert statuses.get("too_large (300.01 KB)") == 1
    assert statuses.get("off_topic") == 1             # brochure.pdf
    # files on disk match the manifest, named under their host
    files = [f for f in (tmp_path / "out" / "files").rglob("*") if f.is_file()]
    assert len(files) == len(expected)
    assert any(f.name.startswith("opcenter-factsheet-") and f.suffix == ".pdf" for f in files)


def test_duplicates_and_reasons(web, tmp_path):
    a, _ = web
    run(make_cfg(a), tmp_path / "out")
    import sqlite3

    db = sqlite3.connect(tmp_path / "out" / "crawl.db")
    reason = dict(db.execute("SELECT url, reason FROM urls").fetchall())
    A = a.base
    assert reason[f"{A}/downloads/opcenter-factsheet-copy.pdf"] == "duplicate_content" or \
        reason[f"{A}/downloads/opcenter-factsheet.pdf"] == "duplicate_content"
    assert reason[f"{A}/blog/copy-of-overview/"] == "duplicate_content" or \
        reason[f"{A}/blog/opcenter-execution-overview/"] == "duplicate_content"
    assert "listing_page" in reason[f"{A}/tag/opcenter/"]
    assert "not_english" in reason[f"{A}/blog/opcenter-francais/"]
    assert "off_topic" in reason[f"{A}/blog/cooking-pasta/"]
    assert reason[f"{A}/blog/?p=12"] == "redirect 301"


def test_budget_is_respected_and_resume_continues(web, tmp_path):
    a, _ = web
    out = tmp_path / "out"
    s1, _ = run(make_cfg(a, workers=1), out, budget=20_000, hard=25_000)
    assert 0 < s1.saved_bytes <= 25_000
    assert "target" in s1.stop_reason or s1.saved_bytes >= 20_000
    first = saved_paths(out)
    hits_before = dict(a.hits)

    s2, _ = run(make_cfg(a, workers=1), out, budget=50_000_000)
    assert s2.saved_bytes > s1.saved_bytes
    assert first <= saved_paths(out)
    for path in first:   # nothing saved in run 1 is downloaded again
        assert a.hits.get(path, 0) == hits_before.get(path, 0) or path == "/"
    assert "frontier exhausted" in s2.stop_reason


def test_hard_cap_never_exceeded_with_parallel_workers(web, tmp_path):
    a, _ = web
    s, _ = run(make_cfg(a, workers=4), tmp_path / "out", budget=60_000, hard=70_000)
    assert s.saved_bytes <= 70_000


def test_search_provider_seeds(web, tmp_path, monkeypatch):
    a, b = web
    A = a.base
    a.file("/hidden/opcenter-only-in-search.pdf", "application/pdf",
           pdf_bytes([f"Opcenter document found only by search, line {i}, Camstar." for i in range(60)]))
    b.routes["/brave?q=Opcenter+pdf&count=20&offset=0"] = (200, {"Content-Type": "application/json"}, json.dumps(
        {"web": {"results": [{"url": f"{A}/hidden/opcenter-only-in-search.pdf", "title": "Opcenter doc",
                              "description": "Opcenter Camstar"}]}}).encode())
    monkeypatch.setenv("BRAVE_API_KEY", "test-key")
    cfg = make_cfg(a)
    cfg["search"].update({"provider": "brave", "brave_url": f"{b.base}/brave", "queries": ["Opcenter pdf"],
                          "pages_per_query": 1, "delay_s": 0})
    run(cfg, tmp_path / "out")
    assert "/hidden/opcenter-only-in-search.pdf" in saved_paths(tmp_path / "out")
    brave_req = next(h for p, _, h in b.log if p.startswith("/brave"))
    assert brave_req.get("X-Subscription-Token") == "test-key"


def test_size_parsing_and_cli_report(web, tmp_path, capsys):
    from rag.crawl.cli import main
    from rag.crawl.config import parse_size

    assert parse_size("15GB") == 15_000_000_000 and parse_size("2 GiB") == 2 * 1024**3
    a, _ = web
    cfg_path = tmp_path / "t.toml"
    cfg_path.write_text(
        '[topic]\nterms = ["opcenter", "camstar"]\n[crawl]\n'
        f'seeds = ["{a.base}/"]\nallowed_domains = ["127.0.0.1"]\ndelay_s = 0.02\nmin_file_kb = 0.5\n'
        'html_min_words = 100\nmax_file_mb = 0.2\n[sitemaps]\nenabled = false\n', encoding="utf-8")
    assert main(["run", str(cfg_path), str(tmp_path / "o"), "--budget", "40KB", "--quiet"]) == 0
    assert main(["report", str(tmp_path / "o"), "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "Saved" in out and "Highest-scoring files" in out


def test_unreachable_host_is_dropped_quickly_with_a_clear_reason(tmp_path):
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()   # nothing listens here
    cfg = load_config(None, {
        "topic": {"terms": ["opcenter"]},
        "crawl": {"seeds": [f"http://127.0.0.1:{port}/opcenter/", f"http://127.0.0.1:{port}/opcenter/b.pdf"],
                  "allowed_domains": ["127.0.0.1"], "delay_s": 0.01, "timeout_s": 2,
                  "retry_unreachable_s": 0.05},
        "sitemaps": {"enabled": False},
    })
    started = time.monotonic()
    summary, c = run(cfg, tmp_path / "out", max_requests=50)
    assert time.monotonic() - started < 20
    assert "no site could be reached" in summary.stop_reason
    import sqlite3

    rows = sqlite3.connect(tmp_path / "out" / "crawl.db").execute("SELECT status, reason FROM urls").fetchall()
    assert rows and all(st == "failed" and "unreachable" in reason for st, reason in rows)
    assert all("robots" != reason for _, reason in rows)
    # The next run (network back) tries those hosts again instead of treating them as done.
    from rag.crawl.state import CrawlState

    st = CrawlState(tmp_path / "out" / "crawl.db")
    try:
        assert st.status_counts() == {"failed": len(rows)}     # a report changes nothing
        assert st.retry_unreachable() == len(rows)
        assert st.status_counts() == {"queued": len(rows)}
    finally:
        st.close()


def test_site_down_on_first_run_is_collected_on_the_next(tmp_path):
    import socket

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    cfg = load_config(None, {
        "topic": {"terms": ["opcenter"]},
        "crawl": {"seeds": [f"http://127.0.0.1:{port}/opcenter/"], "allowed_domains": ["127.0.0.1"],
                  "delay_s": 0.01, "timeout_s": 2, "retry_unreachable_s": 0.05},
    })
    first, _ = run(cfg, tmp_path / "out")
    assert first.saved_files == 0 and "no site could be reached" in first.stop_reason

    site = Site("127.0.0.1", port)
    try:
        site.routes["/robots.txt"] = (200, {"Content-Type": "text/plain"}, b"User-agent: *\nAllow: /\n")
        site.html("/opcenter/", article("Opcenter overview", WORDS))
        second, _ = run(cfg, tmp_path / "out")
    finally:
        site.close()
    assert second.saved_files == 1
    assert "/opcenter/" in site.paths()


def test_crawl_output_builds_into_a_pack(web, tmp_path):
    """Everything the crawler saves must ingest cleanly."""
    from rag.builder.engine import Builder
    from rag.core.workspace import Workspace

    from conftest import TEST_EMBEDDING

    a, _ = web
    run(make_cfg(a), tmp_path / "crawl")
    ws = Workspace.init(tmp_path / "ws", TEST_EMBEDDING)
    stats = Builder(ws).build([tmp_path / "crawl" / "files"])
    assert stats.failed == 0, stats.errors
    assert stats.added == 15


def test_also_required_narrows_to_semiconductor():
    from rag.crawl.content import Relevance

    r = Relevance(["opcenter", "camstar"], [], 2, ["semiconductor", "wafer", "fab"], 2)
    semi = "Opcenter Execution Semiconductor tracks every wafer through the fab. Camstar heritage."
    med = "Opcenter Execution Medical Device keeps the device history record. Camstar heritage."
    assert r.content_ok("", semi)[0] and not r.content_ok("", med)[0]
    assert r.link_score("https://x/opcenter-execution-semiconductor/") > r.link_score("https://x/opcenter-medical/")


def test_document_urls_in_locale_folders_are_judged_by_content(tmp_path):
    cfg = load_config(None, {"topic": {"terms": ["opcenter"]}, "crawl": {"seeds": []}})
    c = Crawler(cfg, tmp_path / "o", budget=1000, hard_cap=1000, log=lambda m: None)
    try:
        assert c._rejected("https://www.plm.automation.siemens.com/media/global/de/Opcenter_FS.pdf") is None
        assert c._rejected("https://www.siemens.com/de-ch/products/opcenter/") == "language_url"
        assert c._rejected("https://www.siemens.com/en-us/products/opcenter/") is None
    finally:
        c.close()


def test_presets_load():
    from rag.crawl.cli import _preset_text
    import tomllib

    for name in ("opcenter", "opcenter-semiconductor"):
        cfg = load_config(None, tomllib.loads(_preset_text(name)))
        assert cfg["topic"]["terms"] and cfg["crawl"]["seeds"]
