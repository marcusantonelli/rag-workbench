"""Finding starting URLs: sitemaps of the seed hosts, and optional web search."""

from __future__ import annotations

import gzip
import json
import os
import re
import time
from typing import Callable, Iterator
from urllib.parse import urlencode, urlsplit

from rag.crawl.net import Fetcher, RobotsCache

_LOC_RE = re.compile(rb"<loc>\s*(.*?)\s*</loc>", re.S | re.I)


def _xml_body(data: bytes) -> bytes:
    if data[:2] == b"\x1f\x8b":
        try:
            return gzip.decompress(data)
        except OSError:
            return b""
    return data


def sitemap_urls(origin_url: str, fetcher: Fetcher, robots: RobotsCache, *, keep: Callable[[str], bool],
                 max_files: int, delay: float, log: Callable[[str], None]) -> Iterator[str]:
    """Walk a host's sitemaps (from robots.txt, else /sitemap.xml) and yield URLs that `keep` accepts.

    Child sitemaps whose own URL mentions a locale other than English are still
    read (their names rarely say), but every page URL is filtered by `keep`.
    """
    from rag.crawl.net import RobotsUnreachable

    parts = urlsplit(origin_url)
    origin = f"{parts.scheme}://{parts.netloc}"
    try:
        todo = robots.sitemaps(origin_url) or [origin + "/sitemap.xml", origin + "/sitemap_index.xml"]
    except RobotsUnreachable as exc:
        log(f"sitemaps skipped: {exc}")
        return
    seen: set[str] = set()
    fetched = 0
    while todo and fetched < max_files:
        sm = todo.pop(0)
        try:
            if sm in seen or not robots.allowed(sm):
                continue
        except RobotsUnreachable:
            continue
        seen.add(sm)
        time.sleep(delay)
        resp = fetcher.get(sm, max_bytes=100_000_000)
        fetched += 1
        if resp.status in (301, 302, 303, 307, 308) and resp.location:
            todo.insert(0, resp.location)
            continue
        if resp.status != 200 or not resp.body:
            continue
        body = _xml_body(resp.body)
        locs = [m.decode("utf-8", "replace").replace("&amp;", "&") for m in _LOC_RE.findall(body)]
        is_index = b"<sitemapindex" in body[:2000].lower()
        if is_index:
            todo.extend(locs)
            continue
        kept = 0
        for loc in locs:
            if keep(loc):
                kept += 1
                yield loc
        log(f"sitemap {sm}: {len(locs)} URLs, {kept} on topic")


def search_urls(cfg: dict, log: Callable[[str], None]) -> Iterator[tuple[str, str]]:
    """Yield (url, title + snippet) from the configured search provider."""
    s = cfg["search"]
    provider = (s.get("provider") or "").lower()
    if not provider or not s.get("queries"):
        return
    fetcher = Fetcher("rag-crawl/0.1", 30)
    pages = max(1, int(s.get("pages_per_query", 5)))
    for query in s["queries"]:
        for page in range(pages):
            if provider == "brave":
                key = os.environ.get("BRAVE_API_KEY", "")
                if not key:
                    log("search: BRAVE_API_KEY is not set; skipping web search")
                    return
                url = s["brave_url"] + "?" + urlencode({"q": query, "count": 20, "offset": min(page, 9)})
                resp = fetcher.get(url, max_bytes=5_000_000, headers={
                    "X-Subscription-Token": key, "Accept": "application/json"})
                items = []
                if resp.status == 200:
                    data = json.loads(resp.body or b"{}")
                    items = [(r.get("url", ""), f"{r.get('title', '')} {r.get('description', '')}")
                             for r in (data.get("web") or {}).get("results", [])]
            elif provider == "searxng":
                base = s["searxng_url"].rstrip("/")
                url = f"{base}/search?" + urlencode({"q": query, "format": "json", "pageno": page + 1})
                resp = fetcher.get(url, max_bytes=5_000_000, headers={"Accept": "application/json"})
                items = []
                if resp.status == 200:
                    data = json.loads(resp.body or b"{}")
                    items = [(r.get("url", ""), f"{r.get('title', '')} {r.get('content', '')}")
                             for r in data.get("results", [])]
            else:
                log(f"search: unknown provider {provider!r}")
                return
            if resp.status != 200:
                log(f"search: {provider} returned HTTP {resp.status} for {query!r}: "
                    f"{(resp.body or b'')[:200].decode('utf-8', 'replace')}")
                break
            log(f"search {query!r} page {page + 1}: {len(items)} results")
            for item in items:
                if item[0]:
                    yield item
            if not items:
                break
            time.sleep(float(s.get("delay_s", 1.1)))
