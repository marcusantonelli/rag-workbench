"""The crawl loop.

Scheduling: every host has its own priority queue and a "next allowed" time.
A worker takes the host that is free soonest, pops that host's best URL, and
holds the host until the request finishes plus the delay, so each host sees
one request at a time no faster than its delay (or robots.txt crawl-delay).

Focus: links whose URL or anchor text mention a topic term are always
followed; other links only up to ``max_tunnel`` hops from the last on-topic
page, so the crawl does not wander across a large site's navigation.

Budget: a download reserves its expected size against the hard cap before it
starts, so parallel downloads cannot overshoot. New fetches stop once the
saved total reaches the target; the hard cap is never exceeded.
"""

from __future__ import annotations

import hashlib
import heapq
import itertools
import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from rag.crawl.config import human
from rag.crawl.content import (Relevance, looks_english, obviously_unsupported, parse_file, parse_html,
                               sniff_type, type_from_headers, type_from_url, url_language)
from rag.crawl.discover import search_urls, sitemap_urls
from rag.crawl.net import (Fetcher, Response, RobotsCache, RobotsUnreachable, canonical_url, domain_match, host_of,
                           hostname_of, retry_after_seconds)
from rag.crawl.state import CrawlState

MAX_ATTEMPTS = 4
_REDIRECT = (301, 302, 303, 307, 308)
_LOGIN_RE = re.compile(r"(login|signin|sign-in|sso|oauth|/auth\b|saml|account/|okta|adfs)", re.I)


@dataclass(order=True)
class _Item:
    neg_priority: float
    seq: int
    url: str = field(compare=False)


@dataclass
class _Host:
    queue: list[_Item] = field(default_factory=list)
    next_time: float = 0.0
    busy: bool = False
    delay: float = 1.0
    fetched: int = 0
    backoff: float = 0.0


@dataclass
class Summary:
    saved_bytes: int
    saved_files: int
    stop_reason: str
    by_type: dict[str, dict[str, int]]
    by_host: dict[str, dict[str, int]]
    statuses: dict[str, int]
    skip_reasons: list[tuple[str, int]]
    seconds: float


class Crawler:
    def __init__(self, cfg: dict[str, Any], out_dir: Path, *, budget: int, hard_cap: int,
                 log: Callable[[str], None] = print, max_requests: int | None = None):
        self.cfg = cfg
        self.c = cfg["crawl"]
        self.out = Path(out_dir)
        self.files_dir = self.out / "files"
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self.state = CrawlState(self.out / "crawl.db")
        self.budget = int(budget)
        self.hard_cap = max(int(hard_cap), self.budget)
        self.log = log
        self.max_requests = max_requests
        self.fetcher = Fetcher(self.c["user_agent"], float(self.c["timeout_s"]))
        token = self.c["user_agent"].split("/")[0].split()[0]
        self.robots = RobotsCache(self.fetcher, token)
        t = cfg["topic"]
        self.rel = Relevance(t["terms"], t["related"], t["min_term_hits"],
                             t.get("also_required") or [], t.get("also_required_min_hits", 1))
        self.languages = [lang.lower() for lang in t.get("languages", [])]
        self.types = set(self.c["types"])
        self.skip_re = [re.compile(p, re.I) for p in self.c["skip_patterns"]]
        self.listing_re = [re.compile(p, re.I) for p in self.c["listing_patterns"]]
        self.max_file = int(float(self.c["max_file_mb"]) * 1_000_000)
        self.min_file = int(float(self.c["min_file_kb"]) * 1000)

        self.lock = threading.Condition()
        self.hosts: dict[str, _Host] = {}
        self.seq = itertools.count()
        self.saved = self.state.saved_bytes()
        self.saved_files = len(self.state.files())
        self.reserved = 0
        self.active = 0
        self.requests = 0
        self.stop_event = threading.Event()
        self.stop_reason = ""
        self.manifest_lock = threading.Lock()
        host_counts = self.state.host_counts()
        for row in self.state.queued():
            self._push(row["url"], row["host"], row["priority"])
        for host, n in host_counts.items():
            self._host(host).fetched = n

    # ------------------------------------------------------------------ policy
    def _host(self, host: str) -> _Host:
        h = self.hosts.get(host)
        if h is None:
            h = self.hosts[host] = _Host(delay=float(self.c["delay_s"]))
        return h

    def _push(self, url: str, host: str, priority: float) -> None:
        heapq.heappush(self._host(host).queue, _Item(-priority, next(self.seq), url))

    def _allowed_domain(self, url: str) -> bool:
        allowed = self.c["allowed_domains"]
        return not allowed or domain_match(hostname_of(url), allowed)

    def _rejected(self, url: str) -> str | None:
        if domain_match(hostname_of(url), self.c["blocked_domains"]):
            return "blocked_domain"
        if obviously_unsupported(url):
            return "unsupported_type"
        guess = type_from_url(url)
        if guess and guess not in self.types and guess != "html":
            return "type_not_wanted"
        if any(r.search(url) for r in self.skip_re):
            return "skip_pattern"
        if self.languages and (guess is None or guess == "html"):
            # Locale folders reliably mean translated web pages; documents are judged by
            # their text instead (Siemens files English PDFs under /media/global/de/ etc.).
            lang = url_language(url)
            if lang and lang not in self.languages:
                return "language_url"
        return None

    def offer(self, url: str, *, parent: str | None, anchor: str, depth: int, parent_offsite: int,
              parent_tunnel: int, parent_relevant: bool, bonus: float = 0.0, redirect_of=None) -> bool:
        """Consider a discovered URL for the queue.

        tunnel = hops since the last on-topic page (0 when the link itself names a term);
        offsite = hops outside the allowed domains.
        """
        url = canonical_url(url, parent)
        if url is None or len(url) > 2000:
            return False
        if self._rejected(url):
            return False
        if depth > int(self.c["max_depth"]):
            return False
        allowed = self._allowed_domain(url)
        link_score = self.rel.link_score(url, anchor)
        if redirect_of is not None:
            # A redirect target inherits the admitted URL's standing.
            tunnel = redirect_of["tunnel"]
            offsite = 0 if allowed else max(redirect_of["offsite"], 1)
            if not allowed and (not self.c["follow_external"] or offsite > int(self.c["max_offsite_hops"])):
                return False
            priority = float(redirect_of["priority"])
        else:
            offsite = 0 if allowed else parent_offsite + 1
            if not allowed:
                if not self.c["follow_external"] or link_score < 10 or offsite > int(self.c["max_offsite_hops"]):
                    return False
            if parent is None or link_score >= 10:
                tunnel = 0
            else:
                tunnel = 1 if parent_relevant else parent_tunnel + 1
                if tunnel > int(self.c["max_tunnel"]):
                    return False
            priority = link_score + bonus - 0.5 * depth
            if parent_relevant:
                priority += 5
            if allowed and self.c["allowed_domains"]:
                priority += 3
            if type_from_url(url) in ("pdf", "docx", "doc", "xlsx", "csv"):
                priority += 4
        host = host_of(url)
        if self.state.add_url(url, host, depth, offsite, tunnel, priority, parent, anchor):
            with self.lock:
                self._push(url, host, priority)
                self.lock.notify_all()
            return True
        return False

    # ---------------------------------------------------------------- discovery
    def seed(self) -> None:
        for s in self.c["seeds"]:
            self.offer(s, parent=None, anchor="", depth=0, parent_offsite=0, parent_tunnel=0,
                       parent_relevant=True, bonus=20)
        if self.state.get_meta("discovered") == "1":
            return
        n_search = 0
        for url, text in search_urls(self.cfg, self.log):
            if self.offer(url, parent=None, anchor=text, depth=0, parent_offsite=0, parent_tunnel=0,
                          parent_relevant=False, bonus=5):
                n_search += 1
        if n_search:
            self.log(f"search added {n_search} URLs")
        if self.cfg["sitemaps"]["enabled"]:
            origins = {f"{urlsplit(s).scheme}://{urlsplit(s).netloc}" for s in self.c["seeds"]}
            for origin in sorted(origins):
                if self.stop_event.is_set():
                    return
                added = 0
                keep = lambda u: self.rel.link_score(u) >= 10 and not self._rejected(u)  # noqa: E731
                try:
                    delay = max(float(self.c["delay_s"]), self.robots.delay(origin + "/"))
                except RobotsUnreachable as exc:
                    self.log(f"sitemaps of {origin} skipped: {exc}")
                    continue
                for loc in sitemap_urls(origin + "/", self.fetcher, self.robots, keep=keep,
                                        max_files=int(self.cfg["sitemaps"]["max_files"]),
                                        delay=delay, log=self.log):
                    if self.offer(loc, parent=None, anchor="", depth=1, parent_offsite=0, parent_tunnel=0,
                                  parent_relevant=False, bonus=2):
                        added += 1
                if added:
                    self.log(f"sitemaps of {origin} added {added} URLs")
        self.state.set_meta("discovered", "1")

    # ---------------------------------------------------------------- scheduling
    def _next(self) -> tuple[str, str] | None:
        """Block until a URL can be fetched; None when the crawl is over."""
        with self.lock:
            while True:
                if self.stop_event.is_set():
                    return None
                if self.saved >= self.budget:
                    self._stop(f"reached the {human(self.budget)} target")
                    return None
                if self.max_requests is not None and self.requests >= self.max_requests:
                    self._stop(f"reached max_requests={self.max_requests}")
                    return None
                now = time.monotonic()
                best_host, best_time = None, None
                cap = int(self.c["max_pages_per_host"])
                for name, h in self.hosts.items():
                    if h.busy or not h.queue:
                        continue
                    if h.fetched >= cap:
                        h.queue.clear()
                        continue
                    if best_time is None or h.next_time < best_time:
                        best_host, best_time = name, h.next_time
                if best_host is None:
                    if self.active == 0:
                        self._stop("no more on-topic URLs to fetch (frontier exhausted)")
                        return None
                    self.lock.wait(0.5)
                    continue
                if best_time > now:
                    self.lock.wait(min(best_time - now, 0.5))
                    continue
                h = self.hosts[best_host]
                item = heapq.heappop(h.queue)
                h.busy = True
                self.active += 1
                self.requests += 1
                return item.url, best_host

    def _release(self, host: str, *, backoff: float = 0.0) -> None:
        with self.lock:
            h = self.hosts[host]
            h.busy = False
            h.fetched += 1
            h.next_time = time.monotonic() + max(h.delay, backoff)
            self.active -= 1
            self.lock.notify_all()

    def _stop(self, reason: str) -> None:
        if not self.stop_event.is_set():
            self.stop_reason = reason
            self.stop_event.set()
            self.lock.notify_all()

    # ---------------------------------------------------------------- work
    def _worker(self) -> None:
        while True:
            nxt = self._next()
            if nxt is None:
                return
            url, host = nxt
            backoff = 0.0
            try:
                backoff = self._process(url, host)
            except Exception as exc:  # noqa: BLE001 - keep crawling
                self.state.set_status(url, "failed", f"error: {type(exc).__name__}: {exc}"[:300], attempt=True)
            finally:
                self._release(host, backoff=backoff)

    def _process(self, url: str, host: str) -> float:
        row = self.state.get(url)
        if row is None:
            return 0.0
        self.state.set_status(url, "active")
        try:
            if not self.robots.known(url):
                d = self.robots.delay(url)
                with self.lock:
                    self._host(host).delay = max(float(self.c["delay_s"]), d)
            allowed = self.robots.allowed(url)
        except RobotsUnreachable as exc:
            if row["attempts"] + 1 < MAX_ATTEMPTS:
                self._requeue(url, row, "robots.txt unreachable")
                return 60.0
            self.state.set_status(url, "failed", str(exc)[:300], attempt=True)
            return 0.0
        if not allowed:
            code = self.robots.status_of(url)
            reason = "robots" if code == 200 or code is None else f"robots.txt HTTP {code} (host treated as off-limits)"
            self.state.set_status(url, "skipped", reason)
            return 0.0

        ctx: dict[str, Any] = {}

        def accept(resp: Response):
            if resp.status != 200:
                return False
            guess = type_from_headers(url, resp.content_type)
            ctx["guess"] = guess
            if guess is None or (guess not in self.types and guess != "html"):
                ctx["skip"] = f"type {resp.content_type.split(';')[0] or 'unknown'}"
                return False
            length = resp.content_length
            limit = self.max_file if guess != "html" else min(self.max_file, 20_000_000)
            if length is not None and length > limit:
                ctx["skip"] = f"too_large ({human(length)})"
                return False
            with self.lock:
                room = self.hard_cap - self.saved - self.reserved
                want = min(limit, length) if length else limit
                if guess != "html" and room < max(self.min_file, length or self.min_file):
                    ctx["skip"] = "budget_room"
                    return False
                reserve = min(want, max(room, 0)) if guess != "html" else 0
                self.reserved += reserve
                ctx["reserved"] = reserve
            return reserve if guess != "html" else limit

        resp = self.fetcher.get(url, max_bytes=self.max_file, accept_body=accept)
        try:
            return self._handle(url, row, resp, ctx)
        finally:
            with self.lock:
                self.reserved -= ctx.get("reserved", 0)

    def _handle(self, url: str, row, resp: Response, ctx: dict[str, Any]) -> float:
        if resp.status == 0:
            if row["attempts"] + 1 < MAX_ATTEMPTS:
                self._requeue(url, row, f"network: {resp.error}")
                return 30.0
            self.state.set_status(url, "failed", f"network: {resp.error}", attempt=True)
            return 0.0
        if resp.status in _REDIRECT:
            target = resp.location
            if not target:
                self.state.set_status(url, "failed", f"redirect without location ({resp.status})")
                return 0.0
            if _LOGIN_RE.search(urlsplit(target).path + "?" + urlsplit(target).query):
                self.state.set_status(url, "skipped", "login_required")
                return 0.0
            self.state.set_status(url, "done", f"redirect {resp.status}")
            self.offer(target, parent=url, anchor=row["anchor"] or "", depth=row["depth"],
                       parent_offsite=row["offsite"], parent_tunnel=row["tunnel"], parent_relevant=False,
                       redirect_of=row)
            return 0.0
        if resp.status in (429, 503):
            wait = min(retry_after_seconds(resp.headers.get("retry-after"), 60.0), 600.0)
            if row["attempts"] + 1 < MAX_ATTEMPTS:
                self._requeue(url, row, f"http {resp.status}")
                self.log(f"{host_of(url)} asked to slow down (HTTP {resp.status}); pausing {wait:.0f}s")
                return wait
            self.state.set_status(url, "failed", f"http {resp.status}", attempt=True)
            return wait
        if resp.status in (401, 403):
            self.state.set_status(url, "skipped", "login_required" if resp.status == 401 else "forbidden")
            return 0.0
        if resp.status >= 500:
            if row["attempts"] + 1 < MAX_ATTEMPTS:
                self._requeue(url, row, f"http {resp.status}")
                return 20.0
            self.state.set_status(url, "failed", f"http {resp.status}", attempt=True)
            return 0.0
        if resp.status != 200:
            self.state.set_status(url, "failed", f"http {resp.status}")
            return 0.0
        if "skip" in ctx:
            self.state.set_status(url, "skipped", ctx["skip"])
            return 0.0
        if resp.truncated:
            self.state.set_status(url, "skipped", "too_large_or_no_budget_room")
            return 0.0
        if resp.error:
            self.state.set_status(url, "failed", f"read: {resp.error}", attempt=True)
            return 0.0

        ftype = sniff_type(resp.body, ctx.get("guess"))
        if ftype is None:
            self.state.set_status(url, "skipped", "unsupported_content")
            return 0.0
        if ftype == "html":
            self._handle_html(url, row, resp.body)
        else:
            self._handle_file(url, row, resp.body, ftype)
        return 0.0

    def _requeue(self, url: str, row, reason: str) -> None:
        self.state.set_status(url, "queued", reason, attempt=True)
        with self.lock:
            self._push(url, host_of(url), row["priority"] - 1)

    # ---------------------------------------------------------------- content
    def _handle_html(self, url: str, row, body: bytes) -> None:
        p = parse_html(body)
        relevant, score = self.rel.content_ok(p.title, p.text)
        english = not self.languages or "en" not in self.languages or looks_english(p.text)
        # Links: follow from every page (focus rules apply), unless the page says nofollow.
        if not p.nofollow and english:
            for href, anchor in p.links[:2000]:
                self.offer(href, parent=url, anchor=anchor, depth=row["depth"] + 1,
                           parent_offsite=row["offsite"], parent_tunnel=row["tunnel"],
                           parent_relevant=relevant)
        listing = any(r.search(url) for r in self.listing_re)
        reasons = []
        if not self.c["save_html"] or "html" not in self.types:
            reasons.append("html_not_saved")
        if listing:
            reasons.append("listing_page")
        if p.noindex:
            reasons.append("noindex")
        if not english:
            reasons.append("not_english")
        if not relevant:
            reasons.append("off_topic")
        if p.words < int(self.c["html_min_words"]):
            reasons.append("thin_page")
        if p.canonical:
            canon = canonical_url(p.canonical, url)
            if canon and canon != url and self.state.known(canon):
                reasons.append("canonical_elsewhere")
        if reasons:
            self.state.set_status(url, "done", ",".join(reasons))
            return
        text_hash = hashlib.sha256(" ".join(p.text.lower().split()).encode()).hexdigest()
        self._save(url, body, "html", text_hash, score, p.title)

    def _handle_file(self, url: str, row, body: bytes, ftype: str) -> None:
        if ftype not in self.types:
            self.state.set_status(url, "skipped", f"type {ftype} not wanted")
            return
        if len(body) < self.min_file:
            self.state.set_status(url, "skipped", "too_small")
            return
        p = parse_file(body, ftype)
        if p.broken:
            self.state.set_status(url, "skipped", f"unreadable: {p.note}")
            return
        relevant, score = self.rel.content_ok(p.title, p.text)
        link_score = self.rel.link_score(url, row["anchor"] or "")
        if not relevant and ftype in ("pdf", "doc", "xlsx", "csv") and p.words < 30 and link_score >= 10:
            relevant, score = True, link_score / 10   # scanned PDF or numeric sheet named for the topic
        if not relevant:
            self.state.set_status(url, "skipped", "off_topic")
            return
        if ftype in ("pdf", "docx", "doc", "txt", "md") and p.words < int(self.c["min_words"]) and link_score < 10:
            self.state.set_status(url, "skipped", "too_little_text")
            return
        if self.languages and "en" in self.languages and ftype not in ("csv", "xlsx", "sql", "cs", "py") \
                and not looks_english(p.text):
            self.state.set_status(url, "skipped", "not_english")
            return
        self._save(url, body, ftype, hashlib.sha256(body).hexdigest(), score, p.title)

    def _save(self, url: str, body: bytes, ftype: str, content_hash: str, score: float, title: str) -> None:
        if self.state.has_content(content_hash):
            self.state.set_status(url, "done", "duplicate_content")
            return
        with self.lock:
            if self.saved + len(body) > self.hard_cap:
                self.state.set_status(url, "skipped", "budget_room")
                return
            self.saved += len(body)
            self.saved_files += 1
        rel_path = self._path_for(url, ftype, content_hash)
        dest = self.out / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(body)
        tmp.replace(dest)
        self.state.add_file(content_hash, url, rel_path.as_posix(), ftype, len(body), score, title[:300])
        self.state.set_status(url, "saved")
        entry = {"path": rel_path.as_posix(), "url": url, "type": ftype, "bytes": len(body),
                 "sha256": content_hash if ftype != "html" else hashlib.sha256(body).hexdigest(),
                 "title": title[:300], "relevance": score, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        with self.manifest_lock, open(self.out / "manifest.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.log(f"saved {human(len(body)):>10}  {ftype:4}  {url}   [total {human(self.saved)}]")

    def _path_for(self, url: str, ftype: str, content_hash: str) -> Path:
        parts = urlsplit(url)
        segs = [unquote(s) for s in parts.path.split("/") if s]
        stem = segs[-1] if segs else "index"
        if "." in stem:
            stem = stem.rsplit(".", 1)[0]
        if len(segs) >= 2 and stem.lower() in ("index", "default", "page", "view", "download"):
            stem = f"{segs[-2]}-{stem}"
        stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._") or "file"
        stem = stem[:80]
        host = re.sub(r"[^A-Za-z0-9.-]+", "_", parts.hostname or "host")
        ext = {"md": "md", "html": "html"}.get(ftype, ftype)
        return Path("files") / host / f"{stem}-{content_hash[:8]}.{ext}"

    # ---------------------------------------------------------------- run
    def run(self) -> Summary:
        started = time.monotonic()
        self.log(f"Target {human(self.budget)} (hard cap {human(self.hard_cap)}); "
                 f"already saved {human(self.saved)} in {self.saved_files} files")
        if self.saved >= self.budget:
            self.stop_reason = "target already reached"
        else:
            self.seed()
            workers = [threading.Thread(target=self._worker, daemon=True, name=f"crawl-{i}")
                       for i in range(max(1, int(self.c["workers"])))]
            for w in workers:
                w.start()
            try:
                last = time.monotonic()
                while any(w.is_alive() for w in workers):
                    for w in workers:
                        w.join(timeout=0.5)
                    if time.monotonic() - last > 60:
                        last = time.monotonic()
                        with self.lock:
                            queued = sum(len(h.queue) for h in self.hosts.values())
                        self.log(f"... {human(self.saved)} saved in {self.saved_files} files, "
                                 f"{self.requests} requests, {queued} URLs queued")
            except KeyboardInterrupt:
                self.log("stopping after in-flight requests finish (Ctrl+C again to force)")
                with self.lock:
                    self._stop("interrupted; run again to resume")
                for w in workers:
                    w.join()
        return self.summary(time.monotonic() - started)

    def summary(self, seconds: float = 0.0) -> Summary:
        files = self.state.files()
        by_type: dict[str, dict[str, int]] = {}
        by_host: dict[str, dict[str, int]] = {}
        for f in files:
            for key, bucket in ((f["type"], by_type), (urlsplit(f["url"]).hostname or "?", by_host)):
                b = bucket.setdefault(key, {"files": 0, "bytes": 0})
                b["files"] += 1
                b["bytes"] += f["bytes"]
        return Summary(saved_bytes=sum(f["bytes"] for f in files), saved_files=len(files),
                       stop_reason=self.stop_reason or "finished", by_type=by_type, by_host=by_host,
                       statuses=self.state.status_counts(), skip_reasons=self.state.skip_reasons(),
                       seconds=round(seconds, 1))

    def close(self) -> None:
        self.state.close()
