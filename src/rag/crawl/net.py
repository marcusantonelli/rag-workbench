"""HTTP fetching: no redirects followed automatically, no cookies, robots.txt per host.

Redirects are returned to the crawler, which queues the target like any other
URL, so every hop is checked against robots.txt and the per-host delay.
Proxies come from the usual HTTP(S)_PROXY environment variables.
"""

from __future__ import annotations

import email.utils
import re
import socket
import ssl
import time
import urllib.error
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qsl, urlencode

_TRACKING = re.compile(r"^(utm_\w+|gclid|fbclid|msclkid|mc_cid|mc_eid|_hsenc|_hsmi|hsctatracking|"
                       r"trk|trkcampaign|cid|sc_cid|ref_src|igshid|_ga|_gl)$", re.I)


def canonical_url(url: str, base: str | None = None) -> str | None:
    """Absolute, normalized URL; None for non-http links."""
    url = url.strip()
    if base:
        url = urljoin(base, url)
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    host = parts.hostname.lower().rstrip(".")
    port = parts.port
    netloc = host if port in (None, 80 if parts.scheme == "http" else 443) else f"{host}:{port}"
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                             if not _TRACKING.match(k)))
    path = re.sub(r"/{2,}", "/", parts.path or "/")
    return urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower()


def hostname_of(url: str) -> str:
    return (urlsplit(url).hostname or "").lower()


def domain_match(hostname: str, domains: list[str]) -> bool:
    hostname = hostname.lower()
    return any(hostname == d.lower() or hostname.endswith("." + d.lower()) for d in domains)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401
        return None


@dataclass
class Response:
    url: str
    status: int
    headers: dict[str, str]
    body: bytes = b""
    location: str | None = None
    truncated: bool = False
    error: str | None = None
    elapsed: float = 0.0

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "")

    @property
    def content_length(self) -> int | None:
        try:
            return int(self.headers.get("content-length", ""))
        except ValueError:
            return None


def retry_after_seconds(value: str | None, default: float) -> float:
    if not value:
        return default
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
            return max(0.0, when.timestamp() - time.time())
        except (TypeError, ValueError):
            return default


class Fetcher:
    def __init__(self, user_agent: str, timeout: float):
        self.user_agent = user_agent
        self.timeout = timeout
        ctx = ssl.create_default_context()
        self.opener = urllib.request.build_opener(
            _NoRedirect(), urllib.request.HTTPSHandler(context=ctx), urllib.request.ProxyHandler())

    def get(self, url: str, *, max_bytes: int, accept_body=None, headers: dict[str, str] | None = None) -> Response:
        """GET a URL. ``accept_body(response)`` sees status and headers first and may
        return False to skip downloading the body, or an int to lower max_bytes."""
        req = urllib.request.Request(url, headers={
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/pdf,application/vnd.openxmlformats-"
                      "officedocument.*,text/*;q=0.9,*/*;q=0.5",
            "Accept-Language": "en;q=1.0,*;q=0.3",
            **(headers or {}),
        })
        started = time.monotonic()
        try:
            resp = self.opener.open(req, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            hdrs = {k.lower(): v for k, v in (exc.headers.items() if exc.headers else [])}
            loc = hdrs.get("location")
            body = b""
            if exc.code < 300 or exc.code >= 400:
                try:
                    body = exc.read(65536)
                except Exception:  # noqa: BLE001
                    pass
            return Response(url=url, status=exc.code, headers=hdrs, body=body,
                            location=urljoin(url, loc) if loc else None, elapsed=time.monotonic() - started)
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, ssl.SSLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            return Response(url=url, status=0, headers={}, error=f"{type(exc).__name__}: {reason}",
                            elapsed=time.monotonic() - started)
        with resp:
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            out = Response(url=url, status=resp.status, headers=hdrs)
            if accept_body is not None:
                verdict = accept_body(out)
                if verdict is False:
                    out.elapsed = time.monotonic() - started
                    return out
                if isinstance(verdict, int) and not isinstance(verdict, bool):
                    max_bytes = min(max_bytes, verdict)
            chunks: list[bytes] = []
            total = 0
            try:
                while True:
                    block = resp.read(65536)
                    if not block:
                        break
                    total += len(block)
                    if total > max_bytes:
                        out.truncated = True
                        break
                    chunks.append(block)
            except (socket.timeout, TimeoutError, ConnectionError, OSError) as exc:
                out.error = f"{type(exc).__name__}: {exc}"
            out.body = b"".join(chunks)
            out.elapsed = time.monotonic() - started
            return out


class RobotsUnreachable(RuntimeError):
    """robots.txt could not be fetched at all (network error); try the host later."""


class RobotsCache:
    """robots.txt per host (scheme://host:port)."""

    def __init__(self, fetcher: Fetcher, agent_token: str):
        self.fetcher = fetcher
        self.agent = agent_token
        self._cache: dict[str, tuple[urllib.robotparser.RobotFileParser, list[str]]] = {}
        self.status: dict[str, int] = {}

    def _load(self, origin: str) -> tuple[urllib.robotparser.RobotFileParser, list[str]]:
        rp = urllib.robotparser.RobotFileParser()
        sitemaps: list[str] = []
        resp = self.fetcher.get(origin + "/robots.txt", max_bytes=512_000)
        hops = 0
        while resp.status in (301, 302, 303, 307, 308) and resp.location and hops < 3:
            resp = self.fetcher.get(resp.location, max_bytes=512_000)
            hops += 1
        if resp.status == 0:
            raise RobotsUnreachable(f"robots.txt unreachable for {origin}: {resp.error}")
        self.status[origin] = resp.status
        if resp.status == 200:
            lines = resp.body.decode("utf-8", "replace").splitlines()
            rp.parse(lines)
            sitemaps = [ln.split(":", 1)[1].strip() for ln in lines if ln.lower().startswith("sitemap:")]
        elif resp.status in (401, 403):
            rp.disallow_all = True
        elif 400 <= resp.status < 500:
            rp.allow_all = True
        else:  # 5xx: RFC 9309 says treat the whole host as disallowed
            rp.disallow_all = True
        return rp, sitemaps

    def get(self, url: str) -> tuple[urllib.robotparser.RobotFileParser, list[str]]:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        if origin not in self._cache:
            self._cache[origin] = self._load(origin)
        return self._cache[origin]

    def allowed(self, url: str) -> bool:
        rp, _ = self.get(url)
        return rp.can_fetch(self.agent, url)

    def delay(self, url: str) -> float:
        rp, _ = self.get(url)
        d = rp.crawl_delay(self.agent)
        rate = rp.request_rate(self.agent)
        values = [float(d)] if d else []
        if rate and rate.requests:
            values.append(rate.seconds / rate.requests)
        return max(values) if values else 0.0

    def sitemaps(self, url: str) -> list[str]:
        return list(self.get(url)[1])

    def status_of(self, url: str) -> int | None:
        parts = urlsplit(url)
        return self.status.get(f"{parts.scheme}://{parts.netloc}")

    def known(self, url: str) -> bool:
        parts = urlsplit(url)
        return f"{parts.scheme}://{parts.netloc}" in self._cache
