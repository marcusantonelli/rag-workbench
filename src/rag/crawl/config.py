"""Crawl configuration: a topic TOML file plus command-line overrides."""

from __future__ import annotations

import copy
import re
import tomllib
from pathlib import Path
from typing import Any

SUPPORTED_TYPES = ["pdf", "docx", "doc", "txt", "md", "html", "csv", "xlsx", "sql", "cs", "py"]

DEFAULTS: dict[str, Any] = {
    "topic": {
        "name": "",
        # A page or file must mention these (min_term_hits times in total) to be kept.
        "terms": [],
        # These raise a link's priority and a file's score but do not qualify it alone.
        "related": [],
        "min_term_hits": 2,
        # Optional second subject: a file must also mention one of these (e.g. narrowing
        # "Opcenter" to its semiconductor edition with ["semiconductor", "wafer", "fab"]).
        "also_required": [],
        "also_required_min_hits": 1,
        # Only English is detected; [] keeps every language.
        "languages": ["en"],
    },
    "crawl": {
        "seeds": [],
        # Domains crawled freely (suffix match: "siemens.com" covers blogs.sw.siemens.com).
        # Empty = the whole web, guided by relevance alone.
        "allowed_domains": [],
        "blocked_domains": [
            "facebook.com", "twitter.com", "x.com", "linkedin.com", "instagram.com", "youtube.com",
            "tiktok.com", "pinterest.com", "reddit.com", "scribd.com", "coursehero.com", "studocu.com",
            "slideshare.net", "dokumen.pub", "pdfcoffee.com", "vdocuments.mx", "doku.pub",
            "amazon.com", "ebay.com", "google.com", "bing.com", "duckduckgo.com",
        ],
        # Off-list hosts may be visited when a link mentions a term, this many hops out.
        "follow_external": True,
        "max_offsite_hops": 1,
        # Links without a term in their URL or anchor text are followed only this many
        # hops away from the last on-topic page (keeps the crawl focused).
        "max_tunnel": 1,
        "max_depth": 8,
        "max_pages_per_host": 20000,
        "delay_s": 1.0,           # minimum gap between requests to one host (robots may raise it)
        "workers": 4,             # parallel hosts; each host still gets one request at a time
        "timeout_s": 60,
        "retry_unreachable_s": 15,  # wait before retrying a host that refused/failed to connect
        "max_file_mb": 200,       # matches the Builder's default per-file limit
        "min_file_kb": 2,
        "min_words": 50,          # text-bearing files with fewer words are skipped
        "html_min_words": 250,    # HTML pages saved only above this much main text
        "save_html": True,
        "types": SUPPORTED_TYPES,
        "user_agent": "rag-crawl/0.1 (document research crawler; obeys robots.txt)",
        # Never fetched.
        "skip_patterns": [
            r"/(login|logout|signin|sign-in|signup|register|sso|oauth|auth)\b", r"/cart\b", r"/checkout\b",
            r"[?&](replytocom|share|print)=", r"/wp-admin/", r"/wp-login", r"/feed/?$", r"/xmlrpc",
            r"/calendar/", r"[?&]s=", r"/search\b",
        ],
        # Fetched for their links but never saved (listing and archive pages).
        "listing_patterns": [
            r"/tag/", r"/tags/", r"/category/", r"/author/", r"/page/\d+/?$", r"/archives?/",
            r"[?&]page=\d+", r"/\d{4}/\d{2}/?$", r"/wp-json/",
        ],
    },
    "search": {
        "provider": "",           # "brave" (BRAVE_API_KEY env var) or "searxng"
        "brave_url": "https://api.search.brave.com/res/v1/web/search",
        "searxng_url": "",        # e.g. http://localhost:8888
        "queries": [],
        "pages_per_query": 5,
        "delay_s": 1.1,
    },
    "sitemaps": {
        "enabled": True,
        "max_files": 300,
    },
}

_SIZE_RE = re.compile(r"^\s*([\d.]+)\s*([kmgt]?i?b?)?\s*$", re.I)
_UNITS = {"": 1, "b": 1, "k": 1e3, "kb": 1e3, "m": 1e6, "mb": 1e6, "g": 1e9, "gb": 1e9, "t": 1e12, "tb": 1e12,
          "kib": 1024, "mib": 1024**2, "gib": 1024**3, "tib": 1024**4}


def parse_size(text: str) -> int:
    """'15GB' -> 15_000_000_000; '500 MiB' -> 524_288_000."""
    m = _SIZE_RE.match(str(text))
    if not m:
        raise ValueError(f"cannot read size {text!r} (examples: 15GB, 500MB, 2GiB)")
    unit = (m.group(2) or "").lower()
    if unit not in _UNITS:
        raise ValueError(f"unknown size unit in {text!r}")
    return int(float(m.group(1)) * _UNITS[unit])


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000
    return str(n)


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: Path | None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if path is not None:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    cfg = _merge(DEFAULTS, data)
    cfg = _merge(cfg, overrides or {})
    unknown = set(cfg["crawl"]["types"]) - set(SUPPORTED_TYPES)
    if unknown:
        raise ValueError(f"unsupported types in crawl.types: {sorted(unknown)}")
    if not cfg["topic"]["terms"]:
        raise ValueError("topic.terms is empty: give at least one term that on-topic files must contain")
    return cfg
