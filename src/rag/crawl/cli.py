"""rag-crawl: collect on-topic documents from the web up to a size budget.

    rag-crawl run CONFIG.toml OUT --budget 15GB [--max 20GB]
    rag-crawl run --preset example OUT --budget 2GB
    rag-crawl report OUT
    rag-crawl preset example > my-topic.toml

Files land in OUT/files/<host>/; OUT/manifest.jsonl records each file's source URL.
Run again with the same OUT to resume (or to raise the budget). Then build:

    rag build my-ws OUT/files
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib import resources
from pathlib import Path

from rag.crawl.config import human, load_config, parse_size


def _preset_text(name: str) -> str:
    try:
        return resources.files("rag.crawl.presets").joinpath(f"{name}.toml").read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise SystemExit(f"no preset named {name!r}") from exc


def _print_summary(s, budget: int | None = None) -> None:
    print(f"\nStopped: {s.stop_reason}")
    print(f"Saved {s.saved_files} files, {human(s.saved_bytes)}" +
          (f" of the {human(budget)} target" if budget else "") + (f" in {s.seconds:.0f}s" if s.seconds else ""))
    if s.by_type:
        print("  by type: " + ", ".join(f"{t} {v['files']} ({human(v['bytes'])})"
                                         for t, v in sorted(s.by_type.items(), key=lambda kv: -kv[1]["bytes"])))
    if s.by_host:
        top = sorted(s.by_host.items(), key=lambda kv: -kv[1]["bytes"])[:8]
        print("  top hosts: " + ", ".join(f"{h} {v['files']} ({human(v['bytes'])})" for h, v in top))
    print("  URLs: " + ", ".join(f"{k} {v}" for k, v in sorted(s.statuses.items())))
    if s.skip_reasons:
        print("  most common skip reasons: " + ", ".join(f"{r} {n}" for r, n in s.skip_reasons[:8]))


def cmd_run(args) -> int:
    from rag.crawl.crawler import Crawler

    if args.preset:
        import tempfile

        tmp = Path(tempfile.mkdtemp()) / f"{args.preset}.toml"
        tmp.write_text(_preset_text(args.preset), encoding="utf-8")
        cfg_path = tmp
    elif args.config:
        cfg_path = Path(args.config)
    else:
        raise SystemExit("give a CONFIG.toml or --preset NAME")
    overrides: dict = {"crawl": {}}
    if args.workers:
        overrides["crawl"]["workers"] = args.workers
    if args.delay is not None:
        overrides["crawl"]["delay_s"] = args.delay
    if args.seed:
        overrides["crawl"]["seeds"] = args.seed
    if args.no_html:
        overrides["crawl"]["save_html"] = False
    cfg = load_config(cfg_path, overrides)
    budget = parse_size(args.budget)
    hard = parse_size(args.max) if args.max else int(budget * 1.15)
    if hard < budget:
        raise SystemExit("--max must be at least --budget")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "crawl-config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    log = (lambda m: None) if args.quiet else (lambda m: print(m, flush=True))
    crawler = Crawler(cfg, out, budget=budget, hard_cap=hard, log=log, max_requests=args.max_requests)
    try:
        summary = crawler.run()
    finally:
        crawler.close()
    _print_summary(summary, budget)
    if summary.saved_files:
        print(f"\nNext: rag build <workspace> \"{out / 'files'}\"")
    return 0


def cmd_report(args) -> int:
    from rag.crawl.crawler import Summary
    from rag.crawl.state import CrawlState

    state = CrawlState(Path(args.out) / "crawl.db")
    try:
        files = state.files()
        by_type: dict = {}
        by_host: dict = {}
        from urllib.parse import urlsplit

        for f in files:
            for key, bucket in ((f["type"], by_type), (urlsplit(f["url"]).hostname or "?", by_host)):
                b = bucket.setdefault(key, {"files": 0, "bytes": 0})
                b["files"] += 1
                b["bytes"] += f["bytes"]
        s = Summary(saved_bytes=sum(f["bytes"] for f in files), saved_files=len(files), stop_reason="(report)",
                    by_type=by_type, by_host=by_host, statuses=state.status_counts(),
                    skip_reasons=state.skip_reasons(), seconds=0)
        _print_summary(s)
        if args.top:
            print("\nHighest-scoring files:")
            for f in sorted(files, key=lambda f: -(f["relevance"] or 0))[: args.top]:
                print(f"  {f['relevance']:7.1f}  {human(f['bytes']):>10}  {f['path']}  <- {f['url']}")
    finally:
        state.close()
    return 0


def cmd_preset(args) -> int:
    sys.stdout.write(_preset_text(args.name))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag-crawl", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("run", help="crawl (or resume) until the budget is reached or nothing on-topic is left")
    s.add_argument("config", nargs="?", help="topic TOML file")
    s.add_argument("out", nargs="?", help="output folder")
    s.add_argument("--preset", help="use a built-in topic preset instead of a config file (e.g. example, a template)")
    s.add_argument("--budget", required=True, help="target total size, e.g. 15GB")
    s.add_argument("--max", help="hard cap, never exceeded (default: budget + 15%%)")
    s.add_argument("--workers", type=int)
    s.add_argument("--delay", type=float, help="minimum seconds between requests to one host")
    s.add_argument("--seed", action="append", help="replace the seed URLs (repeatable)")
    s.add_argument("--no-html", action="store_true", help="follow web pages but save only documents")
    s.add_argument("--max-requests", type=int, help="stop after this many requests (for trial runs)")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_run)

    s = sub.add_parser("report", help="summarize a crawl folder")
    s.add_argument("out")
    s.add_argument("--top", type=int, default=0, help="list the N most relevant files")
    s.set_defaults(fn=cmd_report)

    s = sub.add_parser("preset", help="print a built-in preset")
    s.add_argument("name")
    s.set_defaults(fn=cmd_preset)

    args = p.parse_args(argv)
    if args.cmd == "run" and args.preset and args.config and not args.out:
        args.out, args.config = args.config, None
    if args.cmd == "run" and not args.out:
        p.error("run needs an output folder")
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
