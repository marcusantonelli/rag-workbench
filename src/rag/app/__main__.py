"""rag-engine: the desktop app's engine process.

    rag-engine --stdio            (what the app runs)
    rag-engine --http [PORT]      (browser dev mode; default port 8765)
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys


def main(argv: list[str] | None = None) -> int:
    multiprocessing.freeze_support()
    p = argparse.ArgumentParser(prog="rag-engine", description="RAG Workbench engine")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--stdio", action="store_true", help="JSON lines over stdin/stdout (default)")
    g.add_argument("--http", nargs="?", const=8765, type=int, metavar="PORT", help="serve HTTP on 127.0.0.1")
    p.add_argument("--version", action="store_true")
    args = p.parse_args(argv)
    if args.version:
        from rag import __version__

        print(__version__)
        return 0
    from rag.app import transport

    if args.http:
        return transport.serve_http(args.http)
    return transport.serve_stdio()


if __name__ == "__main__":
    sys.exit(main())
