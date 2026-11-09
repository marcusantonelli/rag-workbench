"""Entry point PyInstaller freezes into rag-engine(.exe)."""

import sys

from rag.app.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
