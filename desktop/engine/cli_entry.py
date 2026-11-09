"""Entry point PyInstaller freezes into rag(.exe): the command-line tool."""

import sys

from rag.cli import main

if __name__ == "__main__":
    sys.exit(main())
