"""Token counting: exact with the Qwen3 tokenizer.json, estimated without it."""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

_PIECES = re.compile(r"\w+|[^\w\s]", re.UNICODE)


class TokenCounter:
    def __init__(self, tokenizer_path: str | None = None):
        self._tok = None
        if tokenizer_path and Path(tokenizer_path).exists():
            from tokenizers import Tokenizer

            self._tok = Tokenizer.from_file(str(tokenizer_path))
        self.exact = self._tok is not None
        self._count = lru_cache(maxsize=65536)(self._count_uncached)

    def _count_uncached(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        # Qwen's BPE averages roughly 1.1-1.3 tokens per word-or-symbol on English
        # technical text; err on the high side so chunks stay under the limits.
        pieces = _PIECES.findall(text)
        long_words = sum(1 for p in pieces if len(p) > 8)
        return max(1, math.ceil(len(pieces) * 1.1 + long_words * 0.6))

    def __call__(self, text: str) -> int:
        return self._count(text)
