"""Configuration: defaults in code, overridden by a ``config.toml``.

Every workspace gets its own ``config.toml`` (written by ``rag init``). The
retrieval side must use the same embedding settings the Builder used; those are
also recorded in the ``meta`` table of ``rag.db`` and in the pack manifest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_INCLUDE = [
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm",
    ".csv", ".xlsx", ".xlsm", ".sql", ".cs", ".py",
]

DEFAULT_EXCLUDE = [
    "~$*", "Thumbs.db", "desktop.ini", ".DS_Store",
    "**/.git/**", "**/.svn/**", "**/.vs/**", "**/.idea/**",
    "**/bin/**", "**/obj/**", "**/__pycache__/**", "**/.venv/**", "**/venv/**",
    "**/node_modules/**", "**/*_files/**",
    "*.Designer.cs", "*.g.cs", "*.g.i.cs", "*.AssemblyInfo.cs",
    "**/Migrations/*.cs",
]

DEFAULTS: dict[str, Any] = {
    "embedding": {
        # Backends:
        #   llama-server - HTTP to llama.cpp's llama-server (--embedding --pooling last);
        #                  runs on CUDA, Vulkan (e.g. Radeon iGPU) or CPU. Default.
        #   llama-cpp    - in-process llama-cpp-python with a local .gguf file.
        #   qwen3-onnx   - Qwen3-Embedding-0.6B on ONNX Runtime (CPU-only option).
        #   hash-test    - deterministic lexical stand-in for tests; never for real packs.
        "backend": "llama-server",
        "model": "Qwen/Qwen3-Embedding-8B-GGUF:Q4_K_M",
        "url": "http://127.0.0.1:8081",
        "model_path": "",        # llama-cpp: path to the .gguf file
        "native_dims": 4096,     # what the model outputs
        "dims": 4096,            # what gets stored (MRL truncation if smaller, e.g. 2048 or 1024)
        "query_template": "Instruct: {task}\nQuery:{query}",
        "query_instruction": (
            "Given a question about internal technical documents, "
            "retrieve passages that answer it"
        ),
        "append_eos": False,     # set true if llama.cpp does not end inputs with <|endoftext|>
        "batch_size": 8,
        "timeout_s": 300,
        "n_gpu_layers": -1,      # llama-cpp: -1 = all layers on the GPU
        "n_ctx": 8192,           # llama-cpp context per sequence
        "tokenizer_path": "",    # Qwen3 tokenizer.json for exact token counts (optional)
        "models_dir": "",        # qwen3-onnx cache dir
        "threads": 0,
    },
    "vectors": {
        # Where a NEW workspace keeps its vectors: "sqlite-vec" (inside rag.db; what the retrieval
        # server reads today) or "shammy" (vectors.shammy; smaller and faster, needs dims to be a
        # power of two). An existing workspace is switched with `rag convert`, not by editing this.
        "backend": "sqlite-vec",
    },
    "chunking": {
        "target_tokens": 400,
        "max_tokens": 500,
        "min_tokens": 40,
        "overlap_tokens": 60,
        "long_section_tokens": 800,
        "topic_shift_percentile": 95,
        "code_max_tokens": 800,
        "table_inline_max_rows": 30,
        # A table too large to sit inline is stored in tables.db for SQL. Up to this many rows it
        # is also chunked in full (row groups under the column names), so a question about one
        # row finds it; beyond that only a preview is chunked.
        "table_chunk_max_rows": 1000,
        "row_chunks": True,          # row-group chunks for small text-heavy tables
        "row_chunk_size": 25,
        "row_chunk_max_rows": 200,
        # What each chunk's embedded text starts with: 1 = file › heading trail; 2 = also a line
        # naming the document (title, document number, revision). New knowledge bases are made
        # with 2; one built earlier keeps what it was built with (see embed_context in rag.db).
        "context": 1,
    },
    "pdf": {
        "min_text_chars": 100,
        "min_printable_ratio": 0.85,
        "ocr_scale": 2.5,            # render scale for OCR (72 dpi * scale)
        "figure_min_area": 0.05,     # fraction of page area
        "figure_ocr": True,
        # Flowcharts drawn as vector shapes (Visio, PowerPoint, draw.io exports) are rebuilt as
        # numbered steps with their connections instead of being read as jumbled text.
        "diagrams": True,
        # Values typed into a fillable PDF form are read from its fields (they aren't in the
        # page text) and placed where the field is.
        "form_fields": True,
        "furniture_min_pages": 3,
    },
    "figures": {
        # A picture of a table or spreadsheet becomes a real table (rows and columns rebuilt from
        # where the OCR found each cell); off = its text is kept as one paragraph.
        "table_ocr": True,
        # OpenAI-compatible vision endpoint on the inference server; empty = off.
        "describe_url": "",
        "describe_model": "",
        "describe_timeout_s": 60,
    },
    "library": {
        "copy_originals": True,
    },
    "ingest": {
        "include": DEFAULT_INCLUDE,
        "exclude": DEFAULT_EXCLUDE,
        "max_file_mb": 200,
        "libreoffice_path": "",      # for legacy .doc; auto-detected when empty
        "sql_dialect": "",           # sqlglot dialect hint, e.g. "tsql", "oracle"
    },
    "search": {
        "fts_k": 40,
        "vec_k": 40,
        "rrf_k": 60,
        "keep": 6,               # results returned (no reranker in v1)
        "neighbors": 1,
        "max_result_tokens": 6000,
        # With a current spots file (structure.spots), a hit inside a procedure comes back with its
        # procedure: whole when it is at most procedure_tokens long, else the hit's step with up
        # to procedure_window steps on each side that still fit.
        "procedures": True,
        "procedure_tokens": 1500,
        "procedure_window": 1,
        "procedure_hits": 3,         # at most this many results from one procedure
    },
    "rerank": {
        # Optional second pass: a small reranker model reorders the best fused candidates before
        # the cut. Needs a /v1/rerank endpoint (llama-server --reranking with a reranker GGUF such
        # as Qwen3-Reranker-0.6B). Off by default; `rag search --rerank` turns it on for one search.
        # The RAG_RERANK_URL environment variable overrides url.
        "enabled": False,
        "url": "http://127.0.0.1:8082",
        "model": "",
        "candidates": 20,
        "timeout_ms": 15000,
    },
    "sql": {
        "row_limit": 200,
        "time_limit_ms": 2000,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    data: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULTS))
    path: Path | None = None

    def __getitem__(self, section: str) -> dict[str, Any]:
        return self.data[section]

    @classmethod
    def load(cls, path: Path | None) -> "Config":
        if path is None or not Path(path).exists():
            return cls(path=Path(path) if path else None)
        with open(path, "rb") as fh:
            user = tomllib.load(fh)
        return cls(data=_deep_merge(DEFAULTS, user), path=Path(path))

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        return Config(data=_deep_merge(self.data, overrides), path=self.path)

    def fingerprint(self) -> str:
        """Hash of the settings that change what a build or search returns."""
        relevant = {k: self.data[k] for k in ("embedding", "chunking", "search")}
        blob = json.dumps(relevant, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]


def render_toml(data: dict[str, Any]) -> str:
    """Write a config dict as TOML (only the value types used here)."""

    def fmt(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, list):
            inner = ",\n  ".join(fmt(v) for v in value)
            return f"[\n  {inner},\n]" if value else "[]"
        raise TypeError(f"unsupported TOML value: {value!r}")

    lines: list[str] = [
        "# RAG Builder workspace config. Embedding settings must match the",
        "# retrieval side; changing them requires a full re-embed (rag build --force).",
        "",
    ]
    for section, values in data.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {fmt(value)}")
        lines.append("")
    return "\n".join(lines)
