"""Embedding backends. All return L2-normalized float32 rows of ``dims`` columns.

Default: Qwen3-Embedding-8B (Q4_K_M GGUF) served by llama.cpp's llama-server:

    llama-server -m Qwen3-Embedding-8B-Q4_K_M.gguf --embedding --pooling last \
                 -ub 8192 -c 8192 --host 127.0.0.1 --port 8081 -ngl 99

``-ngl 99`` offloads all layers to the GPU (CUDA or Vulkan builds); drop it for CPU.
Queries are embedded as ``Instruct: <task>\\nQuery:<question>``; documents as-is.
Storing fewer dims than the model outputs (MRL) truncates, then re-normalizes.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

import numpy as np

EOS = "<|endoftext|>"


class EmbedderError(RuntimeError):
    pass


class Embedder(Protocol):
    backend: str
    model: str
    dims: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...
    def info(self) -> dict[str, str]: ...


def _finish(rows: list[list[float]] | np.ndarray, dims: int) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[1] < dims:
        raise EmbedderError(f"model returned {arr.shape[1]} dims; config asks for {dims}")
    arr = arr[:, :dims]
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms


class _Base:
    backend = "base"

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.model = str(cfg["model"])
        self.dims = int(cfg["dims"])
        self.batch_size = max(1, int(cfg.get("batch_size", 8)))
        self.template = str(cfg.get("query_template", "Instruct: {task}\nQuery:{query}"))
        self.instruction = str(cfg.get("query_instruction", ""))

    def query_text(self, text: str) -> str:
        if not self.instruction:
            return text
        return self.template.format(task=self.instruction, query=text)

    def info(self) -> dict[str, str]:
        return {
            "embed_backend": self.backend,
            "embed_model": self.model,
            "embed_dims": str(self.dims),
            # The exact query prefix; the retrieval server must reproduce it.
            "query_instruction": self.query_text("{query}"),
        }

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.query_text(text)])[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch_size):
            out.append(self._embed(texts[i:i + self.batch_size]))
        return np.vstack(out) if out else np.zeros((0, self.dims), dtype=np.float32)

    def _embed(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError


class LlamaServerEmbedder(_Base):
    """OpenAI-compatible /v1/embeddings on llama.cpp's llama-server."""

    backend = "llama-server"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        self.url = str(cfg.get("url", "http://127.0.0.1:8081")).rstrip("/")
        self.timeout = float(cfg.get("timeout_s", 300))
        self.append_eos = bool(cfg.get("append_eos", False))

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.url}/v1/embeddings", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", "replace")[:500]
                raise EmbedderError(f"llama-server returned HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_exc = exc
                time.sleep(1.5 * (attempt + 1))
        raise EmbedderError(
            f"cannot reach llama-server at {self.url} ({last_exc}).\n"
            "Open RAG Workbench and start its embedding model (it serves on port 8081), or start one yourself:\n"
            "  llama-server -m Qwen3-Embedding-8B-Q4_K_M.gguf --embedding --pooling last "
            "-ub 8192 -c 8192 --port 8081 -ngl 99"
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        inputs = [t + EOS if self.append_eos else t for t in texts]
        data = self._post({"input": inputs, "model": self.model, "encoding_format": "float"})
        rows = sorted(data["data"], key=lambda d: d["index"])
        return _finish([r["embedding"] for r in rows], self.dims)


class LlamaCppEmbedder(_Base):
    """In-process llama-cpp-python with a local GGUF file."""

    backend = "llama-cpp"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        try:
            import llama_cpp
        except ImportError as exc:
            raise EmbedderError("install llama-cpp-python (pip install rag-builder[llamacpp])") from exc
        path = cfg.get("model_path")
        if not path:
            raise EmbedderError("embedding.model_path must point to the .gguf file for backend llama-cpp")
        n_ctx = int(cfg.get("n_ctx", 8192))
        self.append_eos = bool(cfg.get("append_eos", False))
        self.llm = llama_cpp.Llama(
            model_path=str(path), embedding=True, pooling_type=llama_cpp.LLAMA_POOLING_TYPE_LAST,
            n_gpu_layers=int(cfg.get("n_gpu_layers", -1)), n_ctx=n_ctx, n_batch=n_ctx, n_ubatch=n_ctx,
            n_threads=int(cfg.get("threads") or 0) or None, verbose=False,
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        inputs = [t + EOS if self.append_eos else t for t in texts]
        return _finish(self.llm.embed(inputs, normalize=True), self.dims)


class Qwen3OnnxEmbedder(_Base):
    """Qwen3-Embedding-0.6B on ONNX Runtime via qwen3-embed (CPU-only option)."""

    backend = "qwen3-onnx"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)
        try:
            from qwen3_embed import TextEmbedding
        except ImportError as exc:
            raise EmbedderError("install qwen3-embed (pip install rag-builder[onnx])") from exc
        self.m = TextEmbedding(
            model_name=self.model, cache_dir=cfg.get("models_dir") or None,
            threads=int(cfg.get("threads") or 0) or None,
        )

    def _embed(self, texts: list[str]) -> np.ndarray:
        return _finish(list(self.m.embed(texts, batch_size=self.batch_size)), self.dims)

    def embed_query(self, text: str) -> np.ndarray:
        return _finish(list(self.m.embed([self.query_text(text)], batch_size=1)), self.dims)[0]


class HashEmbedder(_Base):
    """Deterministic lexical vectors (signed feature hashing of words and word pairs).

    Lets the whole pipeline run and be tested without a model. Packs built with it
    are only good for testing.
    """

    backend = "hash-test"

    def __init__(self, cfg: dict[str, Any]):
        super().__init__(cfg)

    def query_text(self, text: str) -> str:
        return text

    def _embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", text.lower())
            feats = words + [f"{a}_{b}" for a, b in zip(words, words[1:])]
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dims] += 1.0 if (h >> 63) & 1 else -1.0
        return _finish(out, self.dims)


BACKENDS = {
    "llama-server": LlamaServerEmbedder,
    "llama-cpp": LlamaCppEmbedder,
    "qwen3-onnx": Qwen3OnnxEmbedder,
    "hash-test": HashEmbedder,
}


def make_embedder(cfg: dict[str, Any]) -> _Base:
    backend = cfg.get("backend", "llama-server")
    if backend not in BACKENDS:
        raise EmbedderError(f"unknown embedding backend {backend!r}; choose from {', '.join(BACKENDS)}")
    if int(cfg["dims"]) > int(cfg.get("native_dims", cfg["dims"])):
        raise EmbedderError("embedding.dims cannot exceed embedding.native_dims")
    return BACKENDS[backend](cfg)
