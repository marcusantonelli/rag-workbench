# RAG Builder

Builds a portable RAG **pack** from a mixed document set, and lets you test
retrieval against it before the C# retrieval server exists.

A pack (`.ragpack`, a zip) holds:

| File | Contents |
| --- | --- |
| `rag.db` | documents, chunks, FTS5 keyword index, sqlite-vec vectors |
| `tables.db` | spreadsheets and large document tables as real, typed SQL tables, plus `_catalog` |
| `assets/` | figure crops from PDFs and Word files |
| `manifest.json` | pack id, embedding model and dims, counts, SHA-256 of every file |

Inputs: text PDFs, scanned PDFs (OCR per page), Word (.docx; .doc via LibreOffice),
TXT, Markdown, HTML/HTM, CSV, XLSX, SQL, C#, Python.

This is the Builder engine and CLI. The desktop app (Tauri) will drive the same
engine through `rag build --json`. Reranking is deliberately left out of v1.

## Install

Python 3.11+ (python.org builds on Windows support SQLite extension loading).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux
pip install -e ".[dev]"
```

Optional extras: `[docling]` (higher-quality PDF layout and tables), `[llamacpp]`
(in-process GGUF embedding), `[onnx]` (Qwen3-Embedding-0.6B on CPU).

## Embedding model: Qwen3-Embedding-8B, Q4_K_M

Download `Qwen3-Embedding-8B-Q4_K_M.gguf` (4.68 GB) from
`Qwen/Qwen3-Embedding-8B-GGUF` on Hugging Face, and a llama.cpp release build
for your hardware (CUDA for NVIDIA, Vulkan for AMD/Intel GPUs including the
Radeon 760M, or CPU). Then run it as an embedding server:

```bash
llama-server -m Qwen3-Embedding-8B-Q4_K_M.gguf --embedding --pooling last ^
  -ub 8192 -c 8192 --host 127.0.0.1 --port 8081 -ngl 99
```

`-ngl 99` puts every layer on the GPU; omit it for CPU only. Check the
connection, vector size and query prefix with:

```bash
rag check-embedder my-ws
```

**One-time check:** Qwen3 embeddings are pooled from the final `<|endoftext|>`
token. Run `llama-embedding -m Qwen3-Embedding-8B-Q4_K_M.gguf -p "test" --pooling last --verbose-prompt`
and confirm the last token printed is `<|endoftext|>` (id 151643). If it is not,
set `append_eos = true` under `[embedding]`.

**Dimensions:** the model outputs 4096 dims and supports truncation (MRL) down to 32.
At 4096 dims each chunk's vector is 16 KB, so 50,000 chunks is about 800 MB of
vectors that the retrieval side scans per query. Setting `dims = 2048` halves that
with a small quality cost; measure both with `rag eval`. Changing dims requires
`rag build --force`.

**Retrieval must use the same model.** Query vectors have to come from the same
GGUF and quantization, with the same query prefix (recorded in `rag.db` meta and
the manifest). On a CPU-only retrieval server an 8B query embedding takes noticeably
longer than a 0.6B one; serving it from the GPU inference server is the alternative.

## Use

```bash
rag init my-ws                                # writes my-ws/config.toml; review it
rag build my-ws "D:\KB\Procedures" "D:\KB\Code" --prune
rag status my-ws
rag search my-ws "minimum pull strength for 5 mil aluminum wire"
rag tables my-ws
rag describe my-ws t_burn_in_yields_q3_yields
rag sql my-ws "SELECT device, SUM(test_fail) FROM t_burn_in_yields_q3_yields GROUP BY device"
rag export my-ws kb.ragpack
rag search kb.ragpack "seal leak root cause"   # works on packs too
rag eval my-ws examples/golden.example.jsonl
```

To try everything without a model: `rag init test-ws --embedder hash-test`
(a deterministic lexical stand-in; never ship packs built with it).

Rebuilds are incremental: unchanged files (same SHA-256) are skipped, changed
files replace their old chunks and tables in one transaction, `--prune` removes
documents deleted from the given folders, and identical text is embedded once.
A file that fails is logged (`rag status`) and never stops the build.

## How each type is handled

| Input | Extractor | Becomes |
| --- | --- | --- |
| Text PDF | pdfplumber: lines, font-size headings, ruled tables, embedded images; two-column pages; repeated headers/footers removed | Markdown blocks per page |
| Scanned PDF page | RapidOCR (ONNX, models bundled) on the rendered page | Paragraph blocks with OCR confidence |
| Word | python-docx: heading styles, lists, tables, embedded pictures (OCR'd) | Markdown blocks |
| TXT / Markdown | line-based; front matter, pipe tables, code fences | Blocks with line ranges |
| HTML | nav/footer/scripts dropped; headings, lists, tables, `<pre>` | Markdown blocks |
| CSV / XLSX | region detection, merged and two-row headers, typed columns, notes | `tables.db` tables + a searchable table card |
| SQL | statement splitter aware of strings, comments, BEGIN/END, `GO`; sqlglot for referenced tables | One chunk per statement; bulk INSERTs summarized |
| C# | tree-sitter: type headers (doc comments, fields, member list) and one chunk per method | Code chunks with symbol paths |
| Python | `ast`: module header, class headers, one chunk per function | Code chunks with symbol paths |

Figures get their crop saved, their text OCR'd and, if `[figures] describe_url`
points at an OpenAI-compatible vision endpoint (e.g. on the inference server),
a short description. Logos repeated across pages are dropped.

Document tables over 30 rows also go to `tables.db`; the document keeps a
pointer chunk. Spreadsheet quirks can be overridden per workbook in
`<workspace>/sheets.toml` (see `examples/sheets.example.toml`).

PDF backend: `pdf.backend = "auto"` uses Docling when `[docling]` is installed and
working, otherwise the native extractor. **The Docling path was not exercised in
development** (its models download from Hugging Face, which was unreachable); set
`pdf.backend = "docling"` on first real use to see any errors instead of falling back.

## Chunking

Structure first, about 400 tokens per prose chunk with ~60 tokens of overlap;
sections over 800 tokens without sub-headings are split where the topic shifts
(sentence embeddings from the same model). Tables, figures, code and SQL are
chunked on their own rules. Every chunk's embedded text starts with
`file › heading trail › symbol`; code chunks add split identifiers
(`GetLotYield` → `get lot yield`) so keyword search matches both spellings.
Set `embedding.tokenizer_path` to Qwen3's `tokenizer.json` for exact token counts
(otherwise a conservative estimate is used).

## Retrieval (test path, no reranker)

`rag search` embeds the query with the instruction prefix, takes the top 40 from
the vector index and the top 40 from FTS5 (identifiers as exact phrases), fuses
them with Reciprocal Rank Fusion (k = 60), drops duplicate text, expands prose
hits with their neighbours from the same section, and returns cited results plus
any table cards that ranked. `rag search --json` is the shape the agent will get;
plain output is the compact Markdown form. `Searcher(rerank=...)` is the v2 hook.

`rag sql` runs agent-style SQL read-only: an authorizer allows only reads, one
statement per call, rows capped at 200, a 2-second time limit.

## Desktop app integration

`rag build WS PATH... --json` prints one JSON object per line: `start`,
`file_start`, `stage` (page / ocr / chunk / embed progress), `file_done`,
`file_failed`, `file_unchanged`, `file_removed`, `skipped`, `done`.

## Tests

```bash
pytest
```

The suite generates a corpus with every file type (`tests/fixtures/make_fixtures.py`)
and covers extraction, chunking, incremental builds, pack verification, search,
the SQL guard and the llama-server client (against a fake server).

## Layout

```text
src/rag/
  cli.py                 rag command
  config.py              defaults + config.toml
  eval.py                golden-set evaluation
  builder/engine.py      discovery, incremental build, progress events
  core/
    db/                  schemas, store (rag.db), tables (tables.db), sqlguard, pack
    ingest/              router + one extractor per type, OCR, figures
    chunk/               chunker, tokens, identifiers, topic-shift breakpoints
    models/embed.py      llama-server, llama-cpp, qwen3-onnx, hash-test
    retrieve/search.py   hybrid search (reference for the C# server)
tests/                   pytest suite and fixture generator
examples/                golden set and sheets.toml examples
```
