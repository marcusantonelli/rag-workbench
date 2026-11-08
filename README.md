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

If `rag` reports that Python's sqlite3 cannot load extensions (common with
pyenv or other from-source Python builds), run `pip install sqlean.py`; it is
picked up automatically. Optional extras: `[sqlite]` (that same package), `[docling]` (higher-quality PDF layout and tables), `[llamacpp]`
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
rag eval my-ws golden.jsonl --report eval.md --out eval.json   # + the chunks each question retrieved
```

`--report` writes a readable Markdown file: every question with its top 6
retrieved chunks (file, page, section, full text), the correct ones marked ✅,
and the first correct chunk appended when it ranked lower. Unanswerable
questions show what the agent would be handed. `--show N` changes how many
chunks are listed, and adds them to the `--out` JSON as well.

To try everything without a model: `rag init test-ws --embedder hash-test`
(a deterministic lexical stand-in; never ship packs built with it).

Rebuilds are incremental: unchanged files (same SHA-256) are skipped, changed
files replace their old chunks and tables in one transaction, `--prune` removes
documents deleted from the given folders, and identical text is embedded once.
Files read by an older extractor version are re-extracted on the next build
(chunks whose text didn't change keep their embeddings). A file that fails is logged (`rag status`) and never stops the build.

## How each type is handled

| Input | Extractor | Becomes |
| --- | --- | --- |
| Text PDF | pdfplumber: lines, font-size headings, ruled tables, embedded images; multi-column layouts (sidebars, 2–3 columns, full-width titles) read column by column; repeated headers/footers removed | Markdown blocks per page |
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

## Test corpus with known answers

`tools/make_test_corpus.py` writes a fictional semiconductor back-end plant's
documents (Halden Microsystems: 29 ingestible files across every type, plus files
that must be excluded) with 47 questions, expected answers, answer locations and,
for table questions, the SQL that returns the answer:

```bash
python tools/make_test_corpus.py halden
rag init halden-ws
rag build halden-ws halden/docs
rag eval halden-ws halden/questions/golden.jsonl
```

`halden/questions/ANSWERS.md` is the answer key for asking the same questions
through the agent. The corpus contains the traps real document sets have: a
superseded revision beside the current one, a byte-identical copy, a two-column
newsletter, a table spanning two pages, a table without ruling lines, scanned and
half-scanned PDFs, merged Word and Excel headers, a hidden sheet, a cp1252 log,
a semicolon CSV, and generated code that must be skipped.

## rag-crawl: collect public documents on a topic

`rag-crawl` downloads on-topic documents from the web, only in the file types the
Builder reads, until it reaches a size target:

```bash
rag-crawl run --preset opcenter-semiconductor opcenter-semi --budget 2GB
rag-crawl report opcenter-semi --top 20
rag build my-ws opcenter-semi/files
```

- **Budget:** `--budget` is the target; the crawl stops starting new downloads
  once the saved total reaches it. `--max` (default budget + 15%) is a hard cap
  that is never exceeded, even with parallel downloads. If the topic runs out of
  material first, it stops and says so ("frontier exhausted"); it never pads.
- **Resume:** run the same command again to continue, or with a larger budget to
  collect more. State lives in `OUT/crawl.db`; `OUT/manifest.jsonl` records each
  file's source URL, size, title and relevance.
- **Quality:** a file is kept only if it mentions the topic terms enough
  (`topic.terms`, `min_term_hits`) and, if set, a second subject too
  (`topic.also_required`, used by the semiconductor preset to drop Medical
  Device and Process material). Thin pages, listing/tag pages, non-English
  pages, unreadable files and duplicates (same bytes, or same article text at
  another URL) are skipped.
- **Focus:** links whose URL or anchor text names a term are always followed;
  other links only one hop from the last on-topic page (`max_tunnel`). Other
  domains are visited one hop out, and only for links that name a term.
- **Manners:** robots.txt and crawl-delay are obeyed, each host gets one
  request at a time with at least `delay_s` between requests, HTTP 429/503
  pause the host (honouring Retry-After), nothing behind a login is fetched,
  and sites hosting uploaded copies of licensed manuals (Scribd and similar)
  are blocked by default.
- **Network problems:** an unreachable site is retried twice, then dropped; if no
  site can be reached the crawl stops and says so, and the next run retries them.
  `HTTPS_PROXY` is honoured.
- **Discovery:** seeds, then the seed hosts' sitemaps, then links. Web search is
  optional: set `[search] provider = "brave"` and the `BRAVE_API_KEY`
  environment variable (Brave Search API, free tier), or point
  `provider = "searxng"` at your own SearXNG instance.

Presets: `opcenter-semiconductor` (Opcenter Execution Semiconductor / Camstar
Semiconductor Suite) and `opcenter` (all of Opcenter). Print one with
`rag-crawl preset opcenter-semiconductor > semi.toml`, edit it, and run
`rag-crawl run semi.toml OUT --budget 2GB`.

What it cannot reach: the Opcenter manuals on docs.sw.siemens.com (crawlers are
disallowed there) and anything in the Siemens Support Center (login). Public
material (fact sheets, what's-new sheets, white papers, case studies, blog posts,
trade press) is likely a few hundred MB to a few GB, not tens of GB.

Size planning: 1 GB of PDFs is very roughly 30,000-60,000 chunks. Measure your
embedding rate with `rag check-embedder` and a small build before embedding a
large crawl; with the 8B model, a GPU makes the difference between hours and days.

## Tests

```bash
pytest
```

The suite generates a corpus with every file type (`tests/fixtures/make_fixtures.py`)
and covers extraction, chunking, incremental builds, pack verification, search,
the SQL guard and the llama-server client (against a fake server). It also builds
the Halden corpus and checks that every golden answer location exists and every
SQL answer is right, and runs `rag-crawl` against a fake two-host web (robots,
sitemaps, redirects, login walls, rate limits, duplicates, languages, budgets,
resume, search).

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
  crawl/                 rag-crawl: crawler, robots/fetching, relevance, presets
tools/make_test_corpus.py  Halden test corpus, golden questions, answer key
tests/                   pytest suite and fixture generator
examples/                golden set and sheets.toml examples
```
