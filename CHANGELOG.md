# Changelog

One line per version; the full notes for 0.4.0 and later are in [docs/release-notes](docs/release-notes).

**Versions.** 0.x is the alpha: features are still being added, and settings, file formats and
behaviour can change between versions (knowledge bases keep opening; anything that needs a
rebuild says so in its notes). 1.0 betas come once the feature set is complete, then 1.0.0. A tag
with a hyphen (`v0.4.3-1`) is a test build, published as a GitHub pre-release and offered only to
apps that chose **Also offer pre-releases**.

| Version | Date | What changed |
|---|---|---|
| [0.4.2](docs/release-notes/0.4.2.md) | 2026-11-11 | Sections give every document with headings its context in search; PDF and Word layouts read by shared rules (block-arrow and Word-shape flowcharts, boxed panels, text boxes, titled tables); rebuild from scratch while open; Stop… Now; time left in all and per file; opt-in pre-releases. |
| [0.4.1](docs/release-notes/0.4.1.md) | 2026-11-11 | Procedures (spots file, whole-procedure answers, outlines, reading on), document name in embedded text for new knowledge bases, build progress, auto-update, domain-neutral project. |
| [0.4.0](docs/release-notes/0.4.0.md) | 2026-11-10 | Shammy vector store and shared search engine, optional reranker and picture model, rebuilt PDF flowcharts, procedure tables, pictures of tables, forms; Docling removed. |
| 0.3.3 | 2026-11-09 | Merge knowledge bases, workspace markers, `rag list` and `rag verify`. |
| 0.3.2 | 2026-11-09 | In-app terminal for `rag` commands. |
| 0.3.1 | 2026-11-09 | The `rag` command ships with RAG Workbench, with a command-line guide. |
| 0.3.0 | 2026-11-09 | RAG Workbench desktop app: open, search and evaluate packs, build and export knowledge bases, embedding model set-up for the hardware found. |
| 0.2.3 | 2026-11-07 | `rag eval --report` and `--show` list the chunks each question retrieved. |
| 0.2.2 | 2026-11-07 | PDFs with sidebars and several columns read column by column. |
| 0.2.1 | 2026-11-07 | `rag-crawl` gives up on unreachable sites quickly and retries them next run. |
| 0.2.0 | 2026-11-07 | `rag-crawl` for collecting public documents, a test corpus with known answers, reading fixes. |
| 0.1.1 | 2026-11-07 | Falls back to a drop-in sqlite3 when Python can't load extensions. |
| 0.1.0 | 2026-11-07 | First version: SQLite + sqlite-vec packs built from mixed documents. |
