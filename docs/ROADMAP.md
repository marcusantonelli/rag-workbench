# Long-term dev notes

Decisions and plans that span releases. Short entries; details live in the code, the
release notes and the Shammy plan.

## v1.0: rename to Shammy

Done in v1.0, not in a prerelease.

| Today | From v1.0 |
|---|---|
| RAG Workbench (desktop app) | **Shammy Workbench** |
| `.shammy` vector store and file format | **ShammyDB** |
| `rag` and `shammy` command lines | **Shammy CLI** (one `shammy` command) |
| `shammy-rag` shared retrieval library (Python, C, C#) | **Shammy Engine** |

Things the rename touches: app name, installer and bundle ids, the data folder, the CLI entry
points (keep `rag` as an alias for a release), Python package names, the `.ragworkspace`
marker and `.ragpack` extension (read the old names forever), docs, and the GitHub repository.

## 0.4.1 / 0.4.2

- Page headers and footers: keep the document number, revision and title once per document as
  metadata (shown in citations); detect them by position and repetition over runs of pages,
  including odd/even and per-section variants; remove them from scanned pages too.
- Word text boxes (DrawingML and VML), without reading AlternateContent twice.
- Forms on scanned pages: label/value pairs and checkbox states from OCR.
- Procedure retrieval (0.4.1):
  - Step metadata as real fields: procedure id, step number, document number, revision.
  - Return the whole procedure, or a window around the matched step (parent/child retrieval).
  - Filters (document number, latest revision, type) and grouping of hits by procedure.
  - Cross-document links ("see WI-2210") later.
  - Compatibility: new columns are optional. 0.4.0 packs and knowledge bases open unchanged and
    fall back to document + heading path; vectors are never rebuilt for this. An in-place
    upgrade fills the fields from stored text. 0.4.0 must still open 0.4.1 packs. Test: 0.4.0
    test packs give the same results under 0.4.1.
- Auto-update (0.4.1):
  - Tauri updater plugin; `latest.json` and signed installers on the GitHub releases
    (`releases/latest/download/latest.json`). The repo is public.
  - Signing key secrets: `TAURI_SIGNING_PRIVATE_KEY`, `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`.
  - Public key (minisign 76327FDFB6E9D185), for `tauri.conf.json`:
    `dW50cnVzdGVkIGNvbW1lbnQ6IG1pbmlzaWduIHB1YmxpYyBrZXk6IDc2MzI3RkRGQjZFOUQxODUKUldTRjBlbTIzMzh5ZHJURmtmUDVJVUpFUUpzZE9ucWtTYTAyM3dzdHZDU2RBUzdtakdocDdnbHIK`
  - Shows the version and notes first; never silent. Waits for a running build. Stops the engine
    and helper models before installing. A Settings switch turns checking off.
    `rag update --check` for the CLI.
  - 0.4.1 is installed by hand once; later versions update themselves.
- Build progress (0.4.1): no more "0 chunks" with nothing moving.
  - Overall bar: files done of total, and pages done of total pages, with time elapsed and an
    estimate.
  - Per file, the current stage with its own progress: reading page 12/40, OCR, picture 3/9,
    chunking, embedding 160/420 chunks, writing.
  - Chunk counts climb as embedding batches finish, not only at the end.
  - Recent activity list (finished, skipped as unchanged, failed with the reason). The same
    progress shows in the CLI as a bar.

## To measure on real machines (0.4.0)

- Picture model (Qwen3-VL 4B and 2B): seconds per picture and memory on GPU and on CPU, next to
  the embedding model; whether the GPU-to-CPU fallback triggers; description quality on photos
  and screenshots from work instructions.
- Reranker (Qwen3-Reranker 0.6B, and any 4B/8B file tried): added search time, and whether the
  order is better on real questions.
- Scanned pages with tables: time per page with the second reads, and how many cells they fix.
- Shammy at scale: `shammy bench` on a large real knowledge base to set the stage 1 default, and
  the thread speed-up on 6-8 cores.
- Windows: install, first-run downloads, both helper models starting, a build and search on a
  Shammy knowledge base while another window searches it, `rag convert` and compaction.
- Retrieval server: `ShammyRetriever` in the .NET MCP server against `ragpack/1` and `ragpack/2`.
