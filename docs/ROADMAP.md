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

## 0.4.1 (released)

Procedures (spots file, procedure expansion, outlines, `get_procedure_steps`, document filter),
embedding context 2 for new knowledge bases, build progress, auto-update, domain-neutral project.
Details in `docs/release-notes/0.4.1.md`. Updater public key (minisign 76327FDFB6E9D185) is in
`desktop/src-tauri/tauri.conf.json`; the private key stays in the repository secrets.

## 0.4.2

- Page headers and footers: keep the document number, revision and title once per document as
  metadata (shown in citations); detect them by position and repetition over runs of pages,
  including odd/even and per-section variants; remove them from scanned pages too.
- Word text boxes (DrawingML and VML), without reading AlternateContent twice.
- Forms on scanned pages: label/value pairs and checkbox states from OCR.
- Procedures: grouping by revision ("latest revision only"); bulleted (unnumbered) step lists in
  sections whose title reads as a procedure.
- Documents tab: list a document's chunks while it is being built.

## To measure on real machines (0.4.0, 0.4.1)

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
- (0.4.1) Updates: install 0.4.1 by hand, publish 0.4.2 as a normal release with the signing
  secrets set, and check the banner, the download, the restart and that the engine and model
  servers come back.
- (0.4.1) Procedures on real knowledge bases: `rag spots` on a 0.4.0 knowledge base, how many
  procedures it finds, and whether procedure answers beat the old fragments on real questions.

## Experiments after 0.4.x (structure)

Measure first, with a question set from real documents (a quality-measurement set), on two knowledge bases
side by side: one built with context 1, one with context 2.
- Step graphs in the spots file: prerequisites, required tools and specs per step, branch targets
  from flowcharts, so "what do I need before step 7" is a walk instead of a search.
- Two-level search: one vector per procedure (the mean of its step vectors, or an embedded
  summary) to pick the procedure first, then the step.
- Revisions: link revisions of the same document, prefer the latest, answer "what changed".
- Step-shaped chunks: chunk boundaries at step boundaries (needs a rebuild).
