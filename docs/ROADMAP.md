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

Sections for every document with headings (the spots file beyond procedures), general PDF and
Word layout reading (block-arrow and Word-shape flowcharts, boxed panels, text boxes, titled
tables), rebuild from scratch while
the knowledge base is open elsewhere, a steadier time-left estimate, Stop now. Details in
`docs/release-notes/0.4.2.md`.

## 0.4.3

- Query instruction editable by experienced users: the whole query template, with `${query}`
  (and `${task}`) placeholders instead of `{query}`. Changing it changes what is embedded for a
  question, so it belongs to the knowledge base (recorded like the embedding model) and Details
  warns when it differs from the one the knowledge base was built and evaluated with.
- Search: the "only documents" box becomes a multi-select filter listing the knowledge base's
  documents (by number and title, with a search field).
- Build tab: thumbnails of the pictures being described in the current document, with each
  description as it arrives.
- Low-confidence chunks (OCR confidence, table re-reads, figure text) flagged for review: a simple
  threshold for the first N documents, then outliers by standard deviation over the knowledge
  base. 0.4.3 only finds and lists them (Documents tab, `rag review`); editing them is 0.5.
- Carried over from the earlier 0.4.2 list: page headers and footers (document number, revision
  and title once per document as metadata; detected by position and repetition over runs of
  pages, odd/even and per-section variants; removed from scanned pages too); Word text boxes
  (DrawingML and VML, without reading AlternateContent twice); forms on scanned pages (label/value
  pairs and checkbox states from OCR); procedures grouped by revision ("latest revision only")
  and bulleted step lists in sections whose title reads as a procedure; the Documents tab listing
  a document's chunks while it is being built.
- Picture descriptions kept by picture (a hash of the image, the model and the prompt), so a
  file read again doesn't describe its pictures again. Then the reader version can be bumped to
  re-read every PDF and Word file with the 0.4.2 layout rules without hours of descriptions.
- Word: SmartArt (its data part holds the nodes and their connections), and flowcharts kept only
  as VML or as embedded Visio/EMF pictures (EMF needs converting before it can be described).
- Updates: a pre-release channel (Settings → Updates → "Also offer pre-releases"), which needs a
  second update feed because `releases/latest` never returns a pre-release.

## 0.5 (editing what was read)

Editing chunk text and metadata changes the knowledge base by hand, so it needs more than a
release's worth of UI: an edit has to survive a rebuild of its file (stored as an overlay keyed by
the source file's hash and the chunk's position and text hash, re-applied when the file is read
again and flagged when the text under it changed), be re-embedded on save, travel in packs, be
visible and undoable (an edit log with who and when), and keep both search engines in step.
- Edit a chunk's text (starting from the 0.4.3 review list), re-embedded on save.
- Metadata at document level (title, document number, revision, custom fields such as owner or
  area) and at chunk level (custom tags); custom fields usable as search filters and shown in
  citations.
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
