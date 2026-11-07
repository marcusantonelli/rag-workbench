-- rag.db: documents, chunks, keyword index (FTS5) and vectors (sqlite-vec).
-- {dims} is substituted at creation time from the embedding config.

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id        INTEGER PRIMARY KEY,
  source_key    TEXT NOT NULL UNIQUE,   -- normalized original path
  display_path  TEXT NOT NULL,          -- root folder name + relative path
  library_path  TEXT,                   -- copy inside the workspace library, if kept
  sha256        TEXT NOT NULL,
  file_type     TEXT NOT NULL,          -- pdf, docx, doc, txt, md, html, csv, xlsx, sql, cs, py
  title         TEXT,
  n_pages       INTEGER,
  size_bytes    INTEGER,
  mtime         REAL,
  extractor     TEXT,
  status        TEXT NOT NULL,          -- ok | partial | failed
  warnings      TEXT,                   -- JSON list
  n_chunks      INTEGER NOT NULL DEFAULT 0,
  ingested_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id     INTEGER PRIMARY KEY,
  doc_id       INTEGER NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  ordinal      INTEGER NOT NULL,        -- position within the document
  kind         TEXT NOT NULL,           -- prose | table | figure | code | sql | table_card | rows
  text         TEXT NOT NULL,           -- shown to the agent and cited
  embed_text   TEXT NOT NULL,           -- context header + text (+ split identifiers)
  text_hash    TEXT NOT NULL,
  heading_path TEXT,
  symbol       TEXT,                    -- code: Namespace.Class.Member
  page_start   INTEGER,
  page_end     INTEGER,
  line_start   INTEGER,
  line_end     INTEGER,
  sheet        TEXT,
  cell_range   TEXT,
  table_name   TEXT,                    -- table cards and large doc tables: target in tables.db
  ocr_conf     REAL,
  asset_path   TEXT,                    -- figures: saved image crop, relative to the pack
  n_tokens     INTEGER
);
CREATE INDEX IF NOT EXISTS chunks_doc  ON chunks(doc_id, ordinal);
CREATE INDEX IF NOT EXISTS chunks_hash ON chunks(text_hash);

-- Keyword index over the same text that gets embedded; rowid = chunk_id.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  embed_text,
  tokenize = 'porter unicode61'
);

-- One vector per chunk; metadata columns are filterable inside KNN queries.
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vec USING vec0(
  chunk_id  INTEGER PRIMARY KEY,
  embedding float[{dims}] distance_metric=cosine,
  file_type TEXT,
  doc_id    INTEGER
);

CREATE TABLE IF NOT EXISTS ingest_errors (
  path    TEXT,
  stage   TEXT,
  message TEXT,
  at      TEXT
);
