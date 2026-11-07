-- tables.db: spreadsheet regions and large document tables as real tables.
-- Each data table is named t_<slug>, has typed snake_case columns and keeps the
-- source row number in _row for citation.

CREATE TABLE IF NOT EXISTS _catalog (
  table_name    TEXT PRIMARY KEY,
  doc_id        INTEGER NOT NULL,       -- documents.doc_id in rag.db
  source_path   TEXT NOT NULL,          -- display path of the source file
  origin        TEXT NOT NULL,          -- sheet | csv | doc_table
  sheet         TEXT,
  cell_range    TEXT,
  page          INTEGER,
  caption       TEXT,
  n_rows        INTEGER NOT NULL,
  columns_json  TEXT NOT NULL,          -- [{name, header, type, min, max, examples, nulls}]
  card_chunk_id INTEGER                 -- the table card's chunk_id in rag.db
);
CREATE INDEX IF NOT EXISTS _catalog_doc ON _catalog(doc_id);
