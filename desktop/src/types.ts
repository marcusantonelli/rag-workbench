// Shapes returned by the engine (src/rag/app/service.py).

export interface EngineError {
  code: string;
  message: string;
  hint?: string;
  data?: Record<string, unknown>;
}

export interface AppInfo {
  version: string;
  data_dir: string;
  workspaces_dir: string;
  platform: string;
  python: string;
  frozen: boolean;
  /** Version of the Shammy vector engine, or null when this build doesn't include it. */
  shammy: string | null;
}

export interface Device {
  id: string;
  name: string;
  total_mb: number;
  free_mb: number;
}

export interface Hardware {
  os: string;
  arch: string;
  cpu: string;
  threads: number | null;
  ram_gb: number | null;
  nvidia: { name: string; memory_mb: number; driver: string }[];
  gpus: string[];
}

export interface RuntimeStatus {
  mode: "managed" | "external";
  state: "stopped" | "starting" | "ready" | "failed" | "external";
  ready: boolean;
  url: string | null;
  error: string | null;
  accel: "auto" | "gpu" | "cpu";
  variant: string | null;
  recommended_variant: string;
  recommended_reason: string;
  llama_build: string;
  devices: Device[] | null;
  device_note: string;
  model_path: string | null;
  model_id: string | null;
  model_size: number | null;
  default_model: { id: string; file: string; url: string; approx_bytes: number };
  needs_setup: boolean;
  hardware: Hardware;
  external_url: string;
  log_tail: string[];
}

export interface RuntimeCheck {
  needs_runtime: boolean;
  ready: boolean;
  match: boolean;
  kb_model?: string;
  runtime_model?: string | null;
}

export interface KbSummary {
  id: string;
  kind: "pack" | "workspace";
  name: string;
  path: string;
  root: string;
  readonly: boolean;
  building: boolean;
  size_bytes: number;
  manifest: {
    name?: string;
    built_at?: string;
    app_version?: string;
    pack_id?: string;
    includes_originals?: boolean;
    counts?: Record<string, unknown>;
    embedding?: Record<string, unknown>;
  } | null;
  counts: {
    documents: number;
    failed: number;
    chunks: number;
    vectors: number;
    tables: number;
    with_originals: number;
    documents_by_type: Record<string, number>;
    chunks_by_kind: Record<string, number>;
  };
  embedding: { backend: string; model: string; dims: number; query_instruction?: string };
  vector_store: "sqlite-vec" | "shammy";
  runtime: RuntimeCheck;
  last_build_at: string | null;
  built_with: string | null;
  config_fingerprint: string | null;
  /** 1 = file and headings in each embedded chunk; 2 = also the document's title, number and revision. */
  embed_context?: number | null;
  /** The spots file (structure.spots): procedures, steps and references found in the chunks. */
  structure?: { present: boolean; current: boolean; built_at?: string; documents?: number; procedures?: number; steps?: number; links?: number; error?: string };
}

export interface RecentKb {
  path: string;
  kind: "pack" | "workspace";
  name: string;
  opened_at: string;
  exists: boolean;
}

export interface DocumentRow {
  doc_id: number;
  display_path: string;
  file_type: string;
  title: string | null;
  n_pages: number | null;
  size_bytes: number | null;
  status: "ok" | "partial" | "failed";
  warnings: string[];
  n_chunks: number;
  extractor: string | null;
  ingested_at: string;
  has_original: boolean;
  error: string | null;
}

export interface Chunk {
  chunk_id: number;
  doc_id: number;
  ordinal: number;
  kind: string;
  text: string;
  heading_path: string | null;
  symbol: string | null;
  page_start: number | null;
  page_end: number | null;
  line_start: number | null;
  line_end: number | null;
  sheet: string | null;
  cell_range: string | null;
  table_name: string | null;
  ocr_conf: number | null;
  asset_path: string | null;
  n_tokens: number | null;
  file?: string;
  file_type?: string;
  embed_text?: string;
  title?: string | null;
}

export interface Ranks {
  vector: number | null;
  keyword: number | null;
  fused: number;
  rank: number;
}

export interface SearchHit {
  tag: string;
  chunk_id: number;
  chunk_ids?: number[];
  kind: string;
  score: number;
  text: string;
  ranks: Ranks | null;
  source: {
    file: string;
    file_type: string;
    page: number | null;
    page_end: number | null;
    sheet: string | null;
    range: string | null;
    lines: string | null;
    heading_path: string | null;
    symbol?: string | null;
    /** From the spots file: the document's number and revision. */
    document?: string | null;
    revision?: string | null;
  };
  /** The hit's procedure, when the knowledge base has a current spots file and the hit is in one. */
  procedure?: {
    id: number;
    kind: "numbered" | "list" | "flowchart";
    title: string | null;
    label: string | null;
    steps: number;
    from: string | null;
    to: string | null;
    whole: boolean;
    /** What came back: all of it, steps from-to, sub-steps from-to inside `step`, or part of `step`. */
    level: "procedure" | "steps" | "substeps" | "chunks";
    step?: string | null;
  };
  /** The first result from a long procedure: its steps around this passage. */
  outline?: { steps: { label: string | null; title: string; shown: boolean }[]; before: boolean; after: boolean };
  /** References in the result's text: other steps ("see Section 8.4") and other documents. */
  related?: { kind: "section" | "document"; ref: string; title?: string | null; file?: string | null }[];
}

export interface SearchResponse {
  query: string;
  results: SearchHit[];
  tables: { table_name: string; chunk_id: number; score: number; summary: string; card: string; ranks: Ranks | null }[];
  timings_ms: Record<string, number>;
  candidates: (Chunk & { ranks: Ranks | null })[];
  markdown: string;
  /** Present when reranking was asked for: whether it was applied, and why not if it wasn't. */
  rerank?: { applied: boolean; error?: string; model?: string };
  /** "native" = the shared engine the retrieval server also runs; "reference" = the Python fallback. */
  engine?: string;
}

export interface TableCatalogRow {
  table_name: string;
  source_path: string;
  sheet: string | null;
  cell_range: string | null;
  page: number | null;
  caption: string | null;
  n_rows: number;
}

export interface TableDescription {
  table_name: string;
  create_table: string;
  source_path: string;
  sheet: string | null;
  cell_range: string | null;
  page: number | null;
  caption: string | null;
  n_rows: number;
  columns: { name: string; header: string; type: string; min?: unknown; max?: unknown; examples?: string[] | null; nulls?: number }[];
  sample_rows: Record<string, unknown>[];
}

export interface SqlResult {
  sql: string;
  columns: string[];
  rows: unknown[][];
  row_count: number;
  truncated: boolean;
  elapsed_ms: number;
}

export interface Job {
  id: string;
  kind: "build" | "export" | "eval" | "verify" | "merge" | "runtime_setup";
  title: string;
  kb: string | null;
  state: "running" | "done" | "failed" | "cancelled";
  started_at: string;
  finished_at: string | null;
  progress: Record<string, any>;
  error: (EngineError & { trace?: string }) | null;
  log_size: number;
  result?: any;
  log?: { t: string; [k: string]: any }[];
}

export interface EvalChunk {
  rank: number;
  match: boolean;
  chunk_id: number;
  file: string;
  where: string;
  kind: string;
  heading: string | null;
  text: string;
}

export interface EvalQuestion {
  id: string;
  type: string;
  question: string;
  first_hit_rank: number | null;
  "hit@5": boolean;
  "hit@20": boolean;
  "rr@10": number;
  embed_ms: number;
  search_ms: number;
  expect?: Record<string, unknown>[];
  retrieved?: EvalChunk[];
}

export interface EvalReport {
  overall: { n: number; "recall@5": number; "recall@20": number; "mrr@10": number };
  by_type: Record<string, { n: number; "recall@5": number; "recall@20": number; "mrr@10": number }>;
  latency_ms: Record<string, number>;
  questions: EvalQuestion[];
  unanswerable: { id: string; question: string; retrieved?: EvalChunk[] }[];
  golden?: string;
  kb?: { name: string; path: string };
}
