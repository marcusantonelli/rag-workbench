//! The retrieval path over a RAG Workbench workspace or unpacked pack.
//!
//! One implementation for every caller (the Workbench through Python, the retrieval server through
//! C#): vector search (Shammy or sqlite-vec, whichever the knowledge base uses) and FTS5 keyword
//! search, fused with Reciprocal Rank Fusion, duplicates collapsed, optionally reranked by a model
//! behind an HTTP endpoint, then prose hits expanded with their neighbours and cited.
//!
//! The caller embeds the query and passes the vector; everything after that happens here.
//! `src/rag/core/retrieve/search.py` (in the Workbench) keeps the original Python version as the reference, and a
//! test holds the two to the same output.

mod query;
mod rerank;

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once};
use std::time::Instant;

use rusqlite::{params_from_iter, Connection, OpenFlags, Row};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

pub use query::{fts_query, query_identifiers, split_identifier};
pub use rerank::RerankOptions;

#[derive(Debug)]
pub enum Error {
    /// The request is wrong (bad JSON, wrong vector size).
    Request(String),
    /// The knowledge base can't be opened or read.
    Store(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Request(m) | Error::Store(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

impl From<rusqlite::Error> for Error {
    fn from(e: rusqlite::Error) -> Self {
        Error::Store(format!("database error: {e}"))
    }
}

impl From<shammy_core::Error> for Error {
    fn from(e: shammy_core::Error) -> Self {
        Error::Store(format!("vectors.shammy: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

fn d_true() -> bool { true }
fn d_keep() -> usize { 6 }
fn d_k40() -> usize { 40 }
fn d_rrf() -> f64 { 60.0 }
fn d_neighbors() -> i64 { 1 }
fn d_budget() -> i64 { 6000 }

/// One search. Only `query` is required; the defaults are the Workbench's.
#[derive(Clone, Debug, Deserialize)]
pub struct Request {
    pub query: String,
    /// Results returned.
    #[serde(default = "d_keep")]
    pub k: usize,
    /// Only chunks from files of this type (for example "pdf").
    #[serde(default)]
    pub file_type: Option<String>,
    /// Join a prose hit with its neighbours from the same section.
    #[serde(default = "d_true")]
    pub expand: bool,
    #[serde(default = "d_k40")]
    pub vec_k: usize,
    #[serde(default = "d_k40")]
    pub fts_k: usize,
    #[serde(default = "d_rrf")]
    pub rrf_k: f64,
    #[serde(default = "d_neighbors")]
    pub neighbors: i64,
    #[serde(default = "d_budget")]
    pub max_result_tokens: i64,
    /// Threads for the vector search (Shammy only); 0 = two fewer than the machine's cores.
    #[serde(default)]
    pub threads: usize,
    /// Rerank the fused candidates with a model. Absent or `enabled: false` = no reranking.
    #[serde(default)]
    pub rerank: Option<RerankOptions>,
    /// Also return the compact Markdown form an agent reads.
    #[serde(default)]
    pub markdown: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct Info {
    pub path: String,
    pub vector_store: String,
    pub dims: usize,
    pub model: Option<String>,
    pub query_instruction: Option<String>,
    pub chunks: i64,
    pub vectors: i64,
}

enum Vectors {
    SqliteVec,
    Shammy(shammy_core::Reader),
}

/// A knowledge base opened for searching. Safe to share between threads.
pub struct Retriever {
    conn: Mutex<Connection>,
    vectors: Vectors,
    dir: PathBuf,
    dims: usize,
}

struct ChunkRow {
    chunk_id: i64,
    doc_id: i64,
    ordinal: i64,
    kind: String,
    text: String,
    text_hash: String,
    heading_path: Option<String>,
    symbol: Option<String>,
    page_start: Option<i64>,
    page_end: Option<i64>,
    line_start: Option<i64>,
    line_end: Option<i64>,
    sheet: Option<String>,
    cell_range: Option<String>,
    table_name: Option<String>,
    ocr_conf: Option<f64>,
    asset_path: Option<String>,
    n_tokens: Option<i64>,
    display_path: String,
    file_type: String,
}

const CHUNK_COLUMNS: &str = "c.chunk_id, c.doc_id, c.ordinal, c.kind, c.text, c.text_hash, c.heading_path, c.symbol, \
    c.page_start, c.page_end, c.line_start, c.line_end, c.sheet, c.cell_range, c.table_name, c.ocr_conf, c.asset_path, \
    c.n_tokens, d.display_path, d.file_type";

fn chunk_row(r: &Row<'_>) -> rusqlite::Result<ChunkRow> {
    Ok(ChunkRow {
        chunk_id: r.get(0)?, doc_id: r.get(1)?, ordinal: r.get(2)?, kind: r.get(3)?, text: r.get(4)?,
        text_hash: r.get(5)?, heading_path: r.get(6)?, symbol: r.get(7)?, page_start: r.get(8)?, page_end: r.get(9)?,
        line_start: r.get(10)?, line_end: r.get(11)?, sheet: r.get(12)?, cell_range: r.get(13)?, table_name: r.get(14)?,
        ocr_conf: r.get(15)?, asset_path: r.get(16)?, n_tokens: r.get(17)?, display_path: r.get(18)?, file_type: r.get(19)?,
    })
}

fn register_sqlite_vec() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        // makes the vec0 module available on every connection this process opens through rusqlite
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<*const (), unsafe extern "C" fn(
            *mut rusqlite::ffi::sqlite3, *mut *mut std::os::raw::c_char, *const rusqlite::ffi::sqlite3_api_routines,
        ) -> std::os::raw::c_int>(sqlite_vec::sqlite3_vec_init as *const ())));
    });
}

fn round5(x: f64) -> f64 {
    (x * 1e5).round() / 1e5
}

fn ms(since: Instant) -> f64 {
    (since.elapsed().as_secs_f64() * 1e4).round() / 10.0
}

impl Retriever {
    /// Open the folder of a workspace or an unpacked pack (the one holding rag.db).
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        register_sqlite_vec();
        let dir = dir.as_ref().to_path_buf();
        let db = dir.join("rag.db");
        if !db.is_file() {
            return Err(Error::Store(format!("{} has no rag.db; it isn't a workspace or an unpacked pack", dir.display())));
        }
        let conn = Connection::open_with_flags(&db, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
            .map_err(|e| Error::Store(format!("can't open {}: {e}", db.display())))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))?;
        let meta = |key: &str| -> Option<String> {
            conn.query_row("SELECT value FROM meta WHERE key = ?", [key], |r| r.get::<_, Option<String>>(0)).ok().flatten()
        };
        let backend = meta("vector_backend").unwrap_or_else(|| "sqlite-vec".into());
        let mut dims = meta("embed_dims").and_then(|d| d.parse::<usize>().ok()).unwrap_or(0);
        let vectors = if backend == "shammy" {
            let path = dir.join("vectors.shammy");
            if !path.is_file() {
                return Err(Error::Store(format!("{} is missing; this knowledge base keeps its vectors there", path.display())));
            }
            let reader = shammy_core::Reader::open(&path)?;
            let stored = reader.info().dims as usize;
            if dims == 0 {
                dims = stored;
            } else if dims > stored {
                return Err(Error::Store(format!("rag.db says vectors have {dims} values but vectors.shammy stores {stored}")));
            }
            Vectors::Shammy(reader)
        } else if backend == "sqlite-vec" {
            Vectors::SqliteVec
        } else {
            return Err(Error::Store(format!("unknown vector store {backend:?}; this version reads sqlite-vec and shammy")));
        };
        Ok(Self { conn: Mutex::new(conn), vectors, dir, dims })
    }

    pub fn info(&self) -> Result<Info> {
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let meta = |key: &str| -> Option<String> {
            conn.query_row("SELECT value FROM meta WHERE key = ?", [key], |r| r.get::<_, Option<String>>(0)).ok().flatten()
        };
        let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        let (store, vectors) = match &self.vectors {
            Vectors::Shammy(r) => ("shammy", r.info().live_rows as i64),
            Vectors::SqliteVec => ("sqlite-vec", conn.query_row("SELECT COUNT(*) FROM chunk_vec", [], |r| r.get(0))?),
        };
        Ok(Info { path: self.dir.display().to_string(), vector_store: store.into(), dims: self.dims,
                  model: meta("embed_model"), query_instruction: meta("query_instruction"), chunks, vectors })
    }

    fn vector_ids(&self, conn: &Connection, vector: &[f32], k: usize, file_type: Option<&str>, threads: usize) -> Result<Vec<i64>> {
        match &self.vectors {
            Vectors::Shammy(reader) => {
                if reader.info().live_rows == 0 {
                    return Ok(Vec::new());
                }
                let mut q = vec![0f32; reader.info().dims as usize];
                q[..vector.len()].copy_from_slice(vector);
                let filter = shammy_core::Filter { file_types: file_type.map(|t| vec![t.to_string()]), doc_ids: None };
                let params = shammy_core::Params { threads, ..Default::default() };
                Ok(reader.search(&q, k, &filter, &params)?.into_iter().map(|h| h.id as i64).collect())
            }
            Vectors::SqliteVec => {
                let blob: Vec<u8> = vector.iter().flat_map(|x| x.to_le_bytes()).collect();
                let mut sql = String::from("SELECT chunk_id, distance FROM chunk_vec WHERE embedding MATCH ?1 AND k = ?2");
                if file_type.is_some() {
                    sql.push_str(" AND file_type = ?3");
                }
                let mut stmt = conn.prepare_cached(&sql)?;
                let rows = match file_type {
                    Some(t) => stmt.query_map(rusqlite::params![blob, k as i64, t], |r| r.get::<_, i64>(0))?.collect::<rusqlite::Result<Vec<_>>>(),
                    None => stmt.query_map(rusqlite::params![blob, k as i64], |r| r.get::<_, i64>(0))?.collect::<rusqlite::Result<Vec<_>>>(),
                };
                Ok(rows?)
            }
        }
    }

    fn keyword_ids(&self, conn: &Connection, query: &str, k: usize, file_type: Option<&str>) -> Result<Vec<i64>> {
        let expr = fts_query(query);
        if expr.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = conn.prepare_cached("SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ?1 ORDER BY rank LIMIT ?2")?;
        let mut ids = stmt.query_map(rusqlite::params![expr, (k * 2) as i64], |r| r.get::<_, i64>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if let (Some(t), false) = (file_type, ids.is_empty()) {
            let marks = vec!["?"; ids.len()].join(",");
            let sql = format!("SELECT c.chunk_id FROM chunks c JOIN documents d ON d.doc_id = c.doc_id \
                               WHERE c.chunk_id IN ({marks}) AND d.file_type = ?");
            let mut args: Vec<rusqlite::types::Value> = ids.iter().map(|i| (*i).into()).collect();
            args.push(t.to_string().into());
            let keep: HashSet<i64> = conn.prepare(&sql)?.query_map(params_from_iter(args), |r| r.get::<_, i64>(0))?
                .collect::<rusqlite::Result<_>>()?;
            ids.retain(|i| keep.contains(i));
        }
        ids.truncate(k);
        Ok(ids)
    }

    fn load(&self, conn: &Connection, ids: &[i64]) -> Result<HashMap<i64, ChunkRow>> {
        if ids.is_empty() {
            return Ok(HashMap::new());
        }
        let marks = vec!["?"; ids.len()].join(",");
        let sql = format!("SELECT {CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.doc_id = c.doc_id WHERE c.chunk_id IN ({marks})");
        let rows = conn.prepare(&sql)?.query_map(params_from_iter(ids.iter()), chunk_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows.into_iter().map(|r| (r.chunk_id, r)).collect())
    }

    /// The chunk plus up to `n` prose chunks on each side from the same section.
    fn neighbors(&self, conn: &Connection, hit: &ChunkRow, n: i64) -> Result<Vec<ChunkRow>> {
        let sql = format!("SELECT {CHUNK_COLUMNS} FROM chunks c JOIN documents d ON d.doc_id = c.doc_id \
                           WHERE c.doc_id = ?1 AND c.ordinal BETWEEN ?2 AND ?3 AND c.kind = 'prose' \
                           AND c.heading_path IS ?4 ORDER BY c.ordinal");
        let mut rows = conn.prepare_cached(&sql)?
            .query_map(rusqlite::params![hit.doc_id, hit.ordinal - n, hit.ordinal + n, hit.heading_path], chunk_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        // keep only the unbroken run around the hit
        let Some(i) = rows.iter().position(|r| r.ordinal == hit.ordinal) else { return Ok(Vec::new()) };
        let (mut lo, mut hi) = (i, i);
        while lo > 0 && rows[lo - 1].ordinal == rows[lo].ordinal - 1 {
            lo -= 1;
        }
        while hi < rows.len() - 1 && rows[hi + 1].ordinal == rows[hi].ordinal + 1 {
            hi += 1;
        }
        rows.truncate(hi + 1);
        Ok(rows.split_off(lo))
    }

    /// Search with a JSON request (see `Request`) and get the JSON response.
    pub fn search_json(&self, request: &str, vector: &[f32]) -> Result<String> {
        let req: Request = serde_json::from_str(request).map_err(|e| Error::Request(format!("bad search request: {e}")))?;
        Ok(self.search(&req, vector)?.to_string())
    }

    pub fn search(&self, req: &Request, vector: &[f32]) -> Result<Value> {
        if vector.is_empty() || (self.dims > 0 && vector.len() != self.dims) {
            return Err(Error::Request(format!("the query vector has {} values; this knowledge base uses {}", vector.len(), self.dims)));
        }
        let conn = self.conn.lock().unwrap_or_else(|e| e.into_inner());
        let file_type = req.file_type.as_deref().filter(|t| !t.is_empty());
        let mut timings = Map::new();

        let t = Instant::now();
        let vec_ids = self.vector_ids(&conn, vector, req.vec_k, file_type, req.threads)?;
        let kw_ids = self.keyword_ids(&conn, &req.query, req.fts_k, file_type)?;
        // Reciprocal Rank Fusion; ties keep first-seen order (vector hits first)
        let mut scores: HashMap<i64, f64> = HashMap::new();
        let mut order: Vec<i64> = Vec::new();
        for ranked in [&vec_ids, &kw_ids] {
            for (i, cid) in ranked.iter().enumerate() {
                let add = 1.0 / (req.rrf_k + (i + 1) as f64);
                match scores.get_mut(cid) {
                    Some(s) => *s += add,
                    None => {
                        scores.insert(*cid, add);
                        order.push(*cid);
                    }
                }
            }
        }
        order.sort_by(|a, b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));
        let rows = self.load(&conn, &order)?;
        // collapse identical text (boilerplate repeated across files)
        let mut seen: HashSet<&str> = HashSet::new();
        let ordered: Vec<i64> = order.iter().copied()
            .filter(|cid| rows.get(cid).is_some_and(|r| seen.insert(r.text_hash.as_str()))).collect();
        timings.insert("search".into(), json!(ms(t)));

        let rr = req.rerank.as_ref().filter(|r| r.enabled);
        let pool = rr.map_or(20, |r| r.candidates.max(1));
        let mut candidates: Vec<i64> = ordered.iter().copied().take(pool).collect();
        let mut rerank_scores: HashMap<i64, f64> = HashMap::new();
        let mut rerank_report = Value::Null;
        if let Some(opts) = rr {
            let t = Instant::now();
            let docs: Vec<&str> = candidates.iter().map(|c| rows[c].text.as_str()).collect();
            match rerank::score(opts, &req.query, &docs) {
                Ok(found) => {
                    let mut paired: Vec<(f64, i64)> = found.iter().copied().zip(candidates.iter().copied()).collect();
                    paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                    candidates = paired.iter().map(|p| p.1).collect();
                    rerank_scores = paired.into_iter().map(|(s, c)| (c, s)).collect();
                    rerank_report = json!({"applied": true, "candidates": candidates.len(), "model": opts.model});
                }
                // a reranker that is down must not take search down with it: keep the fused order and say so
                Err(why) => rerank_report = json!({"applied": false, "error": why}),
            }
            timings.insert("rerank".into(), json!(ms(t)));
        }

        let vec_rank: HashMap<i64, usize> = vec_ids.iter().enumerate().map(|(i, c)| (*c, i + 1)).collect();
        let kw_rank: HashMap<i64, usize> = kw_ids.iter().enumerate().map(|(i, c)| (*c, i + 1)).collect();
        let mut ranks = Map::new();
        for (i, cid) in ordered.iter().enumerate() {
            ranks.insert(cid.to_string(), json!({"vector": vec_rank.get(cid), "keyword": kw_rank.get(cid),
                                                 "fused": round5(scores[cid]), "rank": i + 1}));
        }

        let mut results: Vec<Value> = Vec::new();
        let mut tables: Vec<Value> = Vec::new();
        let mut included: HashSet<i64> = HashSet::new();
        let (mut used, mut tag) = (0i64, 0usize);
        for cid in &candidates {
            if results.len() >= req.k {
                break;
            }
            if included.contains(cid) {
                continue;
            }
            let row = &rows[cid];
            if row.kind == "table_card" {
                let mut t = json!({"table_name": row.table_name, "chunk_id": cid, "score": round5(scores[cid]),
                                   "summary": row.text.split('\n').next().unwrap_or(""), "card": row.text});
                if let Some(s) = rerank_scores.get(cid) {
                    t["rerank_score"] = json!(s);
                }
                tables.push(t);
                continue;
            }
            let mut text = row.text.clone();
            let mut span = vec![*cid];
            if req.expand && row.kind == "prose" && req.neighbors > 0 {
                let around: Vec<ChunkRow> = self.neighbors(&conn, row, req.neighbors)?.into_iter()
                    .filter(|r| !included.contains(&r.chunk_id)).collect();
                if around.len() > 1 {
                    text = around.iter().map(|r| r.text.as_str()).collect::<Vec<_>>().join("\n\n");
                    span = around.iter().map(|r| r.chunk_id).collect();
                }
            }
            let tokens = row.n_tokens.unwrap_or(0) * if text != row.text { 3 } else { 1 };
            if used > 0 && used + tokens > req.max_result_tokens {
                break;
            }
            used += tokens;
            included.extend(span.iter().copied());
            tag += 1;
            let lines = match (row.line_start.filter(|l| *l != 0), row.line_end.filter(|l| *l != 0)) {
                (Some(a), Some(b)) => Some(format!("{a}-{b}")),
                (Some(a), None) => Some(a.to_string()),
                _ => None,
            };
            let mut item = json!({
                "tag": format!("S{tag}"), "chunk_id": cid, "kind": row.kind, "score": round5(scores[cid]), "text": text,
                "source": {
                    "file": row.display_path, "file_type": row.file_type, "page": row.page_start, "page_end": row.page_end,
                    "sheet": row.sheet, "range": row.cell_range, "lines": lines, "heading_path": row.heading_path,
                    "symbol": row.symbol, "table_name": row.table_name, "asset_path": row.asset_path, "ocr_conf": row.ocr_conf,
                },
            });
            if span.len() > 1 {
                item["chunk_ids"] = json!(span);
            }
            if let Some(s) = rerank_scores.get(cid) {
                item["rerank_score"] = json!(s);
            }
            results.push(item);
        }

        let mut out = json!({"query": req.query, "results": results, "tables": tables, "timings_ms": timings,
                             "candidates": ordered, "ranks": ranks});
        if !rerank_report.is_null() {
            out["rerank"] = rerank_report;
        }
        if req.markdown {
            out["markdown"] = json!(markdown(&out));
        }
        Ok(out)
    }
}

/// The compact form for an agent: a source line per result, then the passage.
pub fn markdown(response: &Value) -> String {
    let mut out: Vec<String> = Vec::new();
    let text = |v: &Value| v.as_str().filter(|s| !s.is_empty()).map(str::to_string);
    for r in response["results"].as_array().into_iter().flatten() {
        let s = &r["source"];
        let mut loc: Vec<String> = Vec::new();
        if let Some(page) = s["page"].as_i64().filter(|p| *p != 0) {
            let end = s["page_end"].as_i64().filter(|e| *e != page);
            loc.push(format!("p.{page}") + &end.map(|e| format!("-{e}")).unwrap_or_default());
        }
        if let Some(sheet) = text(&s["sheet"]) {
            loc.push(format!("sheet {sheet}"));
        }
        if let Some(range) = text(&s["range"]) {
            loc.push(range);
        }
        if let Some(lines) = text(&s["lines"]) {
            loc.push(format!("lines {lines}"));
        }
        for key in ["heading_path", "symbol"] {
            if let Some(v) = text(&s[key]) {
                loc.push(v);
            }
        }
        let place = if loc.is_empty() { String::new() } else { format!(" · {}", loc.join(" · ")) };
        out.push(format!("[{}] {}{}  (chunk {})", r["tag"].as_str().unwrap_or(""), s["file"].as_str().unwrap_or(""), place, r["chunk_id"]));
        out.push(r["text"].as_str().unwrap_or("").to_string());
        out.push(String::new());
    }
    let tables = response["tables"].as_array().filter(|t| !t.is_empty());
    if let Some(tables) = tables {
        out.push("Tables that may answer this (use describe_table / query_table):".into());
        for t in tables {
            out.push(format!("- {}: {}", t["table_name"].as_str().unwrap_or(""), t["summary"].as_str().unwrap_or("")));
        }
    }
    out.join("\n").trim().to_string()
}
