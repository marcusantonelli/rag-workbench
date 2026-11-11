//! Reading a knowledge base's spots file (`structure.spots`): which chunks form a procedure and its
//! steps, the references between them, and each document's number and revision. The Workbench
//! writes it (`rag spots`, and after every build); here it is only read.
//!
//! A spots file is used only while it matches its rag.db (the chunk fingerprint); a missing, stale
//! or unreadable one is ignored, and search then behaves exactly as without spots files.

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{Connection, OpenFlags, OptionalExtension};

pub(crate) const SPOTS_NAME: &str = "structure.spots";
const FORMAT: &str = "spots/1";

#[derive(Clone)]
pub(crate) struct Procedure {
    pub kind: String,
    pub label: Option<String>,
    pub title: Option<String>,
    pub steps: i64,
    pub tokens: i64,
}

#[derive(Clone)]
pub(crate) struct Step {
    pub step: i64,
    pub label: Option<String>,
    pub title: Option<String>,
    pub tokens: i64,
}

#[derive(Clone)]
pub(crate) struct Substep {
    pub sub: i64,
    pub label: Option<String>,
    pub title: Option<String>,
    pub tokens: i64,
}

pub(crate) struct Link {
    pub kind: String,
    pub reference: String,
    pub proc_id: Option<i64>,
    pub step: Option<i64>,
    pub doc_id: Option<i64>,
}

pub(crate) struct Spots {
    conn: Mutex<Connection>,
}

/// The chunks a rag.db holds, as the Workbench fingerprints them.
pub(crate) fn fingerprint(rag: &Connection) -> rusqlite::Result<String> {
    rag.query_row("SELECT COUNT(*), COALESCE(MAX(chunk_id), 0), COALESCE(SUM(chunk_id), 0) FROM chunks", [], |r| {
        Ok(format!("{}:{}:{}", r.get::<_, i64>(0)?, r.get::<_, i64>(1)?, r.get::<_, i64>(2)?))
    })
}

/// How document numbers compare: case, spaces, dashes and underscores don't matter.
pub(crate) fn norm_number(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace() && *c != '-' && *c != '_').collect::<String>().to_uppercase()
}

impl Spots {
    /// The spots file beside rag.db, when there is one and it matches.
    pub(crate) fn open(dir: &Path, rag: &Connection) -> Option<Spots> {
        let path = dir.join(SPOTS_NAME);
        if !path.is_file() {
            return None;
        }
        let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX).ok()?;
        let meta = |key: &str| -> Option<String> {
            conn.query_row("SELECT value FROM spots_meta WHERE key = ?", [key], |r| r.get::<_, Option<String>>(0)).ok().flatten()
        };
        if meta("format").as_deref() != Some(FORMAT) || meta("chunks") != fingerprint(rag).ok() {
            return None;
        }
        Some(Spots { conn: Mutex::new(conn) })
    }

    fn conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// (proc_id, step, sub) of a chunk inside a procedure.
    pub(crate) fn member(&self, chunk_id: i64) -> rusqlite::Result<Option<(i64, i64, i64)>> {
        self.conn().query_row("SELECT proc_id, step, sub FROM members WHERE chunk_id = ?", [chunk_id],
                              |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .optional()
    }

    pub(crate) fn substeps(&self, proc_id: i64, step: i64) -> rusqlite::Result<Vec<Substep>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached("SELECT sub, label, title, tokens FROM substeps WHERE proc_id = ? AND step = ? ORDER BY sub")?;
        let rows = stmt.query_map([proc_id, step], |r| Ok(Substep { sub: r.get(0)?, label: r.get(1)?, title: r.get(2)?, tokens: r.get(3)? }))?;
        rows.collect()
    }

    pub(crate) fn procedure(&self, proc_id: i64) -> rusqlite::Result<Option<Procedure>> {
        self.conn().query_row("SELECT kind, label, title, steps, tokens FROM procedures WHERE proc_id = ?", [proc_id], |r| {
            Ok(Procedure { kind: r.get(0)?, label: r.get(1)?, title: r.get(2)?, steps: r.get(3)?, tokens: r.get(4)? })
        }).optional()
    }

    pub(crate) fn steps(&self, proc_id: i64) -> rusqlite::Result<Vec<Step>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached("SELECT step, label, title, tokens FROM steps WHERE proc_id = ? ORDER BY step")?;
        let rows = stmt.query_map([proc_id], |r| Ok(Step { step: r.get(0)?, label: r.get(1)?, title: r.get(2)?, tokens: r.get(3)? }))?;
        rows.collect()
    }

    /// (chunk_id, step, sub) of steps lo..=hi, in reading order.
    pub(crate) fn members(&self, proc_id: i64, lo: i64, hi: i64) -> rusqlite::Result<Vec<(i64, i64, i64)>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached(
            "SELECT chunk_id, step, sub FROM members WHERE proc_id = ?1 AND step BETWEEN ?2 AND ?3 ORDER BY step, ordinal")?;
        let rows = stmt.query_map(rusqlite::params![proc_id, lo, hi], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect()
    }

    pub(crate) fn links(&self, chunk_id: i64) -> rusqlite::Result<Vec<Link>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached("SELECT kind, ref, proc_id, step, doc_id FROM links WHERE chunk_id = ? ORDER BY rowid")?;
        let rows = stmt.query_map([chunk_id], |r| {
            Ok(Link { kind: r.get(0)?, reference: r.get(1)?, proc_id: r.get(2)?, step: r.get(3)?, doc_id: r.get(4)? })
        })?;
        rows.collect()
    }

    pub(crate) fn step_info(&self, proc_id: i64, step: i64) -> rusqlite::Result<Option<(Option<String>, Option<String>)>> {
        self.conn().query_row("SELECT label, title FROM steps WHERE proc_id = ? AND step = ?", [proc_id, step], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()
    }

    pub(crate) fn document(&self, doc_id: i64) -> rusqlite::Result<(Option<String>, Option<String>)> {
        Ok(self.conn().query_row("SELECT number, revision FROM documents WHERE doc_id = ?", [doc_id], |r| Ok((r.get(0)?, r.get(1)?)))
            .optional()?.unwrap_or((None, None)))
    }

    pub(crate) fn numbers(&self) -> rusqlite::Result<Vec<(i64, String)>> {
        let conn = self.conn();
        let mut stmt = conn.prepare_cached("SELECT doc_id, number FROM documents WHERE number IS NOT NULL ORDER BY doc_id")?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
        rows.collect()
    }

    pub(crate) fn count(&self, table: &str) -> i64 {
        self.conn().query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0)).unwrap_or(0)
    }
}
