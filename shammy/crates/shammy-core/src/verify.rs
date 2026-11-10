//! Check a file end to end: header, last commit, every section's checksum and size.

use std::path::Path;

use xxhash_rust::xxh3::xxh3_64;

use crate::error::Result;
use crate::format::{self, Header};
use crate::reader::Reader;

#[derive(Clone, Debug, Default)]
pub struct Report {
    pub ok: bool,
    pub generation: u64,
    pub rows: u64,
    pub live_rows: u64,
    pub sections_checked: usize,
    pub bytes_checked: u64,
    /// Bytes after the last whole commit (an interrupted write); harmless, dropped on the next write.
    pub uncommitted_bytes: u64,
    pub problems: Vec<String>,
}

pub fn verify(path: impl AsRef<Path>) -> Result<Report> {
    let mut rep = Report::default();
    let data = std::fs::read(path.as_ref())?;
    if let Err(e) = Header::parse(&data) {
        rep.problems.push(e.to_string());
        return Ok(rep);
    }
    let (footer, end) = match format::find_last_commit(&data) {
        Ok(x) => x,
        Err(e) => {
            rep.problems.push(e.to_string());
            return Ok(rep);
        }
    };
    rep.generation = footer.generation;
    rep.rows = footer.row_count;
    rep.live_rows = footer.live_row_count;
    rep.uncommitted_bytes = data.len() as u64 - end;
    for s in &footer.sections {
        let stop = s.offset.checked_add(s.length);
        if stop.is_none() || stop.unwrap() > end {
            rep.problems.push(format!("segment {}, section type {}: lies outside the committed file", s.segment, s.kind));
            continue;
        }
        let bytes = &data[s.offset as usize..stop.unwrap() as usize];
        if xxh3_64(bytes) != s.checksum {
            rep.problems.push(format!("segment {}, section type {} ({} bytes at offset {}): checksum mismatch",
                                      s.segment, s.kind, s.length, s.offset));
        }
        rep.sections_checked += 1;
        rep.bytes_checked += s.length;
    }
    drop(data);
    match Reader::open(path.as_ref()) {
        Ok(r) => {
            let mut live = 0u64;
            for seg in &r.segments {
                live += (0..seg.rows).filter(|row| !r.deleted(seg, *row)).count() as u64;
            }
            if live != footer.live_row_count {
                rep.problems.push(format!("the footer says {} live rows, the deletion bitmaps leave {live}", footer.live_row_count));
            }
        }
        Err(e) => rep.problems.push(e.to_string()),
    }
    rep.ok = rep.problems.is_empty();
    Ok(rep)
}
