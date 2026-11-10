//! Compaction: copy the live rows of a file into a new file, dropping deleted rows and old
//! tombstone bitmaps. Rows are copied as stored, so searches on the new file score them
//! exactly as before.

use std::path::Path;

use crate::error::{Error, Result};
use crate::reader::Reader;
use crate::writer::Writer;

#[derive(Clone, Copy, Debug, Default)]
pub struct Compacted {
    pub rows_before: u64,
    pub rows_after: u64,
    pub bytes_before: u64,
    pub bytes_after: u64,
}

/// Write the live rows of `src` to `dst` (which must not exist yet). `src` is not changed.
pub fn compact(src: impl AsRef<Path>, dst: impl AsRef<Path>) -> Result<Compacted> {
    compact_with(src, dst, true)
}

/// As `compact`; with `keep_lossless` false the float16 copies are left out, giving a slim file
/// about a third of the size whose searches end at the 8-bit codes.
pub fn compact_with(src: impl AsRef<Path>, dst: impl AsRef<Path>, keep_lossless: bool) -> Result<Compacted> {
    let r = Reader::open(src.as_ref())?;
    if dst.as_ref().exists() {
        return Err(Error::Invalid(format!("{} already exists", dst.as_ref().display())));
    }
    let mut header = r.header.clone();
    if !keep_lossless {
        header.flags &= !crate::format::FLAG_LOSSLESS;
    }
    let mut w = Writer::create_like(dst.as_ref(), &header, &r.footer)?;
    let dims = r.header.dims as usize;
    let sk = r.header.sketch_dims as usize / 8;
    let rest = dims / 8 - sk;
    let mut kept = 0u64;
    for seg in &r.segments {
        let part = |s: crate::reader::Span, row: usize, width: usize| &r.bytes(s)[row * width..(row + 1) * width];
        for row in 0..seg.rows {
            if r.deleted(seg, row) {
                continue;
            }
            w.add_encoded(part(seg.ids, row, 8), part(seg.doc, row, 4), r.bytes(seg.ftype)[row], part(seg.sketch, row, sk),
                          part(seg.bits, row, rest), part(seg.corr, row, 8), part(seg.q8, row, dims), part(seg.q8corr, row, 8),
                          seg.f16.filter(|_| keep_lossless).map(|f| part(f, row, dims * 2)))?;
            kept += 1;
        }
    }
    w.commit()?;
    let info = r.info();
    Ok(Compacted { rows_before: info.rows, rows_after: kept, bytes_before: info.file_bytes,
                   bytes_after: std::fs::metadata(dst.as_ref())?.len() })
}
