//! Reading: the file is memory-mapped and never copied. Opening reads the header and the
//! section directory only, so it takes the same time for any file size.

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::format::{self, section, Footer, Header};
use crate::rotate::Rotation;

#[derive(Clone, Copy, Default)]
pub(crate) struct Span {
    pub at: usize,
    pub len: usize,
}

pub(crate) struct Segment {
    pub rows: usize,
    pub ids: Span,
    pub doc: Span,
    pub ftype: Span,
    pub sketch: Span,
    pub bits: Span,
    pub corr: Span,
    pub q8: Span,
    pub q8corr: Span,
    pub f16: Option<Span>,
    pub tomb: Option<Span>,
}

/// What `info` reports.
#[derive(Clone, Debug)]
pub struct Info {
    pub dims: u32,
    pub sketch_dims: u32,
    pub lossless: bool,
    pub generation: u64,
    pub rows: u64,
    pub live_rows: u64,
    pub segments: u32,
    pub model: String,
    pub instruction_hash: u64,
    pub file_types: Vec<String>,
    pub file_uuid: [u8; 16],
    pub rotation_seed: u64,
    pub file_bytes: u64,
    pub committed_bytes: u64,
}

pub struct Reader {
    pub(crate) map: Mmap,
    pub(crate) header: Header,
    pub(crate) footer: Footer,
    pub(crate) segments: Vec<Segment>,
    pub(crate) rot: Rotation,
    committed: u64,
}

impl Reader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        // Safety: the mapping is read-only, and writers only ever append past the committed end.
        let map = unsafe { Mmap::map(&file)? };
        let header = Header::parse(&map)?;
        let (footer, committed) = format::find_last_commit(&map)?;
        let segments = build_segments(&header, &footer, committed)?;
        let rot = Rotation::new(header.dims as usize, header.rotation_seed)?;
        Ok(Self { map, header, footer, segments, rot, committed })
    }

    pub fn info(&self) -> Info {
        Info {
            dims: self.header.dims,
            sketch_dims: self.header.sketch_dims,
            lossless: self.header.lossless(),
            generation: self.footer.generation,
            rows: self.footer.row_count,
            live_rows: self.footer.live_row_count,
            segments: self.footer.segment_count,
            model: self.footer.model.clone(),
            instruction_hash: self.footer.instruction_hash,
            file_types: self.footer.file_types.clone(),
            file_uuid: self.header.file_uuid,
            rotation_seed: self.header.rotation_seed,
            file_bytes: self.map.len() as u64,
            committed_bytes: self.committed,
        }
    }

    pub fn footer(&self) -> &Footer {
        &self.footer
    }

    pub(crate) fn bytes(&self, s: Span) -> &[u8] {
        &self.map[s.at..s.at + s.len]
    }

    pub(crate) fn deleted(&self, seg: &Segment, row: usize) -> bool {
        seg.tomb.is_some_and(|t| self.map[t.at + row / 8] >> (row % 8) & 1 == 1)
    }
}

fn build_segments(header: &Header, footer: &Footer, committed: u64) -> Result<Vec<Segment>> {
    let dims = header.dims as usize;
    let sketch = header.sketch_dims as usize;
    let mut out = Vec::with_capacity(footer.segment_count as usize);
    for seg in 0..footer.segment_count {
        let find = |kind: u16, width_bits: usize, required: bool| -> Result<Option<(Span, usize)>> {
            // the last entry of a kind wins (a newer tombstone bitmap replaces the older one)
            let Some(e) = footer.sections.iter().rev().find(|s| s.segment == seg && s.kind == kind) else {
                return if required {
                    Err(Error::Format(format!("segment {seg} has no section of type {kind}")))
                } else {
                    Ok(None)
                };
            };
            let expect = if kind == section::TOMB { (e.rows as usize).div_ceil(8) } else { e.rows as usize * width_bits / 8 };
            let end = e.offset.checked_add(e.length);
            if e.length as usize != expect || end.is_none() || end.unwrap() > committed || e.offset % format::ALIGN != 0 {
                return Err(Error::Format(format!("segment {seg}, section type {kind}: size or position is wrong")));
            }
            Ok(Some((Span { at: e.offset as usize, len: e.length as usize }, e.rows as usize)))
        };
        let (ids, rows) = find(section::IDS, 64, true)?.unwrap();
        let need = |kind: u16, bits: usize| -> Result<Span> {
            let (span, r) = find(kind, bits, true)?.unwrap();
            if r != rows {
                return Err(Error::Format(format!("segment {seg}: section type {kind} has {r} rows, expected {rows}")));
            }
            Ok(span)
        };
        out.push(Segment {
            rows,
            ids,
            doc: need(section::DOC, 32)?,
            ftype: need(section::FTYPE, 8)?,
            sketch: need(section::SKETCH, sketch)?,
            bits: need(section::BITS, dims - sketch)?,
            corr: need(section::CORR, 64)?,
            q8: need(section::Q8, dims * 8)?,
            q8corr: need(section::Q8CORR, 64)?,
            f16: if header.lossless() { Some(need(section::F16, dims * 16)?) } else { None },
            tomb: find(section::TOMB, 1, false)?.map(|(s, _)| s),
        });
    }
    let total: u64 = out.iter().map(|s| s.rows as u64).sum();
    if total != footer.row_count {
        return Err(Error::Format(format!("the footer says {} rows, the segments hold {total}", footer.row_count)));
    }
    Ok(out)
}
