//! Byte-level layout: header, section directory (footer) and trailer. See spec/FORMAT.md.

use crate::error::{Error, Result};
use xxhash_rust::xxh3::xxh3_64;

pub const MAGIC: [u8; 8] = *b"SHAMMY\0\x01";
pub const FOOTER_MAGIC: [u8; 8] = *b"SHFOOT\0\x01";
pub const FORMAT_MAJOR: u16 = 1;
pub const FORMAT_MINOR: u16 = 0;
pub const HEADER_LEN: usize = 64;
pub const TRAILER_LEN: usize = 32;
pub const ALIGN: u64 = 4096;
pub const DEFAULT_ROTATION_SEED: u64 = 0x5348_414D_4D59_3031; // "SHAMMY01"
pub const FOOTER_VERSION: u32 = 1;
pub const ENTRY_LEN: usize = 48;

pub const FLAG_UNIT: u32 = 1;
pub const FLAG_LOSSLESS: u32 = 2;

/// Section types. Readers skip types they don't know.
pub mod section {
    pub const IDS: u16 = 1; // u64 chunk id per row
    pub const DOC: u16 = 2; // u32 document id per row
    pub const FTYPE: u16 = 3; // u8 file-type code per row
    pub const SKETCH: u16 = 4; // sign bits of rotated dims [0, sketch_dims)
    pub const BITS: u16 = 5; // sign bits of rotated dims [sketch_dims, dims)
    pub const CORR: u16 = 6; // 2 x f32 per row: estimator scale for the sketch, for all dims
    pub const Q8: u16 = 7; // i8 per dimension
    pub const Q8CORR: u16 = 8; // 2 x f32 per row: factor, reserved (0)
    pub const F16: u16 = 9; // unit vector as f16, unrotated: the lossless tier
    pub const TOMB: u16 = 10; // deletion bitmap, 1 bit per row; the last one for a segment wins
    pub const CLUSTERS: u16 = 20; // reserved
}

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub major: u16,
    pub minor: u16,
    pub dims: u32,
    pub sketch_dims: u32,
    pub flags: u32,
    pub rotation_seed: u64,
    pub file_uuid: [u8; 16],
}

impl Header {
    pub fn to_bytes(&self) -> [u8; HEADER_LEN] {
        let mut b = [0u8; HEADER_LEN];
        b[0..8].copy_from_slice(&MAGIC);
        b[8..10].copy_from_slice(&self.major.to_le_bytes());
        b[10..12].copy_from_slice(&self.minor.to_le_bytes());
        b[12..16].copy_from_slice(&self.dims.to_le_bytes());
        b[16..20].copy_from_slice(&self.sketch_dims.to_le_bytes());
        b[20..24].copy_from_slice(&self.flags.to_le_bytes());
        b[24..32].copy_from_slice(&self.rotation_seed.to_le_bytes());
        b[32..48].copy_from_slice(&self.file_uuid);
        let sum = xxh3_64(&b[..56]);
        b[56..64].copy_from_slice(&sum.to_le_bytes());
        b
    }

    pub fn parse(b: &[u8]) -> Result<Self> {
        if b.len() < HEADER_LEN || b[0..8] != MAGIC {
            return Err(Error::Format("not a Shammy file (wrong magic bytes)".into()));
        }
        if xxh3_64(&b[..56]) != u64_at(b, 56) {
            return Err(Error::Corrupt("the file header is corrupt (checksum mismatch)".into()));
        }
        let h = Header {
            major: u16_at(b, 8),
            minor: u16_at(b, 10),
            dims: u32_at(b, 12),
            sketch_dims: u32_at(b, 16),
            flags: u32_at(b, 20),
            rotation_seed: u64_at(b, 24),
            file_uuid: b[32..48].try_into().unwrap(),
        };
        if h.major != FORMAT_MAJOR {
            return Err(Error::Format(format!(
                "format version {}.{} is not readable by this version (reads {}.x)",
                h.major, h.minor, FORMAT_MAJOR
            )));
        }
        let (d, s) = (h.dims, h.sketch_dims);
        if d < 64 || !d.is_power_of_two() || (s != d / 4 && s != d / 2) {
            return Err(Error::Format(format!("unsupported dimensions {d} with a {s}-dimension sketch")));
        }
        Ok(h)
    }

    pub fn lossless(&self) -> bool {
        self.flags & FLAG_LOSSLESS != 0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SectionEntry {
    pub segment: u32,
    pub kind: u16,
    pub codec: u16,
    pub offset: u64,
    pub length: u64,
    pub rows: u64,
    pub checksum: u64,
    pub block_checksums_offset: u64,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Footer {
    pub generation: u64,
    pub row_count: u64,
    pub live_row_count: u64,
    pub segment_count: u32,
    pub model: String,
    /// XXH3-64 of the query instruction text (0 if none).
    pub instruction_hash: u64,
    /// File-type names; a row's FTYPE byte indexes this list.
    pub file_types: Vec<String>,
    pub sections: Vec<SectionEntry>,
}

impl Footer {
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(128 + self.sections.len() * ENTRY_LEN);
        b.extend_from_slice(&FOOTER_VERSION.to_le_bytes());
        b.extend_from_slice(&self.generation.to_le_bytes());
        b.extend_from_slice(&self.row_count.to_le_bytes());
        b.extend_from_slice(&self.live_row_count.to_le_bytes());
        b.extend_from_slice(&self.segment_count.to_le_bytes());
        b.extend_from_slice(&self.instruction_hash.to_le_bytes());
        put_str(&mut b, &self.model);
        b.extend_from_slice(&(self.file_types.len() as u16).to_le_bytes());
        for t in &self.file_types {
            put_str(&mut b, t);
        }
        b.extend_from_slice(&(self.sections.len() as u32).to_le_bytes());
        for s in &self.sections {
            b.extend_from_slice(&s.segment.to_le_bytes());
            b.extend_from_slice(&s.kind.to_le_bytes());
            b.extend_from_slice(&s.codec.to_le_bytes());
            b.extend_from_slice(&s.offset.to_le_bytes());
            b.extend_from_slice(&s.length.to_le_bytes());
            b.extend_from_slice(&s.rows.to_le_bytes());
            b.extend_from_slice(&s.checksum.to_le_bytes());
            b.extend_from_slice(&s.block_checksums_offset.to_le_bytes());
        }
        b
    }

    pub fn parse(b: &[u8]) -> Result<Self> {
        let mut r = Cursor { b, at: 0 };
        let version = r.u32()?;
        if version != FOOTER_VERSION {
            return Err(Error::Format(format!("footer version {version} is not supported")));
        }
        let mut f = Footer {
            generation: r.u64()?,
            row_count: r.u64()?,
            live_row_count: r.u64()?,
            segment_count: r.u32()?,
            instruction_hash: r.u64()?,
            model: r.string()?,
            ..Default::default()
        };
        for _ in 0..r.u16()? {
            f.file_types.push(r.string()?);
        }
        let n = r.u32()? as usize;
        if n > b.len() / ENTRY_LEN {
            return Err(Error::Format("the section directory is truncated".into()));
        }
        for _ in 0..n {
            f.sections.push(SectionEntry {
                segment: r.u32()?,
                kind: r.u16()?,
                codec: r.u16()?,
                offset: r.u64()?,
                length: r.u64()?,
                rows: r.u64()?,
                checksum: r.u64()?,
                block_checksums_offset: r.u64()?,
            });
        }
        Ok(f)
    }
}

pub fn trailer_bytes(footer_offset: u64, footer: &[u8]) -> [u8; TRAILER_LEN] {
    let mut t = [0u8; TRAILER_LEN];
    t[0..8].copy_from_slice(&footer_offset.to_le_bytes());
    t[8..16].copy_from_slice(&(footer.len() as u64).to_le_bytes());
    t[16..24].copy_from_slice(&xxh3_64(footer).to_le_bytes());
    t[24..32].copy_from_slice(&FOOTER_MAGIC);
    t
}

/// Find the last commit that is whole: scan back from the end for a trailer whose footer sits
/// right before it and matches its checksum. Returns (footer, offset just past that trailer).
pub fn find_last_commit(data: &[u8]) -> Result<(Footer, u64)> {
    let mut end = data.len();
    while end >= HEADER_LEN + TRAILER_LEN {
        let t = &data[end - TRAILER_LEN..end];
        if t[24..32] == FOOTER_MAGIC {
            let (off, len) = (u64_at(t, 0), u64_at(t, 8));
            let ok = off.checked_add(len) == Some((end - TRAILER_LEN) as u64) && off >= HEADER_LEN as u64;
            if ok {
                let body = &data[off as usize..(off + len) as usize];
                if xxh3_64(body) == u64_at(t, 16) {
                    if let Ok(f) = Footer::parse(body) {
                        return Ok((f, end as u64));
                    }
                }
            }
        }
        end -= 1;
    }
    Err(Error::Format("no complete commit found: the file is empty or was cut off before its first commit".into()))
}

pub fn align_up(n: u64) -> u64 {
    n.div_ceil(ALIGN) * ALIGN
}

pub fn u16_at(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes(b[at..at + 2].try_into().unwrap())
}
pub fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
pub fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn put_str(b: &mut Vec<u8>, s: &str) {
    let bytes = &s.as_bytes()[..s.len().min(u16::MAX as usize)];
    b.extend_from_slice(&(bytes.len() as u16).to_le_bytes());
    b.extend_from_slice(bytes);
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8]> {
        let end = self.at.checked_add(n).filter(|e| *e <= self.b.len())
            .ok_or_else(|| Error::Format("the footer is truncated".into()))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String> {
        let n = self.u16()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| Error::Format("a name in the footer is not UTF-8".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_footer_round_trip() {
        let h = Header { major: 1, minor: 0, dims: 4096, sketch_dims: 1024, flags: 3, rotation_seed: 9, file_uuid: [7; 16] };
        assert_eq!(Header::parse(&h.to_bytes()).unwrap(), h);
        let mut bad = h.to_bytes();
        bad[13] ^= 1;
        assert!(matches!(Header::parse(&bad), Err(Error::Corrupt(_))));

        let f = Footer {
            generation: 3, row_count: 10, live_row_count: 9, segment_count: 1, model: "m".into(),
            instruction_hash: 5, file_types: vec!["pdf".into(), "md".into()],
            sections: vec![SectionEntry { segment: 0, kind: 1, codec: 0, offset: 4096, length: 80, rows: 10, checksum: 1, block_checksums_offset: 0 }],
        };
        assert_eq!(Footer::parse(&f.to_bytes()).unwrap(), f);
        assert!(Footer::parse(&f.to_bytes()[..20]).is_err());
    }
}
