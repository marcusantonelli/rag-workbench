//! Writing: every change appends whole sections, then one footer and trailer. Nothing already
//! committed is modified, so a crash mid-write leaves the previous commit readable.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use half::f16;
use xxhash_rust::xxh3::xxh3_64;

use crate::encode::{encode, normalize};
use crate::error::{Error, Result};
use crate::format::{self, section, Footer, Header, SectionEntry};
use crate::rotate::Rotation;

/// What a new file is created with.
#[derive(Clone, Debug)]
pub struct Schema {
    pub dims: u32,
    /// dims / 4 (default) or dims / 2.
    pub sketch_dims: u32,
    /// Keep each vector as float16 so results can be rescored exactly and codes rebuilt.
    pub lossless: bool,
    pub model: String,
    pub query_instruction: String,
    pub rotation_seed: u64,
}

impl Schema {
    pub fn new(dims: u32) -> Self {
        Self { dims, sketch_dims: dims / 4, lossless: true, model: String::new(),
               query_instruction: String::new(), rotation_seed: format::DEFAULT_ROTATION_SEED }
    }
}

#[derive(Default)]
struct Pending {
    ids: Vec<u8>,
    docs: Vec<u8>,
    ftypes: Vec<u8>,
    sketch: Vec<u8>,
    bits: Vec<u8>,
    corr: Vec<u8>,
    q8: Vec<u8>,
    q8corr: Vec<u8>,
    f16: Vec<u8>,
    rows: u64,
}

pub struct Writer {
    path: PathBuf,
    file: File,
    header: Header,
    footer: Footer,
    rot: Rotation,
    pending: Pending,
    scratch: Vec<f32>,
    unit: Vec<f32>,
    /// Where the next section goes (everything before it is committed or written this session).
    end: u64,
    dirty: bool,
    /// Rows buffered in memory before they are written out as a segment.
    pub segment_rows: u64,
}

impl Writer {
    /// Create a new file. Fails if one is already there.
    pub fn create(path: impl AsRef<Path>, schema: &Schema) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let rot = Rotation::new(schema.dims as usize, schema.rotation_seed)?;
        if schema.sketch_dims != schema.dims / 4 && schema.sketch_dims != schema.dims / 2 {
            return Err(Error::Invalid("the sketch must cover a quarter or half of the dimensions".into()));
        }
        let header = Header {
            major: format::FORMAT_MAJOR,
            minor: format::FORMAT_MINOR,
            dims: schema.dims,
            sketch_dims: schema.sketch_dims,
            flags: format::FLAG_UNIT | if schema.lossless { format::FLAG_LOSSLESS } else { 0 },
            rotation_seed: schema.rotation_seed,
            file_uuid: new_uuid(&path),
        };
        let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path)
            .map_err(|e| Error::Invalid(format!("can't create {}: {e}", path.display())))?;
        file.write_all(&header.to_bytes())?;
        let footer = Footer {
            model: schema.model.clone(),
            instruction_hash: if schema.query_instruction.is_empty() { 0 } else { xxh3_64(schema.query_instruction.as_bytes()) },
            ..Default::default()
        };
        let mut w = Self::assemble(path, file, header, footer, rot, format::HEADER_LEN as u64);
        w.dirty = true; // so an empty file still gets its first commit
        Ok(w)
    }

    /// Open an existing file to append to it. Anything after its last whole commit is discarded.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;
        let header = Header::parse(&data)?;
        let (footer, end) = format::find_last_commit(&data)?;
        let len = data.len() as u64;
        drop(data);
        if end != len {
            file.set_len(end)?; // drop an interrupted write
        }
        let rot = Rotation::new(header.dims as usize, header.rotation_seed)?;
        Ok(Self::assemble(path, file, header, footer, rot, end))
    }

    fn assemble(path: PathBuf, file: File, header: Header, footer: Footer, rot: Rotation, end: u64) -> Self {
        let dims = header.dims as u64;
        // about 256 MB of buffered rows per segment
        let per_row = 21 + dims / 8 + dims + if header.lossless() { dims * 2 } else { 0 } + 16;
        Self { path, file, header, footer, rot, pending: Pending::default(), scratch: Vec::new(),
               unit: Vec::new(), end, dirty: false, segment_rows: ((256u64 << 20) / per_row).max(1024) }
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// A new, empty file with the same layout (dimensions, rotation, model, file types) as `like`.
    pub(crate) fn create_like(path: impl AsRef<Path>, like: &Header, footer: &Footer) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let rot = Rotation::new(like.dims as usize, like.rotation_seed)?;
        let header = Header { file_uuid: new_uuid(&path), ..like.clone() };
        let mut file = OpenOptions::new().read(true).write(true).create_new(true).open(&path)
            .map_err(|e| Error::Invalid(format!("can't create {}: {e}", path.display())))?;
        file.write_all(&header.to_bytes())?;
        let footer = Footer { model: footer.model.clone(), instruction_hash: footer.instruction_hash,
                              file_types: footer.file_types.clone(), ..Default::default() };
        let mut w = Self::assemble(path, file, header, footer, rot, format::HEADER_LEN as u64);
        w.dirty = true;
        Ok(w)
    }

    /// Add a row that is already encoded (copied from another file with the same layout).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn add_encoded(&mut self, id: &[u8], doc: &[u8], ftype: u8, sketch: &[u8], bits: &[u8], corr: &[u8],
                              q8: &[u8], q8corr: &[u8], f16: Option<&[u8]>) -> Result<()> {
        let p = &mut self.pending;
        p.ids.extend_from_slice(id);
        p.docs.extend_from_slice(doc);
        p.ftypes.push(ftype);
        p.sketch.extend_from_slice(sketch);
        p.bits.extend_from_slice(bits);
        p.corr.extend_from_slice(corr);
        p.q8.extend_from_slice(q8);
        p.q8corr.extend_from_slice(q8corr);
        if let Some(f) = f16 {
            p.f16.extend_from_slice(f);
        }
        p.rows += 1;
        self.dirty = true;
        if p.rows >= self.segment_rows {
            self.flush_segment()?;
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Add one row. The vector is normalised to unit length; `file_type` is any short label.
    pub fn add(&mut self, id: u64, doc_id: u32, file_type: &str, vector: &[f32]) -> Result<()> {
        let dims = self.header.dims as usize;
        if vector.len() != dims {
            return Err(Error::Invalid(format!("a vector has {} dimensions, the file stores {dims}", vector.len())));
        }
        if vector.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid(format!("the vector for id {id} contains NaN or infinity")));
        }
        let code = match self.footer.file_types.iter().position(|t| t == file_type) {
            Some(i) => i,
            None => {
                if self.footer.file_types.len() >= 255 {
                    return Err(Error::Invalid("more than 255 distinct file types".into()));
                }
                self.footer.file_types.push(file_type.to_string());
                self.footer.file_types.len() - 1
            }
        } as u8;
        self.unit.clear();
        self.unit.extend_from_slice(vector);
        normalize(&mut self.unit);
        let sketch_bytes = self.header.sketch_dims as usize / 8;
        let e = encode(&self.unit, &self.rot, self.header.sketch_dims as usize, &mut self.scratch);
        let p = &mut self.pending;
        p.ids.extend_from_slice(&id.to_le_bytes());
        p.docs.extend_from_slice(&doc_id.to_le_bytes());
        p.ftypes.push(code);
        p.sketch.extend_from_slice(&e.bits[..sketch_bytes]);
        p.bits.extend_from_slice(&e.bits[sketch_bytes..]);
        for c in e.corr {
            p.corr.extend_from_slice(&c.to_le_bytes());
        }
        p.q8.extend(e.q8.iter().map(|c| *c as u8));
        p.q8corr.extend_from_slice(&e.q8_factor.to_le_bytes());
        p.q8corr.extend_from_slice(&0f32.to_le_bytes());
        if self.header.lossless() {
            for x in &self.unit {
                p.f16.extend_from_slice(&f16::from_f32(*x).to_bits().to_le_bytes());
            }
        }
        p.rows += 1;
        self.dirty = true;
        if p.rows >= self.segment_rows {
            self.flush_segment()?;
        }
        Ok(())
    }

    fn write_section(&mut self, segment: u32, kind: u16, rows: u64, bytes: &[u8]) -> Result<()> {
        let offset = format::align_up(self.end);
        self.file.seek(SeekFrom::Start(self.end))?;
        let pad = vec![0u8; (offset - self.end) as usize];
        self.file.write_all(&pad)?;
        self.file.write_all(bytes)?;
        self.end = offset + bytes.len() as u64;
        self.footer.sections.push(SectionEntry {
            segment, kind, codec: 0, offset, length: bytes.len() as u64, rows,
            checksum: xxh3_64(bytes), block_checksums_offset: 0,
        });
        Ok(())
    }

    fn flush_segment(&mut self) -> Result<()> {
        if self.pending.rows == 0 {
            return Ok(());
        }
        let p = std::mem::take(&mut self.pending);
        let seg = self.footer.segment_count;
        let mut parts = vec![
            (section::IDS, p.ids), (section::DOC, p.docs), (section::FTYPE, p.ftypes),
            (section::SKETCH, p.sketch), (section::BITS, p.bits), (section::CORR, p.corr),
            (section::Q8, p.q8), (section::Q8CORR, p.q8corr),
        ];
        if self.header.lossless() {
            parts.push((section::F16, p.f16));
        }
        for (kind, bytes) in parts {
            self.write_section(seg, kind, p.rows, &bytes)?;
        }
        self.footer.segment_count += 1;
        self.footer.row_count += p.rows;
        self.footer.live_row_count += p.rows;
        Ok(())
    }

    fn read_at(&mut self, offset: u64, len: u64) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; len as usize];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }

    /// Mark every row of these documents deleted. Returns how many rows that removed.
    pub fn delete_docs(&mut self, doc_ids: &[u32]) -> Result<u64> {
        self.flush_segment()?;
        let wanted: std::collections::HashSet<u32> = doc_ids.iter().copied().collect();
        let mut removed = 0u64;
        for seg in 0..self.footer.segment_count {
            let Some(doc) = self.footer.sections.iter().find(|s| s.segment == seg && s.kind == section::DOC).cloned() else { continue };
            let docs = self.read_at(doc.offset, doc.length)?;
            let old = self.footer.sections.iter().rev().find(|s| s.segment == seg && s.kind == section::TOMB).cloned();
            let mut tomb = match &old {
                Some(t) => self.read_at(t.offset, t.length)?,
                None => vec![0u8; (doc.rows as usize).div_ceil(8)],
            };
            let mut changed = 0u64;
            for (row, d) in docs.chunks_exact(4).enumerate() {
                let id = u32::from_le_bytes(d.try_into().unwrap());
                if wanted.contains(&id) && tomb[row / 8] >> (row % 8) & 1 == 0 {
                    tomb[row / 8] |= 1 << (row % 8);
                    changed += 1;
                }
            }
            if changed > 0 {
                self.footer.sections.retain(|s| !(s.segment == seg && s.kind == section::TOMB));
                self.write_section(seg, section::TOMB, doc.rows, &tomb)?;
                removed += changed;
            }
        }
        if removed > 0 {
            self.footer.live_row_count -= removed;
            self.dirty = true;
        }
        Ok(removed)
    }

    /// Make everything added or deleted since the last commit durable and visible to readers.
    /// Returns the new generation number.
    pub fn commit(&mut self) -> Result<u64> {
        self.flush_segment()?;
        if !self.dirty {
            return Ok(self.footer.generation);
        }
        self.footer.generation += 1;
        let body = self.footer.to_bytes();
        self.file.seek(SeekFrom::Start(self.end))?;
        self.file.write_all(&body)?;
        self.file.write_all(&format::trailer_bytes(self.end, &body))?;
        self.file.sync_all()?;
        self.end += (body.len() + format::TRAILER_LEN) as u64;
        self.dirty = false;
        Ok(self.footer.generation)
    }
}

fn new_uuid(path: &Path) -> [u8; 16] {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let seed = format!("{}|{}|{}", nanos, std::process::id(), path.display());
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&xxh3_64(seed.as_bytes()).to_le_bytes());
    out[8..].copy_from_slice(&xxh3_64(format!("{seed}|2").as_bytes()).to_le_bytes());
    out
}
