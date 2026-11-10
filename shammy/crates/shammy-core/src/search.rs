//! The query path: sketch scan over every live row, then two reranks on larger codes, then an
//! exact finish on the lossless tier. This is the plain (scalar, single-threaded) version; the
//! SIMD kernels of milestone 2 must return exactly what this returns.

use half::f16;

use crate::encode::{normalize, pack_bits};
use crate::error::{Error, Result};
use crate::kernels;
use crate::reader::{Reader, Segment};

/// Restrict a search. `None` means no restriction on that field.
#[derive(Clone, Debug, Default)]
pub struct Filter {
    pub file_types: Option<Vec<String>>,
    pub doc_ids: Option<Vec<u32>>,
}

/// Candidates kept after each stage. 0 picks the default.
#[derive(Clone, Copy, Debug, Default)]
pub struct Params {
    /// After the sketch scan. Default: 1% of live rows, at least 4,096.
    pub stage1: usize,
    /// After the full 1-bit estimate. Default 256.
    pub stage2: usize,
    /// After the 8-bit rerank, rescored exactly when the file is lossless. Default: 50, at least k.
    pub finish: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub id: u64,
    pub doc_id: u32,
    pub score: f32,
}

/// Per-stage counts for one search, for tuning and tests.
#[derive(Clone, Copy, Debug, Default)]
pub struct Trace {
    pub scanned: usize,
    pub stage1: usize,
    pub stage2: usize,
    pub finish: usize,
    pub exact_finish: bool,
}

#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    seg: u32,
    row: u32,
}

/// Best first; ties broken by position so results never depend on the order of work.
fn keep_best(c: &mut Vec<Cand>, n: usize) {
    let order = |a: &Cand, b: &Cand| b.score.total_cmp(&a.score).then(a.seg.cmp(&b.seg)).then(a.row.cmp(&b.row));
    if c.len() > n {
        c.select_nth_unstable_by(n, order);
        c.truncate(n);
    }
    c.sort_unstable_by(order);
}

impl Reader {
    pub fn search(&self, query: &[f32], k: usize, filter: &Filter, params: &Params) -> Result<Vec<Hit>> {
        Ok(self.search_traced(query, k, filter, params)?.0)
    }

    pub fn search_traced(&self, query: &[f32], k: usize, filter: &Filter, params: &Params) -> Result<(Vec<Hit>, Trace)> {
        let dims = self.header.dims as usize;
        if query.len() != dims {
            return Err(Error::Invalid(format!("the query has {} dimensions, the file stores {dims}", query.len())));
        }
        if query.iter().any(|x| !x.is_finite()) {
            return Err(Error::Invalid("the query contains NaN or infinity".into()));
        }
        let mut trace = Trace::default();
        if k == 0 || self.footer.live_row_count == 0 {
            return Ok((Vec::new(), trace));
        }
        let mut unit = query.to_vec();
        normalize(&mut unit);
        let mut rotated = unit.clone();
        self.rot.apply(&mut rotated);
        let qbits = pack_bits(&rotated);
        let sketch_bytes = self.header.sketch_dims as usize / 8;

        let live = self.footer.live_row_count as usize;
        let stage1 = if params.stage1 == 0 { (live / 100).max(4096) } else { params.stage1 }.max(k);
        let stage2 = if params.stage2 == 0 { 256 } else { params.stage2 }.max(k);
        let finish = if params.finish == 0 { 50 } else { params.finish }.max(k);

        // Filter: allowed file-type codes and document ids.
        let type_ok: Option<[bool; 256]> = filter.file_types.as_ref().map(|names| {
            let mut ok = [false; 256];
            for (code, t) in self.footer.file_types.iter().enumerate() {
                ok[code] = names.iter().any(|n| n == t);
            }
            ok
        });
        let docs_ok: Option<std::collections::HashSet<u32>> = filter.doc_ids.as_ref().map(|d| d.iter().copied().collect());

        // Stage 1: Hamming distance between sign bits of the sketch dimensions, for every row.
        // Distances are small integers, so the best `stage1` rows are found with a histogram
        // instead of a sort; rows at the cut-off distance are taken in file order.
        const SKIP: u16 = u16::MAX;
        let mut dist: Vec<Vec<u16>> = Vec::with_capacity(self.segments.len());
        let mut histogram = vec![0usize; self.header.sketch_dims as usize + 1];
        for seg in &self.segments {
            let mut d = vec![0u16; seg.rows];
            kernels::hamming_rows(self.bytes(seg.sketch), sketch_bytes, &qbits[..sketch_bytes], &mut d);
            let ftype = self.bytes(seg.ftype);
            let doc = self.bytes(seg.doc);
            let filtered = seg.tomb.is_some() || type_ok.is_some() || docs_ok.is_some();
            for (row, x) in d.iter_mut().enumerate() {
                if filtered {
                    let out = self.deleted(seg, row)
                        || type_ok.as_ref().is_some_and(|ok| !ok[ftype[row] as usize])
                        || docs_ok.as_ref().is_some_and(|ok| !ok.contains(&u32::from_le_bytes(doc[row * 4..row * 4 + 4].try_into().unwrap())));
                    if out {
                        *x = SKIP;
                        continue;
                    }
                }
                histogram[*x as usize] += 1;
            }
            dist.push(d);
        }
        trace.scanned = histogram.iter().sum();
        let (mut cutoff, mut below) = (0usize, 0usize);
        while cutoff < histogram.len() && below + histogram[cutoff] < stage1 {
            below += histogram[cutoff];
            cutoff += 1;
        }
        let mut at_cutoff = stage1.saturating_sub(below);
        let mut cands: Vec<Cand> = Vec::with_capacity(stage1.min(trace.scanned));
        for (s, d) in dist.iter().enumerate() {
            for (row, x) in d.iter().enumerate() {
                let x = *x as usize;
                if x < cutoff || (x == cutoff && at_cutoff > 0) {
                    if x == cutoff {
                        at_cutoff -= 1;
                    }
                    cands.push(Cand { score: -(x as f32), seg: s as u32, row: row as u32 });
                }
            }
        }
        drop(dist);
        trace.stage1 = cands.len();

        // Stage 2: all sign bits against the rotated query (RaBitQ estimate of the dot product).
        for c in cands.iter_mut() {
            let seg = &self.segments[c.seg as usize];
            c.score = self.one_bit_estimate(seg, c.row as usize, &rotated, sketch_bytes);
        }
        keep_best(&mut cands, stage2);
        trace.stage2 = cands.len();

        // Stage 3: 8-bit codes.
        for c in cands.iter_mut() {
            let seg = &self.segments[c.seg as usize];
            let row = c.row as usize;
            let code = &self.bytes(seg.q8)[row * dims..(row + 1) * dims];
            let dot: f64 = code.iter().zip(&rotated).map(|(b, q)| (*b as i8) as f64 * *q as f64).sum();
            c.score = (dot * f32_at(self.bytes(seg.q8corr), row * 8) as f64) as f32;
        }
        let lossless = self.header.lossless();
        keep_best(&mut cands, if lossless { finish } else { k });
        trace.finish = cands.len();

        // Finish: the stored float16 vector against the query, exactly.
        if lossless {
            trace.exact_finish = true;
            for c in cands.iter_mut() {
                let seg = &self.segments[c.seg as usize];
                let row = c.row as usize;
                let v = &self.bytes(seg.f16.unwrap())[row * dims * 2..(row + 1) * dims * 2];
                let dot: f64 = v.chunks_exact(2).zip(&unit)
                    .map(|(h, q)| f16::from_bits(u16::from_le_bytes([h[0], h[1]])).to_f32() as f64 * *q as f64).sum();
                c.score = dot as f32;
            }
            keep_best(&mut cands, k);
        }

        let mut hits: Vec<Hit> = cands.iter().map(|c| {
            let seg = &self.segments[c.seg as usize];
            let row = c.row as usize;
            Hit {
                id: u64::from_le_bytes(self.bytes(seg.ids)[row * 8..row * 8 + 8].try_into().unwrap()),
                doc_id: u32::from_le_bytes(self.bytes(seg.doc)[row * 4..row * 4 + 4].try_into().unwrap()),
                score: c.score,
            }
        }).collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.id.cmp(&b.id)));
        Ok((hits, trace))
    }

    fn one_bit_estimate(&self, seg: &Segment, row: usize, rotated: &[f32], sketch_bytes: usize) -> f32 {
        let rest_bytes = rotated.len() / 8 - sketch_bytes;
        let a = &self.bytes(seg.sketch)[row * sketch_bytes..(row + 1) * sketch_bytes];
        let b = &self.bytes(seg.bits)[row * rest_bytes..(row + 1) * rest_bytes];
        let mut dot = 0f64;
        for (i, q) in rotated.iter().enumerate() {
            let byte = if i / 8 < sketch_bytes { a[i / 8] } else { b[i / 8 - sketch_bytes] };
            dot += if byte >> (i % 8) & 1 == 1 { -(*q as f64) } else { *q as f64 };
        }
        (dot * f32_at(self.bytes(seg.corr), row * 8 + 4) as f64) as f32
    }

    /// The stored vector for a chunk id (from the lossless tier), or None.
    pub fn vector(&self, id: u64) -> Option<Vec<f32>> {
        let dims = self.header.dims as usize;
        for seg in &self.segments {
            let f = seg.f16?;
            let ids = self.bytes(seg.ids);
            for row in 0..seg.rows {
                if u64::from_le_bytes(ids[row * 8..row * 8 + 8].try_into().unwrap()) == id && !self.deleted(seg, row) {
                    let v = &self.bytes(f)[row * dims * 2..(row + 1) * dims * 2];
                    return Some(v.chunks_exact(2).map(|h| f16::from_bits(u16::from_le_bytes([h[0], h[1]])).to_f32()).collect());
                }
            }
        }
        None
    }

    /// Ids of every live row, in file order.
    pub fn live_ids(&self) -> Vec<u64> {
        let mut out = Vec::with_capacity(self.footer.live_row_count as usize);
        for seg in &self.segments {
            let ids = self.bytes(seg.ids);
            out.extend((0..seg.rows).filter(|r| !self.deleted(seg, *r))
                .map(|r| u64::from_le_bytes(ids[r * 8..r * 8 + 8].try_into().unwrap())));
        }
        out
    }

    /// Exact search over the lossless tier: the ground truth that `search` is measured against.
    pub fn search_exact(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<Hit>> {
        if !self.header.lossless() {
            return Err(Error::Invalid("this file has no lossless tier to search exactly".into()));
        }
        let all = Params { stage1: usize::MAX, stage2: usize::MAX, finish: usize::MAX };
        self.search(query, k, filter, &all)
    }
}

fn f32_at(b: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
