//! The query path: sketch scan over every live row, then two reranks on larger codes, then an
//! exact finish on the lossless tier. Every score depends only on its own row and the query, and
//! ties are broken by position, so the result is the same for any kernel and any thread count.

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
    /// Worker threads for one search. Default: `default_threads()`.
    pub threads: usize,
}

/// Two fewer than the machine's cores (at least 1), leaving room for whatever else runs there.
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).saturating_sub(2).max(1)
}

/// Run `f(offset, part)` over consecutive parts of `items`, on up to `threads` threads when each
/// would get at least `min_each` items. Results come back in order of position.
fn in_parts<T: Send, R: Send>(items: &mut [T], threads: usize, min_each: usize, f: impl Fn(usize, &mut [T]) -> R + Sync) -> Vec<R> {
    let n = threads.min(items.len() / min_each.max(1)).max(1);
    if n == 1 {
        return vec![f(0, items)];
    }
    let each = items.len().div_ceil(n);
    std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = items.chunks_mut(each).enumerate().map(|(i, part)| s.spawn(move || f(i * each, part))).collect();
        handles.into_iter().map(|h| h.join().expect("a search thread panicked")).collect()
    })
}

/// Rows per thread below which a scan stays on fewer threads (starting a thread costs more).
const SCAN_ROWS_PER_THREAD: usize = 32_768;
const RERANK_ROWS_PER_THREAD: usize = 512;

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
    pub threads: usize,
    /// Time spent in the scan, the 1-bit rerank, the 8-bit rerank and the exact finish.
    pub nanos: [u64; 4],
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
        let threads = if params.threads == 0 { default_threads() } else { params.threads };
        trace.threads = threads;
        let mut clock = std::time::Instant::now();
        let mut lap = |slot: usize, trace: &mut Trace| {
            trace.nanos[slot] = clock.elapsed().as_nanos() as u64;
            clock = std::time::Instant::now();
        };

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
            let sketch = self.bytes(seg.sketch);
            let ftype = self.bytes(seg.ftype);
            let doc = self.bytes(seg.doc);
            let filtered = seg.tomb.is_some() || type_ok.is_some() || docs_ok.is_some();
            let parts = in_parts(&mut d, threads, SCAN_ROWS_PER_THREAD, |start, part| {
                let mut hist = vec![0usize; histogram.len()];
                kernels::hamming_rows(&sketch[start * sketch_bytes..(start + part.len()) * sketch_bytes], sketch_bytes,
                                      &qbits[..sketch_bytes], part);
                for (i, x) in part.iter_mut().enumerate() {
                    if filtered {
                        let row = start + i;
                        let out = self.deleted(seg, row)
                            || type_ok.as_ref().is_some_and(|ok| !ok[ftype[row] as usize])
                            || docs_ok.as_ref().is_some_and(|ok| !ok.contains(&u32::from_le_bytes(doc[row * 4..row * 4 + 4].try_into().unwrap())));
                        if out {
                            *x = SKIP;
                            continue;
                        }
                    }
                    hist[*x as usize] += 1;
                }
                hist
            });
            for hist in parts {
                for (total, n) in histogram.iter_mut().zip(hist) {
                    *total += n;
                }
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
        lap(0, &mut trace);

        // Stage 2: all sign bits against the rotated query (RaBitQ estimate of the dot product).
        in_parts(&mut cands, threads, RERANK_ROWS_PER_THREAD, |_, part| {
            for c in part.iter_mut() {
                let seg = &self.segments[c.seg as usize];
                c.score = self.one_bit_estimate(seg, c.row as usize, &rotated, sketch_bytes);
            }
        });
        keep_best(&mut cands, stage2);
        trace.stage2 = cands.len();
        lap(1, &mut trace);

        // Stage 3: 8-bit codes.
        in_parts(&mut cands, threads, RERANK_ROWS_PER_THREAD, |_, part| {
            for c in part.iter_mut() {
                let seg = &self.segments[c.seg as usize];
                let row = c.row as usize;
                let code = &self.bytes(seg.q8)[row * dims..(row + 1) * dims];
                let mut acc = [0f32; kernels::LANES];
                kernels::i8_dot(code, &rotated, &mut acc);
                c.score = (kernels::reduce(&acc) * f32_at(self.bytes(seg.q8corr), row * 8) as f64) as f32;
            }
        });
        let lossless = self.header.lossless();
        keep_best(&mut cands, if lossless { finish } else { k });
        trace.finish = cands.len();
        lap(2, &mut trace);

        // Finish: the stored float16 vector against the query, exactly.
        if lossless {
            trace.exact_finish = true;
            in_parts(&mut cands, threads, RERANK_ROWS_PER_THREAD / 8, |_, part| {
                for c in part.iter_mut() {
                    let seg = &self.segments[c.seg as usize];
                    let row = c.row as usize;
                    let v = &self.bytes(seg.f16.unwrap())[row * dims * 2..(row + 1) * dims * 2];
                    let dot: f64 = v.chunks_exact(2).zip(&unit)
                        .map(|(h, q)| f16::from_bits(u16::from_le_bytes([h[0], h[1]])).to_f32() as f64 * *q as f64).sum();
                    c.score = dot as f32;
                }
            });
            keep_best(&mut cands, k);
            lap(3, &mut trace);
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
        let split = sketch_bytes * 8;
        let mut acc = [0f32; kernels::LANES];
        kernels::sign_dot(a, &rotated[..split], 0, &mut acc);
        kernels::sign_dot(b, &rotated[split..], split, &mut acc);
        (kernels::reduce(&acc) * f32_at(self.bytes(seg.corr), row * 8 + 4) as f64) as f32
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

    /// The stored vectors for many chunk ids in one pass over the file (None where an id is
    /// absent, deleted, or the file has no lossless tier).
    pub fn vectors(&self, ids: &[u64]) -> Vec<Option<Vec<f32>>> {
        let dims = self.header.dims as usize;
        let mut out: Vec<Option<Vec<f32>>> = vec![None; ids.len()];
        let mut wanted: std::collections::HashMap<u64, Vec<usize>> = std::collections::HashMap::new();
        for (i, id) in ids.iter().enumerate() {
            wanted.entry(*id).or_default().push(i);
        }
        for seg in &self.segments {
            let Some(f) = seg.f16 else { continue };
            let stored = self.bytes(seg.ids);
            for row in 0..seg.rows {
                let id = u64::from_le_bytes(stored[row * 8..row * 8 + 8].try_into().unwrap());
                if let Some(slots) = wanted.get(&id) {
                    if self.deleted(seg, row) {
                        continue;
                    }
                    let v = &self.bytes(f)[row * dims * 2..(row + 1) * dims * 2];
                    let vec: Vec<f32> = v.chunks_exact(2).map(|h| f16::from_bits(u16::from_le_bytes([h[0], h[1]])).to_f32()).collect();
                    for s in slots {
                        out[*s] = Some(vec.clone());
                    }
                }
            }
        }
        out
    }

    /// Document ids of every live row, in file order (pairs with `live_ids`).
    pub fn live_docs(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.footer.live_row_count as usize);
        for seg in &self.segments {
            let docs = self.bytes(seg.doc);
            out.extend((0..seg.rows).filter(|r| !self.deleted(seg, *r))
                .map(|r| u32::from_le_bytes(docs[r * 4..r * 4 + 4].try_into().unwrap())));
        }
        out
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
        let all = Params { stage1: usize::MAX, stage2: usize::MAX, finish: usize::MAX, threads: 0 };
        self.search(query, k, filter, &all)
    }
}

fn f32_at(b: &[u8], at: usize) -> f32 {
    f32::from_le_bytes(b[at..at + 4].try_into().unwrap())
}
