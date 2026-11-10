//! `shammy`: a small developer tool for .shammy files.

use std::process::ExitCode;

use shammy_core::{verify, Filter, Params, Reader, Schema, Writer};

const USAGE: &str = "\
shammy: inspect and test .shammy vector files

  shammy info FILE
  shammy verify FILE
  shammy build FILE --dims D --vectors RAW.f32 [--ids RAW.u64] [--docs RAW.u32]
               [--type NAME] [--slim] [--sketch quarter|half] [--model NAME] [--append]
  shammy delete FILE DOC_ID...
  shammy search FILE --query RAW.f32 [--row N] [-k 20] [--stage1 N] [--stage2 N]
               [--finish N] [--exact] [--type NAME] [--doc ID]
  shammy bench FILE [--queries 100] [-k 20] [--stage2 N] [--finish N] [--stage1 N,N,...]

bench uses stored vectors as queries (the file needs its lossless tier) and reports, for several
stage-1 candidate counts, recall against exact search and the time per search.

RAW files are little-endian arrays with no header: float32 rows of D values for vectors,
uint64 for ids (default: 0, 1, 2, ...), uint32 for document ids (default: 0).
search prints one line per result: id, document id, score (tab-separated).";

struct Args(Vec<String>);

impl Args {
    fn flag(&mut self, name: &str) -> bool {
        match self.0.iter().position(|a| a == name) {
            Some(i) => {
                self.0.remove(i);
                true
            }
            None => false,
        }
    }
    fn value(&mut self, name: &str) -> Result<Option<String>, String> {
        match self.0.iter().position(|a| a == name) {
            Some(i) if i + 1 < self.0.len() => {
                self.0.remove(i);
                Ok(Some(self.0.remove(i)))
            }
            Some(_) => Err(format!("{name} needs a value")),
            None => Ok(None),
        }
    }
    fn number<T: std::str::FromStr>(&mut self, name: &str) -> Result<Option<T>, String> {
        match self.value(name)? {
            Some(v) => v.parse().map(Some).map_err(|_| format!("{name}: '{v}' is not a number")),
            None => Ok(None),
        }
    }
}

fn read_f32(path: &str) -> Result<Vec<f32>, String> {
    let b = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if b.len() % 4 != 0 {
        return Err(format!("{path}: size is not a multiple of 4 bytes"));
    }
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn run() -> Result<ExitCode, String> {
    let mut args = Args(std::env::args().skip(1).collect());
    if args.0.is_empty() || args.flag("--help") || args.flag("-h") {
        println!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    let cmd = args.0.remove(0);
    let e = |e: shammy_core::Error| e.to_string();
    match cmd.as_str() {
        "info" => {
            let path = args.0.first().ok_or("info needs a file")?;
            let i = Reader::open(path).map_err(e)?.info();
            println!("file            {path}");
            println!("rows            {} ({} live, {} deleted)", i.rows, i.live_rows, i.rows - i.live_rows);
            println!("dimensions      {} (sketch: first {})", i.dims, i.sketch_dims);
            println!("lossless tier   {}", if i.lossless { "yes (float16)" } else { "no" });
            println!("segments        {}", i.segments);
            println!("generation      {}", i.generation);
            println!("model           {}", if i.model.is_empty() { "(not recorded)" } else { &i.model });
            println!("file types      {}", i.file_types.join(", "));
            println!("size            {} bytes ({} per row)", i.file_bytes, if i.rows > 0 { i.file_bytes / i.rows } else { 0 });
            println!("uuid            {}", hex(&i.file_uuid));
            println!("rotation seed   {:#018x}", i.rotation_seed);
            if i.file_bytes > i.committed_bytes {
                println!("note            {} bytes after the last commit (an interrupted write) are ignored", i.file_bytes - i.committed_bytes);
            }
        }
        "verify" => {
            let path = args.0.first().ok_or("verify needs a file")?;
            let r = verify(path).map_err(e)?;
            if r.ok {
                println!("OK: {} sections, {} bytes checked; generation {}, {} rows ({} live)",
                         r.sections_checked, r.bytes_checked, r.generation, r.rows, r.live_rows);
                if r.uncommitted_bytes > 0 {
                    println!("note: {} bytes after the last commit are ignored", r.uncommitted_bytes);
                }
            } else {
                for p in &r.problems {
                    eprintln!("problem: {p}");
                }
                return Ok(ExitCode::from(1));
            }
        }
        "build" => {
            let vectors = read_f32(&args.value("--vectors")?.ok_or("build needs --vectors")?)?;
            let append = args.flag("--append");
            let slim = args.flag("--slim");
            let ftype = args.value("--type")?.unwrap_or_else(|| "bin".into());
            let ids = args.value("--ids")?;
            let docs = args.value("--docs")?;
            let dims: Option<u32> = args.number("--dims")?;
            let sketch = args.value("--sketch")?;
            let model = args.value("--model")?;
            let path = args.0.first().ok_or("build needs an output file")?.clone();
            let mut w = if append {
                Writer::open(&path).map_err(e)?
            } else {
                let mut s = Schema::new(dims.ok_or("build needs --dims")?);
                s.lossless = !slim;
                s.model = model.unwrap_or_default();
                if sketch.as_deref() == Some("half") {
                    s.sketch_dims = s.dims / 2;
                }
                Writer::create(&path, &s).map_err(e)?
            };
            let d = w.header().dims as usize;
            if vectors.len() % d != 0 {
                return Err(format!("--vectors holds {} values, not a multiple of {d}", vectors.len()));
            }
            let n = vectors.len() / d;
            let ids: Vec<u64> = match ids {
                Some(p) => std::fs::read(&p).map_err(|x| format!("{p}: {x}"))?.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect(),
                None => (0..n as u64).collect(),
            };
            let docs: Vec<u32> = match docs {
                Some(p) => std::fs::read(&p).map_err(|x| format!("{p}: {x}"))?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(),
                None => vec![0; n],
            };
            if ids.len() != n || docs.len() != n {
                return Err(format!("{n} vectors but {} ids and {} document ids", ids.len(), docs.len()));
            }
            for i in 0..n {
                w.add(ids[i], docs[i], &ftype, &vectors[i * d..(i + 1) * d]).map_err(e)?;
            }
            let generation = w.commit().map_err(e)?;
            println!("wrote {n} rows to {path} (generation {generation})");
        }
        "delete" => {
            if args.0.len() < 2 {
                return Err("delete needs a file and at least one document id".into());
            }
            let ids: Vec<u32> = args.0[1..].iter().map(|a| a.parse().map_err(|_| format!("'{a}' is not a document id"))).collect::<Result<_, _>>()?;
            let mut w = Writer::open(&args.0[0]).map_err(e)?;
            let n = w.delete_docs(&ids).map_err(e)?;
            let generation = w.commit().map_err(e)?;
            println!("deleted {n} rows (generation {generation})");
        }
        "search" => {
            let query = read_f32(&args.value("--query")?.ok_or("search needs --query")?)?;
            let row: usize = args.number("--row")?.unwrap_or(0);
            let k: usize = args.number("-k")?.unwrap_or(20);
            let params = Params {
                stage1: args.number("--stage1")?.unwrap_or(0),
                stage2: args.number("--stage2")?.unwrap_or(0),
                finish: args.number("--finish")?.unwrap_or(0),
            };
            let exact = args.flag("--exact");
            let filter = Filter {
                file_types: args.value("--type")?.map(|t| vec![t]),
                doc_ids: args.number("--doc")?.map(|d| vec![d]),
            };
            let r = Reader::open(args.0.first().ok_or("search needs a file")?).map_err(e)?;
            let d = r.info().dims as usize;
            let q = query.get(row * d..(row + 1) * d).ok_or(format!("--query has no row {row} of {d} values"))?;
            let hits = if exact { r.search_exact(q, k, &filter) } else { r.search(q, k, &filter, &params) }.map_err(e)?;
            for h in hits {
                println!("{}\t{}\t{:.6}", h.id, h.doc_id, h.score);
            }
        }
        "bench" => {
            let n: usize = args.number("--queries")?.unwrap_or(100);
            let k: usize = args.number("-k")?.unwrap_or(20);
            let stage2: usize = args.number("--stage2")?.unwrap_or(0);
            let finish: usize = args.number("--finish")?.unwrap_or(0);
            let custom = args.value("--stage1")?;
            let r = Reader::open(args.0.first().ok_or("bench needs a file")?).map_err(e)?;
            let info = r.info();
            if !info.lossless {
                return Err("bench needs the lossless tier (it is the ground truth)".into());
            }
            let ids = r.live_ids();
            if ids.len() <= k {
                return Err("too few rows to benchmark".into());
            }
            let live = ids.len();
            let step = (live / n.min(live)).max(1);
            let queries: Vec<Vec<f32>> = ids.iter().step_by(step).take(n).filter_map(|id| r.vector(*id)).collect();
            let none = Filter::default();
            let t = std::time::Instant::now();
            let truth: Vec<Vec<shammy_core::Hit>> = queries.iter().map(|q| r.search_exact(q, k, &none)).collect::<Result<_, _>>().map_err(e)?;
            println!("{} live rows x {} dims, {} queries (stored vectors), k = {k}", live, info.dims, queries.len());
            println!("exact search: {:.1} ms per query\n", t.elapsed().as_secs_f64() * 1e3 / queries.len() as f64);
            let grid: Vec<usize> = match custom {
                Some(c) => c.split(',').map(|x| x.trim().parse().map_err(|_| format!("--stage1: '{x}' is not a number"))).collect::<Result<_, _>>()?,
                None => {
                    let mut g: Vec<usize> = [0.001, 0.002, 0.004, 0.01, 0.02, 0.05].iter().map(|f| ((live as f64 * f) as usize).max(k)).collect();
                    g.dedup();
                    g
                }
            };
            println!("{:>8} {:>8} {:>10} {:>10} {:>10} {:>10}", "stage 1", "of rows", "recall", "worst", "p50 ms", "p99 ms");
            for s1 in grid {
                let params = Params { stage1: s1, stage2, finish };
                let (mut hit, mut worst) = (0usize, 1f64);
                let mut times = Vec::with_capacity(queries.len());
                for (q, want) in queries.iter().zip(&truth) {
                    let t = std::time::Instant::now();
                    let got = r.search(q, k, &none, &params).map_err(e)?;
                    times.push(t.elapsed().as_secs_f64() * 1e3);
                    // a result as good as the k-th exact one counts: duplicate vectors tie
                    let floor = want.last().map(|h| h.score).unwrap_or(0.0) - 1e-6;
                    let found = got.iter().filter(|h| h.score >= floor).count().min(want.len());
                    hit += found;
                    worst = worst.min(found as f64 / want.len() as f64);
                }
                times.sort_by(|a, b| a.total_cmp(b));
                let total: usize = truth.iter().map(|w| w.len()).sum();
                println!("{:>8} {:>7.2}% {:>10.4} {:>10.2} {:>10.2} {:>10.2}", s1, s1 as f64 / live as f64 * 100.0,
                         hit as f64 / total as f64, worst, times[times.len() / 2], times[(times.len() * 99 / 100).min(times.len() - 1)]);
            }
            println!("\nrecall: share of the exact top {k} returned; worst: the lowest single query. stage 2 = {}, finish = {}.",
                     if stage2 == 0 { 256 } else { stage2 }, if finish == 0 { 50.max(k) } else { finish });
        }
        other => return Err(format!("unknown command '{other}'\n\n{USAGE}")),
    }
    Ok(ExitCode::SUCCESS)
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::from(2)
        }
    }
}
