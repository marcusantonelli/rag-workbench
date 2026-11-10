use shammy_core::rotate::SplitMix64;
use shammy_core::{verify, Filter, Params, Reader, Schema, Writer};
use std::path::PathBuf;

fn tmp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("shammy-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir.join("vectors.shammy")
}

/// Vectors with structure at three scales (20 topics, groups of 10 near-duplicates, small noise),
/// and early dimensions carrying more of the signal, like Matryoshka embeddings.
fn vectors(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64(seed);
    let mut unit = move || (rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
    let decay: Vec<f32> = (0..dims).map(|i| 1.0 / (1.0 + i as f32 / (dims as f32 / 8.0)).sqrt()).collect();
    let topics: Vec<Vec<f32>> = (0..20).map(|_| (0..dims).map(|i| unit() * decay[i]).collect()).collect();
    let mut out = Vec::with_capacity(n);
    let mut group = vec![0f32; dims];
    for i in 0..n {
        if i % 10 == 0 {
            let t = &topics[(i / 10) % 20];
            group = (0..dims).map(|d| t[d] + 0.7 * unit() * decay[d]).collect();
        }
        out.push((0..dims).map(|d| group[d] + 0.25 * unit() * decay[d]).collect());
    }
    out
}

/// Queries that sit near stored vectors, as real questions sit near their answers.
fn queries(data: &[Vec<f32>], n: usize) -> Vec<Vec<f32>> {
    let noise = vectors(n, data[0].len(), 4242);
    (0..n).map(|i| data[i * 37 % data.len()].iter().zip(&noise[i]).map(|(x, e)| x + 0.15 * e).collect()).collect()
}

fn build(path: &PathBuf, data: &[Vec<f32>], lossless: bool) {
    let mut schema = Schema::new(data[0].len() as u32);
    schema.lossless = lossless;
    schema.model = "test-model".into();
    schema.query_instruction = "Query: {query}".into();
    let mut w = Writer::create(path, &schema).unwrap();
    for (i, v) in data.iter().enumerate() {
        w.add(1000 + i as u64, (i / 10) as u32, if i % 3 == 0 { "pdf" } else { "md" }, v).unwrap();
    }
    assert_eq!(w.commit().unwrap(), 1);
}

fn recall(r: &Reader, queries: &[Vec<f32>], k: usize, params: &Params) -> f64 {
    let mut hit = 0;
    for q in queries {
        let truth: Vec<u64> = r.search_exact(q, k, &Filter::default()).unwrap().iter().map(|h| h.id).collect();
        let got = r.search(q, k, &Filter::default(), params).unwrap();
        hit += got.iter().filter(|h| truth.contains(&h.id)).count();
    }
    hit as f64 / (queries.len() * k) as f64
}

#[test]
fn build_reopen_and_search() {
    let path = tmp("search");
    let data = vectors(3000, 256, 1);
    build(&path, &data, true);
    let r = Reader::open(&path).unwrap();
    let info = r.info();
    assert_eq!((info.rows, info.live_rows, info.segments, info.generation), (3000, 3000, 1, 1));
    assert_eq!(info.file_types, vec!["pdf", "md"]);
    assert!(info.lossless && info.model == "test-model" && info.instruction_hash != 0);

    // A stored vector finds itself first, with a score of 1.
    let hits = r.search(&data[42], 5, &Filter::default(), &Params::default()).unwrap();
    assert_eq!(hits[0].id, 1042);
    assert!((hits[0].score - 1.0).abs() < 1e-3);
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
    let back = r.vector(1042).unwrap();
    let norm: f32 = back.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3);

    // The funnel keeps recall with a small fraction of the rows surviving stage 1.
    let queries = queries(&data, 40);
    let tight = Params { stage1: 300, stage2: 100, finish: 50 };
    let got = recall(&r, &queries, 10, &tight);
    assert!(got >= 0.97, "recall@10 with 10% of rows kept after stage 1 was {got}");
    let (_, trace) = r.search_traced(&queries[0], 10, &Filter::default(), &tight).unwrap();
    assert_eq!((trace.scanned, trace.stage1, trace.stage2, trace.finish, trace.exact_finish), (3000, 300, 100, 50, true));

    // The same search twice gives the same answer.
    let a = r.search(&queries[1], 20, &Filter::default(), &Params::default()).unwrap();
    assert_eq!(a, r.search(&queries[1], 20, &Filter::default(), &Params::default()).unwrap());
    assert!(verify(&path).unwrap().ok);
}

#[test]
fn filters_and_deletes() {
    let path = tmp("filter");
    let data = vectors(600, 64, 2);
    build(&path, &data, true);
    let q = &data[30]; // id 1030, doc 3, "pdf" (30 % 3 == 0)
    let r = Reader::open(&path).unwrap();
    let md = Filter { file_types: Some(vec!["md".into()]), doc_ids: None };
    let hits = r.search(q, 10, &md, &Params::default()).unwrap();
    assert_eq!(hits.len(), 10);
    assert!(hits.iter().all(|h| (h.id - 1000) % 3 != 0));
    let doc = Filter { file_types: None, doc_ids: Some(vec![3, 4]) };
    let hits = r.search(q, 50, &doc, &Params::default()).unwrap();
    assert_eq!(hits.len(), 20);
    assert!(hits.iter().all(|h| h.doc_id == 3 || h.doc_id == 4) && hits[0].id == 1030);
    let none = Filter { file_types: Some(vec!["docx".into()]), doc_ids: None };
    assert!(r.search(q, 10, &none, &Params::default()).unwrap().is_empty());
    drop(r);

    let mut w = Writer::open(&path).unwrap();
    assert_eq!(w.delete_docs(&[3]).unwrap(), 10);
    assert_eq!(w.delete_docs(&[3]).unwrap(), 0);
    assert_eq!(w.delete_docs(&[5, 999]).unwrap(), 10);
    assert_eq!(w.commit().unwrap(), 2);
    let r = Reader::open(&path).unwrap();
    assert_eq!((r.info().rows, r.info().live_rows), (600, 580));
    let hits = r.search(q, 600, &Filter::default(), &Params::default()).unwrap();
    assert_eq!(hits.len(), 580);
    assert!(hits.iter().all(|h| h.doc_id != 3 && h.doc_id != 5));
    assert!(r.vector(1030).is_none());
    assert!(verify(&path).unwrap().ok);
}

#[test]
fn appending_adds_segments_and_slim_files_work() {
    let path = tmp("append");
    let data = vectors(500, 64, 3);
    build(&path, &data[..300].to_vec(), false);
    let mut w = Writer::open(&path).unwrap();
    w.segment_rows = 80; // force several segments
    for (i, v) in data[300..].iter().enumerate() {
        w.add(5000 + i as u64, 77, "html", v).unwrap();
    }
    assert_eq!(w.commit().unwrap(), 2);
    assert_eq!(w.commit().unwrap(), 2, "a commit with nothing new changes nothing");
    let r = Reader::open(&path).unwrap();
    let info = r.info();
    assert_eq!((info.rows, info.segments, info.lossless), (500, 4, false));
    assert_eq!(info.file_types, vec!["pdf", "md", "html"]);
    let hits = r.search(&data[450], 3, &Filter::default(), &Params::default()).unwrap();
    assert_eq!(hits[0].id, 5150);
    assert!(r.search_exact(&data[0], 3, &Filter::default()).is_err());
    assert!(r.vector(5150).is_none());
}

#[test]
fn an_interrupted_write_falls_back_to_the_last_commit() {
    let path = tmp("truncate");
    let data = vectors(200, 64, 4);
    build(&path, &data[..100].to_vec(), true);
    let first = std::fs::metadata(&path).unwrap().len();
    let mut w = Writer::open(&path).unwrap();
    for (i, v) in data[100..].iter().enumerate() {
        w.add(9000 + i as u64, 1, "md", v).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    let whole = std::fs::read(&path).unwrap();
    assert_eq!(Reader::open(&path).unwrap().info().rows, 200);

    // Cut the second commit off at many points: the first commit must stay readable.
    let step = ((whole.len() as u64 - first) / 23).max(1) as usize;
    let mut cut = first as usize;
    while cut < whole.len() {
        std::fs::write(&path, &whole[..cut]).unwrap();
        let r = Reader::open(&path).unwrap();
        assert_eq!((r.info().rows, r.info().generation), (100, 1), "cut at {cut}");
        let rep = verify(&path).unwrap();
        assert!(rep.ok && rep.uncommitted_bytes == cut as u64 - first);
        cut += step;
    }
    // Reopening for writing drops the partial tail and continues cleanly.
    let mut w = Writer::open(&path).unwrap();
    w.add(1, 1, "md", &data[150]).unwrap();
    assert_eq!(w.commit().unwrap(), 2);
    assert_eq!(Reader::open(&path).unwrap().info().rows, 101);

    // Cut inside the first commit: nothing to fall back to.
    std::fs::write(&path, &whole[..first as usize - 5]).unwrap();
    assert!(Reader::open(&path).is_err());
    assert!(!verify(&path).unwrap().ok);
}

#[test]
fn corruption_is_reported_not_served_silently() {
    let path = tmp("corrupt");
    build(&path, &vectors(100, 64, 5), true);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[4096 * 4 + 17] ^= 0x40; // inside a data section
    std::fs::write(&path, &bytes).unwrap();
    let rep = verify(&path).unwrap();
    assert!(!rep.ok && rep.problems[0].contains("checksum mismatch"), "{:?}", rep.problems);

    bytes[4096 * 4 + 17] ^= 0x40;
    bytes[20] ^= 1; // header
    std::fs::write(&path, &bytes).unwrap();
    assert!(Reader::open(&path).is_err());
    std::fs::write(&path, b"this is not a shammy file at all, but it is long enough to have a header.....").unwrap();
    assert!(Reader::open(&path).err().unwrap().to_string().contains("not a Shammy file"));
}

#[test]
fn bad_input_is_refused() {
    let path = tmp("bad");
    assert!(Writer::create(&path, &Schema::new(100)).is_err());
    let mut w = Writer::create(&path, &Schema::new(64)).unwrap();
    assert!(Writer::create(&path, &Schema::new(64)).is_err(), "never overwrite an existing file");
    assert!(w.add(1, 1, "md", &[0.5; 32]).is_err());
    let mut nan = vec![0.1f32; 64];
    nan[3] = f32::NAN;
    assert!(w.add(1, 1, "md", &nan).is_err());
    w.commit().unwrap();
    let r = Reader::open(&path).unwrap();
    assert_eq!(r.info().rows, 0);
    assert!(r.search(&[0.1; 64], 5, &Filter::default(), &Params::default()).unwrap().is_empty());
    assert!(r.search(&[0.1; 32], 5, &Filter::default(), &Params::default()).is_err());
}

#[test]
#[ignore]
fn debug_recall() {
    let path = tmp("dbg");
    let data = vectors(3000, 256, 1);
    build(&path, &data, true);
    let r = Reader::open(&path).unwrap();
    let queries = queries(&data, 40);
    for (a, b, c) in [(3000, 3000, 3000), (3000, 3000, 50), (3000, 100, 50), (300, 300, 300), (300, 100, 50), (600, 100, 50), (1000, 200, 50)] {
        println!("{a} {b} {c}: {:.3}", recall(&r, &queries, 10, &Params { stage1: a, stage2: b, finish: c }));
    }
}
