# Shammy

The purpose-built vector store for this pipeline: one memory-mapped file per knowledge base,
no index structure, a scan of 1-bit sketches followed by reranking on larger codes. Design:
"Shammy: low-level plan for a purpose-built vector store". Not wired into `rag` yet
(that is milestone 3).

| | |
|---|---|
| `crates/shammy-core` | The engine: format, encoder, writer, reader, search, verify |
| `crates/shammy-cli` | `shammy`: info, verify, build, delete, search, bench (developer tool) |
| `python/shammy_ref.py` | The specification as NumPy code; the engine is tested against it |
| `spec/FORMAT.md` | The byte-level format |
| `prototype/m0_funnel.py` | Milestone 0: measures the design on a real workspace or pack |

## Build and test

```bash
cd shammy
cargo test --release                 # engine tests
cargo build --release                # target/release/shammy
cd .. && python -m pytest tests/test_shammy_parity.py   # engine vs NumPy reference
```

## Try it

Vectors go in as raw little-endian float32 (for example `array.astype("<f4").tofile("v.f32")`):

```bash
shammy build kb.shammy --dims 4096 --vectors v.f32 --model Qwen3-Embedding-8B
shammy info kb.shammy
shammy verify kb.shammy
shammy search kb.shammy --query q.f32 -k 10
shammy search kb.shammy --query q.f32 -k 10 --exact     # ground truth from the lossless tier
shammy bench kb.shammy                                  # recall and ms per search, by stage-1 count
```

## Status

- **Milestone 0 (done):** on the 651-chunk Siemens pack (Qwen3-Embedding-8B, 4096 dims) the
  8-bit code alone finds 99.8% of the exact top 20 and the 1-bit code 91%; block rotation costs
  nothing against a full rotation; a centroid doesn't help. The pack is too small to say how
  few candidates stage 1 can keep at a million chunks; `shammy bench` (milestone 2) will measure
  that on larger data.
- **Milestone 1 (done):** format 1.0, append-only writer with commits and tombstones, reader,
  scalar single-threaded search, verify, crash and corruption tests, bit-exact parity with the
  reference.
- **Milestone 2 (done):** `shammy bench`; AVX2 and AVX-512 kernels for all three coded stages;
  every stage split across threads (`Params.threads`, `--threads`; default two fewer than the
  machine's cores, at least 1, no upper limit). Results are identical for any kernel and any
  thread count, and tests check both. On 50,000 x 4096 a search takes 1 to 2 ms on one thread;
  `bench` prints the time spent in each stage.
  Not built: stopping early from the error bounds. Measured, it could only save the 8-bit
  stage (about 0.2 ms), and tuning how many candidates stage 1 keeps per query needs real
  vectors at scale, which the synthetic test file can't stand in for.

## Milestone 0 script

```bash
python shammy/prototype/m0_funnel.py /path/to/pack.ragpack
```

Reads stored vectors (read-only), prints three tables and a verdict, writes `m0_results.json`.
The pack must hold real embeddings; the larger the better.
