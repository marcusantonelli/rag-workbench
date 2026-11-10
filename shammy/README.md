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

## Python

```bash
pip install maturin        # once; needs the Rust toolchain
shammy/build_wheel.sh      # builds the wheel and installs it into the current Python
```

```python
import numpy as np, shammy

with shammy.Index.create("kb.shammy", dims=1024, model="my-embedder") as ix:
    ix.add(ids=[1, 2, 3], vectors=np.random.rand(3, 1024), doc_ids=[10, 10, 11], file_types="pdf")
    ix.commit()                      # durable, and visible to searches

ix = shammy.Index.open("kb.shammy")  # read-only; any number of processes
for hit in ix.search(query_vector, k=10, file_types=["pdf"]):
    print(hit.id, hit.doc_id, hit.score)
```

- `Index.open(path, write=True)` appends; `delete_docs([...])` removes a document's rows;
  `shammy.compact(src, dst)` rewrites a file without deleted rows; `shammy.verify(path)` checks
  every checksum.
- One process writes at a time (`<file>.lock`); readers see the last commit.
- Vectors shorter than the stored size are padded with zeros, which leaves cosine similarity
  unchanged, so a 768- or 1536-dimension model is stored at 1024 or 2048.
- `search(..., exact=True)` scores every row in full; `threads=` sets the thread count.

## Import from SQLite or PostgreSQL

```bash
shammy-convert sqlite kb.db out.shammy --table chunk_vec --id chunk_id --doc doc_id --type file_type
shammy-convert pgvector "postgresql://user@host/db" out.shammy --table items --id id --vector embedding
```

- `sqlite` reads sqlite-vec `vec0` tables (the `sqlite-vec` package must be installed) and ordinary
  tables whose vector column holds float32 blobs or JSON text.
- `pgvector` streams a `vector` column through psycopg (`pip install "psycopg[binary]"`).
- `--where` picks rows, `--doc` and `--type` name columns to filter by later, `--slim` leaves out
  the float16 copies.
- Ids must be whole numbers. Otherwise rows are numbered from 1 and the original ids go to
  `out.shammy.ids.csv`.
- From Python: `shammy.convert.from_sqlite`, `from_pgvector`, or `from_rows` for any source that
  yields `(id, vector, doc_id, file_type)`.

For a RAG Workbench workspace use `rag convert WS` instead: it switches the workspace in place.

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

- **Milestone 3 (done):** Python module (`crates/shammy-py`, one abi3 wheel per platform),
  compaction, a writer lock, importers for SQLite and pgvector, and the Builder's vector-store
  switch (`rag init --vectors shammy`, `rag convert`, `rag compact`).
- **Milestone 4 (done):** the C interface (`crates/shammy-ffi`, `include/shammy.h`), the C#
  wrapper (`dotnet/`, with how to read a `ragpack/2` pack in the retrieval server), and slim
  files (`rag export --slim`, `shammy.compact(src, dst, slim=True)`). C, C# and Python return
  the same hits bit for bit; a test checks it.
- **Next, milestone 5:** the keyword index.

## Milestone 0 script

```bash
python shammy/prototype/m0_funnel.py /path/to/pack.ragpack
```

Reads stored vectors (read-only), prints three tables and a verdict, writes `m0_results.json`.
The pack must hold real embeddings; the larger the better.
