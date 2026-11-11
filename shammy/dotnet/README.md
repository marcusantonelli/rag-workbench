# Shammy for .NET

`Shammy/` is a small class library (net8.0, no package dependencies) that reads `.shammy` files
through the native library (`shammy.dll` on Windows, `libshammy.so` on Linux).

## Get the pieces

There are two DLLs, and only one of them is a .NET assembly:

| File | What it is | How it gets into your server |
|---|---|---|
| `shammy.dll` | The **native** library (Rust, C functions). Not a .NET assembly: Visual Studio's *Add Reference* refuses it ("the reference is invalid or unsupported"). | Copied next to the server's executable; the wrapper loads it at run time. |
| `Shammy.Net.dll` | The **.NET wrapper** (`ShammyIndex`, `ShammyRetriever`). | A project reference to `Shammy/Shammy.csproj`, or its two `.cs` files copied into the server project. |

- **Native library:** release builds attach it as `shammy-native-Windows.zip`, with `shammy.h`
  (or `cargo build --release -p shammy-ffi` in `shammy/` gives `target/release/shammy.dll`).
- **Wrapper**, either:
  - add `Shammy/Shammy.csproj` to the server's solution and reference the *project*
    (Add → Project Reference), or
  - copy `Shammy/ShammyIndex.cs` and `Shammy/ShammyRetriever.cs` into the server project and set
    `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>` in its `.csproj`.
- **Deploy the native library** next to the server's executable. Put it in a `native/` folder of
  the server project and add to its `.csproj`:

  ```xml
  <ItemGroup>
    <None Include="native/shammy.dll" CopyToOutputDirectory="PreserveNewest" Link="shammy.dll" />
  </ItemGroup>
  ```

  A `DllNotFoundException` for "shammy" at the first search means the file isn't next to the
  executable (or is the Linux/macOS build); `BadImageFormatException` means a 32-bit process (the
  library is 64-bit; build the server as x64 or AnyCPU without "Prefer 32-bit").

Before 0.4.1 the wrapper's own assembly was also named `Shammy.dll`. Windows file names ignore
case, so it and the native `shammy.dll` overwrote each other in the output folder. With a 0.4.0
wrapper, copy the two `.cs` files instead of referencing the project.

## Use

```csharp
using Shammy;

using var index = ShammyIndex.Open(@"C:\packs\kb\vectors.shammy");
ShammyHit[] hits = index.Search(queryEmbedding, k: 40);
ShammyHit[] pdfOnly = index.Search(queryEmbedding, 40, new ShammySearchOptions { FileTypes = ["pdf"] });
```

- `ShammyIndex` is safe to search from many threads at once. Keep one per pack for the life of
  the server and dispose it when the pack is unloaded.
- It shows the file as of when it was opened. After a pack is replaced, open a new one. On
  Windows the old one must be disposed before the file can be replaced.
- Scores are cosine similarity. `Search` uses two fewer threads than the machine has cores unless
  `Threads` is set.
- Failures throw `ShammyException` with the native code (-1 argument, -2 I/O, -3 file format).

## The retrieval server: one call per search

`ShammyRetriever` runs the whole search path: vector search, keyword search, rank fusion,
duplicate removal, optional reranking, neighbour expansion and citations. It is the same code the
Workbench runs, so the agent gets exactly what the Workbench's Search tab shows. It reads
`ragpack/1` packs (vectors in sqlite-vec) and `ragpack/2` packs (vectors in `vectors.shammy`)
alike, so the server doesn't need its own SQLite, sqlite-vec or fusion code for search.

```csharp
using Shammy;

// when a pack is loaded (the folder it was unpacked to, the one holding rag.db)
var kb = ShammyRetriever.Open(packFolder);
// kb.Model, kb.Dimensions and kb.QueryInstruction say how queries must be embedded

// per request from the agent
float[] vector = await embedder.EmbedAsync(kb.QueryInstruction + question);   // your existing embedding call
RetrievalResult found = kb.Search(question, vector, new RetrievalOptions
{
    K = 6,
    FileType = fileType,                                            // null for all
    Rerank = rerankOn ? new RerankOptions { Url = "http://127.0.0.1:8082" } : null,
});
return found.Markdown;      // the compact text for the agent; found.Results / found.Json for structure
```

- `found.Reranked` is `true` when the reranker's order was used, `false` when reranking was asked
  for but failed (results are then in the usual order and `found.RerankError` says why), and
  `null` when it wasn't asked for. A reranker that is down never fails the search.
- The reranker is any `/v1/rerank` endpoint over plain http, for example
  `llama-server -m Qwen3-Reranker-0.6B-Q8_0.gguf --reranking --port 8082`.
- `kb.SearchJson(requestJson, vector)` takes and returns raw JSON (fields in `shammy.h`) if the
  server passes the agent's options straight through.
- Table tools (`describe_table`, `query_table`) still read `tables.db` in the server as before;
  `found.Tables` lists the tables whose description ranked.
- `ShammyIndex` (above) remains for using a `.shammy` file on its own.

## Test

`python -m pytest tests/test_shammy_native.py` (from the repository root) builds a vector file and two
knowledge bases with the Python module and checks that the C interface and this wrapper return
exactly what Python returns, for vector search and for whole retrievals (with and without
reranking).
