# Shammy for .NET

`Shammy/` is a small class library (net8.0, no package dependencies) that reads `.shammy` files
through the native library (`shammy.dll` on Windows, `libshammy.so` on Linux).

## Get the pieces

- **Native library:** `cargo build --release -p shammy-ffi` in `shammy/` gives
  `target/release/shammy.dll` (or `libshammy.so`). Release builds attach it as
  `shammy-native-Windows.zip`, with `shammy.h`.
- **Wrapper:** reference `Shammy/Shammy.csproj` from the server project, or copy `ShammyIndex.cs`
  into it (it needs `<AllowUnsafeBlocks>true</AllowUnsafeBlocks>`).
- **Deploy:** put `shammy.dll` next to the server's executable. In the server's `.csproj`:

  ```xml
  <ItemGroup>
    <None Include="native/shammy.dll" CopyToOutputDirectory="PreserveNewest" Link="shammy.dll" />
  </ItemGroup>
  ```

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

## Reading a `ragpack/2` pack in the retrieval server

A `ragpack/2` pack is a `ragpack/1` pack whose vectors are in `vectors.shammy` instead of the
`chunk_vec` table (which is present but empty). `manifest.json` has `"format": "ragpack/2"`,
`"vector_store": "shammy"` and `"vector_tier": "full"` or `"slim"`. Everything else (chunks,
`chunks_fts`, `tables.db`, assets) is unchanged, so only the vector query changes:

```csharp
// when a pack is loaded
if (manifest.Format == "ragpack/2")
{
    var report = ShammyIndex.Verify(vectorsPath);              // optional: checks every checksum
    if (!report.Ok) throw new InvalidDataException(report.FirstProblem);
    pack.Vectors = ShammyIndex.Open(vectorsPath);
    if (pack.Vectors.Dimensions != embedder.Dimensions || pack.Vectors.Model != manifest.Embedding.Model)
        throw new InvalidDataException("the pack was built with a different embedding model");
}

// in the search path, where ragpack/1 runs
//   SELECT chunk_id FROM chunk_vec WHERE embedding MATCH ? AND k = 40 [AND file_type = ?]
long[] vectorIds = pack.Vectors is null
    ? QueryChunkVec(queryEmbedding, 40, fileType)
    : pack.Vectors.Search(queryEmbedding, 40, new ShammySearchOptions { FileTypes = fileType is null ? null : [fileType] })
          .Select(h => (long)h.Id).ToArray();
// then, as before: fuse with the FTS5 ids, load the rows from chunks by chunk_id
```

The ids Shammy returns are `chunks.chunk_id` values, best first, the same list the `chunk_vec`
query gave.

## Test

`python -m pytest tests/test_shammy_native.py` (from the repository root) builds a file with the
Python module and checks that the C interface and this wrapper return exactly the same hits.
