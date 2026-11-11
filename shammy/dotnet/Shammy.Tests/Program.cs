// Checks the wrapper against results written by the Python module for the same file and queries.
// Usage: Shammy.Tests FIXTURE_DIR   (made by tests/test_shammy_native.py; needs the native library
// on the loader path, e.g. LD_LIBRARY_PATH=shammy/target/release)
using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading.Tasks;
using Shammy;

if (args.Length != 1) { Console.Error.WriteLine("usage: Shammy.Tests FIXTURE_DIR"); return 2; }
string dir = args[0];
int failed = 0, passed = 0;
void Expect(bool ok, string what) { if (ok) passed++; else { failed++; Console.Error.WriteLine("FAIL: " + what); } }

using var doc = JsonDocument.Parse(File.ReadAllText(Path.Combine(dir, "expected.json")));
var root = doc.RootElement;
int dims = root.GetProperty("dims").GetInt32();
float[] queries = MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(Path.Combine(dir, "queries.f32"))).ToArray();
string file = Path.Combine(dir, "kb.shammy");

using (var index = ShammyIndex.Open(file))
{
    Expect(index.Dimensions == dims, "dimensions");
    Expect(index.Count == root.GetProperty("live_rows").GetInt64(), "live rows");
    Expect(index.Model == root.GetProperty("model").GetString(), "model name (non-ASCII)");
    Expect(index.Lossless, "lossless");
    Expect(index.FileTypes.SequenceEqual(root.GetProperty("file_types").EnumerateArray().Select(e => e.GetString()!)), "file types");
    Expect(ShammyIndex.NativeVersion.Length > 0 && ShammyIndex.DefaultThreads >= 1, "version and threads");

    foreach (var c in root.GetProperty("cases").EnumerateArray())
    {
        int qi = c.GetProperty("query").GetInt32();
        int len = c.GetProperty("query_len").GetInt32();
        var options = new ShammySearchOptions
        {
            FileTypes = c.TryGetProperty("file_types", out var ft) ? ft.EnumerateArray().Select(e => e.GetString()!).ToArray() : null,
            DocIds = c.TryGetProperty("doc_ids", out var di) ? di.EnumerateArray().Select(e => e.GetUInt32()).ToArray() : null,
            Exact = c.GetProperty("exact").GetBoolean(),
            Threads = c.GetProperty("threads").GetInt32(),
        };
        var got = index.Search(queries.AsSpan(qi * dims, len), c.GetProperty("k").GetInt32(), options);
        var want = c.GetProperty("hits").EnumerateArray()
            .Select(h => new ShammyHit(h[0].GetUInt64(), h[1].GetUInt32(), h[2].GetSingle())).ToArray();
        Expect(got.SequenceEqual(want), $"case {c.GetProperty("name").GetString()}: {got.Length} hits, first {(got.Length > 0 ? got[0] : default)}");
    }

    // a stored vector comes back at unit length and finds itself
    ulong someId = root.GetProperty("some_id").GetUInt64();
    float[]? v = index.GetVector(someId);
    Expect(v is not null && Math.Abs(v.Sum(x => (double)x * x) - 1.0) < 1e-3, "GetVector returns a unit vector");
    Expect(v is not null && index.Search(v, 1)[0].Id == someId, "a stored vector finds itself");
    Expect(index.GetVector(ulong.MaxValue) is null, "GetVector of an unknown id is null");

    // many threads at once give the same answers
    var first = index.Search(queries.AsSpan(0, dims), 10);
    bool same = true;
    Parallel.For(0, 64, _ => { if (!index.Search(queries.AsSpan(0, dims), 10).SequenceEqual(first)) same = false; });
    Expect(same, "concurrent searches agree");

    try { index.Search(new float[dims + 1], 5); Expect(false, "an oversized query is refused"); }
    catch (ArgumentException) { passed++; }
    try { index.Search(queries.AsSpan(0, dims), 5, new ShammySearchOptions { FileTypes = new[] { "no-such-type" } }); passed++; }
    catch (ShammyException e) { Expect(e.Code == -1 && e.Message.Length > 0, "unknown file type error"); }
}

var report = ShammyIndex.Verify(file);
Expect(report.Ok && report.Problems == 0 && report.LiveRows == root.GetProperty("live_rows").GetInt64(), "verify a good file");

string damaged = Path.Combine(dir, "damaged.shammy");
var bytes = File.ReadAllBytes(file);
bytes[8192] ^= 0xFF;
File.WriteAllBytes(damaged, bytes);
var bad = ShammyIndex.Verify(damaged);
Expect(!bad.Ok && bad.Problems > 0 && !string.IsNullOrEmpty(bad.FirstProblem), "verify reports a damaged file");

try { ShammyIndex.Open(Path.Combine(dir, "missing.shammy")); Expect(false, "opening a missing file fails"); }
catch (ShammyException e) { Expect(e.Code < 0 && e.Message.Length > 0, "missing file error"); }
File.WriteAllText(Path.Combine(dir, "not.shammy"), "this is not a shammy file, just some text that is long enough to read a header from........");
try { ShammyIndex.Open(Path.Combine(dir, "not.shammy")); Expect(false, "opening a non-Shammy file fails"); }
catch (ShammyException e) { Expect(e.Code == -3, $"wrong format error code ({e.Code}: {e.Message})"); }

// The whole retrieval path: the wrapper must return what the Python module returned for the same
// knowledge bases, questions and options (rag.json is written by the test that runs this program).
string ragFile = Path.Combine(dir, "rag.json");
if (File.Exists(ragFile))
{
    using var rag = JsonDocument.Parse(File.ReadAllText(ragFile));
    foreach (var kb in rag.RootElement.GetProperty("knowledge_bases").EnumerateArray())
    {
        string folder = kb.GetProperty("folder").GetString()!;
        using var retriever = ShammyRetriever.Open(folder);
        Expect(retriever.VectorStore == kb.GetProperty("vector_store").GetString(), $"vector store of {folder}");
        Expect(retriever.Dimensions == kb.GetProperty("dims").GetInt32() && retriever.Chunks > 0, "retriever info");
        foreach (var c in kb.GetProperty("cases").EnumerateArray())
        {
            float[] v = c.GetProperty("vector").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            string name = c.GetProperty("name").GetString()!;
            string got = retriever.SearchJson(c.GetProperty("request").GetRawText(), v);
            using var gotDoc = JsonDocument.Parse(got);
            Expect(Same(gotDoc.RootElement, c.GetProperty("response")), $"retrieval case {name}");
        }

        // the typed call
        var first = kb.GetProperty("cases")[0];
        float[] qv = first.GetProperty("vector").EnumerateArray().Select(e => e.GetSingle()).ToArray();
        string question = first.GetProperty("request").GetProperty("query").GetString()!;
        var typed = retriever.Search(question, qv, new RetrievalOptions { K = 3 });
        Expect(typed.Results.Count == 3 && typed.Results[0].Tag == "S1" && typed.Results[0].File.Length > 0
               && typed.Markdown!.StartsWith("[S1] ") && typed.Reranked is null, "typed search");
        // structure: the Workbench writes a spots file with every build
        Expect(retriever.Procedures is not null, "spots file read");
        var none = retriever.Search(question, qv, new RetrievalOptions { K = 3, Documents = ["no-such-document"] });
        Expect(none.Results.Count == 0, "document filter");
        if (retriever.Procedures is > 0)
        {
            // read a whole procedure page by page: every page names the next, the last says so
            var page = retriever.ProcedureSteps(1, maxTokens: 200);
            int pages = 1;
            while (page.Next is { } next && pages < 500)
            {
                page = retriever.ProcedureSteps(next, 200);
                pages++;
            }
            Expect(page.Next is null && page.Markdown.Length > 0 && pages < 500, "procedure pages");
        }
        var flat = retriever.Search(question, qv, new RetrievalOptions { K = 3, Procedures = false });
        Expect(flat.Results.All(r => r.Procedure is null), "procedures off");
        if (kb.TryGetProperty("rerank_url", out var url))
        {
            var ranked = retriever.Search(question, qv, new RetrievalOptions { K = 3, Rerank = new RerankOptions { Url = url.GetString()! } });
            Expect(ranked.Reranked == true && ranked.Results[0].RerankScore is not null, "typed search with reranking");
            var down = retriever.Search(question, qv, new RetrievalOptions { K = 3, Rerank = new RerankOptions { Url = "http://127.0.0.1:9" } });
            Expect(down.Reranked == false && !string.IsNullOrEmpty(down.RerankError)
                   && down.Results.Select(r => r.ChunkId).SequenceEqual(typed.Results.Select(r => r.ChunkId)), "reranker down falls back");
        }
        try { retriever.Search("x", new float[7]); Expect(false, "a wrong-size vector is refused"); }
        catch (ShammyException e) { Expect(e.Code == -1, "wrong-size vector error"); }
    }
    try { ShammyRetriever.Open(dir); Expect(false, "a folder without rag.db is refused"); }
    catch (ShammyException e) { Expect(e.Code == -2 && e.Message.Contains("rag.db"), "not a knowledge base error"); }
}

Console.WriteLine($"{passed} passed, {failed} failed");
return failed == 0 ? 0 : 1;

// Same JSON value: numbers compared as numbers, object keys in any order, timings ignored.
static bool Same(JsonElement a, JsonElement b)
{
    if (a.ValueKind != b.ValueKind) return false;
    switch (a.ValueKind)
    {
        case JsonValueKind.Number: return a.GetDouble() == b.GetDouble();
        case JsonValueKind.String: return a.GetString() == b.GetString();
        case JsonValueKind.Array:
            return a.GetArrayLength() == b.GetArrayLength() && a.EnumerateArray().Zip(b.EnumerateArray()).All(p => Same(p.First, p.Second));
        case JsonValueKind.Object:
            var left = a.EnumerateObject().Where(p => p.Name != "timings_ms").ToList();
            if (left.Count != b.EnumerateObject().Count(p => p.Name != "timings_ms")) return false;
            return left.All(p => b.TryGetProperty(p.Name, out var other) && Same(p.Value, other));
        default: return true;
    }
}
