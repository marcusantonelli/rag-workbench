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

Console.WriteLine($"{passed} passed, {failed} failed");
return failed == 0 ? 0 : 1;
