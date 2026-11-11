using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text.Json;

namespace Shammy;

/// <summary>Reranking through a <c>/v1/rerank</c> endpoint (llama-server --reranking, Jina-style servers).</summary>
public sealed class RerankOptions
{
    public bool Enabled { get; init; } = true;

    /// <summary>For example <c>http://127.0.0.1:8082</c> ("/v1/rerank" is added) or a full endpoint. Plain http.</summary>
    public string Url { get; init; } = "";

    public string Model { get; init; } = "";

    /// <summary>How many fused candidates are reranked (default 20).</summary>
    public int? Candidates { get; init; }

    public int? TimeoutMs { get; init; }
}

/// <summary>Optional settings for one retrieval. Unset values use the Workbench's defaults.</summary>
public sealed class RetrievalOptions
{
    /// <summary>Results returned (default 6).</summary>
    public int? K { get; init; }

    /// <summary>Only chunks from files of this type, for example "pdf".</summary>
    public string? FileType { get; init; }

    /// <summary>Join a prose hit with its neighbours from the same section (default true).</summary>
    public bool? Expand { get; init; }

    public int? VectorCandidates { get; init; }
    public int? KeywordCandidates { get; init; }
    public int? Neighbors { get; init; }
    public int? MaxResultTokens { get; init; }
    public int? Threads { get; init; }

    /// <summary>Also produce <see cref="RetrievalResult.Markdown"/>, the compact text an agent reads (default true).</summary>
    public bool Markdown { get; init; } = true;

    /// <summary>Rerank the fused candidates with a model. Null means no reranking.</summary>
    public RerankOptions? Rerank { get; init; }

    /// <summary>With a spots file (structure.spots): a hit inside a procedure brings its procedure (default true).</summary>
    public bool? Procedures { get; init; }

    /// <summary>A procedure up to this many tokens comes whole; a longer one as the hit's step and the steps around it (default 1500).</summary>
    public int? ProcedureTokens { get; init; }

    /// <summary>Steps on each side of the hit's step, when they fit (default 1).</summary>
    public int? ProcedureWindow { get; init; }

    /// <summary>At most this many results from one procedure while other results are left (default 3).</summary>
    public int? ProcedureHits { get; init; }

    /// <summary>Only these documents: document numbers ("DOC-12", needs the spots file) or parts of file paths.</summary>
    public IReadOnlyList<string>? Documents { get; init; }
}

/// <summary>The procedure a result belongs to (from the spots file). Level says what was returned:
/// "procedure" (all of it), "steps" (From..To), "substeps" (From..To inside Step) or "chunks" (part of Step).</summary>
public sealed record RetrievalProcedure(long Id, string Kind, string? Title, string? Label, int Steps, string? From, string? To,
                                        bool Whole, string Level, string? Step);

/// <summary>One line of a long procedure's outline; Shown marks the steps in the result.</summary>
public sealed record RetrievalOutlineStep(string? Label, string Title, bool Shown);

/// <summary>A reference in a result's text: another step ("section") or another document ("document"; File is null when it isn't in the knowledge base).</summary>
public sealed record RetrievalReference(string Kind, string Ref, string? Title, string? File);

public sealed record RetrievalHit(string Tag, long ChunkId, string Kind, double Score, double? RerankScore, string Text,
                                  string File, string FileType, int? Page, int? PageEnd, string? Sheet, string? Range,
                                  string? Lines, string? HeadingPath, string? Symbol, string? TableName, string? AssetPath)
{
    /// <summary>The document's number and revision, from the spots file.</summary>
    public string? Document { get; init; }
    public string? Revision { get; init; }

    /// <summary>The procedure this result was expanded to, when it is in one.</summary>
    public RetrievalProcedure? Procedure { get; init; }

    public IReadOnlyList<RetrievalReference> Related { get; init; } = [];

    /// <summary>For the first result from a long procedure: its steps around this passage (empty otherwise).</summary>
    public IReadOnlyList<RetrievalOutlineStep> Outline { get; init; } = [];

    /// <summary>When the result is only part of a procedure: the request that reads on from it. Pass it to
    /// <see cref="ShammyRetriever.ProcedureSteps(ProcedurePosition, int?)"/>. Null when there is nothing more.</summary>
    public ProcedurePosition? More { get; init; }
}

/// <summary>Where to continue reading a procedure: a step label, and inside a step too long for one page,
/// the last chunk already read.</summary>
public sealed record ProcedurePosition(long Procedure, string? From, long? AfterChunk);

/// <summary>One page of a procedure, from <see cref="ShammyRetriever.ProcedureSteps(long, string?, long?, int?)"/>.</summary>
public sealed class ProcedurePage
{
    /// <summary>The whole response as JSON, exactly as the engine wrote it.</summary>
    public required string Json { get; init; }

    /// <summary>The steps on this page, with a heading per step.</summary>
    public required string Text { get; init; }

    /// <summary>The compact text for an agent, ending with the call that reads on (or "the procedure ends here").</summary>
    public required string Markdown { get; init; }

    public RetrievalProcedure? Procedure { get; init; }

    /// <summary>The request for the next page; null at the end of the procedure.</summary>
    public ProcedurePosition? Next { get; init; }
}

public sealed record RetrievalTable(string TableName, long ChunkId, double Score, string Summary, string Card);

public sealed class RetrievalResult
{
    /// <summary>The whole response as JSON, exactly as the engine wrote it.</summary>
    public required string Json { get; init; }

    public required IReadOnlyList<RetrievalHit> Results { get; init; }

    /// <summary>Tables whose description ranked; the agent can query them with SQL.</summary>
    public required IReadOnlyList<RetrievalTable> Tables { get; init; }

    /// <summary>The compact text form for an agent (source line, then passage), when asked for.</summary>
    public string? Markdown { get; init; }

    /// <summary>True when the results are in the reranker's order; false when reranking was asked for but failed
    /// (the results are then in fused order and <see cref="RerankError"/> says why); null when it wasn't asked for.</summary>
    public bool? Reranked { get; init; }

    public string? RerankError { get; init; }
}

/// <summary>
/// The whole retrieval path over a RAG Workbench workspace or unpacked pack folder: vector and keyword
/// search, rank fusion, optional reranking, neighbour expansion and citations. This is the same code
/// the Workbench runs, so the agent gets what the Workbench shows. Works for <c>ragpack/1</c>
/// (sqlite-vec) and <c>ragpack/2</c> (Shammy) alike. Safe to use from many threads.
/// </summary>
public sealed class ShammyRetriever : IDisposable
{
    private readonly RetrieverHandle _handle;

    /// <summary>"shammy" or "sqlite-vec".</summary>
    public string VectorStore { get; }

    /// <summary>The size the query vector must have.</summary>
    public int Dimensions { get; }

    /// <summary>The embedding model the knowledge base was built with; embed queries with the same one.</summary>
    public string? Model { get; }

    /// <summary>The instruction to prefix queries with before embedding, if the knowledge base uses one.</summary>
    public string? QueryInstruction { get; }

    public long Chunks { get; }

    private ShammyRetriever(RetrieverHandle handle)
    {
        _handle = handle;
        Check(RagNative.shammy_rag_info(handle, out var ptr));
        using var doc = JsonDocument.Parse(Take(ptr));
        var root = doc.RootElement;
        VectorStore = root.GetProperty("vector_store").GetString() ?? "";
        Dimensions = root.GetProperty("dims").GetInt32();
        Model = root.GetProperty("model").GetString();
        QueryInstruction = root.GetProperty("query_instruction").GetString();
        Chunks = root.GetProperty("chunks").GetInt64();
        Procedures = root.TryGetProperty("procedures", out var pr) && pr.ValueKind == JsonValueKind.Number ? pr.GetInt64() : null;
    }

    /// <summary>Procedures in the knowledge base's spots file; null when it has no current one (results then carry no structure).</summary>
    public long? Procedures { get; }

    /// <summary>Open the folder that holds <c>rag.db</c> (a workspace, or a pack after unpacking).</summary>
    public static ShammyRetriever Open(string folder)
    {
        ArgumentNullException.ThrowIfNull(folder);
        Check(RagNative.shammy_rag_open(folder, out var handle));
        try
        {
            return new ShammyRetriever(handle);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>Search with the embedded question; returns typed results plus the raw JSON and Markdown.</summary>
    public RetrievalResult Search(string query, ReadOnlySpan<float> vector, RetrievalOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(query);
        options ??= new RetrievalOptions();
        var request = new Dictionary<string, object?> { ["query"] = query, ["markdown"] = options.Markdown };
        void Set(string key, object? value) { if (value is not null) request[key] = value; }
        Set("k", options.K);
        Set("file_type", options.FileType);
        Set("expand", options.Expand);
        Set("vec_k", options.VectorCandidates);
        Set("fts_k", options.KeywordCandidates);
        Set("neighbors", options.Neighbors);
        Set("max_result_tokens", options.MaxResultTokens);
        Set("threads", options.Threads);
        Set("procedures", options.Procedures);
        Set("procedure_tokens", options.ProcedureTokens);
        Set("procedure_window", options.ProcedureWindow);
        Set("procedure_hits", options.ProcedureHits);
        if (options.Documents is { Count: > 0 } docs) request["documents"] = docs;
        if (options.Rerank is { } r)
        {
            var rerank = new Dictionary<string, object?> { ["enabled"] = r.Enabled, ["url"] = r.Url, ["model"] = r.Model };
            if (r.Candidates is not null) rerank["candidates"] = r.Candidates;
            if (r.TimeoutMs is not null) rerank["timeout_ms"] = r.TimeoutMs;
            request["rerank"] = rerank;
        }
        string json = SearchJson(JsonSerializer.Serialize(request), vector);

        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        static string? Str(JsonElement e, string name) =>
            e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.String ? v.GetString() : null;
        static int? Int(JsonElement e, string name) =>
            e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetInt32() : null;
        static double? Num(JsonElement e, string name) =>
            e.TryGetProperty(name, out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;

        var hits = new List<RetrievalHit>();
        foreach (var h in root.GetProperty("results").EnumerateArray())
        {
            var s = h.GetProperty("source");
            RetrievalProcedure? procedure = null;
            if (h.TryGetProperty("procedure", out var p) && p.ValueKind == JsonValueKind.Object)
                procedure = new RetrievalProcedure(p.GetProperty("id").GetInt64(), Str(p, "kind") ?? "", Str(p, "title"), Str(p, "label"),
                                                   Int(p, "steps") ?? 0, Str(p, "from"), Str(p, "to"),
                                                   p.TryGetProperty("whole", out var w) && w.ValueKind == JsonValueKind.True,
                                                   Str(p, "level") ?? "", Str(p, "step"));
            var outline = new List<RetrievalOutlineStep>();
            if (h.TryGetProperty("outline", out var ol) && ol.TryGetProperty("steps", out var ols) && ols.ValueKind == JsonValueKind.Array)
                foreach (var x in ols.EnumerateArray())
                    outline.Add(new RetrievalOutlineStep(Str(x, "label"), Str(x, "title") ?? "",
                                                         x.TryGetProperty("shown", out var sh) && sh.ValueKind == JsonValueKind.True));
            var related = new List<RetrievalReference>();
            if (h.TryGetProperty("related", out var rel) && rel.ValueKind == JsonValueKind.Array)
                foreach (var x in rel.EnumerateArray())
                    related.Add(new RetrievalReference(Str(x, "kind") ?? "", Str(x, "ref") ?? "", Str(x, "title"), Str(x, "file")));
            hits.Add(new RetrievalHit(Str(h, "tag") ?? "", h.GetProperty("chunk_id").GetInt64(), Str(h, "kind") ?? "",
                                      Num(h, "score") ?? 0, Num(h, "rerank_score"), Str(h, "text") ?? "", Str(s, "file") ?? "",
                                      Str(s, "file_type") ?? "", Int(s, "page"), Int(s, "page_end"), Str(s, "sheet"),
                                      Str(s, "range"), Str(s, "lines"), Str(s, "heading_path"), Str(s, "symbol"),
                                      Str(s, "table_name"), Str(s, "asset_path"))
            {
                Document = Str(s, "document"), Revision = Str(s, "revision"), Procedure = procedure, Related = related,
                Outline = outline, More = Position(h, "more"),
            });
        }
        var tables = new List<RetrievalTable>();
        foreach (var t in root.GetProperty("tables").EnumerateArray())
            tables.Add(new RetrievalTable(Str(t, "table_name") ?? "", t.GetProperty("chunk_id").GetInt64(), Num(t, "score") ?? 0,
                                          Str(t, "summary") ?? "", Str(t, "card") ?? ""));
        bool? reranked = null;
        string? rerankError = null;
        if (root.TryGetProperty("rerank", out var rr))
        {
            reranked = rr.GetProperty("applied").GetBoolean();
            rerankError = Str(rr, "error");
        }
        return new RetrievalResult { Json = json, Results = hits, Tables = tables, Markdown = Str(root, "markdown"),
                                     Reranked = reranked, RerankError = rerankError };
    }

    private static ProcedurePosition? Position(JsonElement e, string name)
    {
        if (!e.TryGetProperty(name, out var m) || m.ValueKind != JsonValueKind.Object) return null;
        return new ProcedurePosition(m.GetProperty("procedure").GetInt64(),
                                     m.TryGetProperty("from", out var f) && f.ValueKind == JsonValueKind.String ? f.GetString() : null,
                                     m.TryGetProperty("after_chunk", out var c) && c.ValueKind == JsonValueKind.Number ? c.GetInt64() : null);
    }

    /// <summary>Read on through a procedure (needs the knowledge base's spots file): its steps from
    /// <paramref name="from"/> (the first step when null), as many as fit in <paramref name="maxTokens"/> (4000).
    /// A step longer than that is paged by its chunks; <paramref name="afterChunk"/> continues inside it.</summary>
    public ProcedurePage ProcedureSteps(long procedure, string? from = null, long? afterChunk = null, int? maxTokens = null)
    {
        var request = new Dictionary<string, object?> { ["procedure"] = procedure, ["markdown"] = true };
        if (from is not null) request["from"] = from;
        if (afterChunk is not null) request["after_chunk"] = afterChunk;
        if (maxTokens is not null) request["max_tokens"] = maxTokens;
        string json = ProcedureJson(JsonSerializer.Serialize(request));
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        RetrievalProcedure? info = null;
        if (root.TryGetProperty("procedure", out var p) && p.ValueKind == JsonValueKind.Object)
            info = new RetrievalProcedure(p.GetProperty("id").GetInt64(), p.GetProperty("kind").GetString() ?? "",
                                          p.TryGetProperty("title", out var t) && t.ValueKind == JsonValueKind.String ? t.GetString() : null,
                                          p.TryGetProperty("label", out var l) && l.ValueKind == JsonValueKind.String ? l.GetString() : null,
                                          p.GetProperty("steps").GetInt32(),
                                          p.TryGetProperty("from", out var fr) && fr.ValueKind == JsonValueKind.String ? fr.GetString() : null,
                                          p.TryGetProperty("to", out var to) && to.ValueKind == JsonValueKind.String ? to.GetString() : null,
                                          p.TryGetProperty("whole", out var w) && w.ValueKind == JsonValueKind.True,
                                          p.TryGetProperty("level", out var lv) ? lv.GetString() ?? "" : "",
                                          p.TryGetProperty("step", out var st) && st.ValueKind == JsonValueKind.String ? st.GetString() : null);
        return new ProcedurePage
        {
            Json = json, Procedure = info, Next = Position(root, "next"),
            Text = root.TryGetProperty("text", out var tx) ? tx.GetString() ?? "" : "",
            Markdown = root.TryGetProperty("markdown", out var md) ? md.GetString() ?? "" : "",
        };
    }

    /// <summary>Continue from a search result's <see cref="RetrievalHit.More"/> or a page's <see cref="ProcedurePage.Next"/>.</summary>
    public ProcedurePage ProcedureSteps(ProcedurePosition position, int? maxTokens = null)
        => ProcedureSteps(position.Procedure, position.From, position.AfterChunk, maxTokens);

    /// <summary>Read on through a procedure with a raw JSON request (fields in shammy.h); returns the raw JSON page.</summary>
    public string ProcedureJson(string requestJson)
    {
        ArgumentNullException.ThrowIfNull(requestJson);
        Check(RagNative.shammy_rag_procedure(_handle, requestJson, out var ptr));
        return Take(ptr);
    }

    /// <summary>Search with a raw JSON request (the fields are listed in shammy.h) and get the raw JSON response.</summary>
    public unsafe string SearchJson(string requestJson, ReadOnlySpan<float> vector)
    {
        ArgumentNullException.ThrowIfNull(requestJson);
        if (vector.IsEmpty) throw new ArgumentException("The query vector is empty.", nameof(vector));
        int code;
        IntPtr ptr;
        fixed (float* v = vector) code = RagNative.shammy_rag_search(_handle, requestJson, v, (nuint)vector.Length, out ptr);
        Check(code);
        return Take(ptr);
    }

    public void Dispose() => _handle.Dispose();

    private static string Take(IntPtr ptr)
    {
        try
        {
            return Marshal.PtrToStringUTF8(ptr) ?? "";
        }
        finally
        {
            RagNative.shammy_string_free(ptr);
        }
    }

    private static void Check(int code)
    {
        if (code == 0) return;
        throw new ShammyException(code, Marshal.PtrToStringUTF8(Native.shammy_last_error()) ?? $"shammy error {code}");
    }
}

internal sealed class RetrieverHandle : SafeHandle
{
    public RetrieverHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        RagNative.shammy_rag_close(handle);
        return true;
    }
}

internal static unsafe class RagNative
{
    private const string Lib = "shammy";

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_rag_open([MarshalAs(UnmanagedType.LPUTF8Str)] string folder, out RetrieverHandle retriever);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void shammy_rag_close(IntPtr retriever);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_rag_info(RetrieverHandle retriever, out IntPtr json);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_rag_search(RetrieverHandle retriever, [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
                                                 float* vector, nuint vectorLen, out IntPtr json);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_rag_procedure(RetrieverHandle retriever, [MarshalAs(UnmanagedType.LPUTF8Str)] string requestJson,
                                                    out IntPtr json);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void shammy_string_free(IntPtr s);
}
