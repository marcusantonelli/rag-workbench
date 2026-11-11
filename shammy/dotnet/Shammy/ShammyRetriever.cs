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
}

public sealed record RetrievalHit(string Tag, long ChunkId, string Kind, double Score, double? RerankScore, string Text,
                                  string File, string FileType, int? Page, int? PageEnd, string? Sheet, string? Range,
                                  string? Lines, string? HeadingPath, string? Symbol, string? TableName, string? AssetPath);

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
    }

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
            hits.Add(new RetrievalHit(Str(h, "tag") ?? "", h.GetProperty("chunk_id").GetInt64(), Str(h, "kind") ?? "",
                                      Num(h, "score") ?? 0, Num(h, "rerank_score"), Str(h, "text") ?? "", Str(s, "file") ?? "",
                                      Str(s, "file_type") ?? "", Int(s, "page"), Int(s, "page_end"), Str(s, "sheet"),
                                      Str(s, "range"), Str(s, "lines"), Str(s, "heading_path"), Str(s, "symbol"),
                                      Str(s, "table_name"), Str(s, "asset_path")));
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
    internal static extern void shammy_string_free(IntPtr s);
}
