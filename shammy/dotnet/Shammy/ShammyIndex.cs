using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

namespace Shammy;

/// <summary>One search result: the chunk id, its document id and the cosine similarity.</summary>
public readonly record struct ShammyHit(ulong Id, uint DocId, float Score);

/// <summary>Optional settings for one search. Leave everything unset for the defaults.</summary>
public sealed class ShammySearchOptions
{
    /// <summary>Only rows whose file type is one of these (for example "pdf").</summary>
    public IReadOnlyList<string>? FileTypes { get; init; }

    /// <summary>Only rows of these documents.</summary>
    public IReadOnlyList<uint>? DocIds { get; init; }

    /// <summary>Worker threads; 0 means two fewer than the machine's cores (at least 1).</summary>
    public int Threads { get; init; }

    /// <summary>Score every row in full. Slow; for checking results. Needs a lossless file.</summary>
    public bool Exact { get; init; }

    /// <summary>Rows kept by the first scan; 0 means 1% of the rows, at least 4096.</summary>
    public long Stage1 { get; init; }

    /// <summary>Rows kept by the 1-bit rerank; 0 means 256.</summary>
    public long Stage2 { get; init; }

    /// <summary>Rows rescored exactly at the end; 0 means 50.</summary>
    public long Finish { get; init; }
}

/// <summary>What <see cref="ShammyIndex.Verify"/> found.</summary>
public readonly record struct ShammyReport(bool Ok, int Problems, string? FirstProblem, ulong Generation, long Rows,
                                           long LiveRows, long UncommittedBytes);

public sealed class ShammyException : Exception
{
    /// <summary>The native error code (negative): -1 argument, -2 I/O, -3 file format, -4 internal.</summary>
    public int Code { get; }

    public ShammyException(int code, string message) : base(message) => Code = code;
}

/// <summary>
/// A <c>.shammy</c> vector file, opened read-only. Safe to search from many threads at once.
/// It shows the file as of its last commit when opened; open a new one to see later commits.
/// </summary>
public sealed class ShammyIndex : IDisposable
{
    private readonly ReaderHandle _handle;

    /// <summary>Stored values per vector. Queries may be shorter (they are padded with zeros).</summary>
    public int Dimensions { get; }

    /// <summary>Rows that can be found (deleted rows are not counted).</summary>
    public long Count { get; }

    /// <summary>True when the file keeps float16 copies: results are rescored exactly and vectors can be read back.</summary>
    public bool Lossless { get; }

    /// <summary>The embedding model the vectors were made with.</summary>
    public string Model { get; }

    /// <summary>Commits made to the file so far.</summary>
    public ulong Generation { get; }

    /// <summary>XXH3-64 of the query instruction the vectors were built with (0 when none).</summary>
    public ulong InstructionHash { get; }

    /// <summary>The file types present, usable in <see cref="ShammySearchOptions.FileTypes"/>.</summary>
    public IReadOnlyList<string> FileTypes { get; }

    public long FileBytes { get; }

    private ShammyIndex(ReaderHandle handle)
    {
        _handle = handle;
        Check(Native.shammy_info(handle, out var info));
        Dimensions = checked((int)info.Dims);
        Count = checked((long)info.LiveRows);
        Lossless = info.Lossless != 0;
        Generation = info.Generation;
        InstructionHash = info.InstructionHash;
        FileBytes = checked((long)info.FileBytes);
        unsafe
        {
            Model = Marshal.PtrToStringUTF8((IntPtr)info.Model) ?? "";
            var types = new string[info.FileTypeCount];
            byte* buf = stackalloc byte[256];
            for (uint i = 0; i < info.FileTypeCount; i++)
            {
                Check(Native.shammy_file_type(handle, i, buf, 256));
                types[i] = Marshal.PtrToStringUTF8((IntPtr)buf) ?? "";
            }
            FileTypes = types;
        }
    }

    public static ShammyIndex Open(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        Check(Native.shammy_open(path, out var handle));
        try
        {
            return new ShammyIndex(handle);
        }
        catch
        {
            handle.Dispose();
            throw;
        }
    }

    /// <summary>The <paramref name="k"/> most similar rows, best first.</summary>
    public unsafe ShammyHit[] Search(ReadOnlySpan<float> query, int k, ShammySearchOptions? options = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(k);
        if (query.IsEmpty || query.Length > Dimensions)
            throw new ArgumentException($"The query has {query.Length} values; this file stores {Dimensions}.", nameof(query));
        if (k == 0) return Array.Empty<ShammyHit>();

        var p = new Native.Params();
        IntPtr[] typePtrs = Array.Empty<IntPtr>();
        uint[] docs = Array.Empty<uint>();
        if (options is not null)
        {
            p.Threads = checked((uint)options.Threads);
            p.Exact = options.Exact ? 1u : 0u;
            p.Stage1 = checked((ulong)options.Stage1);
            p.Stage2 = checked((ulong)options.Stage2);
            p.Finish = checked((ulong)options.Finish);
            if (options.DocIds is { Count: > 0 } d) docs = [.. d];
            if (options.FileTypes is { Count: > 0 } t)
            {
                typePtrs = new IntPtr[t.Count];
                for (int i = 0; i < t.Count; i++) typePtrs[i] = Marshal.StringToCoTaskMemUTF8(t[i]);
            }
        }
        var hits = new Native.Hit[k];
        try
        {
            nuint count;
            int code;
            fixed (float* q = query)
            fixed (IntPtr* types = typePtrs)
            fixed (uint* docPtr = docs)
            fixed (Native.Hit* outHits = hits)
            {
                code = Native.shammy_search(_handle, q, (nuint)query.Length, (nuint)k, types, (nuint)typePtrs.Length,
                                            docPtr, (nuint)docs.Length, &p, outHits, &count);
            }
            Check(code);
            var result = new ShammyHit[(int)count];
            for (int i = 0; i < result.Length; i++) result[i] = new ShammyHit(hits[i].Id, hits[i].DocId, hits[i].Score);
            return result;
        }
        finally
        {
            foreach (var ptr in typePtrs) Marshal.FreeCoTaskMem(ptr);
        }
    }

    /// <summary>The stored vector of a chunk (unit length, float16 precision), or null when it isn't stored.</summary>
    public unsafe float[]? GetVector(ulong id)
    {
        var v = new float[Dimensions];
        uint found;
        int code;
        fixed (float* p = v) code = Native.shammy_vector(_handle, id, p, (nuint)v.Length, &found);
        Check(code);
        return found != 0 ? v : null;
    }

    /// <summary>Check every checksum in a file without opening it for search.</summary>
    public static ShammyReport Verify(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        Check(Native.shammy_verify(path, out var r));
        string? first = r.Problems > 0 ? Marshal.PtrToStringUTF8(Native.shammy_last_error()) : null;
        return new ShammyReport(r.Ok != 0, (int)r.Problems, first, r.Generation, (long)r.Rows, (long)r.LiveRows, (long)r.UncommittedBytes);
    }

    /// <summary>The version of the native library in use.</summary>
    public static string NativeVersion => Marshal.PtrToStringUTF8(Native.shammy_version()) ?? "";

    /// <summary>The thread count a search uses when none is given.</summary>
    public static int DefaultThreads => (int)Native.shammy_default_threads();

    public void Dispose() => _handle.Dispose();

    private static void Check(int code)
    {
        if (code == 0) return;
        throw new ShammyException(code, Marshal.PtrToStringUTF8(Native.shammy_last_error()) ?? $"shammy error {code}");
    }
}

/// <summary>Keeps the native reader alive while any call is using it and closes it exactly once.</summary>
internal sealed class ReaderHandle : SafeHandle
{
    public ReaderHandle() : base(IntPtr.Zero, ownsHandle: true) { }

    public override bool IsInvalid => handle == IntPtr.Zero;

    protected override bool ReleaseHandle()
    {
        Native.shammy_close(handle);
        return true;
    }
}

internal static unsafe class Native
{
    private const string Lib = "shammy";   // shammy.dll, libshammy.so or libshammy.dylib

    [StructLayout(LayoutKind.Sequential)]
    internal struct Info
    {
        public uint Dims, SketchDims, Lossless, Segments, FileTypeCount, Reserved;
        public ulong Generation, Rows, LiveRows, FileBytes, InstructionHash;
        public fixed byte Model[128];
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct Hit
    {
        public ulong Id;
        public uint DocId;
        public float Score;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct Params
    {
        public uint Threads, Exact;
        public ulong Stage1, Stage2, Finish;
    }

    [StructLayout(LayoutKind.Sequential)]
    internal struct Report
    {
        public uint Ok, Problems;
        public ulong Generation, Rows, LiveRows, UncommittedBytes;
    }

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr shammy_version();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern IntPtr shammy_last_error();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern uint shammy_default_threads();

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_open([MarshalAs(UnmanagedType.LPUTF8Str)] string path, out ReaderHandle reader);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern void shammy_close(IntPtr reader);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_info(ReaderHandle reader, out Info info);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_file_type(ReaderHandle reader, uint index, byte* buf, nuint cap);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_search(ReaderHandle reader, float* query, nuint queryLen, nuint k,
                                             IntPtr* fileTypes, nuint fileTypeCount, uint* docIds, nuint docIdCount,
                                             Params* p, Hit* hits, nuint* count);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_vector(ReaderHandle reader, ulong id, float* dst, nuint dstLen, uint* found);

    [DllImport(Lib, CallingConvention = CallingConvention.Cdecl)]
    internal static extern int shammy_verify([MarshalAs(UnmanagedType.LPUTF8Str)] string path, out Report report);
}
