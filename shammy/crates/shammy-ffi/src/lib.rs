//! C interface for reading Shammy files. See include/shammy.h for the contract.
//!
//! Every function returns 0 on success or a negative `SHAMMY_E_*` code; the message for the last
//! failure on the calling thread is `shammy_last_error()`. A panic never crosses the boundary.
//! A handle can be used from many threads at once; close it once, when no call is running.

use std::cell::RefCell;
use std::ffi::{c_char, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};

use shammy_core as core;

pub const SHAMMY_OK: i32 = 0;
pub const SHAMMY_E_ARGUMENT: i32 = -1;
pub const SHAMMY_E_IO: i32 = -2;
pub const SHAMMY_E_FORMAT: i32 = -3;
pub const SHAMMY_E_INTERNAL: i32 = -4;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn fail(code: i32, message: impl Into<String>) -> i32 {
    let text = message.into().replace('\0', " ");
    LAST_ERROR.with(|e| *e.borrow_mut() = CString::new(text).unwrap_or_default());
    code
}

fn from_core(e: core::Error) -> i32 {
    match &e {
        core::Error::Io(_) => fail(SHAMMY_E_IO, e.to_string()),
        core::Error::Invalid(_) => fail(SHAMMY_E_ARGUMENT, e.to_string()),
        _ => fail(SHAMMY_E_FORMAT, e.to_string()),
    }
}

fn guard(f: impl FnOnce() -> i32) -> i32 {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|_| fail(SHAMMY_E_INTERNAL, "internal error (panic) in shammy"))
}

unsafe fn text<'a>(p: *const c_char, what: &str) -> Result<&'a str, i32> {
    if p.is_null() {
        return Err(fail(SHAMMY_E_ARGUMENT, format!("{what} is null")));
    }
    CStr::from_ptr(p).to_str().map_err(|_| fail(SHAMMY_E_ARGUMENT, format!("{what} is not UTF-8")))
}

/// Copy `s` into a caller buffer of `cap` bytes, always NUL-terminated (cut at a character boundary).
unsafe fn copy_text(s: &str, buf: *mut c_char, cap: usize) {
    if buf.is_null() || cap == 0 {
        return;
    }
    let mut n = s.len().min(cap - 1);
    while !s.is_char_boundary(n) {
        n -= 1;
    }
    std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, n);
    *buf.add(n) = 0;
}

pub struct ShammyReader {
    reader: core::Reader,
}

#[repr(C)]
pub struct ShammyInfo {
    pub dims: u32,
    pub sketch_dims: u32,
    pub lossless: u32,
    pub segments: u32,
    pub file_type_count: u32,
    pub reserved: u32,
    pub generation: u64,
    pub rows: u64,
    pub live_rows: u64,
    pub file_bytes: u64,
    pub instruction_hash: u64,
    pub model: [c_char; 128],
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct ShammyHit {
    pub id: u64,
    pub doc_id: u32,
    pub score: f32,
}

/// All zero means "defaults". `exact` scores every row in full (needs a lossless file).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct ShammyParams {
    pub threads: u32,
    pub exact: u32,
    pub stage1: u64,
    pub stage2: u64,
    pub finish: u64,
}

#[repr(C)]
pub struct ShammyReport {
    pub ok: u32,
    pub problems: u32,
    pub generation: u64,
    pub rows: u64,
    pub live_rows: u64,
    pub uncommitted_bytes: u64,
}

/// Version of this library, e.g. "0.1.0". Never null; do not free.
#[no_mangle]
pub extern "C" fn shammy_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

/// Message for the last failed call on this thread. Valid until the next failed call on this thread.
#[no_mangle]
pub extern "C" fn shammy_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// The default thread count for a search: two fewer than the machine's cores, at least 1.
#[no_mangle]
pub extern "C" fn shammy_default_threads() -> u32 {
    core::default_threads() as u32
}

/// Open a file read-only (UTF-8 path). The file is seen as of its last commit at this moment.
///
/// # Safety
/// `path` must be a NUL-terminated string and `out` a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn shammy_open(path: *const c_char, out: *mut *mut ShammyReader) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(SHAMMY_E_ARGUMENT, "out is null");
        }
        *out = std::ptr::null_mut();
        let path = match text(path, "path") {
            Ok(p) => p,
            Err(code) => return code,
        };
        match core::Reader::open(path) {
            Ok(reader) => {
                *out = Box::into_raw(Box::new(ShammyReader { reader }));
                SHAMMY_OK
            }
            Err(e) => from_core(e),
        }
    })
}

/// Close a handle from `shammy_open`. Null is ignored.
///
/// # Safety
/// `reader` must come from `shammy_open`, not be closed twice, and have no call in progress.
#[no_mangle]
pub unsafe extern "C" fn shammy_close(reader: *mut ShammyReader) {
    if !reader.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(reader))));
    }
}

/// # Safety
/// `reader` must be an open handle and `out` a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn shammy_info(reader: *const ShammyReader, out: *mut ShammyInfo) -> i32 {
    guard(|| {
        let (Some(r), false) = (reader.as_ref(), out.is_null()) else {
            return fail(SHAMMY_E_ARGUMENT, "reader or out is null");
        };
        let i = r.reader.info();
        let mut info = ShammyInfo {
            dims: i.dims, sketch_dims: i.sketch_dims, lossless: i.lossless as u32, segments: i.segments,
            file_type_count: i.file_types.len() as u32, reserved: 0, generation: i.generation, rows: i.rows,
            live_rows: i.live_rows, file_bytes: i.file_bytes, instruction_hash: i.instruction_hash, model: [0; 128],
        };
        copy_text(&i.model, info.model.as_mut_ptr(), info.model.len());
        out.write(info);
        SHAMMY_OK
    })
}

/// Name of file type number `index` (0 .. file_type_count-1), copied into `buf`.
///
/// # Safety
/// `reader` must be an open handle and `buf` point to `cap` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn shammy_file_type(reader: *const ShammyReader, index: u32, buf: *mut c_char, cap: usize) -> i32 {
    guard(|| {
        let Some(r) = reader.as_ref() else { return fail(SHAMMY_E_ARGUMENT, "reader is null") };
        match r.reader.info().file_types.get(index as usize) {
            Some(name) => {
                copy_text(name, buf, cap);
                SHAMMY_OK
            }
            None => fail(SHAMMY_E_ARGUMENT, format!("no file type number {index}")),
        }
    })
}

/// Find the `k` most similar rows, best first.
///
/// `query` holds `query_len` float32 values (shorter than the stored size is padded with zeros).
/// `file_types` (NUL-terminated names) and `doc_ids` restrict the search; pass null and 0 for no
/// restriction. `params` may be null for defaults. Up to `k` hits are written to `hits` and the
/// number written to `count`.
///
/// # Safety
/// Every pointer must be valid for the length given with it.
#[no_mangle]
pub unsafe extern "C" fn shammy_search(
    reader: *const ShammyReader, query: *const f32, query_len: usize, k: usize,
    file_types: *const *const c_char, file_type_count: usize, doc_ids: *const u32, doc_id_count: usize,
    params: *const ShammyParams, hits: *mut ShammyHit, count: *mut usize,
) -> i32 {
    guard(|| {
        let Some(r) = reader.as_ref() else { return fail(SHAMMY_E_ARGUMENT, "reader is null") };
        if query.is_null() || count.is_null() || (hits.is_null() && k > 0) {
            return fail(SHAMMY_E_ARGUMENT, "query, hits or count is null");
        }
        *count = 0;
        let dims = r.reader.info().dims as usize;
        if query_len == 0 || query_len > dims {
            return fail(SHAMMY_E_ARGUMENT, format!("the query has {query_len} values; this file stores {dims}"));
        }
        let mut q = vec![0f32; dims];
        q[..query_len].copy_from_slice(std::slice::from_raw_parts(query, query_len));
        let mut filter = core::Filter::default();
        if file_type_count > 0 {
            if file_types.is_null() {
                return fail(SHAMMY_E_ARGUMENT, "file_types is null");
            }
            let mut names = Vec::with_capacity(file_type_count);
            for p in std::slice::from_raw_parts(file_types, file_type_count) {
                match text(*p, "a file type") {
                    Ok(s) => names.push(s.to_string()),
                    Err(code) => return code,
                }
            }
            filter.file_types = Some(names);
        }
        if doc_id_count > 0 {
            if doc_ids.is_null() {
                return fail(SHAMMY_E_ARGUMENT, "doc_ids is null");
            }
            filter.doc_ids = Some(std::slice::from_raw_parts(doc_ids, doc_id_count).to_vec());
        }
        let p = params.as_ref().copied().unwrap_or_default();
        let params = if p.exact != 0 {
            if !r.reader.info().lossless {
                return fail(SHAMMY_E_ARGUMENT, "this file has no lossless tier to search exactly");
            }
            core::Params { stage1: usize::MAX, stage2: usize::MAX, finish: usize::MAX, threads: p.threads as usize }
        } else {
            core::Params { stage1: p.stage1 as usize, stage2: p.stage2 as usize, finish: p.finish as usize, threads: p.threads as usize }
        };
        match r.reader.search(&q, k, &filter, &params) {
            Ok(found) => {
                for (i, h) in found.iter().enumerate() {
                    hits.add(i).write(ShammyHit { id: h.id, doc_id: h.doc_id, score: h.score });
                }
                *count = found.len();
                SHAMMY_OK
            }
            Err(e) => from_core(e),
        }
    })
}

/// Copy the stored vector for chunk `id` into `out` (`dims` float32 values). `found` is set to 0
/// when the id isn't stored or the file has no lossless tier.
///
/// # Safety
/// `out` must hold `out_len` floats; `found` must be valid.
#[no_mangle]
pub unsafe extern "C" fn shammy_vector(reader: *const ShammyReader, id: u64, out: *mut f32, out_len: usize, found: *mut u32) -> i32 {
    guard(|| {
        let Some(r) = reader.as_ref() else { return fail(SHAMMY_E_ARGUMENT, "reader is null") };
        if out.is_null() || found.is_null() {
            return fail(SHAMMY_E_ARGUMENT, "out or found is null");
        }
        *found = 0;
        if let Some(v) = r.reader.vectors(&[id]).pop().flatten() {
            if out_len < v.len() {
                return fail(SHAMMY_E_ARGUMENT, format!("the buffer holds {out_len} values; a vector has {}", v.len()));
            }
            std::ptr::copy_nonoverlapping(v.as_ptr(), out, v.len());
            *found = 1;
        }
        SHAMMY_OK
    })
}

/// Check every checksum in a file. Returns 0 when the check ran; `out.ok` says whether the file is
/// whole. The first problem found, if any, is in `shammy_last_error()`.
///
/// # Safety
/// `path` must be a NUL-terminated string and `out` a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn shammy_verify(path: *const c_char, out: *mut ShammyReport) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(SHAMMY_E_ARGUMENT, "out is null");
        }
        let path = match text(path, "path") {
            Ok(p) => p,
            Err(code) => return code,
        };
        match core::verify(path) {
            Ok(rep) => {
                if let Some(first) = rep.problems.first() {
                    fail(0, first.clone());
                }
                out.write(ShammyReport { ok: rep.ok as u32, problems: rep.problems.len() as u32, generation: rep.generation,
                                         rows: rep.rows, live_rows: rep.live_rows, uncommitted_bytes: rep.uncommitted_bytes });
                SHAMMY_OK
            }
            Err(e) => from_core(e),
        }
    })
}

// ---------------------------------------------------------------------------------------------
// The retrieval path over a workspace or unpacked pack (shammy-rag): one call per search.

pub struct ShammyRetriever {
    inner: shammy_rag::Retriever,
}

fn from_rag(e: shammy_rag::Error) -> i32 {
    match e {
        shammy_rag::Error::Request(m) => fail(SHAMMY_E_ARGUMENT, m),
        shammy_rag::Error::Store(m) => fail(SHAMMY_E_IO, m),
    }
}

unsafe fn give_string(s: String, out: *mut *mut c_char) -> i32 {
    match CString::new(s) {
        Ok(c) => {
            *out = c.into_raw();
            SHAMMY_OK
        }
        Err(_) => fail(SHAMMY_E_INTERNAL, "the result contains a NUL character"),
    }
}

/// Open the folder of a workspace or unpacked pack (the one holding rag.db) for searching.
///
/// # Safety
/// `folder` must be a NUL-terminated string and `out` a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn shammy_rag_open(folder: *const c_char, out: *mut *mut ShammyRetriever) -> i32 {
    guard(|| {
        if out.is_null() {
            return fail(SHAMMY_E_ARGUMENT, "out is null");
        }
        *out = std::ptr::null_mut();
        let folder = match text(folder, "folder") {
            Ok(p) => p,
            Err(code) => return code,
        };
        match shammy_rag::Retriever::open(folder) {
            Ok(inner) => {
                *out = Box::into_raw(Box::new(ShammyRetriever { inner }));
                SHAMMY_OK
            }
            Err(e) => from_rag(e),
        }
    })
}

/// # Safety
/// `retriever` must come from `shammy_rag_open`, not be closed twice, and have no call in progress.
#[no_mangle]
pub unsafe extern "C" fn shammy_rag_close(retriever: *mut ShammyRetriever) {
    if !retriever.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(retriever))));
    }
}

/// JSON describing the knowledge base. Free the string with `shammy_string_free`.
///
/// # Safety
/// `retriever` must be an open handle and `out_json` a valid pointer.
#[no_mangle]
pub unsafe extern "C" fn shammy_rag_info(retriever: *const ShammyRetriever, out_json: *mut *mut c_char) -> i32 {
    guard(|| {
        let (Some(r), false) = (retriever.as_ref(), out_json.is_null()) else {
            return fail(SHAMMY_E_ARGUMENT, "retriever or out_json is null");
        };
        *out_json = std::ptr::null_mut();
        match r.inner.info() {
            Ok(info) => give_string(serde_json::to_string(&info).unwrap_or_default(), out_json),
            Err(e) => from_rag(e),
        }
    })
}

/// Run one search. `request_json` is a JSON object (only "query" is required; see shammy.h);
/// `vector` is the embedded query. The JSON response is returned in `out_json`; free it with
/// `shammy_string_free`.
///
/// # Safety
/// Every pointer must be valid; `vector` must hold `vector_len` floats.
#[no_mangle]
pub unsafe extern "C" fn shammy_rag_search(retriever: *const ShammyRetriever, request_json: *const c_char,
                                           vector: *const f32, vector_len: usize, out_json: *mut *mut c_char) -> i32 {
    guard(|| {
        let (Some(r), false) = (retriever.as_ref(), out_json.is_null()) else {
            return fail(SHAMMY_E_ARGUMENT, "retriever or out_json is null");
        };
        *out_json = std::ptr::null_mut();
        if vector.is_null() {
            return fail(SHAMMY_E_ARGUMENT, "vector is null");
        }
        let request = match text(request_json, "request_json") {
            Ok(t) => t,
            Err(code) => return code,
        };
        match r.inner.search_json(request, std::slice::from_raw_parts(vector, vector_len)) {
            Ok(json) => give_string(json, out_json),
            Err(e) => from_rag(e),
        }
    })
}

/// Free a string returned by this library. Null is ignored.
///
/// # Safety
/// `s` must come from this library and not be freed twice.
#[no_mangle]
pub unsafe extern "C" fn shammy_string_free(s: *mut c_char) {
    if !s.is_null() {
        let _ = catch_unwind(AssertUnwindSafe(|| drop(CString::from_raw(s))));
    }
}
