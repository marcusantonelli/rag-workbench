//! Python bindings. Vectors and id lists cross as little-endian bytes (float32, u64, u32) so no
//! NumPy build dependency is needed; `shammy/__init__.py` wraps this with NumPy arrays.

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use shammy_core as core;

fn err(e: core::Error) -> PyErr {
    match e {
        core::Error::Io(e) => PyOSError::new_err(e.to_string()),
        other => PyValueError::new_err(other.to_string()),
    }
}

fn floats(b: &[u8]) -> PyResult<Vec<f32>> {
    if b.len() % 4 != 0 {
        return Err(PyValueError::new_err("vector bytes are not a whole number of float32 values"));
    }
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
}

fn closed() -> PyErr {
    PyValueError::new_err("this file has been closed")
}

#[pyclass]
struct Writer {
    inner: Option<core::Writer>,
}

#[pymethods]
impl Writer {
    #[staticmethod]
    #[pyo3(signature = (path, dims, lossless = true, sketch_dims = None, model = "", query_instruction = ""))]
    fn create(path: &str, dims: u32, lossless: bool, sketch_dims: Option<u32>, model: &str, query_instruction: &str) -> PyResult<Self> {
        let mut schema = core::Schema::new(dims);
        schema.lossless = lossless;
        if let Some(s) = sketch_dims {
            schema.sketch_dims = s;
        }
        schema.model = model.to_string();
        schema.query_instruction = query_instruction.to_string();
        Ok(Self { inner: Some(core::Writer::create(path, &schema).map_err(err)?) })
    }

    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        Ok(Self { inner: Some(core::Writer::open(path).map_err(err)?) })
    }

    /// Add rows: ids as u64 bytes, doc_ids as u32 bytes, one file type per row (or one for all),
    /// vectors as float32 bytes, row after row.
    fn add(&mut self, py: Python<'_>, ids: &[u8], doc_ids: &[u8], file_types: Vec<String>, vectors: &[u8]) -> PyResult<usize> {
        let w = self.inner.as_mut().ok_or_else(closed)?;
        let dims = w.header().dims as usize;
        let n = ids.len() / 8;
        if ids.len() % 8 != 0 || doc_ids.len() != n * 4 || vectors.len() != n * dims * 4 || !(file_types.len() == n || file_types.len() == 1) {
            return Err(PyValueError::new_err(format!(
                "{n} ids need {n} doc ids, {n} (or 1) file types and {n} x {dims} float32 values")));
        }
        let all = floats(vectors)?;
        py.allow_threads(|| {
            for i in 0..n {
                let id = u64::from_le_bytes(ids[i * 8..i * 8 + 8].try_into().unwrap());
                let doc = u32::from_le_bytes(doc_ids[i * 4..i * 4 + 4].try_into().unwrap());
                let ft = &file_types[if file_types.len() == 1 { 0 } else { i }];
                w.add(id, doc, ft, &all[i * dims..(i + 1) * dims])?;
            }
            Ok(n)
        }).map_err(err)
    }

    fn delete_docs(&mut self, doc_ids: Vec<u32>) -> PyResult<u64> {
        self.inner.as_mut().ok_or_else(closed)?.delete_docs(&doc_ids).map_err(err)
    }

    /// Make everything since the last commit durable and visible to readers.
    fn commit(&mut self, py: Python<'_>) -> PyResult<u64> {
        let w = self.inner.as_mut().ok_or_else(closed)?;
        py.allow_threads(|| w.commit()).map_err(err)
    }

    fn close(&mut self) {
        self.inner = None;
    }
}

#[pyclass]
struct Reader {
    inner: Option<core::Reader>,
}

#[pymethods]
impl Reader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Ok(Self { inner: Some(core::Reader::open(path).map_err(err)?) })
    }

    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let i = self.inner.as_ref().ok_or_else(closed)?.info();
        let d = PyDict::new(py);
        d.set_item("dims", i.dims)?;
        d.set_item("sketch_dims", i.sketch_dims)?;
        d.set_item("lossless", i.lossless)?;
        d.set_item("generation", i.generation)?;
        d.set_item("rows", i.rows)?;
        d.set_item("live_rows", i.live_rows)?;
        d.set_item("segments", i.segments)?;
        d.set_item("model", i.model)?;
        d.set_item("file_types", i.file_types)?;
        d.set_item("file_bytes", i.file_bytes)?;
        Ok(d)
    }

    /// Returns (id, doc_id, score) triples, best first.
    #[pyo3(signature = (query, k = 10, file_types = None, doc_ids = None, exact = false, threads = 0, stage1 = 0, stage2 = 0, finish = 0))]
    #[allow(clippy::too_many_arguments)]
    fn search(&self, py: Python<'_>, query: &[u8], k: usize, file_types: Option<Vec<String>>, doc_ids: Option<Vec<u32>>,
              exact: bool, threads: usize, stage1: usize, stage2: usize, finish: usize) -> PyResult<Vec<(u64, u32, f32)>> {
        let r = self.inner.as_ref().ok_or_else(closed)?;
        let q = floats(query)?;
        let filter = core::Filter { file_types, doc_ids };
        let params = if exact {
            core::Params { stage1: usize::MAX, stage2: usize::MAX, finish: usize::MAX, threads }
        } else {
            core::Params { stage1, stage2, finish, threads }
        };
        if exact && !r.info().lossless {
            return Err(PyValueError::new_err("this file has no lossless tier to search exactly"));
        }
        let hits = py.allow_threads(|| r.search(&q, k, &filter, &params)).map_err(err)?;
        Ok(hits.into_iter().map(|h| (h.id, h.doc_id, h.score)).collect())
    }

    /// Stored vectors (float32 bytes) for these ids; None where an id isn't stored.
    fn vectors<'py>(&self, py: Python<'py>, ids: Vec<u64>) -> PyResult<Vec<Option<Bound<'py, PyBytes>>>> {
        let r = self.inner.as_ref().ok_or_else(closed)?;
        let found = py.allow_threads(|| r.vectors(&ids));
        Ok(found.into_iter().map(|v| v.map(|v| {
            let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
            PyBytes::new(py, &bytes)
        })).collect())
    }

    /// (ids as u64 bytes, doc ids as u32 bytes) of every live row, in file order.
    fn live<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyBytes>, Bound<'py, PyBytes>)> {
        let r = self.inner.as_ref().ok_or_else(closed)?;
        let ids: Vec<u8> = r.live_ids().iter().flat_map(|x| x.to_le_bytes()).collect();
        let docs: Vec<u8> = r.live_docs().iter().flat_map(|x| x.to_le_bytes()).collect();
        Ok((PyBytes::new(py, &ids), PyBytes::new(py, &docs)))
    }

    fn close(&mut self) {
        self.inner = None;
    }
}

/// Check every checksum in a file. Returns a dict with `ok` and `problems`.
#[pyfunction]
fn verify<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let rep = py.allow_threads(|| core::verify(path)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("ok", rep.ok)?;
    d.set_item("generation", rep.generation)?;
    d.set_item("rows", rep.rows)?;
    d.set_item("live_rows", rep.live_rows)?;
    d.set_item("sections_checked", rep.sections_checked)?;
    d.set_item("uncommitted_bytes", rep.uncommitted_bytes)?;
    d.set_item("problems", rep.problems)?;
    Ok(d)
}

/// Copy the live rows of `src` into a new file `dst`.
#[pyfunction]
fn compact<'py>(py: Python<'py>, src: &str, dst: &str) -> PyResult<Bound<'py, PyDict>> {
    let c = py.allow_threads(|| core::compact(src, dst)).map_err(err)?;
    let d = PyDict::new(py);
    d.set_item("rows_before", c.rows_before)?;
    d.set_item("rows_after", c.rows_after)?;
    d.set_item("bytes_before", c.bytes_before)?;
    d.set_item("bytes_after", c.bytes_after)?;
    Ok(d)
}

#[pyfunction]
fn default_threads() -> usize {
    core::default_threads()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Writer>()?;
    m.add_class::<Reader>()?;
    m.add_function(wrap_pyfunction!(verify, m)?)?;
    m.add_function(wrap_pyfunction!(compact, m)?)?;
    m.add_function(wrap_pyfunction!(default_threads, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
