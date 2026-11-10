//! Shammy: a single-file vector store built for one workload: read-mostly knowledge bases of
//! up to a few million high-dimensional, unit-length, Matryoshka-trained embeddings, searched
//! one query at a time with filters.
//!
//! There is no index to walk. Each vector is stored as three nested codes (a 1-bit sketch of
//! the leading dimensions, 1 bit for every dimension, 8 bits for every dimension) plus,
//! optionally, the float16 original. A search scans the sketch of every live row and reranks
//! a shrinking candidate set on the larger codes. The byte layout is in `spec/FORMAT.md`.

pub mod encode;
pub mod error;
pub mod format;
pub mod kernels;
pub mod reader;
pub mod rotate;
pub mod search;
pub mod verify;
pub mod writer;

pub use error::{Error, Result};
pub use reader::{Info, Reader};
pub use search::{Filter, Hit, Params, Trace};
pub use verify::{verify, Report};
pub use writer::{Schema, Writer};
