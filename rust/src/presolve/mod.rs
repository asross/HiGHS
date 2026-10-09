//! Presolve. Ported so far: the undo side of the postsolve stack
//! (postsolve.rs); HPresolve and the recording of reductions stay C++.

pub(crate) mod ffi;
pub mod hpresolve;
pub mod postsolve;
pub mod symmetry;
pub(crate) mod symmetry_ffi;
