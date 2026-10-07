//! Presolve. Ported so far: the undo side of the postsolve stack
//! (postsolve.rs); HPresolve and the recording of reductions stay C++.

mod ffi;
pub mod hpresolve;
pub mod postsolve;
pub mod symmetry;
mod symmetry_ffi;
