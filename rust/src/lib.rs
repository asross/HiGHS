//! Rust ports of HiGHS kernels, called from C++ through `extern "C"` shims.
//! Ported so far: the LU factor HFactor (factor.rs), the double-precision
//! PRICE kernels of HighsSparseMatrix, the MPS and LP readers, and the
//! utilities (hashing, random numbers, double-double arithmetic).

pub mod io;
pub mod matrix;
pub mod util;

pub mod factor;
mod ffi;
pub mod hvector;
