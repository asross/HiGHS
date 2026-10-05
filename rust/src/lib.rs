//! Rust ports of HiGHS kernels, called from C++ through `extern "C"` shims.
//! Ported so far: the LU factor HFactor (factor.rs).

pub mod factor;
mod ffi;
pub mod hvector;
