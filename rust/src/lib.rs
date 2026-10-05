//! Rust ports of HiGHS kernels, called from C++ through `extern "C"` shims.
//! Ported so far: the LU factor HFactor (factor.rs), the double-precision
//! PRICE kernels of HighsSparseMatrix, the dual simplex CHUZC HEkkDualRow
//! (simplex/dual_row.rs), its CHUZR HEkkDualRHS (simplex/dual_rhs.rs), the
//! numerical kernels of HEkk on a view of its data (simplex/ekk.rs), the
//! primal simplex solver HEkkPrimal (simplex/primal.rs), the
//! MPS and LP readers, and the utilities
//! (hashing, random numbers, double-double arithmetic).

pub mod io;
pub mod matrix;
pub mod simplex;
pub mod util;

pub mod basiclu;
pub mod factor;
mod ffi;
pub mod hvector;
