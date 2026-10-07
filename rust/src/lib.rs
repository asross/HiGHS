//! Rust ports of HiGHS kernels, called from C++ through `extern "C"` shims.
//! Ported so far: the LU factor HFactor (factor.rs), the double-precision
//! PRICE kernels of HighsSparseMatrix, the dual simplex CHUZC HEkkDualRow
//! (simplex/dual_row.rs), its CHUZR HEkkDualRHS (simplex/dual_rhs.rs), the
//! numerical kernels of HEkk on a view of its data (simplex/ekk.rs), the
//! serial dual and the primal simplex drivers (simplex/dual.rs, primal.rs),
//! BASICLU and the interior point solver IPX (basiclu/, ipx/), the PDLP
//! solver cuPDLP-C (pdlp/), the QP solver (qp/), the MPS and
//! LP readers, and the utilities
//! (hashing, random numbers, double-double arithmetic).

pub mod io;
pub mod lp_data;
pub mod matrix;
pub mod mip;
pub mod simplex;
pub mod util;

pub mod basiclu;
pub mod ipx;
pub mod pdlp;
pub mod presolve;
pub mod qp;
pub mod factor;
mod ffi;
pub mod hvector;
