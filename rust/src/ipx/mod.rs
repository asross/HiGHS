//! IPX (highs/ipm/ipx/): the interior point solver with basis
//! preconditioning and crossover. A line-by-line port of the C++ code: same
//! floating point operations in the same order, so that it runs
//! bit-identically. Clang contracts `a ± b*c` within one expression into a
//! fused multiply-add on arm64 (it emits `llvm.fmuladd` for exactly those
//! expressions); the port mirrors every such place with `mul_add` (the
//! inventory was taken from the IR of the C++ sources). The std::valarray
//! expressions (`lhs += alpha*step` and friends) are not contracted, since
//! the product and the sum live in different template functions; the port
//! keeps them as separate multiplications and additions.
//!
//! The C++ class `ipx::LpSolver` becomes a thin wrapper around the opaque
//! Rust handle of lp_solver.rs (see ffi.rs and highs/ipm/ipx/lp_solver_rs.cc).

// Faithful port: index loops as in the C++ code
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

mod basis;
mod control;
mod crossover;
pub mod ffi;
pub(crate) mod fmt;
mod guess_basis;
mod indexed_vector;
mod ipm;
mod iterate;
mod kkt;
mod lp_solver;
mod lu;
mod maxvolume;
mod model;
mod sparse_matrix;
mod sparse_utils;
mod starting_basis;
mod symbolic_invert;
pub(crate) mod utils;

#[cfg(test)]
mod tests;

pub use control::Hooks;
pub use lp_solver::LpSolver;

/// ipxint = HighsInt (32 bit)
pub type Int = i32;

/// std::valarray<double>
pub type Vector = Vec<f64>;

// A vector is treated sparse if it has no more than kHypersparseThreshold *
// dim nonzeros.
pub(crate) const HYPERSPARSE_THRESHOLD: f64 = 0.1;

// When LU factorization is used for rank detection, columns of the active
// submatrix whose maximum entry is <= kLuDependencyTol are removed
// immediately without choosing a pivot.
pub(crate) const LU_DEPENDENCY_TOL: f64 = 1e-3;

// A fresh LU factorization is considered unstable if
//   ||b-Bx|| / (||b||+||B||*||x||) > kLuStabilityThreshold,
// where x=B\b is computed from the LU factors, b has components +/- 1 that
// are chosen to make x large, and ||.|| is the 1-norm. An unstable
// factorization triggers tightening of the pivot tolerance and
// refactorization.
pub(crate) const LU_STABILITY_THRESHOLD: f64 = 1e-12;

// A Forrest-Tomlin LU update is declared numerically unstable if the
// relative error in the new diagonal entry of U is larger than
// kFtDiagErrorTol.
pub(crate) const FT_DIAG_ERROR_TOL: f64 = 1e-8;

// ipx_status.h
pub const STATUS_NOT_RUN: Int = 0;
pub const STATUS_SOLVED: Int = 1000;
pub const STATUS_STOPPED: Int = 1005;
pub const STATUS_NO_MODEL: Int = 1006;
pub const STATUS_OUT_OF_MEMORY: Int = 1003;
pub const STATUS_INTERNAL_ERROR: Int = 1004;
pub const STATUS_OPTIMAL: Int = 1;
pub const STATUS_IMPRECISE: Int = 2;
pub const STATUS_PRIMAL_INFEAS: Int = 3;
pub const STATUS_DUAL_INFEAS: Int = 4;
pub const STATUS_USER_INTERRUPT: Int = 5;
pub const STATUS_TIME_LIMIT: Int = 6;
pub const STATUS_ITER_LIMIT: Int = 7;
pub const STATUS_NO_PROGRESS: Int = 8;
pub const STATUS_FAILED: Int = 9;
pub const STATUS_DEBUG: Int = 10;
pub const ERROR_ARGUMENT_NULL: Int = 102;
pub const ERROR_INVALID_DIMENSION: Int = 103;
pub const ERROR_INVALID_MATRIX: Int = 104;
pub const ERROR_INVALID_VECTOR: Int = 105;
pub const ERROR_INVALID_BASIS: Int = 107;
pub const ERROR_CR_ITER_LIMIT: Int = 201;
pub const ERROR_CR_MATRIX_NOT_POSDEF: Int = 202;
pub const ERROR_CR_PRECOND_NOT_POSDEF: Int = 203;
pub const ERROR_CR_NO_PROGRESS: Int = 204;
pub const ERROR_CR_INF_OR_NAN: Int = 205;
pub const ERROR_BASIS_SINGULAR: Int = 301;
pub const ERROR_BASIS_ALMOST_SINGULAR: Int = 302;
pub const ERROR_BASIS_UPDATE_SINGULAR: Int = 303;
pub const ERROR_BASIS_REPAIR_OVERFLOW: Int = 304;
pub const ERROR_BASIS_REPAIR_SEARCH: Int = 305;
pub const ERROR_BASIS_TOO_ILL_CONDITIONED: Int = 306;
pub const ERROR_LAPACK_CHOL: Int = 401;
pub const ERROR_NOT_IMPLEMENTED: Int = 901;
pub const ERROR_USER_INTERRUPT: Int = 998;
pub const ERROR_TIME_INTERRUPT: Int = 999;
pub const BASIC: Int = 0;
pub const NONBASIC: Int = -1;
pub const NONBASIC_LB: Int = -1;
pub const NONBASIC_UB: Int = -2;
pub const SUPERBASIC: Int = -3;

/// struct ipx_info (ipx_info.h), zero-initialized like ipx::Info
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Info {
    pub status: Int,
    pub status_ipm: Int,
    pub status_crossover: Int,
    pub errflag: Int,
    pub num_var: Int,
    pub num_constr: Int,
    pub num_entries: Int,
    pub num_rows_solver: Int,
    pub num_cols_solver: Int,
    pub num_entries_solver: Int,
    pub dualized: Int,
    pub dense_cols: Int,
    pub centring_tried: Int,
    pub centring_success: Int,
    pub dependent_rows: Int,
    pub dependent_cols: Int,
    pub rows_inconsistent: Int,
    pub cols_inconsistent: Int,
    pub primal_dropped: Int,
    pub dual_dropped: Int,
    pub abs_presidual: f64,
    pub abs_dresidual: f64,
    pub rel_presidual: f64,
    pub rel_dresidual: f64,
    pub pobjval: f64,
    pub dobjval: f64,
    pub rel_objgap: f64,
    pub complementarity: f64,
    pub normx: f64,
    pub normy: f64,
    pub normz: f64,
    pub objval: f64,
    pub primal_infeas: f64,
    pub dual_infeas: f64,
    pub iter: Int,
    pub kktiter1: Int,
    pub kktiter2: Int,
    pub basis_repairs: Int,
    pub updates_start: Int,
    pub updates_ipm: Int,
    pub updates_crossover: Int,
    pub time_total: f64,
    pub time_ipm1: f64,
    pub time_ipm2: f64,
    pub time_starting_basis: f64,
    pub time_crossover: f64,
    pub time_kkt_factorize: f64,
    pub time_kkt_solve: f64,
    pub time_maxvol: f64,
    pub time_cr1: f64,
    pub time_cr1_aat: f64,
    pub time_cr1_pre: f64,
    pub time_cr2: f64,
    pub time_cr2_nnt: f64,
    pub time_cr2_b: f64,
    pub time_cr2_bt: f64,
    pub ftran_sparse: f64,
    pub btran_sparse: f64,
    pub time_ftran: f64,
    pub time_btran: f64,
    pub time_lu_invert: f64,
    pub time_lu_update: f64,
    pub mean_fill: f64,
    pub max_fill: f64,
    pub time_symb_invert: f64,
    pub maxvol_updates: Int,
    pub maxvol_skipped: Int,
    pub maxvol_passes: Int,
    pub tbl_nnz: Int,
    pub tbl_max: f64,
    pub frobnorm_squared: f64,
    pub lambdamax: f64,
    pub volume_increase: f64,
}

/// struct ipx_parameters (ipx_parameters.h); the C++ side passes it by
/// pointer
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Parameters {
    pub display: Int,
    pub logfile: *const std::ffi::c_char,
    pub print_interval: f64,
    pub time_limit: f64,
    pub analyse_basis_data: bool,
    pub dualize: Int,
    pub scale: Int,
    pub ipm_maxiter: Int,
    pub ipm_feasibility_tol: f64,
    pub ipm_optimality_tol: f64,
    pub ipm_drop_primal: f64,
    pub ipm_drop_dual: f64,
    pub kkt_tol: f64,
    pub crash_basis: Int,
    pub dependency_tol: f64,
    pub volume_tol: f64,
    pub rows_per_slice: Int,
    pub maxskip_updates: Int,
    pub lu_kernel: Int,
    pub lu_pivottol: f64,
    pub run_crossover: Int,
    pub start_crossover_tol: f64,
    pub pfeasibility_tol: f64,
    pub dfeasibility_tol: f64,
    pub debug: Int,
    pub switchiter: Int,
    pub stop_at_switch: Int,
    pub update_heuristic: Int,
    pub maxpasses: Int,
    pub run_centring: Int,
    pub max_centring_steps: Int,
    pub centring_ratio_tolerance: f64,
    pub centring_ratio_reduction: f64,
    pub centring_alpha_scaling: f64,
    pub bad_products_tolerance: Int,
    pub highs_logging: bool,
    pub timeless_log: bool,
    pub log_options: *const std::ffi::c_void,
}

impl Default for Parameters {
    /// The defaults of ipx::Parameters()
    fn default() -> Self {
        Parameters {
            display: 1,
            logfile: std::ptr::null(),
            print_interval: 60.0,
            time_limit: -1.0,
            analyse_basis_data: false,
            dualize: -1,
            scale: 1,
            ipm_maxiter: 300,
            ipm_feasibility_tol: 1e-6,
            ipm_optimality_tol: 1e-8,
            ipm_drop_primal: 1e-9,
            ipm_drop_dual: 1e-9,
            kkt_tol: 0.3,
            crash_basis: 1,
            dependency_tol: 1e-6,
            volume_tol: 2.0,
            rows_per_slice: 10000,
            maxskip_updates: 10,
            lu_kernel: 0,
            lu_pivottol: 0.0625,
            run_crossover: 1,
            start_crossover_tol: 1e-8,
            pfeasibility_tol: 1e-7,
            dfeasibility_tol: 1e-7,
            debug: 0,
            switchiter: -1,
            stop_at_switch: 0,
            update_heuristic: 1,
            maxpasses: -1,
            run_centring: 0,
            max_centring_steps: 5,
            centring_ratio_tolerance: 100.0,
            centring_ratio_reduction: 1.5,
            centring_alpha_scaling: 0.5,
            bad_products_tolerance: 3,
            highs_logging: false,
            timeless_log: false,
            log_options: std::ptr::null(),
        }
    }
}

// The C layouts (checked on the C++ side too, lp_solver_rs.cc)
const _: () = assert!(std::mem::size_of::<Parameters>() == 240);
const _: () = assert!(std::mem::offset_of!(Parameters, log_options) == 232);
const _: () = assert!(std::mem::size_of::<Info>() == 464);
const _: () = assert!(std::mem::offset_of!(Info, volume_increase) == 456);

/// std::max for doubles: `(a < b) ? b : a` (differs from f64::max on NaN)
#[inline]
pub(crate) fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// std::min for doubles: `(b < a) ? b : a`
#[inline]
pub(crate) fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// The binary exponent of std::frexp: |x| = f * 2^exp with f in [0.5,1)
/// (0 for zero; x is finite)
pub(crate) fn frexp_exp(x: f64) -> i32 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }
    let bits = x.abs().to_bits();
    let e = ((bits >> 52) & 0x7ff) as i32;
    if e == 0 {
        // subnormal: x = mant * 2^-1074 with the top bit of mant at
        // 63 - leading_zeros
        let mant = bits & ((1u64 << 52) - 1);
        -1010 - mant.leading_zeros() as i32
    } else {
        e - 1022
    }
}

/// std::ldexp(1.0, e), exact for the exponents IPX uses
pub(crate) fn ldexp1(e: i32) -> f64 {
    if (-1022..=1023).contains(&e) {
        f64::from_bits(((e + 1023) as u64) << 52)
    } else {
        2f64.powi(e)
    }
}

