//! `extern "C"` entry of the Rust cuPDLP-C, called by solveLpCupdlp
//! (highs/pdlp/CupdlpWrapperRs.cpp)

use super::{solve, Log, Lp, Params, Solution};
use crate::ffi::{sl, sl_mut};
use std::ffi::c_char;

/// The HiGHS LP; mirrored by PdlpRsLp in C++
#[repr(C)]
pub struct PdlpRsLp {
    pub num_col: i32,
    pub num_row: i32,
    pub a_start: *const i32,
    pub a_index: *const i32,
    pub a_value: *const f64,
    pub col_cost: *const f64,
    pub col_lower: *const f64,
    pub col_upper: *const f64,
    pub row_lower: *const f64,
    pub row_upper: *const f64,
    pub offset: f64,
    pub sense: f64,
}

/// Solves the LP; returns the termination code (termination_code of
/// cupdlp_defs.h) and sets the solution, its validity and the iteration
/// count
///
/// # Safety
/// lp and params point to valid structs whose arrays have the lengths of
/// the LP (a_start: num_col+1, a_index and a_value: a_start[num_col]);
/// the solution arrays have num_col / num_row entries; print, if not null,
/// is callable with a NUL-terminated string
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn pdlp_rs_solve(
    lp: *const PdlpRsLp,
    params: *const Params,
    print: Option<extern "C" fn(*const c_char)>,
    col_value: *mut f64,
    col_dual: *mut f64,
    row_value: *mut f64,
    row_dual: *mut f64,
    value_valid: *mut i32,
    dual_valid: *mut i32,
    num_iter: *mut i32,
) -> i32 {
    let (lp, params) = (&*lp, &*params);
    let (n, m) = (lp.num_col, lp.num_row);
    let start = sl(lp.a_start, n + 1);
    let nnz = start[n as usize];
    let highs_lp = Lp {
        start,
        index: sl(lp.a_index, nnz),
        value: sl(lp.a_value, nnz),
        col_cost: sl(lp.col_cost, n),
        col_lower: sl(lp.col_lower, n),
        col_upper: sl(lp.col_upper, n),
        row_lower: sl(lp.row_lower, m),
        row_upper: sl(lp.row_upper, m),
        offset: lp.offset,
        sense: lp.sense,
    };
    let mut sol = Solution {
        col_value: sl_mut(col_value, n),
        col_dual: sl_mut(col_dual, n),
        row_value: sl_mut(row_value, m),
        row_dual: sl_mut(row_dual, m),
        value_valid: *value_valid != 0,
        dual_valid: *dual_valid != 0,
    };
    let log = Log {
        level: params.log_level,
        print,
    };
    let (code, iters) = solve(&highs_lp, params, &log, &mut sol);
    *value_valid = sol.value_valid as i32;
    *dual_valid = sol.dual_valid as i32;
    *num_iter = iters;
    code as i32
}
