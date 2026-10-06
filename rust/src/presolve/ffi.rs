//! `extern "C"` shims for HighsPostsolveStack (highs/presolve/
//! HighsPostsolveStack.h under HIGHS_RUST)

use super::postsolve::{
    compress_index_maps, reduced_primal_solution, undo, Basis, DuplicateColumn, Reduction, Solution, Stack,
    Tolerances,
};
use std::slice::{from_raw_parts, from_raw_parts_mut};

/// The stack's data; mirrored by PostsolveRsStack in C++
#[repr(C)]
pub struct CStack {
    data: *const u8,
    data_len: usize,
    reductions: *const Reduction,
    num_reductions: usize,
    orig_col_index: *const i32,
    num_col: usize,
    orig_row_index: *const i32,
    num_row: usize,
}

/// A solution and basis in the original space; mirrored by
/// PostsolveRsSolution in C++. The dual (status) arrays are read only if
/// dual_valid (basis_valid).
#[repr(C)]
pub struct CSolution {
    col_value: *mut f64,
    col_dual: *mut f64,
    col_status: *mut u8,
    num_col: usize,
    row_value: *mut f64,
    row_dual: *mut f64,
    row_status: *mut u8,
    num_row: usize,
    dual_valid: bool,
    basis_valid: bool,
}

unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    if n == 0 {
        &[]
    } else {
        from_raw_parts(p, n)
    }
}

unsafe fn sl_mut<'a, T>(p: *mut T, n: usize) -> &'a mut [T] {
    if n == 0 {
        &mut []
    } else {
        from_raw_parts_mut(p, n)
    }
}

unsafe fn stack<'a>(s: &CStack) -> Stack<'a> {
    Stack {
        data: sl(s.data, s.data_len),
        reductions: sl(s.reductions, s.num_reductions),
        orig_col_index: sl(s.orig_col_index, s.num_col),
        orig_row_index: sl(s.orig_row_index, s.num_row),
    }
}

/// HighsPostsolveStack::undo / undoUntil
///
/// # Safety
/// The pointers of `s` and `x` are valid for their lengths (the dual and
/// status arrays only if dual_valid / basis_valid); the stack holds the
/// records HighsPostsolveStack pushed
#[no_mangle]
pub unsafe extern "C" fn highs_rs_postsolve_undo(
    s: *const CStack,
    tol: *const Tolerances,
    x: *const CSolution,
    until: usize,
    report_col: i32,
) {
    let s = stack(&*s);
    let x = &*x;
    let (nc, nr) = if x.dual_valid { (x.num_col, x.num_row) } else { (0, 0) };
    let (bc, br) = if x.basis_valid { (x.num_col, x.num_row) } else { (0, 0) };
    let mut sol = Solution {
        col_value: sl_mut(x.col_value, x.num_col),
        row_value: sl_mut(x.row_value, x.num_row),
        col_dual: sl_mut(x.col_dual, nc),
        row_dual: sl_mut(x.row_dual, nr),
        dual_valid: x.dual_valid,
    };
    let mut basis = Basis {
        col_status: sl_mut(x.col_status, bc),
        row_status: sl_mut(x.row_status, br),
        valid: x.basis_valid,
    };
    undo(&s, &*tol, &mut sol, &mut basis, until, report_col);
}

/// HighsPostsolveStack::getReducedPrimalSolution on sol (num_orig_col
/// entries, a copy of the original solution)
///
/// # Safety
/// As for highs_rs_postsolve_undo; sol has the original number of columns
#[no_mangle]
pub unsafe extern "C" fn highs_rs_postsolve_reduced_primal(s: *const CStack, sol: *mut f64, num_orig_col: usize) {
    reduced_primal_solution(&stack(&*s), sl_mut(sol, num_orig_col));
}

/// HighsPostsolveStack::compressIndexMaps; writes the new map sizes
///
/// # Safety
/// The pointers are valid for their lengths; the new indices are -1 or less
/// than their position
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_postsolve_compress_index_maps(
    orig_row_index: *mut i32,
    num_orig_row: usize,
    orig_col_index: *mut i32,
    num_orig_col: usize,
    new_row_index: *const i32,
    num_new_row: usize,
    new_col_index: *const i32,
    num_new_col: usize,
    new_num_row: *mut usize,
    new_num_col: *mut usize,
) {
    (*new_num_row, *new_num_col) = compress_index_maps(
        sl_mut(orig_row_index, num_orig_row),
        sl_mut(orig_col_index, num_orig_col),
        sl(new_row_index, num_new_row),
        sl(new_col_index, num_new_col),
    );
}

/// DuplicateColumn::okMerge
///
/// # Safety
/// r points to a HighsPostsolveStack::DuplicateColumn
#[no_mangle]
pub unsafe extern "C" fn highs_rs_postsolve_duplicate_col_ok_merge(r: *const DuplicateColumn, tolerance: f64) -> bool {
    (*r).ok_merge(tolerance)
}
