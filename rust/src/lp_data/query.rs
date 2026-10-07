//! The checks and data logic of Highs' basis and tableau queries
//! (getBasisInverseRow/Col, getBasisSolve, getBasisTransposeSolve,
//! getReducedRow/Column and basisSolveInterface's extraction of the
//! solution) and of setSolution. The solves themselves (HEkk btran/ftran)
//! stay C++ (Highs.cpp, HighsInterface.cpp).
//!
//! clang fuses the reduced row's `value += a * binv[row]`.

use super::ffi::RsMut;
use super::{Log, LogType, Status};
use crate::log_user;
use crate::util::fma::ClangFma;

/// kHighsTiny
const TINY: f64 = 1e-14;

/// The argument checks of a basis query: `null` names a NULL argument;
/// `index_kind` ("Row" or "Column") with `index` out of [0, dim) is an
/// error; then an INVERT is required
pub fn check_query(
    log: &Log,
    method: &str,
    null: Option<&str>,
    index_kind: Option<&str>,
    index: i32,
    dim: i32,
    has_invert: bool,
) -> Status {
    if let Some(arg) = null {
        log_user!(log, LogType::Error, "%s: %s is NULL\n", method, arg);
        return Status::Error;
    }
    if let Some(kind) = index_kind {
        if index < 0 || index >= dim {
            log_user!(
                log,
                LogType::Error,
                "%s index %d out of range [0, %d] in %s\n",
                kind,
                index,
                dim - 1,
                method
            );
            return Status::Error;
        }
    }
    if !has_invert {
        log_user!(log, LogType::Error, "No invertible representation for %s\n", method);
        return Status::Error;
    }
    Status::Ok
}

/// The row of B^{-1}A from the row of B^{-1} (getReducedRow): values
/// below kHighsTiny are zeroed; returns the number of nonzeros, whose
/// indices are put in `indices` if not empty
pub fn reduced_row(start: &[i32], index: &[i32], value: &[f64], binv_row: &[f64], row: &mut [f64], indices: &mut [i32]) -> i32 {
    let mut num_nz = 0;
    for col in 0..row.len() {
        let mut v = 0.0;
        for el in start[col] as usize..start[col + 1] as usize {
            v = value[el].mul_add_c(binv_row[index[el] as usize], v);
        }
        row[col] = 0.0;
        if v.abs() > TINY {
            if !indices.is_empty() {
                indices[num_nz as usize] = col as i32;
            }
            num_nz += 1;
            row[col] = v;
        }
    }
    num_nz
}

/// The solution of basisSolveInterface from the solved HVector (count,
/// index, array): returns the number of nonzeros if indices are wanted
/// (`indices` not empty), else -1
pub fn extract_solve(count: i32, index: &[i32], array: &[f64], solution: &mut [f64], indices: &mut [i32]) -> i32 {
    let num_row = solution.len() as i32;
    if count > num_row {
        if !indices.is_empty() {
            // The C++ writes through a null pointer here
            panic!("basisSolveInterface: dense solve with indices");
        }
        solution.copy_from_slice(&array[..solution.len()]);
        return -1;
    }
    solution.fill(0.0);
    for (x, &i) in index[..count as usize].iter().enumerate() {
        solution[i as usize] = array[i as usize];
        if !indices.is_empty() {
            indices[x] = i;
        }
    }
    if indices.is_empty() {
        -1
    } else {
        count
    }
}

/// The checks of setSolution(num_entries, index, value): returns Error on
/// an index out of range or an infeasible value, Warning on duplicates
pub fn check_sparse_solution(log: &Log, index: &[i32], value: &[f64], lower: &[f64], upper: &[f64], pft: f64) -> Status {
    let num_col = lower.len() as i32;
    let mut is_set = vec![false; lower.len()];
    let mut num_duplicates = 0;
    for (ix, (&col, &v)) in index.iter().zip(value).enumerate() {
        if col < 0 || col >= num_col {
            log_user!(
                log,
                LogType::Error,
                "setSolution: User solution index %d has value %d out of range [0, %d)\n",
                ix,
                col,
                num_col
            );
            return Status::Error;
        }
        let c = col as usize;
        if v < lower[c] - pft || upper[c] + pft < v {
            log_user!(
                log,
                LogType::Error,
                "setSolution: User solution value %d of %g is infeasible for bounds [%g, %g]\n",
                ix,
                v,
                lower[c],
                upper[c]
            );
            return Status::Error;
        }
        if is_set[c] {
            num_duplicates += 1;
        }
        is_set[c] = true;
    }
    if num_duplicates > 0 {
        log_user!(
            log,
            LogType::Warning,
            "setSolution: User set of indices has %d duplicate%s: last value used\n",
            num_duplicates,
            if num_duplicates == 1 { "" } else { "s" }
        );
        return Status::Warning;
    }
    Status::Ok
}

/// Which parts of setSolution(solution) are new: bit 0 primal, bit 1
/// dual; neither is an error, logged
pub fn new_solution_parts(log: &Log, num_col: i32, num_row: i32, col_value_size: usize, row_dual_size: usize) -> i32 {
    let new_primal = num_col > 0 && col_value_size >= num_col as usize;
    let new_dual = num_row > 0 && row_dual_size >= num_row as usize;
    if !new_primal && !new_dual {
        log_user!(
            log,
            LogType::Error,
            "setSolution: User solution is rejected due to mismatch between size of col_value and row_dual vectors (%d, %d) and number of columns and rows in the model (%d, %d)\n",
            col_value_size as i32,
            row_dual_size as i32,
            num_col,
            num_row
        );
    }
    new_primal as i32 | (new_dual as i32) << 1
}

// The C++ entry points

/// # Safety
/// `log` valid; strings valid for their lengths
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_check_query(
    log: *const Log,
    method: *const u8,
    method_len: usize,
    null: *const u8,
    null_len: usize,
    index_kind: *const u8,
    index_kind_len: usize,
    index: i32,
    dim: i32,
    has_invert: bool,
) -> i32 {
    let s = |p: *const u8, n: usize| -> Option<&str> {
        if p.is_null() {
            None
        } else {
            Some(std::str::from_utf8_unchecked(std::slice::from_raw_parts(p, n)))
        }
    };
    check_query(
        &*log,
        s(method, method_len).unwrap_or(""),
        s(null, null_len),
        s(index_kind, index_kind_len),
        index,
        dim,
        has_invert,
    ) as i32
}

/// # Safety
/// The arrays valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_reduced_row(
    start: RsMut<i32>,
    index: RsMut<i32>,
    value: RsMut<f64>,
    binv_row: RsMut<f64>,
    row: RsMut<f64>,
    indices: RsMut<i32>,
) -> i32 {
    reduced_row(start.get(), index.get(), value.get(), binv_row.get(), row.get_mut(), indices.get_mut())
}

/// # Safety
/// The arrays valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_extract_solve(
    count: i32,
    index: RsMut<i32>,
    array: RsMut<f64>,
    solution: RsMut<f64>,
    indices: RsMut<i32>,
) -> i32 {
    extract_solve(count, index.get(), array.get(), solution.get_mut(), indices.get_mut())
}

/// # Safety
/// The arrays valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_check_sparse_solution(
    log: *const Log,
    index: RsMut<i32>,
    value: RsMut<f64>,
    lower: RsMut<f64>,
    upper: RsMut<f64>,
    pft: f64,
) -> i32 {
    check_sparse_solution(&*log, index.get(), value.get(), lower.get(), upper.get(), pft) as i32
}

/// # Safety
/// `log` valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_new_solution_parts(
    log: *const Log,
    num_col: i32,
    num_row: i32,
    col_value_size: usize,
    row_dual_size: usize,
) -> i32 {
    new_solution_parts(&*log, num_col, num_row, col_value_size, row_dual_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduced_and_solve() {
        // A = [[1, 2], [3, 0]] column-wise
        let start = [0, 2, 3];
        let index = [0, 1, 0];
        let value = [1.0, 3.0, 2.0];
        let mut row = [9.0; 2];
        let mut ind = [0; 2];
        assert_eq!(reduced_row(&start, &index, &value, &[1.0, -1.0 / 3.0], &mut row, &mut ind), 1);
        assert_eq!(row, [0.0, 2.0]);
        assert_eq!(ind[0], 1);
        let mut sol = [5.0; 3];
        let mut ind = [0; 3];
        assert_eq!(extract_solve(1, &[2, 0, 0], &[0.0, 0.0, 4.0], &mut sol, &mut ind), 1);
        assert_eq!(sol, [0.0, 0.0, 4.0]);
        assert_eq!(check_sparse_solution(&Log::none(), &[0, 0], &[1.0, 2.0], &[0.0], &[3.0], 1e-7), Status::Warning);
        assert_eq!(check_sparse_solution(&Log::none(), &[1], &[1.0], &[0.0], &[3.0], 1e-7), Status::Error);
        assert_eq!(new_solution_parts(&Log::none(), 2, 1, 2, 0), 1);
    }
}
