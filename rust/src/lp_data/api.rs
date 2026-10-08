//! Methods of the `Highs` class whose logic is Rust (HighsInterface.cpp,
//! Highs.cpp): the standard form LP (formStandardFormLp).
//!
//! clang fuses the standard form's `offset += cost * bound` and `rhs -=
//! value * bound`.

use super::ffi::{RsMut, RsVec};
use super::{Log, LogType, INF};
use crate::log_user;
use crate::util::fma::ClangFma;

/// The standard form LP min c'x s.t. Ax = b, x >= 0 of an LP
#[derive(Debug, Default, PartialEq)]
pub struct StandardForm {
    pub offset: f64,
    pub cost: Vec<f64>,
    pub rhs: Vec<f64>,
    /// Column-wise
    pub start: Vec<i32>,
    pub index: Vec<i32>,
    pub value: Vec<f64>,
}

/// The LP data the standard form reads: the matrix row-wise
pub struct StdLp<'a> {
    pub num_col: usize,
    pub sense: i32,
    pub offset: f64,
    pub col_cost: &'a [f64],
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    pub row_lower: &'a [f64],
    pub row_upper: &'a [f64],
    pub ar_start: &'a [i32],
    pub ar_index: &'a [i32],
    pub ar_value: &'a [f64],
}

/// formStandardFormLp: the rows (one-sided rows get a slack, boxed rows
/// two), the boxed columns' upper bounds as rows, the columns shifted to
/// nonnegativity (free columns split), then the slacks
pub fn form_standard_form_lp(log: &Log, lp: &StdLp) -> StandardForm {
    let sense = lp.sense as f64;
    let mut f = StandardForm { offset: sense * lp.offset, ..Default::default() };
    for j in 0..lp.num_col {
        f.cost.push(sense * lp.col_cost[j]);
    }
    // The row-wise matrix (HighsSparseMatrix::addRows of one row)
    let mut num_matrix_col = lp.num_col;
    let mut ar_start: Vec<i32> = vec![0];
    let mut ar_index: Vec<i32> = Vec::new();
    let mut ar_value: Vec<f64> = Vec::new();
    let mut add_row = |index: &[i32], value: &[f64]| {
        ar_index.extend_from_slice(index);
        ar_value.extend_from_slice(value);
        ar_start.push(ar_index.len() as i32);
    };
    let (mut num_fixed_row, mut num_boxed_row, mut num_lower_row, mut num_upper_row, mut num_free_row) =
        (0, 0, 0, 0, 0);
    let (mut num_fixed_col, mut num_boxed_col, mut num_lower_col, mut num_upper_col, mut num_free_col) =
        (0, 0, 0, 0, 0);
    let mut slack_ix: Vec<i32> = Vec::new();
    for i in 0..lp.row_lower.len() {
        let lower = lp.row_lower[i];
        let upper = lp.row_upper[i];
        if lower <= -INF && upper >= INF {
            num_free_row += 1;
            continue;
        }
        let els = lp.ar_start[i] as usize..lp.ar_start[i + 1] as usize;
        let (index, value) = (&lp.ar_index[els.clone()], &lp.ar_value[els]);
        if lower == upper {
            num_fixed_row += 1;
            add_row(index, value);
            f.rhs.push(upper);
        } else if lower <= -INF {
            num_upper_row += 1;
            slack_ix.push(f.rhs.len() as i32 + 1);
            add_row(index, value);
            f.rhs.push(upper);
        } else if upper >= INF {
            num_lower_row += 1;
            slack_ix.push(-(f.rhs.len() as i32 + 1));
            add_row(index, value);
            f.rhs.push(lower);
        } else {
            num_boxed_row += 1;
            slack_ix.push(-(f.rhs.len() as i32 + 1));
            add_row(index, value);
            f.rhs.push(lower);
            slack_ix.push(f.rhs.len() as i32 + 1);
            add_row(index, value);
            f.rhs.push(upper);
        }
    }
    // Rows x + s = u of the boxed columns
    for j in 0..lp.num_col {
        if lp.col_lower[j] > -INF && lp.col_upper[j] < INF {
            f.cost.push(0.0);
            num_matrix_col += 1;
            add_row(&[j as i32, num_matrix_col as i32 - 1], &[1.0, 1.0]);
            f.rhs.push(lp.col_upper[j]);
        }
    }
    // HighsSparseMatrix::ensureColwise
    let num_row = ar_start.len() - 1;
    let num_nz = ar_start[num_row] as usize;
    let mut start = vec![0i32; num_matrix_col + 1];
    let mut index = vec![0i32; num_nz];
    let mut value = vec![0f64; num_nz];
    if num_nz > 0 {
        let mut length = vec![0i32; num_matrix_col];
        for &j in &ar_index[..num_nz] {
            length[j as usize] += 1;
        }
        for j in 0..num_matrix_col {
            start[j + 1] = start[j] + length[j];
        }
        for i in 0..num_row {
            for el in ar_start[i] as usize..ar_start[i + 1] as usize {
                let j = ar_index[el] as usize;
                let to = start[j] as usize;
                index[to] = i as i32;
                value[to] = ar_value[el];
                start[j] += 1;
            }
        }
        start[0] = 0;
        for j in 0..num_matrix_col {
            start[j + 1] = start[j] + length[j];
        }
    }
    // Columns to nonnegativity
    for j in 0..lp.num_col {
        let cost = sense * lp.col_cost[j];
        let lower = lp.col_lower[j];
        let upper = lp.col_upper[j];
        if lower > -INF {
            if upper < INF {
                if lower == upper {
                    num_fixed_col += 1;
                } else {
                    num_boxed_col += 1;
                }
            } else {
                num_lower_col += 1;
            }
            if lower != 0.0 {
                f.offset = cost.mul_add_c(lower, f.offset);
                for el in start[j] as usize..start[j + 1] as usize {
                    let i = index[el] as usize;
                    f.rhs[i] = (-value[el]).mul_add_c(lower, f.rhs[i]);
                }
            }
        } else if upper < INF {
            num_upper_col += 1;
            f.offset = cost.mul_add_c(upper, f.offset);
            f.cost[j] = -cost;
            for el in start[j] as usize..start[j + 1] as usize {
                let i = index[el] as usize;
                f.rhs[i] = (-value[el]).mul_add_c(upper, f.rhs[i]);
                value[el] = -value[el];
            }
        } else {
            num_free_col += 1;
            f.cost.push(-cost);
            for el in start[j] as usize..start[j + 1] as usize {
                index.push(index[el]);
                value.push(-value[el]);
            }
            start.push(index.len() as i32);
        }
    }
    for &i in &slack_ix {
        f.cost.push(0.0);
        if i > 0 {
            index.push(i - 1);
            value.push(1.0);
        } else {
            index.push(-i - 1);
            value.push(-1.0);
        }
        start.push(index.len() as i32);
    }
    log_user!(
        log,
        LogType::Info,
        "Standard form LP obtained for LP with (free / lower / upper / boxed / fixed) variables (%d / %d / %d / %d / %d) and constraints (%d / %d / %d / %d / %d) \n",
        num_free_col,
        num_lower_col,
        num_upper_col,
        num_boxed_col,
        num_fixed_col,
        num_free_row,
        num_lower_row,
        num_upper_row,
        num_boxed_row,
        num_fixed_row
    );
    f.start = start;
    f.index = index;
    f.value = value;
    f
}

/// The standard form's C++ vectors (Highs::standard_form_*)
#[repr(C)]
pub struct CStandardForm {
    pub offset: *mut f64,
    pub cost: RsVec<f64>,
    pub rhs: RsVec<f64>,
    pub start: RsVec<i32>,
    pub index: RsVec<i32>,
    pub value: RsVec<f64>,
}

/// formStandardFormLp between the C++ ensureRowwise and ensureColwise of
/// the LP's matrix; the C++ sets the matrix dimensions and format
///
/// # Safety
/// The views are the C++ objects'; the matrix is row-wise
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_form_standard_form_lp(
    log: *const Log,
    num_col: i32,
    sense: i32,
    offset: f64,
    col_cost: RsMut<f64>,
    col_lower: RsMut<f64>,
    col_upper: RsMut<f64>,
    row_lower: RsMut<f64>,
    row_upper: RsMut<f64>,
    ar_start: RsMut<i32>,
    ar_index: RsMut<i32>,
    ar_value: RsMut<f64>,
    out: *mut CStandardForm,
) {
    let lp = StdLp {
        num_col: num_col as usize,
        sense,
        offset,
        col_cost: col_cost.get(),
        col_lower: col_lower.get(),
        col_upper: col_upper.get(),
        row_lower: row_lower.get(),
        row_upper: row_upper.get(),
        ar_start: ar_start.get(),
        ar_index: ar_index.get(),
        ar_value: ar_value.get(),
    };
    let f = form_standard_form_lp(&*log, &lp);
    let out = &mut *out;
    *out.offset = f.offset;
    out.cost.assign(&f.cost);
    out.rhs.assign(&f.rhs);
    out.start.assign(&f.start);
    out.index.assign(&f.index);
    out.value.assign(&f.value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_form() {
        // min x0 - x1 s.t. 1 <= x0 + x1 <= 3, x0 in [1, 2], x1 free
        let lp = StdLp {
            num_col: 2,
            sense: 1,
            offset: 0.5,
            col_cost: &[1.0, -1.0],
            col_lower: &[1.0, -INF],
            col_upper: &[2.0, INF],
            row_lower: &[1.0],
            row_upper: &[3.0],
            ar_start: &[0, 2],
            ar_index: &[0, 1],
            ar_value: &[1.0, 1.0],
        };
        let f = form_standard_form_lp(&Log::none(), &lp);
        // Columns x0, x1, s (boxed x0), x1-, slacks of the two rows
        assert_eq!(f.cost, vec![1.0, -1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(f.offset, 1.5);
        assert_eq!(f.rhs, vec![0.0, 2.0, 1.0]);
        assert_eq!(f.start, vec![0, 3, 5, 6, 8, 9, 10]);
        assert_eq!(f.index, vec![0, 1, 2, 0, 1, 2, 0, 1, 0, 1]);
    }
}
