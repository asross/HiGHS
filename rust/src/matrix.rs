//! PRICE kernels of HighsSparseMatrix (double precision paths). The matrix
//! is CSC for `price_by_column` and row-wise for the others; `end[i]` is
//! `start[i + 1]`, or `p_end[i]` for a partitioned row-wise matrix.

use crate::util::fma::ClangFma;

const K_HIGHS_TINY: f64 = 1e-14;
const K_HIGHS_ZERO: f64 = 1e-50;

/// result = column^T A over all columns, storing every index and advancing
/// the count only for nonzeros (branchless; result is cleared on entry).
/// Returns the count.
pub fn price_by_column(
    start: &[i32],
    index: &[i32],
    value: &[f64],
    column: &[f64],
    result: &mut [f64],
    result_index: &mut [i32],
) -> usize {
    let mut count = 0;
    for col in 0..start.len() - 1 {
        let (from, to) = (start[col] as usize, start[col + 1] as usize);
        let mut v = 0.0;
        for (&i, &a) in index[from..to].iter().zip(&value[from..to]) {
            v = column[i as usize].mul_add_c(a, v);
        }
        let nonzero = v.abs() > K_HIGHS_TINY;
        result[col] = if nonzero { v } else { 0.0 };
        result_index[count] = col as i32;
        count += nonzero as usize;
    }
    count
}

/// result += multiplier * row, keeping the nonzero list and replacing tiny
/// values by kHighsZero.
#[inline]
fn add_row(index: &[i32], value: &[f64], multiplier: f64, result: &mut [f64], result_index: &mut [i32], count: &mut usize) {
    for (&j, &a) in index.iter().zip(value) {
        let col = j as usize;
        let value0 = result[col];
        let value1 = multiplier.mul_add_c(a, value0);
        if value0 == 0.0 {
            result_index[*count] = j;
            *count += 1;
        }
        result[col] = if value1.abs() < K_HIGHS_TINY { K_HIGHS_ZERO } else { value1 };
    }
}

/// Hyper-sparse row-wise PRICE from `from_index` of the column's nonzeros,
/// stopping before a row whose fill would make the result too dense.
/// Returns the index of the first row not priced; updates `count`.
#[allow(clippy::too_many_arguments)]
pub fn price_by_row_sparse(
    start: &[i32],
    end: &[i32],
    index: &[i32],
    value: &[f64],
    num_col: usize,
    column_index: &[i32],
    column: &[f64],
    from_index: usize,
    switch_density: f64,
    result: &mut [f64],
    result_index: &mut [i32],
    count: &mut usize,
) -> usize {
    let inv_num_col = 1.0 / num_col as f64;
    let mut next_index = from_index;
    for (ix, &row) in column_index.iter().enumerate().skip(from_index) {
        let row = row as usize;
        let (from, to) = (start[row] as usize, end[row] as usize);
        let local_density = *count as f64 * inv_num_col;
        if *count + (to - from) >= num_col || local_density > switch_density {
            break;
        }
        let multiplier = column[row];
        if multiplier != 0.0 {
            add_row(&index[from..to], &value[from..to], multiplier, result, result_index, count);
        }
        next_index = ix + 1;
    }
    next_index
}

/// Row-wise PRICE into a dense result, from `from_index` of the column's
/// nonzeros (the result is zeroed beforehand, or holds a partial PRICE).
pub fn price_by_row_dense(
    start: &[i32],
    end: &[i32],
    index: &[i32],
    value: &[f64],
    column_index: &[i32],
    column: &[f64],
    from_index: usize,
    result: &mut [f64],
) {
    for &row in &column_index[from_index..] {
        let row = row as usize;
        let multiplier = column[row];
        let (from, to) = (start[row] as usize, end[row] as usize);
        for (&j, &a) in index[from..to].iter().zip(&value[from..to]) {
            let x = &mut result[j as usize];
            let value1 = multiplier.mul_add_c(a, *x);
            *x = if value1.abs() < K_HIGHS_TINY { K_HIGHS_ZERO } else { value1 };
        }
    }
}

/// Sets result_index to the nonzeros of a dense result, zeroing tiny values.
/// Returns the count.
pub fn index_dense_result(result: &mut [f64], result_index: &mut [i32]) -> usize {
    let mut count = 0;
    for (col, x) in result.iter_mut().enumerate() {
        if x.abs() < K_HIGHS_TINY {
            *x = 0.0;
        } else {
            result_index[count] = col as i32;
            count += 1;
        }
    }
    count
}

/// HighsSparseMatrix::priceByRowWithSwitch (double precision): row-wise
/// PRICE from `from_index` of the column's nonzeros, hyper-sparse if
/// `hyper` until the result is too dense, then dense. `end` is p_end for a
/// partitioned matrix. Returns the result count.
#[allow(clippy::too_many_arguments)]
pub fn price_by_row_with_switch(
    start: &[i32],
    end: &[i32],
    index: &[i32],
    value: &[f64],
    num_col: usize,
    column_index: &[i32],
    column: &[f64],
    hyper: bool,
    from_index: usize,
    switch_density: f64,
    result_count: usize,
    result: &mut [f64],
    result_index: &mut [i32],
) -> usize {
    let mut count = result_count;
    let mut next = from_index;
    if hyper {
        next = price_by_row_sparse(start, end, index, value, num_col, column_index, column, next,
            switch_density, result, result_index, &mut count);
    }
    if next < column_index.len() {
        price_by_row_dense(start, end, index, value, column_index, column, next, result);
        count = index_dense_result(result, result_index);
    } else {
        // complete: remove small values (HVector::tight)
        let mut kept = 0;
        for i in 0..count {
            let col = result_index[i];
            if result[col as usize].abs() < K_HIGHS_TINY {
                result[col as usize] = 0.0;
            } else {
                result_index[kept] = col;
                kept += 1;
            }
        }
        count = kept;
    }
    count
}

mod ffi {
    use std::slice::{from_raw_parts as s, from_raw_parts_mut as m};

    /// # Safety
    /// `start` has num_col + 1 entries, `index`/`value` start[num_col]
    /// entries, `column` covers the row indices, and the results num_col.
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_price_by_column(
        num_col: i32, start: *const i32, index: *const i32, value: *const f64,
        num_row: i32, column: *const f64, result: *mut f64, result_index: *mut i32,
    ) -> i32 {
        let n = num_col as usize;
        let start = s(start, n + 1);
        let nnz = start[n] as usize;
        super::price_by_column(start, s(index, nnz), s(value, nnz), s(column, num_row as usize),
            m(result, n), m(result_index, n)) as i32
    }

    /// Row-wise PRICE with possible switch to a dense result, as in
    /// HighsSparseMatrix::priceByRowWithSwitch (double precision): returns
    /// the result count. `end` is p_end or &start[1]; `num_nz` covers both.
    ///
    /// # Safety
    /// Arrays as described, `start`/`end` cover the column's row indices.
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_price_by_row(
        num_row: i32, num_col: i32, start: *const i32, end: *const i32, num_nz: i32,
        index: *const i32, value: *const f64, column_count: i32, column_index: *const i32,
        column: *const f64, hyper: bool, from_index: i32, switch_density: f64,
        result_count: i32, result: *mut f64, result_index: *mut i32,
    ) -> i32 {
        let (nr, nc, nz) = (num_row as usize, num_col as usize, num_nz as usize);
        let (start, end) = (s(start, nr), s(end, nr));
        let (index, value) = (s(index, nz), s(value, nz));
        let column_index = s(column_index, column_count as usize);
        let column = s(column, nr);
        let (result, result_index) = (m(result, nc), m(result_index, nc));
        super::price_by_row_with_switch(start, end, index, value, nc, column_index, column, hyper,
            from_index as usize, switch_density, result_count as usize, result, result_index) as i32
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_and_column_price_agree() {
        // A = [[1, 2, 0], [0, 3, 4]]: CSC and row-wise
        let (cs, ci, cv) = ([0, 1, 3, 4], [0, 0, 1, 1], [1.0, 2.0, 3.0, 4.0]);
        let (rs, ri, rv) = ([0, 2, 4], [0, 1, 1, 2], [1.0, 2.0, 3.0, 4.0]);
        let y = [2.0, -1.0];
        let (mut a, mut ai) = ([0.0; 3], [0; 3]);
        let n = price_by_column(&cs, &ci, &cv, &y, &mut a, &mut ai);
        assert_eq!((n, a), (3, [2.0, 1.0, -4.0]));
        let (mut b, mut bi, mut count) = ([0.0; 3], [0; 3], 0);
        let next = price_by_row_sparse(&rs, &rs[1..], &ri, &rv, 3, &[0, 1], &y, 0, f64::INFINITY,
            &mut b, &mut bi, &mut count);
        assert_eq!(next, 1); // the second row would fill the result
        price_by_row_dense(&rs, &rs[1..], &ri, &rv, &[0, 1], &y, next, &mut b);
        assert_eq!(index_dense_result(&mut b, &mut bi), 3);
        assert_eq!(b, a);
    }
}
