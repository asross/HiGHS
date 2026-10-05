//! Rust ports of HiGHS kernels, called from C++ through `extern "C"` shims.
//! Ported so far: the hyper-sparse triangular solve of HFactor (solveHyper)
//! and the double-precision PRICE kernels of HighsSparseMatrix.

pub mod matrix;
pub mod util;

const K_HIGHS_TINY: f64 = 1e-14;

/// A triangular factor in HFactor's layout: column (or row) `i` has entries
/// `index/value[start[i]..end[i]]` and pivot `pivot_index[i]`; `lookup` maps
/// a row to its pivot position. `pivot_value` is `None` for unit L.
pub struct Factor<'a> {
    pub lookup: &'a [i32],
    pub pivot_index: &'a [i32],
    pub pivot_value: Option<&'a [f64]>,
    pub start: &'a [i32],
    pub end: &'a [i32],
    pub index: &'a [i32],
    pub value: &'a [f64],
}

/// The sparse right-hand side, in HVector's layout.
pub struct Rhs<'a> {
    pub count: usize,
    pub index: &'a mut [i32],
    pub array: &'a mut [f64],
    pub mark: &'a mut [u8],
    pub iwork: &'a mut [i32],
    pub synthetic_tick: f64,
}

/// Same algorithm and floating-point operations as solveHyper in HFactor.cpp
/// (clang contracts `x -= m * v` into a fused multiply-add, hence mul_add),
/// so results are bit-identical.
pub fn solve_hyper(h_size: usize, h: &Factor, rhs: &mut Rhs) {
    let (list_index, list_stack) = rhs.iwork.split_at_mut(h_size);
    let mark = &mut *rhs.mark;
    let mut list_count = 0;
    let mut count_pivot = 0usize;
    let mut count_entry = 0usize;

    // Depth-first search for the topological order of the nonzeros
    for &row in &rhs.index[..rhs.count] {
        let mut hi = h.lookup[row as usize] as usize;
        if mark[hi] != 0 {
            continue;
        }
        let mut hk = h.start[hi] as usize;
        let mut n_stack = 0;
        mark[hi] = 1;
        loop {
            if hk < h.end[hi] as usize {
                let sub = h.lookup[h.index[hk] as usize] as usize;
                hk += 1;
                if mark[sub] == 0 {
                    mark[sub] = 1;
                    list_stack[n_stack] = hi as i32;
                    list_stack[n_stack + 1] = hk as i32;
                    n_stack += 2;
                    hi = sub;
                    hk = h.start[hi] as usize;
                    if hi >= h_size {
                        count_pivot += 1;
                        count_entry += (h.end[hi] - h.start[hi]) as usize;
                    }
                }
            } else {
                list_index[list_count] = hi as i32;
                list_count += 1;
                if n_stack == 0 {
                    break;
                }
                n_stack -= 2;
                hi = list_stack[n_stack] as usize;
                hk = list_stack[n_stack + 1] as usize;
            }
        }
    }
    rhs.synthetic_tick += (count_pivot * 20 + count_entry * 10) as f64;

    // Solve in reverse topological order
    let mut count = 0;
    for &i in list_index[..list_count].iter().rev() {
        let i = i as usize;
        mark[i] = 0;
        let pivot_row = h.pivot_index[i] as usize;
        let mut multiplier = rhs.array[pivot_row];
        if multiplier.abs() > K_HIGHS_TINY {
            if let Some(pivot_value) = h.pivot_value {
                multiplier /= pivot_value[i];
                rhs.array[pivot_row] = multiplier;
            }
            rhs.index[count] = pivot_row as i32;
            count += 1;
            let (start, end) = (h.start[i] as usize, h.end[i] as usize);
            for (&j, &v) in h.index[start..end].iter().zip(&h.value[start..end]) {
                let x = &mut rhs.array[j as usize];
                *x = (-multiplier).mul_add(v, *x);
            }
        } else {
            rhs.array[pivot_row] = 0.0;
        }
    }
    rhs.count = count;
}

/// C entry point. Every pointer comes with its length, so the Rust side is
/// bounds-checked; `pivot_value` may be null.
///
/// # Safety
/// Each pointer must be valid for its stated length, and the output arrays
/// must not alias the factor.
#[no_mangle]
pub unsafe extern "C" fn highs_rs_solve_hyper(
    h_size: i32,
    lookup: *const i32,
    n_lookup: i32,
    pivot_index: *const i32,
    pivot_value: *const f64,
    start: *const i32,
    end: *const i32,
    n_pivot: i32,
    index: *const i32,
    value: *const f64,
    n_entry: i32,
    rhs_count: *mut i32,
    rhs_index: *mut i32,
    rhs_array: *mut f64,
    n_row: i32,
    cwork: *mut u8,
    n_cwork: i32,
    iwork: *mut i32,
    n_iwork: i32,
    synthetic_tick: *mut f64,
) {
    use std::slice::{from_raw_parts as s, from_raw_parts_mut as m};
    let (np, ne) = (n_pivot as usize, n_entry as usize);
    let h = Factor {
        lookup: s(lookup, n_lookup as usize),
        pivot_index: s(pivot_index, np),
        pivot_value: (!pivot_value.is_null()).then(|| s(pivot_value, np)),
        start: s(start, np),
        end: s(end, np),
        index: s(index, ne),
        value: s(value, ne),
    };
    let mut rhs = Rhs {
        count: *rhs_count as usize,
        index: m(rhs_index, n_row as usize),
        array: m(rhs_array, n_row as usize),
        mark: m(cwork, n_cwork as usize),
        iwork: m(iwork, n_iwork as usize),
        synthetic_tick: *synthetic_tick,
    };
    solve_hyper(h_size as usize, &h, &mut rhs);
    *rhs_count = rhs.count as i32;
    *synthetic_tick = rhs.synthetic_tick;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upper_triangular_solve() {
        // U = [[2, 1], [0, 4]] stored by column, pivots on the diagonal:
        // column 1 has the off-diagonal 1 in row 0. Solve U x = (3, 8).
        let h = Factor {
            lookup: &[0, 1],
            pivot_index: &[0, 1],
            pivot_value: Some(&[2.0, 4.0]),
            start: &[0, 0],
            end: &[0, 1],
            index: &[0],
            value: &[1.0],
        };
        let (mut index, mut array) = ([0, 1], [3.0, 8.0]);
        let (mut mark, mut iwork) = ([0u8; 2], [0i32; 8]);
        let mut rhs = Rhs {
            count: 2,
            index: &mut index,
            array: &mut array,
            mark: &mut mark,
            iwork: &mut iwork,
            synthetic_tick: 0.0,
        };
        solve_hyper(2, &h, &mut rhs);
        assert_eq!(rhs.count, 2);
        assert_eq!(array, [0.5, 2.0]);
        assert_eq!(mark, [0, 0]);
    }
}
