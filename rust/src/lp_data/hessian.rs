//! HighsHessian and its utilities (model/HighsHessian.cpp,
//! HighsHessianUtils.cpp, HighsModel.cpp's objective): assessment
//! (dimensions, the matrix, normalising to the lower triangle with the
//! diagonal first, explicit zero diagonals), completion to the model's
//! dimension, the diagonal sign check, triangular / square conversions,
//! column deletion, products, the objective and user scaling. The C++
//! class keeps its data (`CHessian` views it, resizing through `RsVec`).
//!
//! clang fuses `p[i] += v * x[j]`, `y[i] += alpha * v * x[j]` and the
//! objective's `f += 0.5 * x * v * x` / `f += x * v * y` (the last
//! product fused); the double-double objective is not fused.

use super::edit::{update_out_in_index, OutIn};
use super::ffi::{CIndexCollection, RsVec};
use super::lp_utils::{assess_matrix, assess_matrix_dimensions, cmax, cmin, IndexCollection};
use super::user_scale::UserScaleData;
use super::{Log, LogType, Status, INF};
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use crate::util::printf::sprintf;
use crate::{log_dev, log_user};

/// HessianFormat
pub const TRIANGULAR: i32 = 1;
pub const SQUARE: i32 = 2;

/// A HighsHessian, copied out of the C++ class or a view of it
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Hessian {
    pub dim: i32,
    pub format: i32,
    pub start: Vec<i32>,
    pub index: Vec<i32>,
    pub value: Vec<f64>,
}

/// The C++ HighsHessian (HighsRust.h: rsHessian)
#[repr(C)]
pub struct CHessian {
    pub dim: *mut i32,
    pub format: *mut i32,
    pub start: RsVec<i32>,
    pub index: RsVec<i32>,
    pub value: RsVec<f64>,
}

impl CHessian {
    /// # Safety
    /// The pointers are the C++ HighsHessian's
    pub unsafe fn load(&self) -> Hessian {
        Hessian {
            dim: *self.dim,
            format: *self.format,
            start: self.start.to_vec(),
            index: self.index.to_vec(),
            value: self.value.to_vec(),
        }
    }
    /// # Safety
    /// As load
    pub unsafe fn store(&mut self, h: &Hessian) {
        *self.dim = h.dim;
        *self.format = h.format;
        self.start.assign(&h.start);
        self.index.assign(&h.index);
        self.value.assign(&h.value);
    }
}

impl Hessian {
    /// HighsHessian::clear
    pub fn clear(&mut self) {
        self.dim = 0;
        self.start.clear();
        self.index.clear();
        self.value.clear();
        self.format = TRIANGULAR;
        self.start.push(0);
    }
    pub fn num_nz(&self) -> i32 {
        self.start[self.dim as usize]
    }
    /// HighsHessian::exactResize
    pub fn exact_resize(&mut self) {
        if self.dim != 0 {
            self.start.resize(self.dim as usize + 1, 0);
            let num_nz = self.start[self.dim as usize] as usize;
            self.index.resize(num_nz, 0);
            self.value.resize(num_nz, 0.0);
        } else {
            self.clear();
        }
    }
}

/// assessHessianDimensions
pub fn assess_hessian_dimensions(log: &Log, h: &Hessian) -> Status {
    if h.dim == 0 {
        return Status::Ok;
    }
    assess_matrix_dimensions(log, h.dim, false, &h.start, &[], h.index.len(), h.value.len())
}

/// assessHessian
pub fn assess_hessian(log: &Log, h: &mut Hessian, small_matrix_value: f64, large_matrix_value: f64) -> Status {
    let mut return_status = log.interpret(assess_hessian_dimensions(log, h), Status::Ok, "assessHessianDimensions");
    if return_status == Status::Error {
        return return_status;
    }
    if h.dim == 0 {
        h.clear();
        return Status::Ok;
    }
    if h.start[0] != 0 {
        log_user!(log, LogType::Error, "Hessian has nonzero value (%d) for the start of column 0\n", h.start[0]);
        return Status::Error;
    }
    let dim = h.dim;
    let call = assess_matrix(log, "Hessian", dim, dim, false, &mut h.start, &[], &mut h.index, &mut h.value, 0.0, INF, true);
    return_status = log.interpret(call, return_status, "assessMatrix");
    if return_status == Status::Error {
        return return_status;
    }
    let call = normalise_hessian(log, h);
    return_status = log.interpret(call, return_status, "normaliseHessian");
    if return_status == Status::Error {
        return return_status;
    }
    let call = assess_matrix(
        log,
        "Hessian",
        dim,
        dim,
        false,
        &mut h.start,
        &[],
        &mut h.index,
        &mut h.value,
        small_matrix_value,
        large_matrix_value,
        false,
    );
    return_status = log.interpret(call, return_status, "assessMatrix");
    if return_status == Status::Error {
        return return_status;
    }
    let mut num_nz = h.num_nz();
    if num_nz != 0 {
        complete_hessian_diagonal(log, h);
        num_nz = h.num_nz();
    }
    let num_nz = num_nz as usize;
    if h.index.len() > num_nz {
        h.index.truncate(num_nz);
    }
    if h.value.len() > num_nz {
        h.value.truncate(num_nz);
    }
    // Any warning is not returned (as the C++)
    Status::Ok
}

/// completeHessianDiagonal: inserts explicit zeros for missing diagonal
/// entries (each column's diagonal entry first)
pub fn complete_hessian_diagonal(log: &Log, h: &mut Hessian) {
    let dim = h.dim;
    let num_nz = h.num_nz();
    let mut num_missing = 0;
    for col in 0..dim {
        let el = h.start[col as usize];
        if el < num_nz {
            if h.index[el as usize] != col {
                num_missing += 1;
            }
        } else {
            num_missing += 1;
        }
    }
    if num_missing > 0 {
        log_dev!(
            log,
            LogType::Info,
            "Hessian has dimension %d and %d nonzeros: inserting %d zeros onto the diagonal\n",
            dim,
            num_nz,
            num_missing
        );
    }
    if num_missing == 0 {
        return;
    }
    let new_num_nz = num_nz + num_missing;
    let mut to_el = new_num_nz;
    h.index.resize(new_num_nz as usize, 0);
    h.value.resize(new_num_nz as usize, 0.0);
    let mut next_start = num_nz;
    h.start[dim as usize] = to_el;
    for col in (0..dim).rev() {
        let c = col as usize;
        let mut el = next_start - 1;
        while el > h.start[c] {
            to_el -= 1;
            h.index[to_el as usize] = h.index[el as usize];
            h.value[to_el as usize] = h.value[el as usize];
            el -= 1;
        }
        let no_diagonal_entry = if h.start[c] < next_start {
            let el = h.start[c] as usize;
            to_el -= 1;
            h.index[to_el as usize] = h.index[el];
            h.value[to_el as usize] = h.value[el];
            h.index[el] != col
        } else {
            true
        };
        if no_diagonal_entry {
            to_el -= 1;
            h.index[to_el as usize] = col;
            h.value[to_el as usize] = 0.0;
        }
        next_start = h.start[c];
        h.start[c] = to_el;
    }
}

/// okHessianDiagonal: the diagonal entries (first in each column), signed
/// by the objective sense, must be nonnegative
pub fn ok_hessian_diagonal(log: &Log, dim: i32, start: &[i32], value: &[f64], sense: i32) -> bool {
    let mut min_diagonal_value = INF;
    let mut num_illegal = 0;
    for col in 0..dim as usize {
        let diagonal_value = sense as f64 * value[start[col] as usize];
        min_diagonal_value = cmin(diagonal_value, min_diagonal_value);
        if diagonal_value < 0.0 {
            num_illegal += 1;
        }
    }
    if num_illegal > 0 {
        if sense == 1 {
            log_user!(
                log,
                LogType::Error,
                "Hessian has %d diagonal entries in [%g, 0) so is not positive semidefinite for minimization\n",
                num_illegal,
                min_diagonal_value
            );
        } else {
            log_user!(
                log,
                LogType::Error,
                "Hessian has %d diagonal entries in (0, %g] so is not negative semidefinite for maximization\n",
                num_illegal,
                -min_diagonal_value
            );
        }
    }
    num_illegal == 0
}

/// extractTriangularHessian: drops the strict upper triangle (column-wise),
/// moving each diagonal entry first
pub fn extract_triangular_hessian(log: &Log, h: &mut Hessian) -> Status {
    let mut return_status = Status::Ok;
    let dim = h.dim as usize;
    let mut nnz = 0usize;
    for col in 0..dim {
        let nnz0 = nnz;
        for el in h.start[col] as usize..h.start[col + 1] as usize {
            let row = h.index[el];
            if (row as usize) < col {
                continue;
            }
            h.index[nnz] = row;
            h.value[nnz] = h.value[el];
            if row as usize == col && nnz > nnz0 {
                h.index[nnz] = h.index[nnz0];
                h.value[nnz] = h.value[nnz0];
                h.index[nnz0] = row;
                h.value[nnz0] = h.value[el];
            }
            nnz += 1;
        }
        h.start[col] = nnz0 as i32;
    }
    let num_ignored_nz = h.start[dim] - nnz as i32;
    if num_ignored_nz != 0 {
        if h.format == TRIANGULAR {
            log_user!(log, LogType::Warning, "Ignored %d entries of Hessian in opposite triangle\n", num_ignored_nz);
            return_status = Status::Warning;
        }
        h.start[dim] = nnz as i32;
    }
    h.format = TRIANGULAR;
    return_status
}

/// triangularToSquareHessian: the square matrix with each column's
/// entries in row order
pub fn triangular_to_square_hessian(h: &Hessian) -> (Vec<i32>, Vec<i32>, Vec<f64>) {
    let dim = h.dim;
    if dim <= 0 {
        return (vec![0], vec![], vec![]);
    }
    let dim = dim as usize;
    let nnz = h.start[dim] as usize;
    let square_nnz = nnz + (nnz - dim);
    let mut start = vec![0i32; dim + 1];
    let mut index = vec![0i32; square_nnz];
    let mut value = vec![0f64; square_nnz];
    let mut length = vec![0i32; dim];
    for col in 0..dim {
        length[col] += 1;
        for el in h.start[col] as usize + 1..h.start[col + 1] as usize {
            length[h.index[el] as usize] += 1;
            length[col] += 1;
        }
    }
    for col in 0..dim {
        start[col + 1] = start[col] + length[col];
    }
    for col in 0..dim {
        let el = h.start[col] as usize;
        let to = start[col] as usize;
        index[to] = h.index[el];
        value[to] = h.value[el];
        start[col] += 1;
        for el in h.start[col] as usize + 1..h.start[col + 1] as usize {
            let row = h.index[el] as usize;
            let to = start[row] as usize;
            index[to] = col as i32;
            value[to] = h.value[el];
            start[row] += 1;
            let to = start[col] as usize;
            index[to] = row as i32;
            value[to] = h.value[el];
            start[col] += 1;
        }
    }
    start[0] = 0;
    for col in 0..dim {
        start[col + 1] = start[col] + length[col];
    }
    (start, index, value)
}

/// normaliseHessian: to the lower triangle, summing a triangular
/// Hessian's upper entries into it and checking a square one's symmetry
/// (averaging entries within 1e-10)
pub fn normalise_hessian(log: &Log, h: &mut Hessian) -> Status {
    let dim = h.dim as usize;
    let triangular = h.format == TRIANGULAR;
    let square = !triangular;
    let mut upper_start: Vec<i32> = vec![0];
    let mut upper_index: Vec<i32> = Vec::new();
    let mut upper_value: Vec<f64> = Vec::new();
    let mut upper_length = vec![0i32; dim];
    let mut num_upper_nz = 0;
    let mut num_lower_nz = 0;
    for col in 0..dim {
        for el in h.start[col] as usize..h.start[col + 1] as usize {
            let row = h.index[el] as usize;
            if row < col {
                upper_length[row] += 1;
                num_upper_nz += 1;
            } else if row > col {
                num_lower_nz += 1;
            }
        }
    }
    if triangular && num_upper_nz == 0 {
        return Status::Ok;
    }
    if square && num_upper_nz != num_lower_nz {
        log_user!(
            log,
            LogType::Error,
            "Hessian has %d / %d lower / upper triangular entries so is not symmetric\n",
            num_lower_nz,
            num_upper_nz
        );
        return Status::Error;
    }
    if num_upper_nz > 0 {
        let mut num_lower_nz = 0;
        for row in 0..dim {
            upper_start.push(upper_start[row] + upper_length[row]);
            upper_length[row] = upper_start[row];
        }
        upper_index.resize(num_upper_nz as usize, 0);
        upper_value.resize(num_upper_nz as usize, 0.0);
        for col in 0..dim {
            let from_el = h.start[col] as usize;
            h.start[col] = num_lower_nz;
            for el in from_el..h.start[col + 1] as usize {
                let row = h.index[el] as usize;
                if row < col {
                    let u = upper_length[row] as usize;
                    upper_index[u] = col as i32;
                    upper_value[u] = h.value[el];
                    upper_length[row] += 1;
                } else {
                    h.index[num_lower_nz as usize] = row as i32;
                    h.value[num_lower_nz as usize] = h.value[el];
                    num_lower_nz += 1;
                }
            }
        }
        h.start[dim] = num_lower_nz;
    } else {
        upper_start.resize(dim + 1, 0);
    }
    // A triangular Hessian is gathered from a copy
    let copy = if triangular { Some((h.start.clone(), h.index.clone(), h.value.clone())) } else { None };

    let mut lower_on_below_diagonal = vec![0f64; dim];
    let mut upper_off_diagonal = vec![0f64; dim];
    let mut num_summation = 0;
    let mut num_upper_triangle = 0;
    let mut num_hessian_el = 0usize;
    const TOL: f64 = 1e-10;
    let mut num_illegal_asymmetry = 0;
    let mut min_illegal_asymmetry = INF;
    let mut max_illegal_asymmetry = 0.0;
    let mut num_ok_asymmetry = 0;
    let mut min_ok_asymmetry = INF;
    let mut max_ok_asymmetry = 0.0;
    // The source of the gathering: the copy, or (square) the Hessian
    // itself, overwritten behind (and, as the C++, possibly at) its reads
    let from_start = |h: &Hessian, i: usize| copy.as_ref().map_or(h.start[i], |c| c.0[i]) as usize;
    let from_index = |h: &Hessian, i: usize| copy.as_ref().map_or(h.index[i], |c| c.1[i]) as usize;
    let from_value = |h: &Hessian, i: usize| copy.as_ref().map_or(h.value[i], |c| c.2[i]);
    for col in 0..dim {
        let from_el = from_start(h, col);
        let mut el = from_el;
        while el < from_start(h, col + 1) {
            lower_on_below_diagonal[from_index(h, el)] = from_value(h, el);
            el += 1;
        }
        for el in upper_start[col] as usize..upper_start[col + 1] as usize {
            let row = upper_index[el] as usize;
            upper_off_diagonal[row] = upper_value[el];
            if square {
                let asymmetry = (upper_off_diagonal[row] - lower_on_below_diagonal[row]).abs();
                if asymmetry > TOL {
                    num_illegal_asymmetry += 1;
                    min_illegal_asymmetry = cmin(asymmetry, min_illegal_asymmetry);
                    max_illegal_asymmetry = cmax(asymmetry, max_illegal_asymmetry);
                } else if asymmetry != 0.0 {
                    num_ok_asymmetry += 1;
                    min_ok_asymmetry = cmin(asymmetry, min_ok_asymmetry);
                    max_ok_asymmetry = cmax(asymmetry, max_ok_asymmetry);
                    let average = (upper_off_diagonal[row] + lower_on_below_diagonal[row]) * 0.5;
                    upper_off_diagonal[row] = average;
                    lower_on_below_diagonal[row] = average;
                }
            } else {
                num_upper_triangle += 1;
                if lower_on_below_diagonal[row] != 0.0 {
                    num_summation += 1;
                }
                lower_on_below_diagonal[row] += upper_off_diagonal[row];
                upper_off_diagonal[row] = 0.0;
            }
        }
        // Gather the nonzeros: the diagonal, below it, then the entries
        // summed in from above
        h.start[col] = num_hessian_el as i32;
        if lower_on_below_diagonal[col] != 0.0 {
            h.index[num_hessian_el] = col as i32;
            h.value[num_hessian_el] = lower_on_below_diagonal[col];
            num_hessian_el += 1;
            lower_on_below_diagonal[col] = 0.0;
        }
        let mut el = from_el;
        while el < from_start(h, col + 1) {
            let row = from_index(h, el);
            if lower_on_below_diagonal[row] != 0.0 {
                h.index[num_hessian_el] = row as i32;
                h.value[num_hessian_el] = lower_on_below_diagonal[row];
                num_hessian_el += 1;
                lower_on_below_diagonal[row] = 0.0;
            }
            el += 1;
        }
        for el in upper_start[col] as usize..upper_start[col + 1] as usize {
            let row = upper_index[el] as usize;
            if lower_on_below_diagonal[row] != 0.0 {
                h.index[num_hessian_el] = row as i32;
                h.value[num_hessian_el] = lower_on_below_diagonal[row];
                num_hessian_el += 1;
                lower_on_below_diagonal[row] = 0.0;
            }
            upper_off_diagonal[row] = 0.0;
        }
    }
    h.start[dim] = num_hessian_el as i32;
    h.format = TRIANGULAR;
    h.index.resize(num_hessian_el, 0);
    h.value.resize(num_hessian_el, 0.0);

    let mut warning_found = false;
    let mut error_found = false;
    if num_ok_asymmetry != 0 {
        log_user!(
            log,
            LogType::Info,
            "Square Hessian contains %d non-symmetr%s in [%.2g, %.2g] within tolerance of %.1g\n",
            num_ok_asymmetry,
            if num_ok_asymmetry == 1 { "y" } else { "ies" },
            min_ok_asymmetry,
            max_ok_asymmetry,
            TOL
        );
    }
    if num_illegal_asymmetry != 0 {
        log_user!(
            log,
            LogType::Error,
            "Square Hessian contains %d non-symmetr%s in [%.2g, %.2g] exceeding tolerance of %.1g\n",
            num_illegal_asymmetry,
            if num_illegal_asymmetry == 1 { "y" } else { "ies" },
            min_illegal_asymmetry,
            max_illegal_asymmetry,
            TOL
        );
        error_found = true;
    }
    if num_upper_triangle != 0 {
        log_user!(
            log,
            LogType::Warning,
            "Triangular Hessian contains %d entr%s in upper triangle: added to lower triangle, requiring %d non-trivial summation%s\n",
            num_upper_triangle,
            if num_upper_triangle == 1 { "y" } else { "ies" },
            num_summation,
            if num_summation == 1 { "" } else { "s" }
        );
        warning_found = true;
    }
    if error_found {
        Status::Error
    } else if warning_found {
        Status::Warning
    } else {
        Status::Ok
    }
}

/// completeHessian: explicit zero diagonal entries up to full_dim
pub fn complete_hessian(full_dim: i32, h: &mut Hessian) {
    if h.dim == full_dim {
        return;
    }
    let mut nnz = h.num_nz();
    h.exact_resize();
    for col in h.dim..full_dim {
        h.index.push(col);
        h.value.push(0.0);
        nnz += 1;
        h.start.push(nnz);
    }
    h.dim = full_dim;
}

/// reportHessian
pub fn report_hessian(log: &Log, dim: i32, num_nz: i32, start: &[i32], index: &[i32], value: &[f64]) {
    if dim <= 0 {
        return;
    }
    log_user!(log, LogType::Info, "Hessian Index              Value\n");
    for col in 0..dim {
        log_user!(log, LogType::Info, "    %8d Start   %10d\n", col, start[col as usize]);
        let to_el = if col < dim - 1 { start[col as usize + 1] } else { num_nz };
        for el in start[col as usize]..to_el {
            log_user!(log, LogType::Info, "          %8d %12g\n", index[el as usize], value[el as usize]);
        }
    }
    log_user!(log, LogType::Info, "             Start   %10d\n", num_nz);
}

/// userScaleHessian
pub fn user_scale_hessian(dim: i32, start: &[i32], value: &mut [f64], d: &mut UserScaleData, apply: bool) {
    d.num_infinite_hessian_values = 0;
    if dim == 0 {
        return;
    }
    if d.user_objective_scale == 0 && d.user_bound_scale == 0 {
        return;
    }
    let objective_scale_value = 2f64.powf(d.user_objective_scale as f64);
    let bound_scale_value = 2f64.powf(-d.user_bound_scale as f64);
    for v in value[..start[dim as usize] as usize].iter_mut() {
        let value = *v * objective_scale_value * bound_scale_value;
        if value.abs() > d.infinite_cost {
            d.num_infinite_hessian_values += 1;
        }
        if apply {
            *v = value;
        }
    }
}

/// HighsHessian::deleteCols (of a triangular Hessian)
pub fn delete_cols(h: &mut Hessian, ic: &IndexCollection) {
    if h.dim == 0 {
        return;
    }
    let (from_k, to_k) = ic.limits();
    if from_k > to_k {
        return;
    }
    let dim = h.dim;
    let mut new_index = vec![-1i32; dim as usize];
    let mut new_dim = 0;
    let mut s = OutIn::new();
    for k in from_k..=to_k {
        update_out_in_index(ic, &mut s);
        if k == from_k {
            for col in 0..s.out_from {
                new_index[col as usize] = new_dim;
                new_dim += 1;
            }
        }
        for col in s.in_from..=s.in_to {
            new_index[col as usize] = new_dim;
            new_dim += 1;
        }
        if s.in_to >= dim - 1 {
            break;
        }
    }
    let mut s = OutIn::new();
    new_dim = 0;
    let mut new_num_nz = 0;
    let mut new_num_entries = 0usize;
    let save_start = h.start.clone();
    let mut keep_col = |h: &mut Hessian, col: i32, new_dim: &mut i32| {
        for el in save_start[col as usize] as usize..save_start[col as usize + 1] as usize {
            let row = new_index[h.index[el] as usize];
            if row < 0 {
                continue;
            }
            h.index[new_num_entries] = row;
            h.value[new_num_entries] = h.value[el];
            if h.value[new_num_entries] != 0.0 {
                new_num_nz += 1;
            }
            new_num_entries += 1;
        }
        *new_dim += 1;
        h.start[*new_dim as usize] = new_num_entries as i32;
    };
    for k in from_k..=to_k {
        update_out_in_index(ic, &mut s);
        if k == from_k {
            for col in 0..s.out_from {
                keep_col(h, col, &mut new_dim);
            }
        }
        for col in s.in_from..=s.in_to {
            keep_col(h, col, &mut new_dim);
        }
        if s.in_to >= dim - 1 {
            break;
        }
    }
    h.dim = new_dim;
    if new_num_nz == 0 {
        h.clear();
    } else {
        h.exact_resize();
    }
}

/// HighsHessian::scaleOk
pub fn scale_ok(dim: i32, start: &[i32], value: &[f64], hessian_scale: i32, small: f64, large: f64) -> bool {
    if dim == 0 {
        return true;
    }
    let scale_value = 2f64.powf(hessian_scale as f64);
    for &v in &value[..start[dim as usize] as usize] {
        let abs_new_value = (v * scale_value).abs();
        if abs_new_value >= large || abs_new_value <= small {
            return false;
        }
    }
    true
}

/// A read-only HighsHessian (HighsRust.h: rsHessianView)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HessianView {
    pub dim: i32,
    pub format: i32,
    pub start: *const i32,
    pub index: *const i32,
    pub value: *const f64,
}

impl HessianView {
    fn parts(&self) -> (usize, &[i32], &[i32], &[f64]) {
        let dim = self.dim.max(0) as usize;
        if dim == 0 {
            return (0, &[], &[], &[]);
        }
        // SAFETY: the C++ HighsHessian's arrays: start has dim + 1
        // entries, index and value start[dim]
        unsafe {
            let start = std::slice::from_raw_parts(self.start, dim + 1);
            let nnz = start[dim] as usize;
            (
                dim,
                start,
                std::slice::from_raw_parts(self.index, nnz),
                std::slice::from_raw_parts(self.value, nnz),
            )
        }
    }
}

/// HighsHessian::product (product has dim entries, assigned 0 first)
pub fn product(h: &HessianView, x: &[f64], product: &mut [f64]) {
    let (dim, start, index, value) = h.parts();
    let triangular = h.format == TRIANGULAR;
    product[..dim].fill(0.0);
    for col in 0..dim {
        for el in start[col] as usize..start[col + 1] as usize {
            let row = index[el] as usize;
            product[row] = value[el].mul_add_c(x[col], product[row]);
            if triangular && row != col {
                product[col] = value[el].mul_add_c(x[row], product[col]);
            }
        }
    }
}

/// HighsHessian::alphaProductPlusY
pub fn alpha_product_plus_y(h: &HessianView, alpha: f64, x: &[f64], y: &mut [f64]) {
    let (dim, start, index, value) = h.parts();
    let triangular = h.format == TRIANGULAR;
    for col in 0..dim {
        for el in start[col] as usize..start[col + 1] as usize {
            let row = index[el] as usize;
            y[row] = (alpha * value[el]).mul_add_c(x[col], y[row]);
            if triangular && row != col {
                y[col] = (alpha * value[el]).mul_add_c(x[row], y[col]);
            }
        }
    }
}

/// HighsHessian::objectiveValue (triangular, diagonal first)
pub fn objective_value(h: &HessianView, x: &[f64]) -> f64 {
    let (dim, start, index, value) = h.parts();
    let mut f = 0.0;
    for col in 0..dim {
        let el = start[col] as usize;
        f = (0.5 * x[col] * value[el]).mul_add_c(x[col], f);
        for el in el + 1..start[col + 1] as usize {
            f = (x[col] * value[el]).mul_add_c(x[index[el] as usize], f);
        }
    }
    f
}

/// HighsHessian::objectiveCDoubleValue
pub fn objective_cdouble_value(h: &HessianView, x: &[f64]) -> CDouble {
    let (dim, start, index, value) = h.parts();
    let mut f = CDouble::from(0.0);
    for col in 0..dim {
        let el = start[col] as usize;
        f += 0.5 * x[col] * value[el] * x[col];
        for el in el + 1..start[col + 1] as usize {
            f += x[col] * value[el] * x[index[el] as usize];
        }
    }
    f
}

/// HighsHessian::isDiagonal
pub fn is_diagonal(h: &HessianView) -> bool {
    let (dim, start, index, _) = h.parts();
    (0..dim).all(|col| (start[col] as usize..start[col + 1] as usize).all(|el| index[el] as usize == col))
}

/// HighsHessian::toSquare of a triangular Hessian: each column's diagonal
/// first, then its entries below and above in the order met
pub fn to_square(h: &Hessian) -> Hessian {
    let dim = h.dim as usize;
    let mut iwork = vec![0i32; dim];
    for col in 0..dim {
        for el in h.start[col] as usize + 1..h.start[col + 1] as usize {
            iwork[h.index[el] as usize] += 1;
        }
    }
    let mut sq = Hessian { dim: h.dim, format: SQUARE, start: vec![0], index: vec![], value: vec![] };
    for col in 0..dim {
        let col_nnz = iwork[col] + h.start[col + 1] - h.start[col];
        sq.start.push(sq.start[col] + col_nnz);
        iwork[col] = sq.start[col] + 1;
    }
    let nnz = sq.start[dim] as usize;
    sq.index.resize(nnz, 0);
    sq.value.resize(nnz, 0.0);
    for col in 0..dim {
        let el = h.start[col] as usize;
        let sq_el = sq.start[col] as usize;
        sq.index[sq_el] = col as i32;
        sq.value[sq_el] = h.value[el];
        for el in el + 1..h.start[col + 1] as usize {
            let row = h.index[el] as usize;
            let v = h.value[el];
            let e = iwork[col] as usize;
            sq.index[e] = row as i32;
            sq.value[e] = v;
            iwork[col] += 1;
            let e = iwork[row] as usize;
            sq.index[e] = col as i32;
            sq.value[e] = v;
            iwork[row] += 1;
        }
    }
    sq
}

/// HighsHessian::print (to C's stdout)
pub fn print(h: &Hessian, message: &str) {
    let mut s = sprintf(
        "%s Hessian of dimension %d and %d entries: %s\n",
        &[(if h.format == TRIANGULAR { "Triangular" } else { "Square" }).into(), h.dim.into(), h.num_nz().into(), message.into()],
    );
    s += &sprintf(
        "Start; Index; Value of sizes %d; %d; %d\n",
        &[(h.start.len() as i32).into(), (h.index.len() as i32).into(), (h.value.len() as i32).into()],
    );
    if h.dim > 0 {
        let dim = h.dim as usize;
        s += " Row|";
        for col in 0..dim {
            s += &sprintf(" %4d", &[(col as i32).into()]);
        }
        s += "\n-----";
        s += &"-----".repeat(dim);
        s += "\n";
        let mut cells = vec![String::new(); dim];
        for col in 0..dim {
            let els = h.start[col] as usize..h.start[col + 1] as usize;
            for el in els.clone() {
                cells[h.index[el] as usize] = sprintf("%4g", &[h.value[el].into()]);
            }
            s += &sprintf("%4d|", &[(col as i32).into()]);
            for cell in &cells {
                s += &sprintf(" %4s", &[cell.as_str().into()]);
            }
            s += "\n";
            for el in els {
                cells[h.index[el] as usize].clear();
            }
        }
    }
    crate::io::log::c_stdout(s.as_bytes());
}

// ------------------------------------------------------------ entry points

/// HighsRust.h: RsHessianOptions, what the assessment reads
#[repr(C)]
pub struct CHessianOptions {
    pub log: Log,
    pub small_matrix_value: f64,
    pub large_matrix_value: f64,
}

/// What a C++ HighsHessian method or utility asks of Rust (codes of
/// highs_rs_hessian)
#[repr(i32)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // made by transmute from the C++ code
enum Call {
    Assess = 0,
    AssessDimensions,
    Complete,
    ExtractTriangular,
    Normalise,
    CompleteDiagonal,
    Print,
    ToSquare,
}

/// The HighsHessian utilities that change a Hessian: Rust works on a copy
/// and stores it back (`arg` is completeHessian's full_dim; `out` the
/// toSquare result; `msg` print's message)
///
/// # Safety
/// The views are the C++ objects'; `msg` holds `msg_len` bytes
#[no_mangle]
pub unsafe extern "C" fn highs_rs_hessian(
    call: i32,
    h: *mut CHessian,
    o: *const CHessianOptions,
    arg: i32,
    out: *mut CHessian,
    msg: *const u8,
    msg_len: usize,
) -> i32 {
    let mut x = (*h).load();
    let log = if o.is_null() { Log::none() } else { (*o).log };
    let call = std::mem::transmute::<i32, Call>(call);
    let status = match call {
        Call::Assess => assess_hessian(&log, &mut x, (*o).small_matrix_value, (*o).large_matrix_value),
        Call::AssessDimensions => return assess_hessian_dimensions(&log, &x) as i32,
        Call::Complete => {
            complete_hessian(arg, &mut x);
            Status::Ok
        }
        Call::ExtractTriangular => extract_triangular_hessian(&log, &mut x),
        Call::Normalise => normalise_hessian(&log, &mut x),
        Call::CompleteDiagonal => {
            complete_hessian_diagonal(&log, &mut x);
            Status::Ok
        }
        Call::Print => {
            let m = if msg_len == 0 { &[][..] } else { std::slice::from_raw_parts(msg, msg_len) };
            print(&x, std::str::from_utf8_unchecked(m));
            return 0;
        }
        Call::ToSquare => {
            if x.format == SQUARE {
                (*out).store(&x);
            } else {
                (*out).store(&to_square(&x));
            }
            return 0;
        }
    };
    (*h).store(&x);
    status as i32
}

/// HighsHessian::deleteCols
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
pub unsafe extern "C" fn highs_rs_hessian_delete_cols(h: *mut CHessian, ic: *const CIndexCollection) {
    let mut x = (*h).load();
    delete_cols(&mut x, &(*ic).view());
    (*h).store(&x);
}

/// triangularToSquareHessian into the three vectors
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
pub unsafe extern "C" fn highs_rs_triangular_to_square_hessian(
    h: HessianView,
    start: *mut RsVec<i32>,
    index: *mut RsVec<i32>,
    value: *mut RsVec<f64>,
) {
    let (dim, s, i, v) = h.parts();
    let x = Hessian { dim: dim as i32, format: h.format, start: s.to_vec(), index: i.to_vec(), value: v.to_vec() };
    let (s, i, v) = triangular_to_square_hessian(&x);
    if dim == 0 {
        // The C++ assigns only start
        (*start).assign(&s);
        return;
    }
    (*start).resize(s.len());
    (*start).as_mut_slice().copy_from_slice(&s);
    (*index).resize(i.len());
    (*index).as_mut_slice().copy_from_slice(&i);
    (*value).resize(v.len());
    (*value).as_mut_slice().copy_from_slice(&v);
}

/// okHessianDiagonal
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
pub unsafe extern "C" fn highs_rs_ok_hessian_diagonal(log: *const Log, h: HessianView, sense: i32) -> bool {
    let dim = h.dim.max(0) as usize;
    if dim == 0 {
        return true;
    }
    let start = std::slice::from_raw_parts(h.start, dim + 1);
    let value = std::slice::from_raw_parts(h.value, start[dim] as usize);
    ok_hessian_diagonal(&*log, h.dim, start, value, sense)
}

/// reportHessian
///
/// # Safety
/// The arrays hold dim + 1 starts and num_nz entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_report_hessian(
    log: *const Log,
    dim: i32,
    num_nz: i32,
    start: *const i32,
    index: *const i32,
    value: *const f64,
) {
    if dim <= 0 {
        return;
    }
    let nnz = num_nz.max(0) as usize;
    let start = std::slice::from_raw_parts(start, dim as usize);
    let index = if nnz == 0 { &[][..] } else { std::slice::from_raw_parts(index, nnz) };
    let value = if nnz == 0 { &[][..] } else { std::slice::from_raw_parts(value, nnz) };
    report_hessian(&*log, dim, num_nz, start, index, value)
}

/// userScaleHessian
///
/// # Safety
/// The views are the C++ objects'
#[no_mangle]
pub unsafe extern "C" fn highs_rs_user_scale_hessian(h: HessianView, d: *mut UserScaleData, apply: bool) {
    let (dim, start, _, value) = h.parts();
    let value = std::slice::from_raw_parts_mut(value.as_ptr() as *mut f64, value.len());
    user_scale_hessian(dim as i32, start, value, &mut *d, apply)
}

/// HighsHessian::scaleOk
///
/// # Safety
/// The view is the C++ object's
#[no_mangle]
pub unsafe extern "C" fn highs_rs_hessian_scale_ok(h: HessianView, scale: i32, small: f64, large: f64) -> bool {
    let (dim, start, _, value) = h.parts();
    scale_ok(dim as i32, start, value, scale, small, large)
}

/// HighsHessian::product, alphaProductPlusY, objectiveValue (codes 0, 1,
/// 2) and isDiagonal (3)
///
/// # Safety
/// The view is the C++ object's; x and y hold at least dim entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_hessian_numeric(
    call: i32,
    h: HessianView,
    alpha: f64,
    x: *const f64,
    y: *mut f64,
) -> f64 {
    let dim = h.dim.max(0) as usize;
    let xs = if dim == 0 { &[][..] } else { std::slice::from_raw_parts(x, dim) };
    match call {
        0 => product(&h, xs, std::slice::from_raw_parts_mut(y, dim)),
        1 => alpha_product_plus_y(&h, alpha, xs, std::slice::from_raw_parts_mut(y, dim)),
        2 => return objective_value(&h, xs),
        _ => return is_diagonal(&h) as i32 as f64,
    }
    0.0
}

/// HighsHessian::objectiveCDoubleValue: (hi, lo) into out
///
/// # Safety
/// The view is the C++ object's; x holds dim entries
#[no_mangle]
pub unsafe extern "C" fn highs_rs_hessian_objective_cdouble(h: HessianView, x: *const f64, out: *mut CDouble) {
    let dim = h.dim.max(0) as usize;
    let xs = if dim == 0 { &[][..] } else { std::slice::from_raw_parts(x, dim) };
    *out = objective_cdouble_value(&h, xs);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(dim: i32, format: i32, start: &[i32], index: &[i32], value: &[f64]) -> Hessian {
        Hessian { dim, format, start: start.to_vec(), index: index.to_vec(), value: value.to_vec() }
    }

    #[test]
    fn normalise_square_and_triangular() {
        // [[2, 1], [1, 3]] square, column-wise with the diagonal second
        let mut x = h(2, SQUARE, &[0, 2, 4], &[1, 0, 0, 1], &[1.0, 2.0, 1.0, 3.0]);
        assert_eq!(normalise_hessian(&Log::none(), &mut x), Status::Ok);
        assert_eq!(x, h(2, TRIANGULAR, &[0, 2, 3], &[0, 1, 1], &[2.0, 1.0, 3.0]));
        // Upper entry of a triangular Hessian summed into the lower one
        let mut x = h(2, TRIANGULAR, &[0, 1, 3], &[0, 0, 1], &[2.0, 1.0, 3.0]);
        assert_eq!(normalise_hessian(&Log::none(), &mut x), Status::Warning);
        assert_eq!(x, h(2, TRIANGULAR, &[0, 2, 3], &[0, 1, 1], &[2.0, 1.0, 3.0]));
    }

    #[test]
    fn complete_diagonal_and_square() {
        let mut x = h(3, TRIANGULAR, &[0, 1, 1, 2], &[1, 2], &[5.0, 7.0]);
        complete_hessian_diagonal(&Log::none(), &mut x);
        assert_eq!(x, h(3, TRIANGULAR, &[0, 2, 3, 4], &[0, 1, 1, 2], &[0.0, 5.0, 0.0, 7.0]));
        let (s, i, v) = triangular_to_square_hessian(&x);
        assert_eq!((s, i, v), (vec![0, 2, 4, 5], vec![0, 1, 0, 1, 2], vec![0.0, 5.0, 5.0, 0.0, 7.0]));
        let sq = to_square(&x);
        assert_eq!(sq.start, vec![0, 2, 4, 5]);
        assert_eq!(sq.index, vec![0, 1, 1, 0, 2]);
    }
}
