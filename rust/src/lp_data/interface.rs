//! The model modification interfaces of HighsInterface.cpp: adding and
//! deleting columns and rows, changing costs, bounds, integrality and a
//! matrix coefficient, and scaling a column or row. They edit an LP and a
//! HiGHS basis generic over their vectors ([`LpG`], [`BasisG`]): a C++
//! HighsLp and HighsBasis in place (resized by C++, see HighsRust.h:
//! RsLpVec), or Rust-owned ones. What lies outside the LP and basis (the
//! names, the model status, solution and info, the simplex engine's
//! status and basis, the Hessian) is reached through [`IfaceHost`].

use super::edit::{
    append_nonbasic_cols, change_bounds, change_costs, change_integrality, change_matrix_coefficient,
    delete_basis_entries, delete_from_vectors, delete_scale, set_nonbasic_status,
};
use super::lp::LpG;
use super::lp_utils::{assess_bounds, assess_costs, assess_matrix, IndexCollection};
use super::report::apply_scaling_to_lp;
use super::sparse::{Buf, SparseMatrix};
use super::{matrix_format, var_type, Log, Status};
use crate::simplex::lp_solver::LpSolver;
use crate::util::sort::{increasing_set_ok_int, sort_set_permutation};

// HighsBasisStatus
const LOWER: u8 = 0;
const BASIC: u8 = 1;
const UPPER: u8 = 2;
// LpAction
const LP_NEW_COSTS: i32 = 1;
const LP_NEW_BOUNDS: i32 = 2;
const LP_NEW_COLS: i32 = 4;
const LP_NEW_ROWS: i32 = 5;
const LP_DEL_COLS: i32 = 6;
const LP_SCALED_COL: i32 = 10;
const LP_SCALED_ROW: i32 = 11;

/// HighsBasis over status vectors `S` (the debug origin name stays with
/// its owner)
#[repr(C)]
#[derive(Clone, Debug, PartialEq)]
pub struct BasisG<S> {
    pub valid: bool,
    pub alien: bool,
    pub useful: bool,
    pub was_alien: bool,
    pub debug_id: i32,
    pub debug_update_count: i32,
    pub col_status: S,
    pub row_status: S,
}

/// The option values the interfaces read
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IfaceOptions {
    pub log: Log,
    pub infinite_cost: f64,
    pub infinite_bound: f64,
    pub small_matrix_value: f64,
    pub large_matrix_value: f64,
    pub allowed_matrix_scale_factor: i32,
}

/// The invalidations of the Highs object
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Invalidate {
    /// invalidateModelStatusSolutionAndInfo
    StatusSolutionAndInfo = 0,
    /// invalidateModelStatusAndInfo
    StatusAndInfo = 1,
    /// invalidateModelStatus
    Status = 2,
    /// model_status_ = kNotset
    StatusNotset = 3,
}

/// What the interfaces reach outside the LP and basis
pub trait IfaceHost {
    /// The simplex engine (no borrow is held across a host call)
    #[allow(clippy::mut_from_ref)]
    fn lps(&self) -> &mut LpSolver;
    /// The names of the columns (`cols`) or rows, if there are any, sized
    /// to `num`, new ones blank
    fn names_resize(&mut self, cols: bool, num: i32);
    /// The names of the kept entries moved to the front (kept[new] = old)
    /// and the names sized to new_num, if there are any
    fn names_delete(&mut self, cols: bool, kept: &[i32], new_num: i32);
    /// Clear the name hash of the columns or rows
    fn names_hash_clear(&mut self, cols: bool);
    fn invalidate(&mut self, what: Invalidate);
    /// Highs::feasibleWrtBounds(columns)
    fn feasible_wrt_bounds(&self, columns: bool) -> bool;
    /// What HEkk::addCols does but the status update: setNlaLp(lp) if
    /// there is a simplex NLA
    fn ekk_nla_lp(&mut self);
    /// The C++ HEkk shell's clear (after the engine cleared itself)
    fn ekk_clear_shell(&mut self);
    /// completeHessian(num_col) if the Hessian is not empty
    fn hessian_complete(&mut self, num_col: i32);
    /// HighsHessian::deleteCols
    fn hessian_delete_cols(&mut self, ic: &IndexCollection);
}

/// HEkk::updateStatus
fn ekk_update_status(h: &mut impl IfaceHost, action: i32) {
    if h.lps().update_status(action) {
        h.ekk_clear_shell();
    }
}

fn interval(dim: i32) -> IndexCollection<'static> {
    IndexCollection::interval(dim, 0, dim - 1)
}

/// HighsSparseMatrix::applyRowScale / applyColScale of a matrix with the
/// given scale factors
fn apply_scale_factors(m: &mut SparseMatrix, row: Option<&[f64]>, col: Option<&[f64]>) {
    let colwise = m.is_colwise();
    let num_vec = if colwise { m.num_col } else { m.num_row } as usize;
    for v in 0..num_vec {
        for el in m.start[v] as usize..m.start[v + 1] as usize {
            let o = m.index[el] as usize;
            let (c, r) = if colwise { (v, o) } else { (o, v) };
            if let Some(s) = row {
                m.value[el] *= s[r];
            }
            if let Some(s) = col {
                m.value[el] *= s[c];
            }
        }
    }
}

/// HighsSparseMatrix::considerColScaling / considerRowScaling: scale
/// factors (powers of two in the allowed range) of the columns of a
/// column-wise or the rows of a row-wise matrix, applied to it
pub fn consider_scaling(m: &mut SparseMatrix, max_scale_factor_exponent: i32, scale: &mut [f64]) {
    let log2 = 2f64.ln();
    let max_allow_scale = 2f64.powi(max_scale_factor_exponent);
    let min_allow_scale = 1.0 / max_allow_scale;
    let num_vec = if m.is_colwise() { m.num_col } else { m.num_row } as usize;
    for v in 0..num_vec {
        let (from, to) = (m.start[v] as usize, m.start[v + 1] as usize);
        let mut max_value = 0.0f64;
        for &x in &m.value[from..to] {
            max_value = cmax(x.abs(), max_value);
        }
        if max_value != 0.0 {
            let mut scale_value = 1.0 / max_value;
            scale_value = 2f64.powf(((scale_value.ln() / log2) + 0.5).floor());
            scale_value = cmin(cmax(min_allow_scale, scale_value), max_allow_scale);
            scale[v] = scale_value;
            for x in &mut m.value[from..to] {
                *x *= scale[v];
            }
        } else {
            scale[v] = 1.0;
        }
    }
}

/// std::max (the first if neither is greater)
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// std::min
fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Highs::addColsInterface after the null data checks
#[allow(clippy::too_many_arguments)]
pub fn add_cols<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>, S: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    basis: &mut BasisG<S>,
    h: &mut impl IfaceHost,
    o: &IfaceOptions,
    num_new_col: i32,
    cost: &[f64],
    lower: &[f64],
    upper: &[f64],
    num_new_nz: i32,
    start: &[i32],
    index: &[i32],
    value: &[f64],
) -> Status {
    let log = &o.log;
    let mut return_status = Status::Ok;
    if lp.num_row <= 0 && num_new_nz > 0 {
        return Status::Error;
    }
    let num_col = lp.num_col as usize;
    let n = num_new_col as usize;
    let new_num_col = num_col + n;
    let ic = interval(num_new_col);
    let mut local_cost = cost[..n].to_vec();
    let mut local_lower = lower[..n].to_vec();
    let mut local_upper = upper[..n].to_vec();
    let mut local_has_infinite_cost = false;
    return_status = log.interpret(
        assess_costs(log, &ic, &mut local_cost, &mut local_has_infinite_cost, o.infinite_cost),
        return_status,
        "assessCosts",
    );
    if return_status == Status::Error {
        return return_status;
    }
    return_status = log.interpret(
        assess_bounds(log, "Col", lp.num_col, &ic, &mut local_lower, &mut local_upper, o.infinite_bound, None),
        return_status,
        "assessBounds",
    );
    if return_status == Status::Error {
        return return_status;
    }
    // appendColsToLpVectors
    lp.col_cost.resize(new_num_col);
    lp.col_lower.resize(new_num_col);
    lp.col_upper.resize(new_num_col);
    let have_integrality = !lp.integrality.is_empty();
    if have_integrality {
        lp.integrality.resize(new_num_col);
    }
    h.names_resize(true, new_num_col as i32);
    for k in 0..n {
        lp.col_cost.sl_mut()[num_col + k] = local_cost[k];
        lp.col_lower.sl_mut()[num_col + k] = local_lower[k];
        lp.col_upper.sl_mut()[num_col + k] = local_upper[k];
        if have_integrality {
            lp.integrality.sl_mut()[num_col + k] = var_type::CONTINUOUS;
        }
    }
    // The new columns, column-wise
    let mut local = SparseMatrix {
        format: matrix_format::COLWISE,
        num_col: num_new_col,
        num_row: lp.num_row,
        start: Vec::new(),
        p_end: Vec::new(),
        index: Vec::new(),
        value: Vec::new(),
    };
    if num_new_nz != 0 {
        local.start = start[..n].to_vec();
        local.start.push(num_new_nz);
        local.index = index[..num_new_nz as usize].to_vec();
        local.value = value[..num_new_nz as usize].to_vec();
        return_status = log.interpret(
            assess_matrix(
                log,
                "LP",
                local.num_row,
                local.num_col,
                false,
                &mut local.start,
                &[],
                &mut local.index,
                &mut local.value,
                o.small_matrix_value,
                o.large_matrix_value,
                false,
            ),
            return_status,
            "assessMatrix",
        );
        if return_status == Status::Error {
            return return_status;
        }
    } else {
        local.start = vec![0; n + 1];
    }
    lp.a.add_cols(&local.start, &local.index, &local.value, num_new_col);
    if lp.scale.has_scaling {
        lp.scale.col.resize(new_num_col);
        for k in 0..n {
            lp.scale.col.sl_mut()[num_col + k] = 1.0;
        }
        lp.scale.num_col = new_num_col as i32;
        apply_scale_factors(&mut local, Some(lp.scale.row.sl()), None);
        consider_scaling(&mut local, o.allowed_matrix_scale_factor, &mut lp.scale.col.sl_mut()[num_col..]);
    }
    if basis.useful {
        append_nonbasic_cols_to_basis(lp, basis, h, num_new_col);
    }
    lp.num_col += num_new_col;
    lp.has_infinite_cost = lp.has_infinite_cost || local_has_infinite_cost;
    h.invalidate(Invalidate::StatusSolutionAndInfo);
    // HEkk::addCols
    if h.lps().sh.status.has_nla {
        h.ekk_nla_lp();
    }
    ekk_update_status(h, LP_NEW_COLS);
    h.hessian_complete(lp.num_col);
    return_status
}

/// Highs::appendNonbasicColsToBasisInterface (for a useful basis)
fn append_nonbasic_cols_to_basis<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>, S: Buf<u8>>(
    lp: &LpG<F, I, U>,
    basis: &mut BasisG<S>,
    h: &mut impl IfaceHost,
    num_new_col: i32,
) {
    if num_new_col == 0 {
        return;
    }
    let lps = h.lps();
    let has_simplex_basis = lps.sh.status.has_basis;
    let (num_col, num_row) = (lp.num_col as usize, lp.num_row as usize);
    let new_num_col = num_col + num_new_col as usize;
    basis.col_status.resize(new_num_col);
    if has_simplex_basis {
        lps.resize_basis(new_num_col + num_row);
    }
    let b = &mut lps.basis;
    let (flag, mv, basic_index): (&mut [i8], &mut [i8], &mut [i32]) = if has_simplex_basis {
        (&mut b.nonbasic_flag, &mut b.nonbasic_move, &mut b.basic_index)
    } else {
        (&mut [], &mut [], &mut [])
    };
    append_nonbasic_cols(
        num_col,
        num_row,
        num_new_col as usize,
        basis.col_status.sl_mut(),
        lp.col_lower.sl(),
        lp.col_upper.sl(),
        flag,
        mv,
        basic_index,
    );
}

/// Highs::addRowsInterface after the null data checks
#[allow(clippy::too_many_arguments)]
pub fn add_rows<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>, S: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    basis: &mut BasisG<S>,
    h: &mut impl IfaceHost,
    o: &IfaceOptions,
    num_new_row: i32,
    lower: &[f64],
    upper: &[f64],
    num_new_nz: i32,
    start: &[i32],
    index: &[i32],
    value: &[f64],
) -> Status {
    let log = &o.log;
    let mut return_status = Status::Ok;
    if lp.num_col <= 0 && num_new_nz > 0 {
        return Status::Error;
    }
    let num_row = lp.num_row as usize;
    let n = num_new_row as usize;
    let new_num_row = num_row + n;
    let ic = interval(num_new_row);
    let mut local_lower = lower[..n].to_vec();
    let mut local_upper = upper[..n].to_vec();
    return_status = log.interpret(
        assess_bounds(log, "Row", lp.num_row, &ic, &mut local_lower, &mut local_upper, o.infinite_bound, None),
        return_status,
        "assessBounds",
    );
    if return_status == Status::Error {
        return return_status;
    }
    // appendRowsToLpVectors
    lp.row_lower.resize(new_num_row);
    lp.row_upper.resize(new_num_row);
    h.names_resize(false, new_num_row as i32);
    for k in 0..n {
        lp.row_lower.sl_mut()[num_row + k] = local_lower[k];
        lp.row_upper.sl_mut()[num_row + k] = local_upper[k];
    }
    // The new rows, row-wise
    let mut local = SparseMatrix {
        format: matrix_format::ROWWISE,
        num_col: lp.num_col,
        num_row: num_new_row,
        start: Vec::new(),
        p_end: Vec::new(),
        index: Vec::new(),
        value: Vec::new(),
    };
    if num_new_nz != 0 {
        local.start = start[..n].to_vec();
        local.start.push(num_new_nz);
        local.index = index[..num_new_nz as usize].to_vec();
        local.value = value[..num_new_nz as usize].to_vec();
        return_status = log.interpret(
            assess_matrix(
                log,
                "LP",
                local.num_col,
                local.num_row,
                false,
                &mut local.start,
                &[],
                &mut local.index,
                &mut local.value,
                o.small_matrix_value,
                o.large_matrix_value,
                false,
            ),
            return_status,
            "assessMatrix",
        );
        if return_status == Status::Error {
            return return_status;
        }
    } else {
        local.start = vec![0; n + 1];
    }
    lp.a.add_rows(&local.start, &local.index, &local.value, num_new_row);
    if lp.scale.has_scaling {
        lp.scale.row.resize(new_num_row);
        for k in 0..n {
            lp.scale.row.sl_mut()[num_row + k] = 1.0;
        }
        lp.scale.num_row = new_num_row as i32;
        apply_scale_factors(&mut local, None, Some(lp.scale.col.sl()));
        consider_scaling(&mut local, o.allowed_matrix_scale_factor, &mut lp.scale.row.sl_mut()[num_row..]);
    }
    if basis.useful {
        // appendBasicRowsToBasisInterface
        basis.row_status.resize(new_num_row);
        for s in &mut basis.row_status.sl_mut()[num_row..] {
            *s = BASIC;
        }
        let lps = h.lps();
        if lps.sh.status.has_basis {
            lps.append_basic_rows(lp.num_col, lp.num_row, new_num_row as i32);
        }
    }
    lp.num_row += num_new_row;
    h.invalidate(Invalidate::StatusSolutionAndInfo);
    // HEkk::addRows: new rows come in with basic logicals, which leaves
    // the DSE weights of the existing rows unchanged
    let lps = h.lps();
    lps.lp.num_row = lp.num_row;
    lps.add_rows(lp.num_row - num_new_row, lp.num_row);
    h.ekk_clear_shell();
    return_status
}

/// The entries of the collection are deleted from a basis' statuses:
/// whether a basic and whether a nonbasic entry was deleted
fn delete_basis_statuses<S: Buf<u8>>(status: &mut S, ic: &IndexCollection) -> (bool, bool) {
    let (from_k, to_k) = ic.limits();
    if from_k > to_k {
        return (false, false);
    }
    let (n, deleted_basic, deleted_nonbasic) = delete_basis_entries(status.sl_mut(), ic);
    status.resize(n);
    (deleted_basic, deleted_nonbasic)
}

/// HighsLp::deleteColsFromVectors / deleteRowsFromVectors and the matrix
/// part of HighsLp::deleteCols / deleteRows
fn delete_from_lp<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    h: &mut impl IfaceHost,
    ic: &IndexCollection,
    cols: bool,
) {
    let dim = if cols { lp.num_col } else { lp.num_row };
    let mut kept = vec![0i32; dim as usize];
    let new_num = {
        let mut empty: Vec<u8> = Vec::new();
        if cols {
            let integrality = lp.integrality.sl_mut();
            delete_from_vectors(
                ic,
                &mut [lp.col_cost.sl_mut(), lp.col_lower.sl_mut(), lp.col_upper.sl_mut()],
                integrality,
                &mut kept,
            )
        } else {
            delete_from_vectors(ic, &mut [lp.row_lower.sl_mut(), lp.row_upper.sl_mut()], &mut empty, &mut kept)
        }
    };
    let (from_k, to_k) = ic.limits();
    if from_k <= to_k {
        if cols {
            lp.col_cost.resize(new_num);
            lp.col_lower.resize(new_num);
            lp.col_upper.resize(new_num);
            if !lp.integrality.is_empty() {
                lp.integrality.resize(new_num);
            }
        } else {
            lp.row_lower.resize(new_num);
            lp.row_upper.resize(new_num);
        }
        h.names_delete(cols, &kept, new_num as i32);
    }
    if cols {
        lp.a.delete_cols(ic);
        lp.num_col = new_num as i32;
    } else {
        lp.a.delete_rows(ic);
        lp.num_row = new_num as i32;
    }
}

/// Highs::deleteColsInterface: whether a mask collection is to be set to
/// the new indices (renumber_mask)
pub fn delete_cols<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>, S: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    basis: &mut BasisG<S>,
    h: &mut impl IfaceHost,
    ic: &IndexCollection,
) -> bool {
    lp.a.ensure_colwise();
    let original_num_col = lp.num_col;
    delete_from_lp(lp, h, ic, true);
    h.hessian_delete_cols(ic);
    if lp.num_col == original_num_col {
        return false;
    }
    h.invalidate(Invalidate::StatusNotset);
    if basis.useful {
        let (deleted_basic, _) = delete_basis_statuses(&mut basis.col_status, ic);
        if deleted_basic {
            basis.valid = false;
        }
    }
    if lp.scale.has_scaling {
        delete_scale(lp.scale.col.sl_mut(), ic);
        lp.scale.col.resize(lp.num_col as usize);
        lp.scale.num_col = lp.num_col;
    }
    h.invalidate(Invalidate::StatusSolutionAndInfo);
    ekk_update_status(h, LP_DEL_COLS);
    h.names_hash_clear(true);
    ic.is_mask
}

/// The mask of a deletion set to the new indices of the kept entries (-1
/// for deleted ones)
pub fn renumber_mask(mask: &mut [i32]) {
    let mut new_ix = 0;
    for m in mask {
        if *m == 0 {
            *m = new_ix;
            new_ix += 1;
        } else {
            *m = -1;
        }
    }
}

/// Highs::deleteRowsInterface: whether a mask collection is to be set to
/// the new indices (renumber_mask)
pub fn delete_rows<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>, S: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    basis: &mut BasisG<S>,
    h: &mut impl IfaceHost,
    ic: &IndexCollection,
) -> bool {
    lp.a.ensure_colwise();
    let original_num_row = lp.num_row;
    delete_from_lp(lp, h, ic, false);
    if lp.num_row == original_num_row {
        return false;
    }
    h.invalidate(Invalidate::StatusNotset);
    if basis.useful {
        let (_, deleted_nonbasic) = delete_basis_statuses(&mut basis.row_status, ic);
        if deleted_nonbasic {
            basis.valid = false;
        }
    }
    if lp.scale.has_scaling {
        delete_scale(lp.scale.row.sl_mut(), ic);
        lp.scale.row.resize(lp.num_row as usize);
        lp.scale.num_row = lp.num_row;
    }
    h.invalidate(Invalidate::StatusSolutionAndInfo);
    // HEkk::deleteRows: deleting rows with basic logicals leaves the DSE
    // weights of the remaining rows unchanged
    h.lps().delete_rows(ic);
    h.ekk_clear_shell();
    h.names_hash_clear(false);
    ic.is_mask
}

/// Highs::changeIntegralityInterface for a positive number of entries
pub fn change_integrality_iface<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    h: &mut impl IfaceHost,
    ic: &IndexCollection,
    integrality: &[u8],
) -> Status {
    if ic.is_set {
        debug_assert!(increasing_set_ok_int(&ic.set[..ic.set_num_entries as usize], 0, ic.dimension, true));
    }
    // changeLpIntegrality
    let (from_k, to_k) = ic.limits();
    if from_k <= to_k {
        if lp.integrality.is_empty() {
            lp.integrality.assign(lp.num_col as usize, var_type::CONTINUOUS);
        }
        change_integrality(lp.integrality.sl_mut(), ic, integrality);
        // HighsLp::isMip
        let is_mip = lp.integrality.sl().iter().any(|&t| t != var_type::CONTINUOUS);
        if !is_mip {
            lp.integrality.clear();
        }
    }
    h.invalidate(Invalidate::Status);
    Status::Ok
}

/// HighsLp::hasInfiniteCost
fn has_infinite_cost(cost: &[f64], infinite_cost: f64) -> bool {
    cost.iter().any(|&c| c >= infinite_cost || c <= -infinite_cost)
}

/// Highs::changeCostsInterface for a positive number of costs
pub fn change_costs_iface<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    h: &mut impl IfaceHost,
    o: &IfaceOptions,
    ic: &IndexCollection,
    cost: &[f64],
) -> Status {
    let log = &o.log;
    let mut local_cost = cost.to_vec();
    let mut local_has_infinite_cost = false;
    let return_status = log.interpret(
        assess_costs(log, ic, &mut local_cost, &mut local_has_infinite_cost, o.infinite_cost),
        Status::Ok,
        "assessCosts",
    );
    if return_status == Status::Error {
        return return_status;
    }
    // changeLpCosts
    let (from_k, to_k) = ic.limits();
    if from_k <= to_k {
        change_costs(lp.col_cost.sl_mut(), ic, &local_cost);
        if lp.has_infinite_cost {
            lp.has_infinite_cost = has_infinite_cost(&lp.col_cost.sl()[..lp.num_col as usize], o.infinite_cost);
        }
    }
    lp.has_infinite_cost = lp.has_infinite_cost || local_has_infinite_cost;
    h.invalidate(Invalidate::StatusSolutionAndInfo);
    ekk_update_status(h, LP_NEW_COSTS);
    Status::Ok
}

/// sortSetData for the bounds of a set: the set sorted and the bounds
/// gathered into its order
pub fn sort_set_bounds(set: &mut [i32], lower: &[f64], upper: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut local_lower = lower.to_vec();
    let mut local_upper = upper.to_vec();
    if !set.is_empty() {
        let perm = sort_set_permutation(set);
        for (k, &p) in perm.iter().enumerate() {
            local_lower[k] = lower[p as usize];
            local_upper[k] = upper[p as usize];
        }
    }
    (local_lower, local_upper)
}

/// Highs::changeColBoundsInterface / changeRowBoundsInterface for a
/// positive number of bounds, after the set of a set collection has been
/// sorted with the bounds (sort_set_bounds)
#[allow(clippy::too_many_arguments)]
pub fn change_bounds_iface<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>, S: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    basis: &mut BasisG<S>,
    h: &mut impl IfaceHost,
    o: &IfaceOptions,
    ic: &IndexCollection,
    columns: bool,
    mut local_lower: Vec<f64>,
    mut local_upper: Vec<f64>,
) -> Status {
    let log = &o.log;
    let return_status = log.interpret(
        assess_bounds(
            log,
            if columns { "col" } else { "row" },
            0,
            ic,
            &mut local_lower,
            &mut local_upper,
            o.infinite_bound,
            None,
        ),
        Status::Ok,
        "assessBounds",
    );
    if return_status == Status::Error {
        return return_status;
    }
    let (from_k, to_k) = ic.limits();
    if from_k <= to_k {
        if columns {
            change_bounds(lp.col_lower.sl_mut(), lp.col_upper.sl_mut(), ic, &local_lower, &local_upper);
        } else {
            change_bounds(lp.row_lower.sl_mut(), lp.row_upper.sl_mut(), ic, &local_lower, &local_upper);
        }
    }
    // setNonbasicStatusInterface
    if basis.valid {
        let lps = h.lps();
        let has_simplex_basis = lps.sh.status.has_basis;
        let b = &mut lps.basis;
        let (flag, mv): (&mut [i8], &mut [i8]) = if has_simplex_basis {
            (&mut b.nonbasic_flag, &mut b.nonbasic_move)
        } else {
            (&mut [], &mut [])
        };
        let (status, lo, up, offset) = if columns {
            (basis.col_status.sl_mut(), lp.col_lower.sl(), lp.col_upper.sl(), 0)
        } else {
            (basis.row_status.sl_mut(), lp.row_lower.sl(), lp.row_upper.sl(), lp.num_col as usize)
        };
        set_nonbasic_status(ic, columns, status, lo, up, flag, mv, offset);
    }
    if !basis.useful && h.feasible_wrt_bounds(columns) {
        h.invalidate(Invalidate::StatusAndInfo);
    } else {
        h.invalidate(Invalidate::StatusSolutionAndInfo);
    }
    ekk_update_status(h, LP_NEW_BOUNDS);
    Status::Ok
}

/// Highs::changeCoefficientInterface
#[allow(clippy::too_many_arguments)]
pub fn change_coefficient<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>, S: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    basis: &mut BasisG<S>,
    h: &mut impl IfaceHost,
    o: &IfaceOptions,
    row: i32,
    col: i32,
    new_value: f64,
) {
    lp.a.ensure_colwise();
    let zero_new_value = new_value.abs() <= o.small_matrix_value;
    // changeLpMatrixCoefficient: room for an inserted entry (the C++ only
    // resizes when inserting)
    let a = &mut lp.a;
    let old_size = a.index.len();
    let num_nz = a.start.sl()[lp.num_col as usize] as usize;
    a.index.resize(old_size.max(num_nz + 1));
    a.value.resize(old_size.max(num_nz + 1));
    let new_num_nz = change_matrix_coefficient(
        a.start.sl_mut(),
        a.index.sl_mut(),
        a.value.sl_mut(),
        lp.num_col as usize,
        row,
        col as usize,
        new_value,
        zero_new_value,
    );
    let size = if new_num_nz < 0 { old_size } else { new_num_nz as usize };
    a.index.resize(size);
    a.value.resize(size);
    // (the C++ reads the status without checking that the basis has one)
    let basic_column = basis.col_status.sl().get(col as usize).copied() == Some(BASIC);
    h.invalidate(Invalidate::StatusSolutionAndInfo);
    if basic_column {
        basis.was_alien = true;
        basis.alien = true;
    }
    ekk_update_status(h, LP_NEW_ROWS);
}

/// Highs::scaleColInterface / scaleRowInterface
pub fn scale_col_row<F: Buf<f64>, I: Buf<i32>, U: Buf<u8>, S: Buf<u8>>(
    lp: &mut LpG<F, I, U>,
    basis: &mut BasisG<S>,
    h: &mut impl IfaceHost,
    log: &Log,
    is_col: bool,
    ix: i32,
    scale_value: f64,
) -> Status {
    lp.a.ensure_colwise();
    let dim = if is_col { lp.num_col } else { lp.num_row };
    if ix < 0 || ix >= dim || scale_value == 0.0 {
        return Status::Error;
    }
    let mut v = lp.view();
    let call = apply_scaling_to_lp(&mut v, is_col, ix, scale_value);
    let return_status = log.interpret(call, Status::Ok, if is_col { "applyScalingToLpCol" } else { "applyScalingToLpRow" });
    if return_status == Status::Error {
        return return_status;
    }
    if scale_value < 0.0 && basis.valid {
        let s = &mut if is_col { basis.col_status.sl_mut() } else { basis.row_status.sl_mut() }[ix as usize];
        if *s == LOWER {
            *s = UPPER;
        } else if *s == UPPER {
            *s = LOWER;
        }
    }
    let lps = h.lps();
    if lps.sh.status.initialised_for_solve && scale_value < 0.0 && lps.sh.status.has_basis {
        let var = if is_col { ix } else { lp.num_col + ix } as usize;
        lps.flip_nonbasic_move(var);
    }
    h.invalidate(Invalidate::StatusSolutionAndInfo);
    ekk_update_status(h, if is_col { LP_SCALED_COL } else { LP_SCALED_ROW });
    Status::Ok
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scaling_factors_are_powers_of_two() {
        let mut m = SparseMatrix {
            format: matrix_format::COLWISE,
            num_col: 2,
            num_row: 2,
            start: vec![0, 2, 2],
            p_end: vec![],
            index: vec![0, 1],
            value: vec![3.0, -0.5],
        };
        let mut s = [0.0; 2];
        consider_scaling(&mut m, 20, &mut s);
        assert_eq!(s, [0.25, 1.0]);
        assert_eq!(m.value, vec![0.75, -0.125]);
    }
}

// The C++ host and entry points
pub mod ffi {
    use super::*;
    use crate::lp_data::ffi::{CIndexCollection, RsVec};
    use crate::lp_data::lp::CppLp;
    use std::ffi::c_void;

    pub type CppBasis = BasisG<RsVec<u8>>;

    /// The C++ side of the interfaces (HighsInterface.cpp)
    #[repr(C)]
    pub struct CIfaceHost {
        pub ctx: *mut c_void,
        /// (ctx, op, arg, p, n) -> result
        pub op: unsafe extern "C" fn(*mut c_void, i32, i32, *const c_void, i32) -> i32,
        pub lps: *mut LpSolver,
    }

    const OP_NAMES_RESIZE: i32 = 1;
    const OP_NAMES_DELETE: i32 = 2;
    const OP_NAMES_HASH_CLEAR: i32 = 3;
    const OP_INVALIDATE: i32 = 4;
    const OP_FEASIBLE_WRT_BOUNDS: i32 = 5;
    const OP_EKK_NLA_LP: i32 = 6;
    const OP_EKK_CLEAR_SHELL: i32 = 7;
    const OP_HESSIAN_COMPLETE: i32 = 8;
    const OP_HESSIAN_DELETE_COLS: i32 = 9;

    impl CIfaceHost {
        fn op(&self, code: i32, arg: i32, p: *const c_void, n: i32) -> i32 {
            // SAFETY: the C++ op with its context
            unsafe { (self.op)(self.ctx, code, arg, p, n) }
        }
    }

    impl IfaceHost for CIfaceHost {
        fn lps(&self) -> &mut LpSolver {
            // SAFETY: the C++ HEkk's engine, borrowed between host calls
            unsafe { &mut *self.lps }
        }
        fn names_resize(&mut self, cols: bool, num: i32) {
            self.op(OP_NAMES_RESIZE, cols as i32, std::ptr::null(), num);
        }
        fn names_delete(&mut self, cols: bool, kept: &[i32], new_num: i32) {
            self.op(OP_NAMES_DELETE, cols as i32, kept.as_ptr() as *const c_void, new_num);
        }
        fn names_hash_clear(&mut self, cols: bool) {
            self.op(OP_NAMES_HASH_CLEAR, cols as i32, std::ptr::null(), 0);
        }
        fn invalidate(&mut self, what: Invalidate) {
            self.op(OP_INVALIDATE, what as i32, std::ptr::null(), 0);
        }
        fn feasible_wrt_bounds(&self, columns: bool) -> bool {
            self.op(OP_FEASIBLE_WRT_BOUNDS, columns as i32, std::ptr::null(), 0) != 0
        }
        fn ekk_nla_lp(&mut self) {
            self.op(OP_EKK_NLA_LP, 0, std::ptr::null(), 0);
        }
        fn ekk_clear_shell(&mut self) {
            self.op(OP_EKK_CLEAR_SHELL, 0, std::ptr::null(), 0);
        }
        fn hessian_complete(&mut self, num_col: i32) {
            self.op(OP_HESSIAN_COMPLETE, num_col, std::ptr::null(), 0);
        }
        fn hessian_delete_cols(&mut self, ic: &IndexCollection) {
            let set = ic.set.as_ptr();
            let mask = ic.mask.as_ptr();
            let c = CIndexCollection {
                dimension: ic.dimension,
                is_interval: ic.is_interval,
                from: ic.from,
                to: ic.to,
                is_set: ic.is_set,
                set_num_entries: ic.set_num_entries,
                set: crate::lp_data::ffi::RsMut { ptr: set as *mut i32, len: ic.set.len() },
                is_mask: ic.is_mask,
                mask: crate::lp_data::ffi::RsMut { ptr: mask as *mut i32, len: ic.mask.len() },
            };
            self.op(OP_HESSIAN_DELETE_COLS, 0, &c as *const CIndexCollection as *const c_void, 0);
        }
    }

    /// The data of an interface call
    #[repr(C)]
    pub struct CIfaceCall {
        pub host: CIfaceHost,
        pub lp: *mut CppLp,
        pub basis: *mut CppBasis,
        pub o: IfaceOptions,
        pub ic: CIndexCollection,
    }

    /// Highs::addColsInterface (`num` columns) or addRowsInterface after
    /// the null data checks: `x0` the costs (columns) and `x1`, `x2` the
    /// bounds
    ///
    /// # Safety
    /// The views valid, the data of their lengths
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_iface_add(
        c: *mut CIfaceCall,
        cols: bool,
        num: i32,
        x0: *const f64,
        x1: *const f64,
        x2: *const f64,
        num_nz: i32,
        start: *const i32,
        index: *const i32,
        value: *const f64,
    ) -> i32 {
        use crate::ffi::sl;
        let c = &mut *c;
        let (lp, basis) = (&mut *c.lp, &mut *c.basis);
        let nz = num_nz.max(0);
        let (start, index, value) = if nz > 0 { (sl(start, num), sl(index, nz), sl(value, nz)) } else { (&[][..], &[][..], &[][..]) };
        if cols {
            add_cols(lp, basis, &mut c.host, &c.o, num, sl(x0, num), sl(x1, num), sl(x2, num), num_nz, start, index, value)
                as i32
        } else {
            add_rows(lp, basis, &mut c.host, &c.o, num, sl(x1, num), sl(x2, num), num_nz, start, index, value) as i32
        }
    }

    /// Highs::deleteColsInterface (`cols`) or deleteRowsInterface
    ///
    /// # Safety
    /// As highs_rs_iface_add
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_iface_delete(c: *mut CIfaceCall, cols: bool) {
        let c = &mut *c;
        let (lp, basis) = (&mut *c.lp, &mut *c.basis);
        let dim = if cols { lp.num_col } else { lp.num_row } as usize;
        let renumber = {
            let ic = c.ic.view();
            if cols {
                delete_cols(lp, basis, &mut c.host, &ic)
            } else {
                delete_rows(lp, basis, &mut c.host, &ic)
            }
        };
        if renumber {
            renumber_mask(&mut c.ic.mask.get_mut()[..dim]);
        }
    }

    /// Highs::changeCostsInterface (what 0), changeColBoundsInterface (1),
    /// changeRowBoundsInterface (2) or changeIntegralityInterface (3, `x0`
    /// the integrality bytes) for a positive number of entries
    ///
    /// # Safety
    /// As highs_rs_iface_add
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_iface_change(
        c: *mut CIfaceCall,
        what: i32,
        num: i32,
        x0: *const c_void,
        x1: *const f64,
    ) -> i32 {
        use crate::ffi::sl;
        let c = &mut *c;
        let (lp, basis) = (&mut *c.lp, &mut *c.basis);
        match what {
            0 => change_costs_iface(lp, &mut c.host, &c.o, &c.ic.view(), sl(x0 as *const f64, num)) as i32,
            1 | 2 => {
                let (lower, upper) = (sl(x0 as *const f64, num), sl(x1, num));
                let (local_lower, local_upper) = if c.ic.is_set {
                    let n = c.ic.set_num_entries.max(0) as usize;
                    sort_set_bounds(&mut c.ic.set.get_mut()[..n], lower, upper)
                } else {
                    (lower.to_vec(), upper.to_vec())
                };
                let ic = c.ic.view();
                change_bounds_iface(lp, basis, &mut c.host, &c.o, &ic, what == 1, local_lower, local_upper) as i32
            }
            _ => change_integrality_iface(lp, &mut c.host, &c.ic.view(), sl(x0 as *const u8, num)) as i32,
        }
    }

    /// Highs::changeCoefficientInterface
    ///
    /// # Safety
    /// As highs_rs_iface_add
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_iface_change_coefficient(c: *mut CIfaceCall, row: i32, col: i32, value: f64) {
        let c = &mut *c;
        change_coefficient(&mut *c.lp, &mut *c.basis, &mut c.host, &c.o, row, col, value);
    }

    /// Highs::scaleColInterface (`is_col`) or scaleRowInterface
    ///
    /// # Safety
    /// As highs_rs_iface_add
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_iface_scale(c: *mut CIfaceCall, is_col: bool, ix: i32, scale: f64) -> i32 {
        let c = &mut *c;
        let log = c.o.log;
        scale_col_row(&mut *c.lp, &mut *c.basis, &mut c.host, &log, is_col, ix, scale) as i32
    }
}
