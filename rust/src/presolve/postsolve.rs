//! The undo side of HighsPostsolveStack (highs/presolve/HighsPostsolveStack.cpp).
//!
//! HPresolve (C++) records the reductions: each is a record struct, then any
//! vectors of nonzeros, pushed as raw bytes on a HighsDataStack
//! (highs/util/HighsDataStack.h), with its type in a list of reductions.
//! The C++ class keeps owning that byte stack and the index maps; Rust reads
//! them through `Stack` and undoes the reductions on the solution and basis.
//! The record structs below are `#[repr(C)]` copies of the C++ ones
//! (HighsInt is 32 bits, RowType an int, HighsBasisStatus and bool a byte),
//! so a record is read back with the bytes the C++ pushed.

use crate::util::fma::ClangFma;

use crate::util::cdouble::CDouble;
use crate::util::printf::{sprintf, Arg};
use std::mem::size_of;

const INF: f64 = f64::INFINITY;

// HighsBasisStatus
pub const LOWER: u8 = 0;
pub const BASIC: u8 = 1;
pub const UPPER: u8 = 2;
pub const ZERO: u8 = 3;
pub const NONBASIC: u8 = 4;

// RowType
pub(crate) const GEQ: i32 = 0;
pub(crate) const LEQ: i32 = 1;
pub(crate) const EQ: i32 = 2;

// ReductionType
pub(crate) const LINEAR_TRANSFORM: u8 = 0;
pub(crate) const FREE_COL_SUBSTITUTION: u8 = 1;
pub(crate) const DOUBLETON_EQUATION: u8 = 2;
pub(crate) const EQUALITY_ROW_ADDITION: u8 = 3;
pub(crate) const EQUALITY_ROW_ADDITIONS: u8 = 4;
pub(crate) const SINGLETON_ROW: u8 = 5;
pub(crate) const FIXED_COL: u8 = 6;
pub(crate) const REDUNDANT_ROW: u8 = 7;
pub(crate) const FORCING_ROW: u8 = 8;
pub(crate) const FORCING_COLUMN: u8 = 9;
pub(crate) const FORCING_COLUMN_REMOVED_ROW: u8 = 10;
pub(crate) const DUPLICATE_ROW: u8 = 11;
pub(crate) const DUPLICATE_COLUMN: u8 = 12;
pub(crate) const SLACK_COL_SUBSTITUTION: u8 = 13;

/// A plain-old-data record: `#[repr(C)]`, only integer and float fields, so
/// every byte pattern is a valid value.
///
/// # Safety
/// Implement only for such types.
unsafe trait Pod: Copy {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Nonzero {
    pub index: i32,
    pub value: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct LinearTransform {
    pub(crate) scale: f64,
    pub(crate) constant: f64,
    pub(crate) col: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FreeColSubstitution {
    pub(crate) rhs: f64,
    pub(crate) col_cost: f64,
    pub(crate) row: i32,
    pub(crate) col: i32,
    pub(crate) row_type: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DoubletonEquation {
    pub(crate) coef: f64,
    pub(crate) coef_subst: f64,
    pub(crate) rhs: f64,
    pub(crate) subst_lower: f64,
    pub(crate) subst_upper: f64,
    pub(crate) subst_cost: f64,
    pub(crate) row: i32,
    pub(crate) col_subst: i32,
    pub(crate) col: i32,
    pub(crate) lower_tightened: u8,
    pub(crate) upper_tightened: u8,
    pub(crate) row_type: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct EqualityRowAddition {
    pub(crate) row: i32,
    pub(crate) added_eq_row: i32,
    pub(crate) eq_row_scale: f64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct EqualityRowAdditions {
    pub(crate) added_eq_row: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SingletonRow {
    pub(crate) coef: f64,
    pub(crate) row: i32,
    pub(crate) col: i32,
    pub(crate) col_lower_tightened: u8,
    pub(crate) col_upper_tightened: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct FixedCol {
    pub(crate) fix_value: f64,
    pub(crate) col_cost: f64,
    pub(crate) col: i32,
    pub(crate) fix_type: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RedundantRow {
    pub(crate) row: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ForcingRow {
    pub(crate) side: f64,
    pub(crate) row: i32,
    pub(crate) row_type: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ForcingColumn {
    pub(crate) col_cost: f64,
    pub(crate) col_bound: f64,
    pub(crate) col: i32,
    pub(crate) at_infinite_upper: u8,
    pub(crate) col_integral: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct ForcingColumnRemovedRow {
    pub(crate) rhs: f64,
    pub(crate) row: i32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct DuplicateRow {
    pub(crate) duplicate_row_scale: f64,
    pub(crate) duplicate_row: i32,
    pub(crate) row: i32,
    pub(crate) row_lower_tightened: u8,
    pub(crate) row_upper_tightened: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DuplicateColumn {
    pub(crate) col_scale: f64,
    pub(crate) col_lower: f64,
    pub(crate) col_upper: f64,
    pub(crate) duplicate_col_lower: f64,
    pub(crate) duplicate_col_upper: f64,
    pub(crate) col: i32,
    pub(crate) duplicate_col: i32,
    pub(crate) col_integral: u8,
    pub(crate) duplicate_col_integral: u8,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct SlackColSubstitution {
    pub(crate) rhs: f64,
    pub(crate) row: i32,
    pub(crate) col: i32,
}

// SAFETY: all are #[repr(C)] with integer and float fields only
unsafe impl Pod for Nonzero {}
unsafe impl Pod for LinearTransform {}
unsafe impl Pod for FreeColSubstitution {}
unsafe impl Pod for DoubletonEquation {}
unsafe impl Pod for EqualityRowAddition {}
unsafe impl Pod for EqualityRowAdditions {}
unsafe impl Pod for SingletonRow {}
unsafe impl Pod for FixedCol {}
unsafe impl Pod for RedundantRow {}
unsafe impl Pod for ForcingRow {}
unsafe impl Pod for ForcingColumn {}
unsafe impl Pod for ForcingColumnRemovedRow {}
unsafe impl Pod for DuplicateRow {}
unsafe impl Pod for DuplicateColumn {}
unsafe impl Pod for SlackColSubstitution {}
unsafe impl Pod for usize {}

// The sizes of the C++ structs (checked there too by static_assert)
const _: () = {
    assert!(size_of::<Nonzero>() == 16);
    assert!(size_of::<LinearTransform>() == 24);
    assert!(size_of::<FreeColSubstitution>() == 32);
    assert!(size_of::<DoubletonEquation>() == 72);
    assert!(size_of::<EqualityRowAddition>() == 16);
    assert!(size_of::<EqualityRowAdditions>() == 4);
    assert!(size_of::<SingletonRow>() == 24);
    assert!(size_of::<FixedCol>() == 24);
    assert!(size_of::<RedundantRow>() == 4);
    assert!(size_of::<ForcingRow>() == 16);
    assert!(size_of::<ForcingColumn>() == 24);
    assert!(size_of::<ForcingColumnRemovedRow>() == 16);
    assert!(size_of::<DuplicateRow>() == 24);
    assert!(size_of::<DuplicateColumn>() == 56);
    assert!(size_of::<SlackColSubstitution>() == 16);
};

/// std::pair<ReductionType, size_t>
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Reduction {
    pub kind: u8,
    pub position: usize,
}

/// HighsDataStack::pop, reading backwards from `pos`
struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn pop<T: Pod>(&mut self) -> T {
        self.pos -= size_of::<T>();
        let b = &self.data[self.pos..self.pos + size_of::<T>()];
        // SAFETY: b has size_of::<T>() bytes and T: Pod accepts any bytes
        unsafe { std::ptr::read_unaligned(b.as_ptr() as *const T) }
    }

    fn pop_vec<T: Pod>(&mut self, v: &mut Vec<T>) {
        let n: usize = self.pop();
        v.clear();
        if n == 0 {
            return;
        }
        let bytes = n * size_of::<T>();
        self.pos -= bytes;
        let b = &self.data[self.pos..self.pos + bytes];
        v.reserve(n);
        // SAFETY: b has n * size_of::<T>() bytes, v room for n Ts, T: Pod
        unsafe {
            std::ptr::copy_nonoverlapping(b.as_ptr(), v.as_mut_ptr() as *mut u8, bytes);
            v.set_len(n);
        }
    }
}

/// The recorded reductions (owned by the C++ HighsPostsolveStack)
pub struct Stack<'a> {
    pub data: &'a [u8],
    pub reductions: &'a [Reduction],
    pub orig_col_index: &'a [i32],
    pub orig_row_index: &'a [i32],
}

/// The tolerances of HighsOptions used by postsolve
#[repr(C)]
pub struct Tolerances {
    pub primal_feasibility: f64,
    pub dual_feasibility: f64,
    pub mip_feasibility: f64,
}

/// HighsSolution's vectors, already sized to the original space (the duals
/// only if dual_valid)
pub struct Solution<'a> {
    pub col_value: &'a mut [f64],
    pub row_value: &'a mut [f64],
    pub col_dual: &'a mut [f64],
    pub row_dual: &'a mut [f64],
    pub dual_valid: bool,
}

impl Solution<'_> {
    #[inline]
    fn is_model_row(&self, row: i32) -> bool {
        (row as usize) < self.row_value.len()
    }
}

/// HighsBasis's status vectors, sized to the original space if valid
pub struct Basis<'a> {
    pub col_status: &'a mut [u8],
    pub row_status: &'a mut [u8],
    pub valid: bool,
}

fn c_print(s: &str) {
    extern "C" {
        fn printf(fmt: *const std::ffi::c_char, ...) -> i32;
    }
    let s = std::ffi::CString::new(s).unwrap();
    // SAFETY: both are NUL-terminated strings
    unsafe {
        printf(c"%s".as_ptr(), s.as_ptr());
    }
}

/// HighsUtils fractionality
#[inline]
fn fractionality(v: f64) -> f64 {
    (v - v.round()).abs()
}

fn undo_iterate_backwards<T: Copy>(values: &mut [T], index: &[i32]) {
    for i in (0..index.len()).rev() {
        values[index[i] as usize] = values[i];
    }
}

/// HighsPostsolveStack::undo (until = 0) and undoUntil: undoes the
/// reductions after the first `until`, after expanding the solution and
/// basis to the original index space
pub fn undo(
    stack: &Stack,
    tol: &Tolerances,
    sol: &mut Solution,
    basis: &mut Basis,
    until: usize,
    report_col: i32,
) {
    undo_iterate_backwards(sol.col_value, stack.orig_col_index);
    undo_iterate_backwards(sol.row_value, stack.orig_row_index);
    if sol.dual_valid {
        undo_iterate_backwards(sol.col_dual, stack.orig_col_index);
        undo_iterate_backwards(sol.row_dual, stack.orig_row_index);
    }
    if basis.valid {
        undo_iterate_backwards(basis.col_status, stack.orig_col_index);
        undo_iterate_backwards(basis.row_status, stack.orig_row_index);
    }

    let mut rd = Reader { data: stack.data, pos: stack.data.len() };
    let mut row_values: Vec<Nonzero> = Vec::new();
    let mut col_values: Vec<Nonzero> = Vec::new();
    for i in (until..stack.reductions.len()).rev() {
        let kind = stack.reductions[i].kind;
        if report_col >= 0 {
            c_print(&sprintf(
                "Before  reduction %2d (type %2d): col_value[%2d] = %g\n",
                &[
                    Arg::I(i as i64),
                    Arg::I(kind as i64),
                    Arg::I(report_col as i64),
                    Arg::F(sol.col_value[report_col as usize]),
                ],
            ));
        }
        match kind {
            LINEAR_TRANSFORM => {
                let r: LinearTransform = rd.pop();
                r.undo(sol);
            }
            FREE_COL_SUBSTITUTION => {
                rd.pop_vec(&mut col_values);
                rd.pop_vec(&mut row_values);
                let r: FreeColSubstitution = rd.pop();
                r.undo(&row_values, &col_values, sol, basis);
            }
            DOUBLETON_EQUATION => {
                rd.pop_vec(&mut col_values);
                let r: DoubletonEquation = rd.pop();
                r.undo(tol, &col_values, sol, basis);
            }
            EQUALITY_ROW_ADDITION => {
                rd.pop_vec(&mut row_values);
                let r: EqualityRowAddition = rd.pop();
                r.undo(sol);
            }
            EQUALITY_ROW_ADDITIONS => {
                rd.pop_vec(&mut col_values);
                rd.pop_vec(&mut row_values);
                let r: EqualityRowAdditions = rd.pop();
                r.undo(&col_values, sol);
            }
            SINGLETON_ROW => {
                let r: SingletonRow = rd.pop();
                r.undo(tol, sol, basis);
            }
            FIXED_COL => {
                rd.pop_vec(&mut col_values);
                let r: FixedCol = rd.pop();
                r.undo(&col_values, sol, basis);
            }
            REDUNDANT_ROW => {
                let r: RedundantRow = rd.pop();
                r.undo(sol, basis);
            }
            FORCING_ROW => {
                rd.pop_vec(&mut row_values);
                let r: ForcingRow = rd.pop();
                r.undo(&row_values, sol, basis);
            }
            FORCING_COLUMN => {
                rd.pop_vec(&mut col_values);
                let r: ForcingColumn = rd.pop();
                r.undo(tol, &col_values, sol, basis);
            }
            FORCING_COLUMN_REMOVED_ROW => {
                rd.pop_vec(&mut row_values);
                let r: ForcingColumnRemovedRow = rd.pop();
                r.undo(&row_values, sol, basis);
            }
            DUPLICATE_ROW => {
                let r: DuplicateRow = rd.pop();
                r.undo(tol, sol, basis);
            }
            DUPLICATE_COLUMN => {
                let r: DuplicateColumn = rd.pop();
                r.undo(tol, sol, basis);
            }
            SLACK_COL_SUBSTITUTION => {
                rd.pop_vec(&mut row_values);
                let r: SlackColSubstitution = rd.pop();
                r.undo(&row_values, sol, basis);
            }
            _ => unreachable!("reduction type {kind}"),
        }
    }
    if report_col >= 0 {
        c_print(&sprintf(
            "After last reduction: col_value[%2d] = %g\n",
            &[Arg::I(report_col as i64), Arg::F(sol.col_value[report_col as usize])],
        ));
    }
}

/// HighsPostsolveStack::getReducedPrimalSolution on a copy of the original
/// solution; the first orig_col_index.len() entries are the result
pub fn reduced_primal_solution(stack: &Stack, sol: &mut [f64]) {
    for red in stack.reductions {
        let mut rd = Reader { data: stack.data, pos: red.position };
        match red.kind {
            DUPLICATE_COLUMN => {
                let r: DuplicateColumn = rd.pop();
                let (c, d) = (r.col as usize, r.duplicate_col as usize);
                // fused by clang
                sol[c] = r.col_scale.mul_add_c(sol[d], sol[c]);
            }
            LINEAR_TRANSFORM => {
                let r: LinearTransform = rd.pop();
                let c = r.col as usize;
                sol[c] -= r.constant;
                sol[c] /= r.scale;
            }
            _ => {}
        }
    }
    for (i, &j) in stack.orig_col_index.iter().enumerate() {
        sol[i] = sol[j as usize];
    }
}

/// HighsPostsolveStack::compressIndexMaps; returns the new sizes of the maps
pub fn compress_index_maps(
    orig_row_index: &mut [i32],
    orig_col_index: &mut [i32],
    new_row_index: &[i32],
    new_col_index: &[i32],
) -> (usize, usize) {
    fn compress(orig: &mut [i32], new: &[i32]) -> usize {
        let mut n = orig.len();
        for (i, &k) in new.iter().enumerate() {
            if k == -1 {
                n -= 1;
            } else {
                orig[k as usize] = orig[i];
            }
        }
        n
    }
    (compress(orig_row_index, new_row_index), compress(orig_col_index, new_col_index))
}

fn compute_row_status(dual: f64, row_type: i32) -> u8 {
    if row_type == EQ {
        if dual < 0.0 {
            UPPER
        } else {
            LOWER
        }
    } else if row_type == GEQ {
        LOWER
    } else {
        UPPER
    }
}

/// computeStatus with a basis status to keep when the dual is zero
fn compute_status_keep(dual: f64, status: &mut u8, tol: f64) -> u8 {
    if dual > tol {
        *status = LOWER;
    } else if dual < -tol {
        *status = UPPER;
    }
    *status
}

fn compute_status(dual: f64, tol: f64) -> u8 {
    if dual > tol {
        LOWER
    } else if dual < -tol {
        UPPER
    } else {
        BASIC
    }
}

impl LinearTransform {
    fn undo(&self, sol: &mut Solution) {
        let c = self.col as usize;
        sol.col_value[c] *= self.scale;
        sol.col_value[c] += self.constant;
        if sol.dual_valid {
            sol.col_dual[c] /= self.scale;
        }
    }
}

/// The value of `col` making the row hold with `rhs`: sets the row value if
/// the row is in the model; returns the coefficient of col
fn substitute_col(row_values: &[Nonzero], row: i32, col: i32, rhs: f64, sol: &mut Solution) -> f64 {
    let mut col_coef = 0.0;
    let mut row_value = CDouble::from(0.0);
    for rv in row_values {
        if rv.index == col {
            col_coef = rv.value;
        } else {
            row_value += rv.value * sol.col_value[rv.index as usize];
        }
    }
    let c = col as usize;
    if sol.is_model_row(row) {
        sol.row_value[row as usize] = f64::from(row_value + col_coef * sol.col_value[c]);
    }
    sol.col_value[c] = f64::from((rhs - row_value) / col_coef);
    col_coef
}

impl FreeColSubstitution {
    fn undo(&self, row_values: &[Nonzero], col_values: &[Nonzero], sol: &mut Solution, basis: &mut Basis) {
        let col_coef = substitute_col(row_values, self.row, self.col, self.rhs, sol);
        if !sol.dual_valid {
            return;
        }
        let row = self.row as usize;
        if sol.is_model_row(self.row) {
            sol.row_dual[row] = 0.0;
            let mut dualval = CDouble::from(self.col_cost);
            for cv in col_values {
                if sol.is_model_row(cv.index) {
                    dualval -= cv.value * sol.row_dual[cv.index as usize];
                }
            }
            sol.row_dual[row] = f64::from(dualval / col_coef);
        }
        sol.col_dual[self.col as usize] = 0.0;
        if !basis.valid {
            return;
        }
        basis.col_status[self.col as usize] = BASIC;
        if sol.is_model_row(self.row) {
            basis.row_status[row] = compute_row_status(sol.row_dual[row], self.row_type);
        }
    }
}

impl DoubletonEquation {
    fn undo(&self, tol: &Tolerances, col_values: &[Nonzero], sol: &mut Solution, basis: &mut Basis) {
        let (col, col_subst) = (self.col as usize, self.col_subst as usize);
        sol.col_value[col_subst] =
            f64::from((self.rhs - CDouble::from(self.coef) * sol.col_value[col]) / self.coef_subst);

        if self.row == -1 || !sol.dual_valid {
            return;
        }
        let col_status = if !basis.valid {
            compute_status(sol.col_dual[col], tol.dual_feasibility)
        } else {
            compute_status_keep(sol.col_dual[col], &mut basis.col_status[col], tol.dual_feasibility)
        };

        let row = self.row as usize;
        let mut row_dual = CDouble::from(0.0);
        if sol.is_model_row(self.row) {
            sol.row_dual[row] = 0.0;
            for cv in col_values {
                if sol.is_model_row(cv.index) {
                    row_dual -= cv.value * sol.row_dual[cv.index as usize];
                }
            }
            row_dual /= self.coef_subst;
            sol.row_dual[row] = f64::from(row_dual);
        }
        sol.col_dual[col_subst] = self.subst_cost;
        sol.col_dual[col] += self.subst_cost * self.coef / self.coef_subst;

        if (self.upper_tightened != 0 && col_status == UPPER)
            || (self.lower_tightened != 0 && col_status == LOWER)
        {
            let row_dual_delta = sol.col_dual[col] / self.coef;
            if sol.is_model_row(self.row) {
                sol.row_dual[row] = f64::from(row_dual + row_dual_delta);
            }
            sol.col_dual[col] = 0.0;
            sol.col_dual[col_subst] =
                f64::from(CDouble::from(sol.col_dual[col_subst]) - row_dual_delta * self.coef_subst);
            if basis.valid {
                let same_sign = self.coef.is_sign_negative() == self.coef_subst.is_sign_negative();
                basis.col_status[col_subst] = if (same_sign && basis.col_status[col] == UPPER)
                    || (!same_sign && basis.col_status[col] == LOWER)
                {
                    LOWER
                } else {
                    UPPER
                };
                basis.col_status[col] = BASIC;
            }
        } else {
            let row_dual_delta = sol.col_dual[col_subst] / self.coef_subst;
            if sol.is_model_row(self.row) {
                sol.row_dual[row] = f64::from(row_dual + row_dual_delta);
            }
            sol.col_dual[col_subst] = 0.0;
            sol.col_dual[col] = f64::from(CDouble::from(sol.col_dual[col]) - row_dual_delta * self.coef);
            if basis.valid {
                basis.col_status[col_subst] = BASIC;
            }
        }

        if !basis.valid {
            return;
        }
        if sol.is_model_row(self.row) {
            basis.row_status[row] = compute_row_status(sol.row_dual[row], self.row_type);
        }
    }
}

impl EqualityRowAddition {
    fn undo(&self, sol: &mut Solution) {
        if !sol.is_model_row(self.row) || !sol.is_model_row(self.added_eq_row) {
            return;
        }
        let (row, eq) = (self.row as usize, self.added_eq_row as usize);
        if !sol.dual_valid || sol.row_dual[row] == 0.0 {
            return;
        }
        sol.row_dual[eq] = f64::from(CDouble::from(self.eq_row_scale) * sol.row_dual[row] + sol.row_dual[eq]);
    }
}

impl EqualityRowAdditions {
    fn undo(&self, target_rows: &[Nonzero], sol: &mut Solution) {
        if !sol.is_model_row(self.added_eq_row) || !sol.dual_valid {
            return;
        }
        let eq = self.added_eq_row as usize;
        let mut eq_row_dual = CDouble::from(sol.row_dual[eq]);
        for t in target_rows {
            if sol.is_model_row(t.index) {
                eq_row_dual += CDouble::from(t.value) * sol.row_dual[t.index as usize];
            }
        }
        sol.row_dual[eq] = f64::from(eq_row_dual);
    }
}

impl ForcingColumn {
    fn undo(&self, tol: &Tolerances, col_values: &[Nonzero], sol: &mut Solution, basis: &mut Basis) {
        let mut nonbasic_row: i32 = -1;
        let mut nonbasic_row_status = NONBASIC;
        let mut col_val_from_nonbasic_row = self.col_bound;
        let direction: f64 = if self.at_infinite_upper != 0 { 1.0 } else { -1.0 };
        for cv in col_values {
            if sol.is_model_row(cv.index) {
                let col_val_from_row = sol.row_value[cv.index as usize] / cv.value;
                if direction * col_val_from_row > direction * col_val_from_nonbasic_row {
                    nonbasic_row = cv.index;
                    col_val_from_nonbasic_row = col_val_from_row;
                    nonbasic_row_status = if direction * cv.value > 0.0 { LOWER } else { UPPER };
                }
            }
        }
        if nonbasic_row != -1 && self.col_integral != 0 {
            // direction * x is exact, so fusing the subtraction does not matter
            col_val_from_nonbasic_row =
                direction * (direction * col_val_from_nonbasic_row - tol.mip_feasibility).ceil();
        }
        let col = self.col as usize;
        sol.col_value[col] = col_val_from_nonbasic_row;
        if !sol.dual_valid {
            return;
        }
        sol.col_dual[col] = 0.0;
        if !basis.valid {
            return;
        }
        if nonbasic_row == -1 {
            basis.col_status[col] = if self.at_infinite_upper != 0 { LOWER } else { UPPER };
        } else {
            basis.col_status[col] = BASIC;
            basis.row_status[nonbasic_row as usize] = nonbasic_row_status;
        }
    }
}

impl ForcingColumnRemovedRow {
    fn undo(&self, row_values: &[Nonzero], sol: &mut Solution, basis: &mut Basis) {
        if !sol.is_model_row(self.row) {
            return;
        }
        let mut val = CDouble::from(self.rhs);
        for rv in row_values {
            val -= rv.value * sol.col_value[rv.index as usize];
        }
        let row = self.row as usize;
        sol.row_value[row] = f64::from(val);
        if sol.dual_valid {
            sol.row_dual[row] = 0.0;
        }
        if basis.valid {
            basis.row_status[row] = BASIC;
        }
    }
}

impl SingletonRow {
    fn undo(&self, tol: &Tolerances, sol: &mut Solution, basis: &mut Basis) {
        if !sol.dual_valid {
            return;
        }
        let (row, col) = (self.row as usize, self.col as usize);
        let col_status = if !basis.valid {
            compute_status(sol.col_dual[col], tol.dual_feasibility)
        } else {
            compute_status_keep(sol.col_dual[col], &mut basis.col_status[col], tol.dual_feasibility)
        };
        if (self.col_lower_tightened == 0 || col_status != LOWER)
            && (self.col_upper_tightened == 0 || col_status != UPPER)
        {
            if sol.is_model_row(self.row) {
                if basis.valid {
                    basis.row_status[row] = BASIC;
                }
                sol.row_dual[row] = 0.0;
            }
            return;
        }
        if sol.is_model_row(self.row) {
            sol.row_dual[row] = sol.col_dual[col] / self.coef;
        }
        sol.col_dual[col] = 0.0;
        if !basis.valid {
            return;
        }
        if sol.is_model_row(self.row) {
            match col_status {
                LOWER => basis.row_status[row] = if self.coef > 0.0 { LOWER } else { UPPER },
                UPPER => basis.row_status[row] = if self.coef > 0.0 { UPPER } else { LOWER },
                _ => {}
            }
        }
        basis.col_status[col] = BASIC;
    }
}

impl FixedCol {
    fn undo(&self, col_values: &[Nonzero], sol: &mut Solution, basis: &mut Basis) {
        let col = self.col as usize;
        sol.col_value[col] = self.fix_value;
        if !sol.dual_valid {
            return;
        }
        let mut reduced_cost = CDouble::from(self.col_cost);
        for cv in col_values {
            if sol.is_model_row(cv.index) {
                reduced_cost -= cv.value * sol.row_dual[cv.index as usize];
            }
        }
        sol.col_dual[col] = f64::from(reduced_cost);
        if basis.valid {
            basis.col_status[col] = self.fix_type;
            if self.fix_type == NONBASIC {
                basis.col_status[col] = if sol.col_dual[col] >= 0.0 { LOWER } else { UPPER };
            }
        }
    }
}

impl RedundantRow {
    fn undo(&self, sol: &mut Solution, basis: &mut Basis) {
        if !sol.is_model_row(self.row) || !sol.dual_valid {
            return;
        }
        sol.row_dual[self.row as usize] = 0.0;
        if basis.valid {
            basis.row_status[self.row as usize] = BASIC;
        }
    }
}

impl ForcingRow {
    fn undo(&self, row_values: &[Nonzero], sol: &mut Solution, basis: &mut Basis) {
        if !sol.dual_valid {
            return;
        }
        let mut basic_col: i32 = -1;
        let mut dual_delta = 0.0;
        let direction: f64 = if self.row_type == LEQ { 1.0 } else { -1.0 };
        for rv in row_values {
            // a - b * c, fused by clang
            let col_dual = (-rv.value).mul_add_c(dual_delta, sol.col_dual[rv.index as usize]);
            if direction * col_dual * rv.value < 0.0 {
                dual_delta = sol.col_dual[rv.index as usize] / rv.value;
                basic_col = rv.index;
            }
        }
        if basic_col != -1 {
            let row = self.row as usize;
            if sol.is_model_row(self.row) {
                sol.row_dual[row] += dual_delta;
            }
            for rv in row_values {
                let j = rv.index as usize;
                sol.col_dual[j] = f64::from(sol.col_dual[j] - CDouble::from(dual_delta) * rv.value);
            }
            sol.col_dual[basic_col as usize] = 0.0;
            if basis.valid {
                if sol.is_model_row(self.row) {
                    basis.row_status[row] = if self.row_type == GEQ { LOWER } else { UPPER };
                }
                basis.col_status[basic_col as usize] = BASIC;
            }
        }
    }
}

impl DuplicateRow {
    fn undo(&self, tol: &Tolerances, sol: &mut Solution, basis: &mut Basis) {
        if !sol.is_model_row(self.row) || !sol.dual_valid {
            return;
        }
        let (row, dup) = (self.row as usize, self.duplicate_row as usize);
        let dup_in_model = sol.is_model_row(self.duplicate_row);
        let make_dup_basic = |sol: &mut Solution, basis: &mut Basis| {
            if dup_in_model {
                sol.row_dual[dup] = 0.0;
                if basis.valid {
                    basis.row_status[dup] = BASIC;
                }
            }
        };
        if self.row_upper_tightened == 0 && self.row_lower_tightened == 0 {
            make_dup_basic(sol, basis);
            return;
        }
        let row_status = if !basis.valid {
            compute_status(sol.row_dual[row], tol.dual_feasibility)
        } else {
            compute_status_keep(sol.row_dual[row], &mut basis.row_status[row], tol.dual_feasibility)
        };
        let tightened = match row_status {
            BASIC => {
                make_dup_basic(sol, basis);
                return;
            }
            UPPER => self.row_upper_tightened != 0,
            LOWER => self.row_lower_tightened != 0,
            _ => return,
        };
        if tightened {
            if dup_in_model {
                sol.row_dual[dup] = sol.row_dual[row] / self.duplicate_row_scale;
                if basis.valid {
                    basis.row_status[dup] = if self.duplicate_row_scale > 0.0 { UPPER } else { LOWER };
                }
            }
            sol.row_dual[row] = 0.0;
            if basis.valid {
                basis.row_status[row] = BASIC;
            }
        } else {
            make_dup_basic(sol, basis);
        }
    }
}

impl DuplicateColumn {
    fn undo(&self, tol: &Tolerances, sol: &mut Solution, basis: &mut Basis) {
        let (col, dup) = (self.col as usize, self.duplicate_col as usize);
        let merge_val = sol.col_value[col];
        let ok_residual = |x: f64, y: f64| {
            // x + colScale * y, fused by clang
            let check = self.col_scale.mul_add_c(y, x);
            (check - merge_val).abs() <= tol.primal_feasibility
        };
        let is_at_bound = |value: f64, bound: f64| {
            if value < bound - tol.primal_feasibility {
                return false;
            }
            value <= bound + tol.primal_feasibility
        };

        if sol.dual_valid {
            sol.col_dual[dup] = sol.col_dual[col] * self.col_scale;
        }

        if basis.valid {
            match basis.col_status[col] {
                LOWER => {
                    sol.col_value[col] = self.col_lower;
                    if self.col_scale > 0.0 {
                        basis.col_status[dup] = LOWER;
                        sol.col_value[dup] = self.duplicate_col_lower;
                    } else {
                        basis.col_status[dup] = UPPER;
                        sol.col_value[dup] = self.duplicate_col_upper;
                    }
                    return;
                }
                UPPER => {
                    sol.col_value[col] = self.col_upper;
                    if self.col_scale > 0.0 {
                        basis.col_status[dup] = UPPER;
                        sol.col_value[dup] = self.duplicate_col_upper;
                    } else {
                        basis.col_status[dup] = LOWER;
                        sol.col_value[dup] = self.duplicate_col_lower;
                    }
                    return;
                }
                ZERO => {
                    sol.col_value[col] = 0.0;
                    basis.col_status[dup] = ZERO;
                    sol.col_value[dup] = 0.0;
                    return;
                }
                _ => {}
            }
        }

        sol.col_value[col] = if self.col_lower != -INF { self.col_lower } else { 0.0f64.min(self.col_upper) };
        sol.col_value[dup] = f64::from((CDouble::from(merge_val) - sol.col_value[col]) / self.col_scale);

        let mut recompute_col = false;
        if basis.valid {
            basis.col_status[dup] = NONBASIC;
        }
        if sol.col_value[dup] > self.duplicate_col_upper {
            sol.col_value[dup] = self.duplicate_col_upper;
            recompute_col = true;
            if basis.valid {
                basis.col_status[dup] = UPPER;
            }
        } else if sol.col_value[dup] < self.duplicate_col_lower {
            sol.col_value[dup] = self.duplicate_col_lower;
            recompute_col = true;
            if basis.valid {
                basis.col_status[dup] = LOWER;
            }
        } else if self.duplicate_col_integral != 0 && fractionality(sol.col_value[dup]) > tol.mip_feasibility {
            sol.col_value[dup] = sol.col_value[dup].floor();
            recompute_col = true;
        }

        if recompute_col {
            // mergeVal - colScale * y, fused by clang
            sol.col_value[col] = (-self.col_scale).mul_add_c(sol.col_value[dup], merge_val);
            if self.duplicate_col_integral == 0 && self.col_integral != 0 {
                sol.col_value[col] = (sol.col_value[col] - tol.mip_feasibility).ceil();
                sol.col_value[dup] = f64::from((CDouble::from(merge_val) - sol.col_value[col]) / self.col_scale);
            }
        } else if basis.valid {
            basis.col_status[dup] = basis.col_status[col];
            if self.col_lower != -INF {
                basis.col_status[col] = LOWER;
            } else if self.col_upper <= 0.0 {
                basis.col_status[col] = UPPER;
            } else {
                basis.col_status[col] = ZERO;
                if self.col_upper < INF {
                    c_print(&sprintf(
                        "HighsPostsolveStack::DuplicateColumn::undo Col is nonbasic at zero with upper bound of %g\n",
                        &[Arg::F(self.col_upper)],
                    ));
                }
            }
        }

        let (x, y) = (sol.col_value[col], sol.col_value[dup]);
        let mip = tol.mip_feasibility;
        let has_error = y < self.duplicate_col_lower - mip
            || y > self.duplicate_col_upper + mip
            || x < self.col_lower - mip
            || x > self.col_upper + mip
            || !ok_residual(x, y);
        if !has_error {
            return;
        }
        self.undo_fix(tol, sol, merge_val);

        if basis.valid {
            let (x, y) = (sol.col_value[col], sol.col_value[dup]);
            let mut dup_basic = false;
            if self.duplicate_col_lower <= -INF && self.duplicate_col_upper >= INF {
                if y == 0.0 {
                    basis.col_status[col] = BASIC;
                    basis.col_status[dup] = ZERO;
                } else {
                    dup_basic = true;
                }
            } else if is_at_bound(y, self.duplicate_col_lower) {
                basis.col_status[col] = BASIC;
                basis.col_status[dup] = LOWER;
            } else if is_at_bound(y, self.duplicate_col_upper) {
                basis.col_status[col] = BASIC;
                basis.col_status[dup] = UPPER;
            } else {
                dup_basic = true;
            }
            if dup_basic {
                basis.col_status[dup] = BASIC;
                basis.col_status[col] = if is_at_bound(x, self.col_lower) {
                    LOWER
                } else if is_at_bound(x, self.col_upper) {
                    UPPER
                } else {
                    NONBASIC
                };
            }
        }
    }

    /// DuplicateColumn::okMerge
    pub fn ok_merge(&self, tolerance: f64) -> bool {
        let scale = self.col_scale;
        let x_int = self.col_integral != 0;
        let y_int = self.duplicate_col_integral != 0;
        let x_lo = if x_int { (self.col_lower - tolerance).ceil() } else { self.col_lower };
        let x_up = if x_int { (self.col_upper + tolerance).floor() } else { self.col_upper };
        let y_lo = if y_int { (self.duplicate_col_lower - tolerance).ceil() } else { self.duplicate_col_lower };
        let y_up = if y_int { (self.duplicate_col_upper + tolerance).floor() } else { self.duplicate_col_upper };
        let x_len = x_up - x_lo;
        let y_len = y_up - y_lo;
        let mut ok = scale != 0.0;
        let abs_scale = scale.abs();
        if x_int {
            if y_int {
                if fractionality(scale) > tolerance {
                    ok = false;
                }
                if abs_scale > x_len + 1.0 + tolerance {
                    ok = false;
                }
            } else if y_len == 0.0 || abs_scale < 1.0 / y_len {
                ok = false;
            }
        } else if y_int && abs_scale > x_len {
            ok = false;
        }
        ok
    }

    fn undo_fix(&self, tol: &Tolerances, sol: &mut Solution, merge_value: f64) {
        let mip = tol.mip_feasibility;
        let primal = tol.primal_feasibility;
        let is_integer = |v: f64| fractionality(v) <= mip;
        let is_feasible = |l: f64, v: f64, u: f64| v >= l - primal && v <= u + primal;
        let scale = self.col_scale;
        let x_int = self.col_integral != 0;
        let y_int = self.duplicate_col_integral != 0;
        let x_lo = if x_int { (self.col_lower - mip).ceil() } else { self.col_lower };
        let x_up = if x_int { (self.col_upper + mip).floor() } else { self.col_upper };
        let y_lo = if y_int { (self.duplicate_col_lower - mip).ceil() } else { self.duplicate_col_lower };
        let y_up = if y_int { (self.duplicate_col_upper + mip).floor() } else { self.duplicate_col_upper };
        let mut x_v: f64;
        let mut y_v: f64;

        // (z_0, z_d, z_1)
        let check_int_var = |z_lo: f64, z_up: f64| -> (f64, f64, f64) {
            const VALUE_MAX: f64 = 1000.0;
            if z_lo <= -INF {
                if z_up >= INF {
                    (0.0, 1.0, VALUE_MAX)
                } else {
                    (z_up, -1.0, -VALUE_MAX)
                }
            } else {
                (z_lo, 1.0, if z_up >= INF { VALUE_MAX } else { z_up })
            }
        };
        let compute_value = |value: f64| f64::from((CDouble::from(merge_value) - value) / scale);
        let compute_inv_value = |value: f64| f64::from(CDouble::from(merge_value) - CDouble::from(value) * scale);
        // Returns (z_value, other_value)
        let find_value = |z_0: f64, z_1: f64, z_delta: f64, other_lower: f64, other_upper: f64, other_int: bool| {
            const EPS: f64 = 1e-8;
            let mut z_value = z_0;
            loop {
                let other_value = compute_value(z_value);
                if is_feasible(other_lower, other_value, other_upper) && (!other_int || is_integer(other_value)) {
                    return (z_value, other_value);
                }
                if z_delta > 0.0 && z_value + z_delta >= z_1 + EPS {
                    return (z_value, other_value);
                }
                if z_delta < 0.0 && z_value + z_delta <= z_1 - EPS {
                    return (z_value, other_value);
                }
                z_value += z_delta;
            }
        };
        let set_x = |value: f64| (value, compute_value(value));
        let set_y = |value: f64| (compute_inv_value(value), value);

        if x_int {
            let (x_0, x_d, x_1) = check_int_var(x_lo, x_up);
            (x_v, y_v) = find_value(x_0, x_1, x_d, y_lo, y_up, y_int);
        } else if y_int {
            let (y_0, y_d, y_1) = check_int_var(y_lo, y_up);
            (y_v, x_v) = find_value(y_0, y_1, y_d, x_lo, x_up, x_int);
        } else if scale > 0.0 {
            if y_up < INF {
                (x_v, y_v) = set_y(y_up);
                if x_v < x_lo - primal {
                    (x_v, y_v) = set_x(x_lo);
                    if y_v < y_lo - primal {
                        (x_v, y_v) = set_x(x_lo - primal);
                    }
                }
            } else if y_lo > -INF {
                (x_v, y_v) = set_y(y_lo);
                if x_v > x_up + primal {
                    (x_v, y_v) = set_x(x_up);
                    if y_v > y_up + primal {
                        (x_v, y_v) = set_x(x_up + primal);
                    }
                }
            } else {
                (x_v, y_v) = set_x(0.0f64.max(x_lo));
            }
        } else if y_lo > -INF {
            (x_v, y_v) = set_y(y_lo);
            if x_v < x_lo - primal {
                (x_v, y_v) = set_x(x_lo);
                if y_v > y_up + primal {
                    (x_v, y_v) = set_x(x_lo - primal);
                }
            }
        } else if y_up < INF {
            (x_v, y_v) = set_y(y_up);
            if x_v > x_up + primal {
                (x_v, y_v) = set_x(x_up);
                if y_v < y_lo - primal {
                    (x_v, y_v) = set_x(x_up + primal);
                }
            }
        } else {
            (x_v, y_v) = set_x(0.0f64.max(x_lo));
        }
        let residual = f64::from(CDouble::from(x_v) - compute_inv_value(y_v)).abs();
        let x_y_ok = is_feasible(x_lo, x_v, x_up)
            && is_feasible(y_lo, y_v, y_up)
            && (!x_int || is_integer(x_v))
            && (!y_int || is_integer(y_v))
            && x_v.abs() < INF
            && y_v.abs() < INF
            && residual <= 1e-12;
        if x_y_ok {
            sol.col_value[self.col as usize] = x_v;
            sol.col_value[self.duplicate_col as usize] = y_v;
        }
    }
}

impl SlackColSubstitution {
    fn undo(&self, row_values: &[Nonzero], sol: &mut Solution, basis: &mut Basis) {
        let col_coef = substitute_col(row_values, self.row, self.col, self.rhs, sol);
        if !sol.dual_valid {
            return;
        }
        let (row, col) = (self.row as usize, self.col as usize);
        if sol.is_model_row(self.row) {
            sol.col_dual[col] = -sol.row_dual[row] / col_coef;
        }
        if !basis.valid {
            return;
        }
        if sol.is_model_row(self.row) {
            if basis.row_status[row] == BASIC {
                basis.col_status[col] = BASIC;
                basis.row_status[row] = compute_row_status(sol.row_dual[row], EQ);
            } else if basis.row_status[row] == LOWER {
                basis.col_status[col] = if col_coef > 0.0 { UPPER } else { LOWER };
            } else {
                basis.col_status[col] = if col_coef > 0.0 { LOWER } else { UPPER };
            }
        } else {
            basis.col_status[col] = NONBASIC;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push<T: Pod>(data: &mut Vec<u8>, r: &T) {
        // SAFETY: T: Pod is plain bytes
        data.extend_from_slice(unsafe { std::slice::from_raw_parts(r as *const T as *const u8, size_of::<T>()) });
    }

    /// A fixed column x0 = 2 with cost 3 in row 0 (coefficient 2, dual 1),
    /// pushed as HighsDataStack does, on a reduced problem with one row
    #[test]
    fn fixed_col_undo() {
        let mut data = Vec::new();
        push(&mut data, &FixedCol { fix_value: 2.0, col_cost: 3.0, col: 0, fix_type: NONBASIC });
        push(&mut data, &Nonzero { index: 0, value: 2.0 });
        push(&mut data, &1usize);
        let reductions = [Reduction { kind: FIXED_COL, position: data.len() }];
        let stack = Stack { data: &data, reductions: &reductions, orig_col_index: &[1], orig_row_index: &[0] };
        let tol = Tolerances { primal_feasibility: 1e-7, dual_feasibility: 1e-7, mip_feasibility: 1e-6 };
        let (mut cv, mut rv, mut cd, mut rd) = ([5.0, 0.0], [4.0], [0.5, 0.0], [1.0]);
        let (mut cs, mut rs) = ([BASIC, 0], [UPPER]);
        let mut sol =
            Solution { col_value: &mut cv, row_value: &mut rv, col_dual: &mut cd, row_dual: &mut rd, dual_valid: true };
        let mut basis = Basis { col_status: &mut cs, row_status: &mut rs, valid: true };
        undo(&stack, &tol, &mut sol, &mut basis, 0, -1);
        assert_eq!(cv, [2.0, 5.0]);
        assert_eq!(cd, [1.0, 0.5]); // 3 - 2 * 1
        assert_eq!(cs, [LOWER, BASIC]);
    }

    #[test]
    fn compress() {
        let (mut r, mut c) = ([0, 1, 2], [0, 1]);
        assert_eq!(compress_index_maps(&mut r, &mut c, &[0, -1, 1], &[-1, 0]), (2, 1));
        assert_eq!((r[..2].to_vec(), c[0]), (vec![0, 2], 1));
    }
}
