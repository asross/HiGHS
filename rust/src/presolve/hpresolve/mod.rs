//! HPresolve (highs/presolve/HPresolve.cpp): the presolve engine of LP and
//! MIP presolve and of the MIP restarts, step for step as the C++ so that
//! the presolved models, the postsolve stacks and hence the search paths
//! are bit-identical.
//!
//! Rust owns the presolve state and a copy of the model (costs, bounds,
//! integrality, offset), which it writes back to the C++ HighsLp
//! (`sync_model`) whenever C++ code reads it: the MIP rebuilds of
//! shrinkProblem, probing, and the end of the run. The postsolve stack stays
//! C++: the reductions are recorded here (record.rs) and appended to it by
//! `flush`. What runs in C++ is called through [`Host`]: logging, the
//! timer, the analysis setup, the HFactor of the dependent equations, and
//! the MIP solver's C++ parts: the setup of the domain and clique table for
//! probing, the cut pool, and the glue of the domain, the clique table and
//! implications.runProbing (ffi.rs, highs/presolve/HPresolveRust.cpp). The
//! probing and enumeration loops (probing.rs, enumeration.rs) work on the
//! Rust clique table and implications directly and on the domain through
//! its Rust view (mip/domain.rs).
//!
//! The matrix is stored as in the C++: triplets with a linked list per
//! column and a splay tree per row (keyed by the column), whose shapes
//! decide the orders in which rows are scanned.

mod cols;
pub mod ffi;
mod mip;
mod parallel;
mod record;
mod reduce;
mod rows;
mod driver;
mod enumeration;
mod probing;

use crate::mip::clique::CliqueTable;
use crate::mip::implications::Implications;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use crate::util::linear_sum_bounds::LinearSumBounds;
use crate::util::splay::Tree;
use ffi::Host;
use record::Recorder;
use std::collections::BTreeSet;

pub(crate) const INF: f64 = f64::INFINITY;
pub(crate) const TINY: f64 = 1e-14;
pub(crate) const IINF: i32 = i32::MAX;

// HighsVarType
pub(crate) const CONTINUOUS: u8 = 0;
pub(crate) const INTEGER: u8 = 1;
pub(crate) const IMPLICIT_INTEGER: u8 = 4;

// HighsLogType
pub(crate) const LOG_INFO: i32 = 1;
pub(crate) const LOG_DETAILED: i32 = 2;
pub(crate) const LOG_WARNING: i32 = 4;
pub(crate) const LOG_ERROR: i32 = 5;

// Presolve rules (kPresolveRule*)
pub(crate) const RULE_EMPTY_ROW: usize = 0;
pub(crate) const RULE_SINGLETON_ROW: usize = 1;
pub(crate) const RULE_REDUNDANT_ROW: usize = 2;
pub(crate) const RULE_EMPTY_COL: usize = 3;
pub(crate) const RULE_FIXED_COL: usize = 4;
pub(crate) const RULE_DOMINATED_COL: usize = 5;
pub(crate) const RULE_FORCING_ROW: usize = 6;
pub(crate) const RULE_FORCING_COL: usize = 7;
pub(crate) const RULE_FREE_COL_SUBSTITUTION: usize = 8;
pub(crate) const RULE_DOUBLETON_EQUATION: usize = 9;
pub(crate) const RULE_DEPENDENT_EQUATIONS: usize = 10;
pub(crate) const RULE_DEPENDENT_FREE_COLS: usize = 11;
pub(crate) const RULE_AGGREGATOR: usize = 12;
pub(crate) const RULE_PARALLEL_ROWS_AND_COLS: usize = 13;
pub(crate) const RULE_SPARSIFY: usize = 14;
pub(crate) const RULE_PROBING: usize = 15;
pub(crate) const RULE_ENUMERATION: usize = 16;
pub(crate) const RULE_COL_STUFFING: usize = 18;
pub(crate) const RULE_COUNT: usize = 20;
pub(crate) const RULE_ILLEGAL: i32 = -1;

/// The non-ok outcomes of HPresolve::Result
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Stop {
    PrimalInfeasible,
    DualInfeasible,
    Stopped,
}

pub(crate) type R = Result<(), Stop>;

/// HPresolve::StatusResult: a flag, or a non-ok result
pub(crate) type SR = Result<bool, Stop>;

/// HighsUtils fractionality
#[inline]
pub(crate) fn fractionality(v: f64) -> f64 {
    (v - v.round()).abs()
}

/// The bounds of the columns, for impliedRowBounds
macro_rules! rb {
    ($s:expr) => {
        crate::util::linear_sum_bounds::VarBounds {
            lower: &$s.col_lower,
            upper: &$s.col_upper,
            impl_lower: &$s.impl_col_lower,
            impl_upper: &$s.impl_col_upper,
            impl_lower_src: &$s.col_lower_source,
            impl_upper_src: &$s.col_upper_source,
        }
    };
}

/// The bounds of the row duals, for impliedDualRowBounds
macro_rules! db {
    ($s:expr) => {
        crate::util::linear_sum_bounds::VarBounds {
            lower: &$s.row_dual_lower,
            upper: &$s.row_dual_upper,
            impl_lower: &$s.impl_row_dual_lower,
            impl_upper: &$s.impl_row_dual_upper,
            impl_lower_src: &$s.row_dual_lower_source,
            impl_upper_src: &$s.row_dual_upper_source,
        }
    };
}
pub(crate) use rb;

/// The option values HPresolve reads
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Options {
    pub primal_feasibility_tolerance: f64,
    pub dual_feasibility_tolerance: f64,
    pub mip_feasibility_tolerance: f64,
    pub small_matrix_value: f64,
    pub time_limit: f64,
    pub presolve_pivot_threshold: f64,
    pub presolve_substitution_maxfillin: i32,
    pub presolve_rule_test: i32,
    pub presolve_rule_off: i32,
    pub log_dev_level: i32,
    pub random_seed: i32,
    pub mip_lifting_for_probing: i32,
    pub presolve_off: bool,
    pub lp_presolve_requires_basis_postsolve: bool,
    pub presolve_remove_slacks: bool,
    pub output_flag: bool,
    pub timeless_log: bool,
    pub use_implied_bounds_from_presolve: bool,
    pub presolve_rule_logging: bool,
}

/// The MIP solver's values HPresolve reads (when presolving a MIP)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MipInfo {
    pub epsilon: f64,
    /// mipdata_->feastol
    pub feastol: f64,
    /// the MIP solver's clique table and implications (Rust-owned, the C++
    /// classes hold these handles)
    pub cliquetable: *mut CliqueTable,
    pub implications: *mut Implications,
    pub orig_num_row: i32,
    pub num_restarts: i32,
    pub submip: bool,
}

/// HPresolveAnalysis
pub(crate) struct Analysis {
    pub allow_rule: [bool; RULE_COUNT],
    pub allow_logging: bool,
    pub logging_on: bool,
    pub log_rule_type: i32,
    pub num_deleted_rows0: i32,
    pub num_deleted_cols0: i32,
    /// (call, col_removed, row_removed) per rule
    pub log: [(i32, i32, i32); RULE_COUNT],
    pub original_num_col: i32,
    pub original_num_row: i32,
}

impl Default for Analysis {
    fn default() -> Self {
        Analysis {
            allow_rule: [true; RULE_COUNT],
            allow_logging: false,
            logging_on: false,
            log_rule_type: RULE_ILLEGAL,
            num_deleted_rows0: 0,
            num_deleted_cols0: 0,
            log: [(0, 0, 0); RULE_COUNT],
            original_num_col: 0,
            original_num_row: 0,
        }
    }
}

/// The lifting opportunities of probing: per row a hash tree, iterated as
/// the C++ std::unordered_map<HighsInt, ...> (reserved for all rows, so
/// each key has its own bucket and libc++ puts every new key first: the
/// iteration order is the reverse order of first insertion)
#[derive(Default)]
pub(crate) struct Lifting {
    pub trees: Vec<Option<crate::util::hash_tree::HighsHashTree<(i32, i32), f64>>>,
    pub order: Vec<i32>,
}

impl Lifting {
    pub fn clear_row(&mut self, row: i32) {
        if let Some(Some(t)) = self.trees.get_mut(row as usize) {
            t.clear();
        }
    }
    pub fn clear(&mut self) {
        for &r in &self.order {
            self.trees[r as usize] = None;
        }
        self.order.clear();
    }
    pub fn entry(&mut self, row: i32) -> &mut crate::util::hash_tree::HighsHashTree<(i32, i32), f64> {
        let r = row as usize;
        if self.trees.len() <= r {
            self.trees.resize_with(r + 1, || None);
        }
        if self.trees[r].is_none() {
            self.trees[r] = Some(Default::default());
            self.order.push(row);
        }
        self.trees[r].as_mut().unwrap()
    }
    pub fn len(&self) -> usize {
        self.order.len()
    }
}

/// A preorder walk of a row's splay tree (HighsTripletTreeSlicePreOrder);
/// the bodies of the loops do not change the tree they walk
pub(crate) struct PreOrder {
    stack: SStack,
    cur: i32,
}

/// std::set<HighsInt> as a sorted vector (the sets are small)
#[derive(Clone, Default)]
pub(crate) struct SortedSet(Vec<i32>);

impl SortedSet {
    #[inline]
    pub fn insert(&mut self, v: i32) {
        if let Err(i) = self.0.binary_search(&v) {
            self.0.insert(i, v);
        }
    }
    #[inline]
    pub fn remove(&mut self, v: &i32) {
        if let Ok(i) = self.0.binary_search(v) {
            self.0.remove(i);
        }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, i32> {
        self.0.iter()
    }
}

impl<'a> IntoIterator for &'a SortedSet {
    type Item = &'a i32;
    type IntoIter = std::slice::Iter<'a, i32>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// from an increasing sequence
impl FromIterator<i32> for SortedSet {
    fn from_iter<I: IntoIterator<Item = i32>>(it: I) -> Self {
        SortedSet(it.into_iter().collect())
    }
}

/// A stack of tree nodes without allocation for the usual depths
pub(crate) struct SStack {
    buf: [i32; 32],
    len: usize,
    spill: Vec<i32>,
}

impl SStack {
    #[inline]
    pub fn new() -> Self {
        SStack { buf: [0; 32], len: 0, spill: Vec::new() }
    }
    #[inline]
    pub fn push(&mut self, v: i32) {
        if self.len < 32 {
            self.buf[self.len] = v;
            self.len += 1;
        } else {
            self.spill.push(v);
        }
    }
    #[inline]
    pub fn pop(&mut self) -> Option<i32> {
        if let Some(v) = self.spill.pop() {
            Some(v)
        } else if self.len > 0 {
            self.len -= 1;
            Some(self.buf[self.len])
        } else {
            None
        }
    }
}

impl PreOrder {
    #[inline]
    pub fn new(root: i32) -> Self {
        PreOrder { stack: SStack::new(), cur: root }
    }
    #[inline]
    pub fn next(&mut self, left: &[i32], right: &[i32]) -> Option<usize> {
        if self.cur == -1 {
            return None;
        }
        let c = self.cur as usize;
        let (l, r) = (left[c], right[c]);
        if l != -1 {
            if r != -1 {
                self.stack.push(r);
            }
            self.cur = l;
        } else if r != -1 {
            self.cur = r;
        } else {
            self.cur = self.stack.pop().unwrap_or(-1);
        }
        Some(c)
    }
}

/// The nonzeros (row, value) of a column list
pub(crate) struct ColIter<'a> {
    pub index: &'a [i32],
    pub value: &'a [f64],
    pub next: &'a [i32],
    pub cur: i32,
}

impl Iterator for ColIter<'_> {
    type Item = (i32, f64);
    #[inline]
    fn next(&mut self) -> Option<(i32, f64)> {
        if self.cur == -1 {
            return None;
        }
        let c = self.cur as usize;
        self.cur = self.next[c];
        Some((self.index[c], self.value[c]))
    }
}

/// The nonzeros (col, value) of a row tree in preorder
pub(crate) struct RowIter<'a> {
    pub index: &'a [i32],
    pub value: &'a [f64],
    pub left: &'a [i32],
    pub right: &'a [i32],
    pub walk: PreOrder,
}

impl Iterator for RowIter<'_> {
    type Item = (i32, f64);
    #[inline]
    fn next(&mut self) -> Option<(i32, f64)> {
        let p = self.walk.next(self.left, self.right)?;
        Some((self.index[p], self.value[p]))
    }
}

/// The nonzeros (col, value) at the given positions
pub(crate) struct PosIter<'a> {
    pub index: &'a [i32],
    pub value: &'a [f64],
    pub pos: std::slice::Iter<'a, i32>,
}

impl Iterator for PosIter<'_> {
    type Item = (i32, f64);
    #[inline]
    fn next(&mut self) -> Option<(i32, f64)> {
        let &p = self.pos.next()?;
        Some((self.index[p as usize], self.value[p as usize]))
    }
}

macro_rules! col_iter {
    ($s:expr, $col:expr) => {
        crate::presolve::hpresolve::ColIter {
            index: &$s.a_row,
            value: &$s.a_value,
            next: &$s.a_next,
            cur: $s.colhead[$col as usize],
        }
    };
}

macro_rules! row_iter {
    ($s:expr, $row:expr) => {
        crate::presolve::hpresolve::RowIter {
            index: &$s.a_col,
            value: &$s.a_value,
            left: &$s.ar_left,
            right: &$s.ar_right,
            walk: crate::presolve::hpresolve::PreOrder::new($s.rowroot[$row as usize]),
        }
    };
}

/// getStoredRow
macro_rules! stored_row {
    ($s:expr) => {
        crate::presolve::hpresolve::PosIter {
            index: &$s.a_col,
            value: &$s.a_value,
            pos: $s.rowpositions[..$s.rp_len].iter(),
        }
    };
}
pub(crate) use {col_iter, row_iter, stored_row};

pub struct Presolve<'h> {
    pub(crate) host: &'h Host,
    pub(crate) opt: Options,
    pub(crate) mip: Option<MipInfo>,
    pub(crate) primal_feastol: f64,

    // the model
    pub(crate) num_col: i32,
    pub(crate) num_row: i32,
    pub(crate) col_cost: Vec<f64>,
    pub(crate) col_lower: Vec<f64>,
    pub(crate) col_upper: Vec<f64>,
    pub(crate) row_lower: Vec<f64>,
    pub(crate) row_upper: Vec<f64>,
    pub(crate) integrality: Vec<u8>,
    pub(crate) offset: f64,
    pub(crate) maximize: bool,
    pub(crate) model_name: String,

    // triplet storage
    pub(crate) a_value: Vec<f64>,
    pub(crate) a_row: Vec<i32>,
    pub(crate) a_col: Vec<i32>,
    pub(crate) colhead: Vec<i32>,
    pub(crate) a_next: Vec<i32>,
    pub(crate) a_prev: Vec<i32>,
    pub(crate) rowroot: Vec<i32>,
    pub(crate) ar_left: Vec<i32>,
    pub(crate) ar_right: Vec<i32>,
    pub(crate) rowsize: Vec<i32>,
    pub(crate) rowsize_integer: Vec<i32>,
    pub(crate) rowsize_impl_int: Vec<i32>,
    pub(crate) colsize: Vec<i32>,
    /// the positions of a stored row: entries past rp_len keep stale values
    /// as the C++ vector's capacity does
    pub(crate) rowpositions: Vec<i32>,
    pub(crate) rp_len: usize,
    pub(crate) freeslots: Vec<i32>,

    pub(crate) impl_col_lower: Vec<f64>,
    pub(crate) impl_col_upper: Vec<f64>,
    pub(crate) col_lower_source: Vec<i32>,
    pub(crate) col_upper_source: Vec<i32>,
    pub(crate) row_dual_lower: Vec<f64>,
    pub(crate) row_dual_upper: Vec<f64>,
    pub(crate) impl_row_dual_lower: Vec<f64>,
    pub(crate) impl_row_dual_upper: Vec<f64>,
    pub(crate) row_dual_lower_source: Vec<i32>,
    pub(crate) row_dual_upper_source: Vec<i32>,
    pub(crate) col_impl_source_by_row: Vec<SortedSet>,
    pub(crate) impl_row_dual_source_by_col: Vec<SortedSet>,

    pub(crate) implied_row_bounds: LinearSumBounds,
    pub(crate) implied_dual_row_bounds: LinearSumBounds,

    pub(crate) changed_row_indices: Vec<i32>,
    pub(crate) changed_row_flag: Vec<u8>,
    pub(crate) changed_col_indices: Vec<i32>,
    pub(crate) changed_col_flag: Vec<u8>,

    pub(crate) substitution_opportunities: Vec<(i32, i32)>,
    pub(crate) lifting: Lifting,

    /// std::set<pair<size, row>> of the equations; eqsize[row] is the size
    /// it was inserted with, -1 if not in the set (eqiters == end())
    pub(crate) equations: BTreeSet<(i32, i32)>,
    pub(crate) eqsize: Vec<i32>,

    pub(crate) shrink_problem_enabled: bool,
    pub(crate) reduction_limit: usize,

    pub(crate) singleton_rows: Vec<i32>,
    pub(crate) singleton_columns: Vec<i32>,
    pub(crate) row_deleted: Vec<u8>,
    pub(crate) col_deleted: Vec<u8>,
    pub(crate) single_equation_checked: Vec<u8>,

    pub(crate) num_probes: Vec<u16>,
    pub(crate) probing_contingent: i64,
    pub(crate) probing_num_del_col: i32,
    pub(crate) num_probed: i32,

    pub(crate) num_deleted_rows: i32,
    pub(crate) num_deleted_cols: i32,
    pub(crate) old_num_col: i32,
    pub(crate) old_num_row: i32,
    pub(crate) probing_early_abort: bool,

    pub(crate) presolve_status: i32,
    pub(crate) analysis: Analysis,
    pub(crate) ps: Recorder,
}

// HighsPresolveStatus
pub(crate) const PS_NOT_PRESOLVED: i32 = -1;
pub(crate) const PS_NOT_REDUCED: i32 = 0;
pub(crate) const PS_INFEASIBLE: i32 = 1;
pub(crate) const PS_UNBOUNDED_OR_INFEASIBLE: i32 = 2;
pub(crate) const PS_REDUCED: i32 = 3;
pub(crate) const PS_REDUCED_TO_EMPTY: i32 = 4;
pub(crate) const PS_NOT_SET: i32 = 8;

// HighsModelStatus
pub(crate) const MS_NOTSET: i32 = 0;
pub(crate) const MS_OPTIMAL: i32 = 7;
pub(crate) const MS_INFEASIBLE: i32 = 8;
pub(crate) const MS_UNBOUNDED_OR_INFEASIBLE: i32 = 9;

impl Presolve<'_> {
    // ---------------------------------------------------------------- logging

    pub(crate) fn log_user(&self, t: i32, msg: &str) {
        self.host.log(0, t, msg);
    }

    pub(crate) fn log_dev(&self, t: i32, msg: &str) {
        self.host.log(1, t, msg);
    }

    pub(crate) fn start_rule_log(&mut self, rule: usize) {
        let a = &mut self.analysis;
        debug_assert!(a.logging_on);
        a.logging_on = false;
        a.log[rule].0 += 1;
        debug_assert!(a.log_rule_type == RULE_ILLEGAL);
        a.log_rule_type = rule as i32;
        if a.num_deleted_rows0 != self.num_deleted_rows || a.num_deleted_cols0 != self.num_deleted_cols {
            let msg = crate::util::printf::sprintf(
                "ERROR: Model %s: %d = num_deleted_rows0_ != *numDeletedRows = %d ||%d = num_deleted_cols0_ != *numDeletedCols = %d\n",
                &[
                    (&self.model_name).into(),
                    a.num_deleted_rows0.into(),
                    self.num_deleted_rows.into(),
                    a.num_deleted_cols0.into(),
                    self.num_deleted_cols.into(),
                ],
            );
            self.host.log(3, 0, &msg);
        }
        a.num_deleted_rows0 = self.num_deleted_rows;
        a.num_deleted_cols0 = self.num_deleted_cols;
    }

    pub(crate) fn stop_rule_log(&mut self, rule: usize) {
        let a = &mut self.analysis;
        debug_assert!(a.logging_on);
        debug_assert!(rule as i32 == a.log_rule_type);
        let num_removed_row = self.num_deleted_rows - a.num_deleted_rows0;
        let num_removed_col = self.num_deleted_cols - a.num_deleted_cols0;
        a.log[rule].1 += num_removed_col;
        a.log[rule].2 += num_removed_row;
        a.log_rule_type = RULE_ILLEGAL;
        a.num_deleted_rows0 = self.num_deleted_rows;
        a.num_deleted_cols0 = self.num_deleted_cols;
    }

    // ---------------------------------------------------- simple predicates

    #[inline]
    pub(crate) fn is_lower_implied(&self, col: i32) -> bool {
        let c = col as usize;
        self.col_lower[c] == -INF || self.impl_col_lower[c] >= self.col_lower[c] - self.primal_feastol
    }

    #[inline]
    pub(crate) fn is_lower_strictly_implied(&self, col: i32, tol: Option<f64>) -> bool {
        let c = col as usize;
        self.col_lower[c] == -INF || self.impl_col_lower[c] > self.col_lower[c] + tol.unwrap_or(self.primal_feastol)
    }

    #[inline]
    pub(crate) fn is_upper_implied(&self, col: i32) -> bool {
        let c = col as usize;
        self.col_upper[c] == INF || self.impl_col_upper[c] <= self.col_upper[c] + self.primal_feastol
    }

    #[inline]
    pub(crate) fn is_upper_strictly_implied(&self, col: i32, tol: Option<f64>) -> bool {
        let c = col as usize;
        self.col_upper[c] == INF || self.impl_col_upper[c] < self.col_upper[c] - tol.unwrap_or(self.primal_feastol)
    }

    #[inline]
    pub(crate) fn is_implied_free(&self, col: i32) -> bool {
        self.is_lower_implied(col) && self.is_upper_implied(col)
    }

    #[inline]
    pub(crate) fn is_dual_implied_free(&self, row: i32) -> bool {
        let r = row as usize;
        self.is_equation(row)
            || (self.row_upper[r] != INF && self.impl_row_dual_upper[r] <= self.opt.dual_feasibility_tolerance)
            || (self.row_lower[r] != -INF && self.impl_row_dual_lower[r] >= -self.opt.dual_feasibility_tolerance)
    }

    /// dualImpliedFreeGetRhsAndRowType: (rhs, row type)
    pub(crate) fn dual_implied_free_get_rhs_and_row_type(&mut self, row: i32, relax_row_dual_bounds: bool) -> (f64, i32) {
        use crate::presolve::postsolve::{EQ, GEQ, LEQ};
        let r = row as usize;
        if self.is_equation(row) {
            (self.row_upper[r], EQ)
        } else if self.row_upper[r] != INF && self.impl_row_dual_upper[r] <= self.opt.dual_feasibility_tolerance {
            let rhs = self.row_upper[r];
            if relax_row_dual_bounds {
                self.change_row_dual_upper(row, INF);
            }
            (rhs, LEQ)
        } else {
            let rhs = self.row_lower[r];
            if relax_row_dual_bounds {
                self.change_row_dual_lower(row, -INF);
            }
            (rhs, GEQ)
        }
    }

    #[inline]
    pub(crate) fn is_equation(&self, row: i32) -> bool {
        self.row_lower[row as usize] == self.row_upper[row as usize]
    }

    #[inline]
    pub(crate) fn is_ranged(&self, row: i32) -> bool {
        self.row_lower[row as usize] != -INF && self.row_upper[row as usize] != INF
    }

    #[inline]
    pub(crate) fn is_redundant(&self, row: i32) -> bool {
        let r = row as usize;
        self.implied_row_bounds.sum_lower(row) >= self.row_lower[r] - self.primal_feastol
            && self.implied_row_bounds.sum_upper(row) <= self.row_upper[r] + self.primal_feastol
    }

    #[inline]
    pub(crate) fn yields_implied_lower_bound(&self, row: i32, val: f64) -> bool {
        let r = row as usize;
        (val < 0.0 && self.row_upper[r] != INF) || (val > 0.0 && self.row_lower[r] != -INF)
    }

    #[inline]
    pub(crate) fn yields_implied_upper_bound(&self, row: i32, val: f64) -> bool {
        self.yields_implied_lower_bound(row, -val)
    }

    #[inline]
    pub(crate) fn is_implied_equation_at_lower(&self, row: i32) -> bool {
        self.impl_row_dual_lower[row as usize] > self.opt.dual_feasibility_tolerance
    }

    #[inline]
    pub(crate) fn is_implied_equation_at_upper(&self, row: i32) -> bool {
        self.impl_row_dual_upper[row as usize] < -self.opt.dual_feasibility_tolerance
    }

    pub(crate) fn row_coefficients_integral(&self, row: i32, scale: f64) -> bool {
        for (_, v) in row_iter!(self, row) {
            if fractionality(v * scale) > self.opt.small_matrix_value {
                return false;
            }
        }
        true
    }

    pub(crate) fn is_implied_integral(&mut self, col: i32) -> SR {
        debug_assert!(self.integrality[col as usize] == INTEGER);
        let mut run_dual_detection = true;
        for (row, val) in col_iter!(self, col) {
            let r = row as usize;
            if self.rowsize[r] < 2 || self.rowsize_integer[r] < self.rowsize[r] {
                run_dual_detection = false;
                continue;
            }
            let row_lower =
                if self.is_implied_equation_at_upper(row) { self.row_upper[r] } else { self.row_lower[r] };
            let row_upper =
                if self.is_implied_equation_at_lower(row) { self.row_lower[r] } else { self.row_upper[r] };
            if row_upper == row_lower {
                run_dual_detection = false;
                let scale = 1.0 / val;
                if !self.row_coefficients_integral(row, scale) {
                    continue;
                }
                if fractionality(row_lower * scale) > self.primal_feastol {
                    return Err(Stop::PrimalInfeasible);
                }
                return Ok(true);
            }
        }
        if !run_dual_detection {
            return Ok(false);
        }
        let mut p = self.colhead[col as usize];
        while p != -1 {
            let pu = p as usize;
            let row = self.a_row[pu];
            let val = self.a_value[pu];
            let r = row as usize;
            let scale = 1.0 / val;
            if !self.row_coefficients_integral(row, scale) {
                return Ok(false);
            }
            if self.row_upper[r] != INF {
                let r_upper = val.abs() * (self.row_upper[r].mul_add_c(scale.abs(), self.primal_feastol)).floor();
                if (self.row_upper[r] - r_upper).abs() > self.opt.small_matrix_value {
                    self.row_upper[r] = r_upper;
                    self.mark_changed_row(row);
                }
            }
            if self.row_lower[r] != -INF {
                let r_lower = val.abs() * (self.row_lower[r].mul_add_c(scale.abs(), -self.primal_feastol)).ceil();
                if (self.row_lower[r] - r_lower).abs() > self.opt.small_matrix_value {
                    self.row_lower[r] = r_lower;
                    self.mark_changed_row(row);
                }
            }
            p = self.a_next[pu];
        }
        Ok(true)
    }

    pub(crate) fn is_implied_integer(&self, col: i32) -> SR {
        debug_assert!(self.integrality[col as usize] == CONTINUOUS);
        let mut run_dual_detection = true;
        for (row, val) in col_iter!(self, col) {
            let r = row as usize;
            if self.rowsize[r] < 2 || self.rowsize_integer[r] + self.rowsize_impl_int[r] < self.rowsize[r] - 1 {
                run_dual_detection = false;
                continue;
            }
            let row_lower =
                if self.is_implied_equation_at_upper(row) { self.row_upper[r] } else { self.row_lower[r] };
            let row_upper =
                if self.is_implied_equation_at_lower(row) { self.row_lower[r] } else { self.row_upper[r] };
            if row_upper == row_lower {
                run_dual_detection = false;
                let scale = 1.0 / val;
                if fractionality(row_lower * scale) > self.primal_feastol {
                    continue;
                }
                if !self.row_coefficients_integral(row, scale) {
                    continue;
                }
                return Ok(true);
            }
        }
        if !run_dual_detection {
            return Ok(false);
        }
        let c = col as usize;
        if (self.col_lower[c] != -INF && fractionality(self.col_lower[c]) > self.opt.small_matrix_value)
            || (self.col_upper[c] != INF && fractionality(self.col_upper[c]) > self.opt.small_matrix_value)
        {
            return Ok(false);
        }
        for (row, val) in col_iter!(self, col) {
            let r = row as usize;
            let scale = 1.0 / val;
            if self.row_upper[r] != INF && fractionality(self.row_upper[r] * scale) > self.primal_feastol {
                return Ok(false);
            }
            if self.row_lower[r] != -INF && fractionality(self.row_lower[r] * scale) > self.primal_feastol {
                return Ok(false);
            }
            if !self.row_coefficients_integral(row, scale) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub(crate) fn convert_implied_integer(&mut self, col: i32, row: i32, skip_input_checks: bool) -> SR {
        let c = col as usize;
        if self.col_deleted[c] != 0 {
            return Ok(false);
        }
        if !skip_input_checks {
            if self.integrality[c] != CONTINUOUS {
                return Ok(false);
            }
            let implied = self.is_implied_integer(col)?;
            if !implied {
                return Ok(false);
            }
        }
        self.integrality[c] = IMPLICIT_INTEGER;
        if row != -1 {
            self.rowsize_impl_int[row as usize] += 1;
        } else {
            let mut p = self.colhead[c];
            while p != -1 {
                self.rowsize_impl_int[self.a_row[p as usize] as usize] += 1;
                p = self.a_next[p as usize];
            }
        }
        self.change_col_bounds(col, self.col_lower[c], self.col_upper[c])?;
        Ok(true)
    }

    // ---------------------------------------------------------- the matrix

    #[inline]
    fn row_tree(&mut self) -> Tree<'_> {
        Tree { left: &mut self.ar_left, right: &mut self.ar_right, key: &self.a_col }
    }

    pub(crate) fn link(&mut self, pos: i32) {
        let p = pos as usize;
        let col = self.a_col[p];
        let row = self.a_row[p];
        let c = col as usize;
        let r = row as usize;
        self.a_next[p] = self.colhead[c];
        self.a_prev[p] = -1;
        self.colhead[c] = pos;
        if self.a_next[p] != -1 {
            let n = self.a_next[p] as usize;
            self.a_prev[n] = pos;
        }
        self.colsize[c] += 1;

        self.ar_left[p] = -1;
        self.ar_right[p] = -1;
        let mut root = self.rowroot[r];
        self.row_tree().link(pos, &mut root);
        self.rowroot[r] = root;

        let val = self.a_value[p];
        self.implied_row_bounds.add(row, col, val, &rb!(self));
        self.implied_dual_row_bounds.add(col, row, val, &db!(self));
        self.rowsize[r] += 1;
        if self.integrality[c] == INTEGER {
            self.rowsize_integer[r] += 1;
        } else if self.integrality[c] == IMPLICIT_INTEGER {
            self.rowsize_impl_int[r] += 1;
        }
    }

    pub(crate) fn unlink(&mut self, pos: i32) {
        let p = pos as usize;
        let next = self.a_next[p];
        let prev = self.a_prev[p];
        let col = self.a_col[p];
        let row = self.a_row[p];
        let c = col as usize;
        let r = row as usize;
        if next != -1 {
            self.a_prev[next as usize] = prev;
        }
        if prev != -1 {
            self.a_next[prev as usize] = next;
        } else {
            self.colhead[c] = next;
        }
        self.colsize[c] -= 1;

        if self.col_deleted[c] == 0 {
            if self.colsize[c] == 1 {
                self.singleton_columns.push(col);
            } else {
                self.mark_changed_col(col);
            }
            let val = self.a_value[p];
            self.implied_dual_row_bounds.remove(col, row, val, &db!(self));
        }

        let mut root = self.rowroot[r];
        self.row_tree().unlink(pos, &mut root);
        self.rowroot[r] = root;
        self.rowsize[r] -= 1;
        if self.integrality[c] == INTEGER {
            self.rowsize_integer[r] -= 1;
        } else if self.integrality[c] == IMPLICIT_INTEGER {
            self.rowsize_impl_int[r] -= 1;
        }

        if self.row_deleted[r] == 0 {
            if self.rowsize[r] == 1 {
                self.singleton_rows.push(row);
            } else {
                self.mark_changed_row(row);
            }
            let val = self.a_value[p];
            self.implied_row_bounds.remove(row, col, val, &rb!(self));
        }

        self.reset_row_dual_implied_bounds_derived_from_col(col);
        self.reset_col_implied_bounds_derived_from_row(row);
        self.lifting.clear_row(row);

        self.a_value[p] = 0.0;
        self.freeslots.push(pos);
    }

    #[inline]
    pub(crate) fn mark_changed_row(&mut self, row: i32) {
        let r = row as usize;
        if self.changed_row_flag[r] == 0 {
            self.changed_row_indices.push(row);
            self.changed_row_flag[r] = 1;
        }
        self.single_equation_checked[r] = 0;
    }

    #[inline]
    pub(crate) fn mark_changed_col(&mut self, col: i32) {
        let c = col as usize;
        if self.changed_col_flag[c] == 0 {
            self.changed_col_indices.push(col);
            self.changed_col_flag[c] = 1;
        }
    }

    pub(crate) fn get_max_abs_col_val(&self, col: i32) -> f64 {
        let mut max_val = 0.0f64;
        for (_, v) in col_iter!(self, col) {
            max_val = std_max(v.abs(), max_val);
        }
        max_val
    }

    pub(crate) fn get_max_abs_row_val(&self, row: i32) -> f64 {
        let mut max_val = 0.0f64;
        for (_, v) in row_iter!(self, row) {
            max_val = std_max(v.abs(), max_val);
        }
        max_val
    }

    pub(crate) fn find_nonzero(&mut self, row: i32, col: i32) -> i32 {
        let r = row as usize;
        if self.rowroot[r] == -1 {
            return -1;
        }
        let root = self.rowroot[r];
        let root = self.row_tree().splay(col, root);
        self.rowroot[r] = root;
        if self.a_col[root as usize] == col {
            root
        } else {
            -1
        }
    }

    /// getRowPositions into rowpositions (in order): storeRow
    pub(crate) fn store_row(&mut self, row: i32) {
        let n = in_order_positions(self.rowroot[row as usize], &self.ar_left, &self.ar_right, &mut self.rowpositions);
        self.rp_len = n;
    }

    /// getRowPositions into a fresh vector
    pub(crate) fn get_row_positions(&self, row: i32) -> Vec<i32> {
        let mut v = Vec::new();
        let n = in_order_positions(self.rowroot[row as usize], &self.ar_left, &self.ar_right, &mut v);
        v.truncate(n);
        v
    }

    pub(crate) fn mark_row_deleted(&mut self, row: i32) {
        let r = row as usize;
        debug_assert!(self.row_deleted[r] == 0);
        if self.is_equation(row) && self.eqsize[r] != -1 {
            self.equations.remove(&(self.eqsize[r], row));
            self.eqsize[r] = -1;
        }
        self.changed_row_flag[r] = 1;
        self.row_deleted[r] = 1;
        self.num_deleted_rows += 1;
    }

    pub(crate) fn mark_col_deleted(&mut self, col: i32) {
        let c = col as usize;
        debug_assert!(self.col_deleted[c] == 0);
        self.changed_col_flag[c] = 1;
        self.col_deleted[c] = 1;
        self.num_deleted_cols += 1;
    }

    pub(crate) fn reinsert_equation(&mut self, row: i32) {
        let r = row as usize;
        if self.is_equation(row) && self.eqsize[r] != -1 && self.eqsize[r] != self.rowsize[r] {
            self.equations.remove(&(self.eqsize[r], row));
            self.eqsize[r] = self.rowsize[r];
            self.equations.insert((self.rowsize[r], row));
        }
    }

    pub(crate) fn add_to_matrix(&mut self, row: i32, col: i32, val: f64) {
        let mut pos = self.find_nonzero(row, col);
        self.mark_changed_row(row);
        self.mark_changed_col(col);
        if pos == -1 {
            if let Some(p) = self.freeslots.pop() {
                pos = p;
                let p = p as usize;
                self.a_value[p] = val;
                self.a_row[p] = row;
                self.a_col[p] = col;
                self.a_prev[p] = -1;
            } else {
                pos = self.a_value.len() as i32;
                self.a_value.push(val);
                self.a_row.push(row);
                self.a_col.push(col);
                self.a_next.push(-1);
                self.a_prev.push(-1);
                self.ar_left.push(-1);
                self.ar_right.push(-1);
            }
            self.link(pos);
            self.reset_row_dual_implied_bounds_derived_from_col(col);
            self.reset_col_implied_bounds_derived_from_row(row);
            self.lifting.clear_row(row);
        } else {
            let p = pos as usize;
            let sum = self.a_value[p] + val;
            if sum.abs() <= self.opt.small_matrix_value {
                self.unlink(pos);
            } else {
                self.reset_row_dual_implied_bounds_derived_from_col(col);
                self.reset_col_implied_bounds_derived_from_row(row);
                self.lifting.clear_row(row);
                let old = self.a_value[p];
                self.implied_row_bounds.remove(row, col, old, &rb!(self));
                self.implied_dual_row_bounds.remove(col, row, old, &db!(self));
                self.a_value[p] = sum;
                self.implied_row_bounds.add(row, col, sum, &rb!(self));
                self.implied_dual_row_bounds.add(col, row, sum, &db!(self));
            }
        }
    }

    // ------------------------------------------------------- bound changes

    pub(crate) fn change_col_upper(&mut self, col: i32, mut new_upper: f64) -> R {
        let c = col as usize;
        if self.integrality[c] != CONTINUOUS {
            new_upper = (new_upper + self.primal_feastol).floor();
            if new_upper == self.col_upper[c] {
                return Ok(());
            }
        }
        let old_upper = self.col_upper[c];
        self.col_upper[c] = new_upper;
        self.check_col_bounds(col, None)?;
        let mut p = self.colhead[c];
        while p != -1 {
            let pu = p as usize;
            let row = self.a_row[pu];
            let val = self.a_value[pu];
            self.implied_row_bounds.updated_var_upper(row, col, val, old_upper, &rb!(self));
            self.mark_changed_row(row);
            p = self.a_next[pu];
        }
        Ok(())
    }

    pub(crate) fn change_col_lower(&mut self, col: i32, mut new_lower: f64) -> R {
        let c = col as usize;
        if self.integrality[c] != CONTINUOUS {
            new_lower = (new_lower - self.primal_feastol).ceil();
            if new_lower == self.col_lower[c] {
                return Ok(());
            }
        }
        let old_lower = self.col_lower[c];
        self.col_lower[c] = new_lower;
        self.check_col_bounds(col, None)?;
        let mut p = self.colhead[c];
        while p != -1 {
            let pu = p as usize;
            let row = self.a_row[pu];
            let val = self.a_value[pu];
            self.implied_row_bounds.updated_var_lower(row, col, val, old_lower, &rb!(self));
            self.mark_changed_row(row);
            p = self.a_next[pu];
        }
        Ok(())
    }

    pub(crate) fn change_col_bounds(&mut self, col: i32, new_lower: f64, new_upper: f64) -> R {
        if new_lower > self.col_upper[col as usize] {
            self.change_col_upper(col, new_upper)?;
            self.change_col_lower(col, new_lower)?;
        } else {
            self.change_col_lower(col, new_lower)?;
            self.change_col_upper(col, new_upper)?;
        }
        Ok(())
    }

    pub(crate) fn check_col_bounds(&self, col: i32, is_fixed: Option<&mut bool>) -> R {
        let c = col as usize;
        let bound_diff = self.col_upper[c] - self.col_lower[c];
        let mut fixed = false;
        if bound_diff <= self.primal_feastol
            && (bound_diff <= self.opt.small_matrix_value
                || self.get_max_abs_col_val(col) * bound_diff <= self.primal_feastol)
        {
            if bound_diff < -self.primal_feastol {
                return Err(Stop::PrimalInfeasible);
            }
            if self.col_lower[c].abs() == INF {
                return Err(Stop::DualInfeasible);
            }
            fixed = true;
        }
        if let Some(f) = is_fixed {
            *f = fixed;
        }
        Ok(())
    }

    pub(crate) fn change_row_dual_upper(&mut self, row: i32, new_upper: f64) {
        let r = row as usize;
        let old_upper = self.row_dual_upper[r];
        self.row_dual_upper[r] = new_upper;
        let mut w = PreOrder::new(self.rowroot[r]);
        while let Some(p) = w.next(&self.ar_left, &self.ar_right) {
            let col = self.a_col[p];
            let val = self.a_value[p];
            self.implied_dual_row_bounds.updated_var_upper(col, row, val, old_upper, &db!(self));
            self.mark_changed_col(col);
        }
    }

    pub(crate) fn change_row_dual_lower(&mut self, row: i32, new_lower: f64) {
        let r = row as usize;
        let old_lower = self.row_dual_lower[r];
        self.row_dual_lower[r] = new_lower;
        let mut w = PreOrder::new(self.rowroot[r]);
        while let Some(p) = w.next(&self.ar_left, &self.ar_right) {
            let col = self.a_col[p];
            let val = self.a_value[p];
            self.implied_dual_row_bounds.updated_var_lower(col, row, val, old_lower, &db!(self));
            self.mark_changed_col(col);
        }
    }

    pub(crate) fn change_impl_col_upper(&mut self, col: i32, new_upper: f64, origin_row: i32) {
        let c = col as usize;
        let old_impl_upper = self.impl_col_upper[c];
        let old_upper_source = self.col_upper_source[c];
        if old_impl_upper >= self.col_upper[c] - self.primal_feastol
            && new_upper < self.col_upper[c] - self.primal_feastol
        {
            self.mark_changed_col(col);
        }
        let new_implied_free = self.is_lower_implied(col)
            && old_impl_upper > self.col_upper[c] + self.primal_feastol
            && new_upper <= self.col_upper[c] + self.primal_feastol;
        if old_upper_source != origin_row {
            if old_upper_source != -1 && old_upper_source != self.col_lower_source[c] {
                self.col_impl_source_by_row[old_upper_source as usize].remove(&col);
            }
            if origin_row != -1 {
                self.col_impl_source_by_row[origin_row as usize].insert(col);
            }
            self.col_upper_source[c] = origin_row;
        }
        self.impl_col_upper[c] = new_upper;
        if !new_implied_free && std_min(old_impl_upper, new_upper) >= self.col_upper[c] {
            return;
        }
        let mut p = self.colhead[c];
        while p != -1 {
            let pu = p as usize;
            let row = self.a_row[pu];
            let val = self.a_value[pu];
            self.implied_row_bounds.updated_impl_var_upper(row, col, val, old_impl_upper, old_upper_source, &rb!(self));
            if new_implied_free && self.is_dual_implied_free(row) {
                self.substitution_opportunities.push((row, col));
            }
            self.mark_changed_row(row);
            p = self.a_next[pu];
        }
    }

    pub(crate) fn change_impl_col_lower(&mut self, col: i32, new_lower: f64, origin_row: i32) {
        let c = col as usize;
        let old_impl_lower = self.impl_col_lower[c];
        let old_lower_source = self.col_lower_source[c];
        if old_impl_lower <= self.col_lower[c] + self.primal_feastol
            && new_lower > self.col_lower[c] + self.primal_feastol
        {
            self.mark_changed_col(col);
        }
        let new_implied_free = self.is_upper_implied(col)
            && old_impl_lower < self.col_lower[c] - self.primal_feastol
            && new_lower >= self.col_lower[c] - self.primal_feastol;
        if old_lower_source != origin_row {
            if old_lower_source != -1 && old_lower_source != self.col_upper_source[c] {
                self.col_impl_source_by_row[old_lower_source as usize].remove(&col);
            }
            if origin_row != -1 {
                self.col_impl_source_by_row[origin_row as usize].insert(col);
            }
            self.col_lower_source[c] = origin_row;
        }
        self.impl_col_lower[c] = new_lower;
        if !new_implied_free && std_max(old_impl_lower, new_lower) <= self.col_lower[c] {
            return;
        }
        let mut p = self.colhead[c];
        while p != -1 {
            let pu = p as usize;
            let row = self.a_row[pu];
            let val = self.a_value[pu];
            self.implied_row_bounds.updated_impl_var_lower(row, col, val, old_impl_lower, old_lower_source, &rb!(self));
            if new_implied_free && self.is_dual_implied_free(row) {
                self.substitution_opportunities.push((row, col));
            }
            self.mark_changed_row(row);
            p = self.a_next[pu];
        }
    }

    pub(crate) fn change_impl_row_dual_upper(&mut self, row: i32, new_upper: f64, origin_col: i32) {
        let r = row as usize;
        let old_impl_upper = self.impl_row_dual_upper[r];
        let old_upper_source = self.row_dual_upper_source[r];
        let dtol = self.opt.dual_feasibility_tolerance;
        if old_impl_upper >= -dtol && new_upper < -dtol {
            self.mark_changed_row(row);
        }
        let new_dual_implied = !self.is_dual_implied_free(row)
            && old_impl_upper > self.row_dual_upper[r] + dtol
            && new_upper <= self.row_dual_upper[r] + dtol;
        if old_upper_source != origin_col {
            if old_upper_source != -1 && old_upper_source != self.row_dual_lower_source[r] {
                self.impl_row_dual_source_by_col[old_upper_source as usize].remove(&row);
            }
            if origin_col != -1 {
                self.impl_row_dual_source_by_col[origin_col as usize].insert(row);
            }
            self.row_dual_upper_source[r] = origin_col;
        }
        self.impl_row_dual_upper[r] = new_upper;
        if !new_dual_implied && std_min(old_impl_upper, new_upper) >= self.row_dual_upper[r] {
            return;
        }
        let mut w = PreOrder::new(self.rowroot[r]);
        while let Some(p) = w.next(&self.ar_left, &self.ar_right) {
            let col = self.a_col[p];
            let val = self.a_value[p];
            self.implied_dual_row_bounds.updated_impl_var_upper(
                col,
                row,
                val,
                old_impl_upper,
                old_upper_source,
                &db!(self),
            );
            self.mark_changed_col(col);
            if new_dual_implied && self.is_implied_free(col) {
                self.substitution_opportunities.push((row, col));
            }
        }
    }

    pub(crate) fn change_impl_row_dual_lower(&mut self, row: i32, new_lower: f64, origin_col: i32) {
        let r = row as usize;
        let old_impl_lower = self.impl_row_dual_lower[r];
        let old_lower_source = self.row_dual_lower_source[r];
        let dtol = self.opt.dual_feasibility_tolerance;
        if old_impl_lower <= dtol && new_lower > dtol {
            self.mark_changed_row(row);
        }
        let new_dual_implied = !self.is_dual_implied_free(row)
            && old_impl_lower < self.row_dual_lower[r] - dtol
            && new_lower >= self.row_dual_lower[r] - dtol;
        if old_lower_source != origin_col {
            if old_lower_source != -1 && old_lower_source != self.row_dual_upper_source[r] {
                self.impl_row_dual_source_by_col[old_lower_source as usize].remove(&row);
            }
            if origin_col != -1 {
                self.impl_row_dual_source_by_col[origin_col as usize].insert(row);
            }
            self.row_dual_lower_source[r] = origin_col;
        }
        self.impl_row_dual_lower[r] = new_lower;
        if !new_dual_implied && std_max(old_impl_lower, new_lower) <= self.row_dual_lower[r] {
            return;
        }
        let mut w = PreOrder::new(self.rowroot[r]);
        while let Some(p) = w.next(&self.ar_left, &self.ar_right) {
            let col = self.a_col[p];
            let val = self.a_value[p];
            self.implied_dual_row_bounds.updated_impl_var_lower(
                col,
                row,
                val,
                old_impl_lower,
                old_lower_source,
                &db!(self),
            );
            self.mark_changed_col(col);
            if new_dual_implied && self.is_implied_free(col) {
                self.substitution_opportunities.push((row, col));
            }
        }
    }

    // ------------------------------------------------------ implied bounds

    /// checkUpdateRowDualImpliedBounds: (ok, dualRowLower, dualRowUpper)
    pub(crate) fn check_update_row_dual_implied_bounds(&self, col: i32) -> (bool, f64, f64) {
        let c = col as usize;
        let implied_margin = if self.colsize[c] != 1 { self.primal_feastol } else { -self.primal_feastol };
        let dual_row_lower =
            if self.is_lower_strictly_implied(col, Some(implied_margin)) { self.col_cost[c] } else { -INF };
        let dual_row_upper =
            if self.is_upper_strictly_implied(col, Some(implied_margin)) { self.col_cost[c] } else { INF };
        let ok = (dual_row_lower != -INF && self.implied_dual_row_bounds.num_inf_sum_upper_orig(col) <= 1)
            || (dual_row_upper != INF && self.implied_dual_row_bounds.num_inf_sum_lower_orig(col) <= 1);
        (ok, dual_row_lower, dual_row_upper)
    }

    pub(crate) fn update_row_dual_implied_bounds_nz(&mut self, row: i32, col: i32, val: f64) {
        let (ok, dual_row_lower, dual_row_upper) = self.check_update_row_dual_implied_bounds(col);
        if !ok {
            return;
        }
        let threshold = 1000.0 * self.opt.dual_feasibility_tolerance;
        let check = |s: &mut Self, dual_row_bnd: f64, residual_act: f64, direction: i32| {
            if direction as f64 * residual_act <= -INF {
                return;
            }
            let implied_bound = ((CDouble::from(dual_row_bnd) - residual_act) / val).to_f64();
            if implied_bound.abs() * TINY > s.opt.dual_feasibility_tolerance {
                return;
            }
            if direction as f64 * val > 0.0 {
                if implied_bound < s.impl_row_dual_upper[row as usize] - threshold {
                    s.change_impl_row_dual_upper(row, implied_bound, col);
                }
            } else if implied_bound > s.impl_row_dual_lower[row as usize] + threshold {
                s.change_impl_row_dual_lower(row, implied_bound, col);
            }
        };
        if dual_row_upper != INF {
            let res = self.implied_dual_row_bounds.residual_sum_lower_orig(col, row, val, -1, -INF, -INF, &db!(self));
            check(self, dual_row_upper, res, 1);
        }
        if dual_row_lower != -INF {
            let res = self.implied_dual_row_bounds.residual_sum_upper_orig(col, row, val, -1, -INF, -INF, &db!(self));
            check(self, dual_row_lower, res, -1);
        }
    }

    pub(crate) fn update_row_dual_implied_bounds(&mut self, col: i32) {
        if !self.check_update_row_dual_implied_bounds(col).0 {
            return;
        }
        let mut p = self.colhead[col as usize];
        while p != -1 {
            let pu = p as usize;
            let row = self.a_row[pu];
            let val = self.a_value[pu];
            self.update_row_dual_implied_bounds_nz(row, col, val);
            p = self.a_next[pu];
        }
    }

    /// checkUpdateColImpliedBounds: (ok, rowLower, rowUpper)
    pub(crate) fn check_update_col_implied_bounds(&self, row: i32) -> (bool, f64, f64) {
        let r = row as usize;
        let my_row_lower = if self.is_implied_equation_at_upper(row) { self.row_upper[r] } else { self.row_lower[r] };
        let my_row_upper = if self.is_implied_equation_at_lower(row) { self.row_lower[r] } else { self.row_upper[r] };
        let ok = (my_row_lower != -INF && self.implied_row_bounds.num_inf_sum_upper_orig(row) <= 1)
            || (my_row_upper != INF && self.implied_row_bounds.num_inf_sum_lower_orig(row) <= 1);
        (ok, my_row_lower, my_row_upper)
    }

    pub(crate) fn update_col_implied_bounds_nz(&mut self, row: i32, col: i32, val: f64) -> R {
        let (ok, row_lower, row_upper) = self.check_update_col_implied_bounds(row);
        if !ok {
            return Ok(());
        }
        let threshold = 1000.0 * self.primal_feastol;
        let check = |s: &mut Self, row_bnd: f64, residual_act: f64, direction: i32| -> R {
            if direction as f64 * residual_act <= -INF {
                return Ok(());
            }
            let implied_bound = ((CDouble::from(row_bnd) - residual_act) / val).to_f64();
            if implied_bound.abs() * TINY > s.primal_feastol {
                return Ok(());
            }
            let use_impl_bound = match s.mip {
                None => true,
                Some(m) => s.ps.orig_row_index[row as usize] < m.orig_num_row,
            };
            let c = col as usize;
            if direction as f64 * val > 0.0 {
                let update_col_bound = s.mip.is_some()
                    && ((s.integrality[c] != CONTINUOUS && implied_bound < s.col_upper[c] - s.primal_feastol)
                        || (!use_impl_bound && implied_bound < s.col_upper[c] - threshold));
                if use_impl_bound && implied_bound < s.impl_col_upper[c] - threshold {
                    s.change_impl_col_upper(col, implied_bound, row);
                }
                if update_col_bound {
                    s.change_col_upper(col, implied_bound)?;
                }
            } else {
                let update_col_bound = s.mip.is_some()
                    && ((s.integrality[c] != CONTINUOUS && implied_bound > s.col_lower[c] + s.primal_feastol)
                        || (!use_impl_bound && implied_bound > s.col_lower[c] + threshold));
                if use_impl_bound && implied_bound > s.impl_col_lower[c] + threshold {
                    s.change_impl_col_lower(col, implied_bound, row);
                }
                if update_col_bound {
                    s.change_col_lower(col, implied_bound)?;
                }
            }
            Ok(())
        };
        if row_upper != INF {
            let res = self.implied_row_bounds.residual_sum_lower_orig(row, col, val, -1, -INF, -INF, &rb!(self));
            check(self, row_upper, res, 1)?;
        }
        if row_lower != -INF {
            let res = self.implied_row_bounds.residual_sum_upper_orig(row, col, val, -1, -INF, -INF, &rb!(self));
            check(self, row_lower, res, -1)?;
        }
        Ok(())
    }

    pub(crate) fn update_col_implied_bounds(&mut self, row: i32) -> R {
        if !self.check_update_col_implied_bounds(row).0 {
            return Ok(());
        }
        let mut w = PreOrder::new(self.rowroot[row as usize]);
        while let Some(p) = w.next(&self.ar_left, &self.ar_right) {
            let col = self.a_col[p];
            let val = self.a_value[p];
            self.update_col_implied_bounds_nz(row, col, val)?;
        }
        Ok(())
    }

    pub(crate) fn reset_col_implied_bounds(&mut self, col: i32, row: i32) {
        let c = col as usize;
        if self.col_deleted[c] == 0 {
            if self.col_lower_source[c] != -1 && (row == -1 || self.col_lower_source[c] == row) {
                self.change_impl_col_lower(col, -INF, -1);
            }
            if self.col_upper_source[c] != -1 && (row == -1 || self.col_upper_source[c] == row) {
                self.change_impl_col_upper(col, INF, -1);
            }
        } else if row != -1 && self.row_deleted[row as usize] == 0 {
            self.col_impl_source_by_row[row as usize].remove(&col);
        }
    }

    pub(crate) fn reset_row_dual_implied_bounds(&mut self, row: i32, col: i32) {
        let r = row as usize;
        if self.row_deleted[r] == 0 {
            if self.row_dual_lower_source[r] != -1 && (col == -1 || self.row_dual_lower_source[r] == col) {
                self.change_impl_row_dual_lower(row, -INF, -1);
            }
            if self.row_dual_upper_source[r] != -1 && (col == -1 || self.row_dual_upper_source[r] == col) {
                self.change_impl_row_dual_upper(row, INF, -1);
            }
        } else if col != -1 && self.col_deleted[col as usize] == 0 {
            self.impl_row_dual_source_by_col[col as usize].remove(&row);
        }
    }

    pub(crate) fn reset_col_implied_bounds_derived_from_row(&mut self, row: i32) {
        let r = row as usize;
        if self.col_impl_source_by_row[r].is_empty() {
            return;
        }
        let affected = std::mem::take(&mut self.col_impl_source_by_row[r]);
        for &col in &affected {
            self.reset_col_implied_bounds(col, row);
        }
    }

    pub(crate) fn reset_row_dual_implied_bounds_derived_from_col(&mut self, col: i32) {
        let c = col as usize;
        if self.impl_row_dual_source_by_col[c].is_empty() {
            return;
        }
        let affected = std::mem::take(&mut self.impl_row_dual_source_by_col[c]);
        for &row in &affected {
            self.reset_row_dual_implied_bounds(row, col);
        }
    }

    /// checkLimits
    pub(crate) fn check_limits(&mut self) -> R {
        let numreductions = self.ps.num_reductions;
        if (numreductions & 1023) == 0 {
            self.check_time_limit()?;
        }
        if numreductions >= self.reduction_limit {
            Err(Stop::Stopped)
        } else {
            Ok(())
        }
    }

    pub(crate) fn check_time_limit(&self) -> R {
        if self.opt.time_limit < INF && self.host.timer_read() >= self.opt.time_limit {
            return Err(Stop::Stopped);
        }
        Ok(())
    }

    pub(crate) fn store_current_problem_size(&mut self) {
        self.old_num_col = self.num_col - self.num_deleted_cols;
        self.old_num_row = self.num_row - self.num_deleted_rows;
    }

    pub(crate) fn problem_size_reduction(&self) -> f64 {
        let col_reduction =
            100.0 * (self.old_num_col - (self.num_col - self.num_deleted_cols)) as f64 / self.old_num_col as f64;
        let row_reduction =
            100.0 * (self.old_num_row - (self.num_row - self.num_deleted_rows)) as f64 / self.old_num_row as f64;
        std_max(row_reduction, col_reduction)
    }

    pub(crate) fn num_nonzeros(&self) -> i32 {
        (self.a_value.len() - self.freeslots.len()) as i32
    }

    pub(crate) fn silent_log(&self) -> bool {
        matches!(self.mip, Some(m) if m.num_restarts > 0)
    }

    /// The MIP solver's clique table. Calls into C++ (bound changes,
    /// implications.runProbing, shrinking) may change it: no borrow may
    /// live across one
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn cliquetable(&self) -> &mut CliqueTable {
        // SAFETY: a live table for the presolve run (MipInfo), only reached
        // here and from C++ calls, across which no borrow is held
        unsafe { &mut *self.mip.expect("MIP presolve").cliquetable }
    }

    /// The MIP solver's implications, as cliquetable
    #[allow(clippy::mut_from_ref)]
    pub(crate) fn implications(&self) -> &mut Implications {
        // SAFETY: as in cliquetable
        unsafe { &mut *self.mip.expect("MIP presolve").implications }
    }
}

/// std::min(a, b): b < a ? b : a
#[inline]
pub(crate) fn std_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// std::max(a, b): a < b ? b : a
#[inline]
pub(crate) fn std_max(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// The positions of a row tree in order into buf (whose entries past the
/// returned length are kept); returns the number of positions
pub(crate) fn in_order_positions(root: i32, left: &[i32], right: &[i32], buf: &mut Vec<i32>) -> usize {
    let mut n = 0usize;
    if root == -1 {
        return 0;
    }
    let mut put = |buf: &mut Vec<i32>, v: i32| {
        if n < buf.len() {
            buf[n] = v;
        } else {
            buf.push(v);
        }
        n += 1;
    };
    let mut stack = SStack::new();
    stack.push(-1);
    let mut cur = root;
    while left[cur as usize] != -1 {
        stack.push(cur);
        cur = left[cur as usize];
    }
    loop {
        put(buf, cur);
        let c = cur as usize;
        if right[c] != -1 {
            cur = right[c];
            while left[cur as usize] != -1 {
                stack.push(cur);
                cur = left[cur as usize];
            }
        } else {
            cur = stack.pop().unwrap();
        }
        if cur == -1 {
            break;
        }
    }
    n
}
