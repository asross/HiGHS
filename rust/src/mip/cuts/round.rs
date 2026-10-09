//! The data of one separation round (HighsSeparation::separationRound):
//! views of the LP relaxation and its solution, the global domain's column
//! bounds, a copy of the LP rows, and the per-column data of
//! HighsTransformedLp's constructor packed into one array.
//!
//! # The boundary
//!
//! C++ fills a [`CSepaLp`] (highs/mip/HighsTransformedLpRust.cpp) with
//! pointer+length pairs and a [`Host`] of callbacks, and owns everything
//! the pointers reach for the lifetime of the round:
//! - the LP (bounds, column-wise matrix) and its solution do not change
//!   during a round;
//! - the global domain's column bounds and row activities can change in
//!   any call of `Host::add_cut` (the cut pool propagates and extracts
//!   cliques), so column bounds are read through raw pointers each time,
//!   and the slack bounds of LP rows, which come from the domain's
//!   activities, are cached per [`SepaRound::generation`], which every
//!   add_cut bumps; the column bounds are cached the same way, next to the
//!   column's other data;
//! - LP rows are copied: a cut row's storage in the cut pool can move
//!   when cuts are added.

use crate::hvector::OwnedHVec;
use crate::lp_data::lp_handle::LpHandle;
use crate::util::fma::ClangFma;
use crate::util::sparse_vector_sum::HighsSparseVectorSum;
use std::ffi::c_void;

pub const K_HIGHS_INF: f64 = f64::INFINITY;

/// HighsImplications::VarBound
#[repr(C)]
#[derive(Clone, Copy, Default, Debug)]
pub struct VarBound {
    pub coef: f64,
    pub constant: f64,
}

impl VarBound {
    #[inline]
    pub fn min_value(&self) -> f64 {
        (crate::util::cdouble::CDouble::from(self.constant) + self.coef.min(0.0)).to_f64()
    }
    #[inline]
    pub fn max_value(&self) -> f64 {
        (crate::util::cdouble::CDouble::from(self.constant) + self.coef.max(0.0)).to_f64()
    }
}

/// Callbacks into C++; `ctx` is the C++ side's state for the round (or
/// for a conflict).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Host {
    pub ctx: *mut c_void,
    /// HighsImplications::cleanupVarbounds(col)
    pub cleanup_varbounds: unsafe extern "C" fn(*mut c_void, i32),
    /// globaldom.infeasible()
    pub dom_infeasible: unsafe extern "C" fn(*mut c_void) -> bool,
    /// HighsImplications::getBestVub/Vlb(col, lpSolution, bound, globaldom):
    /// returns the bound's column (-1 if none), updates *bound
    pub best_vub: unsafe extern "C" fn(*mut c_void, i32, *mut f64, *mut VarBound) -> i32,
    pub best_vlb: unsafe extern "C" fn(*mut c_void, i32, *mut f64, *mut VarBound) -> i32,
    /// HighsLpRelaxation::slackLower/slackUpper(row, globaldom)
    pub slack_lower: unsafe extern "C" fn(*mut c_void, i32) -> f64,
    pub slack_upper: unsafe extern "C" fn(*mut c_void, i32) -> f64,
    /// HighsLpRelaxation::getRow; also the row's integrality and maximal
    /// absolute coefficient
    pub get_row:
        unsafe extern "C" fn(*mut c_void, i32, *mut i32, *mut *const i32, *mut *const f64, *mut u8, *mut f64),
    /// HighsCutPool::addCut(mipsolver, inds, vals, len, rhs, integral, true,
    /// true, isConflict); sorts inds/vals
    pub add_cut: unsafe extern "C" fn(*mut c_void, *mut i32, *mut f64, i32, f64, bool, bool) -> i32,
    /// cutpool.getNumCuts(), cutpool.getNumAvailableCuts()
    pub num_cuts: unsafe extern "C" fn(*mut c_void) -> i32,
    pub num_available_cuts: unsafe extern "C" fn(*mut c_void) -> i32,
    /// nodequeue.numNodesDown/Up(col)
    pub num_nodes_down: unsafe extern "C" fn(*mut c_void, i32) -> i64,
    pub num_nodes_up: unsafe extern "C" fn(*mut c_void, i32) -> i64,
    /// lpRelaxation.getNumLpIterations()
    pub num_lp_iterations: unsafe extern "C" fn(*mut c_void) -> i64,
}

/// What C++ passes to start a round
#[repr(C)]
pub struct CSepaLp {
    pub num_col: i32,
    pub num_row: i32,
    /// globaldom.col_lower_/col_upper_ (num_col; written by C++ during
    /// add_cut)
    pub col_lower: *const f64,
    pub col_upper: *const f64,
    /// the LP solution (num_col, num_row, num_row)
    pub col_value: *const f64,
    pub row_value: *const f64,
    pub row_dual: *const f64,
    /// the LP's row bounds (num_row)
    pub row_lower: *const f64,
    pub row_upper: *const f64,
    /// the LP's column-wise matrix (num_col + 1 starts)
    pub a_start: *const i32,
    pub a_index: *const i32,
    pub a_value: *const f64,
    /// mipsolver.model_->integrality_ (num_col; 0 = continuous)
    pub integrality: *const u8,
    pub continuous_cols: *const i32,
    pub num_continuous_cols: i32,
    pub integral_cols: *const i32,
    pub num_integral_cols: i32,
    pub feastol: f64,
    pub epsilon: f64,
    pub small_matrix_value: f64,
    pub parallel_lock_active: bool,
    pub mip_pool_soft_limit: i32,
    pub host: Host,
    /// The LP solver (its basis inverse rows and DSE weights)
    pub lph: *mut LpHandle,
}

/// HighsTransformedLp::BoundType
pub const K_SIMPLE_UB: u8 = 0;
pub const K_SIMPLE_LB: u8 = 1;
pub const K_VARIABLE_UB: u8 = 2;
pub const K_VARIABLE_LB: u8 = 3;

/// The per-column data of HighsTransformedLp, for the columns and then the
/// row slacks, one cache line each. bound_dist = min(lb_dist, ub_dist), as
/// in every branch of the C++ constructor.
#[derive(Clone, Copy, Default)]
#[repr(C, align(64))]
pub struct ColData {
    /// the global bounds (getLb, getUb), valid if gen == the round's
    /// generation
    pub lb: f64,
    pub ub: f64,
    pub gen: u32,
    pub lb_dist: f64,
    pub ub_dist: f64,
    pub simple_lb_dist: f64,
    pub simple_ub_dist: f64,
    pub vub_col: i32,
    pub vlb_col: i32,
    pub bound_type: u8,
    /// lprelaxation.isColIntegral
    pub integral: bool,
    /// HighsTransformedLp::isFractional
    pub fractional: bool,
}

impl ColData {
    #[inline]
    pub fn bound_dist(&self) -> f64 {
        // std::min(lbDist, ubDist)
        if self.ub_dist < self.lb_dist {
            self.ub_dist
        } else {
            self.lb_dist
        }
    }
}

/// A slice that C++ owns and does not change during the round
#[derive(Clone, Copy)]
pub struct View<T> {
    ptr: *const T,
    len: usize,
}

impl<T> View<T> {
    fn new(ptr: *const T, len: i64) -> Self {
        View { ptr, len: if ptr.is_null() || len <= 0 { 0 } else { len as usize } }
    }
    #[inline]
    pub fn get(&self) -> &[T] {
        if self.len == 0 {
            &[]
        } else {
            // SAFETY: C++ keeps the data valid and unchanged for the round
            // (see the module comment)
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        }
    }
}

/// An array that C++ may change in a callback: read element by element
#[derive(Clone, Copy)]
pub struct Live {
    ptr: *const f64,
    len: usize,
}

impl Live {
    pub fn new(ptr: *const f64, len: usize) -> Self {
        Live { ptr: if len == 0 { std::ptr::NonNull::dangling().as_ptr() } else { ptr }, len }
    }
    #[inline]
    pub fn at(&self, i: usize) -> f64 {
        assert!(i < self.len);
        // SAFETY: in bounds; C++ only writes it during callbacks, when no
        // Rust reference to it exists
        unsafe { self.ptr.add(i).read() }
    }
}

pub struct SepaRound {
    pub num_col: usize,
    pub num_row: usize,
    pub col_lower: Live,
    pub col_upper: Live,
    pub col_value: View<f64>,
    pub row_value: View<f64>,
    pub row_dual: View<f64>,
    pub row_lower: View<f64>,
    pub row_upper: View<f64>,
    pub a_start: View<i32>,
    pub a_index: View<i32>,
    pub a_value: View<f64>,
    pub integrality: View<u8>,
    pub continuous_cols: View<i32>,
    pub integral_cols: View<i32>,
    pub feastol: f64,
    pub epsilon: f64,
    pub small_matrix_value: f64,
    pub parallel_lock_active: bool,
    pub mip_pool_soft_limit: i32,
    pub host: Host,
    /// The LP solver, and the vector of its basis inverse rows
    pub lph: *mut LpHandle,
    pub row_ep: OwnedHVec,

    /// LP rows (row-wise copy)
    pub ar_start: Vec<usize>,
    pub ar_index: Vec<i32>,
    pub ar_value: Vec<f64>,
    pub row_max_abs: Vec<f64>,

    /// HighsTransformedLp's data
    pub cols: Vec<ColData>,
    pub vub: Vec<VarBound>,
    pub vlb: Vec<VarBound>,
    /// bumped by add_cut, which can change the global domain
    pub generation: u32,
    /// HighsTransformedLp's vectorsum
    pub vectorsum: HighsSparseVectorSum,
    /// HighsLpAggregator's vectorsum
    pub aggr: HighsSparseVectorSum,
}

impl SepaRound {
    /// HighsTransformedLp's constructor
    ///
    /// # Safety
    /// The pointers of `c` must be valid as described for [`CSepaLp`]
    pub unsafe fn new(c: &CSepaLp) -> Box<SepaRound> {
        let num_col = c.num_col as usize;
        let num_row = c.num_row as usize;
        let n = num_col + num_row;
        let mut r = Box::new(SepaRound {
            num_col,
            num_row,
            col_lower: Live { ptr: c.col_lower, len: num_col },
            col_upper: Live { ptr: c.col_upper, len: num_col },
            col_value: View::new(c.col_value, num_col as i64),
            row_value: View::new(c.row_value, num_row as i64),
            row_dual: View::new(c.row_dual, num_row as i64),
            row_lower: View::new(c.row_lower, num_row as i64),
            row_upper: View::new(c.row_upper, num_row as i64),
            a_start: View::new(c.a_start, num_col as i64 + 1),
            a_index: View::new(c.a_index, if c.a_start.is_null() { 0 } else { *c.a_start.add(num_col) as i64 }),
            a_value: View::new(c.a_value, if c.a_start.is_null() { 0 } else { *c.a_start.add(num_col) as i64 }),
            integrality: View::new(c.integrality, num_col as i64),
            continuous_cols: View::new(c.continuous_cols, c.num_continuous_cols as i64),
            integral_cols: View::new(c.integral_cols, c.num_integral_cols as i64),
            feastol: c.feastol,
            epsilon: c.epsilon,
            small_matrix_value: c.small_matrix_value,
            parallel_lock_active: c.parallel_lock_active,
            mip_pool_soft_limit: c.mip_pool_soft_limit,
            host: c.host,
            lph: c.lph,
            row_ep: OwnedHVec::new(0),
            ar_start: Vec::with_capacity(num_row + 1),
            ar_index: Vec::new(),
            ar_value: Vec::new(),
            row_max_abs: vec![0.0; num_row],
            cols: vec![ColData { vub_col: -1, vlb_col: -1, gen: u32::MAX, ..Default::default() }; n],
            vub: vec![VarBound::default(); num_col],
            vlb: vec![VarBound::default(); num_col],
            generation: 0,
            vectorsum: HighsSparseVectorSum::new(n),
            aggr: HighsSparseVectorSum::new(n),
        });
        r.copy_rows();
        r.setup();
        r
    }

    fn copy_rows(&mut self) {
        let h = &self.host;
        self.ar_start.push(0);
        for row in 0..self.num_row {
            let (mut len, mut inds, mut vals, mut integral, mut maxabs) =
                (0i32, std::ptr::null(), std::ptr::null(), 0u8, 0.0);
            // SAFETY: C++ fills the outputs with a row valid until the next
            // call into C++
            unsafe {
                (h.get_row)(h.ctx, row as i32, &mut len, &mut inds, &mut vals, &mut integral, &mut maxabs);
                self.ar_index.extend_from_slice(crate::ffi::sl(inds, len));
                self.ar_value.extend_from_slice(crate::ffi::sl(vals, len));
            }
            self.ar_start.push(self.ar_index.len());
            self.cols[self.num_col + row].integral = integral != 0;
            self.row_max_abs[row] = maxabs;
        }
        let integrality = self.integrality;
        for (c, &t) in self.cols[..self.num_col].iter_mut().zip(integrality.get()) {
            c.integral = t != 0;
        }
    }

    #[inline]
    pub fn row(&self, row: usize) -> (&[i32], &[f64]) {
        let (s, e) = (self.ar_start[row], self.ar_start[row + 1]);
        (&self.ar_index[s..e], &self.ar_value[s..e])
    }

    #[inline]
    pub fn dom_infeasible(&self) -> bool {
        // SAFETY: a C++ query
        unsafe { (self.host.dom_infeasible)(self.host.ctx) }
    }

    /// slackLower/slackUpper(row, globaldom), cached until the next add_cut
    #[inline]
    pub fn slack_bounds(&mut self, row: usize) -> (f64, f64) {
        self.bounds(self.num_col + row)
    }

    /// The global bounds of a column or row slack (getLb, getUb), cached
    /// until the next add_cut
    #[inline(always)]
    pub fn bounds(&mut self, col: usize) -> (f64, f64) {
        let d = &self.cols[col];
        if d.gen == self.generation {
            (d.lb, d.ub)
        } else {
            self.load_bounds(col)
        }
    }

    #[cold]
    #[inline(never)]
    fn load_bounds(&mut self, col: usize) -> (f64, f64) {
        let (lb, ub) = if col < self.num_col {
            (self.col_lower.at(col), self.col_upper.at(col))
        } else {
            let row = (col - self.num_col) as i32;
            // SAFETY: C++ queries
            unsafe {
                (
                    (self.host.slack_lower)(self.host.ctx, row),
                    (self.host.slack_upper)(self.host.ctx, row),
                )
            }
        };
        let d = &mut self.cols[col];
        d.lb = lb;
        d.ub = ub;
        d.gen = self.generation;
        (lb, ub)
    }

    /// HighsCutPool::addCut; the domain may change
    pub fn add_cut(&mut self, inds: &mut [i32], vals: &mut [f64], rhs: f64, integral: bool, conflict: bool) -> i32 {
        self.generation = self.generation.wrapping_add(1);
        if self.generation == u32::MAX {
            for d in &mut self.cols {
                d.gen = u32::MAX;
            }
            self.generation = 0;
        }
        // SAFETY: inds and vals have len entries; C++ only reorders them
        unsafe {
            (self.host.add_cut)(
                self.host.ctx,
                inds.as_mut_ptr(),
                vals.as_mut_ptr(),
                inds.len() as i32,
                rhs,
                integral,
                conflict,
            )
        }
    }

    pub fn num_cuts(&self) -> i32 {
        // SAFETY: a C++ query
        unsafe { (self.host.num_cuts)(self.host.ctx) }
    }

    /// The constructor of HighsTransformedLp after the allocations
    fn setup(&mut self) {
        let feastol = self.feastol;
        let col_value = self.col_value;
        let col_value = col_value.get();
        let h = std::ptr::addr_of!(self.host);
        // SAFETY: the callbacks are C++ functions taking the context
        let host = unsafe { &*h };

        let continuous_cols = self.continuous_cols;
        for &col in continuous_cols.get() {
            let c = col as usize;
            if !self.parallel_lock_active_flag() {
                unsafe { (host.cleanup_varbounds)(host.ctx, col) };
            }
            if self.dom_infeasible() {
                return;
            }
            let (lb, ub) = (self.col_lower.at(c), self.col_upper.at(c));
            // globaldom_.isFixed(col)
            if lb == ub {
                continue;
            }
            let x = col_value[c];
            let mut bestub = ub;
            let d = &mut self.cols[c];
            d.simple_ub_dist = bestub - x;
            if d.simple_ub_dist <= feastol {
                d.simple_ub_dist = 0.0;
            }
            let mut vb = VarBound::default();
            d.vub_col = unsafe { (host.best_vub)(host.ctx, col, &mut bestub, &mut vb) };
            self.vub[c] = vb;

            let mut bestlb = self.col_lower.at(c);
            let d = &mut self.cols[c];
            d.simple_lb_dist = x - bestlb;
            if d.simple_lb_dist <= feastol {
                d.simple_lb_dist = 0.0;
            }
            d.vlb_col = unsafe { (host.best_vlb)(host.ctx, col, &mut bestlb, &mut vb) };
            self.vlb[c] = vb;

            let d = &mut self.cols[c];
            d.lb_dist = x - bestlb;
            if d.lb_dist <= feastol {
                d.lb_dist = 0.0;
            }
            d.ub_dist = bestub - x;
            if d.ub_dist <= feastol {
                d.ub_dist = 0.0;
            }
        }

        let integral_cols = self.integral_cols;
        for &col in integral_cols.get() {
            let c = col as usize;
            let mut bestub = self.col_upper.at(c);
            let mut bestlb = self.col_lower.at(c);
            if !self.parallel_lock_active_flag() {
                unsafe { (host.cleanup_varbounds)(host.ctx, col) };
            }
            if self.dom_infeasible() {
                return;
            }
            let x = col_value[c];
            let d = &mut self.cols[c];
            d.simple_ub_dist = bestub - x;
            if d.simple_ub_dist <= feastol {
                d.simple_ub_dist = 0.0;
            }
            d.simple_lb_dist = x - bestlb;
            if d.simple_lb_dist <= feastol {
                d.simple_lb_dist = 0.0;
            }
            let simple_bnd_dist = d.simple_lb_dist.min_cpp(d.simple_ub_dist);
            if simple_bnd_dist > 0.0 && (super::integers::nearest_integer(x) as f64 - x).abs() < feastol {
                let mut vb = VarBound::default();
                let vub_col = unsafe { (host.best_vub)(host.ctx, col, &mut bestub, &mut vb) };
                self.vub[c] = vb;
                let vlb_col = unsafe { (host.best_vlb)(host.ctx, col, &mut bestlb, &mut vb) };
                self.vlb[c] = vb;
                let d = &mut self.cols[c];
                d.vub_col = vub_col;
                d.vlb_col = vlb_col;
                d.lb_dist = x - bestlb;
                if d.lb_dist <= feastol {
                    d.lb_dist = 0.0;
                }
                d.ub_dist = bestub - x;
                if d.ub_dist <= feastol {
                    d.ub_dist = 0.0;
                }
                if d.bound_dist() > simple_bnd_dist + feastol {
                    d.lb_dist = d.simple_lb_dist;
                    d.ub_dist = d.simple_ub_dist;
                    d.vub_col = -1;
                    d.vlb_col = -1;
                }
            } else {
                d.lb_dist = d.simple_lb_dist;
                d.ub_dist = d.simple_ub_dist;
            }
        }

        // slack variables
        let num_col = self.num_col;
        let row_value = self.row_value;
        for row in 0..self.num_row {
            let (bestlb, bestub) = self.slack_bounds(row);
            if bestlb == bestub {
                continue;
            }
            let y = row_value.get()[row];
            let d = &mut self.cols[num_col + row];
            d.lb_dist = y - bestlb;
            if d.lb_dist <= feastol {
                d.lb_dist = 0.0;
            }
            d.simple_lb_dist = d.lb_dist;
            d.ub_dist = bestub - y;
            if d.ub_dist <= feastol {
                d.ub_dist = 0.0;
            }
            d.simple_ub_dist = d.ub_dist;
        }

        let is_frac = |v: f64| (v - v.round()).abs() > feastol;
        for col in 0..num_col {
            let d = self.cols[col];
            let x = col_value[col];
            let mut frac = d.integral && is_frac(x);
            if d.vub_col != -1 {
                let vb = self.vub[col];
                let y = col_value[d.vub_col as usize];
                frac = frac || is_frac(y) || vb.coef.mul_add_c(y, vb.constant) < x - feastol;
            }
            if d.vlb_col != -1 {
                let vb = self.vlb[col];
                let y = col_value[d.vlb_col as usize];
                frac = frac || is_frac(y) || vb.coef.mul_add_c(y, vb.constant) > x + feastol;
            }
            self.cols[col].fractional = frac;
        }
        for row in 0..self.num_row {
            let d = &mut self.cols[num_col + row];
            d.fractional = d.integral && is_frac(row_value.get()[row]);
        }
    }

    fn parallel_lock_active_flag(&self) -> bool {
        self.parallel_lock_active
    }
}

/// std::min(a, b) on doubles: b if b < a, else a
pub trait MinCpp {
    fn min_cpp(self, b: f64) -> f64;
    fn max_cpp(self, b: f64) -> f64;
}

impl MinCpp for f64 {
    #[inline]
    fn min_cpp(self, b: f64) -> f64 {
        if b < self {
            b
        } else {
            self
        }
    }
    /// std::max(a, b): b if a < b, else a
    #[inline]
    fn max_cpp(self, b: f64) -> f64 {
        if self < b {
            b
        } else {
            self
        }
    }
}
