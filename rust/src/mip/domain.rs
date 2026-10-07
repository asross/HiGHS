//! HighsDomain (highs/mip/HighsDomain.cpp): bound changes with their stack
//! (changeBound, doChangeBound, backtrack, backtrackToGlobal), the row
//! activities with their infinity counts and capacity thresholds, bound
//! propagation of model rows, cuts and conflicts, objective propagation
//! (objprop.rs) and the whole propagate() loop.
//!
//! # The view
//!
//! HighsDomain keeps owning its data (external code reads col_lower_,
//! col_upper_, the domain change stack, ... everywhere).
//! `highs_rs::DomainAccess` (highs/mip/HighsDomainRust.h) fills a
//! `#[repr(C)]` [`CDomain`] with pointer+length pairs of the model matrices
//! (column-wise from the model, row-wise from mipdata), of the domain's
//! fixed-size vectors, of the cut pools' dynamic row matrices and
//! propagation arrays ([`CCutProp`]), of the conflict pools' watched
//! literals ([`CConfProp`]) and of the objective propagation
//! ([`objprop::CObjProp`]). The vectors that grow during propagation (the
//! domain change stack, its reasons and previous bounds, the branching
//! positions, the changed columns, the lists of rows/cuts/conflicts to
//! propagate) are passed as pointers to the std::vector objects
//! ([`StdVec`], the begin/end/capacity layout of libc++ and libstdc++,
//! checked on the C++ side): Rust reads their live size and appends in
//! place, calling a C++ `reserve` when full. Activities are HighsCDouble
//! arrays, laid out as [`CDouble`] {hi, lo}.
//!
//! HighsDomain caches the view (rsView_) and drops it wherever a vector
//! with a cached length may move or resize: copy, assignment,
//! computeRowActivities, setupObjectivePropagation, adding or clearing
//! pools, cutAdded and conflictAdded. Debug builds compare the cache with a
//! fresh fill on every use. The const methods (computeMin/MaxActivity,
//! propagateRowUpper/Lower), which threads may call concurrently on the
//! global domain, get a small [`CBounds`] of shared data instead.
//!
//! # Calls back into C++
//!
//! A bound change that fixes a binary column applies the clique table's and
//! the implications' fixings (HighsCliqueTable::addImplications,
//! HighsImplications::applyImplications), which stay C++ and call
//! HighsDomain::changeBound, i.e. this code, recursively. That is the only
//! reentrant call, and it is made only by [`Ctx::change_bound`]. [`Ctx`]
//! holds just the raw pointer to the view; the slices of [`Dom`] are
//! borrowed from the `Ctx` (`Ctx::dom(&mut self)`), so the borrow checker
//! guarantees that no slice is alive across a `change_bound`. The other
//! calls into C++ are leaves: the cut and conflict pools' resetAge and the
//! insertion of a redundant row (HighsHashTable).
//!
//! Every floating-point operation is the C++ one in the same order
//! (HighsCDouble arithmetic via util::cdouble); the only contraction clang
//! makes here is `bound +- 1000.0 * feastol` in adjustedUb/adjustedLb.

use super::objprop::CObjProp;
use crate::ffi::{sl, sl_mut};
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use std::ffi::c_void;

pub(crate) const INF: f64 = f64::INFINITY;
const K_HIGHS_TINY: f64 = 1e-14;

// HighsDomain::Reason types
pub const REASON_BRANCHING: i32 = -1;
pub const REASON_UNKNOWN: i32 = -2;
pub const REASON_MODEL_ROW_UPPER: i32 = -3;
pub const REASON_MODEL_ROW_LOWER: i32 = -4;
pub const REASON_CONFLICTING_BOUNDS: i32 = -6;
pub const REASON_OBJECTIVE: i32 = -7;

// HighsBoundType
pub const LOWER: i32 = 0;
pub const UPPER: i32 = 1;

/// HighsDomainChange
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DomChg {
    pub boundval: f64,
    pub column: i32,
    pub boundtype: i32,
}

/// HighsDomain::Reason
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Reason {
    pub kind: i32,
    pub index: i32,
}

impl Reason {
    pub const UNSPECIFIED: Reason = Reason { kind: REASON_UNKNOWN, index: 0 };
}

/// std::pair<double, HighsInt> of prevboundval_
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PrevBound {
    pub val: f64,
    pub pos: i32,
}

/// ConflictPoolPropagation::WatchedLiteral
#[repr(C)]
#[derive(Clone, Copy)]
pub struct WatchedLiteral {
    pub domchg: DomChg,
    pub prev: i32,
    pub next: i32,
}

/// A pointer and length, as std::vector's data() and size(); the pointer
/// is never null (C++ passes a dangling aligned one for an empty vector).
/// Indexing is bounds-checked
#[repr(C)]
pub struct CSlice<T> {
    p: *mut T,
    n: i32,
}

impl<T> Clone for CSlice<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for CSlice<T> {}

impl<T> CSlice<T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n as usize
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// # Safety
    /// `p` valid for `n` reads during 'a
    #[inline(always)]
    pub(crate) unsafe fn get<'a>(&self) -> &'a [T] {
        debug_assert!(!self.p.is_null() && self.n >= 0);
        std::slice::from_raw_parts(self.p, self.n as usize)
    }

    /// # Safety
    /// `p` valid for `n` reads and writes during 'a, unaliased
    #[inline(always)]
    pub(crate) unsafe fn get_mut<'a>(&self) -> &'a mut [T] {
        debug_assert!(!self.p.is_null() && self.n >= 0);
        std::slice::from_raw_parts_mut(self.p, self.n as usize)
    }
}

// The view's slices are valid while a view of the domain exists (CDomain is
// only reachable through Ctx::dom and the ffi entry points)
impl<T, I: std::slice::SliceIndex<[T]>> std::ops::Index<I> for CSlice<T> {
    type Output = I::Output;
    #[inline(always)]
    fn index(&self, i: I) -> &I::Output {
        // SAFETY: see above
        unsafe { &self.get()[i] }
    }
}

impl<T, I: std::slice::SliceIndex<[T]>> std::ops::IndexMut<I> for CSlice<T> {
    #[inline(always)]
    fn index_mut(&mut self, i: I) -> &mut I::Output {
        // SAFETY: see above; distinct CSlices point to distinct vectors
        unsafe { &mut self.get_mut()[i] }
    }
}

/// A pointer to a C++ object of the domain (a std::vector or a scalar)
#[repr(transparent)]
pub struct Ptr<T>(*mut T);

impl<T> Clone for Ptr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Ptr<T> {}

impl<T> std::ops::Deref for Ptr<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &T {
        // SAFETY: as for CSlice
        unsafe { &*self.0 }
    }
}

impl<T> std::ops::DerefMut for Ptr<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut T {
        // SAFETY: as for CSlice; distinct Ptrs point to distinct objects
        unsafe { &mut *self.0 }
    }
}

/// Makes a std::vector's capacity at least the given number of elements
/// (and at least twice the old one)
pub type ReserveFn = unsafe extern "C" fn(*mut c_void, usize);

/// A std::vector of a trivially copyable type with the default allocator:
/// begin, end and capacity end pointers (libc++, libstdc++ and MSVC). New
/// elements are written in place; the C++ `reserve` grows the buffer
#[repr(C)]
pub struct StdVec<T> {
    begin: *mut T,
    end: *mut T,
    cap: *mut T,
}

impl<T: Copy> StdVec<T> {
    #[inline(always)]
    pub fn len(&self) -> usize {
        if self.begin.is_null() {
            0
        } else {
            // SAFETY: both point into (or one past) the same buffer
            unsafe { self.end.offset_from(self.begin) as usize }
        }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.end == self.begin
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: the vector's elements
        unsafe { sl(self.begin, self.len() as i32) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: the vector's elements
        unsafe { sl_mut(self.begin, self.len() as i32) }
    }

    /// Capacity for `n` elements
    #[inline]
    pub fn reserve(&mut self, n: usize, reserve: ReserveFn) {
        let cap = if self.begin.is_null() {
            0
        } else {
            // SAFETY: as in len
            unsafe { self.cap.offset_from(self.begin) as usize }
        };
        if n > cap {
            // SAFETY: C++ reserves the std::vector this is
            unsafe { reserve(self as *mut Self as *mut c_void, n) };
        }
    }

    #[inline(always)]
    pub fn push(&mut self, v: T, reserve: ReserveFn) {
        if self.end == self.cap {
            let n = self.len() + 1;
            // SAFETY: as in reserve
            unsafe { reserve(self as *mut Self as *mut c_void, n) };
        }
        // SAFETY: end < cap, within the allocation
        unsafe {
            self.end.write(v);
            self.end = self.end.add(1);
        }
    }

    #[inline(always)]
    pub fn pop(&mut self) {
        if !self.is_empty() {
            // SAFETY: one element less
            self.end = unsafe { self.end.sub(1) };
        }
    }

    #[inline(always)]
    pub fn truncate(&mut self, n: usize) {
        if n < self.len() {
            // SAFETY: within the elements
            self.end = unsafe { self.begin.add(n) };
        }
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.end = self.begin;
    }

    /// Sets the length to `n` (the vector must have that capacity), the new
    /// elements set to `fill`
    #[inline]
    pub fn set_len_filled(&mut self, n: usize, fill: T) {
        let len = self.len();
        // SAFETY: capacity reserved by the caller
        unsafe {
            for i in len..n {
                self.begin.add(i).write(fill);
            }
            self.end = self.begin.add(n);
        }
    }

    /// std::vector::swap
    #[inline(always)]
    pub fn swap(&mut self, other: &mut StdVec<T>) {
        std::mem::swap(self, other);
    }
}

impl<T: Copy> std::ops::Index<usize> for StdVec<T> {
    type Output = T;
    #[inline(always)]
    fn index(&self, i: usize) -> &T {
        &self.as_slice()[i]
    }
}

/// A CutpoolPropagation and its cut pool's matrix, mirrored by
/// highs_rs::CutProp in highs/mip/HighsDomainRustView.h
#[repr(C)]
pub struct CCutProp {
    cutpoolindex: i32,
    cutpool: *const c_void,
    activitycuts: CSlice<CDouble>,
    activitycutsinf: CSlice<i32>,
    propagatecutflags: CSlice<u8>,
    capacity_threshold: CSlice<f64>,
    propagatecutinds: Ptr<StdVec<i32>>,
    // HighsDynamicRowMatrix
    ar_range: CSlice<[i32; 2]>,
    ar_index: CSlice<i32>,
    ar_value: CSlice<f64>,
    ar_rowindex: CSlice<i32>,
    next_pos: CSlice<i32>,
    next_neg: CSlice<i32>,
    head_pos: CSlice<i32>,
    head_neg: CSlice<i32>,
    rhs: CSlice<f64>,
}

/// A ConflictPoolPropagation and its pool's conflicts, mirrored by
/// highs_rs::ConfProp
#[repr(C)]
pub struct CConfProp {
    col_lower_watched: CSlice<i32>,
    col_upper_watched: CSlice<i32>,
    watched: CSlice<WatchedLiteral>,
    conflict_flag: CSlice<u8>,
    propagate_conflict_inds: Ptr<StdVec<i32>>,
    // HighsConflictPool (read live: syncConflictPool clears them)
    entries: Ptr<StdVec<DomChg>>,
    ranges: Ptr<StdVec<[i32; 2]>>,
}

/// HighsDomain's data, filled by highs_rs::DomainAccess, mirrored by
/// highs_rs::Domain in highs/mip/HighsDomainRustView.h
#[repr(C)]
pub struct CDomain {
    feastol: *const f64,
    epsilon: *const f64,
    upper_limit: *const f64,
    // model, column-wise
    a_start: CSlice<i32>,
    a_index: CSlice<i32>,
    a_value: CSlice<f64>,
    // mipdata, row-wise
    ar_start: CSlice<i32>,
    ar_index: CSlice<i32>,
    ar_value: CSlice<f64>,
    row_lower: CSlice<f64>,
    row_upper: CSlice<f64>,
    pub(crate) integrality: CSlice<u8>,
    // the domain
    pub(crate) col_lower: CSlice<f64>,
    pub(crate) col_upper: CSlice<f64>,
    activitymin: CSlice<CDouble>,
    activitymax: CSlice<CDouble>,
    activitymininf: CSlice<i32>,
    activitymaxinf: CSlice<i32>,
    capacity_threshold: CSlice<f64>,
    propagateflags: CSlice<u8>,
    col_lower_pos: CSlice<i32>,
    col_upper_pos: CSlice<i32>,
    changedcolsflags: CSlice<u8>,
    propagateinds: Ptr<StdVec<i32>>,
    changedcols: Ptr<StdVec<i32>>,
    branchpos: Ptr<StdVec<i32>>,
    domchgstack: Ptr<StdVec<DomChg>>,
    domchgreason: Ptr<StdVec<Reason>>,
    prevboundval: Ptr<StdVec<PrevBound>>,
    // scratch of propagate()
    scratch_inds: Ptr<StdVec<i32>>,
    scratch_bounds: Ptr<StdVec<DomChg>>,
    scratch_counts: Ptr<StdVec<[i32; 2]>>,
    pub(crate) infeasible: Ptr<bool>,
    infeasible_reason: Ptr<Reason>,
    infeasible_pos: Ptr<i32>,
    record_redundant_rows: Ptr<bool>,
    cutpools: CSlice<CCutProp>,
    conflictpools: CSlice<CConfProp>,
    pub(crate) objprop: CObjProp,
    // calls into C++, with `dom` (the HighsDomain)
    dom: *mut c_void,
    implications: unsafe extern "C" fn(*mut c_void, i32, i32),
    redundant_row: unsafe extern "C" fn(*mut c_void, i32),
    cut_reset_age: unsafe extern "C" fn(*mut c_void, i32, i32),
    conflict_reset_age: unsafe extern "C" fn(*mut c_void, i32, i32),
    reserve_i32: ReserveFn,
    reserve_domchg: ReserveFn,
    reserve_reason: ReserveFn,
    reserve_prev: ReserveFn,
    reserve_pair: ReserveFn,
}

/// The data of the const methods (computeMin/MaxActivity,
/// propagateRowUpper/Lower), mirrored by highs_rs::Bounds
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CBounds {
    feastol: f64,
    epsilon: f64,
    col_lower: CSlice<f64>,
    col_upper: CSlice<f64>,
    integrality: CSlice<u8>,
    col_lower_pos: CSlice<i32>,
    col_upper_pos: CSlice<i32>,
    prevboundval: CSlice<PrevBound>,
    infeasible: bool,
    infeasible_pos: i32,
}

/// The column bounds and what the bound-only kernels read
#[derive(Clone, Copy)]
pub struct Bounds<'a> {
    pub feastol: f64,
    pub epsilon: f64,
    pub col_lower: &'a [f64],
    pub col_upper: &'a [f64],
    integrality: &'a [u8],
    col_lower_pos: &'a [i32],
    col_upper_pos: &'a [i32],
    pub(crate) prevboundval: &'a [PrevBound],
    infeasible: bool,
    infeasible_pos: i32,
}

impl CBounds {
    /// # Safety
    /// Every pointer valid for its length
    #[inline(always)]
    pub unsafe fn view<'a>(&self) -> Bounds<'a> {
        Bounds {
            feastol: self.feastol,
            epsilon: self.epsilon,
            col_lower: self.col_lower.get(),
            col_upper: self.col_upper.get(),
            integrality: self.integrality.get(),
            col_lower_pos: self.col_lower_pos.get(),
            col_upper_pos: self.col_upper_pos.get(),
            prevboundval: self.prevboundval.get(),
            infeasible: self.infeasible,
            infeasible_pos: self.infeasible_pos,
        }
    }
}

/// A view of the domain: the C++ data behind CDomain (to which it derefs)
/// and the tolerances
pub struct Dom<'a> {
    c: &'a mut CDomain,
    pub feastol: f64,
    pub epsilon: f64,
    pub upper_limit: f64,
}

impl std::ops::Deref for Dom<'_> {
    type Target = CDomain;
    #[inline(always)]
    fn deref(&self) -> &CDomain {
        self.c
    }
}

impl std::ops::DerefMut for Dom<'_> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut CDomain {
        self.c
    }
}

impl CDomain {
    /// # Safety
    /// Every pointer must be valid for its length and unaliased for the
    /// view's lifetime
    #[inline(always)]
    pub unsafe fn view<'a>(c: *mut CDomain) -> Dom<'a> {
        let c = &mut *c;
        Dom { feastol: *c.feastol, epsilon: *c.epsilon, upper_limit: *c.upper_limit, c }
    }

    /// The cut pool `pool`, as a reference independent of the view's borrow
    #[inline(always)]
    fn cutpool<'b>(&self, pool: usize) -> &'b mut CCutProp {
        assert!(pool < self.cutpools.len());
        // SAFETY: the pools' data is distinct from CDomain itself, and the
        // callers hold one pool at a time
        unsafe { &mut *self.cutpools.p.add(pool) }
    }

    /// The conflict pool `pool`, as cutpool
    #[inline(always)]
    fn conflictpool<'b>(&self, pool: usize) -> &'b mut CConfProp {
        assert!(pool < self.conflictpools.len());
        // SAFETY: as in cutpool
        unsafe { &mut *self.conflictpools.p.add(pool) }
    }
}

/// std::max(a, b)
#[inline(always)]
pub(crate) fn max2(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// std::max({a, b, c})
#[inline(always)]
fn max3(a: f64, b: f64, c: f64) -> f64 {
    max2(max2(a, b), c)
}

#[inline(always)]
fn activity_contribution_min(coef: f64, lb: f64, ub: f64) -> f64 {
    if coef < 0.0 {
        if ub == INF {
            return -INF;
        }
        coef * ub
    } else {
        if lb == -INF {
            return -INF;
        }
        coef * lb
    }
}

#[inline(always)]
fn activity_contribution_max(coef: f64, lb: f64, ub: f64) -> f64 {
    if coef < 0.0 {
        if lb == -INF {
            return INF;
        }
        coef * lb
    } else {
        if ub == INF {
            return INF;
        }
        coef * ub
    }
}

#[inline(always)]
fn compute_delta(val: f64, oldbound: f64, newbound: f64, inf: f64, numinfs: &mut i32) -> CDouble {
    if oldbound == inf {
        *numinfs -= 1;
        CDouble::from(newbound) * val
    } else if newbound == inf {
        *numinfs += 1;
        CDouble::from(-oldbound) * val
    } else {
        (CDouble::from(newbound) - CDouble::from(oldbound)) * val
    }
}

#[inline(always)]
pub(crate) fn bound_range(upper: f64, lower: f64, tolerance: f64, continuous: bool) -> f64 {
    let range = upper - lower;
    range
        - if continuous {
            max2(0.3 * range, 1000.0 * tolerance)
        } else {
            tolerance
        }
}

impl<'a> Bounds<'a> {
    #[inline(always)]
    fn is_continuous(&self, col: usize) -> bool {
        self.integrality[col] == 0
    }

    /// getColLowerPos / getColUpperPos
    fn col_bound_pos(&self, col: usize, stackpos: i32, upper: bool) -> f64 {
        self.col_bound_at(col, stackpos, upper).0
    }

    /// getColLowerPos / getColUpperPos: the bound at stack position
    /// stackpos and the position of its change (-1 if global)
    #[inline]
    pub(crate) fn col_bound_at(&self, col: usize, stackpos: i32, upper: bool) -> (f64, i32) {
        let (mut b, mut pos) = if upper {
            (self.col_upper[col], self.col_upper_pos[col])
        } else {
            (self.col_lower[col], self.col_lower_pos[col])
        };
        while pos > stackpos || (pos != -1 && self.prevboundval[pos as usize].val == b) {
            let p = self.prevboundval[pos as usize];
            b = p.val;
            pos = p.pos;
        }
        (b, pos)
    }

    /// computeMinActivity (max = false) / computeMaxActivity (max = true)
    pub fn compute_activity(&self, index: &[i32], value: &[f64], max: bool) -> (i32, CDouble) {
        let mut act = CDouble::from(0.0);
        let mut ninf = 0;
        let inf = if max { INF } else { -INF };
        if self.infeasible {
            let stackpos = self.infeasible_pos - 1;
            for (&col, &val) in index.iter().zip(value) {
                let col = col as usize;
                let lb = self.col_bound_pos(col, stackpos, false);
                let ub = self.col_bound_pos(col, stackpos, true);
                let c = if max {
                    activity_contribution_max(val, lb, ub)
                } else {
                    activity_contribution_min(val, lb, ub)
                };
                if c == inf {
                    ninf += 1;
                } else {
                    act += c;
                }
            }
        } else {
            for (&col, &val) in index.iter().zip(value) {
                let col = col as usize;
                let (lb, ub) = (self.col_lower[col], self.col_upper[col]);
                let c = if max {
                    activity_contribution_max(val, lb, ub)
                } else {
                    activity_contribution_min(val, lb, ub)
                };
                if c == inf {
                    ninf += 1;
                } else {
                    act += c;
                }
            }
        }
        act.renormalize();
        (ninf, act)
    }

    /// The capacity threshold of a row (recomputeCapacityThreshold)
    fn capacity_threshold_of(&self, index: &[i32], value: &[f64]) -> f64 {
        let feastol = self.feastol;
        let mut cap = -feastol;
        for (&col, &val) in index.iter().zip(value) {
            let col = col as usize;
            let (lb, ub) = (self.col_lower[col], self.col_upper[col]);
            if ub == lb {
                continue;
            }
            let threshold = val.abs() * bound_range(ub, lb, feastol, self.is_continuous(col));
            cap = max3(cap, threshold, feastol);
        }
        cap
    }

    /// adjustedUb
    fn adjusted_ub(&self, col: usize, bound_val: CDouble) -> (f64, bool) {
        let feastol = self.feastol;
        let (lb, ub) = (self.col_lower[col], self.col_upper[col]);
        if !self.is_continuous(col) {
            let bound = (bound_val + feastol).floor().to_f64();
            let accept = bound < ub && ub - bound > 1000.0 * feastol * bound.abs();
            (bound, accept)
        } else {
            let bv = bound_val.to_f64();
            let bound = if (bv - lb).abs() <= self.epsilon { lb } else { bv };
            let accept = if ub == INF {
                true
            } else if 1000.0f64.mul_add_c(feastol, bound) < ub {
                let mut relative_improve = ub - bound;
                if lb != -INF {
                    relative_improve /= ub - lb;
                } else {
                    relative_improve /= max2(ub.abs(), bound.abs());
                }
                relative_improve >= 0.3
            } else {
                false
            };
            (bound, accept)
        }
    }

    /// adjustedLb
    fn adjusted_lb(&self, col: usize, bound_val: CDouble) -> (f64, bool) {
        let feastol = self.feastol;
        let (lb, ub) = (self.col_lower[col], self.col_upper[col]);
        if !self.is_continuous(col) {
            let bound = (bound_val - feastol).ceil().to_f64();
            let accept = bound > lb && bound - lb > 1000.0 * feastol * bound.abs();
            (bound, accept)
        } else {
            let bv = bound_val.to_f64();
            let bound = if (ub - bv).abs() <= self.epsilon { ub } else { bv };
            let accept = if lb == -INF {
                true
            } else if (-1000.0f64).mul_add_c(feastol, bound) > lb {
                let mut relative_improve = bound - lb;
                if ub != INF {
                    relative_improve /= ub - lb;
                } else {
                    relative_improve /= max2(lb.abs(), bound.abs());
                }
                relative_improve >= 0.3
            } else {
                false
            };
            (bound, accept)
        }
    }

    /// tightenCoefficients of the row sum vals[i] x[inds[i]] <= rhs
    pub fn tighten_coefficients(&self, inds: &[i32], vals: &mut [f64], rhs: &mut f64) {
        let mut maxactivity = CDouble::from(0.0);
        for (&c, &v) in inds.iter().zip(vals.iter()) {
            let c = c as usize;
            if v > 0.0 {
                if self.col_upper[c] == INF {
                    return;
                }
                maxactivity += self.col_upper[c] * v;
            } else {
                if self.col_lower[c] == -INF {
                    return;
                }
                maxactivity += self.col_lower[c] * v;
            }
        }

        let maxabscoef = maxactivity - *rhs;
        if maxabscoef.to_f64() > self.feastol {
            let mut upper = CDouble::from(*rhs);
            let mut tightened = 0;
            let m = maxabscoef.to_f64();
            let negm = (-maxabscoef).to_f64();
            for (&c, v) in inds.iter().zip(vals.iter_mut()) {
                let c = c as usize;
                if self.is_continuous(c) {
                    continue;
                }
                if *v > m {
                    let delta = *v - maxabscoef;
                    upper -= delta * self.col_upper[c];
                    *v = m;
                    tightened += 1;
                } else if *v < negm {
                    let delta = -*v - maxabscoef;
                    upper += delta * self.col_lower[c];
                    *v = -m;
                    tightened += 1;
                }
            }
            if tightened != 0 {
                *rhs = upper.to_f64();
            }
        }
    }

    /// propagateRowUpper (lower = false) / propagateRowLower (lower = true):
    /// the bound changes implied by the row side `rhs` and the min (max)
    /// activity, written to `out`; returns their number
    #[allow(clippy::too_many_arguments)]
    pub fn propagate_row(
        &self,
        index: &[i32],
        value: &[f64],
        rhs: f64,
        activity: CDouble,
        ninf: i32,
        lower: bool,
        out: &mut [DomChg],
    ) -> usize {
        let mut cap = 0.0;
        self.propagate_row_impl::<false>(index, value, rhs, activity, ninf, lower, out, &mut cap)
    }

    /// propagate_row and the row's capacity threshold
    /// (recomputeCapacityThreshold, at the bounds before the changes), in
    /// one pass
    #[allow(clippy::too_many_arguments)]
    pub fn propagate_row_cap(
        &self,
        index: &[i32],
        value: &[f64],
        rhs: f64,
        activity: CDouble,
        ninf: i32,
        lower: bool,
        out: &mut [DomChg],
    ) -> (usize, f64) {
        let mut cap = -self.feastol;
        let n = self.propagate_row_impl::<true>(index, value, rhs, activity, ninf, lower, out, &mut cap);
        (n, cap)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn propagate_row_impl<const CAP: bool>(
        &self,
        index: &[i32],
        value: &[f64],
        rhs: f64,
        activity: CDouble,
        ninf: i32,
        lower: bool,
        out: &mut [DomChg],
        cap: &mut f64,
    ) -> usize {
        if ninf > 1 {
            if CAP {
                *cap = self.capacity_threshold_of(index, value);
            }
            return 0;
        }
        let feastol = self.feastol;
        let mut numchgs = 0;
        let act = activity.to_f64();
        let inf = if lower { INF } else { -INF };
        for (&c, &val) in index.iter().zip(value) {
            let col = c as usize;
            let (lb, ub) = (self.col_lower[col], self.col_upper[col]);
            if CAP && ub != lb {
                let threshold = val.abs() * bound_range(ub, lb, feastol, self.is_continuous(col));
                *cap = max3(*cap, threshold, feastol);
            }
            let actcontribution = if lower {
                activity_contribution_max(val, lb, ub)
            } else {
                activity_contribution_min(val, lb, ub)
            };
            let resact;
            if ninf == 1 {
                if actcontribution != inf {
                    continue;
                }
                resact = activity;
            } else {
                // A bound can only be accepted if the implied value
                // (rhs - (act - actcontribution)) / val is strictly inside the
                // current bound: skip the exact (double-double) computation
                // where a double estimate, with a generous margin for its
                // rounding errors, shows that it is not. (The C++'s
                // impliedBoundNoTighter, multiplied by val: no division)
                let n = rhs - (act - actcontribution);
                let margin = 1e-12 * ((rhs.abs() + act.abs() + actcontribution.abs()) + n.abs());
                let hopeless = if (val > 0.0) != lower {
                    // an upper bound
                    if val > 0.0 {
                        n - margin >= ub * val
                    } else {
                        n + margin <= ub * val
                    }
                } else if val > 0.0 {
                    n + margin <= lb * val
                } else {
                    n - margin >= lb * val
                };
                if hopeless {
                    continue;
                }
                resact = activity - actcontribution;
            }

            let bound_val = (rhs - resact) / val;
            if (bound_val.to_f64() * K_HIGHS_TINY).abs() > feastol {
                continue;
            }

            if (val > 0.0) != lower {
                let (bound, accept) = self.adjusted_ub(col, bound_val);
                if accept {
                    out[numchgs] = DomChg { boundval: bound, column: c, boundtype: UPPER };
                    numchgs += 1;
                }
            } else {
                let (bound, accept) = self.adjusted_lb(col, bound_val);
                if accept {
                    out[numchgs] = DomChg { boundval: bound, column: c, boundtype: LOWER };
                    numchgs += 1;
                }
            }
        }
        numchgs
    }
}

impl<'a> Dom<'a> {
    /// The bound-only part of the view
    #[inline(always)]
    pub fn bounds(&self) -> Bounds<'_> {
        Bounds {
            feastol: self.feastol,
            epsilon: self.epsilon,
            // SAFETY: valid for the view's lifetime, not written while the
            // Bounds (borrowing self) lives
            col_lower: unsafe { self.col_lower.get() },
            col_upper: unsafe { self.col_upper.get() },
            integrality: unsafe { self.integrality.get() },
            col_lower_pos: unsafe { self.col_lower_pos.get() },
            col_upper_pos: unsafe { self.col_upper_pos.get() },
            prevboundval: self.prevboundval.as_slice(),
            infeasible: *self.infeasible,
            infeasible_pos: *self.infeasible_pos,
        }
    }

    #[inline(always)]
    pub(crate) fn is_continuous(&self, col: usize) -> bool {
        self.integrality[col] == 0
    }

    #[inline(always)]
    pub(crate) fn is_fixed(&self, col: usize) -> bool {
        self.col_lower[col] == self.col_upper[col]
    }

    #[inline(always)]
    fn push_i32(&self, mut v: Ptr<StdVec<i32>>, x: i32) {
        v.push(x, self.reserve_i32);
    }

    /// The entries of a model row, as slices independent of the view's
    /// borrow (the model is never written)
    #[inline(always)]
    pub(crate) fn row<'x>(&self, row: usize) -> (&'x [i32], &'x [f64]) {
        let (s, e) = (self.ar_start[row] as usize, self.ar_start[row + 1] as usize);
        // SAFETY: read-only model data, valid for the view's lifetime
        unsafe { (&self.ar_index.get()[s..e], &self.ar_value.get()[s..e]) }
    }

    // The accessors of conflict analysis (conflict.rs)

    pub(crate) fn stack<'x>(&self) -> &'x [DomChg] {
        // SAFETY: the stack is not changed during conflict analysis
        unsafe { &*(self.domchgstack.as_slice() as *const [DomChg]) }
    }

    pub(crate) fn reasons<'x>(&self) -> &'x [Reason] {
        // SAFETY: as in stack
        unsafe { &*(self.domchgreason.as_slice() as *const [Reason]) }
    }

    pub(crate) fn prev_bounds<'x>(&self) -> &'x [PrevBound] {
        // SAFETY: as in stack
        unsafe { &*(self.prevboundval.as_slice() as *const [PrevBound]) }
    }

    pub(crate) fn branch_positions<'x>(&self) -> &'x [i32] {
        // SAFETY: as in stack
        unsafe { &*(self.branchpos.as_slice() as *const [i32]) }
    }

    /// infeasible_, infeasible_reason and infeasible_pos
    pub(crate) fn infeasibility(&self) -> (bool, Reason, i32) {
        (*self.infeasible, *self.infeasible_reason, *self.infeasible_pos)
    }

    // The accessors of presolve's probing and enumeration loops
    // (presolve/hpresolve/probing.rs, enumeration.rs)

    /// getChangedCols
    pub(crate) fn changed_cols(&self) -> &[i32] {
        self.changedcols.as_slice()
    }

    /// isChangedCol
    pub(crate) fn is_changed_col(&self, col: usize) -> bool {
        self.changedcolsflags[col] != 0
    }

    /// clearChangedCols(start)
    pub(crate) fn clear_changed_cols(&mut self, start: usize) {
        for i in start..self.changedcols.len() {
            let col = self.changedcols[i] as usize;
            self.changedcolsflags[col] = 0;
        }
        self.changedcols.truncate(start);
    }

    /// isBinary
    pub(crate) fn is_binary(&self, col: usize) -> bool {
        self.integrality[col] != 0 && self.col_lower[col] == 0.0 && self.col_upper[col] == 1.0
    }

    /// isRedundantRow
    pub(crate) fn is_redundant_row(&self, row: usize) -> bool {
        let (lower, upper) = self.row_bounds(row);
        self.min_activity(row) >= lower - self.feastol && self.max_activity(row) <= upper + self.feastol
    }

    pub(crate) fn row_bounds(&self, row: usize) -> (f64, f64) {
        (self.row_lower[row], self.row_upper[row])
    }

    /// getMinActivity
    pub(crate) fn min_activity(&self, row: usize) -> f64 {
        if self.activitymininf[row] == 0 {
            self.activitymin[row].to_f64()
        } else {
            -INF
        }
    }

    /// getMaxActivity
    pub(crate) fn max_activity(&self, row: usize) -> f64 {
        if self.activitymaxinf[row] == 0 {
            self.activitymax[row].to_f64()
        } else {
            INF
        }
    }

    pub(crate) fn num_cutpools(&self) -> usize {
        self.cutpools.len()
    }

    /// HighsCutPool::getCut and the cut's rhs
    pub(crate) fn cut<'x>(&self, pool: usize, cut: usize) -> (&'x [i32], &'x [f64], f64) {
        let cp = self.cutpool(pool);
        let [s, e] = cp.ar_range[cut];
        // SAFETY: the pool's matrix, not changed during conflict analysis
        unsafe { (&cp.ar_index.get()[s as usize..e as usize], &cp.ar_value.get()[s as usize..e as usize], cp.rhs[cut]) }
    }

    /// The HighsCutPool of a cut pool
    pub(crate) fn cutpool_id(&self, pool: usize) -> *const c_void {
        self.cutpool(pool).cutpool
    }

    /// getMinCutActivity
    pub(crate) fn min_cut_activity(&self, cutpool: *const c_void, cut: usize) -> f64 {
        for pool in 0..self.cutpools.len() {
            let cp = self.cutpool(pool);
            if cp.cutpool == cutpool {
                return if cut < cp.propagatecutflags.len()
                    && (cp.propagatecutflags[cut] & 2) == 0
                    && cp.activitycutsinf[cut] == 0
                {
                    cp.activitycuts[cut].to_f64()
                } else {
                    -INF
                };
            }
        }
        -INF
    }

    /// Whether a conflict of conflict pool `pool` has been deleted
    pub(crate) fn conflict_deleted(&self, pool: usize, conflict: usize) -> bool {
        self.conflictpool(pool).conflict_flag[conflict] & 8 != 0
    }

    /// The entries of a conflict
    pub(crate) fn conflict<'x>(&self, pool: usize, conflict: usize) -> &'x [DomChg] {
        let cp = self.conflictpool(pool);
        let [s, e] = cp.ranges[conflict];
        // SAFETY: the pool's entries, not changed while this is used
        unsafe { &*(&cp.entries.as_slice()[s as usize..e as usize] as *const [DomChg]) }
    }

    /// HighsDomain::isActive
    #[inline(always)]
    pub(crate) fn is_active(&self, d: &DomChg) -> bool {
        let c = d.column as usize;
        if d.boundtype == LOWER {
            d.boundval <= self.col_lower[c]
        } else {
            d.boundval >= self.col_upper[c]
        }
    }

    /// HighsDomain::flip
    #[inline]
    pub(crate) fn flip(&self, d: &DomChg) -> DomChg {
        let integral = !self.is_continuous(d.column as usize);
        if d.boundtype == LOWER {
            let v = d.boundval - self.feastol;
            DomChg { boundval: if integral { v.floor() } else { v }, column: d.column, boundtype: UPPER }
        } else {
            let v = d.boundval + self.feastol;
            DomChg { boundval: if integral { v.ceil() } else { v }, column: d.column, boundtype: LOWER }
        }
    }

    pub fn recompute_capacity_threshold(&mut self, row: usize) {
        let (index, value) = self.row(row);
        self.capacity_threshold[row] = self.bounds().capacity_threshold_of(index, value);
    }

    /// Whether updateThresholdLbChange (upper = false) / UbChange (upper =
    /// true) of col changes thresholds, and the bound range it multiplies
    /// by the coefficient
    #[inline(always)]
    fn threshold_range(&self, col: usize, newbound: f64, upper: bool) -> (bool, f64) {
        let cont = self.is_continuous(col);
        if upper {
            let lb = self.col_lower[col];
            (newbound != lb, bound_range(newbound, lb, self.feastol, cont))
        } else {
            let ub = self.col_upper[col];
            (newbound != ub, bound_range(ub, newbound, self.feastol, cont))
        }
    }

    /// markPropagate
    pub fn mark_propagate(&mut self, row: usize) {
        if self.propagateflags[row] != 0 {
            return;
        }
        let (rl, ru) = (self.row_lower[row], self.row_upper[row]);
        let feastol = self.feastol;
        let cap = self.capacity_threshold[row];
        let (amin, amax) = (self.activitymin[row], self.activitymax[row]);
        let (nmin, nmax) = (self.activitymininf[row], self.activitymaxinf[row]);
        let proplower = rl != -INF
            && (nmin != 0 || amin < rl - feastol)
            && (nmax == 1 || (amax.to_f64() - rl) <= cap);
        let propupper = ru != INF
            && (nmax != 0 || amax > ru + feastol)
            && (nmin == 1 || (ru - amin.to_f64()) <= cap);
        if proplower || propupper {
            self.push_i32(self.propagateinds, row as i32);
            self.propagateflags[row] = 1;
        }
    }

    /// computeRowActivities
    pub fn compute_row_activities(&mut self) {
        let nrow = self.row_lower.len();
        for i in 0..nrow {
            let (index, value) = self.row(i);
            let (n, a) = self.bounds().compute_activity(index, value, false);
            (self.activitymininf[i], self.activitymin[i]) = (n, a);
            let (n, a) = self.bounds().compute_activity(index, value, true);
            (self.activitymaxinf[i], self.activitymax[i]) = (n, a);
            self.recompute_capacity_threshold(i);
            if (self.activitymininf[i] <= 1 && self.row_upper[i] != INF)
                || (self.activitymaxinf[i] <= 1 && self.row_lower[i] != -INF)
            {
                self.mark_propagate(i);
            }
        }
    }

    pub(crate) fn set_infeasible(&mut self, kind: i32, index: i32) {
        *self.infeasible = true;
        *self.infeasible_pos = self.domchgstack.len() as i32;
        *self.infeasible_reason = Reason { kind, index };
    }

    /// updateRedundantRows
    fn update_redundant_rows(&mut self, row: usize) {
        // isRedundantRow
        let minact = if self.activitymininf[row] == 0 { self.activitymin[row].to_f64() } else { -INF };
        let maxact = if self.activitymaxinf[row] == 0 { self.activitymax[row].to_f64() } else { INF };
        if minact >= self.row_lower[row] - self.feastol && maxact <= self.row_upper[row] + self.feastol {
            // SAFETY: inserts into the domain's hash table, which no view
            // points to
            unsafe { (self.c.redundant_row)(self.c.dom, row as i32) };
        }
    }

    /// updateActivityLbChange (upper = false) / updateActivityUbChange
    /// (upper = true)
    #[inline]
    pub fn update_activity(&mut self, col: usize, oldbound: f64, newbound: f64, upper: bool) {
        if upper {
            self.update_activity_side::<true>(col, oldbound, newbound)
        } else {
            self.update_activity_side::<false>(col, oldbound, newbound)
        }
    }

    fn update_activity_side<const UPPER_SIDE: bool>(&mut self, col: usize, oldbound: f64, newbound: f64) {
        let upper = UPPER_SIDE;
        debug_assert!(!*self.infeasible);
        if self.objprop.active {
            self.obj_update_activity(col, oldbound, newbound, upper);
            if *self.infeasible {
                return;
            }
        }

        let feastol = self.feastol;
        let inf = if upper { INF } else { -INF };
        let start = self.a_start[col] as usize;
        let mut end = self.a_start[col + 1] as usize;
        let record = *self.record_redundant_rows;
        // local copies of the slices, so that they stay in registers
        let (mut amin, mut amax, mut nmin, mut nmax) =
            (self.activitymin, self.activitymax, self.activitymininf, self.activitymaxinf);
        let (mut cap, flags, row_lower, row_upper) =
            (self.capacity_threshold, self.propagateflags, self.row_lower, self.row_upper);
        // updateThresholdLbChange / UbChange: the same bound range for every
        // entry
        let (thr_applies, thr_range) = self.threshold_range(col, newbound, upper);
        // SAFETY: read-only model data, valid for the view's lifetime
        let (a_index, a_value) = unsafe { (&self.a_index.get()[start..end], &self.a_value.get()[start..end]) };
        for (k, (&r, &val)) in a_index.iter().zip(a_value).enumerate() {
            let row = r as usize;
            // a lower bound change moves the min activity of rows with a
            // positive coefficient, an upper bound change that of rows with a
            // negative one
            if (val > 0.0) != upper {
                let delta = compute_delta(val, oldbound, newbound, inf, &mut nmin[row]);
                amin[row] += delta;
                if record && row_lower[row] != -INF && row_upper[row] == INF {
                    self.update_redundant_rows(row);
                }
                if delta <= 0.0 {
                    if thr_applies {
                        cap[row] = max3(cap[row], val.abs() * thr_range, feastol);
                    }
                    continue;
                }
                let ru = row_upper[row];
                if ru != INF && nmin[row] == 0 && amin[row] - ru > feastol {
                    self.set_infeasible(REASON_MODEL_ROW_UPPER, row as i32);
                    end = start + k + 1;
                    break;
                }
                if nmin[row] <= 1 && flags[row] == 0 && ru != INF {
                    self.mark_propagate(row);
                }
            } else {
                let delta = compute_delta(val, oldbound, newbound, inf, &mut nmax[row]);
                amax[row] += delta;
                if record && row_lower[row] == -INF && row_upper[row] != INF {
                    self.update_redundant_rows(row);
                }
                if delta >= 0.0 {
                    if thr_applies {
                        cap[row] = max3(cap[row], val.abs() * thr_range, feastol);
                    }
                    continue;
                }
                let rl = row_lower[row];
                if rl != -INF && nmax[row] == 0 && rl - amax[row] > feastol {
                    self.set_infeasible(REASON_MODEL_ROW_LOWER, row as i32);
                    end = start + k + 1;
                    break;
                }
                if nmax[row] <= 1 && flags[row] == 0 && rl != -INF {
                    self.mark_propagate(row);
                }
            }
        }

        if !*self.infeasible {
            // If cut pool i finds infeasibility, the pools after it still
            // update their thresholds and the pools before it revert their
            // activity changes
            let mut infeascutpool = -1;
            for i in 0..self.cutpools.len() {
                if !*self.infeasible {
                    self.cut_update_activity::<UPPER_SIDE>(i, col, oldbound, newbound, true, true, false);
                    if *self.infeasible {
                        infeascutpool = i as i32;
                    }
                } else {
                    self.cut_update_activity::<UPPER_SIDE>(i, col, oldbound, newbound, true, false, true);
                }
            }
            for i in 0..infeascutpool.max(0) as usize {
                self.cut_update_activity::<UPPER_SIDE>(i, col, oldbound, newbound, false, true, true);
            }
        }

        if *self.infeasible {
            let (oldbound, newbound) = (newbound, oldbound);
            for (&r, &val) in a_index[..end - start].iter().zip(a_value) {
                let row = r as usize;
                if (val > 0.0) != upper {
                    let d = compute_delta(val, oldbound, newbound, inf, &mut nmin[row]);
                    amin[row] += d;
                } else {
                    let d = compute_delta(val, oldbound, newbound, inf, &mut nmax[row]);
                    amax[row] += d;
                }
            }
            if self.objprop.active {
                self.obj_update_activity(col, oldbound, newbound, upper);
            }
        } else {
            for p in 0..self.conflictpools.len() {
                self.conflict_update_activity(p, col, oldbound, newbound, upper);
            }
        }
    }

    /// CutpoolPropagation::markPropagateCut
    fn mark_propagate_cut_in(&self, cp: &mut CCutProp, cut: usize) {
        if cp.propagatecutflags[cut] == 0
            && (cp.activitycutsinf[cut] == 1
                || (cp.rhs[cut] - cp.activitycuts[cut].to_f64() <= cp.capacity_threshold[cut]))
        {
            self.push_i32(cp.propagatecutinds, cut as i32);
            cp.propagatecutflags[cut] |= 1;
        }
    }

    /// HighsDomain::markPropagateCut
    pub fn mark_propagate_cut(&mut self, reason: Reason) {
        if reason.kind < 0 {
            return;
        }
        let k = reason.kind as usize;
        let i = reason.index as usize;
        if k < self.cutpools.len() {
            let cp = self.cutpool(k);
            self.mark_propagate_cut_in(cp, i);
        } else {
            let cp = self.conflictpool(k - self.cutpools.len());
            self.mark_propagate_conflict(cp, i);
        }
    }

    /// ConflictPoolPropagation::markPropagateConflict
    #[inline]
    fn mark_propagate_conflict(&self, cp: &mut CConfProp, conflict: usize) {
        if cp.conflict_flag[conflict] < 2 {
            self.push_i32(cp.propagate_conflict_inds, conflict as i32);
            cp.conflict_flag[conflict] |= 4;
        }
    }

    /// CutpoolPropagation::updateActivityLbChange / UbChange
    #[allow(clippy::too_many_arguments)]
    fn cut_update_activity<const UPPER_SIDE: bool>(
        &mut self,
        pool: usize,
        col: usize,
        oldbound: f64,
        newbound: f64,
        threshold: bool,
        activity: bool,
        infeasdomain: bool,
    ) {
        let upper = UPPER_SIDE;
        let cp = self.cutpool(pool);
        let feastol = self.feastol;
        let inf = if upper { INF } else { -INF };
        let (rowindex, value, mut cap) = (cp.ar_rowindex, cp.ar_value, cp.capacity_threshold);
        let (thr_applies, thr_range) = self.threshold_range(col, newbound, upper);
        // the entries whose threshold a relaxation changes, and those whose
        // min activity the bound change moves
        let (relax_head, relax_next, act_head, act_next) = if upper {
            (cp.head_pos, cp.next_pos, cp.head_neg, cp.next_neg)
        } else {
            (cp.head_neg, cp.next_neg, cp.head_pos, cp.next_pos)
        };
        let relaxed = if upper { newbound > oldbound } else { newbound < oldbound };
        if relaxed && threshold && thr_applies {
            let mut it = relax_head[col];
            while it != -1 {
                let row = rowindex[it as usize] as usize;
                let val = value[it as usize];
                cap[row] = max3(cap[row], val.abs() * thr_range, feastol);
                it = relax_next[it as usize];
            }
        }

        if !activity {
            return;
        }

        let (mut act, mut ninf, rhs) = (cp.activitycuts, cp.activitycutsinf, cp.rhs);
        if !infeasdomain {
            let mut it = act_head[col];
            while it != -1 {
                let row = rowindex[it as usize] as usize;
                let val = value[it as usize];
                let deltamin = compute_delta(val, oldbound, newbound, inf, &mut ninf[row]);
                act[row] += deltamin;
                if deltamin <= 0.0 {
                    if thr_applies {
                        cap[row] = max3(cap[row], val.abs() * thr_range, feastol);
                    }
                } else if ninf[row] == 0 && act[row] - rhs[row] > feastol {
                    self.set_infeasible(cp.cutpoolindex, row as i32);
                    break;
                } else {
                    self.mark_propagate_cut_in(cp, row);
                }
                it = act_next[it as usize];
            }
        }

        if *self.infeasible {
            let reason = *self.infeasible_reason;
            let (oldbound, newbound) = (newbound, oldbound);
            let mut it = act_head[col];
            while it != -1 {
                let row = rowindex[it as usize] as usize;
                let val = value[it as usize];
                let d = compute_delta(val, oldbound, newbound, inf, &mut ninf[row]);
                act[row] += d;
                if reason.index == row as i32 && reason.kind == cp.cutpoolindex {
                    break;
                }
                it = act_next[it as usize];
            }
        }
    }

    /// ConflictPoolPropagation::updateActivityLbChange / UbChange
    fn conflict_update_activity(&mut self, pool: usize, col: usize, oldbound: f64, newbound: f64, upper: bool) {
        let cp = self.conflictpool(pool);
        let mut i = if upper { cp.col_upper_watched[col] } else { cp.col_lower_watched[col] };
        while i != -1 {
            let w = cp.watched[i as usize];
            let conflict = (i >> 1) as usize;
            let b = w.domchg.boundval;
            let delta = if upper {
                (b < newbound) as u8 as i32 - (b < oldbound) as u8 as i32
            } else {
                (b > newbound) as u8 as i32 - (b > oldbound) as u8 as i32
            };
            if delta != 0 {
                cp.conflict_flag[conflict] = cp.conflict_flag[conflict].wrapping_add(delta as u8);
                self.mark_propagate_conflict(cp, conflict);
            }
            i = w.next;
        }
    }

    /// doChangeBound: sets the bound and updates the activities; returns the
    /// old bound
    pub fn do_change_bound(&mut self, chg: DomChg) -> f64 {
        let col = chg.column as usize;
        let upper = chg.boundtype != LOWER;
        let oldbound = if upper { self.col_upper[col] } else { self.col_lower[col] };
        if upper {
            self.col_upper[col] = chg.boundval;
        } else {
            self.col_lower[col] = chg.boundval;
        }
        if oldbound != chg.boundval {
            if !*self.infeasible {
                self.update_activity(col, oldbound, chg.boundval, upper);
            }
            if self.changedcolsflags[col] == 0 {
                self.changedcolsflags[col] = 1;
                self.push_i32(self.changedcols, col as i32);
            }
        }
        oldbound
    }

    /// changeBound without the fixings implied by a fixed binary; returns
    /// the column and value of such a fixing, if any
    pub fn change_bound_no_implications(&mut self, mut chg: DomChg, reason: Reason) -> Option<(i32, i32)> {
        let col = chg.column as usize;
        let prev_pos;
        let stacksize = self.domchgstack.len() as i32;
        if chg.boundtype == LOWER {
            if chg.boundval <= self.col_lower[col] {
                if reason.kind != REASON_BRANCHING {
                    return None;
                }
                chg.boundval = self.col_lower[col];
            }
            if chg.boundval > self.col_upper[col] {
                if chg.boundval - self.col_upper[col] > self.feastol {
                    if !*self.infeasible {
                        *self.infeasible_pos = stacksize;
                        *self.infeasible = true;
                        *self.infeasible_reason = Reason { kind: REASON_CONFLICTING_BOUNDS, index: stacksize };
                    }
                } else {
                    chg.boundval = self.col_upper[col];
                    if chg.boundval == self.col_lower[col] {
                        return None;
                    }
                }
            }
            prev_pos = self.col_lower_pos[col];
            self.col_lower_pos[col] = stacksize;
        } else {
            if chg.boundval >= self.col_upper[col] {
                if reason.kind != REASON_BRANCHING {
                    return None;
                }
                chg.boundval = self.col_upper[col];
            }
            if chg.boundval < self.col_lower[col] {
                if self.col_lower[col] - chg.boundval > self.feastol {
                    if !*self.infeasible {
                        *self.infeasible_pos = stacksize;
                        *self.infeasible = true;
                        *self.infeasible_reason = Reason { kind: REASON_CONFLICTING_BOUNDS, index: stacksize };
                    }
                } else {
                    chg.boundval = self.col_lower[col];
                    if chg.boundval == self.col_upper[col] {
                        return None;
                    }
                }
            }
            prev_pos = self.col_upper_pos[col];
            self.col_upper_pos[col] = stacksize;
        }

        if reason.kind == REASON_BRANCHING {
            self.push_i32(self.branchpos, stacksize);
        }

        // isBinary
        let binary = !self.is_continuous(col) && self.col_lower[col] == 0.0 && self.col_upper[col] == 1.0;

        let oldbound = self.do_change_bound(chg);

        let (rp, rd, rr) = (self.reserve_prev, self.reserve_domchg, self.reserve_reason);
        self.prevboundval.push(PrevBound { val: oldbound, pos: prev_pos }, rp);
        self.domchgstack.push(chg, rd);
        self.domchgreason.push(reason, rr);

        if binary && !*self.infeasible && self.is_fixed(col) {
            Some((col as i32, (self.col_lower[col] > 0.5) as i32))
        } else {
            None
        }
    }

    /// backtrack (to_global = false) / backtrackToGlobal (to_global = true)
    pub fn backtrack(&mut self, to_global: bool) -> DomChg {
        let stacksize = self.domchgstack.len() as i32;
        let mut k = stacksize - 1;
        let old_infeasible = *self.infeasible;
        let old_reason = *self.infeasible_reason;

        if *self.infeasible && *self.infeasible_pos == stacksize {
            *self.infeasible = false;
            *self.infeasible_reason = Reason::UNSPECIFIED;
        }

        while k >= 0 {
            let ku = k as usize;
            let prev = self.prevboundval.as_slice()[ku];
            let chg = self.domchgstack.as_slice()[ku];
            let col = chg.column as usize;
            if chg.boundtype == LOWER {
                self.col_lower_pos[col] = prev.pos;
            } else {
                self.col_upper_pos[col] = prev.pos;
            }

            // change back to the previous bound (backtrackToGlobal skips the
            // call when the value is the same, which does nothing)
            if !to_global || prev.val != chg.boundval {
                self.do_change_bound(DomChg { boundval: prev.val, column: chg.column, boundtype: chg.boundtype });
            }

            if *self.infeasible && *self.infeasible_pos == k {
                *self.infeasible = false;
                *self.infeasible_reason = Reason::UNSPECIFIED;
            }

            if !to_global && self.domchgreason.as_slice()[ku].kind == REASON_BRANCHING {
                self.branchpos.pop();
                break;
            }

            k -= 1;
        }

        if old_infeasible {
            self.mark_propagate_cut(old_reason);
            *self.infeasible_reason = Reason::UNSPECIFIED;
            *self.infeasible = false;
        }

        let numreason = self.domchgreason.len();
        for i in (k + 1) as usize..numreason {
            let r = self.domchgreason.as_slice()[i];
            self.mark_propagate_cut(r);
        }

        if k < 0 {
            self.domchgstack.clear();
            self.prevboundval.clear();
            self.domchgreason.clear();
            self.branchpos.clear();
            return DomChg { boundval: 0.0, column: -1, boundtype: LOWER };
        }

        let ku = k as usize;
        let chg = self.domchgstack.as_slice()[ku];
        self.domchgstack.truncate(ku);
        self.domchgreason.truncate(ku);
        self.prevboundval.truncate(ku);
        chg
    }

    /// Whether propagate() has anything to do
    fn have_propagation_rows(&self) -> bool {
        if !self.propagateinds.is_empty() {
            return true;
        }
        if self.objprop.active && self.obj_should_be_propagated() {
            return true;
        }
        for pool in 0..self.cutpools.len() {
            if !self.cutpools[pool].propagatecutinds.is_empty() {
                return true;
            }
        }
        for pool in 0..self.conflictpools.len() {
            if !self.conflictpools[pool].propagate_conflict_inds.is_empty() {
                return true;
            }
        }
        false
    }

    /// The model-row part of propagate(): the bound changes of each row in
    /// `rows`, at changedbounds[2 * ARstart[row]..], their numbers (upper
    /// side, lower side) in `counts`
    pub fn propagate_model_rows(&mut self, rows: &[i32], counts: &mut [[i32; 2]], changedbounds: &mut [DomChg]) {
        let feastol = self.feastol;
        for (k, &r) in rows.iter().enumerate() {
            let i = r as usize;
            let s = self.ar_start[i] as usize;
            let (index, value) = self.row(i);
            let (rl, ru) = (self.row_lower[i], self.row_upper[i]);
            // the capacity threshold (recomputed if a side is propagated) is
            // computed in the first propagation pass
            let mut cap = None;
            counts[k] = [0, 0];

            if ru != INF && (self.activitymaxinf[i] != 0 || self.activitymax[i] > ru + feastol) {
                self.activitymin[i].renormalize();
                let (n, c) = self.bounds().propagate_row_cap(
                    index,
                    value,
                    ru,
                    self.activitymin[i],
                    self.activitymininf[i],
                    false,
                    &mut changedbounds[2 * s..],
                );
                counts[k][0] = n as i32;
                cap = Some(c);
            }

            if rl != -INF && (self.activitymininf[i] != 0 || self.activitymin[i] < rl - feastol) {
                self.activitymax[i].renormalize();
                let (amax, nmax) = (self.activitymax[i], self.activitymaxinf[i]);
                let out = &mut changedbounds[2 * s + counts[k][0] as usize..];
                let n = if cap.is_none() {
                    let (n, c) = self.bounds().propagate_row_cap(index, value, rl, amax, nmax, true, out);
                    cap = Some(c);
                    n
                } else {
                    self.bounds().propagate_row(index, value, rl, amax, nmax, true, out)
                };
                counts[k][1] = n as i32;
            }

            if let Some(c) = cap {
                self.capacity_threshold[i] = c;
            }
        }
    }

    /// The cut part of propagate() for cut pool `pool`: as
    /// propagate_model_rows, at changedbounds[rowstart(cut)..], the number in
    /// counts[k][0]
    pub fn propagate_cuts(&mut self, pool: usize, cuts: &[i32], counts: &mut [[i32; 2]], changedbounds: &mut [DomChg]) {
        let cp = self.cutpool(pool);
        for (k, &c) in cuts.iter().enumerate() {
            let i = c as usize;
            counts[k] = [0, 0];
            // first check if cut is marked as deleted
            if cp.propagatecutflags[i] & 2 != 0 {
                continue;
            }
            let [s, e] = cp.ar_range[i];
            let (s, e) = (s as usize, e as usize);
            // SAFETY: the pool's matrix, which nothing writes here
            let (index, value) = unsafe { (&cp.ar_index.get()[s..e], &cp.ar_value.get()[s..e]) };
            cp.activitycuts[i].renormalize();
            let (n, c) = self.bounds().propagate_row_cap(
                index,
                value,
                cp.rhs[i],
                cp.activitycuts[i],
                cp.activitycutsinf[i],
                false,
                &mut changedbounds[s..],
            );
            counts[k][0] = n as i32;
            cp.capacity_threshold[i] = c;
        }
    }

    /// The scratch of a propagate() batch of n rows: the rows, their
    /// counts (sized n) and the bound changes
    #[allow(clippy::type_complexity)]
    fn scratch(&mut self, n: usize) -> (*const [i32], *mut [[i32; 2]], *mut [DomChg]) {
        let r = self.reserve_pair;
        self.scratch_counts.reserve(n, r);
        self.scratch_counts.set_len_filled(n, [0, 0]);
        (self.scratch_inds.as_slice(), self.scratch_counts.as_mut_slice(), self.scratch_bounds.as_mut_slice())
    }

    /// The size propagate() needs for the bound changes of a batch
    fn changed_bound_size(&self) -> usize {
        let mut size = 2 * self.ar_value.len();
        for pool in 0..self.cutpools.len() {
            size = size.max(self.cutpools[pool].ar_value.len());
        }
        size
    }

    /// ConflictPoolPropagation::linkWatchedLiteral (which links a literal
    /// only into a nonempty list, as the C++)
    fn link_watched_literal(cp: &mut CConfProp, pos: usize) {
        let w = cp.watched[pos].domchg;
        let col = w.column as usize;
        let head = if w.boundtype == LOWER { cp.col_lower_watched[col] } else { cp.col_upper_watched[col] };
        cp.watched[pos].prev = -1;
        cp.watched[pos].next = head;
        if head != -1 {
            cp.watched[head as usize].prev = pos as i32;
            if w.boundtype == LOWER {
                cp.col_lower_watched[col] = pos as i32;
            } else {
                cp.col_upper_watched[col] = pos as i32;
            }
        }
    }

    /// ConflictPoolPropagation::unlinkWatchedLiteral
    fn unlink_watched_literal(cp: &mut CConfProp, pos: usize) {
        let w = cp.watched[pos].domchg;
        if w.column == -1 {
            return;
        }
        let col = w.column as usize;
        cp.watched[pos].domchg.column = -1;
        let (prev, next) = (cp.watched[pos].prev, cp.watched[pos].next);
        if prev != -1 {
            cp.watched[prev as usize].next = next;
        } else if w.boundtype == LOWER {
            cp.col_lower_watched[col] = next;
        } else {
            cp.col_upper_watched[col] = next;
        }
        if next != -1 {
            cp.watched[next as usize].prev = prev;
        }
    }

    /// The part of ConflictPoolPropagation::propagateConflict before a bound
    /// change: returns the bound change to make, if any
    fn propagate_conflict_check(&mut self, pool: usize, conflict: usize) -> Option<DomChg> {
        let cp = self.conflictpool(pool);
        // remove propagate flag, but keep watched and deleted information
        cp.conflict_flag[conflict] &= 3 | 8;
        // if two inactive literals are watched or conflict has been deleted skip
        if cp.conflict_flag[conflict] >= 2 {
            return None;
        }
        if *self.infeasible {
            return None;
        }
        let [start, end] = cp.ranges[conflict];
        if start == -1 {
            Self::unlink_watched_literal(cp, 2 * conflict);
            Self::unlink_watched_literal(cp, 2 * conflict + 1);
            return None;
        }
        let mut inactive = [0usize; 2];
        let mut num_inactive = 0;
        for i in start as usize..end as usize {
            if self.is_active(&cp.entries[i]) {
                continue;
            }
            inactive[num_inactive] = i;
            num_inactive += 1;
            if num_inactive == 2 {
                break;
            }
        }
        cp.conflict_flag[conflict] = num_inactive as u8;
        let reason_kind = (self.cutpools.len() + pool) as i32;
        match num_inactive {
            0 => {
                self.set_infeasible(reason_kind, conflict as i32);
                // SAFETY: resetAge touches only the pool's ages
                unsafe { (self.c.conflict_reset_age)(self.c.dom, pool as i32, conflict as i32) };
                None
            }
            1 => {
                let domchg = self.flip(&cp.entries[inactive[0]]);
                if !self.is_active(&domchg) {
                    Some(domchg)
                } else {
                    None
                }
            }
            _ => {
                for w in 0..2 {
                    let pos = 2 * conflict + w;
                    let e = cp.entries[inactive[w]];
                    if cp.watched[pos].domchg != e {
                        Self::unlink_watched_literal(cp, pos);
                        cp.watched[pos].domchg = e;
                        Self::link_watched_literal(cp, pos);
                    }
                }
                None
            }
        }
    }
}

/// The domain behind a view, for the code that changes bounds: Dom views
/// are borrowed from it and so cannot be alive across a change_bound, which
/// may call back into C++ and re-enter this code
pub struct Ctx {
    c: *mut CDomain,
}

impl Ctx {
    /// # Safety
    /// `c` a valid view whose pointers stay valid during the Ctx's use
    pub unsafe fn new(c: *const CDomain) -> Ctx {
        Ctx { c: c as *mut CDomain }
    }

    #[inline(always)]
    pub fn dom(&mut self) -> Dom<'_> {
        // SAFETY: Ctx::new's contract; the view is the only one alive (the
        // borrow of self)
        unsafe { CDomain::view(self.c) }
    }

    #[inline(always)]
    pub fn infeasible(&mut self) -> bool {
        *self.dom().infeasible
    }

    /// changeBound
    pub fn change_bound(&mut self, chg: DomChg, reason: Reason) {
        let fixing = self.dom().change_bound_no_implications(chg, reason);
        if let Some((col, val)) = fixing {
            // SAFETY: C++ applies the clique table's and the implications'
            // fixings through HighsDomain::changeBound; no view is alive
            unsafe { ((*self.c).implications)((*self.c).dom, col, val) };
        }
    }

    /// checkChangeBound
    pub fn check_change_bound(&mut self, boundtype: i32, col: usize, boundval: CDouble, reason: Reason) -> bool {
        let (bound, accept) = {
            let d = self.dom();
            if (boundval * K_HIGHS_TINY).to_f64().abs() > d.feastol {
                return false;
            }
            if boundtype == LOWER {
                d.bounds().adjusted_lb(col, boundval)
            } else {
                d.bounds().adjusted_ub(col, boundval)
            }
        };
        if !accept {
            return false;
        }
        self.change_bound(DomChg { boundval: bound, column: col as i32, boundtype }, reason);
        true
    }

    /// ConflictPoolPropagation::propagateConflict
    fn propagate_conflict(&mut self, pool: usize, conflict: usize) {
        let chg = self.dom().propagate_conflict_check(pool, conflict);
        if let Some(chg) = chg {
            let ncutpools = self.dom().cutpools.len();
            self.change_bound(chg, Reason { kind: (ncutpools + pool) as i32, index: conflict as i32 });
            // SAFETY: resetAge touches only the pool's ages
            unsafe { ((*self.c).conflict_reset_age)((*self.c).dom, pool as i32, conflict as i32) };
        }
    }

    /// Applies the bound changes changedbounds[start..start + n] with the
    /// given reason, stopping at infeasibility
    fn apply_changes(&mut self, start: usize, n: usize, reason: Reason) {
        for j in start..start + n {
            let chg = {
                let d = self.dom();
                if *d.infeasible {
                    return;
                }
                d.scratch_bounds.as_slice()[j]
            };
            self.change_bound(chg, reason);
        }
    }

    /// Whether chg would not tighten the current bound
    fn redundant(&mut self, chg: &DomChg) -> bool {
        let d = self.dom();
        let c = chg.column as usize;
        if chg.boundtype == LOWER {
            chg.boundval <= d.col_lower[c]
        } else {
            chg.boundval >= d.col_upper[c]
        }
    }

    /// changeBound with an unspecified reason, then propagate(); returns
    /// whether the domain is infeasible
    fn change_and_propagate(&mut self, chg: DomChg, reason: Reason) -> bool {
        self.change_bound(chg, reason);
        if !self.infeasible() {
            self.propagate();
        }
        self.infeasible()
    }

    /// setDomainChangeStack (branching = None) / setDomainChangeStack with
    /// branching positions
    pub fn set_domain_change_stack(&mut self, stack: &[DomChg], branching: Option<&[i32]>) {
        {
            let mut d = self.dom();
            *d.infeasible = false;
            let (mut lpos, mut upos) = (d.col_lower_pos, d.col_upper_pos);
            for chg in d.domchgstack.as_slice() {
                if chg.boundtype == LOWER {
                    lpos[chg.column as usize] = -1;
                } else {
                    upos[chg.column as usize] = -1;
                }
            }
            d.prevboundval.clear();
            d.domchgstack.clear();
            d.domchgreason.clear();
            d.branchpos.clear();
        }
        let Some(branching) = branching else {
            for chg in stack {
                if self.redundant(chg) {
                    continue;
                }
                self.change_bound(*chg, Reason::UNSPECIFIED);
                if self.infeasible() {
                    break;
                }
            }
            return;
        };

        let stacksize = stack.len();
        let mut k = 0;
        for &branch_pos in branching {
            while k < branch_pos as usize {
                if !self.redundant(&stack[k]) && self.change_and_propagate(stack[k], Reason::UNSPECIFIED) {
                    return;
                }
                k += 1;
            }

            if k == stacksize {
                return;
            }

            // For redundant branching bound changes we need to be more
            // careful due to symmetry handling. If these bound changes are
            // redundant simply because the corresponding subtree was
            // enumerated and hence the global bound updated, then we still
            // need to keep their status as branching variables for computing
            // correct stabilizers. They can, however, be safely dropped if
            // they are either strictly redundant in the global domain, or if
            // there is already a local bound change that makes the branching
            // change redundant.
            let chg = stack[k];
            {
                let d = self.dom();
                let c = chg.column as usize;
                if chg.boundtype == LOWER {
                    if chg.boundval <= d.col_lower[c] && (chg.boundval < d.col_lower[c] || d.col_lower_pos[c] != -1) {
                        continue;
                    }
                } else if chg.boundval >= d.col_upper[c] && (chg.boundval > d.col_upper[c] || d.col_upper_pos[c] != -1) {
                    continue;
                }
            }

            if self.change_and_propagate(chg, Reason { kind: REASON_BRANCHING, index: 0 }) {
                return;
            }
        }

        while k < stacksize {
            if !self.redundant(&stack[k]) && self.change_and_propagate(stack[k], Reason::UNSPECIFIED) {
                break;
            }
            k += 1;
        }
    }

    /// propagate()
    pub fn propagate(&mut self) -> bool {
        {
            let d = self.dom();
            if !d.have_propagation_rows() {
                return false;
            }
            let size = d.changed_bound_size();
            let mut b = d.scratch_bounds;
            b.reserve(size, d.reserve_domchg);
            let n = size.max(b.len());
            b.set_len_filled(n, DomChg::default());
        }

        while self.dom().have_propagation_rows() {
            if self.dom().objprop.active {
                self.obj_propagate();
            }

            let numconflictpools = self.dom().conflictpools.len();
            for pool in 0..numconflictpools {
                loop {
                    let n = {
                        let d = self.dom();
                        let mut inds = d.conflictpools[pool].propagate_conflict_inds;
                        if inds.is_empty() {
                            break;
                        }
                        let mut scratch = d.scratch_inds;
                        scratch.swap(&mut inds);
                        scratch.len()
                    };
                    for k in 0..n {
                        let conflict = self.dom().scratch_inds.as_slice()[k];
                        self.propagate_conflict(pool, conflict as usize);
                    }
                    self.dom().scratch_inds.clear();
                }
            }

            let n = {
                let mut d = self.dom();
                if d.propagateinds.is_empty() {
                    0
                } else {
                    let (mut scratch, mut inds) = (d.scratch_inds, d.propagateinds);
                    scratch.swap(&mut inds);
                    for &row in scratch.as_slice() {
                        d.propagateflags[row as usize] = 0;
                    }
                    scratch.len()
                }
            };
            if n != 0 {
                if !self.infeasible() {
                    {
                        let mut d = self.dom();
                        let (rows, counts, bounds) = d.scratch(n);
                        // SAFETY: the scratch vectors are distinct from the
                        // data propagate_model_rows reads and writes
                        unsafe { d.propagate_model_rows(&*rows, &mut *counts, &mut *bounds) };
                    }
                    for k in 0..n {
                        let (row, cnt, start) = {
                            let d = self.dom();
                            let row = d.scratch_inds.as_slice()[k];
                            (row, d.scratch_counts.as_slice()[k], 2 * d.ar_start[row as usize] as usize)
                        };
                        if cnt[0] != 0 {
                            self.apply_changes(start, cnt[0] as usize, Reason { kind: REASON_MODEL_ROW_UPPER, index: row });
                            if self.infeasible() {
                                break;
                            }
                        }
                        if cnt[1] != 0 {
                            self.apply_changes(
                                start + cnt[0] as usize,
                                cnt[1] as usize,
                                Reason { kind: REASON_MODEL_ROW_LOWER, index: row },
                            );
                            if self.infeasible() {
                                break;
                            }
                        }
                    }
                }
                self.dom().scratch_inds.clear();
            }

            let numpools = self.dom().cutpools.len();
            for pool in 0..numpools {
                let n = {
                    let d = self.dom();
                    let cp = d.cutpool(pool);
                    if cp.propagatecutinds.is_empty() {
                        continue;
                    }
                    let mut scratch = d.scratch_inds;
                    scratch.swap(&mut cp.propagatecutinds);
                    for &cut in scratch.as_slice() {
                        cp.propagatecutflags[cut as usize] &= 2;
                    }
                    scratch.len()
                };
                if !self.infeasible() {
                    {
                        let mut d = self.dom();
                        let (cuts, counts, bounds) = d.scratch(n);
                        // SAFETY: as for the model rows
                        unsafe { d.propagate_cuts(pool, &*cuts, &mut *counts, &mut *bounds) };
                    }
                    for k in 0..n {
                        let (cut, cnt, start) = {
                            let d = self.dom();
                            let cut = d.scratch_inds.as_slice()[k];
                            let start = d.cutpools[pool].ar_range[cut as usize][0];
                            (cut, d.scratch_counts.as_slice()[k], start as usize)
                        };
                        if cnt[0] != 0 {
                            // SAFETY: resetAge touches only the pool's ages
                            // and its set of propagation rows
                            unsafe { ((*self.c).cut_reset_age)((*self.c).dom, pool as i32, cut) };
                            self.apply_changes(start, cnt[0] as usize, Reason { kind: pool as i32, index: cut });
                        }
                        if self.infeasible() {
                            break;
                        }
                    }
                }
                self.dom().scratch_inds.clear();
            }
        }
        true
    }
}

pub mod ffi {
    //! The `extern "C"` entry points called by HighsDomain (under HIGHS_RUST)
    use super::*;

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_set_domain_change_stack(
        d: *const CDomain,
        stack: *const DomChg,
        len: i32,
        branching: *const i32,
        nbranching: i32,
    ) {
        let branching = if branching.is_null() { None } else { Some(sl(branching, nbranching)) };
        Ctx::new(d).set_domain_change_stack(sl(stack, len), branching);
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_tighten_coefficients(
        b: *const CBounds,
        inds: *const i32,
        vals: *mut f64,
        len: i32,
        rhs: *mut f64,
    ) {
        (*b).view().tighten_coefficients(sl(inds, len), sl_mut(vals, len), &mut *rhs);
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_change_bound(d: *const CDomain, chg: DomChg, reason: Reason) {
        Ctx::new(d).change_bound(chg, reason);
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_propagate(d: *const CDomain) -> bool {
        Ctx::new(d).propagate()
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_backtrack(d: *const CDomain, to_global: bool) -> DomChg {
        CDomain::view(d as *mut CDomain).backtrack(to_global)
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_compute_row_activities(d: *const CDomain) {
        CDomain::view(d as *mut CDomain).compute_row_activities();
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_compute_activity(
        b: *const CBounds,
        index: *const i32,
        value: *const f64,
        len: i32,
        max: bool,
        ninf: *mut i32,
        activity: *mut CDouble,
    ) {
        let (n, a) = (*b).view().compute_activity(sl(index, len), sl(value, len), max);
        *ninf = n;
        *activity = a;
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_propagate_row(
        b: *const CBounds,
        index: *const i32,
        value: *const f64,
        len: i32,
        rhs: f64,
        activity: *const CDouble,
        ninf: i32,
        lower: bool,
        out: *mut DomChg,
    ) -> i32 {
        // propagate_row writes at most one change per entry
        (*b).view().propagate_row(sl(index, len), sl(value, len), rhs, *activity, ninf, lower, sl_mut(out, len))
            as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn std_vec_layout() {
        assert_eq!(std::mem::size_of::<StdVec<i32>>(), 3 * std::mem::size_of::<usize>());
        assert_eq!(std::mem::size_of::<DomChg>(), 16);
        assert_eq!(std::mem::size_of::<PrevBound>(), 16);
        assert_eq!(std::mem::size_of::<WatchedLiteral>(), 24);
    }

    /// x0 + 2 x1 <= 4 with integer x0, x1 in [0, 10]
    #[test]
    fn propagate_knapsack_row() {
        let index = [0, 1];
        let value = [1.0, 2.0];
        let (col_lower, col_upper) = (vec![0.0, 0.0], vec![10.0, 10.0]);
        let b = Bounds {
            feastol: 1e-6,
            epsilon: 1e-9,
            col_lower: &col_lower,
            col_upper: &col_upper,
            integrality: &[1, 1],
            col_lower_pos: &[-1, -1],
            col_upper_pos: &[-1, -1],
            prevboundval: &[],
            infeasible: false,
            infeasible_pos: 0,
        };
        let (ninf, act) = b.compute_activity(&index, &value, false);
        assert_eq!((ninf, act.to_f64()), (0, 0.0));
        let mut chg = [DomChg::default(); 2];
        assert_eq!(b.propagate_row(&index, &value, 4.0, act, ninf, false, &mut chg), 2);
        assert_eq!(chg[0], DomChg { boundval: 4.0, column: 0, boundtype: UPPER });
        assert_eq!(chg[1], DomChg { boundval: 2.0, column: 1, boundtype: UPPER });
    }
}
