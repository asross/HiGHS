//! The propagation engine of HighsDomain (highs/mip/HighsDomain.cpp): row
//! activities with their infinity counts, the capacity thresholds that
//! decide when a row is worth propagating, bound propagation of model rows
//! and cuts, and the watched literals of the conflict pools.
//!
//! # The view
//!
//! HighsDomain keeps owning its data (external code reads col_lower_,
//! col_upper_, the domain change stack, ... everywhere).
//! `highs_rs::DomainAccess` (highs/mip/HighsDomainRust.h) fills a
//! `#[repr(C)]` [`CDomain`] with pointer+length pairs of the model matrices
//! (column-wise from the model, row-wise from mipdata), of the domain's
//! vectors, of the cut pools' dynamic row matrices and propagation arrays
//! ([`CCutProp`]) and of the conflict pools' watched literals
//! ([`CConfProp`]); [`CDomain::view`] turns it into a [`Dom`] of slices
//! (the pools' on use). Activities are HighsCDouble arrays, laid out as
//! [`CDouble`] {hi, lo} (static_assert in HighsDomainRust.h).
//!
//! Filling the view per call cost as much as the C++ kernels saved, so
//! HighsDomain caches it (rsView_, highs/mip/HighsDomainRustView.h) and
//! drops it wherever a vector it points to may move: copy, assignment,
//! computeRowActivities, adding or clearing pools, cutAdded (which resizes
//! the pool arrays and may move the cut matrix) and conflictAdded. Only the
//! sizes of the domain change stack are refreshed per call. Debug builds
//! compare the cache with a fresh fill on every use. The const methods
//! (computeMin/MaxActivity, propagateRowUpper/Lower), which threads may call
//! concurrently on the global domain, fill a temporary view instead.
//!
//! The kernels here are leaves: they never call back into HighsDomain, so
//! nothing a view points to is resized during a call. The only C++ vectors
//! they grow are the lists of rows, cuts and conflicts to propagate, through
//! the `push` function pointer (each entry is pushed at most once per
//! propagation round, flagged as in the C++). changeBound, backtracking, the
//! clique table and implications, objective propagation and conflict
//! analysis stay in C++ and call these kernels.
//!
//! Every floating-point operation is the C++ one in the same order
//! (HighsCDouble arithmetic via util::cdouble); the only contraction clang
//! makes here is `bound +- 1000.0 * feastol` in adjustedUb/adjustedLb.

use crate::ffi::{sl, sl_mut};
use crate::util::cdouble::CDouble;
use std::ffi::c_void;

const INF: f64 = f64::INFINITY;
const K_HIGHS_TINY: f64 = 1e-14;

// HighsDomain::Reason types
const REASON_MODEL_ROW_UPPER: i32 = -3;
const REASON_MODEL_ROW_LOWER: i32 = -4;

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

/// std::pair<double, HighsInt> of prevboundval_
#[repr(C)]
#[derive(Clone, Copy)]
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

/// A pointer and length, as std::vector's data() and size()
#[repr(C)]
pub struct CSlice<T> {
    p: *mut T,
    n: i32,
}

impl<T> CSlice<T> {
    /// # Safety
    /// `p` valid for `n` reads
    unsafe fn get<'a>(&self) -> &'a [T] {
        sl(self.p, self.n)
    }
    /// # Safety
    /// `p` valid for `n` reads and writes, unaliased
    unsafe fn get_mut<'a>(&self) -> &'a mut [T] {
        sl_mut(self.p, self.n)
    }
}

/// Appends a value to a std::vector<HighsInt>
pub type PushFn = unsafe extern "C" fn(*mut c_void, i32);

/// A CutpoolPropagation and its cut pool's matrix, mirrored by
/// highs_rs::CutProp in highs/mip/HighsDomainRust.h
#[repr(C)]
pub struct CCutProp {
    cutpoolindex: i32,
    activitycuts: CSlice<CDouble>,
    activitycutsinf: CSlice<i32>,
    propagatecutflags: CSlice<u8>,
    capacity_threshold: CSlice<f64>,
    propagatecutinds: *mut c_void,
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

/// A ConflictPoolPropagation, mirrored by highs_rs::ConfProp
#[repr(C)]
pub struct CConfProp {
    col_lower_watched: CSlice<i32>,
    col_upper_watched: CSlice<i32>,
    watched: CSlice<WatchedLiteral>,
    conflict_flag: CSlice<u8>,
    propagate_conflict_inds: *mut c_void,
}

/// HighsDomain's data, filled by HighsDomain::rustView(), mirrored by
/// highs_rs::Domain in highs/mip/HighsDomainRust.h
#[repr(C)]
pub struct CDomain {
    feastol: f64,
    epsilon: f64,
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
    integrality: CSlice<u8>,
    // the domain
    col_lower: CSlice<f64>,
    col_upper: CSlice<f64>,
    activitymin: CSlice<CDouble>,
    activitymax: CSlice<CDouble>,
    activitymininf: CSlice<i32>,
    activitymaxinf: CSlice<i32>,
    capacity_threshold: CSlice<f64>,
    propagateflags: CSlice<u8>,
    propagateinds: *mut c_void,
    col_lower_pos: CSlice<i32>,
    col_upper_pos: CSlice<i32>,
    prevboundval: CSlice<PrevBound>,
    infeasible: *mut bool,
    infeasible_reason: *mut Reason,
    infeasible_pos: *mut i32,
    domchgstack_size: i32,
    cutpools: CSlice<CCutProp>,
    conflictpools: CSlice<CConfProp>,
    push: PushFn,
}

pub struct CutProp<'a> {
    cutpoolindex: i32,
    activitycuts: &'a mut [CDouble],
    activitycutsinf: &'a mut [i32],
    propagatecutflags: &'a mut [u8],
    capacity_threshold: &'a mut [f64],
    propagatecutinds: *mut c_void,
    ar_range: &'a [[i32; 2]],
    ar_index: &'a [i32],
    ar_value: &'a [f64],
    ar_rowindex: &'a [i32],
    next_pos: &'a [i32],
    next_neg: &'a [i32],
    head_pos: &'a [i32],
    head_neg: &'a [i32],
    rhs: &'a [f64],
}

pub struct ConfProp<'a> {
    col_lower_watched: &'a [i32],
    col_upper_watched: &'a [i32],
    watched: &'a [WatchedLiteral],
    conflict_flag: &'a mut [u8],
    propagate_conflict_inds: *mut c_void,
}

pub struct Dom<'a> {
    pub feastol: f64,
    pub epsilon: f64,
    a_start: &'a [i32],
    a_index: &'a [i32],
    a_value: &'a [f64],
    ar_start: &'a [i32],
    ar_index: &'a [i32],
    ar_value: &'a [f64],
    row_lower: &'a [f64],
    row_upper: &'a [f64],
    integrality: &'a [u8],
    col_lower: &'a [f64],
    col_upper: &'a [f64],
    activitymin: &'a mut [CDouble],
    activitymax: &'a mut [CDouble],
    activitymininf: &'a mut [i32],
    activitymaxinf: &'a mut [i32],
    capacity_threshold: &'a mut [f64],
    propagateflags: &'a mut [u8],
    propagateinds: *mut c_void,
    col_lower_pos: &'a [i32],
    col_upper_pos: &'a [i32],
    prevboundval: &'a [PrevBound],
    infeasible: &'a mut bool,
    infeasible_reason: &'a mut Reason,
    infeasible_pos: &'a mut i32,
    domchgstack_size: i32,
    cutpools: &'a [CCutProp],
    conflictpools: &'a [CConfProp],
    push: PushFn,
}

impl CDomain {
    /// # Safety
    /// Every pointer must be valid for its length and unaliased for the
    /// view's lifetime; `push` must append to the vectors it is given
    #[inline(always)]
    pub unsafe fn view<'a>(&self) -> Dom<'a> {
        Dom {
            feastol: self.feastol,
            epsilon: self.epsilon,
            a_start: self.a_start.get(),
            a_index: self.a_index.get(),
            a_value: self.a_value.get(),
            ar_start: self.ar_start.get(),
            ar_index: self.ar_index.get(),
            ar_value: self.ar_value.get(),
            row_lower: self.row_lower.get(),
            row_upper: self.row_upper.get(),
            integrality: self.integrality.get(),
            col_lower: self.col_lower.get(),
            col_upper: self.col_upper.get(),
            activitymin: self.activitymin.get_mut(),
            activitymax: self.activitymax.get_mut(),
            activitymininf: self.activitymininf.get_mut(),
            activitymaxinf: self.activitymaxinf.get_mut(),
            capacity_threshold: self.capacity_threshold.get_mut(),
            propagateflags: self.propagateflags.get_mut(),
            propagateinds: self.propagateinds,
            col_lower_pos: self.col_lower_pos.get(),
            col_upper_pos: self.col_upper_pos.get(),
            prevboundval: self.prevboundval.get(),
            infeasible: &mut *self.infeasible,
            infeasible_reason: &mut *self.infeasible_reason,
            infeasible_pos: &mut *self.infeasible_pos,
            domchgstack_size: self.domchgstack_size,
            cutpools: self.cutpools.get(),
            conflictpools: self.conflictpools.get(),
            push: self.push,
        }
    }
}

impl CCutProp {
    /// # Safety
    /// As for CDomain::view; at most one view of a pool at a time
    #[inline(always)]
    unsafe fn view<'a>(&self) -> CutProp<'a> {
        CutProp {
            cutpoolindex: self.cutpoolindex,
            activitycuts: self.activitycuts.get_mut(),
            activitycutsinf: self.activitycutsinf.get_mut(),
            propagatecutflags: self.propagatecutflags.get_mut(),
            capacity_threshold: self.capacity_threshold.get_mut(),
            propagatecutinds: self.propagatecutinds,
            ar_range: self.ar_range.get(),
            ar_index: self.ar_index.get(),
            ar_value: self.ar_value.get(),
            ar_rowindex: self.ar_rowindex.get(),
            next_pos: self.next_pos.get(),
            next_neg: self.next_neg.get(),
            head_pos: self.head_pos.get(),
            head_neg: self.head_neg.get(),
            rhs: self.rhs.get(),
        }
    }
}

impl CConfProp {
    /// # Safety
    /// As for CCutProp::view
    #[inline(always)]
    unsafe fn view<'a>(&self) -> ConfProp<'a> {
        ConfProp {
            col_lower_watched: self.col_lower_watched.get(),
            col_upper_watched: self.col_upper_watched.get(),
            watched: self.watched.get(),
            conflict_flag: self.conflict_flag.get_mut(),
            propagate_conflict_inds: self.propagate_conflict_inds,
        }
    }
}

/// std::max(a, b)
#[inline(always)]
fn max2(a: f64, b: f64) -> f64 {
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
fn bound_range(upper: f64, lower: f64, tolerance: f64, continuous: bool) -> f64 {
    let range = upper - lower;
    range
        - if continuous {
            max2(0.3 * range, 1000.0 * tolerance)
        } else {
            tolerance
        }
}

impl<'a> Dom<'a> {
    #[inline]
    fn is_continuous(&self, col: usize) -> bool {
        self.integrality[col] == 0
    }

    #[inline]
    fn push(&self, vec: *mut c_void, v: i32) {
        // SAFETY: C++ passes a std::vector<HighsInt>* and a function that
        // appends to it; no slice of the view points into these vectors
        unsafe { (self.push)(vec, v) }
    }

    /// getColLowerPos / getColUpperPos
    fn col_bound_pos(&self, col: usize, stackpos: i32, upper: bool) -> f64 {
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
        b
    }

    /// computeMinActivity (max = false) / computeMaxActivity (max = true)
    pub fn compute_activity(&self, index: &[i32], value: &[f64], max: bool) -> (i32, CDouble) {
        let mut act = CDouble::from(0.0);
        let mut ninf = 0;
        let inf = if max { INF } else { -INF };
        if *self.infeasible {
            let stackpos = *self.infeasible_pos - 1;
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

    pub fn recompute_capacity_threshold(&mut self, row: usize) {
        let (s, e) = (self.ar_start[row] as usize, self.ar_start[row + 1] as usize);
        self.capacity_threshold[row] = self.capacity_threshold_of(&self.ar_index[s..e], &self.ar_value[s..e]);
    }

    /// updateThresholdLbChange
    #[inline]
    fn threshold_lb_change(&self, col: usize, newbound: f64, val: f64, threshold: &mut f64) {
        let ub = self.col_upper[col];
        if newbound != ub {
            let t = val.abs() * bound_range(ub, newbound, self.feastol, self.is_continuous(col));
            *threshold = max3(*threshold, t, self.feastol);
        }
    }

    /// updateThresholdUbChange
    #[inline]
    fn threshold_ub_change(&self, col: usize, newbound: f64, val: f64, threshold: &mut f64) {
        let lb = self.col_lower[col];
        if newbound != lb {
            let t = val.abs() * bound_range(newbound, lb, self.feastol, self.is_continuous(col));
            *threshold = max3(*threshold, t, self.feastol);
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
            self.push(self.propagateinds, row as i32);
            self.propagateflags[row] = 1;
        }
    }

    /// computeRowActivities
    pub fn compute_row_activities(&mut self) {
        let nrow = self.row_lower.len();
        for i in 0..nrow {
            let (s, e) = (self.ar_start[i] as usize, self.ar_start[i + 1] as usize);
            let (index, value) = (&self.ar_index[s..e], &self.ar_value[s..e]);
            (self.activitymininf[i], self.activitymin[i]) = self.compute_activity(index, value, false);
            (self.activitymaxinf[i], self.activitymax[i]) = self.compute_activity(index, value, true);
            self.recompute_capacity_threshold(i);
            if (self.activitymininf[i] <= 1 && self.row_upper[i] != INF)
                || (self.activitymaxinf[i] <= 1 && self.row_lower[i] != -INF)
            {
                self.mark_propagate(i);
            }
        }
    }

    fn set_infeasible(&mut self, kind: i32, index: i32) {
        *self.infeasible = true;
        *self.infeasible_pos = self.domchgstack_size;
        *self.infeasible_reason = Reason { kind, index };
    }

    /// updateActivityLbChange (upper = false) / updateActivityUbChange
    /// (upper = true) without the objective propagation, which the C++
    /// caller does before and, if this returns infeasible, after
    pub fn update_activity(&mut self, col: usize, oldbound: f64, newbound: f64, upper: bool) {
        debug_assert!(!*self.infeasible);
        let feastol = self.feastol;
        let inf = if upper { INF } else { -INF };
        let start = self.a_start[col] as usize;
        let mut end = self.a_start[col + 1] as usize;
        for i in start..end {
            let row = self.a_index[i] as usize;
            let val = self.a_value[i];
            // a lower bound change moves the min activity of rows with a
            // positive coefficient, an upper bound change that of rows with a
            // negative one
            if (val > 0.0) != upper {
                let delta = compute_delta(val, oldbound, newbound, inf, &mut self.activitymininf[row]);
                self.activitymin[row] += delta;
                if delta <= 0.0 {
                    let mut t = self.capacity_threshold[row];
                    if upper {
                        self.threshold_ub_change(col, newbound, val, &mut t);
                    } else {
                        self.threshold_lb_change(col, newbound, val, &mut t);
                    }
                    self.capacity_threshold[row] = t;
                    continue;
                }
                let ru = self.row_upper[row];
                if ru != INF && self.activitymininf[row] == 0 && self.activitymin[row] - ru > feastol {
                    self.set_infeasible(REASON_MODEL_ROW_UPPER, row as i32);
                    end = i + 1;
                    break;
                }
                if self.activitymininf[row] <= 1 && self.propagateflags[row] == 0 && ru != INF {
                    self.mark_propagate(row);
                }
            } else {
                let delta = compute_delta(val, oldbound, newbound, inf, &mut self.activitymaxinf[row]);
                self.activitymax[row] += delta;
                if delta >= 0.0 {
                    let mut t = self.capacity_threshold[row];
                    if upper {
                        self.threshold_ub_change(col, newbound, val, &mut t);
                    } else {
                        self.threshold_lb_change(col, newbound, val, &mut t);
                    }
                    self.capacity_threshold[row] = t;
                    continue;
                }
                let rl = self.row_lower[row];
                if rl != -INF && self.activitymaxinf[row] == 0 && rl - self.activitymax[row] > feastol {
                    self.set_infeasible(REASON_MODEL_ROW_LOWER, row as i32);
                    end = i + 1;
                    break;
                }
                if self.activitymaxinf[row] <= 1 && self.propagateflags[row] == 0 && rl != -INF {
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
                    self.cut_update_activity(i, col, oldbound, newbound, upper, true, true, false);
                    if *self.infeasible {
                        infeascutpool = i as i32;
                    }
                } else {
                    self.cut_update_activity(i, col, oldbound, newbound, upper, true, false, true);
                }
            }
            for i in 0..infeascutpool.max(0) as usize {
                self.cut_update_activity(i, col, oldbound, newbound, upper, false, true, true);
            }
        }

        if *self.infeasible {
            let (oldbound, newbound) = (newbound, oldbound);
            for i in start..end {
                let row = self.a_index[i] as usize;
                let val = self.a_value[i];
                if (val > 0.0) != upper {
                    let d = compute_delta(val, oldbound, newbound, inf, &mut self.activitymininf[row]);
                    self.activitymin[row] += d;
                } else {
                    let d = compute_delta(val, oldbound, newbound, inf, &mut self.activitymaxinf[row]);
                    self.activitymax[row] += d;
                }
            }
        } else {
            for p in 0..self.conflictpools.len() {
                self.conflict_update_activity(p, col, oldbound, newbound, upper);
            }
        }
    }

    /// CutpoolPropagation::markPropagateCut
    fn mark_propagate_cut(&self, cp: &mut CutProp, cut: usize) {
        if cp.propagatecutflags[cut] == 0
            && (cp.activitycutsinf[cut] == 1
                || (cp.rhs[cut] - cp.activitycuts[cut].to_f64() <= cp.capacity_threshold[cut]))
        {
            self.push(cp.propagatecutinds, cut as i32);
            cp.propagatecutflags[cut] |= 1;
        }
    }

    /// CutpoolPropagation::updateActivityLbChange / UbChange
    #[allow(clippy::too_many_arguments)]
    fn cut_update_activity(
        &mut self,
        pool: usize,
        col: usize,
        oldbound: f64,
        newbound: f64,
        upper: bool,
        threshold: bool,
        activity: bool,
        infeasdomain: bool,
    ) {
        // take the pool out of self to borrow both
        // SAFETY: the only view of this pool
        let mut cp = unsafe { self.cutpools[pool].view() };
        self.cut_update_activity_in(&mut cp, col, oldbound, newbound, upper, threshold, activity, infeasdomain);
    }

    #[allow(clippy::too_many_arguments)]
    fn cut_update_activity_in(
        &mut self,
        cp: &mut CutProp,
        col: usize,
        oldbound: f64,
        newbound: f64,
        upper: bool,
        threshold: bool,
        activity: bool,
        infeasdomain: bool,
    ) {
        let inf = if upper { INF } else { -INF };
        // the entries whose threshold a relaxation changes, and those whose
        // min activity the bound change moves
        let (relax_head, relax_next, act_head, act_next) = if upper {
            (cp.head_pos, cp.next_pos, cp.head_neg, cp.next_neg)
        } else {
            (cp.head_neg, cp.next_neg, cp.head_pos, cp.next_pos)
        };
        let relaxed = if upper { newbound > oldbound } else { newbound < oldbound };
        if relaxed && threshold {
            let mut it = relax_head[col];
            while it != -1 {
                let row = cp.ar_rowindex[it as usize] as usize;
                let val = cp.ar_value[it as usize];
                if upper {
                    self.threshold_ub_change(col, newbound, val, &mut cp.capacity_threshold[row]);
                } else {
                    self.threshold_lb_change(col, newbound, val, &mut cp.capacity_threshold[row]);
                }
                it = relax_next[it as usize];
            }
        }

        if !activity {
            return;
        }

        if !infeasdomain {
            let mut it = act_head[col];
            while it != -1 {
                let row = cp.ar_rowindex[it as usize] as usize;
                let val = cp.ar_value[it as usize];
                let deltamin = compute_delta(val, oldbound, newbound, inf, &mut cp.activitycutsinf[row]);
                cp.activitycuts[row] += deltamin;
                if deltamin <= 0.0 {
                    if upper {
                        self.threshold_ub_change(col, newbound, val, &mut cp.capacity_threshold[row]);
                    } else {
                        self.threshold_lb_change(col, newbound, val, &mut cp.capacity_threshold[row]);
                    }
                } else if cp.activitycutsinf[row] == 0 && cp.activitycuts[row] - cp.rhs[row] > self.feastol {
                    self.set_infeasible(cp.cutpoolindex, row as i32);
                    break;
                } else {
                    self.mark_propagate_cut(cp, row);
                }
                it = act_next[it as usize];
            }
        }

        if *self.infeasible {
            let reason = *self.infeasible_reason;
            let (oldbound, newbound) = (newbound, oldbound);
            let mut it = act_head[col];
            while it != -1 {
                let row = cp.ar_rowindex[it as usize] as usize;
                let val = cp.ar_value[it as usize];
                let d = compute_delta(val, oldbound, newbound, inf, &mut cp.activitycutsinf[row]);
                cp.activitycuts[row] += d;
                if reason.index == row as i32 && reason.kind == cp.cutpoolindex {
                    break;
                }
                it = act_next[it as usize];
            }
        }
    }

    /// ConflictPoolPropagation::updateActivityLbChange / UbChange
    fn conflict_update_activity(&mut self, pool: usize, col: usize, oldbound: f64, newbound: f64, upper: bool) {
        let push = self.push;
        // SAFETY: the only view of this pool
        let cp = unsafe { self.conflictpools[pool].view() };
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
                // markPropagateConflict
                if cp.conflict_flag[conflict] < 2 {
                    // SAFETY: as in Dom::push
                    unsafe { push(cp.propagate_conflict_inds, conflict as i32) };
                    cp.conflict_flag[conflict] |= 4;
                }
            }
            i = w.next;
        }
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
            } else if 1000.0f64.mul_add(feastol, bound) < ub {
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
            } else if (-1000.0f64).mul_add(feastol, bound) > lb {
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

    /// impliedBoundNoTighter
    #[inline]
    fn implied_bound_no_tighter(&self, col: usize, upper: bool, estimate: f64, scale: f64) -> bool {
        let margin = 1e-12 * (scale + estimate.abs());
        if upper {
            estimate - margin >= self.col_upper[col]
        } else {
            estimate + margin <= self.col_lower[col]
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
        if ninf > 1 {
            return 0;
        }
        let mut numchgs = 0;
        let act = activity.to_f64();
        let inf = if lower { INF } else { -INF };
        for (&c, &val) in index.iter().zip(value) {
            let col = c as usize;
            let (lb, ub) = (self.col_lower[col], self.col_upper[col]);
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
                // A bound can only be accepted if the implied value is strictly
                // inside the current bound: skip the exact (double-double)
                // computation where a double estimate, with a generous margin
                // for its rounding errors, shows that it is not
                if self.implied_bound_no_tighter(
                    col,
                    (val > 0.0) != lower,
                    (rhs - (act - actcontribution)) / val,
                    (rhs.abs() + act.abs() + actcontribution.abs()) / val.abs(),
                ) {
                    continue;
                }
                resact = activity - actcontribution;
            }

            let bound_val = (rhs - resact) / val;
            if (bound_val.to_f64() * K_HIGHS_TINY).abs() > self.feastol {
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

    /// The model-row part of propagate(): the bound changes of each row in
    /// `rows`, at changedbounds[2 * ARstart[row]..], their numbers (upper
    /// side, lower side) in `counts`
    pub fn propagate_model_rows(&mut self, rows: &[i32], counts: &mut [[i32; 2]], changedbounds: &mut [DomChg]) {
        let feastol = self.feastol;
        for (k, &r) in rows.iter().enumerate() {
            let i = r as usize;
            let (s, e) = (self.ar_start[i] as usize, self.ar_start[i + 1] as usize);
            let (index, value) = (&self.ar_index[s..e], &self.ar_value[s..e]);
            let (rl, ru) = (self.row_lower[i], self.row_upper[i]);
            let mut recompute = false;
            counts[k] = [0, 0];

            if ru != INF && (self.activitymaxinf[i] != 0 || self.activitymax[i] > ru + feastol) {
                self.activitymin[i].renormalize();
                counts[k][0] = self.propagate_row(
                    index,
                    value,
                    ru,
                    self.activitymin[i],
                    self.activitymininf[i],
                    false,
                    &mut changedbounds[2 * s..],
                ) as i32;
                recompute = true;
            }

            if rl != -INF && (self.activitymininf[i] != 0 || self.activitymin[i] < rl - feastol) {
                self.activitymax[i].renormalize();
                counts[k][1] = self.propagate_row(
                    index,
                    value,
                    rl,
                    self.activitymax[i],
                    self.activitymaxinf[i],
                    true,
                    &mut changedbounds[2 * s + counts[k][0] as usize..],
                ) as i32;
                recompute = true;
            }

            if recompute {
                self.capacity_threshold[i] = self.capacity_threshold_of(index, value);
            }
        }
    }

    /// The cut part of propagate() for cut pool `pool`: as
    /// propagate_model_rows, at changedbounds[rowstart(cut)..], the number in
    /// counts[k][0]
    pub fn propagate_cuts(&mut self, pool: usize, cuts: &[i32], counts: &mut [[i32; 2]], changedbounds: &mut [DomChg]) {
        // SAFETY: the only view of this pool
        let cp = unsafe { self.cutpools[pool].view() };
        for (k, &c) in cuts.iter().enumerate() {
            let i = c as usize;
            counts[k] = [0, 0];
            // first check if cut is marked as deleted
            if cp.propagatecutflags[i] & 2 != 0 {
                continue;
            }
            let [s, e] = cp.ar_range[i];
            let (s, e) = (s as usize, e as usize);
            let (index, value) = (&cp.ar_index[s..e], &cp.ar_value[s..e]);
            cp.activitycuts[i].renormalize();
            counts[k][0] = self.propagate_row(
                index,
                value,
                cp.rhs[i],
                cp.activitycuts[i],
                cp.activitycutsinf[i],
                false,
                &mut changedbounds[s..],
            ) as i32;
            cp.capacity_threshold[i] = self.capacity_threshold_of(index, value);
        }
    }

    /// CutpoolPropagation::recomputeCapacityThreshold
    pub fn cut_recompute_capacity_threshold(&mut self, pool: usize, cut: usize) {
        // SAFETY: the only view of this pool
        let cp = unsafe { self.cutpools[pool].view() };
        let [s, e] = cp.ar_range[cut];
        let (s, e) = (s as usize, e as usize);
        cp.capacity_threshold[cut] = self.capacity_threshold_of(&cp.ar_index[s..e], &cp.ar_value[s..e]);
    }
}

pub mod ffi {
    //! The `extern "C"` entry points called by HighsDomain (under HIGHS_RUST)
    use super::*;

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_update_activity(
        d: *const CDomain,
        col: i32,
        oldbound: f64,
        newbound: f64,
        upper: bool,
    ) {
        (*d).view().update_activity(col as usize, oldbound, newbound, upper);
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_compute_row_activities(d: *const CDomain) {
        (*d).view().compute_row_activities();
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_compute_activity(
        d: *const CDomain,
        index: *const i32,
        value: *const f64,
        len: i32,
        max: bool,
        ninf: *mut i32,
        activity: *mut CDouble,
    ) {
        let (n, a) = (*d).view().compute_activity(sl(index, len), sl(value, len), max);
        *ninf = n;
        *activity = a;
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_propagate_row(
        d: *const CDomain,
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
        (*d).view().propagate_row(sl(index, len), sl(value, len), rhs, *activity, ninf, lower, sl_mut(out, len))
            as i32
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_propagate_model_rows(
        d: *const CDomain,
        rows: *const i32,
        nrows: i32,
        counts: *mut [i32; 2],
        changedbounds: *mut DomChg,
        nchangedbounds: i32,
    ) {
        (*d).view().propagate_model_rows(sl(rows, nrows), sl_mut(counts, nrows), sl_mut(changedbounds, nchangedbounds));
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_propagate_cuts(
        d: *const CDomain,
        pool: i32,
        cuts: *const i32,
        ncuts: i32,
        counts: *mut [i32; 2],
        changedbounds: *mut DomChg,
        nchangedbounds: i32,
    ) {
        (*d).view().propagate_cuts(
            pool as usize,
            sl(cuts, ncuts),
            sl_mut(counts, ncuts),
            sl_mut(changedbounds, nchangedbounds),
        );
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_mark_propagate(d: *const CDomain, row: i32) {
        (*d).view().mark_propagate(row as usize);
    }

    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_domain_cut_recompute_capacity_threshold(d: *const CDomain, pool: i32, cut: i32) {
        (*d).view().cut_recompute_capacity_threshold(pool as usize, cut as usize);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "C" fn push(v: *mut c_void, x: i32) {
        (*(v as *mut Vec<i32>)).push(x);
    }

    fn cs<T>(v: &mut [T]) -> CSlice<T> {
        CSlice { p: v.as_mut_ptr(), n: v.len() as i32 }
    }

    /// x0 + 2 x1 <= 4 with integer x0, x1 in [0, 10]
    #[test]
    fn propagate_knapsack_row() {
        let (mut a_start, mut a_index, mut a_value) = (vec![0, 1, 2], vec![0, 0], vec![1.0, 2.0]);
        let (mut ar_start, mut ar_index, mut ar_value) = (vec![0, 2], vec![0, 1], vec![1.0, 2.0]);
        let (mut row_lower, mut row_upper, mut integrality) = (vec![-INF], vec![4.0], vec![1u8, 1]);
        let (mut col_lower, mut col_upper) = (vec![0.0, 0.0], vec![10.0, 10.0]);
        let (mut amin, mut amax) = (vec![CDouble::default()], vec![CDouble::default()]);
        let (mut nmin, mut nmax, mut cap, mut flags) = (vec![0], vec![0], vec![0.0], vec![0u8]);
        let mut inds: Vec<i32> = vec![];
        let (mut infeasible, mut reason, mut infeasible_pos) = (false, Reason { kind: -2, index: 0 }, 0);
        let c = CDomain {
            feastol: 1e-6,
            epsilon: 1e-9,
            a_start: cs(&mut a_start),
            a_index: cs(&mut a_index),
            a_value: cs(&mut a_value),
            ar_start: cs(&mut ar_start),
            ar_index: cs(&mut ar_index),
            ar_value: cs(&mut ar_value),
            row_lower: cs(&mut row_lower),
            row_upper: cs(&mut row_upper),
            integrality: cs(&mut integrality),
            col_lower: cs(&mut col_lower),
            col_upper: cs(&mut col_upper),
            activitymin: cs(&mut amin),
            activitymax: cs(&mut amax),
            activitymininf: cs(&mut nmin),
            activitymaxinf: cs(&mut nmax),
            capacity_threshold: cs(&mut cap),
            propagateflags: cs(&mut flags),
            propagateinds: &mut inds as *mut Vec<i32> as *mut c_void,
            col_lower_pos: cs(&mut []),
            col_upper_pos: cs(&mut []),
            prevboundval: cs(&mut []),
            infeasible: &mut infeasible,
            infeasible_reason: &mut reason,
            infeasible_pos: &mut infeasible_pos,
            domchgstack_size: 0,
            cutpools: cs(&mut []),
            conflictpools: cs(&mut []),
            push,
        };
        unsafe { c.view() }.compute_row_activities();
        assert_eq!((amin[0].to_f64(), amax[0].to_f64()), (0.0, 30.0));
        assert_eq!(inds, vec![0]);

        let mut counts = [[0; 2]];
        let mut chg = [DomChg::default(); 4];
        unsafe { c.view() }.propagate_model_rows(&[0], &mut counts, &mut chg);
        assert_eq!(counts, [[2, 0]]);
        assert_eq!(chg[0], DomChg { boundval: 4.0, column: 0, boundtype: UPPER });
        assert_eq!(chg[1], DomChg { boundval: 2.0, column: 1, boundtype: UPPER });

        // x1 >= 1 moves the min activity to 2; x1 >= 3 makes the row
        // infeasible, which reverts the activity
        col_lower[1] = 1.0;
        unsafe { c.view() }.update_activity(1, 0.0, 1.0, false);
        assert_eq!(amin[0].to_f64(), 2.0);
        col_lower[1] = 3.0;
        unsafe { c.view() }.update_activity(1, 1.0, 3.0, false);
        assert!(infeasible);
        assert_eq!((reason.kind, reason.index), (REASON_MODEL_ROW_UPPER, 0));
        assert_eq!(amin[0].to_f64(), 2.0);
    }
}
