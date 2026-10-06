//! HighsRedcostFixing (highs/mip/HighsRedcostFixing.cpp): reduced cost
//! fixing at the nodes, and the root's lurking bounds (bounds that hold
//! once the cutoff bound drops below a value). Rust-owned; the C++ class is
//! a handle. The domains are reached through [`CDom`] (the clique table's
//! domain callbacks); the dual proof and the reconvergence analysis of
//! propagateRedCost are C++ callbacks ([`CRedcost`]).
//!
//! The lurking bounds per column are std::multimaps from the required
//! cutoff bound; here BTreeMaps keyed by (bound, sequence), where the
//! sequence decreases so that an element inserted at the hint
//! lower_bound(key) comes before the equal keys, as in libc++. clang fuses
//! `1 - 10 * feastol` and `frac * redcost + lpobj`.

use super::clique::CDom;
use super::conflictpool::ConflictPool;
use super::domain::{DomChg, LOWER, REASON_UNKNOWN, UPPER};
use super::nodequeue::Ordf;
use crate::ffi::sl;
use crate::util::cdouble::CDouble;
use crate::util::fma::ClangFma;
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ops::Bound::{Excluded, Included, Unbounded};

const INF: f64 = f64::INFINITY;

type Lurk = BTreeMap<(Ordf, u64), i32>;

#[derive(Default)]
pub struct RedcostFixing {
    lurking_upper: Vec<Lurk>,
    lurking_lower: Vec<Lurk>,
    /// the sequence of the next inserted element (decreasing)
    seq: u64,
}

impl RedcostFixing {
    pub fn new() -> Self {
        RedcostFixing { seq: u64::MAX - 1, ..Default::default() }
    }

    /// getLurkingBounds
    pub fn lurking_bounds(&self, integral_cols: &[i32], lower: &[f64], upper: &[f64]) -> Vec<(f64, DomChg)> {
        let mut out = Vec::new();
        if self.lurking_lower.is_empty() {
            return out;
        }
        for &col in integral_cols {
            let c = col as usize;
            for (&(k, _), &b) in &self.lurking_lower[c] {
                if b as f64 > lower[c] {
                    out.push((k.0, DomChg { boundval: b as f64, column: col, boundtype: LOWER }));
                }
            }
            for (&(k, _), &b) in &self.lurking_upper[c] {
                if (b as f64) < upper[c] {
                    out.push((k.0, DomChg { boundval: b as f64, column: col, boundtype: UPPER }));
                }
            }
        }
        out
    }

    /// propagateRootRedcost on the global domain
    pub fn propagate_root(&mut self, dom: &CDom, integral_cols: &[i32], lower_bound: f64, upper_limit: f64) {
        if self.lurking_lower.is_empty() {
            return;
        }
        for &col in integral_cols {
            let c = col as usize;
            for m in [&mut self.lurking_lower[c], &mut self.lurking_upper[c]] {
                let keep = m.split_off(&(Ordf(lower_bound), u64::MAX));
                *m = keep;
            }
            for (&_, &b) in self.lurking_lower[c].range((Included((Ordf(upper_limit), 0)), Unbounded)) {
                if b as f64 > dom.lower(col) {
                    dom.change_bound(LOWER, col, b as f64, REASON_UNKNOWN, 0);
                    if dom.infeasible() {
                        return;
                    }
                }
            }
            for (&_, &b) in self.lurking_upper[c].range((Included((Ordf(upper_limit), 0)), Unbounded)) {
                if (b as f64) < dom.upper(col) {
                    dom.change_bound(UPPER, col, b as f64, REASON_UNKNOWN, 0);
                    if dom.infeasible() {
                        return;
                    }
                }
            }
        }
        dom.propagate();
    }

    /// The lurking bounds of a column (findLurkingBounds of addRootRedcost)
    #[allow(clippy::too_many_arguments)]
    fn find_lurking(
        seq: &mut u64,
        direction: i32,
        bound: i32,
        other_bound: i32,
        other_finite: bool,
        lp_objective: f64,
        redcost: f64,
        max_steps: i32,
        max_steps_exp: i32,
        feastol: f64,
        lower_bound: f64,
        lurk: &mut Lurk,
        other: &mut Lurk,
    ) {
        if direction as f64 * redcost == INF {
            lurk.clear();
            other.clear();
            lurk.insert((Ordf(-INF), *seq), bound);
            *seq -= 1;
            return;
        }
        let last = if !other_finite { bound + direction * max_steps } else { other_bound - direction };
        let mut step = 1;
        let range = direction * (last - bound);
        if range > max_steps {
            step = (range + max_steps - 1) >> max_steps_exp;
        }
        // 1 - 10 * feastol, fused
        let shift = direction as f64 * feastol.mul_add_c(-10.0, 1.0);
        step *= direction;
        let mut lurking = bound;
        while direction * lurking <= direction * last {
            let frac = (lurking - bound) as f64 + shift;
            let required = frac.mul_add_c(redcost, lp_objective);
            if required < lower_bound + feastol {
                lurking += step;
                continue;
            }
            // a better lurking bound stored already?
            let useful = lurk
                .range((Included((Ordf(required), 0)), Unbounded))
                .all(|(_, &b)| direction * b >= direction * (lurking + step));
            if !useful {
                lurking += step;
                continue;
            }
            let key = (Ordf(required), *seq);
            *seq -= 1;
            lurk.insert(key, lurking);
            let dominated: Vec<(Ordf, u64)> = lurk
                .range((Unbounded, Excluded(key)))
                .filter(|(_, &b)| direction * b >= direction * lurking)
                .map(|(&k, _)| k)
                .collect();
            for k in dominated {
                lurk.remove(&k);
            }
            lurking += step;
        }
    }

    /// addRootRedcost after the degenerate duals (global bounds `lower`,
    /// `upper`)
    #[allow(clippy::too_many_arguments)]
    pub fn add_root(
        &mut self,
        ncol: usize,
        integral_cols: &[i32],
        lower: &[f64],
        upper: &[f64],
        redcost: &[f64],
        lp_objective: f64,
        feastol: f64,
        lower_bound: f64,
    ) {
        self.lurking_lower.resize_with(ncol, Lurk::new);
        self.lurking_upper.resize_with(ncol, Lurk::new);
        let mut large = 0i32;
        for &col in integral_cols {
            let c = col as usize;
            if upper[c] - lower[c] >= 512.0 && redcost[c].abs() > feastol {
                large += 1;
            }
        }
        let mut max_steps_exp = 10;
        // std::frexp of an integer (large / 10): its bit length
        let q = large / 10;
        let expshift = if q == 0 { 0 } else { 32 - (q as u32).leading_zeros() as i32 };
        if expshift > 5 {
            let expshift = expshift.min(max_steps_exp);
            max_steps_exp = max_steps_exp - expshift + 5;
        }
        let max_steps = 1i32 << max_steps_exp;
        for &col in integral_cols {
            let c = col as usize;
            if redcost[c] > feastol {
                Self::find_lurking(
                    &mut self.seq,
                    1,
                    lower[c] as i32,
                    upper[c] as i32,
                    upper[c] != INF,
                    lp_objective,
                    redcost[c],
                    max_steps,
                    max_steps_exp,
                    feastol,
                    lower_bound,
                    &mut self.lurking_upper[c],
                    &mut self.lurking_lower[c],
                );
            } else if redcost[c] < -feastol {
                Self::find_lurking(
                    &mut self.seq,
                    -1,
                    upper[c] as i32,
                    lower[c] as i32,
                    lower[c] != -INF,
                    lp_objective,
                    redcost[c],
                    max_steps,
                    max_steps_exp,
                    feastol,
                    lower_bound,
                    &mut self.lurking_lower[c],
                    &mut self.lurking_upper[c],
                );
            }
        }
    }
}

/// The data and C++ calls of propagateRedCost
#[repr(C)]
pub struct CRedcost {
    pub local: *const CDom,
    pub global: *const CDom,
    pub integral_cols: *const i32,
    pub num_integral: i32,
    pub redcost: *const f64,
    pub lp_objective: f64,
    pub upper_limit: f64,
    pub feastol: f64,
    pub epsilon: f64,
    pub pool: *const ConflictPool,
    pub ctx: *mut c_void,
    /// lp.computeDualProof(globaldom, upper_limit, ..., false): the proof's
    /// arrays and rhs, false if none
    pub dual_proof: unsafe extern "C" fn(*mut c_void, *mut *const i32, *mut *const f64, *mut i32, *mut f64) -> bool,
    /// localdomain.conflictAnalyzeReconvergence(domchg, proof...)
    pub reconvergence: unsafe extern "C" fn(*mut c_void, DomChg, *const i32, *const f64, i32, f64),
}

fn is_active(d: &CDom, c: &DomChg) -> bool {
    if c.boundtype == LOWER {
        c.boundval <= d.lower(c.column)
    } else {
        c.boundval >= d.upper(c.column)
    }
}

/// propagateRedCost
///
/// # Safety
/// the pointers of `r` valid (redcost for the local domain's columns)
pub unsafe fn propagate_redcost(r: &CRedcost) {
    let local = &*r.local;
    let global = &*r.global;
    let redcost = sl(r.redcost, local.num_col);
    let gap = CDouble::from(r.upper_limit) - r.lp_objective;
    let tol = {
        let a = 10.0 * r.feastol;
        let b = r.epsilon * gap.to_f64();
        if a < b {
            b
        } else {
            a
        }
    };
    let cols = sl(r.integral_cols, r.num_integral);
    let mut changes: Vec<DomChg> = Vec::with_capacity(cols.len());
    for &col in cols {
        let c = col as usize;
        let (lb, ub) = (local.lower(col), local.upper(col));
        if ub == lb || redcost[c].abs() <= tol {
            continue;
        }
        let max_increase = redcost[c] * (ub - lb);
        if max_increase > gap.to_f64() {
            let newub = ((gap / redcost[c] + lb + r.feastol).floor()).to_f64();
            if newub >= ub {
                continue;
            }
            let d = DomChg { boundval: newub, column: col, boundtype: UPPER };
            if global.is_binary(col) {
                changes.push(d);
            } else {
                local.change_bound(UPPER, col, newub, REASON_UNKNOWN, 0);
                if local.infeasible() {
                    return;
                }
            }
        } else if max_increase < (-gap).to_f64() {
            let newlb = ((gap / redcost[c] + ub - r.feastol).ceil()).to_f64();
            if newlb <= lb {
                continue;
            }
            let d = DomChg { boundval: newlb, column: col, boundtype: LOWER };
            if global.is_binary(col) {
                changes.push(d);
            } else {
                local.change_bound(LOWER, col, newlb, REASON_UNKNOWN, 0);
                if local.infeasible() {
                    return;
                }
            }
        }
    }
    if changes.is_empty() {
        return;
    }
    let apply = |changes: &[DomChg]| {
        for d in changes {
            local.change_bound(d.boundtype, d.column, d.boundval, REASON_UNKNOWN, 0);
            if local.infeasible() {
                break;
            }
        }
        if !local.infeasible() {
            local.propagate();
        }
    };
    let (mut inds, mut vals, mut len, mut rhs) = (std::ptr::null(), std::ptr::null(), 0, 0.0);
    if changes.len() <= 100 && (r.dual_proof)(r.ctx, &mut inds, &mut vals, &mut len, &mut rhs) {
        let old = (*r.pool).num_conflicts();
        for d in &changes {
            if is_active(local, d) {
                continue;
            }
            (r.reconvergence)(r.ctx, *d, inds, vals, len, rhs);
        }
        if (*r.pool).num_conflicts() != old {
            local.propagate();
            if local.infeasible() {
                return;
            }
            changes.retain(|d| !is_active(local, d));
        }
        if !changes.is_empty() {
            apply(&changes);
        }
    } else {
        apply(&changes);
    }
}

mod ffi {
    use super::*;

    #[no_mangle]
    pub extern "C" fn highs_rs_redcost_new() -> *mut RedcostFixing {
        Box::into_raw(Box::new(RedcostFixing::new()))
    }

    /// # Safety
    /// `r` from highs_rs_redcost_new, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_redcost_free(r: *mut RedcostFixing) {
        if !r.is_null() {
            drop(Box::from_raw(r));
        }
    }

    /// getLurkingBounds: the number of bounds; with `out` non-null, writes
    /// them (call twice)
    ///
    /// # Safety
    /// live object, arrays valid
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_redcost_lurking(
        r: *const RedcostFixing,
        cols: *const i32,
        ncols: i32,
        lower: *const f64,
        upper: *const f64,
        ncol: i32,
        keys: *mut f64,
        out: *mut DomChg,
    ) -> i32 {
        let v = (*r).lurking_bounds(sl(cols, ncols), sl(lower, ncol), sl(upper, ncol));
        if !out.is_null() {
            for (k, (key, d)) in v.iter().enumerate() {
                *keys.add(k) = *key;
                *out.add(k) = *d;
            }
        }
        v.len() as i32
    }

    /// propagateRootRedcost
    ///
    /// # Safety
    /// live object, valid domain callbacks
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_redcost_propagate_root(
        r: *mut RedcostFixing,
        dom: *const CDom,
        cols: *const i32,
        ncols: i32,
        lower_bound: f64,
        upper_limit: f64,
    ) {
        (*r).propagate_root(&*dom, sl(cols, ncols), lower_bound, upper_limit);
    }

    /// addRootRedcost (after the C++ computed the degenerate duals)
    ///
    /// # Safety
    /// live object, arrays valid for `ncol` / `ncols`
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_redcost_add_root(
        r: *mut RedcostFixing,
        ncol: i32,
        cols: *const i32,
        ncols: i32,
        lower: *const f64,
        upper: *const f64,
        redcost: *const f64,
        lp_objective: f64,
        feastol: f64,
        lower_bound: f64,
    ) {
        (*r).add_root(
            ncol as usize,
            sl(cols, ncols),
            sl(lower, ncol),
            sl(upper, ncol),
            sl(redcost, ncol),
            lp_objective,
            feastol,
            lower_bound,
        );
    }

    /// propagateRedCost
    ///
    /// # Safety
    /// see propagate_redcost
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_redcost_propagate(r: *const CRedcost) {
        propagate_redcost(&*r);
    }
}
