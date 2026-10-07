//! HighsSeparation (highs/mip/HighsSeparation.cpp): the separation loop of
//! a node, rounds of propagation, LP resolves, implied bound and clique
//! separation, the separators, the cut pools' cuts, and the termination
//! tests. Each step on a C++ object (domains, clique table, implications,
//! reduced cost fixing, the separators, the cut set) is one callback; the
//! LP relaxation is Rust and used directly.

use super::lp_relaxation::{self as lpr, LpRelax};
use std::ffi::c_void;

/// The C++ steps, called with the HighsSeparation's context
#[repr(C)]
pub struct CSepaFns {
    /// 0 propdomain or the global domain infeasible; 1 propagate, then
    /// propdomain infeasible; 2 cleanupFixed of the clique table (master
    /// domain only), then the global domain infeasible; 3
    /// propdomain.clearChangedCols(); 4 the number of changed columns; 5
    /// setObjectiveLimit(worker upper limit); 6 / 7 the root reduced costs
    /// (master domain only; propagated if the worker's / the MIP's upper
    /// limit is finite); 8 separateImpliedBounds; 9 separateCliques; 10
    /// computeBasicDegenerateDuals (not for the global domain); 11 the
    /// separators, then the global domain infeasible; 12 the cut pools'
    /// separation, returns the number of cuts; 13 addCuts; 14 the
    /// separation's LP iterations to the statistics (`arg`); 15 the cut
    /// pool's aging
    pub op: unsafe extern "C" fn(*mut c_void, i32, i64) -> i64,
    /// the HighsDomain to propagate (for resolveLp)
    pub propdomain: *mut c_void,
    /// mipdata->rootlpsolobj, and the worker's optimality limit (read live)
    pub rootlpsolobj: f64,
    pub optimality_limit: *const f64,
    pub feastol: f64,
}

struct Sepa<'a> {
    f: &'a CSepaFns,
    ctx: *mut c_void,
    lp: &'a mut LpRelax,
}

impl Sepa<'_> {
    fn op(&self, which: i32, arg: i64) -> i64 {
        // SAFETY: the C++ steps of the separation, called with its context
        unsafe { (self.f.op)(self.ctx, which, arg) }
    }

    /// propagateAndResolve
    fn propagate_and_resolve(&mut self, status: &mut i32) -> i32 {
        let fail = |s: &Self, status: &mut i32| {
            *status = lpr::INFEASIBLE;
            s.op(3, 0);
            -1
        };
        if self.op(0, 0) != 0 || self.op(1, 0) != 0 || self.op(2, 0) != 0 {
            return fail(self, status);
        }
        let n = self.op(4, 0) as i32;
        while self.op(4, 0) != 0 {
            self.op(5, 0);
            *status = self.lp.resolve_lp(self.f.propdomain);
            if !lpr::scaled_optimal(*status) {
                return -1;
            }
            if lpr::unscaled_dual_feasible(*status) {
                self.op(6, 0);
            }
        }
        n
    }

    fn round(&mut self, status: &mut i32) -> i32 {
        self.op(8, 0);
        let mut ncuts = 0;
        let n = self.propagate_and_resolve(status);
        if n == -1 {
            return 0;
        }
        ncuts += n;
        self.op(9, 0);
        let n = self.propagate_and_resolve(status);
        if n == -1 {
            return 0;
        }
        ncuts += n;
        self.op(10, 0);
        if self.op(11, 0) != 0 {
            *status = lpr::INFEASIBLE;
            return 0;
        }
        let n = self.propagate_and_resolve(status);
        if n == -1 {
            return 0;
        }
        ncuts += n;
        let numcuts = self.op(12, 0) as i32;
        if numcuts > 0 {
            ncuts += numcuts;
            self.op(13, 0);
            *status = self.lp.resolve_lp(self.f.propdomain);
            self.lp.perform_aging(true);
            if lpr::unscaled_dual_feasible(*status) {
                self.op(7, 0);
            }
        }
        ncuts
    }

    fn separate(&mut self) {
        let mut status = self.lp.sh.status;
        if lpr::scaled_optimal(status) && !self.lp.frac.is_empty() {
            let firstobj = self.f.rootlpsolobj;
            // SAFETY: the worker's field, live during the call
            while self.lp.sh.objective < unsafe { *self.f.optimality_limit } {
                let lastobj = self.lp.sh.objective;
                let mut nlpiters = -self.lp.sh.numlpiters;
                let ncuts = self.round(&mut status);
                nlpiters += self.lp.sh.numlpiters;
                self.op(14, nlpiters);
                if ncuts == 0 || !lpr::scaled_optimal(status) || self.lp.frac.is_empty() {
                    break;
                }
                // continue only if the objective improved considerably
                let progress = lastobj - firstobj;
                let tol = if progress < self.f.feastol { self.f.feastol } else { progress };
                if self.lp.sh.objective - firstobj <= tol * 1.01 {
                    break;
                }
            }
        } else {
            self.lp.perform_aging(true);
            self.op(15, 0);
        }
    }
}

/// separationRound; `status` in and out
///
/// # Safety
/// live `fns`, `lp`, called by the C++ separation
#[no_mangle]
pub unsafe extern "C" fn highs_rs_separation_round(
    fns: *const CSepaFns,
    ctx: *mut c_void,
    lp: *mut LpRelax,
    status: *mut i32,
) -> i32 {
    Sepa { f: &*fns, ctx, lp: &mut *lp }.round(&mut *status)
}

/// separate
///
/// # Safety
/// as highs_rs_separation_round
#[no_mangle]
pub unsafe extern "C" fn highs_rs_separation_separate(fns: *const CSepaFns, ctx: *mut c_void, lp: *mut LpRelax) {
    Sepa { f: &*fns, ctx, lp: &mut *lp }.separate()
}
