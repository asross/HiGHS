//! HighsMipWorker's solutions (highs/mip/HighsMipWorker.cpp: addIncumbent,
//! trySolution and transformNewIntegerFeasibleSolution, which a worker
//! uses under the parallel lock: its bounds are updated and the solution
//! buffered for the main solver) and the synchronization of the workers'
//! solutions and global domains with the solver's (HighsMipSolver::run's
//! syncSolutions, syncGlobalDomain and resetGlobalDomain). The workers are
//! C++ objects reached through CMipFns (`worker`, `worker_view`, the
//! solution buffer and scratch solution) and op codes from 400 below.

use super::domain::{DomChg, Reason, LOWER, UPPER};
use super::glue::{fns, Dom, MipData, ScratchView, Worker, P};

/// The operations on the workers (CMipFns::op codes from 400)
pub mod op {
    /// worker i's solution buffer cleared
    pub const CLEAR_SOLUTIONS: i32 = 400;
    /// cliquetable.cleanupFixed(getDomain())
    pub const CLEANUP_FIXED: i32 = 401;
    /// worker i's global domain: empty domain change stack, the search's
    /// local domain reset, clearChangedCols
    pub const RESET_WORKER_DOMAIN_END: i32 = 402;
    /// setParallelLock(i != 0)
    pub const SET_PARALLEL_LOCK: i32 = 403;
}

/// A buffered solution of a worker (CMipFns::worker_solution)
#[repr(C)]
pub struct WorkerSol {
    pub x: *const f64,
    pub n: i32,
    pub obj: f64,
    pub source: i32,
}

/// The worker `k` of the solver
pub fn worker(m: &MipData, k: i32) -> Worker {
    // SAFETY: the C++ operation on the live solver
    Worker::new(unsafe { (fns().worker)(m.mipsolver, k) })
}

impl MipData {
    /// HighsMipWorker::transformNewIntegerFeasibleSolution: (feasible, the
    /// objective in the transformed space)
    fn worker_transform(&self, w: &Worker, sol: &[f64]) -> (bool, f64) {
        let mut v = ScratchView { col: std::ptr::null(), ncol: 0, row: std::ptr::null(), nrow: 0 };
        // SAFETY: the C++ operation on the live worker's scratch solution
        unsafe { (fns().worker_scratch)(self.mipsolver, w.p, sol.as_ptr(), sol.len() as i32, &mut v) };
        let (feasible, _, _, _, quad_obj) = self.solution_feasible_orig(v.col(), v.row());
        let sense = if self.orig_maximize { -1.0 } else { 1.0 };
        (feasible, (quad_obj * sense - self.offset).to_f64())
    }

    /// HighsMipWorker::addIncumbent
    pub fn worker_add_incumbent(&self, w: &Worker, sol: &[f64], solobj: f64, source: i32) -> bool {
        let d = &w.d;
        // SAFETY: the worker's fields, used by its own thread only
        unsafe {
            if solobj < *d.upper_bound {
                let (feasible, transformed) = self.worker_transform(w, sol);
                if feasible && transformed < *d.upper_bound {
                    *d.upper_bound = transformed;
                    let new_upper_limit = self.compute_new_upper_limit(transformed, 0.0, 0.0);
                    if new_upper_limit < *d.upper_limit {
                        *d.upper_limit = new_upper_limit;
                        *d.optimality_limit =
                            self.compute_new_upper_limit(transformed, self.opts.mip_abs_gap, self.opts.mip_rel_gap);
                    }
                }
                // infeasible ones too: they cannot be repaired locally
                (fns().worker_push_solution)(w.p, sol.as_ptr(), sol.len() as i32, solobj, source);
            }
        }
        true
    }

    /// HighsMipWorker::trySolution
    pub fn worker_try_solution(&self, w: &Worker, solution: &[f64], source: i32) -> bool {
        match self.checked_objective(solution) {
            Some(obj) => self.worker_add_incumbent(w, solution, obj, source),
            None => false,
        }
    }

    /// addIncumbent of the heuristics: through the worker under the
    /// parallel lock
    pub fn add_incumbent_any(&self, w: &Worker, sol: &[f64], obj: f64, source: i32) -> bool {
        if self.parallel_lock_active() {
            self.worker_add_incumbent(w, sol, obj, source)
        } else {
            self.add_incumbent_rs(sol, obj, source, true, false)
        }
    }

    /// trySolution of the heuristics
    pub fn try_solution_any(&self, w: &Worker, sol: &[f64], source: i32) -> bool {
        if self.parallel_lock_active() {
            self.worker_try_solution(w, sol, source)
        } else {
            self.try_solution_rs(sol, source)
        }
    }

    /// syncSolutions: the workers' buffered solutions to the solver
    pub fn sync_solutions(&self) {
        for k in 0..self.num_workers() {
            let w = worker(self, k);
            let mut j = 0;
            loop {
                let mut s = WorkerSol { x: std::ptr::null(), n: 0, obj: 0.0, source: 0 };
                // SAFETY: the C++ operation on the live worker
                if !unsafe { (fns().worker_solution)(w.p, j, &mut s) } {
                    break;
                }
                // SAFETY: the buffered vector, copied before the solver
                // changes anything
                let x = unsafe { crate::ffi::sl(s.x, s.n) }.to_vec();
                self.add_incumbent_rs(&x, s.obj, s.source, true, false);
                j += 1;
            }
            self.op(op::CLEAR_SOLUTIONS, None, k as i64, 0.0);
        }
    }

    /// syncGlobalDomain: the bound changes of the first `n` workers' global
    /// domains that are tighter, into the solver's
    pub fn sync_global_domain(&self, n: i32) {
        if self.num_workers() <= 1 {
            return;
        }
        let gd = self.domain();
        let b = gd.bnd();
        for k in 0..n {
            let w = worker(self, k);
            let wd = w.globaldom();
            for &chg in wd.stack() {
                let col = chg.column as usize;
                if (chg.boundtype == LOWER && chg.boundval > b.lo(col))
                    || (chg.boundtype == UPPER && chg.boundval < b.up(col))
                {
                    gd.change_bound(chg, Reason::UNSPECIFIED);
                }
            }
        }
    }

    /// The start of resetGlobalDomain: cleanupFixed, and if `reset_workers`
    /// the first `n` workers' global domains take the solver's changes
    /// (serially: the local domains change the main pool's propagation
    /// domains)
    pub fn reset_worker_domains(&self, reset_workers: bool, n: i32) {
        self.op(op::CLEANUP_FIXED, None, 0, 0.0);
        if !reset_workers || n <= 0 {
            return;
        }
        self.op(op::SET_PARALLEL_LOCK, None, 0, 0.0);
        let gd = self.domain();
        for k in 0..n {
            let w = worker(self, k);
            let wd: Dom = w.globaldom();
            let changes: Vec<DomChg> = gd.stack().to_vec();
            for chg in changes {
                wd.change_bound(chg, Reason::UNSPECIFIED);
            }
            self.op(op::RESET_WORKER_DOMAIN_END, None, k as i64, 0.0);
        }
        self.op(op::SET_PARALLEL_LOCK, None, 0, 0.0);
    }
}

/// HighsMipWorker::addIncumbent / trySolution (C++ callers)
///
/// # Safety
/// `f` the C++ functions, `m` filled for this call, `w` a worker, `sol`
/// valid for `n` reads
#[no_mangle]
pub unsafe extern "C" fn highs_rs_worker_solution(
    f: *const super::glue::CMipFns,
    m: *const MipData,
    w: P,
    sol: *const f64,
    n: i32,
    obj: f64,
    source: i32,
    try_solution: bool,
) -> bool {
    super::glue::set_fns(f);
    let x = crate::ffi::sl(sol, n).to_vec();
    let w = Worker::new(w);
    if try_solution {
        (*m).worker_try_solution(&w, &x, source)
    } else {
        (*m).worker_add_incumbent(&w, &x, obj, source)
    }
}

