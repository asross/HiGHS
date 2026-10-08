//! HighsMipWorker's solutions (highs/mip/HighsMipWorker.cpp: addIncumbent,
//! trySolution and transformNewIntegerFeasibleSolution, which a worker
//! uses under the parallel lock: its bounds are updated and the solution
//! buffered for the main solver) and the synchronization of the workers'
//! solutions and global domains with the solver's (HighsMipSolver::run's
//! syncSolutions, syncGlobalDomain and resetGlobalDomain). A worker's
//! state (bounds, statistics, generator, solution buffer) is Rust's
//! ([`WorkerState`]); the C++ HighsMipWorker owns it and keeps its
//! pointers to the C++ objects it works on (LP relaxation, domains, pools,
//! pseudocosts, search), reached through CMipFns (`worker`,
//! `worker_view`, the scratch solution) and op codes from 400 below.

use super::domain::{DomChg, Reason, LOWER, UPPER};
use super::glue::{fns, Dom, HeurStats, MipData, ScratchView, Worker, P};
use crate::util::random::HighsRandom;

/// HighsMipWorker's state; the fields up to `heuristics_allowed` have the
/// layout of HighsMipWorker::RsState (C++ reads and writes them in place)
#[repr(C)]
pub struct WorkerState {
    pub upper_bound: f64,
    pub upper_limit: f64,
    pub optimality_limit: f64,
    pub heur: HeurStats,
    pub num_neighbourhood_queries: i64,
    pub sepa_lp_iterations: i64,
    pub randgen: HighsRandom,
    pub heuristics_allowed: bool,
    /// the buffered solutions (solution, objective, source)
    pub solutions: Vec<(Vec<f64>, f64, i32)>,
}

/// A worker's state with the solver's bounds and the seeded generator
/// (freed by highs_rs_worker_state_free)
#[no_mangle]
pub extern "C" fn highs_rs_worker_state_new(
    seed: i32,
    upper_bound: f64,
    upper_limit: f64,
    optimality_limit: f64,
) -> *mut WorkerState {
    Box::into_raw(Box::new(WorkerState {
        upper_bound,
        upper_limit,
        optimality_limit,
        heur: HeurStats::default(),
        num_neighbourhood_queries: 0,
        sepa_lp_iterations: 0,
        randgen: HighsRandom::new(seed as u32),
        heuristics_allowed: true,
        solutions: Vec::new(),
    }))
}

/// # Safety
/// `s` from highs_rs_worker_state_new, or null
#[no_mangle]
pub unsafe extern "C" fn highs_rs_worker_state_free(s: *mut WorkerState) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

/// The operations on the workers (CMipFns::op codes from 400)
pub mod op {
    /// cliquetable.cleanupFixed(getDomain())
    pub const CLEANUP_FIXED: i32 = 401;
    /// worker i's global domain: empty domain change stack, the search's
    /// local domain reset, clearChangedCols
    pub const RESET_WORKER_DOMAIN_END: i32 = 402;
    /// setParallelLock(i != 0)
    pub const SET_PARALLEL_LOCK: i32 = 403;
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
        if solobj < w.state().upper_bound {
            let (feasible, transformed) = self.worker_transform(w, sol);
            let s = w.state();
            if feasible && transformed < s.upper_bound {
                s.upper_bound = transformed;
                let new_upper_limit = self.compute_new_upper_limit(transformed, 0.0, 0.0);
                if new_upper_limit < s.upper_limit {
                    s.upper_limit = new_upper_limit;
                    s.optimality_limit =
                        self.compute_new_upper_limit(transformed, self.opts.mip_abs_gap, self.opts.mip_rel_gap);
                }
            }
            // infeasible ones too: they cannot be repaired locally
            s.solutions.push((sol.to_vec(), solobj, source));
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
            // taken first: adding an incumbent does not use the buffer
            let sols = std::mem::take(&mut w.state().solutions);
            for (x, obj, source) in sols {
                self.add_incumbent_rs(&x, obj, source, true, false);
            }
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

