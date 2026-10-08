//! The concurrent LNS helper (HighsMipSolverData's startConcurrentLns,
//! syncConcurrentLns, crossoverWithMain, publishRootCuts, importRootCuts
//! and stopConcurrentLns): a second MIP solver on a copy of the presolved
//! model, run in its own thread, that only does the root LP, cuts and
//! graph LNS. The two exchange incumbents, bounds and root cuts through a
//! [`Pool`], owned (with the thread) by the main solver's [`Main`] in
//! HighsMipScalars::concurrent_lns; the helper's HighsMipSolver points to
//! the pool (MipData::helper_pool), and so do the main solver's sub-MIPs,
//! which stop once the helper has closed the gap (MipData::lns_target).
//!
//! C++ keeps the object shells: the helper's options, model and basis
//! (CMipFns::helper_new) and its HighsMipSolver with its scheduler, timer
//! and profiling (CMipFns::helper_run), the LP rows and cut pool of the
//! root cut exchange. The atomics have the C++'s orderings (sequentially
//! consistent, the limit checks relaxed, the root cuts release/acquire).

use super::glue::{self, lp_status, MipData, Worker, P};
use super::root::op as rop;
use crate::lp_data::LogType;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, Ordering::*};
use std::sync::{Arc, Mutex, OnceLock};

const INF: f64 = f64::INFINITY;
/// kSolutionSourceGraphLns
const SOURCE_GRAPH_LNS: i32 = 3;

/// An f64 in an AtomicU64 (std::atomic<double>)
struct AtomicF64(AtomicU64);
impl AtomicF64 {
    fn new(x: f64) -> Self {
        Self(AtomicU64::new(x.to_bits()))
    }
    fn load(&self) -> f64 {
        f64::from_bits(self.0.load(SeqCst))
    }
    fn store(&self, x: f64) {
        self.0.store(x.to_bits(), SeqCst)
    }
}

#[derive(Default)]
struct Best {
    /// best solution offered so far
    solution: Vec<f64>,
    objective: f64,
    /// each one's best while they search independently ([0]: main solver)
    own_solution: [Vec<f64>; 2],
    own_objective: [f64; 2],
    /// the crossover's log values
    crossover_differ: i32,
    crossover_before: f64,
    crossover_after: f64,
}

/// The helper's root cuts (rows a.x <= rhs)
pub struct RootCuts {
    pub start: Vec<i32>,
    pub index: Vec<i32>,
    pub value: Vec<f64>,
    pub rhs: Vec<f64>,
    pub integral: Vec<u8>,
}

/// HighsConcurrentLns
pub struct Pool {
    best: Mutex<Best>,
    version: AtomicI64,
    stop: AtomicBool,
    /// the main solver's lower bound, and whether the helper has found a
    /// solution within the target gap of it (the main solver then stops)
    main_lower_bound: AtomicF64,
    target_reached: AtomicBool,
    /// the helper's lower bound (its root LP with its own cuts), which the
    /// main solver takes during its root node
    helper_lower_bound: AtomicF64,
    /// written once (rootCutsReady)
    root_cuts: OnceLock<RootCuts>,
    /// until the helper has crossed its incumbent with the main solver's,
    /// at the end of its own quick search, the two search independently
    independent: AtomicBool,
    main_quick_done: AtomicBool,
    /// the main solver's settings of the heuristics that the helper turns
    /// off
    pub run_rins: bool,
    pub run_rens: bool,
    pub run_root_reduced_cost: bool,
    /// the crossover, for the main solver's log: 1 running, 2 done (then
    /// the main solver logs it and sets 3)
    crossover_state: AtomicI32,
}

impl Pool {
    fn offer(&self, sol: &[f64], obj: f64) {
        let mut b = self.best.lock().unwrap();
        if obj >= b.objective {
            return;
        }
        b.objective = obj;
        b.solution.clear();
        b.solution.extend_from_slice(sol);
        self.version.fetch_add(1, SeqCst);
    }
    /// The best solution if it is newer than seen and better than obj
    fn take(&self, seen: &mut i64, obj: f64) -> Option<Vec<f64>> {
        if self.version.load(SeqCst) == *seen {
            return None;
        }
        let b = self.best.lock().unwrap();
        *seen = self.version.load(SeqCst);
        if b.objective >= obj {
            return None;
        }
        Some(b.solution.clone())
    }
    fn offer_own(&self, who: usize, sol: &[f64], obj: f64) {
        let mut b = self.best.lock().unwrap();
        if obj >= b.own_objective[who] {
            return;
        }
        b.own_objective[who] = obj;
        b.own_solution[who].clear();
        b.own_solution[who].extend_from_slice(sol);
    }
    fn own_best(&self, who: usize) -> Option<(Vec<f64>, f64)> {
        let b = self.best.lock().unwrap();
        if b.own_solution[who].is_empty() {
            return None;
        }
        Some((b.own_solution[who].clone(), b.own_objective[who]))
    }
    /// The helper is to stop (relaxed, as the C++ limit check)
    pub fn stopped(&self) -> bool {
        self.stop.load(Relaxed)
    }
    /// The helper has closed the main solver's gap (relaxed)
    pub fn target_reached(&self) -> bool {
        self.target_reached.load(Relaxed)
    }
    pub fn independent(&self) -> bool {
        self.independent.load(SeqCst)
    }
}

/// The main solver's helper: the pool and the thread
pub struct Main {
    pub pool: Arc<Pool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Drop for Main {
    /// stopConcurrentLns
    fn drop(&mut self) {
        self.pool.stop.store(true, SeqCst);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

/// A raw pointer that may cross to the helper thread
struct SendPtr(P);
// SAFETY: the helper's data is handed to the thread and used only there
unsafe impl Send for SendPtr {}

impl MipData {
    /// The main solver's own pool (null without a helper)
    pub fn own_lns(&self) -> Option<&Pool> {
        let p = self.sc().concurrent_lns;
        // SAFETY: a Main from start_concurrent_lns, freed only by
        // stop_concurrent_lns on this thread
        (!p.is_null()).then(|| unsafe { &*(*p).pool })
    }
    /// The pool shared with the main solver, in a helper
    pub fn helper_lns(&self) -> Option<&Pool> {
        // SAFETY: the main solver's pool outlives its helper's solve
        (!self.helper_pool.is_null()).then(|| unsafe { &*self.helper_pool })
    }
    /// The pool whose target ends this sub-MIP's solve
    pub fn lns_target(&self) -> Option<&Pool> {
        // SAFETY: the pool of a solver above this sub-MIP, alive during it
        (!self.lns_target.is_null()).then(|| unsafe { &*self.lns_target })
    }
    /// The pool for a sub-MIP's lns_target_reached_
    pub fn sub_mip_lns_target(&self) -> *const Pool {
        self.own_lns().map_or(self.lns_target, |p| p as *const Pool)
    }

    /// The concurrent bits of checkLimits, in its order
    pub fn concurrent_limit(&self) -> bool {
        self.helper_lns().is_some_and(|p| p.stopped())
            || self.own_lns().is_some_and(|p| p.target_reached())
            || self.lns_target().is_some_and(|p| p.target_reached())
    }

    /// addIncumbent in a helper: offer the incumbent
    pub fn concurrent_offer(&self, upper_bound: f64) {
        if let Some(p) = self.helper_lns() {
            p.offer(self.incumbent(), upper_bound);
        }
    }
    /// addIncumbent in a helper: its solution within the target gap of its
    /// main solver's bound finishes the main solve
    pub fn concurrent_target(&self, optimality_limit: f64) {
        if let Some(p) = self.helper_lns() {
            if p.main_lower_bound.load() > optimality_limit {
                p.target_reached.store(true, SeqCst);
            }
        }
    }

    /// startConcurrentLns
    pub fn start_concurrent_lns(&self) {
        if self.own_lns().is_some()
            || self.op(rop::USE_CONCURRENT_HELPER, None, 0, 0.0) == 0.0
            || self.op(rop::FIRSTROOTBASIS_VALID, None, 0, 0.0) == 0.0
        {
            return;
        }
        let time_left = self.opts.time_limit - self.timer_read();
        if time_left < 1.0 {
            return;
        }
        let pool = Arc::new(Pool {
            best: Mutex::new(Best { objective: INF, own_objective: [INF; 2], ..Default::default() }),
            version: AtomicI64::new(0),
            stop: AtomicBool::new(false),
            main_lower_bound: AtomicF64::new(-INF),
            target_reached: AtomicBool::new(false),
            helper_lower_bound: AtomicF64::new(-INF),
            root_cuts: OnceLock::new(),
            independent: AtomicBool::new(self.opts.mip_concurrent_crossover),
            main_quick_done: AtomicBool::new(false),
            run_rins: self.opts.run_rins,
            run_rens: self.opts.run_rens,
            run_root_reduced_cost: self.opts.run_root_reduced_cost,
            crossover_state: AtomicI32::new(0),
        });
        let sc = self.sc();
        if !self.incumbent().is_empty() {
            pool.offer(self.incumbent(), sc.upper_bound);
        }
        sc.concurrent_lns_seen = pool.version.load(SeqCst);
        // the helper solves a copy of the presolved model from the root
        // basis, with its own random seed, without the heuristics that
        // solve sub-MIPs
        let data = SendPtr(glue::fns_call_helper_new(self, time_left));
        let run = glue::fns().helper_run;
        let shared = pool.clone();
        let thread = std::thread::Builder::new()
            // a C++ std::thread's stack on Linux (the solve recurses into
            // sub-MIPs)
            .stack_size(8 << 20)
            .spawn(move || {
                let data = data;
                // SAFETY: the C++ helper run on its data, which it frees;
                // the pool outlives it (this thread holds a reference)
                unsafe { run(data.0, Arc::as_ptr(&shared) as *const std::ffi::c_void) }
            })
            .expect("concurrent LNS helper thread");
        sc.concurrent_lns = Box::into_raw(Box::new(Main { pool, thread: Some(thread) }));
        crate::log_user!(self.log, LogType::Info, "Concurrent LNS helper thread started\n");
    }

    /// stopConcurrentLns
    pub fn stop_concurrent_lns(&self) {
        let sc = self.sc();
        let p = std::mem::replace(&mut sc.concurrent_lns, std::ptr::null_mut());
        if !p.is_null() {
            // SAFETY: from Box::into_raw in start_concurrent_lns
            drop(unsafe { Box::from_raw(p) });
        }
    }

    /// syncConcurrentLns
    pub fn sync_concurrent_lns(&self) {
        let helper = self.helper_lns();
        let Some(pool) = helper.or(self.own_lns()) else { return };
        // while the two search independently (until the crossover), each
        // only offers its incumbents: the main solver takes the best at
        // the end
        let independent = pool.independent();
        if helper.is_some() {
            let sc = self.sc();
            pool.helper_lower_bound.store(sc.lower_bound);
            // the helper's bound with its incumbent may close the gap on its
            // own
            if sc.upper_bound < INF && cmax(sc.lower_bound, pool.main_lower_bound.load()) > sc.optimality_limit {
                pool.target_reached.store(true, SeqCst);
            }
        } else {
            // the helper's bound is valid for the same model; the tree
            // search has its own
            let helper_bound = pool.helper_lower_bound.load();
            if self.sc().num_nodes == 0 && helper_bound > self.sc().lower_bound {
                self.update_lower_bound_ex(helper_bound, true, true);
            }
            pool.main_lower_bound.store(self.sc().lower_bound);
            let state = pool.crossover_state.load(SeqCst);
            if state == 1 && !self.sc().crossover_start_logged {
                self.sc().crossover_start_logged = true;
                crate::log_user!(self.log, LogType::Info, "Crossover of the two searches' solutions started\n");
            } else if state == 2 && pool.crossover_state.compare_exchange(2, 3, SeqCst, SeqCst).is_ok() {
                let (differ, before, after) = {
                    let b = pool.best.lock().unwrap();
                    (b.crossover_differ, b.crossover_before, b.crossover_after)
                };
                crate::log_user!(
                    self.log,
                    LogType::Info,
                    "Crossover (%d integer columns differ): %.12g -> %.12g\n",
                    differ,
                    before + self.offset,
                    after + self.offset
                );
            }
        }
        let ub = self.sc().upper_bound;
        if !independent {
            if let Some(sol) = pool.take(&mut self.sc().concurrent_lns_seen, ub) {
                self.try_solution_rs(&sol, SOURCE_GRAPH_LNS);
            }
        }
        let inc = self.incumbent();
        if !inc.is_empty() {
            let ub = self.sc().upper_bound;
            pool.offer(inc, ub);
            if independent {
                pool.offer_own(helper.is_some() as usize, inc, ub);
            }
        }
    }

    /// The main solver's quick search is done: its partner solution for
    /// the helper's crossover
    pub fn concurrent_main_quick_done(&self) {
        if let Some(p) = self.own_lns() {
            if p.independent() {
                self.sync_concurrent_lns();
                p.main_quick_done.store(true, SeqCst);
            }
        }
    }

    /// The end of the solve: no longer independent, so the helper's best
    /// solution is taken even if no crossover took place; returns the
    /// helper's lower bound
    pub fn concurrent_final_sync(&self) -> f64 {
        if let Some(p) = self.own_lns() {
            p.independent.store(false, SeqCst);
        }
        self.sync_concurrent_lns();
        self.own_lns().map_or(-INF, |p| p.helper_lower_bound.load())
    }

    /// crossoverWithMain: after its quick search, the helper crosses its
    /// incumbent with the main solver's (from the main solver's own quick
    /// search, with another seed): solutions from different seeds differ
    /// much more than solutions of one search over time. Then the two
    /// exchange incumbents as usual
    pub fn crossover_with_main(&self, w: &Worker) {
        let Some(pool) = self.helper_lns() else { return };
        // the main solver's partner solution is the end of its quick search
        if !pool.independent() || !pool.main_quick_done.load(SeqCst) {
            return;
        }
        self.sync_concurrent_lns();
        let other = pool.own_best(0);
        pool.independent.store(false, SeqCst);
        let Some((other, other_objective)) = other else { return };
        if self.incumbent().is_empty() {
            return;
        }
        pool.best.lock().unwrap().crossover_before = cmin(self.sc().upper_bound, other_objective);
        pool.crossover_state.store(1, SeqCst);
        let time_limit = self.opts.time_limit;
        // SAFETY: the solver's heuristics, used as by the C++ crossover call
        let heur = unsafe { &*self.heur };
        let differ =
            heur.crossover(self, w, &other, other_objective, if time_limit < INF { 0.2 * time_limit } else { INF });
        {
            let mut b = pool.best.lock().unwrap();
            b.crossover_differ = differ;
            b.crossover_after = self.sc().upper_bound;
        }
        pool.crossover_state.store(2, SeqCst);
        self.sync_concurrent_lns();
    }

    /// publishRootCuts: the helper hands its root cuts to the main solver,
    /// which adds them to its own when it gets to its root cuts. The LP's
    /// cut rows come from C++ (setRootCuts)
    pub fn publish_root_cuts(&self) {
        let Some(pool) = self.helper_lns() else { return };
        if pool.root_cuts.get().is_some() {
            return;
        }
        self.op(rop::PUBLISH_ROOT_CUTS, None, 0, 0.0);
    }

    /// importRootCuts: returns whether the LP is infeasible
    pub fn import_root_cuts(&self, w: &Worker) -> bool {
        let Some(pool) = self.own_lns() else { return false };
        if self.sc().root_cuts_imported {
            return false;
        }
        let Some(cuts) = pool.root_cuts.get() else { return false };
        self.sc().root_cuts_imported = true;
        for i in 0..cuts.rhs.len() {
            let (s, e) = (cuts.start[i] as usize, cuts.start[i + 1] as usize);
            glue::add_root_cut(self, &cuts.index[s..e], &cuts.value[s..e], cuts.rhs[i], cuts.integral[i] != 0);
        }
        // bring the violated ones into the LP until none is
        for _ in 0..20 {
            if self.op(rop::SEPARATE_POOL_INTO_LP, None, 0, 0.0) == 0.0 {
                break;
            }
            if self.evaluate_root_lp(w) == lp_status::INFEASIBLE {
                return true;
            }
        }
        false
    }
}

/// std::min / std::max
fn cmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}
fn cmax(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// The helper's root cuts from its LP (publishRootCuts)
///
/// # Safety
/// `pool` a helper's pool; the arrays of the given lengths
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_concurrent_lns_set_root_cuts(
    pool: *const Pool,
    start: *const i32,
    num_cuts: i32,
    index: *const i32,
    value: *const f64,
    nnz: i32,
    rhs: *const f64,
    integral: *const u8,
) {
    use crate::ffi::sl;
    let _ = (*pool).root_cuts.set(RootCuts {
        start: sl(start, num_cuts + 1).to_vec(),
        index: sl(index, nnz).to_vec(),
        value: sl(value, nnz).to_vec(),
        rhs: sl(rhs, num_cuts).to_vec(),
        integral: sl(integral, num_cuts).to_vec(),
    });
}

/// ~HighsMipSolverData: stops a helper still running
///
/// # Safety
/// `main` the address of HighsMipScalars::concurrent_lns
#[no_mangle]
pub unsafe extern "C" fn highs_rs_concurrent_lns_stop(main: *mut *mut Main) {
    let p = std::mem::replace(&mut *main, std::ptr::null_mut());
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool() -> Pool {
        Pool {
            best: Mutex::new(Best { objective: INF, own_objective: [INF; 2], ..Default::default() }),
            version: AtomicI64::new(0),
            stop: AtomicBool::new(false),
            main_lower_bound: AtomicF64::new(-INF),
            target_reached: AtomicBool::new(false),
            helper_lower_bound: AtomicF64::new(-INF),
            root_cuts: OnceLock::new(),
            independent: AtomicBool::new(false),
            main_quick_done: AtomicBool::new(false),
            run_rins: true,
            run_rens: false,
            run_root_reduced_cost: true,
            crossover_state: AtomicI32::new(0),
        }
    }

    #[test]
    fn offer_take() {
        let p = pool();
        let mut seen = 0;
        assert!(p.take(&mut seen, INF).is_none());
        p.offer(&[1.0, 2.0], 5.0);
        p.offer(&[3.0], 6.0); // worse: ignored
        assert_eq!(p.version.load(SeqCst), 1);
        assert!(p.take(&mut seen, 5.0).is_none()); // not better
        assert_eq!(seen, 1);
        p.offer(&[0.0, 1.0], 4.0);
        assert_eq!(p.take(&mut seen, 5.0), Some(vec![0.0, 1.0]));
        assert!(p.take(&mut seen, 5.0).is_none()); // seen
        p.offer_own(1, &[7.0], 3.0);
        assert!(p.own_best(0).is_none());
        assert_eq!(p.own_best(1), Some((vec![7.0], 3.0)));
    }
}
