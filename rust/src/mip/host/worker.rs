//! HighsMipWorker's shell (highs/mip/HighsMipWorker.cpp): the worker's
//! objects (LP relaxation, global domain, pools, pseudocosts, its search,
//! separation and node queue) and its Rust state (workers.rs
//! `WorkerState`); its solutions go through workers.rs.

use super::dom::DomS;
use super::lp::LpS;
use super::pools::{ConflictPoolS, CutPoolS};
use super::search::SearchS;
use super::sepa::SepaS;
use super::solver::MipSolver;
use super::tables::PscostS;
use super::Own;
use crate::lp_data::lp_run::Solution;
use crate::mip::glue::Worker;
use crate::mip::nodequeue::NodeQueue;
use crate::mip::workers::WorkerState;

/// HighsMipWorker
pub struct WorkerS {
    pub mipsolver: *mut MipSolver,
    pub lp: *mut LpS,
    pub globaldom: *mut DomS,
    pub cutpool: *mut CutPoolS,
    pub conflictpool: *mut ConflictPoolS,
    pub pseudocost: *mut PscostS,
    pub rs: Own<WorkerState>,
    pub search: Option<Own<SearchS>>,
    pub sepa: Option<Own<SepaS>>,
    pub nodequeue: Box<NodeQueue>,
    /// transformNewIntegerFeasibleSolution's solution in the original space
    pub scratch: Solution,
}

// SAFETY: a worker is used by one task at a time
unsafe impl Send for WorkerS {}

impl Drop for WorkerS {
    fn drop(&mut self) {
        // the search and separation before the state, as the C++
        self.search = None;
        self.sepa = None;
    }
}

impl WorkerS {
    /// HighsMipWorker(mipsolver, lp, domain, cutpool, conflictpool,
    /// pseudocost)
    pub fn new(
        ms: &MipSolver,
        lp: *mut LpS,
        domain: *mut DomS,
        cutpool: *mut CutPoolS,
        conflictpool: *mut ConflictPoolS,
        pseudocost: *mut PscostS,
    ) -> Box<WorkerS> {
        let sc = &ms.d().sc;
        let rs = Own(crate::mip::workers::highs_rs_worker_state_new(
            ms.opts.random_seed,
            sc.upper_bound,
            sc.upper_limit,
            sc.optimality_limit,
        ));
        let mut w = Box::new(WorkerS {
            mipsolver: ms as *const MipSolver as *mut MipSolver,
            lp,
            globaldom: domain,
            cutpool,
            conflictpool,
            pseudocost,
            rs,
            search: None,
            sepa: None,
            nodequeue: Box::new(NodeQueue::new()),
            scratch: Solution::default(),
        });
        let wp: *mut WorkerS = &mut *w;
        w.search = Some(Own::new(SearchS::new(wp, pseudocost)));
        w.sepa = Some(Own::new(SepaS::new(wp)));
        w.search.as_mut().unwrap().set_lp(lp);
        w.sepa.as_mut().unwrap().lp = lp;
        w
    }

    pub fn ms<'a>(&self) -> &'a MipSolver {
        // SAFETY: the solver outlives its workers
        unsafe { &*self.mipsolver }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn st<'a>(&self) -> &'a mut WorkerState {
        self.rs.get()
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_global_domain<'a>(&self) -> &'a mut DomS {
        // SAFETY: the worker's global domain, live
        unsafe { &mut *self.globaldom }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_lp<'a>(&self) -> &'a mut LpS {
        // SAFETY: the worker's LP, live
        unsafe { &mut *self.lp }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_cut_pool<'a>(&self) -> &'a mut CutPoolS {
        // SAFETY: the worker's pool, live
        unsafe { &mut *self.cutpool }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_conflict_pool<'a>(&self) -> &'a mut ConflictPoolS {
        // SAFETY: the worker's pool, live
        unsafe { &mut *self.conflictpool }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_pseudocost<'a>(&self) -> &'a mut PscostS {
        // SAFETY: the worker's pseudocosts, live
        unsafe { &mut *self.pseudocost }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn search<'a>(&self) -> &'a mut SearchS {
        self.search.as_ref().unwrap().get()
    }
    #[allow(clippy::mut_from_ref)]
    pub fn sepa<'a>(&self) -> &'a mut SepaS {
        self.sepa.as_ref().unwrap().get()
    }

    /// resetSearch
    pub fn reset_search(&mut self) {
        self.search = None;
        let wp: *mut WorkerS = self;
        self.search = Some(Own::new(SearchS::new(wp, self.pseudocost)));
        let lp = self.lp;
        self.search.as_mut().unwrap().set_lp(lp);
    }
    /// resetSepa
    pub fn reset_sepa(&mut self) {
        self.sepa = None;
        let wp: *mut WorkerS = self;
        self.sepa = Some(Own::new(SepaS::new(wp)));
        self.sepa.as_mut().unwrap().lp = self.lp;
    }

    /// The glue.rs handle of this worker
    pub fn handle(&self) -> Worker {
        Worker::new(self as *const WorkerS as *mut std::ffi::c_void)
    }

    /// addIncumbent (workers.rs)
    pub fn add_incumbent(&self, sol: &[f64], obj: f64, source: i32) -> bool {
        let m = self.ms().mip_data();
        m.worker_add_incumbent(&self.handle(), sol, obj, source)
    }
    /// trySolution (workers.rs)
    pub fn try_solution(&self, sol: &[f64], source: i32) -> bool {
        let m = self.ms().mip_data();
        m.worker_try_solution(&self.handle(), sol, source)
    }

    /// resetSepaStats
    pub fn reset_sepa_stats(&self) {
        let s = self.st();
        s.num_neighbourhood_queries = 0;
        s.sepa_lp_iterations = 0;
    }
    /// resetHeurStats
    pub fn reset_heur_stats(&self) {
        let h = &mut self.st().heur;
        h.total_repair_lp = 0;
        h.total_repair_lp_feasible = 0;
        h.total_repair_lp_iterations = 0;
        h.lp_iterations = 0;
        h.max_submip_level = 0;
        h.termination_status = 0;
        h.success_observations = 0.0;
        h.num_success_observations = 0;
        h.infeas_observations = 0.0;
        h.num_infeas_observations = 0;
    }
}
