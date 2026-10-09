//! HighsMipSolver and HighsMipSolverData (highs/mip/HighsMipSolver.h,
//! HighsMipSolverData.h): the solver's options, models, solution and
//! result, and its data's containers (LP relaxations, domains, pools,
//! pseudocosts, workers), tables (clique table, implications, reduced
//! cost fixing, objective function, symmetries), postsolve stack and
//! presolved model, root basis and node queue. The scalars and vectors the
//! ported code reads in place are the `MipScalars` and `MipVecs` of
//! glue.rs and mip_data.rs; [`MipSolver::mip_data`] builds the glue.rs
//! `MipData` view of them.

use super::dom::DomS;
use super::lp::LpS;
use super::pools::{ConflictPoolS, CutPoolS};
use super::tables::{ImplicsS, ObjFunc, PscostInit, PscostS, StabS};
use super::worker::WorkerS;
use super::{Own, Prof, P};
use crate::lp_data::lp::Lp;
use crate::lp_data::lp_handle::Timer;
use crate::lp_data::lp_presolve::PostsolveStack;
use crate::lp_data::lp_run::{Basis, Solution};
use crate::lp_data::opts::Opts;
use crate::lp_data::Log;
use crate::mip::clique::CliqueTable;
use crate::mip::concurrent::Pool;
use crate::mip::glue::{MipData, MipOptions, MipScalars, OrigModel, SolutionPtrs};
use crate::mip::mip_data::MipVecs;
use crate::mip::nodequeue::NodeQueue;
use crate::mip::primal::Heur;
use crate::mip::redcost::RedcostFixing;
use crate::presolve::symmetry::Symmetries;
use crate::util::cdouble::CDouble;
use std::sync::atomic::{AtomicBool, AtomicI32};
use std::sync::Arc;

pub const MODEL_STATUS_NOTSET: i32 = 0;
pub const MODEL_STATUS_INTERRUPT: i32 = 19;

/// HighsTerminator: the termination records of concurrent instances
#[derive(Clone, Copy)]
pub struct Terminator {
    pub num_instance: i32,
    pub my_instance: i32,
    pub record: *mut i32,
}

impl Terminator {
    pub fn none() -> Self {
        // kNoThreadInstance
        Terminator { num_instance: 0, my_instance: -1, record: std::ptr::null_mut() }
    }
}

/// HighsMipSolver
pub struct MipSolver {
    /// The C++ MipHost of the solve (the user callback, the improving
    /// solution file), shared by the sub-MIPs; null in a concurrent helper
    pub host: P,
    pub prof: Prof,
    /// options_mip_
    pub opts: Opts,
    /// options_mip_->log_options
    pub log: Log,
    /// model_: the original model's or the presolved model (in mipdata)
    pub model: *mut Lp,
    /// orig_model_
    pub orig_model: *mut Lp,
    /// The model passed in (the original model)
    pub input: Box<Lp>,
    pub model_name: Vec<u8>,
    pub modelstatus: i32,
    pub solution: Vec<f64>,
    pub solution_objective: f64,
    pub bound_violation: f64,
    pub integrality_violation: f64,
    pub row_violation: f64,
    pub dual_bound: f64,
    pub primal_bound: f64,
    pub gap: f64,
    pub node_count: i64,
    pub total_lp_iterations: i64,
    pub primal_dual_integral: f64,
    pub improving_file_open: bool,
    pub saved_objective_and_solution: Vec<(f64, Vec<f64>)>,
    pub submip: bool,
    pub submip_level: i32,
    pub max_submip_level: i32,
    pub rootbasis: *const Basis,
    pub concurrent_lns: *const Pool,
    pub lns_target_reached: *const Pool,
    pub pscostinit: *const PscostInit,
    pub clqtableinit: *const CliqueTable,
    pub implicinit: *const ImplicsS,
    pub mipdata: Option<Own<SolverData>>,
    /// this solver (its own address, for the views of its fields)
    this: *mut MipSolver,
    pub termination_status: i32,
    pub terminator: Terminator,
    pub timer: Timer,
}

// SAFETY: a solver is used by the threads of its tasks, as the C++
unsafe impl Send for MipSolver {}

impl MipSolver {
    /// HighsMipSolver(callback, options, lp, solution, submip,
    /// submip_level): `solution` the start solution's column and row values
    /// if valid
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        host: P,
        prof: Prof,
        opts: Opts,
        log: Log,
        lp: Lp,
        model_name: Vec<u8>,
        solution: Option<(&[f64], &[f64])>,
        submip: bool,
        submip_level: i32,
    ) -> Box<MipSolver> {
        let mut ms = Box::new(MipSolver {
            host,
            prof,
            opts,
            log,
            model: std::ptr::null_mut(),
            orig_model: std::ptr::null_mut(),
            input: Box::new(lp),
            model_name,
            modelstatus: MODEL_STATUS_NOTSET,
            solution: Vec::new(),
            solution_objective: f64::INFINITY,
            bound_violation: 0.0,
            integrality_violation: 0.0,
            row_violation: 0.0,
            dual_bound: 0.0,
            primal_bound: 0.0,
            gap: 0.0,
            node_count: 0,
            total_lp_iterations: 0,
            primal_dual_integral: 0.0,
            improving_file_open: false,
            saved_objective_and_solution: Vec::new(),
            submip,
            submip_level,
            max_submip_level: 0,
            rootbasis: std::ptr::null(),
            concurrent_lns: std::ptr::null(),
            lns_target_reached: std::ptr::null(),
            pscostinit: std::ptr::null(),
            clqtableinit: std::ptr::null(),
            implicinit: std::ptr::null(),
            mipdata: None,
            this: std::ptr::null_mut(),
            termination_status: MODEL_STATUS_NOTSET,
            terminator: Terminator::none(),
            timer: Timer::default(),
        });
        ms.this = &mut *ms;
        let input: *mut Lp = &mut *ms.input;
        ms.model = input;
        ms.orig_model = input;
        if let Some((col, row)) = solution {
            // the initial solution can be infeasible: its violations and
            // objective
            let (_, bv, rv, iv, obj) = ms.solution_feasible(col, Some(row));
            ms.bound_violation = bv;
            ms.row_violation = rv;
            ms.integrality_violation = iv;
            ms.solution_objective = obj.to_f64();
            ms.solution = col.to_vec();
        }
        ms
    }

    #[inline(always)]
    pub fn model<'a>(&self) -> &'a Lp {
        // SAFETY: the original or the presolved model, alive with the solver
        unsafe { &*self.model }
    }
    #[inline(always)]
    pub fn model_mut<'a>(&self) -> &'a mut Lp {
        // SAFETY: as model; the C++ changed the presolved model in place
        // where no slice of it was held
        unsafe { &mut *self.model }
    }
    #[inline(always)]
    pub fn orig<'a>(&self) -> &'a Lp {
        // SAFETY: as model
        unsafe { &*self.orig_model }
    }
    #[inline(always)]
    pub fn num_col(&self) -> i32 {
        self.model().num_col
    }
    #[inline(always)]
    pub fn num_row(&self) -> i32 {
        self.model().num_row
    }
    /// numNonzero (a column-wise model)
    pub fn num_nonzero(&self) -> i32 {
        let m = self.model();
        if m.a.format == crate::lp_data::matrix_format::ROWWISE {
            m.a.start.get(m.num_row as usize).copied().unwrap_or(0)
        } else {
            m.a.start.get(m.num_col as usize).copied().unwrap_or(0)
        }
    }
    #[inline(always)]
    pub fn is_col_integral(&self, col: i32) -> bool {
        self.model().integrality[col as usize] != 0
    }
    #[inline(always)]
    pub fn is_col_continuous(&self, col: i32) -> bool {
        self.model().integrality[col as usize] == 0
    }

    /// The solver's data
    #[allow(clippy::mut_from_ref)]
    #[inline(always)]
    pub fn d<'a>(&self) -> &'a mut SolverData {
        self.mipdata.as_ref().expect("mipdata").get()
    }

    /// solutionFeasible on the original model: (feasible, bound violation,
    /// row violation, integrality violation, objective)
    pub fn solution_feasible(&self, col_value: &[f64], row_value: Option<&[f64]>) -> (bool, f64, f64, f64, CDouble) {
        let lp = self.orig();
        let tol = self.opts.mip_feasibility_tolerance;
        let mut bound_violation: f64 = 0.0;
        let mut row_violation: f64 = 0.0;
        let mut integrality_violation: f64 = 0.0;
        let mut obj = CDouble::from(lp.offset);
        for i in 0..lp.num_col as usize {
            let value = col_value[i];
            obj += lp.col_cost[i] * value;
            if lp.integrality[i] == 1 {
                integrality_violation = super::max2((value - value.round()).abs(), integrality_violation);
            }
            let infeas = if value < lp.col_lower[i] - tol {
                lp.col_lower[i] - value
            } else if value > lp.col_upper[i] + tol {
                value - lp.col_upper[i]
            } else {
                continue;
            };
            bound_violation = super::max2(bound_violation, infeas);
        }
        if lp.num_row > 0 {
            let computed;
            let rows = match row_value {
                Some(r) => r,
                None => {
                    let mut r = vec![0.0; lp.num_row as usize];
                    crate::lp_data::edit::calculate_row_values_quad(&lp.a.start, &lp.a.index, &lp.a.value, col_value, &mut r);
                    computed = r;
                    &computed
                }
            };
            for i in 0..lp.num_row as usize {
                let value = rows[i];
                let infeas = if value < lp.row_lower[i] - tol {
                    lp.row_lower[i] - value
                } else if value > lp.row_upper[i] + tol {
                    value - lp.row_upper[i]
                } else {
                    continue;
                };
                row_violation = super::max2(row_violation, infeas);
            }
        }
        let feasible = bound_violation <= tol && integrality_violation <= tol && row_violation <= tol;
        (feasible, bound_violation, row_violation, integrality_violation, obj)
    }

    /// initialiseTerminator(mip_solver) of a sub-MIP
    pub fn initialise_terminator_from(&mut self, parent: &MipSolver) {
        self.terminator = Terminator::none();
        if parent.terminator.num_instance <= 0 {
            return;
        }
        self.termination_status = MODEL_STATUS_NOTSET;
        self.terminator = parent.terminator;
    }

    /// terminate()
    pub fn terminate(&self) -> bool {
        self.termination_status != MODEL_STATUS_NOTSET
    }

    /// getMaxNumWorkers
    pub fn get_max_num_workers(&self) -> i32 {
        let threads = crate::parallel::num_threads();
        if threads == 1 || self.opts.parallel.as_slice() != b"on" || self.submip {
            return 1;
        }
        (1.7 * threads as f64).ceil() as i32
    }

    /// setParallelLock
    pub fn set_parallel_lock(&self, lock: bool) {
        let d = self.d();
        if !d.has_multiple_workers() {
            return;
        }
        d.parallel_lock = lock;
        for p in &d.conflictpools {
            p.set_age_lock(lock);
        }
        d.cliquetable.allow_parallel = !lock && !self.submip;
    }

    /// The glue.rs view of the solver's data (refilled after the model
    /// changes)
    pub fn mip_data(&self) -> MipData {
        let model = self.model();
        let d = self.d();
        let o = &self.opts;
        let orig = self.orig();
        let ms = self.this;
        // SAFETY: the fields of the live solver (its own address)
        let msm = unsafe { &mut *ms };
        MipData {
            mipsolver: ms as P,
            log: if *self.log_flag() { self.log } else { Log::none() },
            num_col: model.num_col,
            num_row: model.num_row,
            colwise: model.a.format != crate::lp_data::matrix_format::ROWWISE,
            minimize: model.sense == 1,
            orig_maximize: orig.sense == -1,
            submip: self.submip,
            concurrent_helper: !self.concurrent_lns.is_null(),
            root_presolve_only: o.mip_root_presolve_only,
            offset: model.offset,
            a_start: &model.a.start,
            a_index: &model.a.index,
            a_value: &model.a.value,
            col_cost: &model.col_cost,
            col_lower: &model.col_lower,
            col_upper: &model.col_upper,
            row_lower: &model.row_lower,
            row_upper: &model.row_upper,
            integrality: &model.integrality,
            ar_start: &d.vecs.ar_start,
            ar_index: &d.vecs.ar_index,
            ar_value: &d.vecs.ar_value,
            uplocks: &d.vecs.uplocks,
            downlocks: &d.vecs.downlocks,
            integer_cols: &d.vecs.integer_cols,
            integral_cols: &d.vecs.integral_cols,
            continuous_cols: &d.vecs.continuous_cols,
            rootlpsol: &d.vecs.rootlpsol,
            firstlpsol: &d.vecs.firstlpsol,
            analytic_center: &d.vecs.analytic_center,
            incumbent: &d.vecs.incumbent,
            scalars: &mut *d.sc,
            clique: &*d.cliquetable,
            redcost: &*d.redcostfixing,
            nodequeue: &mut *d.nodequeue,
            globaldom: d.get_domain() as *const DomS as P,
            lp: d.get_lp() as *const LpS as P,
            modelstatus: &mut msm.modelstatus,
            solution: SolutionPtrs {
                objective: &mut msm.solution_objective,
                bound_violation: &mut msm.bound_violation,
                integrality_violation: &mut msm.integrality_violation,
                row_violation: &mut msm.row_violation,
                solution: &msm.solution,
            },
            orig: OrigModel {
                num_col: orig.num_col,
                num_row: orig.num_row,
                offset: orig.offset,
                col_cost: &orig.col_cost,
                col_lower: &orig.col_lower,
                col_upper: &orig.col_upper,
                row_lower: &orig.row_lower,
                row_upper: &orig.row_upper,
                integrality: &orig.integrality,
                a_start: &orig.a.start,
                a_index: &orig.a.index,
                a_value: &orig.a.value,
            },
            opts: MipOptions {
                objective_bound: o.objective_bound,
                objective_target: o.objective_target,
                mip_abs_gap: o.mip_abs_gap,
                mip_rel_gap: o.mip_rel_gap,
                mip_feasibility_tolerance: o.mip_feasibility_tolerance,
                time_limit: o.time_limit,
                mip_min_logging_interval: o.mip_min_logging_interval,
                mip_max_nodes: o.mip_max_nodes,
                mip_max_leaves: o.mip_max_leaves,
                mip_max_improving_sols: o.mip_max_improving_sols,
                output_flag: *self.log_flag(),
                timeless_log: o.timeless_log,
                run_zi_round: o.mip_heuristic_run_zi_round,
                run_shifting: o.mip_heuristic_run_shifting,
                run_graph_lns: o.mip_heuristic_run_graph_lns,
                run_root_reduced_cost: o.mip_heuristic_run_root_reduced_cost,
                run_rens: o.mip_heuristic_run_rens,
                run_rins: o.mip_heuristic_run_rins,
                mip_allow_restart: o.mip_allow_restart,
                presolve_off: o.presolve.as_slice() == b"off",
                run_feasibility_jump: o.mip_heuristic_run_feasibility_jump,
                output_flag_option: o.output_flag,
                mip_max_stall_nodes: o.mip_max_stall_nodes,
                small_matrix_value: o.small_matrix_value,
                mip_heuristic_effort: o.mip_heuristic_effort,
                mip_report_level: o.mip_report_level,
                restart_presolve_reduction_limit: o.restart_presolve_reduction_limit,
                presolve_reduction_limit: o.presolve_reduction_limit,
                mip_detect_symmetry: o.mip_detect_symmetry,
                mip_improving_solution_save: o.mip_improving_solution_save,
                mip_concurrent_crossover: o.mip_concurrent_crossover,
            },
            helper_pool: self.concurrent_lns,
            lns_target: self.lns_target_reached,
            heur: &*d.heuristics,
            vecs: &mut *d.vecs,
        }
    }

    /// *log_options.output_flag: the options' output_flag (the log options
    /// point to it)
    fn log_flag(&self) -> &bool {
        &self.opts.output_flag
    }
}

/// HighsMipSolverData
pub struct SolverData {
    pub vecs: Box<MipVecs>,
    pub mipsolver: *mut MipSolver,
    pub sc: Box<MipScalars>,
    pub lps: Vec<Own<LpS>>,
    pub cutpools: Vec<Own<CutPoolS>>,
    pub conflictpools: Vec<Own<ConflictPoolS>>,
    pub domains: Vec<Own<DomS>>,
    pub pseudocosts: Vec<Own<PscostS>>,
    pub workers: Vec<Own<WorkerS>>,
    pub parallel_lock: bool,
    pub heuristics: Box<Heur>,
    pub cliquetable: Box<CliqueTable>,
    pub implications: Box<ImplicsS>,
    pub redcostfixing: Box<RedcostFixing>,
    pub objective_function: ObjFunc,
    pub postsolve_stack: PostsolveStack,
    pub presolve_status: i32,
    pub presolved_model: Lp,
    pub analytic_center_status: AtomicI32,
    /// set when graph LNS suits the model: a computation of the analytic
    /// centre not started yet is skipped
    pub skip_analytic_center: AtomicBool,
    pub symmetries: Symmetries,
    pub global_orbits: Option<Arc<StabS>>,
    pub firstrootbasis: Basis,
    pub nodequeue: Box<NodeQueue>,
    /// transformNewIntegerFeasibleSolution's solution in the original space
    pub scratch: Solution,
    pub root: Option<Box<super::fns::RootCtx>>,
    pub restart: Option<Box<super::fns::RestartCtx>>,
    pub run: Option<Box<super::fns::RunCtx>>,
}

// SAFETY: as MipSolver
unsafe impl Send for SolverData {}

impl SolverData {
    /// HighsMipSolverData(mipsolver), as mipsolver.mipdata_ (no member's
    /// construction reads mipdata_)
    pub fn create(ms: &mut MipSolver) {
        let msp: *mut MipSolver = ms;
        let ncol = ms.num_col();
        let seed = ms.opts.random_seed;
        let d = Box::new(SolverData {
            vecs: Box::new(MipVecs::new()),
            mipsolver: msp,
            sc: Box::new(super::fns::scalars_default()),
            lps: Vec::new(),
            cutpools: Vec::new(),
            conflictpools: Vec::new(),
            domains: Vec::new(),
            pseudocosts: vec![Own::new(Box::new(PscostS::null()))],
            workers: Vec::new(),
            parallel_lock: false,
            heuristics: Box::new(Heur::new(seed)),
            cliquetable: Box::new(CliqueTable::new(ncol)),
            implications: Box::new(ImplicsS::new(ms)),
            redcostfixing: Box::new(RedcostFixing::new()),
            objective_function: ObjFunc::new(ms),
            postsolve_stack: PostsolveStack::default(),
            presolve_status: super::presolve::PS_NOT_SET,
            presolved_model: Lp::default(),
            analytic_center_status: AtomicI32::new(MODEL_STATUS_NOTSET),
            skip_analytic_center: AtomicBool::new(false),
            symmetries: Symmetries::default(),
            global_orbits: None,
            firstrootbasis: Basis::default(),
            nodequeue: Box::new(NodeQueue::new()),
            scratch: Solution::default(),
            root: None,
            restart: None,
            run: None,
        });
        ms.mipdata = Some(Own::new(d));
        let d = ms.d();
        // lps(1, HighsLpRelaxation(mipsolver)): a copy of a fresh one
        let tmp = LpS::new(ms);
        d.lps.push(Own::new(LpS::copy(&tmp)));
        drop(tmp);
        // domains(1, HighsDomain(mipsolver)): a copy of a fresh one
        let tmp = DomS::new(ms);
        d.domains.push(Own::new(DomS::copy(&tmp)));
        drop(tmp);
        let age = ms.opts.mip_pool_age_limit;
        let soft = ms.opts.mip_pool_soft_limit;
        d.conflictpools.push(Own::new(ConflictPoolS::new(5 * age, soft)));
        d.cutpools.push(Own::new(CutPoolS::new(ms.num_col(), age, soft, 0)));
        let (cp, cfp) = (d.get_cut_pool() as *mut CutPoolS, d.get_conflict_pool() as *mut ConflictPoolS);
        // SAFETY: the solver's own pools, alive with its domains
        unsafe {
            d.get_domain().add_cutpool(&mut *cp);
            d.get_domain().add_conflict_pool(&mut *cfp);
        }
        d.cliquetable.allow_parallel = !ms.submip;
    }

    pub fn ms<'a>(&self) -> &'a mut MipSolver {
        // SAFETY: the solver owning this data
        unsafe { &mut *self.mipsolver }
    }
    pub fn parallel_lock_active(&self) -> bool {
        self.parallel_lock && self.has_multiple_workers()
    }
    pub fn has_multiple_workers(&self) -> bool {
        self.workers.len() > 1
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_domain<'a>(&self) -> &'a mut DomS {
        self.domains[0].get()
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_lp<'a>(&self) -> &'a mut LpS {
        self.lps[0].get()
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_cut_pool<'a>(&self) -> &'a mut CutPoolS {
        self.cutpools[0].get()
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_conflict_pool<'a>(&self) -> &'a mut ConflictPoolS {
        self.conflictpools[0].get()
    }
    #[allow(clippy::mut_from_ref)]
    pub fn get_pseudo_cost<'a>(&self) -> &'a mut PscostS {
        self.pseudocosts[0].get()
    }
    #[allow(clippy::mut_from_ref)]
    pub fn worker<'a>(&self, k: usize) -> &'a mut WorkerS {
        self.workers[k].get()
    }
    /// The row-wise matrix's row
    pub fn get_row(&self, row: i32) -> (&[i32], &[f64]) {
        let s = self.vecs.ar_start.as_slice();
        let (b, e) = (s[row as usize] as usize, s[row as usize + 1] as usize);
        (&self.vecs.ar_index.as_slice()[b..e], &self.vecs.ar_value.as_slice()[b..e])
    }
    /// terminatorActive / terminatorTerminated / terminatorTerminate
    pub fn terminator_active(&self) -> bool {
        self.ms().terminator.num_instance > 0
    }
    pub fn terminator_terminated(&self) -> bool {
        let ms = self.ms();
        if self.terminator_active() {
            let t = ms.terminator;
            let mut status = MODEL_STATUS_NOTSET;
            for k in 0..t.num_instance as usize {
                // SAFETY: the concurrent instances' records
                let r = unsafe { *t.record.add(k) };
                if r != MODEL_STATUS_NOTSET {
                    status = r;
                    break;
                }
            }
            ms.termination_status = status;
        }
        ms.termination_status != MODEL_STATUS_NOTSET
    }
    pub fn terminator_terminate(&self) {
        let t = self.ms().terminator;
        // SAFETY: this instance's record
        unsafe { *t.record.add(t.my_instance as usize) = MODEL_STATUS_INTERRUPT };
    }
}

impl Drop for SolverData {
    /// ~HighsMipSolverData: the helper stopped, then the members in the
    /// C++ reverse order of declaration (the workers and their searches
    /// before the domains they registered with the pools, the domains
    /// before the pools)
    fn drop(&mut self) {
        // SAFETY: the address of the scalars' helper
        unsafe { crate::mip::concurrent::highs_rs_concurrent_lns_stop(&mut self.sc.concurrent_lns) };
        self.run = None;
        self.restart = None;
        self.root = None;
        self.global_orbits = None;
        self.workers.clear();
        self.pseudocosts.clear();
        self.domains.clear();
        self.conflictpools.clear();
        self.cutpools.clear();
        self.lps.clear();
    }
}
