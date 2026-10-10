//! The top level of a C++ `Highs` object on its engine ([`LpHandle`] on a
//! host): the store of what its runs read and write besides the LP data
//! (the Hessian, the semi-variable modifications, the MIP solver between
//! its steps, the saved improving solutions, the records of the presolve
//! component) and the steps of the run that an LP relaxation's handle
//! does not take (QP, MIP and semi-variable models, completing a user
//! solution, the clock, profiling and the PDLP parameters).
//!
//! A run of the Highs object (calledOptimizeModel) is run.rs on the
//! handle's own `CHighs`, every step made here or in lp_handle.rs on Rust
//! data. The C++ object's copies (the model, solution, basis, info, model
//! status, run data, presolve status and component, saved solutions) are
//! mirrors: imported before the call where C++ wrote them
//! ([`highs_rs_lph_top_import`]) and taken back after it
//! ([`highs_rs_lph_top_export`], with flags for what changed). What stays
//! C++ is reached through the host's `top_op`: the HighsProfiling
//! objects, the global scheduler's initialization, the MIP solver's
//! callback host (the user callback and the improving solution file), the
//! dev report of small matrix values and the C++ copy's names.

use super::super::drivers::MipResult;
use super::super::hessian::{self, Hessian, HessianView};
use super::super::lp::Lp;
use super::super::lp_run::CRunData;
use super::super::report::{assess_lp_primal_solution, PrimalAssessment};
use super::super::run::{Facts, Op, RunData};
use super::super::semi;
use super::super::solution::{get_kkt_failures, get_primal_dual_basis_errors, LpRef, PrimalDualErrors, SolRef};
use super::super::var_type::{INTEGER, SEMI_CONTINUOUS, SEMI_INTEGER, CONTINUOUS};
use super::*;
use crate::lp_data::ffi::RsName;
use crate::mip::host::solver::MipSolver;
use crate::mip::host::Prof;

// The host's top_op codes (HighsRunRust.cpp: rsTopOp)
/// Profiling for optimizeModel: a HighsProfiling made if there is none,
/// its pointer into p; returns whether there was one
const H_PROFILING_BEGIN: i32 = 1;
/// The end of optimizeModel's profiling (arg: there was one before)
const H_PROFILING_END: i32 = 2;
/// resetProfiling (if profiling)
const H_PROFILING_RESET: i32 = 3;
/// initializeMultiThreading -> status
const H_MULTITHREADING: i32 = 4;
/// A sub-solver clock: arg (which << 1) | start, which 0 MIP, 1 QP ASM
const H_SUBSOLVER: i32 = 5;
/// The MIP solver's callback host: made (arg 1, has semi-variables 3)
/// into p, or freed (arg 0, p the host)
const H_MIP_HOST: i32 = 6;
/// analyseVectorValues of the small matrix values in p (an RsMut)
const H_SMALL_VALUES: i32 = 7;
/// The C++ copy's column and row names into p ([RsMut<RsName>; 2],
/// valid until the next call)
const H_NAMES: i32 = 8;
/// Single-threaded profiling (a set-up HighsProfiling if there is none,
/// its pointer into p; returns whether there was one)
const H_PROFILING_SINGLE_BEGIN: i32 = 9;
/// Its end (arg: there was one before)
const H_PROFILING_SINGLE_END: i32 = 10;
/// logHeader()
const H_LOG_HEADER: i32 = 11;
/// The matrix and Hessian images of the C++ copy (write_matrix_image,
/// write_hessian_image)
const H_MATRIX_IMAGES: i32 = 12;

// What changed for the C++ copies (the flags of highs_rs_lph_top_export)
/// The LP data: the C++ copy takes them (lpFromRust)
pub const X_MODEL: u32 = 1;
/// The Hessian
pub const X_HESSIAN: u32 = 2;
/// clearIis and invalidateRanging
pub const X_CLEAR_IIS: u32 = 4;
/// clearPresolve and clearStandardFormLp (clearDerivedModelProperties)
pub const X_CLEAR_DERIVED: u32 = 8;
/// presolve_.clear()
pub const X_PRESOLVE_CLEAR: u32 = 16;
/// presolve_.init(model) and the presolve data exported
pub const X_PRESOLVE: u32 = 32;
/// The saved improving solutions
pub const X_SAVED: u32 = 64;
/// The postsolve status and the presolve's removed counts and times
pub const X_PRESOLVE_RECORDS: u32 = 128;
/// clearPresolve()
pub const X_CLEAR_PRESOLVE: u32 = 256;
/// presolved_model_ = the model (presolved_which 0) or the reduced LP (1)
pub const X_PRESOLVED_MODEL: u32 = 512;
/// clearModel()'s C++ part (the copy, its names, the multiple objectives
/// and saved solutions)
pub const X_CLEAR_MODEL: u32 = 1024;
/// The copy takes the passed model (its names and the members the
/// engine's model does not hold; "Original" origin)
pub const X_TAKE_MODEL: u32 = 2048;
/// The presolve was the MIP's: its status and (empty) log
pub const X_PRESOLVE_MIP: u32 = 4096;

/// The host functions of a Highs object's top level (HighsRunRust.cpp)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CTop {
    pub top_op: unsafe extern "C" fn(*mut c_void, i32, i64, *mut c_void) -> i64,
    /// The HighsTimer clocks of a run (run.rs Clock, action)
    pub clock: unsafe extern "C" fn(*mut c_void, i32, i32) -> f64,
}

/// HighsLpMods' semi-variable records
#[derive(Default)]
pub(crate) struct SemiMods {
    non_semi_index: Vec<i32>,
    inconsistent_index: Vec<i32>,
    inconsistent_lower: Vec<f64>,
    inconsistent_upper: Vec<f64>,
    inconsistent_type: Vec<u8>,
    relaxed_index: Vec<i32>,
    relaxed_value: Vec<f64>,
    tightened_index: Vec<i32>,
    tightened_value: Vec<f64>,
}

impl SemiMods {
    fn is_clear(&self) -> bool {
        self.non_semi_index.is_empty()
            && self.inconsistent_index.is_empty()
            && self.relaxed_index.is_empty()
            && self.tightened_index.is_empty()
    }
}

/// PresolveComponent's records that C++ reads (its info and postsolve
/// status)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct PresolveRecords {
    pub postsolve_status: i32,
    pub n_cols_removed: i32,
    pub n_rows_removed: i32,
    pub n_nnz_removed: i32,
    pub presolve_time: f64,
    pub postsolve_time: f64,
}

/// The top level's data: see the module comment
pub(crate) struct Top {
    pub host: CTop,
    /// model_.hessian_
    pub hessian: Hessian,
    /// The semi-variable modifications of a run
    semi: SemiMods,
    /// The MIP solver between MipRun and MipFinish, with the callback
    /// host it was given
    mip: Option<(Box<MipSolver>, *mut c_void)>,
    /// The LP of a MIP with semi-variables (withoutSemiVariables)
    mip_lp: Option<Lp>,
    /// saved_objective_and_solution_
    pub saved: Vec<(f64, Vec<f64>)>,
    saved_tolerance: f64,
    saved_mip_max_nodes: i32,
    saved_bounds: (Vec<f64>, Vec<f64>, Vec<u8>),
    pub records: PresolveRecords,
    /// What changed for the C++ copies (X_*)
    pub changed: u32,
    /// presolved_model_'s source (X_PRESOLVED_MODEL)
    pub presolved_which: i32,
    /// The solution passed to postsolve and crossover, the Hessian to
    /// passModel / passHessian
    user_solution: Option<super::super::lp_run::Solution>,
    user_hessian: Option<Hessian>,
}

impl Top {
    pub(crate) fn new(host: CTop) -> Top {
        let mut hessian = Hessian::default();
        hessian.clear();
        Top {
            host,
            hessian,
            semi: SemiMods::default(),
            mip: None,
            mip_lp: None,
            saved: Vec::new(),
            saved_tolerance: 0.0,
            saved_mip_max_nodes: 0,
            saved_bounds: Default::default(),
            records: PresolveRecords::default(),
            changed: 0,
            presolved_which: 0,
            user_solution: None,
            user_hessian: None,
        }
    }
}

fn rmv<T>(v: &mut Vec<T>) -> RsMut<T> {
    RsMut { ptr: v.as_mut_ptr(), len: v.len() }
}

/// The reduced LP's nonzeros (as Facts of the reduced LP)
fn reduced_num_nz(lp: &Lp) -> i32 {
    let nc = lp.num_col as usize;
    if lp.is_colwise() {
        lp.a.start.get(nc).copied().unwrap_or(0)
    } else {
        lp.a.value.len() as i32
    }
}

impl LpHandle {
    pub(crate) fn top(&mut self) -> &mut Top {
        self.top.as_mut().expect("a Highs object's engine")
    }

    /// The host's top_op
    fn host_top(&mut self, code: i32, arg: i64, p: *mut c_void) -> i64 {
        let ctx = self.host.as_ref().expect("a Highs object's engine").ctx;
        let f = self.top().host.top_op;
        // SAFETY: the Highs object's function with its context
        unsafe { f(ctx, code, arg, p) }
    }

    /// The run clock of a Highs object (its HighsTimer)
    pub(crate) fn top_clock(&mut self, which: i32, action: i32) -> f64 {
        let ctx = self.host.as_ref().expect("a Highs object's engine").ctx;
        let f = self.top().host.clock;
        // SAFETY: as host_top
        unsafe { f(ctx, which, action) }
    }

    fn hessian_view(&self) -> HessianView {
        let h = &self.top.as_ref().expect("a Highs object's engine").hessian;
        HessianView { dim: h.dim, format: h.format, start: h.start.as_ptr(), index: h.index.as_ptr(), value: h.value.as_ptr() }
    }

    /// The C++ copy's names (valid until the next call)
    fn cpp_names(&mut self) -> (&'static [RsName], &'static [RsName]) {
        let mut v = [RsMut { ptr: std::ptr::null_mut::<RsName>(), len: 0 }; 2];
        self.host_top(H_NAMES, 0, v.as_mut_ptr() as *mut c_void);
        // SAFETY: the C++ lists live until the next call
        unsafe { (v[0].get(), v[1].get()) }
    }

    /// The steps of a Highs object's run that an LP's handle does not
    /// take: `None` for the others
    pub(super) fn top_op(&mut self, op: i32, arg: i64, p: *mut c_void, msg: &[u8]) -> Option<i64> {
        macro_rules! is {
            ($o:ident) => {
                op == Op::$o as i32
            };
        }
        let _ = msg;
        if is!(Facts) && arg == 0 {
            let mut f = self.facts();
            f.is_qp = self.top().hessian.dim != 0;
            let m = &self.model;
            f.is_mip = m.integrality.iter().take(m.num_col as usize).any(|&t| t != CONTINUOUS);
            // SAFETY: the run's Facts
            unsafe { (p as *mut Facts).write(f) };
            return Some(0);
        }
        if is!(ExactResizeModel) {
            let sizes = |m: &Lp| {
                (m.col_cost.len(), m.col_lower.len(), m.row_lower.len(), m.a.index.len(), m.a.start.len(), m.integrality.len())
            };
            let before = sizes(&self.model);
            self.exact_resize_model();
            if sizes(&self.model) != before {
                self.top().changed |= X_MODEL;
            }
            let h = &mut self.top().hessian;
            let before = (h.start.len(), h.index.len(), h.value.len());
            h.exact_resize();
            if (h.start.len(), h.index.len(), h.value.len()) != before {
                self.top().changed |= X_HESSIAN;
            }
            return Some(0);
        }
        if is!(EnsureColwise) {
            if !self.model.is_colwise() {
                self.model.a.ensure_colwise();
                self.top().changed |= X_MODEL;
            }
            return Some(0);
        }
        if is!(LpView) && arg == 2 {
            self.top().changed |= X_MODEL;
            return Some(0);
        }
        if is!(ClearSolver) || is!(ClearSolver2) {
            return Some(self.clear_solver() as i64);
        }
        if is!(CompleteSolution) {
            return Some(self.with_run(|r| r.complete_solution_from_discrete_assignment()) as i64);
        }
        if is!(AssessSemiVariables) {
            let mut made = false;
            let s = self.assess_semi_variables(&mut made);
            // SAFETY: the run's flag
            unsafe { *(p as *mut bool) = made };
            return Some(s as i64);
        }
        if is!(RelaxSemiVariables) {
            return Some(self.relax_semi_variables() as i64);
        }
        if is!(OkHessianDiagonal) {
            let log = self.log();
            let sense = self.model.sense;
            let h = &self.top().hessian;
            return Some(hessian::ok_hessian_diagonal(&log, h.dim, &h.start, &h.value, sense) as i64);
        }
        if is!(CallSolveQp) {
            return Some(self.call_solve_qp() as i64);
        }
        if is!(CallSolveMip) {
            return Some(self.with_run(|r| r.call_solve_mip()) as i64);
        }
        if is!(HandleInfCost) {
            // The fixings (and the flag) are the model's until undone
            if self.model.has_infinite_cost {
                self.top().changed |= X_MODEL;
            }
            return Some(self.handle_inf_cost() as i64);
        }
        if is!(UndoMods) {
            if !self.inf_cost.index.is_empty() {
                self.top().changed |= X_MODEL;
            }
            let s = self.undo_mods(status_of(arg));
            self.unapply_semi_mods();
            return Some(s as i64);
        }
        // callSolveMip
        if is!(MipRun) {
            let r = self.mip_run();
            // SAFETY: the run's MipResult
            unsafe { (p as *mut MipResult).write(r) };
            return Some(0);
        }
        if is!(MipTakeSolution) {
            self.mip_take_solution();
            return Some(0);
        }
        if is!(ActiveModifiedUpperBounds) {
            let log = self.log();
            let pft = self.opts.primal_feasibility_tolerance;
            let t = &self.top.as_ref().expect("top").semi.tightened_index;
            let r = semi::active_modified_upper_bounds(
                &log,
                t,
                &self.model.col_upper,
                &self.lps.run.solution.col_value,
                pft,
            );
            return Some(r as i64);
        }
        if is!(SwapPrimalTolerance) {
            if arg == 0 {
                self.top().saved_tolerance = self.opts.primal_feasibility_tolerance;
                // SAFETY: the run's f64
                self.opts.primal_feasibility_tolerance = unsafe { *(p as *const f64) };
            } else {
                self.opts.primal_feasibility_tolerance = self.top().saved_tolerance;
            }
            return Some(0);
        }
        if is!(KktFailures) {
            self.kkt_failures();
            return Some(0);
        }
        if is!(MipFinish) {
            if let Some((ms, host)) = self.top().mip.take() {
                drop(ms);
                self.host_top(H_MIP_HOST, 0, host);
            }
            self.top().mip_lp = None;
            return Some(0);
        }
        // completeSolutionFromDiscreteAssignment
        if is!(SolutionHasUndefined) {
            let s = &self.lps.run.solution;
            let undefined = |v: &[f64]| v.iter().any(|&x| x == INF);
            return Some((undefined(&s.col_value) || undefined(&s.row_value)) as i64);
        }
        if is!(SolutionFeasible) {
            return Some(self.solution_feasible() as i64);
        }
        if is!(SaveColBounds) {
            let m = &mut self.model.g;
            let t = self.top.as_mut().expect("top");
            if arg == 0 {
                t.saved_bounds = (m.col_lower.clone(), m.col_upper.clone(), m.integrality.clone());
            } else {
                let (l, u, i) = std::mem::take(&mut t.saved_bounds);
                m.col_lower = l;
                m.col_upper = u;
                m.integrality = i;
                t.changed |= X_MODEL;
            }
            return Some(0);
        }
        if is!(ClearIntegrality) {
            self.model.g.integrality.clear();
            self.top().changed |= X_MODEL;
            return Some(0);
        }
        if is!(SolutionClear) {
            // HighsSolution::clear
            let s = &mut self.lps.run.solution;
            s.value_valid = false;
            s.dual_valid = false;
            s.col_value.clear();
            s.row_value.clear();
            s.col_dual.clear();
            s.row_dual.clear();
            return Some(0);
        }
        if is!(SwapMipMaxNodes) {
            if arg == 0 {
                self.top().saved_mip_max_nodes = self.opts.mip_max_nodes;
                self.opts.mip_max_nodes = self.opts.mip_max_start_nodes;
            } else {
                self.opts.mip_max_nodes = self.top().saved_mip_max_nodes;
            }
            return Some(0);
        }
        if is!(OptimizeModel) {
            let s = self.optimize_model_steps(true);
            return Some(s as i64);
        }
        if is!(SolutionView) {
            let s = &mut self.lps.run.solution;
            let v = super::super::drivers::SolutionView {
                value_valid: s.value_valid,
                dual_valid: s.dual_valid,
                col_value: rmv(&mut s.col_value),
                col_dual: rmv(&mut s.col_dual),
                row_value: rmv(&mut s.row_value),
                row_dual: rmv(&mut s.row_dual),
            };
            // SAFETY: the run's SolutionView
            unsafe { (p as *mut super::super::drivers::SolutionView).write(v) };
            return Some(0);
        }
        // The LP run
        if is!(PdlpTemplate) {
            let t = self.pdlp_template();
            // SAFETY: the run's PdlpTemplate
            unsafe { (p as *mut super::super::lp_run::PdlpTemplate).write(t) };
            return Some(0);
        }
        if is!(AssessSmallValues) {
            self.host_top(H_SMALL_VALUES, 0, p);
            return Some(0);
        }
        if is!(InitializeMultiThreading) {
            return Some(self.host_top(H_MULTITHREADING, 0, std::ptr::null_mut()));
        }
        // The presolve component's records
        if is!(PresolveClear) {
            self.top().changed |= X_PRESOLVE_CLEAR;
            self.top().records = PresolveRecords::default();
            return Some(0);
        }
        if is!(PresolveInit) {
            self.top().changed |= X_PRESOLVE;
            return Some(0);
        }
        if is!(SetPostsolveStatus) {
            let t = self.top();
            t.records.postsolve_status = arg as i32;
            t.changed |= X_PRESOLVE_RECORDS;
            return Some(0);
        }
        if is!(PresolveTime) {
            let t = self.top();
            // SAFETY: the run's f64
            let v = unsafe { *(p as *const f64) };
            if arg != 0 {
                t.records.postsolve_time = v;
            } else {
                t.records.presolve_time = v;
            }
            t.changed |= X_PRESOLVE_RECORDS;
            return Some(0);
        }
        if is!(PresolveRemoved) {
            let t = self.top();
            // SAFETY: the run's [i32; 3]
            let r = unsafe { *(p as *const [i32; 3]) };
            t.records.n_cols_removed = r[0];
            t.records.n_rows_removed = r[1];
            t.records.n_nnz_removed = r[2];
            t.changed |= X_PRESOLVE_RECORDS;
            return Some(0);
        }
        if is!(MipPresolve) {
            return Some(self.mip_presolve() as i64);
        }
        // presolve
        if is!(NeedsMods) {
            let m = &self.model;
            let semi = m.integrality.iter().any(|&t| t == SEMI_CONTINUOUS || t == SEMI_INTEGER);
            return Some((m.has_infinite_cost || semi) as i64);
        }
        if is!(ReportModelStats) {
            self.report_model_stats();
            return Some(0);
        }
        if is!(ClearPresolve) {
            self.presolve_status = super::super::run::PS_NOT_PRESOLVED;
            self.lps.run.presolve = Default::default();
            self.top().changed |= X_CLEAR_PRESOLVE;
            return Some(0);
        }
        if is!(PresolveProfiled) {
            let mut profiling: *mut c_void = std::ptr::null_mut();
            let already = self.host_top(H_PROFILING_SINGLE_BEGIN, 0, &mut profiling as *mut _ as *mut c_void) != 0;
            self.profiling = profiling;
            let force_lp = self.opts.solve_relaxation;
            let status = self.in_lp_mode(|r| r.run_presolve(force_lp, true) as i64);
            self.host_top(H_PROFILING_SINGLE_END, already as i64, std::ptr::null_mut());
            if !already {
                self.profiling = std::ptr::null_mut();
            }
            return Some(status);
        }
        if is!(ReportPresolveReductions) {
            let log = self.log();
            let on = self.opts.output_flag;
            let f = self.facts();
            let r = &self.lps.run.presolve.reduced;
            let reduced = [r.num_col, r.num_row, reduced_num_nz(r)];
            super::super::run::report_presolve_reductions(
                &log,
                on,
                self.presolve_status,
                [f.num_col, f.num_row, f.num_nz],
                reduced,
            );
            return Some(0);
        }
        if is!(PresolvedModel) {
            let t = self.top();
            t.presolved_which = arg as i32;
            t.changed |= X_PRESOLVED_MODEL;
            return Some(0);
        }
        // crossover
        if is!(Crossover) {
            if arg == 0 {
                let s = self.top().user_solution.take().expect("crossover's solution");
                self.lps.run.solution = s;
                let status = self.crossover();
                if status == 2 {
                    self.task_interrupted = true;
                    return Some(i64::MIN);
                }
                return Some(status as i64);
            }
            self.lp_kkt_failures(false);
            return Some(0);
        }
        // postsolve
        if is!(PostsolveArgs) {
            let s = self.top().user_solution.as_ref().expect("postsolve's solution");
            let (cv, cd, rd, dv) = (s.col_value.len(), s.col_dual.len(), s.row_dual.len(), s.dual_valid);
            let b = self.user_basis.as_ref().expect("postsolve's basis");
            let a = super::super::drivers::PostsolveArgs {
                col_value_size: cv as i64,
                col_dual_size: cd as i64,
                row_dual_size: rd as i64,
                dual_valid: dv,
                basis_col_size: b.b.col_status.len() as i64,
                basis_row_size: b.b.row_status.len() as i64,
                basis_valid: b.b.valid,
            };
            // SAFETY: the run's PostsolveArgs
            unsafe { (p as *mut super::super::drivers::PostsolveArgs).write(a) };
            return Some(0);
        }
        if is!(PostsolveBasisConsistent) {
            let r = &self.lps.run.presolve.reduced;
            let (nc, nr) = (r.num_col as usize, r.num_row as usize);
            let b = &self.user_basis.as_ref().expect("postsolve's basis").b;
            let ok = b.col_status.len() == nc && b.row_status.len() == nr && basis_consistent(&b.col_status, &b.row_status);
            return Some(ok as i64);
        }
        if is!(PostsolveSetSolution) {
            let nr = self.lps.run.presolve.reduced.num_row as usize;
            let user_basis = self.user_basis.clone();
            let us = self.top().user_solution.clone();
            let d = &mut self.lps.run.presolve;
            match arg {
                0 => {
                    d.recovered_solution = us.expect("postsolve's solution");
                    d.recovered_solution.row_value.clear();
                    d.recovered_solution.row_value.resize(nr, 0.0);
                    d.recovered_solution.value_valid = true;
                }
                1 => {
                    let s = &mut d.recovered_solution;
                    s.dual_valid = false;
                    s.col_dual.clear();
                    s.row_dual.clear();
                    d.recovered_basis.b.valid = false;
                }
                _ => {
                    d.recovered_solution.dual_valid = (arg - 2) & 1 != 0;
                    d.recovered_basis = user_basis.expect("postsolve's basis");
                    d.recovered_basis.b.valid = (arg - 2) & 2 != 0;
                }
            }
            return Some(0);
        }
        if is!(PostsolveKkt) {
            self.lp_kkt_postsolve(arg != 0);
            return Some(0);
        }
        if is!(PostsolveTakeRecovered) {
            let (nc, nr) = (self.model.num_col as usize, self.model.num_row as usize);
            let r = &mut self.lps.run;
            r.solution = r.presolve.recovered_solution.clone();
            debug_assert!(r.solution.value_valid);
            if !r.solution.dual_valid {
                r.solution.col_dual = vec![0.0; nc];
                r.solution.row_dual = vec![0.0; nr];
            }
            r.basis = r.presolve.recovered_basis.clone();
            r.basis.origin.push_str(": after postsolve");
            return Some(0);
        }
        if is!(OptionsPostsolveCleanup) {
            self.opts.simplex_strategy = STRATEGY_CHOOSE;
            self.opts.simplex_min_concurrency = 1;
            self.opts.simplex_max_concurrency = 1;
            return Some(0);
        }
        // passModel and passHessian
        if is!(LogHeader) {
            self.host_top(H_LOG_HEADER, 0, std::ptr::null_mut());
            return Some(0);
        }
        if is!(ClearModel) {
            self.clear_model();
            let t = self.top();
            t.saved.clear();
            t.hessian.clear();
            t.semi = SemiMods::default();
            t.changed |= X_CLEAR_MODEL | X_MODEL | X_HESSIAN;
            return Some(0);
        }
        if is!(TakeModel) {
            let lp = self.user_model.take().expect("passModel's model");
            self.model = lp;
            let t = self.top();
            if let Some(h) = t.user_hessian.take() {
                t.hessian = h;
            }
            t.changed |= X_TAKE_MODEL | X_MODEL | X_HESSIAN;
            return Some(0);
        }
        if is!(EmptyMatrix) || is!(PrepareModelLp) || is!(AssessLp) {
            self.top().changed |= X_MODEL;
            return None;
        }
        if is!(FormatOk) && arg != 0 {
            let h = &self.top().hessian;
            // HighsHessian::formatOk
            let ok = h.format == hessian::TRIANGULAR || h.format == hessian::SQUARE;
            return Some(ok as i64);
        }
        if is!(MatrixImages) {
            if self.opts.write_matrix_image || self.opts.write_hessian_image {
                self.host_top(H_MATRIX_IMAGES, 0, std::ptr::null_mut());
            }
            return Some(0);
        }
        if is!(HessianDims) {
            let h = &self.top().hessian;
            let d = [h.dim, if h.dim != 0 { h.num_nz() } else { 0 }];
            // SAFETY: the run's [i32; 2]
            unsafe { (p as *mut [i32; 2]).write(d) };
            return Some(0);
        }
        if is!(AssessHessian) {
            let log = self.log();
            let (small, large) = (self.opts.small_matrix_value, self.opts.large_matrix_value);
            let t = self.top();
            t.changed |= X_HESSIAN;
            return Some(hessian::assess_hessian(&log, &mut t.hessian, small, large) as i64);
        }
        if is!(HessianClear) {
            let t = self.top();
            t.hessian.clear();
            t.changed |= X_HESSIAN;
            return Some(0);
        }
        if is!(CompleteHessian) {
            let n = self.model.num_col;
            let t = self.top();
            hessian::complete_hessian(n, &mut t.hessian);
            t.changed |= X_HESSIAN;
            return Some(0);
        }
        if is!(TakeHessian) {
            let t = self.top();
            t.hessian = t.user_hessian.take().expect("passHessian's Hessian");
            t.changed |= X_HESSIAN;
            return Some(0);
        }
        None
    }

    /// Runs `f` on a run of this handle in the LP mode of lp_run.rs (the
    /// presolve and postsolve steps on the engine's presolve data)
    fn in_lp_mode<R>(&mut self, f: impl FnOnce(&Run) -> R) -> R {
        let c = self.chighs();
        let mode = super::super::lp_run::LpMode::new(&c, self as *mut LpHandle);
        let c2 = mode.view();
        // SAFETY: c2's pointers are into this handle, which outlives the run
        let run2 = unsafe { Run::new(&c2) };
        let r = f(&run2);
        if run2.ab() || mode.aborted.get() {
            self.task_interrupted = true;
        }
        r
    }

    /// reportModelStats
    fn report_model_stats(&mut self) {
        if !self.opts.output_flag {
            return;
        }
        let log = self.log();
        let dev = self.opts.log_dev_level != 0;
        let m = &self.model;
        let h = &self.top.as_ref().expect("top").hessian;
        let name = String::from_utf8_lossy(&m.model_name).into_owned();
        let f = self.facts_of_model();
        super::super::model::report_model_stats(
            &log,
            dev,
            &name,
            m.num_col,
            m.num_row,
            f,
            h.dim,
            if h.dim > 0 { h.num_nz() } else { 0 },
            &m.integrality,
            &m.col_lower,
            &m.col_upper,
        );
    }

    /// The model's number of nonzeros (HighsSparseMatrix::numNz)
    fn facts_of_model(&self) -> i32 {
        self.model.a.num_nz()
    }

    /// runPresolve's MIP presolve on the model: the presolved model and
    /// stack are the engine's presolve data
    fn mip_presolve(&mut self) -> i32 {
        let model_changed = self.top().changed & X_MODEL != 0;
        let mut host: *mut c_void = std::ptr::null_mut();
        let arg = 1 | if model_changed { 4 } else { 0 };
        self.host_top(H_MIP_HOST, arg, &mut host as *mut _ as *mut c_void);
        let s = &self.lps.run.solution;
        let start = if s.value_valid { Some((&s.col_value[..], &s.row_value[..])) } else { None };
        let lp = self.model.clone();
        let name = lp.model_name.clone();
        let limit = self.opts.presolve_reduction_limit;
        let mut ms = MipSolver::new(host, Prof { p: self.profiling }, self.opts.clone(), self.log(), lp, name, start, false, 0);
        ms.timer.start(0);
        crate::mip::host::solver::SolverData::create(&mut ms);
        crate::mip::glue::set_fns(&crate::mip::host::fns::MIP_FNS);
        let m = ms.mip_data();
        m.init_rs();
        m.run_mip_presolve(limit);
        let d = ms.d();
        let status = d.presolve_status;
        let pd = &mut self.lps.run.presolve;
        pd.active = true;
        pd.reduced = d.presolved_model.clone();
        pd.stack = d.postsolve_stack.clone();
        pd.status = status;
        pd.log.clear();
        pd.prepared = false;
        drop(ms);
        self.host_top(H_MIP_HOST, 0, host);
        self.top().changed |= X_PRESOLVE | X_PRESOLVE_MIP;
        status
    }

    /// getLpKktFailures of the model (with the objective first if
    /// `objective`): Crossover arg 1
    fn lp_kkt_failures(&mut self, residuals: bool) {
        let o = self.opts.kkt(self.log());
        let r = &mut self.lps.run;
        let m = &mut self.model;
        let v = m.view();
        let sv = r.solution.view();
        // SAFETY: the views live for the call
        let (lp, sol) = unsafe { (LpRef::new(&v), SolRef::new(&sv)) };
        r.info.objective_function_value = lp.objective_value(&r.solution.col_value);
        get_kkt_failures(&o, false, &lp, lp.col_cost, &sol, &mut r.info, residuals);
        let b = &r.basis.b;
        // SAFETY: plain data, filled by the call
        let mut e: PrimalDualErrors = unsafe { std::mem::zeroed() };
        get_primal_dual_basis_errors(&o, &lp, &sol, b.valid, &b.col_status, &b.row_status, &mut e);
    }

    /// PostsolveKkt: getKktFailures of the model LP with residuals, after
    /// objective = computeObjectiveValue if `objective`
    fn lp_kkt_postsolve(&mut self, objective: bool) {
        let o = self.opts.kkt(self.log());
        let r = &mut self.lps.run;
        let m = &mut self.model;
        let v = m.view();
        let sv = r.solution.view();
        // SAFETY: the views live for the call
        let (lp, sol) = unsafe { (LpRef::new(&v), SolRef::new(&sv)) };
        if objective {
            r.info.objective_function_value = super::super::solution::compute_objective_value(&lp, &sol);
        }
        get_kkt_failures(&o, false, &lp, lp.col_cost, &sol, &mut r.info, true);
    }

    /// optimizeModel: the scheduler, profiling and calledOptimizeModel
    /// (`reset`: then resetProfiling, as the OptimizeModel step)
    pub(crate) fn optimize_model_steps(&mut self, reset: bool) -> Status {
        let s = status_of(self.host_top(H_MULTITHREADING, 0, std::ptr::null_mut()));
        if s != Status::Ok {
            return s;
        }
        let mut profiling: *mut c_void = std::ptr::null_mut();
        let already = self.host_top(H_PROFILING_BEGIN, 0, &mut profiling as *mut _ as *mut c_void) != 0;
        self.profiling = profiling;
        let status = self.with_run(|r| r.called_optimize_model());
        self.host_top(H_PROFILING_END, already as i64, std::ptr::null_mut());
        if already {
            if reset {
                self.host_top(H_PROFILING_RESET, 0, std::ptr::null_mut());
            }
        } else {
            self.profiling = std::ptr::null_mut();
        }
        status
    }

    // ---- Semi-variables (HighsLpMods)

    /// assessSemiVariables, its records appended
    fn assess_semi_variables(&mut self, made: &mut bool) -> Status {
        *made = false;
        if self.model.integrality.is_empty() {
            return Status::Ok;
        }
        let n = self.model.num_col as usize;
        let (mut ii, mut ns, mut ti) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        let (mut il, mut iu, mut tv) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut it = vec![0u8; n];
        let mut m = semi::CSemiMods {
            inconsistent_index: rmv(&mut ii),
            inconsistent_lower: rmv(&mut il),
            inconsistent_upper: rmv(&mut iu),
            inconsistent_type: rmv(&mut it),
            non_semi_index: rmv(&mut ns),
            tightened_index: rmv(&mut ti),
            tightened_value: rmv(&mut tv),
            num_inconsistent: 0,
            num_non_semi: 0,
            num_tightened: 0,
            made_mods: false,
        };
        let log = self.log();
        let lp = &mut self.model.g;
        // SAFETY: the buffers hold num_col entries
        let s = unsafe { semi::assess_semi_variables(&log, &mut lp.col_lower, &mut lp.col_upper, &mut lp.integrality, &mut m) };
        fn app<T: Copy>(to: &mut Vec<T>, from: &[T], n: i32) {
            if n < 0 {
                to.clear();
            } else {
                to.extend_from_slice(&from[..n as usize]);
            }
        }
        let t = self.top();
        let r = &mut t.semi;
        app(&mut r.non_semi_index, &ns, m.num_non_semi);
        app(&mut r.inconsistent_index, &ii, m.num_inconsistent);
        app(&mut r.inconsistent_lower, &il, m.num_inconsistent);
        app(&mut r.inconsistent_upper, &iu, m.num_inconsistent);
        app(&mut r.inconsistent_type, &it, m.num_inconsistent);
        app(&mut r.tightened_index, &ti, m.num_tightened);
        app(&mut r.tightened_value, &tv, m.num_tightened);
        *made = m.made_mods;
        if m.made_mods {
            t.changed |= X_MODEL;
        }
        s
    }

    /// relaxSemiVariables -> made modifications
    fn relax_semi_variables(&mut self) -> bool {
        if self.model.integrality.is_empty() {
            return false;
        }
        let n = self.model.num_col as usize;
        let (mut index, mut value) = (vec![0i32; n], vec![0.0; n]);
        let lp = &mut self.model.g;
        let k = semi::relax_semi_variables(&mut lp.col_lower, &lp.integrality, &mut index, &mut value);
        let t = self.top();
        t.semi.relaxed_index.extend_from_slice(&index[..k]);
        t.semi.relaxed_value.extend_from_slice(&value[..k]);
        if k > 0 {
            t.changed |= X_MODEL;
        }
        !t.semi.relaxed_index.is_empty()
    }

    /// HighsLp::unapplyMods' semi-variable part (the infinite costs are
    /// undo_mods')
    fn unapply_semi_mods(&mut self) {
        let t = self.top.as_mut().expect("top");
        if t.semi.is_clear() {
            return;
        }
        let r = std::mem::take(&mut t.semi);
        t.changed |= X_MODEL;
        let m = semi::CLpMods {
            non_semi_index: RsMut { ptr: r.non_semi_index.as_ptr() as *mut i32, len: r.non_semi_index.len() },
            inconsistent_index: RsMut { ptr: r.inconsistent_index.as_ptr() as *mut i32, len: r.inconsistent_index.len() },
            inconsistent_lower: RsMut { ptr: r.inconsistent_lower.as_ptr() as *mut f64, len: r.inconsistent_lower.len() },
            inconsistent_upper: RsMut { ptr: r.inconsistent_upper.as_ptr() as *mut f64, len: r.inconsistent_upper.len() },
            inconsistent_type: RsMut { ptr: r.inconsistent_type.as_ptr() as *mut u8, len: r.inconsistent_type.len() },
            relaxed_index: RsMut { ptr: r.relaxed_index.as_ptr() as *mut i32, len: r.relaxed_index.len() },
            relaxed_value: RsMut { ptr: r.relaxed_value.as_ptr() as *mut f64, len: r.relaxed_value.len() },
            tightened_index: RsMut { ptr: r.tightened_index.as_ptr() as *mut i32, len: r.tightened_index.len() },
            tightened_value: RsMut { ptr: r.tightened_value.as_ptr() as *mut f64, len: r.tightened_value.len() },
        };
        let lp = &mut self.model.g;
        // SAFETY: the records live for the call
        unsafe { semi::unapply_mods(&m, &mut lp.col_lower, &mut lp.col_upper, &mut lp.integrality) };
    }

    // ---- The KKT failures of the model (getKktFailures(options, model, ...))

    fn kkt_failures(&mut self) {
        let o = self.opts.kkt(self.log());
        let hv = self.hessian_view();
        let is_qp = hv.dim != 0;
        let r = &mut self.lps.run;
        let m = &mut self.model;
        let n = m.num_col as usize;
        // HighsModel::objectiveGradient
        let mut gradient = vec![0.0; n];
        if is_qp {
            hessian::product(&hv, &r.solution.col_value, &mut gradient);
        }
        for j in 0..n {
            gradient[j] += m.col_cost[j];
        }
        let v = m.view();
        let sv = r.solution.view();
        // SAFETY: the views live for the call
        let (lp, sol) = unsafe { (LpRef::new(&v), SolRef::new(&sv)) };
        get_kkt_failures(&o, is_qp, &lp, &gradient, &sol, &mut r.info, false);
        let b = &r.basis.b;
        let mut e: PrimalDualErrors = unsafe { std::mem::zeroed() };
        get_primal_dual_basis_errors(&o, &lp, &sol, b.valid, &b.col_status, &b.row_status, &mut e);
    }

    // ---- callSolveQp

    fn call_solve_qp(&mut self) -> Status {
        let log = self.log();
        let o = &self.opts;
        let (qp_iteration_limit, qp_nullspace_limit, simplex_primal_edge_weight_strategy) =
            (o.qp_iteration_limit, o.qp_nullspace_limit, o.simplex_primal_edge_weight_strategy);
        let (qp_allow_hot_start, timeless_log, qp_regularization_value, time_limit, dual_feasibility_tolerance) =
            (o.qp_allow_hot_start, o.timeless_log, o.qp_regularization_value, o.time_limit, o.dual_feasibility_tolerance);
        let ctx = self as *mut LpHandle as *mut c_void;
        let lp = self.model.view();
        let h = &mut self.top.as_mut().expect("top").hessian;
        let hessian_dim = h.dim;
        let (hs, hi, hval) = (rmv(&mut h.start), rmv(&mut h.index), rmv(&mut h.value));
        let r = &mut self.lps.run;
        let host = crate::qp::glue::CQpHost {
            ctx,
            op: qp_op,
            log,
            lp,
            hessian_dim,
            hessian_start: hs,
            hessian_index: hi,
            hessian_value: hval,
            qp_iteration_limit,
            qp_nullspace_limit,
            simplex_primal_edge_weight_strategy,
            qp_allow_hot_start,
            timeless_log,
            qp_regularization_value,
            time_limit,
            dual_feasibility_tolerance,
            col_value: rmv(&mut r.solution.col_value),
            row_value: rmv(&mut r.solution.row_value),
            col_status: rmv(&mut r.basis.b.col_status),
            row_status: rmv(&mut r.basis.b.row_status),
            value_valid: &mut r.solution.value_valid,
            dual_valid: &mut r.solution.dual_valid,
            basis_valid: &mut r.basis.b.valid,
            basis_alien: &mut r.basis.b.alien,
            basis_useful: &mut r.basis.b.useful,
            model_status: &mut r.model_status,
            info: &mut r.info,
        };
        // SAFETY: the views live for the call; the op resizes the
        // solution and basis only after the hot start check read them
        let return_status = unsafe { crate::qp::glue::call_solve_qp(&host) };
        if return_status == Status::Error {
            return return_status;
        }
        // The objective (HighsModel::objectiveValue) and KKT failures
        let hv = self.hessian_view();
        let r = &mut self.lps.run;
        let x = &r.solution.col_value;
        let m = &mut self.model;
        let lp_obj = {
            let v = m.view();
            // SAFETY: the model's view lives for the call
            unsafe { LpRef::new(&v) }.objective_value(x)
        };
        r.info.objective_function_value = hessian::objective_value(&hv, x) + lp_obj;
        self.kkt_failures();
        self.lps.run.info.valid = true;
        if self.lps.run.model_status == super::super::run::MS_OPTIMAL {
            return self.with_run(|r| r.check_optimality("QP"));
        }
        return_status
    }

    // ---- callSolveMip's solver

    /// The MIP solver run (MipRun): the user's solution kept, the solver
    /// data invalidated, semi-variables replaced
    fn mip_run(&mut self) -> MipResult {
        let r = &mut self.lps.run;
        let user_solution = r.solution.value_valid;
        let (cv, rv) = if user_solution {
            (std::mem::take(&mut r.solution.col_value), std::mem::take(&mut r.solution.row_value))
        } else {
            (Vec::new(), Vec::new())
        };
        self.invalidate_solver_data();
        if user_solution {
            let s = &mut self.lps.run.solution;
            s.col_value = cv;
            s.row_value = rv;
            s.value_valid = true;
        }
        let log_dev_level = self.opts.log_dev_level;
        debug_assert!(self.model.a.format != super::super::matrix_format::ROWWISE);
        let has_semi = self.model.integrality.iter().any(|&t| t == SEMI_CONTINUOUS || t == SEMI_INTEGER);
        let lp = if has_semi {
            let mft = self.opts.primal_feasibility_tolerance;
            let lp = without_semi_variables(&self.model, &mut self.lps.run.solution, mft);
            self.top().mip_lp = Some(lp.clone());
            lp
        } else {
            self.model.clone()
        };
        // The host's model is the C++ copy: it takes the run's changes
        let model_changed = self.top().changed & X_MODEL != 0;
        let mut host: *mut c_void = std::ptr::null_mut();
        let arg = 1 | if has_semi { 2 } else { 0 } | if model_changed { 4 } else { 0 };
        self.host_top(H_MIP_HOST, arg, &mut host as *mut _ as *mut c_void);
        self.host_top(H_SUBSOLVER, 1, std::ptr::null_mut());
        let s = &self.lps.run.solution;
        let start = if s.value_valid { Some((&s.col_value[..], &s.row_value[..])) } else { None };
        let name = lp.model_name.clone();
        let mut ms = MipSolver::new(host, Prof { p: self.profiling }, self.opts.clone(), self.log(), lp, name, start, false, 0);
        crate::mip::host::fns::run(&mut ms);
        self.host_top(H_SUBSOLVER, 0, std::ptr::null_mut());
        self.opts.log_dev_level = log_dev_level;
        let r = MipResult {
            model_status: ms.modelstatus,
            solution_objective: ms.solution_objective,
            node_count: ms.node_count,
            total_lp_iterations: ms.total_lp_iterations,
            dual_bound: ms.dual_bound,
            gap: ms.gap,
            primal_dual_integral: ms.primal_dual_integral,
            row_violation: ms.row_violation,
            bound_violation: ms.bound_violation,
            integrality_violation: ms.integrality_violation,
        };
        self.top().mip = Some((ms, host));
        r
    }

    /// MipTakeSolution: the solver's solution, the saved solutions and
    /// the row values (productQuad)
    fn mip_take_solution(&mut self) {
        let t = self.top.as_mut().expect("top");
        let (ms, _) = t.mip.as_ref().expect("the MIP solver");
        t.saved = ms.saved_objective_and_solution.clone();
        t.changed |= X_SAVED;
        let s = &mut self.lps.run.solution;
        s.col_value.clone_from(&ms.solution);
        let m = &mut self.model;
        let v = m.view();
        // SAFETY: the model's view lives for the call
        s.row_value = unsafe { LpRef::new(&v) }.product_quad(&s.col_value);
        s.value_valid = true;
    }

    /// assessLpPrimalSolution("", options, model, solution) -> feasible
    fn solution_feasible(&mut self) -> bool {
        let log = self.log();
        let (pft, mft) = (self.opts.primal_feasibility_tolerance, self.opts.mip_feasibility_tolerance);
        let (cn, rn) = self.cpp_names();
        let v = self.model.view();
        let s = &self.lps.run.solution;
        let mut a = PrimalAssessment::default();
        let status =
            assess_lp_primal_solution(&log, "", pft, mft, &v, cn, rn, s.value_valid, &s.col_value, &s.row_value, &mut a);
        debug_assert!(status != Status::Error);
        a.feasible
    }

    /// The PDLP parameters (getUserParamsFromOptions, with its logs) and
    /// print function
    fn pdlp_template(&mut self) -> super::super::lp_run::PdlpTemplate {
        // HConst.h: PdlpFeatures; pdlp_features_off is not an option (0)
        const SCALING_OFF: i32 = 1;
        const RESTART_OFF: i32 = 2;
        const ADAPTIVE_STEP_SIZE_OFF: i32 = 4;
        // cupdlp_defs.h: PDHG_FIXED_LINESEARCH, PDHG_ADAPTIVE_LINESEARCH
        const FIXED_LINESEARCH: i32 = 0;
        const ADAPTIVE_LINESEARCH: i32 = 2;
        const DEFAULT_KKT_TOLERANCE: f64 = 1e-7;
        let o = &self.opts;
        let log = self.log();
        let features_off = 0;
        let mut p = crate::pdlp::Params {
            primal_tol: o.primal_feasibility_tolerance,
            dual_tol: o.dual_feasibility_tolerance,
            gap_tol: o.pdlp_optimality_tolerance,
            time_lim: o.time_limit,
            iter_lim: o.pdlp_iteration_limit,
            log_level: if o.output_flag { if o.log_dev_level != 0 { 2 } else { 1 } } else { 0 },
            scaling: ((features_off & SCALING_OFF) == 0) as i32,
            line_search: FIXED_LINESEARCH,
            restart: 0,
        };
        if p.scaling == 0 {
            log.user(LogType::Info, "PDLP: Scaling off\n");
        }
        let adaptive = (features_off & ADAPTIVE_STEP_SIZE_OFF) == 0;
        p.line_search = if adaptive { ADAPTIVE_LINESEARCH } else { FIXED_LINESEARCH };
        if !adaptive {
            log.user(LogType::Info, "PDLP: Adaptive line search off\n");
        }
        if o.kkt_tolerance != DEFAULT_KKT_TOLERANCE {
            p.primal_tol = o.kkt_tolerance;
            p.dual_tol = o.kkt_tolerance;
            p.gap_tol = o.kkt_tolerance;
        }
        let mut restart = ((features_off & RESTART_OFF) == 0) as i32;
        if o.pdlp_cupdlpc_restart_method == 0 {
            restart = 0;
        }
        p.restart = restart;
        if restart == 0 {
            log.user(LogType::Info, "PDLP: Restart off\n");
        }
        super::super::lp_run::PdlpTemplate { params: p, print: Some(pdlp_print) }
    }
}

/// cupdlp_printf: printf("%s", msg)
extern "C" fn pdlp_print(msg: *const c_char) {
    // SAFETY: a NUL-terminated message
    let s = unsafe { CStr::from_ptr(msg) }.to_bytes();
    crate::io::log::c_stdout(s);
}

// The QP glue's host ops (qp/glue.rs)
unsafe extern "C" fn qp_op(ctx: *mut c_void, code: i32, out: *mut c_void) -> f64 {
    let h = &mut *(ctx as *mut LpHandle);
    match code {
        0 => {
            h.host_top(H_SUBSOLVER, 2 | 1, std::ptr::null_mut());
        }
        1 => {
            h.host_top(H_SUBSOLVER, 2, std::ptr::null_mut());
        }
        2 => return h.clock_read(),
        3 => {
            let (nc, nr) = (h.model.num_col as usize, h.model.num_row as usize);
            let r = &mut h.lps.run;
            let s = &mut r.solution;
            s.col_value.resize(nc, 0.0);
            s.col_dual.resize(nc, 0.0);
            s.row_value.resize(nr, 0.0);
            s.row_dual.resize(nr, 0.0);
            let b = &mut r.basis.b;
            b.col_status.resize(nc, 0);
            b.row_status.resize(nr, 0);
            *(out as *mut super::super::basis::COut) = super::super::basis::COut {
                col_value: rmv(&mut s.col_value),
                col_dual: rmv(&mut s.col_dual),
                row_value: rmv(&mut s.row_value),
                row_dual: rmv(&mut s.row_dual),
                col_status: rmv(&mut b.col_status),
                row_status: rmv(&mut b.row_status),
            };
        }
        _ => unreachable!("QP op {code}"),
    }
    0.0
}

/// withoutSemiVariables: the semi-variables replaced by a binary each and
/// two rows; the solution's values extended to them
pub fn without_semi_variables(from: &Lp, solution: &mut super::super::lp_run::Solution, mft: f64) -> Lp {
    use crate::util::fma::ClangFma;
    let mut lp = from.clone();
    let num_col = lp.num_col as usize;
    let num_row = lp.num_row as usize;
    let is_semi = |t: u8| t == SEMI_CONTINUOUS || t == SEMI_INTEGER;
    let num_semi = lp.integrality.iter().take(num_col).filter(|&&t| is_semi(t)).count();
    debug_assert!(num_semi > 0);
    let g = &mut lp.g;
    {
        let a = &mut g.a;
        let num_nz = a.start[num_col] as usize;
        let new_num_nz = num_nz + 2 * num_semi;
        let mut new_el = new_num_nz;
        a.index.resize(new_num_nz, 0);
        a.value.resize(new_num_nz, 0.0);
        for c in (0..num_col).rev() {
            let from_el = a.start[c + 1] as i64 - 1;
            a.start[c + 1] = new_el as i32;
            if is_semi(g.integrality[c]) {
                new_el -= 2;
            }
            let mut el = from_el;
            while el >= a.start[c] as i64 {
                new_el -= 1;
                a.index[new_el] = a.index[el as usize];
                a.value[new_el] = a.value[el as usize];
                el -= 1;
            }
        }
        debug_assert_eq!(new_el, 0);
        let mut row_num = num_row as i32;
        for c in 0..num_col {
            if is_semi(g.integrality[c]) {
                let el = a.start[c + 1] as usize - 2;
                a.index[el] = row_num;
                a.value[el] = 1.0;
                row_num += 1;
                a.index[el + 1] = row_num;
                a.value[el + 1] = 1.0;
                row_num += 1;
            }
        }
    }
    let have_solution = solution.value_valid;
    if have_solution {
        solution.row_value.resize(num_row + 2 * num_semi, 0.0);
    }
    let mut row_num = num_row;
    for c in 0..num_col {
        if !is_semi(g.integrality[c]) {
            continue;
        }
        let semi_lower = g.col_lower[c];
        let semi_upper = g.col_upper[c];
        g.col_cost.push(0.0);
        g.col_lower.push(0.0);
        g.col_upper.push(1.0);
        g.row_lower.push(0.0);
        g.row_upper.push(INF);
        g.a.index.push(row_num as i32);
        row_num += 1;
        g.a.value.push(-semi_lower);
        if have_solution {
            let prev_primal = solution.col_value[c];
            if solution.col_value[c] <= mft {
                solution.col_value[c] = 0.0;
                solution.col_value.push(0.0);
            } else {
                solution.col_value[c] = super::super::lp_utils::cmax(semi_lower, solution.col_value[c]);
                solution.col_value.push(1.0);
            }
            let dl_primal = solution.col_value[c] - prev_primal;
            if dl_primal != 0.0 {
                let a = &g.a;
                for el in a.start[c] as usize..a.start[c + 1] as usize {
                    let r = a.index[el] as usize;
                    solution.row_value[r] = dl_primal.mul_add_c(a.value[el], solution.row_value[r]);
                }
            }
            let new_col = g.col_cost.len() - 1;
            let binary_value = solution.col_value[new_col];
            solution.row_value[row_num - 1] = (-semi_lower).mul_add_c(binary_value, solution.col_value[c]);
            solution.row_value[row_num] = (-semi_upper).mul_add_c(binary_value, solution.col_value[c]);
        }
        g.row_lower.push(-INF);
        g.row_upper.push(0.0);
        g.a.index.push(row_num as i32);
        row_num += 1;
        g.a.value.push(-semi_upper);
        let len = g.a.index.len() as i32;
        g.a.start.push(len);
        g.integrality.push(INTEGER);
        g.integrality[c] = if g.integrality[c] == SEMI_CONTINUOUS { CONTINUOUS } else { INTEGER };
        g.col_lower[c] = 0.0;
    }
    g.num_col += num_semi as i32;
    g.num_row += 2 * num_semi as i32;
    g.a.num_col = g.num_col;
    g.a.num_row = g.num_row;
    lp
}

// ---- The C++ mirror of the top level

/// What C++ passes in and takes back (HighsRunRust.cpp: RsTopData): the
/// Highs object's solution, basis, info, model status, run data,
/// presolve status, optimizeModel flag and Hessian, in place
#[repr(C)]
pub struct CTopData {
    pub run: CRunData,
    pub run_data: *mut RunData,
    pub presolve_status: *mut i32,
    pub called_return: *mut bool,
    pub hessian: super::super::hessian::CHessian,
    pub profiling: *mut c_void,
}

/// The Highs object's data into its engine before a call: the solution,
/// basis, info and model status if `run` (the mirror is newer), the
/// scalars and the Hessian
///
/// # Safety
/// `p` a Highs object's engine, `d` valid views
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_import(p: *mut LpHandle, d: *const CTopData, run: bool) {
    let h = &mut *p;
    let d = &*d;
    if run {
        let presolve = std::mem::take(&mut h.lps.run.presolve);
        h.lps.run.import(&d.run);
        h.lps.run.presolve = presolve;
    }
    h.run_data = std::ptr::read(d.run_data);
    h.presolve_status = *d.presolve_status;
    h.called_return = *d.called_return;
    h.profiling = d.profiling;
    let hs = d.hessian.load();
    let t = h.top();
    t.hessian = hs;
    t.changed = 0;
}

/// Whether the engine's solution, basis, info and model status are the
/// mirror's (HIGHS_RS_CHECK_SYNC: a mirror that is not newer must be);
/// the parts that differ are printed
///
/// # Safety
/// As highs_rs_lph_top_import
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_run_matches(p: *mut LpHandle, d: *const CTopData) -> bool {
    let r = &(*p).lps.run;
    let d = &(*d).run;
    let same = |a: &[f64], b: &[f64]| a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    let s = &r.solution;
    let cb = &*d.basis;
    let b = &r.basis.b;
    let mut diff = Vec::new();
    if s.value_valid != *d.value_valid || s.dual_valid != *d.dual_valid {
        diff.push("solution validity");
    }
    if !same(&s.col_value, d.col_value.as_slice())
        || !same(&s.col_dual, d.col_dual.as_slice())
        || !same(&s.row_value, d.row_value.as_slice())
        || !same(&s.row_dual, d.row_dual.as_slice())
    {
        diff.push("solution");
    }
    if b.valid != cb.valid
        || b.alien != cb.alien
        || b.useful != cb.useful
        || b.was_alien != cb.was_alien
        || b.debug_id != cb.debug_id
        || b.debug_update_count != cb.debug_update_count
        || b.col_status.as_slice() != cb.col_status.as_slice()
        || b.row_status.as_slice() != cb.row_status.as_slice()
        || r.basis.origin.as_bytes() != d.origin.get()
    {
        diff.push("basis");
    }
    if !r.info.equal(&*d.info) {
        diff.push("info");
    }
    if r.model_status != *d.model_status {
        diff.push("model status");
    }
    if !diff.is_empty() {
        eprintln!("HIGHS_RS_CHECK_SYNC: the engine's {} differ from the mirror's", diff.join(", "));
    }
    diff.is_empty()
}

/// The X_* flags of what changed in a call besides the solution, basis,
/// info, model status and run data (C++ takes those parts), cleared
///
/// # Safety
/// `p` a Highs object's engine
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_changed(p: *mut LpHandle) -> u32 {
    let h = &mut *p;
    let active = h.lps.run.presolve.active;
    let t = h.top();
    let mut changed = t.changed;
    if !active {
        changed &= !X_PRESOLVE;
    }
    t.changed = 0;
    changed
}

/// The engine's solution, basis, info, model status, run data, presolve
/// status and optimizeModel flag back into the Highs object after a call
///
/// # Safety
/// As highs_rs_lph_top_import
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_export(p: *mut LpHandle, d: *mut CTopData) {
    let h = &mut *p;
    let d = &mut *d;
    h.lps.run.export(&mut d.run);
    std::ptr::write(d.run_data, std::ptr::read(&h.run_data));
    *d.presolve_status = h.presolve_status;
    *d.called_return = h.called_return;
}

/// The engine's Hessian into the Highs object's (X_HESSIAN)
///
/// # Safety
/// `p` a Highs object's engine, `h` the view of a HighsHessian
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_hessian(p: *mut LpHandle, h: *mut super::super::hessian::CHessian) {
    (*h).store(&(*p).top().hessian);
}

/// presolved_model_'s source (X_PRESOLVED_MODEL): 0 the model, 1 the
/// reduced LP
///
/// # Safety
/// `p` a Highs object's engine
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_presolved_which(p: *mut LpHandle) -> i32 {
    (*p).top().presolved_which
}

/// The presolve component's records (X_PRESOLVE_RECORDS)
///
/// # Safety
/// `p` a Highs object's engine
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_records(p: *mut LpHandle) -> PresolveRecords {
    (*p).top().records
}

/// The saved improving solution k (X_SAVED): its objective, values and
/// length; the number of them
///
/// # Safety
/// `p` a Highs object's engine; the values valid until the next call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_saved(p: *mut LpHandle, k: usize, objective: *mut f64, n: *mut usize) -> usize {
    let t = (*p).top();
    if k < t.saved.len() {
        let (obj, v) = &t.saved[k];
        *objective = *obj;
        *n = v.len();
        return v.as_ptr() as usize;
    }
    t.saved.len()
}

/// clearModel's part of the top level (saved_objective_and_solution_)
///
/// # Safety
/// `p` a Highs object's engine
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_clear_saved(p: *mut LpHandle) {
    (*p).top().saved.clear();
}

/// Highs::calledOptimizeModel on the engine; a cancelled task's interrupt
/// in `interrupted` (C++ throws HighsTask::Interrupt)
///
/// # Safety
/// `p` a Highs object's engine, its data imported
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_called_optimize_model(p: *mut LpHandle, interrupted: *mut bool) -> i32 {
    let h = &mut *p;
    h.task_interrupted = false;
    let s = h.with_run(|r| r.called_optimize_model());
    *interrupted = h.task_interrupted;
    s as i32
}

/// A user solution (RsSolution)
unsafe fn solution_of(s: &super::super::solution::CSolution) -> super::super::lp_run::Solution {
    super::super::lp_run::Solution {
        value_valid: s.value_valid,
        dual_valid: s.dual_valid,
        col_value: s.col_value.get().to_vec(),
        col_dual: s.col_dual.get().to_vec(),
        row_value: s.row_value.get().to_vec(),
        row_dual: s.row_dual.get().to_vec(),
    }
}

/// A user basis (RsBasisVec and its origin)
unsafe fn basis_of(b: &super::super::interface::BasisG<crate::lp_data::ffi::RsVec<u8>>, origin: &[u8]) -> super::super::lp_run::Basis {
    super::super::lp_run::Basis {
        b: super::super::interface::BasisG {
            valid: b.valid,
            alien: b.alien,
            useful: b.useful,
            was_alien: b.was_alien,
            debug_id: b.debug_id,
            debug_update_count: b.debug_update_count,
            col_status: b.col_status.as_slice().to_vec(),
            row_status: b.row_status.as_slice().to_vec(),
        },
        origin: String::from_utf8_lossy(origin).into_owned(),
    }
}

/// Highs::presolve on the engine
///
/// # Safety
/// `p` a Highs object's engine, its data imported
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_presolve(p: *mut LpHandle, interrupted: *mut bool) -> i32 {
    let h = &mut *p;
    h.task_interrupted = false;
    let s = h.with_run(|r| r.presolve());
    *interrupted = h.task_interrupted;
    s as i32
}

/// Highs::callRunPostsolve(solution, basis) on the engine
///
/// # Safety
/// As highs_rs_lph_top_presolve; the views valid for the call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_postsolve(
    p: *mut LpHandle,
    solution: *const super::super::solution::CSolution,
    basis: *const super::super::interface::BasisG<crate::lp_data::ffi::RsVec<u8>>,
    origin: *const u8,
    origin_len: usize,
    interrupted: *mut bool,
) -> i32 {
    let h = &mut *p;
    h.task_interrupted = false;
    h.top().user_solution = Some(solution_of(&*solution));
    h.user_basis = Some(basis_of(&*basis, crate::ffi::sl(origin, origin_len as i32)));
    let s = h.in_lp_mode(|r| r.call_run_postsolve());
    h.top().user_solution = None;
    h.user_basis = None;
    *interrupted = h.task_interrupted;
    s as i32
}

/// Highs::crossover(solution) on the engine
///
/// # Safety
/// As highs_rs_lph_top_postsolve
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_crossover(
    p: *mut LpHandle,
    solution: *const super::super::solution::CSolution,
    interrupted: *mut bool,
) -> i32 {
    let h = &mut *p;
    h.task_interrupted = false;
    h.top().user_solution = Some(solution_of(&*solution));
    let s = h.with_run(|r| r.crossover());
    h.top().user_solution = None;
    *interrupted = h.task_interrupted;
    s as i32
}

/// Highs::setBasis(basis, origin) on the engine (the model column-wise)
///
/// # Safety
/// As highs_rs_lph_top_postsolve
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_set_basis(
    p: *mut LpHandle,
    basis: *const super::super::interface::BasisG<crate::lp_data::ffi::RsVec<u8>>,
    basis_origin: *const u8,
    basis_origin_len: usize,
    origin: *const u8,
    origin_len: usize,
) -> i32 {
    let h = &mut *p;
    if !h.model.is_colwise() {
        h.model.a.ensure_colwise();
        h.top().changed |= X_MODEL;
    }
    let b = basis_of(&*basis, crate::ffi::sl(basis_origin, basis_origin_len as i32));
    let origin = String::from_utf8_lossy(crate::ffi::sl(origin, origin_len as i32)).into_owned();
    h.set_basis(b, &origin) as i32
}

/// Highs::passModel(model) (which 0, the LP's view and Hessian) or
/// passHessian (1, the Hessian) on the engine
///
/// # Safety
/// As highs_rs_lph_top_postsolve
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lph_top_pass_model(
    p: *mut LpHandle,
    which: i32,
    lp: *const CLp,
    name: *const u8,
    name_len: usize,
    hessian: *const super::super::hessian::CHessian,
) -> i32 {
    let h = &mut *p;
    h.top().user_hessian = Some((*hessian).load());
    if which == 0 {
        let mut m = Lp::default();
        m.import(&*lp, crate::ffi::sl(name, name_len as i32));
        h.user_model = Some(m);
    }
    let s = h.with_run(|r| if which == 0 { r.pass_model() } else { r.pass_hessian() });
    h.user_model = None;
    h.top().user_hessian = None;
    s as i32
}
