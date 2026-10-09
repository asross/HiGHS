//! The C++ objects of the MIP solver that Rust drives but does not own:
//! HighsDomain (its logic is Rust, see domain.rs, but external code keeps
//! reading its vectors), HighsLpRelaxation (the `Highs` LP solver around
//! the Rust state of lp_relaxation.rs), HighsSearch (the C++ shell of
//! search.rs holding the local domain), HighsMipWorker and
//! HighsMipSolver(Data). Rust reaches them through [`CMipFns`], one C++
//! function per operation (highs/mip/HighsMipRust.cpp), and reads the
//! MIP solver's data in place through [`MipData`] (pointers to the C++
//! fields, so values changed by a callback are seen at once).
//!
//! The handles are plain pointers: a handle created here ([`Dom::copy`],
//! [`Lp::copy`], [`SearchH::new`]) is freed when dropped, the others are
//! borrowed from the C++ objects for the duration of a call.

use super::domain::{DomChg, Reason, StdVec};
use super::lp_relaxation::{LpRelax, LpShared};
use crate::lp_data::lp_handle::LpHandle;
use crate::lp_data::opts::OptValue;
use super::nodequeue::NodeQueue;
use super::pseudocost::Pseudocost;
use super::search::{FracInt, Search, Stats};
use crate::lp_data::Log;
use crate::util::cdouble::CDouble;
use crate::util::random::HighsRandom;
use std::ffi::c_void;
use std::sync::atomic::{AtomicPtr, Ordering};

pub type P = *mut c_void;

/// HighsLpRelaxation::Status
pub mod lp_status {
    pub const NOT_SET: i32 = 0;
    pub const OPTIMAL: i32 = 1;
    pub const INFEASIBLE: i32 = 2;
    pub const UNSCALED_DUAL_FEASIBLE: i32 = 3;
    pub const UNSCALED_PRIMAL_FEASIBLE: i32 = 4;
    pub const UNSCALED_INFEASIBLE: i32 = 5;
    pub const UNBOUNDED: i32 = 6;
    pub const ERROR: i32 = 7;

    /// HighsLpRelaxation::unscaledPrimalFeasible
    pub fn unscaled_primal_feasible(st: i32) -> bool {
        st == OPTIMAL || st == UNSCALED_PRIMAL_FEASIBLE
    }
}

/// HighsModelStatus::kNotset and kInfeasible
pub const MODEL_STATUS_NOTSET: i32 = 0;
pub const MODEL_STATUS_INFEASIBLE: i32 = 8;

/// MipSolutionSource (HighsMipSolverData.h)
pub mod source {
    pub const BRANCHING: i32 = 0;
    pub const CENTRAL_ROUNDING: i32 = 1;
    pub const FEASIBILITY_PUMP: i32 = 2;
    pub const GRAPH_LNS: i32 = 3;
    pub const HEURISTIC: i32 = 4;
    pub const SHIFTING: i32 = 5;
    pub const FEASIBILITY_JUMP: i32 = 6;
    pub const SUB_MIP: i32 = 7;
    pub const RANDOMIZED_ROUNDING: i32 = 9;
    pub const SOLVE_LP: i32 = 10;
    pub const EVALUATE_NODE: i32 = 11;
    pub const UNBOUNDED: i32 = 12;
    pub const ZI_ROUND: i32 = 15;
}

/// The handles of a heuristic's HighsSearch (with its own copy of the
/// pseudocosts), filled by `search_new`
#[repr(C)]
pub struct SearchParts {
    pub cpp: P,
    pub rs: *mut Search,
    pub ps: *mut Pseudocost,
    pub nq: *const NodeQueue,
    pub localdom: P,
}

/// The results of a sub-MIP run that solveSubMip uses
#[repr(C)]
#[derive(Default)]
pub struct SubMipResult {
    pub termination_status: i32,
    pub model_status: i32,
    pub node_count: i64,
    pub total_lp_iterations: i64,
    pub total_repair_lp: i64,
    pub total_repair_lp_feasible: i64,
    pub total_repair_lp_iterations: i64,
    pub max_submip_level: i32,
    pub has_solution: bool,
}

/// A sub-MIP to build and run (CMipFns::sub_mip): its bounds, start and
/// the options that differ from the caller's
#[repr(C)]
pub struct SubMipSpec {
    /// HighsLpRelaxation whose LP and basis are used, or null for the
    /// model and the first root basis
    pub lp: P,
    pub col_lower: *const f64,
    pub col_upper: *const f64,
    /// num_col values and their num_row row activities, or null
    pub start_cols: *const f64,
    pub start_rows: *const f64,
    pub num_start_rows: i32,
    pub mip_max_leaves: i32,
    pub mip_max_nodes: i32,
    pub mip_max_stall_nodes: i32,
    pub mip_pscost_minreliable: i32,
    pub time_limit: f64,
    pub objective_bound: f64,
    /// both NaN to keep the caller's
    pub mip_rel_gap: f64,
    pub mip_abs_gap: f64,
    pub mip_heuristic_effort: f64,
    /// RINS, RENS and root reduced cost (bits 0-2), or -1 to keep the
    /// caller's
    pub heur_flags: i32,
    pub presolve: bool,
    pub output_flag: bool,
    pub mip_detect_symmetry: bool,
    /// the sub-MIP's lns_target_reached_
    pub lns_target: *const super::concurrent::Pool,
}

/// The C++ operations, implemented in highs/mip/HighsMipRust.cpp
#[repr(C)]
pub struct CMipFns {
    // HighsDomain
    pub dom_copy: unsafe extern "C" fn(P) -> P,
    pub dom_free: unsafe extern "C" fn(P),
    pub dom_assign: unsafe extern "C" fn(P, P),
    pub dom_bounds: unsafe extern "C" fn(P, *mut *const f64, *mut *const f64),
    pub dom_change_bound: unsafe extern "C" fn(P, DomChg, Reason),
    pub dom_fix_col: unsafe extern "C" fn(P, i32, f64, Reason),
    pub dom_propagate: unsafe extern "C" fn(P) -> bool,
    pub dom_infeasible: unsafe extern "C" fn(P) -> bool,
    pub dom_backtrack: unsafe extern "C" fn(P) -> DomChg,
    /// conflictAnalysis with the worker's conflict pool, global domain and
    /// pseudocosts
    pub dom_conflict_analysis: unsafe extern "C" fn(P, P),
    pub dom_stack: unsafe extern "C" fn(P, *mut i32) -> *const DomChg,
    pub dom_branch_depth: unsafe extern "C" fn(P) -> i32,
    pub dom_clear_changed_cols: unsafe extern "C" fn(P),
    pub dom_clear_pool_propagation: unsafe extern "C" fn(P),
    pub dom_num_changed_cols: unsafe extern "C" fn(P) -> i32,
    // HighsLpRelaxation
    /// a copy of an LP relaxation for the worker
    pub lp_copy: unsafe extern "C" fn(P, P) -> P,
    /// a fresh LP relaxation of the model (loadModel) for the worker
    pub lp_new: unsafe extern "C" fn(P, P) -> P,
    pub lp_free: unsafe extern "C" fn(P),
    /// the Rust LP relaxation (with the LP solver)
    pub lp_rust: unsafe extern "C" fn(P) -> *mut LpRelax,
    /// setBasis(firstrootbasis, origin)
    pub lp_set_root_basis: unsafe extern "C" fn(P, *const u8),
    /// resolveLp(domain or null)
    pub lp_resolve: unsafe extern "C" fn(P, P) -> i32,
    pub lp_set_objective_limit: unsafe extern "C" fn(P, f64),
    pub lp_flush_domain: unsafe extern "C" fn(P, P),
    pub lp_remove_obsolete_rows: unsafe extern "C" fn(P, bool),
    /// computeDualInfProof, and if it holds, generateConflict on the local
    /// domain with the worker's cut pool
    pub lp_infeasible_conflict: unsafe extern "C" fn(P, P, P),
    // HighsSearch
    pub search_new: unsafe extern "C" fn(P, *mut SearchParts),
    pub search_free: unsafe extern "C" fn(P),
    pub search_set_lp: unsafe extern "C" fn(P, P),
    // HighsMipSolver(Data) and the worker
    pub check_limits: unsafe extern "C" fn(P) -> bool,
    pub update_lower_bound: unsafe extern "C" fn(P, f64),
    pub parallel_lock_active: unsafe extern "C" fn(P) -> bool,
    pub num_workers: unsafe extern "C" fn(P) -> i32,
    /// fills the worker's view
    pub worker_view: unsafe extern "C" fn(P, *mut WorkerData),
    /// solveSubMip's run: the HighsMipSolver of the sub-MIP of the spec,
    /// for the worker; the solution goes to `sol`
    pub sub_mip: unsafe extern "C" fn(P, P, *const SubMipSpec, *mut SubMipResult, *mut f64),
    /// a scalar operation on the solver (mip_data.rs, mod op)
    pub op: unsafe extern "C" fn(P, i32, P, i64, f64) -> f64,
    /// the scratch solution of transformNewIntegerFeasibleSolution:
    /// col_value = sol (if not null), primal postsolve and row values; its
    /// vectors to the view
    pub scratch_solution: unsafe extern "C" fn(P, *const f64, i32, *mut ScratchView),
    /// refills a MipData (after the model changed)
    pub refill: unsafe extern "C" fn(P, *mut MipData),
    /// the master worker (workers[0])
    pub master_worker: unsafe extern "C" fn(P) -> P,
    /// runTask(processNode) over the indices (a parallel lock, tasks if
    /// more than one) with highs_rs_mip_process_node and the context
    pub run_process_nodes: unsafe extern "C" fn(P, *const i32, i32, *const std::ffi::c_void),
    /// the results of cleanupSolve into the HighsMipSolver
    pub set_cleanup_result: unsafe extern "C" fn(P, *const std::ffi::c_void),
    /// the model name (data, length) and max_submip_level
    pub model_name: unsafe extern "C" fn(P, *mut i32) -> *const u8,
    pub max_submip_level: unsafe extern "C" fn(P) -> i32,
    /// the concurrent helper's options, model and root basis (with the
    /// time left) for helper_run
    pub helper_new: unsafe extern "C" fn(P, f64) -> P,
    /// runs the helper's HighsMipSolver on the data of helper_new (which
    /// it frees) with the pool (concurrent.rs) in its own thread
    pub helper_run: unsafe extern "C" fn(P, *const c_void),
    /// getCutPool().addCut of a helper's root cut
    pub add_root_cut: unsafe extern "C" fn(P, *const i32, *const f64, i32, f64, bool),
    /// the data and size of the solver's vector `which` (setup::vptr), or
    /// null
    pub vec_ptr: unsafe extern "C" fn(P, i32, *mut i32) -> *const c_void,
    /// sets the basis `which` (setup::basis): its statuses and flags
    /// (valid, alien, useful)
    pub set_basis: unsafe extern "C" fn(P, i32, *const u8, i32, *const u8, i32, bool, bool, bool),
    /// with no data: callbackActive(type); otherwise the callback's data_out
    /// set from the CallbackOut (setup.rs) and callbackAction(type, the
    /// message of the given length) (returns the interrupt)
    pub callback: unsafe extern "C" fn(P, i32, *const super::setup::CallbackOut, *const u8, i32) -> bool,
    /// the solver's worker k
    pub worker: unsafe extern "C" fn(P, i32) -> P,
    /// the worker's scratch solution: col_value = sol, primal postsolve
    /// (thread safe) and row values; its vectors to the view
    pub worker_scratch: unsafe extern "C" fn(P, P, *const f64, i32, *mut ScratchView),
    /// the repair LP: the original model with these column bounds and no
    /// integers, simplex with the time limit, primal feasibility tolerance
    /// and presolve (choose, or off); the iterations to the last argument;
    /// if primal feasible, its solution becomes the scratch solution
    pub repair_lp: unsafe extern "C" fn(P, *const f64, *const f64, f64, f64, bool, *mut i64) -> bool,
}

static FNS: AtomicPtr<CMipFns> = AtomicPtr::new(std::ptr::null_mut());

/// Sets the C++ functions (each entry point passes them)
pub fn set_fns(f: *const CMipFns) {
    FNS.store(f as *mut CMipFns, Ordering::Relaxed);
}

#[inline(always)]
pub fn fns() -> &'static CMipFns {
    // SAFETY: set by the entry points before any use; a static C++ table
    unsafe { &*FNS.load(Ordering::Relaxed) }
}

macro_rules! c {
    ($f:ident $(, $a:expr)*) => {
        // SAFETY: the C++ operation on a live object
        unsafe { (fns().$f)($($a),*) }
    };
}

/// The primal-dual integral (HighsPrimaDualIntegral)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Pdi {
    pub value: f64,
    pub prev_lb: f64,
    pub prev_ub: f64,
    pub prev_gap: f64,
    pub prev_time: f64,
}

/// The scalars of HighsMipSolverData (HighsMipScalars, same layout)
#[repr(C)]
pub struct MipScalars {
    pub feastol: f64,
    pub epsilon: f64,
    pub heuristic_effort: f64,
    pub dispfreq: i64,
    pub firstlpsolobj: f64,
    pub rootlpsolobj: f64,
    pub numintegercols: i32,
    pub max_tree_size_log2: i32,
    pub pruned_treeweight: CDouble,
    pub avgrootlpiters: f64,
    pub disptime: f64,
    pub last_disptime: f64,
    pub firstrootlpiters: i64,
    pub num_nodes: i64,
    pub num_leaves: i64,
    pub num_leaves_before_run: i64,
    pub num_nodes_before_run: i64,
    pub total_repair_lp: i64,
    pub total_repair_lp_feasible: i64,
    pub total_repair_lp_iterations: i64,
    pub total_lp_iterations: i64,
    pub heuristic_lp_iterations: i64,
    pub sepa_lp_iterations: i64,
    pub sb_lp_iterations: i64,
    pub total_lp_iterations_before_run: i64,
    pub heuristic_lp_iterations_before_run: i64,
    pub sepa_lp_iterations_before_run: i64,
    pub sb_lp_iterations_before_run: i64,
    pub num_disp_lines: i64,
    pub num_improving_sols: i32,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub upper_limit: f64,
    pub optimality_limit: f64,
    pub num_restarts: i32,
    pub num_restarts_root: i32,
    pub num_clique_entries_after_presolve: i32,
    pub num_clique_entries_after_first_presolve: i32,
    pub lns_tree_next: i64,
    pub lns_tree_wait: i64,
    pub lns_quick_lp_iterations: i64,
    pub concurrent_lns_seen: i64,
    pub pdi: Pdi,
    pub cliques_extracted: bool,
    pub row_matrix_set: bool,
    pub analytic_center_computed: bool,
    pub detect_symmetries: bool,
    pub lns_quick_improved: bool,
    pub crossover_start_logged: bool,
    pub root_cuts_imported: bool,
    /// the main solver's concurrent helper (concurrent.rs), or null
    pub concurrent_lns: *mut super::concurrent::Main,
}

/// The options the solver reads (a copy per call)
#[repr(C)]
pub struct MipOptions {
    pub objective_bound: f64,
    pub objective_target: f64,
    pub mip_abs_gap: f64,
    pub mip_rel_gap: f64,
    pub mip_feasibility_tolerance: f64,
    pub time_limit: f64,
    pub mip_min_logging_interval: f64,
    pub mip_max_nodes: i32,
    pub mip_max_leaves: i32,
    pub mip_max_improving_sols: i32,
    pub output_flag: bool,
    pub timeless_log: bool,
    pub run_zi_round: bool,
    pub run_shifting: bool,
    pub run_graph_lns: bool,
    pub run_root_reduced_cost: bool,
    pub run_rens: bool,
    pub run_rins: bool,
    pub mip_allow_restart: bool,
    pub presolve_off: bool,
    pub run_feasibility_jump: bool,
    /// the output_flag option (the log options' flag is output_flag)
    pub output_flag_option: bool,
    pub mip_max_stall_nodes: i32,
    pub small_matrix_value: f64,
    pub mip_heuristic_effort: f64,
    pub mip_report_level: i32,
    pub restart_presolve_reduction_limit: i32,
    pub presolve_reduction_limit: i32,
    pub mip_detect_symmetry: bool,
    pub mip_improving_solution_save: bool,
    pub mip_concurrent_crossover: bool,
}

/// The original model (for solutions in the original space)
#[repr(C)]
pub struct OrigModel {
    pub num_col: i32,
    pub num_row: i32,
    pub offset: f64,
    pub col_cost: *const Vec<f64>,
    pub col_lower: *const Vec<f64>,
    pub col_upper: *const Vec<f64>,
    pub row_lower: *const Vec<f64>,
    pub row_upper: *const Vec<f64>,
    pub integrality: *const Vec<u8>,
    pub a_start: *const Vec<i32>,
    pub a_index: *const Vec<i32>,
    pub a_value: *const Vec<f64>,
}

/// HighsMipSolver's solution fields
#[repr(C)]
pub struct SolutionPtrs {
    pub objective: *mut f64,
    pub bound_violation: *mut f64,
    pub integrality_violation: *mut f64,
    pub row_violation: *mut f64,
    pub solution: *const Vec<f64>,
}

/// HighsMipSolverData (and its HighsMipSolver), read in place. Filled by
/// the C++ per call (HighsPrimalHeuristics.cpp, mipData); the pointers stay
/// valid during the call.
#[repr(C)]
pub struct MipData {
    pub mipsolver: P,
    pub log: Log,
    // the (presolved) model
    pub num_col: i32,
    pub num_row: i32,
    pub colwise: bool,
    pub minimize: bool,
    pub orig_maximize: bool,
    pub submip: bool,
    pub concurrent_helper: bool,
    pub root_presolve_only: bool,
    pub offset: f64,
    pub a_start: *const Vec<i32>,
    pub a_index: *const Vec<i32>,
    pub a_value: *const Vec<f64>,
    pub col_cost: *const Vec<f64>,
    pub col_lower: *const Vec<f64>,
    pub col_upper: *const Vec<f64>,
    pub row_lower: *const Vec<f64>,
    pub row_upper: *const Vec<f64>,
    pub integrality: *const Vec<u8>,
    // HighsMipSolverData
    pub ar_start: *const StdVec<i32>,
    pub ar_index: *const StdVec<i32>,
    pub ar_value: *const StdVec<f64>,
    pub uplocks: *const StdVec<i32>,
    pub downlocks: *const StdVec<i32>,
    pub integer_cols: *const StdVec<i32>,
    pub integral_cols: *const StdVec<i32>,
    pub continuous_cols: *const StdVec<i32>,
    pub rootlpsol: *const StdVec<f64>,
    pub firstlpsol: *const StdVec<f64>,
    pub analytic_center: *const StdVec<f64>,
    pub incumbent: *const StdVec<f64>,
    pub scalars: *mut MipScalars,
    pub clique: *const super::clique::CliqueTable,
    pub redcost: *const super::redcost::RedcostFixing,
    pub nodequeue: *mut super::nodequeue::NodeQueue,
    pub globaldom: P,
    pub lp: P,
    pub modelstatus: *mut i32,
    pub solution: SolutionPtrs,
    pub orig: OrigModel,
    pub opts: MipOptions,
    /// in a concurrent helper: the pool shared with its main solver
    pub helper_pool: *const super::concurrent::Pool,
    /// in a sub-MIP: the pool whose target reached ends the solve
    pub lns_target: *const super::concurrent::Pool,
    /// the solver's heuristics
    pub heur: *const super::primal::Heur,
    /// HighsMipSolverData's vectors
    pub vecs: *mut super::mip_data::MipVecs,
}

macro_rules! vecs {
    ($($name:ident: $t:ty),*) => {$(
        #[inline(always)]
        pub fn $name(&self) -> &[$t] {
            // SAFETY: a live C++ vector (see MipData)
            unsafe { (*self.$name).as_slice() }
        }
    )*};
}
macro_rules! vals {
    ($($name:ident: $t:ty),*) => {$(
        #[inline(always)]
        pub fn $name(&self) -> $t {
            // SAFETY: a live C++ field (see MipData)
            unsafe { (*self.scalars).$name }
        }
    )*};
}

impl MipData {
    vecs!(a_start: i32, a_index: i32, a_value: f64, col_cost: f64, col_lower: f64, col_upper: f64,
          row_lower: f64, row_upper: f64, integrality: u8, ar_start: i32, ar_index: i32, ar_value: f64,
          uplocks: i32, downlocks: i32, integer_cols: i32, integral_cols: i32, continuous_cols: i32,
          rootlpsol: f64, firstlpsol: f64, analytic_center: f64, incumbent: f64);
    vals!(feastol: f64, lower_bound: f64, upper_bound: f64, upper_limit: f64, optimality_limit: f64,
          avgrootlpiters: f64, firstrootlpiters: i64, total_lp_iterations: i64,
          heuristic_lp_iterations: i64, sb_lp_iterations: i64, num_improving_sols: i32);

    pub fn num_nodes(&self) -> i64 {
        // SAFETY: as vals
        unsafe { (*self.scalars).num_nodes }
    }
    pub fn add_num_nodes(&self, n: i64) {
        // SAFETY: as vals; no other reference to the field is live
        unsafe { (*self.scalars).num_nodes += n }
    }
    pub fn check_limits(&self) -> bool {
        c!(check_limits, self.mipsolver)
    }
    pub fn update_lower_bound(&self, lb: f64) {
        c!(update_lower_bound, self.mipsolver, lb)
    }
    pub fn parallel_lock_active(&self) -> bool {
        c!(parallel_lock_active, self.mipsolver)
    }
    pub fn num_workers(&self) -> i32 {
        c!(num_workers, self.mipsolver)
    }
    pub fn clique(&self) -> &super::clique::CliqueTable {
        // SAFETY: the solver's clique table, not changed during the call
        unsafe { &*self.clique }
    }
    pub fn redcost(&self) -> &super::redcost::RedcostFixing {
        // SAFETY: as clique
        unsafe { &*self.redcost }
    }
}

/// HighsMipWorker::HeurStatistics (same layout)
#[repr(C)]
#[derive(Default)]
pub struct HeurStats {
    pub total_repair_lp: i64,
    pub total_repair_lp_feasible: i64,
    pub total_repair_lp_iterations: i64,
    pub lp_iterations: i64,
    pub success_observations: f64,
    pub num_success_observations: i32,
    pub infeas_observations: f64,
    pub num_infeas_observations: i32,
    pub max_submip_level: i32,
    pub termination_status: i32,
}

/// A HighsMipWorker: its fields in place and its objects
#[repr(C)]
pub struct WorkerData {
    pub upper_limit: *mut f64,
    pub heur: *mut HeurStats,
    pub randgen: *mut HighsRandom,
    pub globaldom: P,
    pub lp: P,
    pub upper_bound: *mut f64,
    pub optimality_limit: *mut f64,
    pub state: *mut super::workers::WorkerState,
}

/// A HighsMipWorker
pub struct Worker {
    pub p: P,
    pub d: WorkerData,
}

impl Worker {
    pub fn new(p: P) -> Self {
        let mut d = WorkerData {
            upper_limit: std::ptr::null_mut(),
            heur: std::ptr::null_mut(),
            randgen: std::ptr::null_mut(),
            globaldom: std::ptr::null_mut(),
            lp: std::ptr::null_mut(),
            upper_bound: std::ptr::null_mut(),
            optimality_limit: std::ptr::null_mut(),
            state: std::ptr::null_mut(),
        };
        c!(worker_view, p, &mut d);
        Worker { p, d }
    }
    pub fn upper_limit(&self) -> f64 {
        // SAFETY: the worker's field
        unsafe { *self.d.upper_limit }
    }
    /// The worker's state
    #[allow(clippy::mut_from_ref)]
    pub fn state(&self) -> &mut super::workers::WorkerState {
        // SAFETY: the worker's Rust-owned state, used by its own thread;
        // no other reference is live across a use
        unsafe { &mut *self.d.state }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn heur(&self) -> &mut HeurStats {
        // SAFETY: the worker's statistics; no other reference is live
        // across a use
        unsafe { &mut *self.d.heur }
    }
    #[allow(clippy::mut_from_ref)]
    pub fn randgen(&self) -> &mut HighsRandom {
        // SAFETY: the worker's generator, used by one thread
        unsafe { &mut *self.d.randgen }
    }
    pub fn globaldom(&self) -> Dom {
        Dom::borrowed(self.d.globaldom)
    }
    pub fn terminated(&self) -> bool {
        self.heur().termination_status != MODEL_STATUS_NOTSET
    }
}

/// The column bounds of a domain (see [`Dom::bnd`])
#[derive(Clone, Copy)]
pub struct Bnd {
    pub lo: *const f64,
    pub up: *const f64,
}

impl Bnd {
    #[inline(always)]
    pub fn lo(&self, col: usize) -> f64 {
        // SAFETY: col below numCol, the domain is live
        unsafe { *self.lo.add(col) }
    }
    #[inline(always)]
    pub fn up(&self, col: usize) -> f64 {
        // SAFETY: as lo
        unsafe { *self.up.add(col) }
    }
}

/// A HighsDomain: owned (freed on drop) or borrowed
pub struct Dom {
    pub p: P,
    owned: bool,
}

impl Dom {
    pub fn borrowed(p: P) -> Dom {
        Dom { p, owned: false }
    }
    /// HighsDomain copy(*other)
    pub fn copy(other: &Dom) -> Dom {
        Dom { p: c!(dom_copy, other.p), owned: true }
    }
    /// *this = other
    pub fn assign(&self, other: &Dom) {
        c!(dom_assign, self.p, other.p)
    }
    /// col_lower_ and col_upper_, read through raw pointers since the
    /// domain changes under them (stable while the domain lives: the
    /// vectors keep numCol entries, and assignment reuses their storage)
    pub fn bnd(&self) -> Bnd {
        let (mut lo, mut up) = (std::ptr::null(), std::ptr::null());
        c!(dom_bounds, self.p, &mut lo, &mut up);
        Bnd { lo, up }
    }
    pub fn change_bound(&self, chg: DomChg, reason: Reason) {
        c!(dom_change_bound, self.p, chg, reason)
    }
    pub fn fix_col(&self, col: i32, val: f64, reason: Reason) {
        c!(dom_fix_col, self.p, col, val, reason)
    }
    pub fn propagate(&self) -> bool {
        c!(dom_propagate, self.p)
    }
    pub fn infeasible(&self) -> bool {
        c!(dom_infeasible, self.p)
    }
    pub fn backtrack(&self) -> DomChg {
        c!(dom_backtrack, self.p)
    }
    pub fn conflict_analysis(&self, w: &Worker) {
        c!(dom_conflict_analysis, self.p, w.p)
    }
    pub fn stack(&self) -> &[DomChg] {
        let mut n = 0;
        let p = c!(dom_stack, self.p, &mut n);
        // SAFETY: the stack's elements, until it changes
        unsafe { crate::ffi::sl(p, n) }
    }
    pub fn stack_len(&self) -> usize {
        let mut n = 0;
        c!(dom_stack, self.p, &mut n);
        n as usize
    }
    pub fn branch_depth(&self) -> i32 {
        c!(dom_branch_depth, self.p)
    }
    pub fn clear_changed_cols(&self) {
        c!(dom_clear_changed_cols, self.p)
    }
    pub fn clear_pool_propagation(&self) {
        c!(dom_clear_pool_propagation, self.p)
    }
    pub fn num_changed_cols(&self) -> i32 {
        c!(dom_num_changed_cols, self.p)
    }
}

impl Drop for Dom {
    fn drop(&mut self) {
        if self.owned {
            c!(dom_free, self.p)
        }
    }
}

/// A HighsLpRelaxation: owned (freed on drop) or borrowed
pub struct Lp {
    pub p: P,
    rs: *mut LpRelax,
    owned: bool,
}

impl Lp {
    pub fn borrowed(p: P) -> Lp {
        Lp { p, rs: c!(lp_rust, p), owned: false }
    }
    /// HighsLpRelaxation(other) for the worker
    pub fn copy(other: P, w: &Worker) -> Lp {
        let p = c!(lp_copy, other, w.p);
        Lp { p, rs: c!(lp_rust, p), owned: true }
    }
    /// HighsLpRelaxation(mipsolver) with loadModel, for the worker
    pub fn new(m: &MipData, w: &Worker) -> Lp {
        let p = c!(lp_new, m.mipsolver, w.p);
        Lp { p, rs: c!(lp_rust, p), owned: true }
    }
    fn sh(&self) -> &LpShared {
        // SAFETY: the Rust state of the live relaxation
        unsafe { &(*self.rs).sh }
    }
    /// The LP solver
    #[allow(clippy::mut_from_ref)]
    fn lph(&self) -> &mut LpHandle {
        // SAFETY: the live relaxation's LP solver, not otherwise borrowed
        // across these calls
        unsafe { (*self.rs).lph() }
    }
    pub fn status(&self) -> i32 {
        self.sh().status
    }
    pub fn objective(&self) -> f64 {
        self.sh().objective
    }
    pub fn num_lp_iterations(&self) -> i64 {
        self.sh().numlpiters
    }
    pub fn avg_solve_iters(&self) -> f64 {
        self.sh().avg_solve_iters
    }
    pub fn set_adjust_symmetric_branching_col(&self, adjust: bool) {
        // SAFETY: as sh
        unsafe { (*self.rs).sh.adjust_sym = adjust }
    }
    /// getFractionalIntegers (valid until the next solve)
    #[allow(clippy::mut_from_ref)]
    pub fn frac(&self) -> &mut [FracInt] {
        let sh = self.sh();
        // SAFETY: the Rust vector of fractional integers, sorted in place
        // by the heuristics as in the C++
        unsafe { crate::ffi::sl_mut(sh.frac, sh.num_frac) }
    }
    pub fn set_iteration_limit(&self, limit: i32) {
        self.lph().set_option("simplex_iteration_limit", OptValue::Int(limit));
    }
    /// changeColsBounds(0, numCol - 1, lo, up)
    pub fn change_cols_bounds(&self, lo: &[f64], up: &[f64]) {
        let n = self.lph().model.num_col;
        self.lph().change_col_bounds_interval(0, n - 1, lo, up);
    }
    /// changeColsBounds to a domain's bounds
    pub fn change_cols_bounds_dom(&self, d: &Dom) {
        let b = d.bnd();
        let n = self.lph().model.num_col;
        // SAFETY: the domain's bounds of the model's columns
        let (lo, up) = unsafe { (crate::ffi::sl(b.lo, n), crate::ffi::sl(b.up, n)) };
        self.lph().change_col_bounds_interval(0, n - 1, lo, up);
    }
    pub fn change_col_bounds(&self, col: i32, lo: f64, up: f64) {
        self.lph().change_col_bounds_set(&[col], &[lo], &[up]);
    }
    pub fn change_cols_cost(&self, mask: &[i32], cost: &[f64]) {
        self.lph().change_col_costs_mask(mask, cost);
    }
    /// 0 presolve off, 1 presolve on, 2 simplex_strategy primal, 3
    /// primal_simplex_bound_perturbation_multiplier 0
    pub fn set_option(&self, which: i32) {
        let h = self.lph();
        match which {
            0 => h.set_option("presolve", OptValue::Str(b"off")),
            1 => h.set_option("presolve", OptValue::Str(b"on")),
            // kSimplexStrategyPrimal
            2 => h.set_option("simplex_strategy", OptValue::Int(4)),
            _ => h.set_option("primal_simplex_bound_perturbation_multiplier", OptValue::Double(0.0)),
        };
    }
    /// setBasis(firstrootbasis, origin); `origin` is NUL-terminated
    pub fn set_root_basis(&self, origin: &[u8]) {
        c!(lp_set_root_basis, self.p, origin.as_ptr())
    }
    pub fn resolve(&self, dom: Option<&Dom>) -> i32 {
        c!(lp_resolve, self.p, dom.map_or(std::ptr::null_mut(), |d| d.p))
    }
    /// the LP solver's col_value (valid until the next solve)
    pub fn col_value(&self) -> &[f64] {
        &self.lph().solution().col_value
    }
    pub fn col_dual(&self) -> &[f64] {
        &self.lph().solution().col_dual
    }
    pub fn set_objective_limit(&self, lim: f64) {
        c!(lp_set_objective_limit, self.p, lim)
    }
    pub fn flush_domain(&self, d: &Dom) {
        c!(lp_flush_domain, self.p, d.p)
    }
    pub fn remove_obsolete_rows(&self, notify: bool) {
        c!(lp_remove_obsolete_rows, self.p, notify)
    }
    pub fn infeasible_conflict(&self, w: &Worker, localdom: &Dom) {
        c!(lp_infeasible_conflict, self.p, w.p, localdom.p)
    }
    pub fn put_iterate(&self) -> bool {
        self.lph().put_iterate() == crate::lp_data::Status::Ok
    }
    pub fn get_iterate(&self) {
        self.lph().get_iterate();
    }
}

impl Drop for Lp {
    fn drop(&mut self) {
        if self.owned {
            c!(lp_free, self.p)
        }
    }
}

/// A heuristic's HighsSearch with its own copy of the worker's
/// pseudocosts (freed on drop); the search itself is called in Rust
pub struct SearchH {
    pub parts: SearchParts,
}

impl SearchH {
    pub fn new(w: &Worker) -> SearchH {
        let mut parts = SearchParts {
            cpp: std::ptr::null_mut(),
            rs: std::ptr::null_mut(),
            ps: std::ptr::null_mut(),
            nq: std::ptr::null(),
            localdom: std::ptr::null_mut(),
        };
        c!(search_new, w.p, &mut parts);
        SearchH { parts }
    }
    pub fn set_lp(&self, lp: &Lp) {
        c!(search_set_lp, self.parts.cpp, lp.p)
    }
    pub fn localdom(&self) -> Dom {
        Dom::borrowed(self.parts.localdom)
    }
    /// The Rust search, entered with its pseudocosts and the node queue
    pub fn s(&mut self) -> &mut Search {
        // SAFETY: the live search; C++ does not use it during the call
        let s = unsafe { &mut *self.parts.rs };
        s.enter(self.parts.ps, self.parts.nq);
        s
    }
    pub fn stats(&self) -> &Stats {
        // SAFETY: as s
        unsafe { &(*self.parts.rs).stats }
    }
    pub fn set_min_reliable(&self, r: i32) {
        // SAFETY: the search's own pseudocost copy
        unsafe { (*self.parts.ps).minreliable = r }
    }
}

impl Drop for SearchH {
    fn drop(&mut self) {
        c!(search_free, self.parts.cpp)
    }
}

/// addIncumbent of the heuristics (through the worker under the parallel
/// lock)
pub fn add_incumbent(m: &MipData, w: &Worker, sol: &[f64], obj: f64, source: i32) -> bool {
    m.add_incumbent_any(w, sol, obj, source)
}

/// trySolution of the heuristics
pub fn try_solution(m: &MipData, w: &Worker, sol: &[f64], source: i32) -> bool {
    m.try_solution_any(w, sol, source)
}

/// The sub-MIP run of solveSubMip (see CMipFns::sub_mip): the options
/// that differ from the caller's and the start's row activities; abs_gap
/// NaN to keep the caller's gaps
#[allow(clippy::too_many_arguments)]
pub fn sub_mip(
    m: &MipData,
    w: &Worker,
    lp: Option<&Lp>,
    lo: &[f64],
    up: &[f64],
    maxleaves: i32,
    maxnodes: i32,
    stallnodes: i32,
    start: Option<&[f64]>,
    time_cap: f64,
    abs_gap: f64,
    sol: &mut [f64],
) -> SubMipResult {
    // the start's row activities (calculateRowValuesQuad, which needs a
    // column-wise model)
    let mut rows = Vec::new();
    if let (Some(start), true) = (start, m.colwise) {
        rows = vec![0.0; m.num_row as usize];
        crate::lp_data::edit::calculate_row_values_quad(m.a_start(), m.a_index(), m.a_value(), start, &mut rows);
    }
    let time_limit = m.opts.time_limit - m.timer_read();
    let spec = SubMipSpec {
        lp: lp.map_or(std::ptr::null_mut(), |l| l.p),
        col_lower: lo.as_ptr(),
        col_upper: up.as_ptr(),
        start_cols: start.map_or(std::ptr::null(), |s| s.as_ptr()),
        start_rows: rows.as_ptr(),
        num_start_rows: rows.len() as i32,
        mip_max_leaves: maxleaves,
        mip_max_nodes: maxnodes,
        mip_max_stall_nodes: stallnodes,
        mip_pscost_minreliable: 0,
        // std::min
        time_limit: if time_cap < time_limit { time_cap } else { time_limit },
        objective_bound: w.upper_limit(),
        mip_rel_gap: if abs_gap.is_nan() { f64::NAN } else { 0.0 },
        mip_abs_gap: abs_gap,
        mip_heuristic_effort: 0.8,
        // a concurrent helper runs without the heuristics that solve
        // sub-MIPs; its crossover sub-MIP gets the main solver's settings
        heur_flags: match (start, m.helper_lns()) {
            (Some(_), Some(p)) => p.run_rins as i32 | (p.run_rens as i32) << 1 | (p.run_root_reduced_cost as i32) << 2,
            _ => -1,
        },
        // with only root presolve allowed, none in the sub-MIP
        presolve: !m.root_presolve_only,
        output_flag: false,
        mip_detect_symmetry: false,
        lns_target: m.sub_mip_lns_target(),
    };
    let mut r = SubMipResult::default();
    c!(sub_mip, m.mipsolver, w.p, &spec, &mut r, sol.as_mut_ptr());
    r
}

/// The vectors of the scratch solution (see CMipFns::scratch_solution)
#[repr(C)]
pub struct ScratchView {
    pub col: *const f64,
    pub ncol: i32,
    pub row: *const f64,
    pub nrow: i32,
}

impl ScratchView {
    pub fn col(&self) -> &[f64] {
        // SAFETY: the C++ scratch solution, unchanged until the next call
        unsafe { crate::ffi::sl(self.col, self.ncol) }
    }
    pub fn row(&self) -> &[f64] {
        // SAFETY: as col
        unsafe { crate::ffi::sl(self.row, self.nrow) }
    }
}

/// transformNewIntegerFeasibleSolution's scratch solution: `sol` postsolved
/// (or the current one, after a repair)
pub fn scratch_solution(m: &MipData, sol: Option<&[f64]>) -> ScratchView {
    let mut v = ScratchView { col: std::ptr::null(), ncol: 0, row: std::ptr::null(), nrow: 0 };
    let (p, n) = sol.map_or((std::ptr::null(), -1), |s| (s.as_ptr(), s.len() as i32));
    c!(scratch_solution, m.mipsolver, p, n, &mut v);
    v
}

/// Refills the MipData (after a restart changed the model)
///
/// # Safety
/// `m` points to a MipData that no reference is held to
pub unsafe fn refill(m: *mut MipData) {
    c!(refill, (*m).mipsolver, m)
}

/// A MipData freshly filled for the solver of `m` (after its model
/// changed), for use while `m` is borrowed
pub fn fresh(m: &MipData) -> MipData {
    let mut out = std::mem::MaybeUninit::<MipData>::uninit();
    // SAFETY: refill writes every field of the MipData
    unsafe {
        (fns().refill)(m.mipsolver, out.as_mut_ptr());
        out.assume_init()
    }
}

/// The solver's vectors
#[allow(clippy::mut_from_ref)]
fn vecs(m: &MipData) -> &mut super::mip_data::MipVecs {
    // SAFETY: the solver's MipVecs; no reference to a vector is held
    // across a set
    unsafe { &mut *m.vecs }
}

/// Sets HighsMipSolverData's double vector `which`
pub fn set_vec(m: &MipData, which: i32, v: &[f64]) {
    vecs(m).set_f64(which, v)
}

/// Sets HighsMipSolverData's byte vector `which` (rowintegral)
pub fn set_bytes(m: &MipData, _which: i32, v: &[u8]) {
    vecs(m).set_u8(v)
}

/// Sets HighsMipSolverData's integer vector `which`
pub fn set_int_vec(m: &MipData, which: i32, v: &[i32]) {
    vecs(m).set_i32(which, v)
}

/// HighsMipSolverData's integer vector `which`
pub fn int_vec(m: &MipData, which: i32) -> &[i32] {
    vecs(m).int(which).as_slice()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts() {
        // HighsMipScalars (static_assert in HighsPrimalHeuristics.cpp)
        assert_eq!(std::mem::size_of::<MipScalars>(), 376);
        assert_eq!(std::mem::offset_of!(MipScalars, pdi), 320);
        assert_eq!(std::mem::size_of::<HeurStats>(), 72);
        // HighsMipWorker::RsState (static_assert in HighsMipWorker.cpp)
        use super::super::workers::WorkerState;
        assert_eq!(std::mem::offset_of!(WorkerState, heur), 24);
        assert_eq!(std::mem::offset_of!(WorkerState, randgen), 112);
        assert_eq!(std::mem::offset_of!(WorkerState, heuristics_allowed), 120);
    }
}

/// The concurrent helper's data (CMipFns::helper_new)
pub fn fns_call_helper_new(m: &MipData, time_left: f64) -> P {
    c!(helper_new, m.mipsolver, time_left)
}

/// A helper's root cut into the cut pool (CMipFns::add_root_cut)
pub fn add_root_cut(m: &MipData, index: &[i32], value: &[f64], rhs: f64, integral: bool) {
    c!(add_root_cut, m.mipsolver, index.as_ptr(), value.as_ptr(), index.len() as i32, rhs, integral)
}

/// The repair LP (see CMipFns::repair_lp)
#[allow(clippy::too_many_arguments)]
pub fn repair_lp(
    m: &MipData,
    lower: &[f64],
    upper: &[f64],
    time_limit: f64,
    feasibility_tolerance: f64,
    presolve: bool,
    iterations: &mut i64,
) -> bool {
    c!(repair_lp, m.mipsolver, lower.as_ptr(), upper.as_ptr(), time_limit, feasibility_tolerance, presolve, iterations)
}

/// The master worker
pub fn master_worker(m: &MipData) -> P {
    c!(master_worker, m.mipsolver)
}

/// runTask(processNode) over the indices
pub fn run_process_nodes(m: &MipData, indices: &[i32], ctx: *const std::ffi::c_void) {
    c!(run_process_nodes, m.mipsolver, indices.as_ptr(), indices.len() as i32, ctx)
}

/// The results of cleanupSolve into the HighsMipSolver
pub fn set_cleanup_result(m: &MipData, r: &super::driver::CleanupResult) {
    c!(set_cleanup_result, m.mipsolver, r as *const _ as *const std::ffi::c_void)
}

/// The original model's name
pub fn model_name(m: &MipData) -> String {
    let mut n = 0;
    let p = c!(model_name, m.mipsolver, &mut n);
    // SAFETY: the C++ string's bytes
    String::from_utf8_lossy(unsafe { crate::ffi::sl(p, n) }).into_owned()
}

/// HighsMipSolver::max_submip_level
pub fn max_submip_level(m: &MipData) -> i32 {
    c!(max_submip_level, m.mipsolver)
}
