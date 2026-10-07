//! The setup of HighsMipSolverData (highs/mip/HighsMipSolverData.cpp):
//! init, runMipPresolve, runSetup (the row-wise matrix, the locks, the
//! integral rows, the column classes and the model log), basisTransfer,
//! checkObjIntegrality, performRestart, the end of the analytic centre and
//! symmetry detection tasks, saveReportMipSolution, the user callbacks
//! (the C++ HighsCallback API is called through `CMipFns::callback`) and
//! queryExternalSolution. The C++ objects (presolve, the domain, the LP
//! relaxation, the pools, the postsolve stack, the symmetries, the
//! callback) are reached through CMipFns::op (codes from 300 below).

use super::domain::{DomChg, Reason, StdVec, LOWER, UPPER};
use super::glue::{self, fns, MipData};
use super::mip_data::{op as mop, src, status, vec};
use super::root::op as rop;
use crate::lp_data::LogType;
use crate::util::cdouble::CDouble;

const INF: f64 = f64::INFINITY;

/// The operations of the setup (CMipFns::op codes from 300)
pub mod op {
    /// postSolveStack.initializeIndexMaps and orig_model_ = model_
    pub const INIT_INDEX_MAPS: i32 = 300;
    /// the clique table and implications from the parent solver (sub-MIP),
    /// the clique table's parallelism threshold
    pub const INIT_TABLES: i32 = 301;
    /// mipsolver.timer_ presolve clock: start (i 0) / stop (i 1)
    pub const PRESOLVE_CLOCK: i32 = 302;
    /// HPresolve okSetInput and run with reduction limit i (sets the model
    /// status and presolve status)
    pub const RUN_PRESOLVE: i32 = 303;
    /// reportPresolveReductions
    pub const REPORT_PRESOLVE: i32 = 304;
    /// getLp().setSolvedFirstLp(false)
    pub const LP_NOT_SOLVED_FIRST: i32 = 305;
    /// incumbent = postSolveStack.getReducedPrimalSolution(solution_)
    pub const REDUCE_START_SOLUTION: i32 = 306;
    /// redcostfixing and the pseudocosts reset
    pub const RESET_REDCOST_PSEUDOCOST: i32 = 307;
    /// objectiveFunction.setupCliquePartition, then the global domain's
    /// setupObjectivePropagation, computeRowActivities and propagate
    pub const SETUP_PROPAGATION: i32 = 308;
    /// implications.cleanupVarbounds of the global domain's changed
    /// columns, then clearChangedCols
    pub const CLEANUP_VARBOUNDS: i32 = 309;
    /// getLp().getLpSolver() presolve off
    pub const LP_PRESOLVE_OFF: i32 = 310;
    /// objectiveFunction.checkIntegrality(epsilon)
    pub const OBJ_CHECK_INTEGRALITY: i32 = 311;
    /// heuristics.setupIntCols()
    pub const SETUP_INT_COLS: i32 = 312;
    /// the analytic centre (status, vector) and symmetries cleared
    pub const CLEAR_AC_SYMMETRIES: i32 = 313;
    /// cliquetable.getNumEntries()
    pub const CLIQUE_NUM_ENTRIES: i32 = 314;
    /// highs::parallel::num_threads(), std::thread::hardware_concurrency()
    pub const NUM_THREADS: i32 = 315;
    pub const HARDWARE_CONCURRENCY: i32 = 316;
    /// the restart's locals (root basis, pseudocost initialization, set as
    /// the solver's), created and freed (the solver's pointers to them
    /// cleared)
    pub const RESTART_CTX_NEW: i32 = 317;
    pub const RESTART_CTX_FREE: i32 = 318;
    /// the cuts (i of them) appended to the model, presolvedModel = the LP
    /// (with its integrality and offset kept)
    pub const RESTART_MODEL: i32 = 320;
    /// globalOrbits.reset()
    pub const RESET_GLOBAL_ORBITS: i32 = 321;
    /// postSolveStack.numReductions() (i 0), getOrigNumCol() (i 1),
    /// getOrigNumRow() (i 2)
    pub const NUM_REDUCTIONS: i32 = 322;
    /// postSolveStack.removeCutsFromModel(i)
    pub const REMOVE_CUTS: i32 = 323;
    /// the master worker's pools, domain, pseudocosts and bounds
    pub const RESTART_MASTER_WORKER: i32 = 324;
    /// the analytic centre task: taskGroup.sync(); analyticCenterStatus
    pub const AC_SYNC: i32 = 325;
    pub const AC_STATUS: i32 = 326;
    /// the symmetry detection task: sync, the symmetries taken (returns
    /// the detection time); counts: i 0 numGenerators, 1 numPerms, 2
    /// numOrbitopes, 3 numOrbitopeColumns; the end (symData reset,
    /// orbitope types, the stabilizer orbits if there are permutations)
    pub const SYM_SYNC: i32 = 327;
    pub const SYM_COUNT: i32 = 328;
    pub const SYM_END: i32 = 329;
    /// mip_improving_solution_save: the solution recorded
    pub const SAVE_IMPROVING_SOLUTION: i32 = 330;
    /// the improving solution file: objective and solution written
    pub const WRITE_IMPROVING_SOLUTION: i32 = 331;
    /// the reduced user solution (getReducedPrimalSolution) into the
    /// scratch solution's col_value
    pub const REDUCE_USER_SOLUTION: i32 = 332;
    /// the callback has a user callback (i 0) / the user has a solution
    /// (i 1)
    pub const CALLBACK_STATE: i32 = 333;
    /// model_ == &presolvedModel
    pub const MODEL_IS_PRESOLVED: i32 = 334;
    /// getPseudoCost() = HighsPseudocost(mipsolver)
    pub const RESET_PSEUDOCOST: i32 = 335;
    /// getDomain() = HighsDomain(mipsolver), computeRowActivities()
    pub const RESET_DOMAIN: i32 = 336;
    /// the callback's output cleared and its cut pool fields set from the
    /// LP relaxation's cuts
    pub const CUT_POOL_OUTPUT: i32 = 337;
}

/// HighsMipSolverData's vectors read through CMipFns::vec_ptr
pub mod vptr {
    pub const FIRSTROOTBASIS_COL: i32 = 0;
    pub const FIRSTROOTBASIS_ROW: i32 = 1;
    /// mipsolver.rootbasis (null if none)
    pub const ROOTBASIS_COL: i32 = 2;
    pub const ROOTBASIS_ROW: i32 = 3;
    /// postSolveStack's index maps
    pub const ORIG_COL_INDEX: i32 = 4;
    pub const ORIG_ROW_INDEX: i32 = 5;
    /// callback data_in.user_solution
    pub const USER_SOLUTION: i32 = 6;
    /// the scratch solution's col_value
    pub const SCRATCH_COL: i32 = 7;
}

/// The bases set through CMipFns::set_basis
pub mod basis {
    pub const FIRSTROOTBASIS: i32 = 0;
    /// the restart's root basis in the original space (also set as
    /// mipsolver.rootbasis)
    pub const RESTART_ROOT: i32 = 1;
}

/// HighsBasisStatus::kBasic
const BASIS_BASIC: u8 = 1;

/// Callback types (HighsCallbackType)
pub mod cb {
    pub const MIP_SOLUTION: i32 = 3;
    pub const MIP_IMPROVING_SOLUTION: i32 = 4;
    pub const MIP_LOGGING: i32 = 5;
    pub const MIP_INTERRUPT: i32 = 6;
    pub const MIP_GET_CUT_POOL: i32 = 7;
    pub const MIP_USER_SOLUTION: i32 = 9;
}

/// The solution passed as data_out.mip_solution (CMipFns::callback)
pub mod cbsol {
    pub const NONE: i32 = 0;
    /// mipsolver.solution_
    pub const SOLVER: i32 = 1;
    /// the scratch solution's col_value
    pub const SCRATCH: i32 = 2;
}

/// The callback's data_out of setCallbackDataOut (CMipFns::callback)
#[repr(C)]
pub struct CallbackOut {
    pub running_time: f64,
    pub objective_function_value: f64,
    pub mip_node_count: i64,
    pub mip_total_lp_iterations: i64,
    pub mip_primal_bound: f64,
    pub mip_dual_bound: f64,
    pub mip_gap: f64,
    /// data_out.external_solution_query_origin (-1: not set)
    pub external_solution_query_origin: i32,
    /// clearHighsCallbackOutput first / clearHighsCallbackInput first
    pub clear_output: bool,
    pub clear_input: bool,
    /// cbsol: data_out.mip_solution
    pub solution: i32,
}

/// fractionality(x)
#[inline(always)]
fn fractionality(x: f64) -> f64 {
    (x - x.round()).abs()
}

fn plural(n: i32) -> &'static str {
    if n == 1 {
        ""
    } else {
        "s"
    }
}

/// A C++ vector read through CMipFns::vec_ptr (None if it does not exist)
fn vec_opt<T: Copy>(m: &MipData, which: i32) -> Option<&[T]> {
    let mut n = 0;
    // SAFETY: the C++ operation on the live solver
    let p = unsafe { (fns().vec_ptr)(m.mipsolver, which, &mut n) } as *const T;
    if p.is_null() {
        None
    } else {
        // SAFETY: the vector's data and size, unchanged while the slice is
        // used
        Some(unsafe { crate::ffi::sl(p, n) })
    }
}

fn vec_of<T: Copy>(m: &MipData, which: i32) -> &[T] {
    vec_opt(m, which).unwrap_or(&[])
}

fn set_basis(m: &MipData, which: i32, col: &[u8], row: &[u8], valid: bool, alien: bool, useful: bool) {
    // SAFETY: the C++ operation on the live solver
    unsafe {
        (fns().set_basis)(
            m.mipsolver,
            which,
            col.as_ptr(),
            col.len() as i32,
            row.as_ptr(),
            row.len() as i32,
            valid,
            alien,
            useful,
        )
    }
}

impl MipData {
    // ---- the user callbacks ----

    /// callbackActive(type)
    pub fn callback_active(&self, callback_type: i32) -> bool {
        // SAFETY: the C++ operation on the live solver
        unsafe { (fns().callback)(self.mipsolver, callback_type, std::ptr::null(), std::ptr::null(), 0) }
    }

    /// setCallbackDataOut and callbackAction (interruptFromCallbackWithData
    /// when the callback is active; the output cleared and the solution set
    /// as `out` says). Returns the interrupt.
    fn callback_with_data(&self, callback_type: i32, objective: f64, message: &str, mut out: CallbackOut) -> bool {
        let (dual_bound, primal_bound, gap) = self.limits_to_bounds();
        let sc = self.sc();
        out.running_time = self.timer_read();
        out.objective_function_value = objective;
        out.mip_node_count = sc.num_nodes;
        out.mip_total_lp_iterations = sc.total_lp_iterations;
        out.mip_primal_bound = primal_bound;
        out.mip_dual_bound = dual_bound;
        out.mip_gap = gap;
        // SAFETY: the C++ operation on the live solver; `message` is passed
        // with its length
        unsafe { (fns().callback)(self.mipsolver, callback_type, &out, message.as_ptr(), message.len() as i32) }
    }

    fn out(clear_output: bool, solution: i32) -> CallbackOut {
        CallbackOut {
            running_time: 0.0,
            objective_function_value: 0.0,
            mip_node_count: 0,
            mip_total_lp_iterations: 0,
            mip_primal_bound: 0.0,
            mip_dual_bound: 0.0,
            mip_gap: 0.0,
            external_solution_query_origin: -1,
            clear_output,
            clear_input: false,
            solution,
        }
    }

    /// interruptFromCallbackWithData after clearHighsCallbackOutput (and
    /// data_out.mip_solution set from `solution`, see cbsol)
    pub fn callback_clear_and_call(&self, callback_type: i32, objective: f64, message: &str, solution: i32) -> bool {
        if !self.callback_active(callback_type) {
            return false;
        }
        self.callback_with_data(callback_type, objective, message, Self::out(true, solution))
    }

    /// The MIP interrupt callback of checkLimits
    pub fn user_interrupt(&self) -> bool {
        if self.o(op::CALLBACK_STATE) == 0.0 {
            return false;
        }
        let obj = self.sol_objective();
        self.callback_clear_and_call(cb::MIP_INTERRUPT, obj, "MIP check limits", cbsol::NONE)
    }

    /// The MIP logging callback of printDisplayLine
    pub fn logging_callback(&self) {
        let obj = self.sol_objective();
        let interrupt = self.callback_clear_and_call(cb::MIP_LOGGING, obj, "MIP logging", cbsol::NONE);
        debug_assert!(!interrupt);
    }

    /// The MIP solution callback with the scratch solution
    pub fn solution_callback(&self, objective: f64) {
        let interrupt = self.callback_clear_and_call(cb::MIP_SOLUTION, objective, "Feasible solution", cbsol::SCRATCH);
        debug_assert!(!interrupt);
    }

    /// The cut pool callback (HighsMipSolver::callbackGetCutPool) if active
    pub fn callback_get_cut_pool(&self) {
        if self.submip || self.o(op::CALLBACK_STATE) == 0.0 || !self.callback_active(cb::MIP_GET_CUT_POOL) {
            return;
        }
        self.o(op::CUT_POOL_OUTPUT);
        let obj = self.sol_objective();
        let interrupt = self.callback_with_data(cb::MIP_GET_CUT_POOL, obj, "MIP cut pool", Self::out(false, cbsol::NONE));
        debug_assert!(!interrupt);
    }

    /// saveReportMipSolution
    pub fn save_report_mip_solution(&self, new_upper_limit: f64) {
        let non_improving = new_upper_limit >= self.sc().upper_limit;
        if self.submip || non_improving {
            return;
        }
        if self.o(op::CALLBACK_STATE) != 0.0 {
            let obj = self.sol_objective();
            let interrupt =
                self.callback_clear_and_call(cb::MIP_IMPROVING_SOLUTION, obj, "Improving solution", cbsol::SOLVER);
            debug_assert!(!interrupt);
        }
        if self.opts.mip_improving_solution_save {
            self.o(op::SAVE_IMPROVING_SOLUTION);
        }
        self.o(op::WRITE_IMPROVING_SOLUTION);
    }

    /// queryExternalSolution
    pub fn query_external_solution(&self, objective: f64, origin: i32) {
        if !self.callback_active(cb::MIP_USER_SOLUTION) {
            return;
        }
        let mut out = Self::out(false, cbsol::NONE);
        out.external_solution_query_origin = origin;
        out.clear_input = true;
        let interrupt = self.callback_with_data(cb::MIP_USER_SOLUTION, objective, "MIP User solution", out);
        debug_assert!(!interrupt);
        if self.oi(op::CALLBACK_STATE, 1) == 0.0 {
            return;
        }
        // the user's solution in the original space: its violations and
        // objective (HighsMipSolver::solutionFeasible with the row values
        // computed)
        let user_solution: Vec<f64> = vec_of::<f64>(self, vptr::USER_SOLUTION).to_vec();
        let o = &self.orig;
        let mut row_value = vec![0.0; o.num_row.max(0) as usize];
        if o.num_row > 0 {
            // SAFETY: the original model's matrix, unchanged during the call
            let (start, index, value) = unsafe { ((*o.a_start).as_slice(), (*o.a_index).as_slice(), (*o.a_value).as_slice()) };
            crate::lp_data::edit::calculate_row_values_quad(start, index, value, &user_solution, &mut row_value);
        }
        let (feasible, bound_violation, row_violation, integrality_violation, quad_obj) =
            self.solution_feasible_orig(&user_solution, &row_value);
        let user_objective = quad_obj.to_f64();
        if !feasible {
            crate::log_user!(
                self.log,
                LogType::Warning,
                "User-supplied solution has with objective %g has violations: bound = %.4g; integrality = %.4g; row = %.4g\n",
                user_objective,
                bound_violation,
                integrality_violation,
                row_violation
            );
            return;
        }
        self.o(op::REDUCE_USER_SOLUTION);
        let reduced: Vec<f64> = vec_of::<f64>(self, vptr::SCRATCH_COL).to_vec();
        self.add_incumbent_rs(&reduced, user_objective, src::USER_SOLUTION, true, true);
    }

    // ---- init, presolve and setup ----

    /// HighsMipSolverData::init
    pub fn init_rs(&self) {
        self.o(op::INIT_INDEX_MAPS);
        let sc = self.sc();
        sc.feastol = self.opts.mip_feasibility_tolerance;
        sc.epsilon = self.opts.small_matrix_value;
        self.o(op::INIT_TABLES);
        let sc = self.sc();
        sc.heuristic_effort = self.opts.mip_heuristic_effort;
        sc.detect_symmetries = self.opts.mip_detect_symmetry;
        sc.firstlpsolobj = -INF;
        sc.rootlpsolobj = -INF;
        sc.analytic_center_computed = false;
        sc.max_tree_size_log2 = 0;
        sc.num_restarts = 0;
        sc.num_restarts_root = 0;
        sc.num_improving_sols = 0;
        sc.pruned_treeweight = CDouble::from(0.0);
        sc.avgrootlpiters = 0.0;
        sc.num_nodes = 0;
        sc.num_nodes_before_run = 0;
        sc.num_leaves = 0;
        sc.num_leaves_before_run = 0;
        sc.total_repair_lp = 0;
        sc.total_repair_lp_feasible = 0;
        sc.total_repair_lp_iterations = 0;
        sc.total_lp_iterations = 0;
        sc.heuristic_lp_iterations = 0;
        sc.sepa_lp_iterations = 0;
        sc.sb_lp_iterations = 0;
        sc.total_lp_iterations_before_run = 0;
        sc.heuristic_lp_iterations_before_run = 0;
        sc.sepa_lp_iterations_before_run = 0;
        sc.sb_lp_iterations_before_run = 0;
        sc.num_disp_lines = 0;
        sc.num_clique_entries_after_presolve = 0;
        sc.num_clique_entries_after_first_presolve = 0;
        sc.cliques_extracted = false;
        sc.row_matrix_set = false;
        sc.lower_bound = -INF;
        sc.upper_bound = INF;
        sc.upper_limit = self.opts.objective_bound;
        sc.optimality_limit = self.opts.objective_bound;
        sc.pdi.value = -INF;
        sc.dispfreq = match self.opts.mip_report_level {
            0 => 0,
            1 => 2000,
            _ => 100,
        };
    }

    /// runMipPresolve (the model changes: refill the MipData after it)
    pub fn run_mip_presolve(&self, presolve_reduction_limit: i32) {
        self.oi(op::PRESOLVE_CLOCK, 0);
        self.oi(op::RUN_PRESOLVE, presolve_reduction_limit as i64);
        self.oi(op::PRESOLVE_CLOCK, 1);
        // report the final presolve reductions unless this is a restart
        if !self.opts.presolve_off && self.sc().num_restarts == 0 {
            self.o(op::REPORT_PRESOLVE);
        }
    }

    /// The row-wise matrix ARstart_, ARindex_, ARvalue_
    /// (highsSparseTranspose of the model's matrix)
    fn set_row_matrix(&self) {
        let num_row = self.num_row as usize;
        let (a_start, a_index, a_value) = (self.a_start(), self.a_index(), self.a_value());
        let mut iwork = vec![0i32; num_row];
        for &r in a_index {
            iwork[r as usize] += 1;
        }
        let mut ar_start = vec![0i32; num_row + 1];
        for i in 1..=num_row {
            ar_start[i] = ar_start[i - 1] + iwork[i - 1];
        }
        iwork.copy_from_slice(&ar_start[..num_row]);
        let mut ar_index = vec![0i32; a_index.len()];
        let mut ar_value = vec![0.0f64; a_index.len()];
        for col in 0..self.num_col as usize {
            for k in a_start[col] as usize..a_start[col + 1] as usize {
                let r = a_index[k] as usize;
                let put = iwork[r] as usize;
                iwork[r] += 1;
                ar_index[put] = col as i32;
                ar_value[put] = a_value[k];
            }
        }
        glue::set_int_vec(self, vec::AR_START, &ar_start);
        glue::set_int_vec(self, vec::AR_INDEX, &ar_index);
        glue::set_vec(self, vec::AR_VALUE, &ar_value);
    }

    /// setupDomainPropagation (presolve's probing)
    pub fn setup_domain_propagation(&self) {
        self.set_row_matrix();
        self.o(op::RESET_PSEUDOCOST);
        let (ar_start, ar_value) = (self.ar_start(), self.ar_value());
        let max_abs_row_coef: Vec<f64> = (0..self.num_row as usize)
            .map(|i| {
                let mut maxabsval: f64 = 0.0;
                for &v in &ar_value[ar_start[i] as usize..ar_start[i + 1] as usize] {
                    if maxabsval < v.abs() {
                        maxabsval = v.abs();
                    }
                }
                maxabsval
            })
            .collect();
        glue::set_vec(self, vec::MAX_ABS_ROW_COEF, &max_abs_row_coef);
        self.o(op::RESET_DOMAIN);
    }

    /// basisTransfer
    pub fn basis_transfer(&self) {
        let Some(rcol) = vec_opt::<u8>(self, vptr::ROOTBASIS_COL) else {
            return;
        };
        let rrow = vec_of::<u8>(self, vptr::ROOTBASIS_ROW);
        let orig_col = vec_of::<i32>(self, vptr::ORIG_COL_INDEX);
        let orig_row = vec_of::<i32>(self, vptr::ORIG_ROW_INDEX);
        let row: Vec<u8> = (0..self.num_row as usize).map(|i| rrow[orig_row[i] as usize]).collect();
        let col: Vec<u8> = (0..self.num_col as usize).map(|i| rcol[orig_col[i] as usize]).collect();
        set_basis(self, basis::FIRSTROOTBASIS, &col, &row, true, true, true);
    }

    /// checkObjIntegrality
    pub fn check_obj_integrality(&self) {
        self.o(op::OBJ_CHECK_INTEGRALITY);
        let scale = self.o(mop::OBJ_INT_SCALE);
        if scale != 0.0 && self.sc().num_restarts == 0 {
            crate::log_user!(self.log, LogType::Info, "Objective function is integral with scale %g\n", scale);
        }
    }

    /// runSetup (the MipData is fresh for the presolved model)
    pub fn run_setup(&self) {
        let sc = self.sc();
        // the first LP has not been solved
        self.o(op::LP_NOT_SOLVED_FIRST);
        sc.last_disptime = -INF;
        sc.disptime = 0.0;
        // the objective limit and bounds in the presolved model's space
        let offset = self.offset;
        sc.upper_limit -= offset;
        sc.optimality_limit -= offset;
        sc.lower_bound -= offset;
        sc.upper_bound -= offset;

        let sol_objective = self.sol_objective();
        if sol_objective != INF {
            // the start solution as the incumbent
            self.o(op::REDUCE_START_SOLUTION);
            let sense = if self.orig_maximize { -1.0 } else { 1.0 };
            let solobj = sol_objective * sense - offset;
            let tol = self.opts.mip_feasibility_tolerance;
            let (bv, iv, rv) = self.violations();
            let feasible = bv <= tol && iv <= tol && rv <= tol;
            if sc.num_restarts == 0 {
                crate::log_user!(
                    self.log,
                    LogType::Info,
                    "\nMIP start solution is %s, objective value is %.12g\n",
                    if feasible { "feasible" } else { "infeasible" },
                    sol_objective
                );
            }
            if feasible && solobj < sc.upper_bound {
                let prev_upper_bound = sc.upper_bound;
                sc.upper_bound = solobj;
                if !self.submip && sc.upper_bound != prev_upper_bound {
                    let lb = sc.lower_bound;
                    self.update_primal_dual_integral(lb, lb, prev_upper_bound, sc.upper_bound, true, true);
                }
                let new_upper_limit = self.compute_new_upper_limit(solobj, 0.0, 0.0);
                self.save_report_mip_solution(new_upper_limit);
                if new_upper_limit < sc.upper_limit {
                    sc.upper_limit = new_upper_limit;
                    sc.optimality_limit =
                        self.compute_new_upper_limit(solobj, self.opts.mip_abs_gap, self.opts.mip_rel_gap);
                    self.nodequeue().set_optimality_limit(sc.optimality_limit);
                }
            }
            if !self.submip && feasible && self.callback_active(cb::MIP_SOLUTION) {
                let interrupt = self.callback_clear_and_call(cb::MIP_SOLUTION, sol_objective, "Feasible solution", cbsol::SOLVER);
                debug_assert!(!interrupt);
            }
        }

        if self.num_col == 0 {
            self.add_incumbent_rs(&[], 0.0, src::EMPTY_MIP, true, false);
        }

        self.o(op::RESET_REDCOST_PSEUDOCOST);
        let sc = self.sc();
        self.nodequeue().set_num_col(self.num_col);
        self.nodequeue().set_optimality_limit(sc.optimality_limit);

        for which in [vec::CONTINUOUS_COLS, vec::INTEGER_COLS, vec::IMPLINT_COLS, vec::INTEGRAL_COLS] {
            glue::set_int_vec(self, which, &[]);
        }

        let num_row = self.num_row as usize;
        let num_col = self.num_col as usize;
        let (a_start, a_index, a_value) = (self.a_start(), self.a_index(), self.a_value());
        // the row-wise matrix (highsSparseTranspose) and the locks
        sc.row_matrix_set = true;
        {
            self.set_row_matrix();
            let (row_lower, row_upper) = (self.row_lower(), self.row_upper());
            let mut uplocks = vec![0i32; num_col];
            let mut downlocks = vec![0i32; num_col];
            for col in 0..num_col {
                for j in a_start[col] as usize..a_start[col + 1] as usize {
                    let row = a_index[j] as usize;
                    let neg = a_value[j] < 0.0;
                    if row_lower[row] != -INF {
                        if neg {
                            uplocks[col] += 1;
                        } else {
                            downlocks[col] += 1;
                        }
                    }
                    if row_upper[row] != INF {
                        if neg {
                            downlocks[col] += 1;
                        } else {
                            uplocks[col] += 1;
                        }
                    }
                }
            }
            glue::set_int_vec(self, vec::UPLOCKS, &uplocks);
            glue::set_int_vec(self, vec::DOWNLOCKS, &downlocks);
        }

        // the maximal absolute coefficients (to filter propagation), and the
        // rows whose activity is integral (their sides rounded)
        debug_assert!(self.o(op::MODEL_IS_PRESOLVED) != 0.0);
        {
            let (ar_start, ar_index, ar_value) = (self.ar_start(), self.ar_index(), self.ar_value());
            let integrality = self.integrality();
            let (feastol, epsilon) = (sc.feastol, sc.epsilon);
            let mut rowintegral = vec![0u8; num_row];
            let mut max_abs_row_coef = vec![0.0f64; num_row];
            // SAFETY: the presolved model's row sides (model_ is
            // presolvedModel), changed in place as by the C++; no slice of
            // them is live
            let (rl, ru) = unsafe { ((*(self.row_lower as *mut StdVec<f64>)).as_mut_slice(), (*(self.row_upper as *mut StdVec<f64>)).as_mut_slice()) };
            for i in 0..num_row {
                let mut maxabsval: f64 = 0.0;
                let mut integral = true;
                for j in ar_start[i] as usize..ar_start[i + 1] as usize {
                    integral = integral && integrality[ar_index[j] as usize] != 0 && fractionality(ar_value[j]) <= epsilon;
                    let a = ar_value[j].abs();
                    if maxabsval < a {
                        maxabsval = a;
                    }
                }
                if integral {
                    if rl[i] != -INF {
                        rl[i] = (rl[i] - feastol).ceil();
                    }
                    if ru[i] != INF {
                        ru[i] = (ru[i] + feastol).floor();
                    }
                }
                rowintegral[i] = integral as u8;
                max_abs_row_coef[i] = maxabsval;
            }
            glue::set_bytes(self, vec::ROW_INTEGRAL, &rowintegral);
            glue::set_vec(self, vec::MAX_ABS_ROW_COEF, &max_abs_row_coef);
        }

        // row activities, and all rows propagated once
        self.o(op::SETUP_PROPAGATION);
        let sc = self.sc();
        if self.domain().infeasible() {
            self.set_modelstatus(status::INFEASIBLE);
            self.update_lower_bound_ex(INF, true, true);
            sc.pruned_treeweight = CDouble::from(1.0);
            return;
        }
        if num_col == 0 {
            self.set_modelstatus(status::OPTIMAL);
            return;
        }
        if self.check_limits_rs(0) {
            return;
        }
        self.o(op::CLEANUP_VARBOUNDS);
        self.o(op::LP_PRESOLVE_OFF);
        self.check_obj_integrality();
        glue::set_vec(self, vec::ROOTLPSOL, &[]);
        glue::set_vec(self, vec::FIRSTLPSOL, &[]);

        // the column classes
        let sc = self.sc();
        let b = self.domain().bnd();
        let (integrality, col_lower, col_upper) = (self.integrality(), self.col_lower(), self.col_upper());
        let mut continuous_cols = Vec::new();
        let mut implint_cols = Vec::new();
        let mut integer_cols = Vec::new();
        let mut integral_cols = Vec::new();
        let mut num_binary: i32 = 0;
        let mut num_domain_fixed: i32 = 0;
        sc.max_tree_size_log2 = 0;
        for i in 0..num_col {
            let fixed = b.lo(i) == b.up(i);
            match integrality[i] {
                0 => {
                    if fixed {
                        num_domain_fixed += 1;
                        continue;
                    }
                    continuous_cols.push(i as i32);
                }
                4 => {
                    if fixed {
                        num_domain_fixed += 1;
                        continue;
                    }
                    implint_cols.push(i as i32);
                    integral_cols.push(i as i32);
                }
                1 => {
                    if fixed {
                        num_domain_fixed += 1;
                        if fractionality(b.lo(i)) > sc.feastol {
                            // integer column fixed at a fractional value
                            self.set_modelstatus(status::INFEASIBLE);
                            self.update_lower_bound_ex(INF, true, true);
                            sc.pruned_treeweight = CDouble::from(1.0);
                            for (which, v) in [
                                (vec::CONTINUOUS_COLS, &continuous_cols),
                                (vec::INTEGER_COLS, &integer_cols),
                                (vec::IMPLINT_COLS, &implint_cols),
                                (vec::INTEGRAL_COLS, &integral_cols),
                            ] {
                                glue::set_int_vec(self, which, v);
                            }
                            return;
                        }
                        continue;
                    }
                    integer_cols.push(i as i32);
                    integral_cols.push(i as i32);
                    let range = 1.0 + col_upper[i] - col_lower[i];
                    sc.max_tree_size_log2 += if range < 1024.0 { range } else { 1024.0f64 }.log2().ceil() as i32;
                    num_binary += (col_lower[i] == 0.0 && col_upper[i] == 1.0) as i32;
                }
                _ => {
                    crate::log_user!(
                        self.log,
                        LogType::Error,
                        "Semicontinuous or semiinteger variables should have been reformulated away before HighsMipSolverData::runSetup() is called."
                    );
                    panic!("Unexpected variable type");
                }
            }
        }
        for (which, v) in [
            (vec::CONTINUOUS_COLS, &continuous_cols),
            (vec::INTEGER_COLS, &integer_cols),
            (vec::IMPLINT_COLS, &implint_cols),
            (vec::INTEGRAL_COLS, &integral_cols),
        ] {
            glue::set_int_vec(self, which, v);
        }

        self.basis_transfer();

        let sc = self.sc();
        sc.numintegercols = integer_cols.len() as i32;
        sc.detect_symmetries = sc.detect_symmetries && num_binary > 0;
        sc.num_clique_entries_after_presolve = self.o(op::CLIQUE_NUM_ENTRIES) as i32;
        let num_col = self.num_col;
        let num_general_integer = sc.numintegercols - num_binary;
        let num_implied_integer = implint_cols.len() as i32;
        let num_continuous = continuous_cols.len() as i32;
        let num_row = self.num_row;
        let nnz = self.a_start()[num_col as usize];
        if sc.num_restarts == 0 {
            sc.num_clique_entries_after_first_presolve = sc.num_clique_entries_after_presolve;
            let max_workers = self.op(super::driver::op::MAX_NUM_WORKERS, None, 0, 0.0) as i32;
            crate::log_user!(
                self.log,
                LogType::Info,
                "\nSolving MIP model with:\n   %d row%s\n   %d col%s (%d binary, %d integer, %d implied int., %d continuous, %d domain fixed)\n   %d nonzero%s\n   Thread count %d (of %d threads). Using %d max workers. Parallel search %s\n",
                num_row,
                plural(num_row),
                num_col,
                plural(num_col),
                num_binary,
                num_general_integer,
                num_implied_integer,
                num_continuous,
                num_domain_fixed,
                nnz,
                plural(nnz),
                self.o(op::NUM_THREADS) as i32,
                self.o(op::HARDWARE_CONCURRENCY) as i32,
                max_workers,
                if max_workers > 1 { "on" } else { "off" }
            );
        } else {
            crate::log_user!(
                self.log,
                LogType::Info,
                "Model after restart has %d row%s, %d col%s (%d bin., %d int., %d impl., %d cont., %d dom.fix.), and %d nonzero%s\n",
                num_row,
                plural(num_row),
                num_col,
                plural(num_col),
                num_binary,
                num_general_integer,
                num_implied_integer,
                num_continuous,
                num_domain_fixed,
                nnz,
                plural(nnz)
            );
        }

        self.o(op::SETUP_INT_COLS);

        let sc = self.sc();
        if sc.upper_limit == INF {
            sc.analytic_center_computed = false;
        }
        self.o(op::CLEAR_AC_SYMMETRIES);
        if sc.num_restarts != 0 {
            crate::log_user!(self.log, LogType::Info, "\n");
        }
    }

    // ---- the tasks of the root node ----

    /// finishAnalyticCenterComputation
    pub fn finish_analytic_center(&self) {
        let mip_timing = self.o(rop::PROF_MIP) != 0.0;
        if mip_timing {
            crate::log_user!(
                self.log,
                LogType::Info,
                "MIP-Timing: %11.2g - starting  analytic centre synch\n",
                self.timer_read()
            );
        }
        self.o(op::AC_SYNC);
        if mip_timing {
            crate::log_user!(
                self.log,
                LogType::Info,
                "MIP-Timing: %11.2g - completed analytic centre synch\n",
                self.timer_read()
            );
        }
        self.sc().analytic_center_computed = true;
        if self.o(op::AC_STATUS) as i32 != status::OPTIMAL {
            return;
        }
        let gd = self.domain();
        let b = gd.bnd();
        let ac = self.analytic_center();
        let (col_lower, col_upper, integrality) = (self.col_lower(), self.col_upper(), self.integrality());
        let feastol = self.sc().feastol;
        let mut nfixed: i32 = 0;
        let mut nintfixed: i32 = 0;
        for i in 0..self.num_col as usize {
            let bound_range = b.up(i) - b.lo(i);
            if bound_range == 0.0 {
                continue;
            }
            let tolerance = feastol * if bound_range < 1.0 { bound_range } else { 1.0 };
            if ac[i] <= col_lower[i] + tolerance {
                gd.change_bound(DomChg { boundval: col_lower[i], column: i as i32, boundtype: UPPER }, Reason::UNSPECIFIED);
                if gd.infeasible() {
                    return;
                }
                nfixed += 1;
                if integrality[i] == 1 {
                    nintfixed += 1;
                }
            } else if ac[i] >= col_upper[i] - tolerance {
                gd.change_bound(DomChg { boundval: col_upper[i], column: i as i32, boundtype: LOWER }, Reason::UNSPECIFIED);
                if gd.infeasible() {
                    return;
                }
                nfixed += 1;
                if integrality[i] == 1 {
                    nintfixed += 1;
                }
            }
        }
        if nfixed > 0 {
            crate::log_dev!(
                self.log,
                LogType::Info,
                "Fixing %d columns (%d integers) sitting at bound at analytic center\n",
                nfixed,
                nintfixed
            );
        }
        gd.propagate();
    }

    /// finishSymmetryDetection
    pub fn finish_symmetry_detection(&self) {
        let detection_time = self.o(op::SYM_SYNC);
        let symmetry_time = if self.opts.timeless_log { String::new() } else { crate::sprintf!(" %.1fs", detection_time) };
        crate::log_user!(self.log, LogType::Info, "\nSymmetry detection completed in%s\n", &symmetry_time);
        let num_generators = self.oi(op::SYM_COUNT, 0) as i32;
        let num_perms = self.oi(op::SYM_COUNT, 1) as i32;
        let num_orbitopes = self.oi(op::SYM_COUNT, 2) as i32;
        if num_generators == 0 {
            self.sc().detect_symmetries = false;
            crate::log_user!(self.log, LogType::Info, "No symmetry present\n\n");
        } else if num_orbitopes == 0 {
            crate::log_user!(self.log, LogType::Info, "Found %d generator(s)\n\n", num_generators);
        } else {
            let num_orbitope_columns = self.oi(op::SYM_COUNT, 3) as i32;
            if num_perms != 0 {
                crate::log_user!(
                    self.log,
                    LogType::Info,
                    "Found %d generator(s) and %d full orbitope(s) acting on %d columns\n\n",
                    num_perms,
                    num_orbitopes,
                    num_orbitope_columns
                );
            } else {
                crate::log_user!(
                    self.log,
                    LogType::Info,
                    "Found %d full orbitope(s) acting on %d columns\n\n",
                    num_orbitopes,
                    num_orbitope_columns
                );
            }
        }
        self.o(op::SYM_END);
    }
}

/// performRestart (the model changes: the caller's MipData must be
/// refilled after it)
pub fn perform_restart(md: &MipData) {
    // the helper's solutions would be for the model before the restart
    md.concurrent_final_sync();
    md.stop_concurrent_lns();
    md.o(op::RESTART_CTX_NEW);
    let sc = md.sc();
    sc.num_restarts += 1;
    sc.num_leaves_before_run = sc.num_leaves;
    sc.num_nodes_before_run = sc.num_nodes;
    sc.total_lp_iterations_before_run = sc.total_lp_iterations;
    sc.heuristic_lp_iterations_before_run = sc.heuristic_lp_iterations;
    sc.sepa_lp_iterations_before_run = sc.sepa_lp_iterations;
    sc.sb_lp_iterations_before_run = sc.sb_lp_iterations;
    let num_lp_rows = md.o(mop::LP_NUM_ROWS) as i32;
    let num_cuts = num_lp_rows - md.num_row;
    md.oi(op::RESTART_MODEL, num_cuts as i64);

    // a basis after the root LP, expanded to the original space for the
    // starting basis of the presolved model after the restart
    let frb_col = vec_of::<u8>(md, vptr::FIRSTROOTBASIS_COL);
    if md.o(rop::FIRSTROOTBASIS_VALID) != 0.0 {
        let frb_row = vec_of::<u8>(md, vptr::FIRSTROOTBASIS_ROW);
        let orig_col = vec_of::<i32>(md, vptr::ORIG_COL_INDEX);
        let orig_row = vec_of::<i32>(md, vptr::ORIG_ROW_INDEX);
        let orig_num_col = md.oi(op::NUM_REDUCTIONS, 1) as usize;
        let orig_num_row = md.oi(op::NUM_REDUCTIONS, 2) as usize;
        let mut col = vec![0u8; orig_num_col];
        let mut row = vec![BASIS_BASIC; orig_num_row];
        for i in 0..md.num_col as usize {
            col[orig_col[i] as usize] = frb_col[i];
        }
        for (i, &s) in frb_row.iter().enumerate() {
            row[orig_row[i] as usize] = s;
        }
        set_basis(md, basis::RESTART_ROOT, &col, &row, true, true, true);
    }

    // the objective limit and bounds in the original model's space (the
    // offset generally changes in presolve)
    let offset = md.offset;
    sc.upper_limit += offset;
    sc.optimality_limit += offset;
    sc.upper_bound += offset;
    sc.lower_bound += offset;

    // the incumbent is kept in the original space
    glue::set_vec(md, vec::INCUMBENT, &[]);
    sc.pruned_treeweight = CDouble::from(0.0);
    md.nodequeue().clear();
    md.o(op::RESET_GLOBAL_ORBITS);

    // presolve on the presolved model: the number of further reductions
    // is limited by restart_presolve_reduction_limit
    let num_reductions = md.o(op::NUM_REDUCTIONS) as i32;
    let restart_limit = md.opts.restart_presolve_reduction_limit;
    debug_assert!(restart_limit != 0);
    let further_limit = if restart_limit >= 0 { num_reductions + restart_limit } else { -1 };
    md.run_mip_presolve(further_limit);
    let fresh = glue::fresh(md);
    let md = &fresh;
    let sc = md.sc();

    if md.modelstatus() != status::NOTSET {
        // the objective limit in the current model's space
        let offset = md.offset;
        sc.upper_limit -= offset;
        sc.optimality_limit -= offset;
        if md.modelstatus() == status::OPTIMAL {
            sc.upper_bound = 0.0;
            md.transform_new_integer_feasible_solution(&[], true);
        } else {
            sc.upper_bound -= offset;
        }
        // lower_bound still relates to the original model
        sc.lower_bound -= offset;
        md.update_lower_bound_ex(sc.upper_bound, true, md.modelstatus() != status::OPTIMAL);
        if md.sol_objective() != INF && md.modelstatus() == status::INFEASIBLE {
            md.set_modelstatus(status::OPTIMAL);
        }
        md.o(op::RESTART_CTX_FREE);
        return;
    }
    md.run_setup();
    if md.o(rop::TERMINATE) != 0.0 {
        md.o(op::RESTART_CTX_FREE);
        return;
    }
    md.oi(op::REMOVE_CUTS, num_cuts as i64);
    md.o(op::RESTART_MASTER_WORKER);
    md.o(op::RESTART_CTX_FREE);
}

/// HighsMipSolverData::setupDomainPropagation
///
/// # Safety
/// `f` the C++ functions, `m` filled for this call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_setup_domain_propagation(f: *const glue::CMipFns, m: *const MipData) {
    glue::set_fns(f);
    (*m).setup_domain_propagation();
}

/// HighsMipSolver::runMipPresolve (presolve only, from Highs::runPresolve):
/// init and runMipPresolve
///
/// # Safety
/// `f` the C++ functions, `m` filled for this call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_mip_presolve_only(f: *const glue::CMipFns, m: *const MipData, limit: i32) {
    glue::set_fns(f);
    (*m).init_rs();
    (*m).run_mip_presolve(limit);
}
