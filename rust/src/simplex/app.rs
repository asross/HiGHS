//! solveLpSimplex (simplex/HApp.h) on the LP of the simplex engine
//! ([`LpSolver::lp`]): the LP being solved (a C++ HighsLp, the
//! "incumbent") is copied in, scaled (considerScaling), dualized if asked
//! to, solved, unscaled and possibly solved again unscaled; its scale
//! factors go back to the incumbent, which is never moved or scaled. The
//! solution, basis and HighsInfo of the C++ solver object are written in
//! place (`CSimplexApp`); the steps on the C++ HEkk shell (its solve with
//! the analysis and simplex stats, setBasis from a HighsBasis, the proof
//! of infeasibility, the profiling clocks) are `op`s of
//! highs/simplex/HAppRust.cpp. Dualize and undualize (HEkk.cpp) are here
//! too, on the same LP.

use super::lp_solver::LpSolver;
use crate::lp_data::basis::refine_basis;
use crate::lp_data::ffi::{CLp, RsMut, RsVec};
use crate::lp_data::form_basis::accommodate_alien_basis;
use crate::lp_data::lp::Lp;
use crate::lp_data::ffi::CLpOptions;
use crate::lp_data::lp_utils::{self, SCALE_CHOOSE, SCALE_OFF};
use crate::lp_data::report::unscale_solution;
use crate::lp_data::run::{status_from_model_status, MS_INFEASIBLE, MS_NOTSET, MS_OBJECTIVE_BOUND, MS_OBJECTIVE_TARGET, MS_OPTIMAL, MS_UNBOUNDED, MS_UNBOUNDED_OR_INFEASIBLE, MS_UNKNOWN};
use crate::lp_data::solution::Info;
use crate::lp_data::solve::{reset_model_status_and_info, set_solution_status};
use crate::lp_data::sparse::SparseMatrix;
use crate::lp_data::{matrix_format, Log, LogType, Status, INF};
use crate::util::fma::ClangFma;
use crate::{log_dev, log_user};
use std::ffi::c_void;

const STRATEGY_CHOOSE: i32 = 0;
const STRATEGY_DUAL: i32 = 1;
const STRATEGY_DUAL_TASKS: i32 = 2;
const STRATEGY_DUAL_MULTI: i32 = 3;
const STRATEGY_PRIMAL: i32 = 4;
const UNSCALED_NONE: i32 = 0;
const UNSCALED_REFINE: i32 = 1;
const UNSCALED_DIRECT: i32 = 2;
const OPTION_CHOOSE: i32 = 0;
const OPTION_ON: i32 = 1;
const EDGE_WEIGHT_DEVEX: i32 = 1;
const ILLEGAL_COUNT: i32 = -1;
const NO_RAY_INDEX: i32 = -1;
/// SimplexAlgorithm::kDual
const ALGORITHM_DUAL: i32 = 1;
const BASIS_VALIDITY_VALID: i32 = 1;

// The ops of HAppRust.cpp
/// Start the sub-solver profiling clock of the solve
const OP_PROFILE_START: i32 = 1;
/// returnFromSolveLpSimplex's stop of the running clock
const OP_PROFILE_STOP: i32 = 2;
/// ekk.initialiseSimplexStats()
const OP_STATS_INIT: i32 = 3;
/// ekk.moveLp(solver_object) after the copy of the LP: the pointers and
/// the checks of the Rust move_lp
const OP_MOVE_LP: i32 = 4;
/// ekk.setBasis(basis): the HighsStatus
const OP_SET_BASIS: i32 = 5;
/// ekk.solve(arg != 0): the HighsStatus
const OP_SOLVE: i32 = 6;
/// ekk.clear()
const OP_CLEAR: i32 = 7;
/// ekk.proofOfPrimalInfeasibility()
const OP_PROOF: i32 = 8;
/// ekk.setNlaPointersForLpAndScale(incumbent_lp)
const OP_NLA_INCUMBENT: i32 = 9;
/// The incumbent takes the solver LP's scale (and with arg != 0 its
/// matrix)
const OP_LP_BACK: i32 = 10;
/// The HEkk's LpsEnv for a call into `p` (a *mut LpsEnv)
const OP_ENV: i32 = 11;
/// basis.debug_origin_name of the solver's basis record
const OP_BASIS_ORIGIN: i32 = 12;

/// HighsLpSolverObject for solveLpSimplex: the C++ data in place and
/// the steps on the C++ objects (HAppRust.cpp)
#[repr(C)]
pub struct CSimplexApp {
    pub log: Log,
    /// The log of an HFactor (accommodateAlienBasis)
    pub factor_log: Log,
    pub ctx: *mut c_void,
    pub op: unsafe extern "C" fn(*mut c_void, i32, i64, *mut c_void) -> i64,
    pub lps: *mut LpSolver,
    /// The incumbent LP and its model name
    pub incumbent: CLp,
    pub model_name: RsMut<u8>,
    pub model_status: *mut i32,
    pub info: *mut Info,
    pub value_valid: *mut bool,
    pub dual_valid: *mut bool,
    pub col_value: RsVec<f64>,
    pub col_dual: RsVec<f64>,
    pub row_value: RsVec<f64>,
    pub row_dual: RsVec<f64>,
    pub basis_valid: *mut bool,
    pub basis_alien: *mut bool,
    pub basis_useful: *mut bool,
    pub basis_was_alien: *mut bool,
    pub basis_debug_id: *mut i32,
    pub basis_debug_update_count: *mut i32,
    pub col_status: RsVec<u8>,
    pub row_status: RsVec<u8>,
    /// options.simplex_strategy (changed and restored here)
    pub simplex_strategy: *mut i32,
    pub dual_simplex_cost_perturbation_multiplier: *mut f64,
    pub simplex_unscaled_solution_strategy: i32,
    pub cost_scale_factor: i32,
    pub simplex_dualize_strategy: i32,
    pub simplex_permute_strategy: i32,
    pub lp_options: CLpOptions,
}

impl CSimplexApp {
    fn op(&self, code: i32, arg: i64, p: *mut c_void) -> i64 {
        // SAFETY: the host's op with its context
        unsafe { (self.op)(self.ctx, code, arg, p) }
    }
    fn op0(&self, code: i32, arg: i64) -> i64 {
        self.op(code, arg, std::ptr::null_mut())
    }
    fn status(&self, code: i32, arg: i64) -> Status {
        status_of(self.op0(code, arg))
    }
    /// The engine, borrowed between ops (an op may re-enter it)
    #[allow(clippy::mut_from_ref)]
    fn lps<'a>(&self) -> &'a mut LpSolver {
        // SAFETY: the C++ HEkk's engine; no borrow is held across an op
        unsafe { &mut *self.lps }
    }
    fn info<'a>(&self) -> &'a mut Info {
        // SAFETY: the solver object's HighsInfo
        unsafe { &mut *self.info }
    }
    fn model_status<'a>(&self) -> &'a mut i32 {
        // SAFETY: the solver object's model status
        unsafe { &mut *self.model_status }
    }
    fn options_simplex_strategy<'a>(&self) -> &'a mut i32 {
        // SAFETY: options.simplex_strategy
        unsafe { &mut *self.simplex_strategy }
    }
    /// The HEkk's environment of a call (with the engine's LP)
    fn env(&self) -> super::lp_solver::LpsEnv {
        let mut env = std::mem::MaybeUninit::<super::lp_solver::LpsEnv>::zeroed();
        self.op(OP_ENV, 0, env.as_mut_ptr() as *mut c_void);
        // SAFETY: filled by HEkk::rsEnv
        let env = unsafe { env.assume_init() };
        self.lps().env_of(&env)
    }

    /// moveBackLpAndUnapplyScaling: the LP is unscaled and the incumbent
    /// takes its scale (and its matrix after a dualized solve)
    fn move_back(&self, matrix: bool) {
        self.lps().lp.unapply_scale();
        self.op0(OP_LP_BACK, matrix as i64);
    }

    /// solution = ekk_instance.getSolution()
    fn get_solution(&mut self) {
        let env = self.env();
        let (nc, nr) = (env.lp.num_col as usize, env.lp.num_row as usize);
        self.col_value.resize(0);
        self.col_value.resize(nc);
        self.col_dual.resize(0);
        self.col_dual.resize(nc);
        self.row_value.resize(0);
        self.row_value.resize(nr);
        self.row_dual.resize(0);
        self.row_dual.resize(nr);
        self.lps().get_solution(
            &env,
            self.col_value.as_mut_slice(),
            self.col_dual.as_mut_slice(),
            self.row_value.as_mut_slice(),
            self.row_dual.as_mut_slice(),
        );
        // SAFETY: the solution's flags
        unsafe {
            *self.value_valid = true;
            *self.dual_valid = true;
        }
    }

    /// basis = ekk_instance.getHighsBasis(ekk_lp)
    fn get_highs_basis(&mut self) {
        let lps = self.lps();
        let lp = lps.lp.view();
        let (nc, nr) = (lp.num_col as usize, lp.num_row as usize);
        self.col_status.resize(0);
        self.col_status.resize(nc);
        self.row_status.resize(0);
        self.row_status.resize(nr);
        let sense = lps.lp.sense;
        let (id, count) = lps.get_highs_basis(&lp, sense, self.col_status.as_mut_slice(), self.row_status.as_mut_slice());
        // SAFETY: the basis' scalars
        unsafe {
            *self.basis_debug_id = id;
            *self.basis_debug_update_count = count;
            *self.basis_valid = true;
            *self.basis_alien = false;
            *self.basis_useful = true;
            *self.basis_was_alien = false;
        }
        let origin = &lps.basis.debug_origin_name;
        self.op(OP_BASIS_ORIGIN, origin.len() as i64, origin.as_ptr() as *mut c_void);
    }

    /// returnFromSolveLpSimplex
    fn return_from(&self, return_status: Status) -> Status {
        self.info().simplex_iteration_count = self.lps().sh.iteration_count;
        self.op0(OP_PROFILE_STOP, 0);
        if return_status == Status::Error {
            self.op0(OP_CLEAR, 0);
            return return_status;
        }
        self.op0(OP_NLA_INCUMBENT, 0);
        if *self.model_status() == MS_OPTIMAL {
            let info = self.info();
            info.num_complementarity_violations = 0;
            info.max_complementarity_violation = 0.0;
        }
        return_status
    }
}

fn status_of(v: i64) -> Status {
    match v {
        0 => Status::Ok,
        1 => Status::Warning,
        _ => Status::Error,
    }
}

/// simplexStrategyToString
pub fn simplex_strategy_to_string(simplex_strategy: i32) -> &'static str {
    match simplex_strategy {
        STRATEGY_CHOOSE => "choose simplex solver",
        STRATEGY_DUAL => "serial dual simplex solver",
        STRATEGY_DUAL_TASKS => "parallel dual simplex solver - SIP",
        STRATEGY_DUAL_MULTI => "parallel dual simplex solver - PAMI",
        STRATEGY_PRIMAL => "primal simplex solver",
        _ => "Unknown",
    }
}

/// considerScaling on the engine's LP: whether new scaling was found
pub fn consider_scaling(o: &CLpOptions, lp: &mut Lp) -> bool {
    let allow_scaling = lp.num_col > 0 && o.simplex_scale_strategy != SCALE_OFF;
    if lp.scale.has_scaling && !allow_scaling {
        lp.clear_scale();
        return true;
    }
    let scaling_not_tried = lp.scale.strategy == SCALE_OFF;
    let new_scaling_strategy = o.simplex_scale_strategy != lp.scale.strategy && o.simplex_scale_strategy != SCALE_CHOOSE;
    let try_scaling = allow_scaling && (scaling_not_tried || new_scaling_strategy);
    let mut new_scaling = false;
    if try_scaling {
        lp.unapply_scale();
        scale_lp(o, lp, false);
        new_scaling = lp.is_scaled;
    } else if lp.scale.has_scaling {
        lp.apply_scale();
    }
    new_scaling
}

/// scaleLp on the engine's LP
pub fn scale_lp(o: &CLpOptions, lp: &mut Lp, force_scaling: bool) {
    lp.clear_scaling();
    let (num_col, num_row) = (lp.num_col as usize, lp.num_row as usize);
    lp.scale.col.clear();
    lp.scale.col.resize(num_col, 1.0);
    lp.scale.row.clear();
    lp.scale.row.resize(num_row, 1.0);
    let mut v = lp.view();
    let scaled = lp_utils::scale_lp(&mut v, o, force_scaling);
    lp.take_scalars(&v);
    if !scaled {
        let strategy = lp.scale.strategy;
        lp.clear_scale();
        lp.scale.strategy = strategy;
    }
}

/// HEkk::dualize on the engine's LP
pub fn dualize(lps: &mut LpSolver, log: &Log) {
    let lp = &mut lps.lp.g;
    let d = &mut lps.dz;
    debug_assert!(lp.is_colwise());
    d.original_num_col = lp.num_col;
    d.original_num_row = lp.num_row;
    d.original_num_nz = lp.a.num_nz();
    d.original_offset = lp.offset;
    d.original_col_cost = lp.col_cost.clone();
    d.original_col_lower = lp.col_lower.clone();
    d.original_col_upper = lp.col_upper.clone();
    d.original_row_lower = lp.row_lower.clone();
    d.original_row_upper = lp.row_upper.clone();
    let original_num_col = d.original_num_col as usize;
    let original_num_row = d.original_num_row as usize;
    lp.col_cost.reserve(original_num_row);
    lp.col_lower.reserve(original_num_row);
    lp.col_upper.reserve(original_num_row);
    lp.row_lower.reserve(original_num_col);
    lp.row_upper.reserve(original_num_col);
    lp.col_cost.clear();
    lp.col_lower.clear();
    lp.col_upper.clear();
    lp.row_lower.clear();
    lp.row_upper.clear();
    // The transpose of the primal matrix, row-wise
    let mut dual_matrix = lp.a.clone();
    dual_matrix.num_row = d.original_num_col;
    dual_matrix.num_col = d.original_num_row;
    dual_matrix.format = matrix_format::ROWWISE;
    let mut primal_bound_value: Vec<f64> = Vec::new();
    let mut primal_bound_index: Vec<i32> = Vec::new();
    for i_col in 0..original_num_col {
        let cost = d.original_col_cost[i_col];
        let lower = d.original_col_lower[i_col];
        let upper = d.original_col_upper[i_col];
        let primal_bound;
        let row_lower;
        let row_upper;
        if lower == upper {
            primal_bound = lower;
            row_lower = -INF;
            row_upper = INF;
        } else if -lower < INF {
            primal_bound = lower;
            row_lower = -INF;
            row_upper = cost;
            if upper < INF {
                d.upper_bound_col.push(i_col as i32);
            }
        } else if upper < INF {
            primal_bound = upper;
            row_lower = cost;
            row_upper = INF;
        } else {
            primal_bound = 0.0;
            row_lower = cost;
            row_upper = cost;
        }
        lp.row_lower.push(row_lower);
        lp.row_upper.push(row_upper);
        if primal_bound != 0.0 {
            primal_bound_value.push(primal_bound);
            primal_bound_index.push(i_col as i32);
        }
    }
    for i_row in 0..original_num_row {
        let lower = d.original_row_lower[i_row];
        let upper = d.original_row_upper[i_row];
        let (col_cost, col_lower, col_upper);
        if lower == upper {
            col_cost = lower;
            col_lower = -INF;
            col_upper = INF;
        } else if -lower < INF {
            col_cost = lower;
            col_lower = 0.0;
            col_upper = INF;
            if upper < INF {
                d.upper_bound_row.push(i_row as i32);
            }
        } else if upper < INF {
            col_cost = upper;
            col_lower = -INF;
            col_upper = 0.0;
        } else {
            col_cost = 0.0;
            col_lower = 0.0;
            col_upper = 0.0;
        }
        lp.col_cost.push(col_cost);
        lp.col_lower.push(col_lower);
        lp.col_upper.push(col_upper);
    }
    let start = &lp.a.start;
    let index = &lp.a.index;
    let value = &lp.a.value;
    // Boxed variables and constraints yield extra columns in the dual LP
    let mut extra = SparseMatrix::default();
    extra.num_row = d.original_num_col;
    let num_upper_bound_col = d.upper_bound_col.len();
    let num_upper_bound_row = d.upper_bound_row.len();
    for &i_col in &d.upper_bound_col {
        extra.add_vec(&[i_col], &[1.0], 1.0);
        lp.col_cost.push(d.original_col_upper[i_col as usize]);
        lp.col_lower.push(-INF);
        lp.col_upper.push(0.0);
    }
    if num_upper_bound_row != 0 {
        let dummy_row = num_upper_bound_row as i32;
        let mut indirection = vec![dummy_row; original_num_row];
        let mut count = vec![0i32; num_upper_bound_row + 1];
        for (extra_row, &i_row) in d.upper_bound_row.iter().enumerate() {
            indirection[i_row as usize] = extra_row as i32;
            lp.col_cost.push(d.original_row_upper[i_row as usize]);
            lp.col_lower.push(-INF);
            lp.col_upper.push(0.0);
        }
        for &i in &index[..d.original_num_nz as usize] {
            count[indirection[i as usize] as usize] += 1;
        }
        extra.start.resize(num_upper_bound_col + num_upper_bound_row + 1, 0);
        for i_row in 0..num_upper_bound_row {
            extra.start[num_upper_bound_col + i_row + 1] = extra.start[num_upper_bound_col + i_row] + count[i_row];
            count[i_row] = extra.start[num_upper_bound_col + i_row];
        }
        let extra_num_nz = extra.start[num_upper_bound_col + num_upper_bound_row] as usize;
        extra.index.resize(extra_num_nz, 0);
        extra.value.resize(extra_num_nz, 0.0);
        for i_col in 0..original_num_col {
            for el in start[i_col] as usize..start[i_col + 1] as usize {
                let i_row = indirection[index[el] as usize] as usize;
                if i_row < num_upper_bound_row {
                    let e = count[i_row] as usize;
                    extra.index[e] = i_col as i32;
                    extra.value[e] = value[el];
                    count[i_row] += 1;
                }
            }
        }
        extra.num_col += num_upper_bound_row as i32;
    }
    // Incorporate the cost shift by subtracting A*primal_bound from the
    // cost vector; compute the objective offset
    let mut delta_offset = 0.0f64;
    for (&i_col, &multiplier) in primal_bound_index.iter().zip(&primal_bound_value) {
        let i_col = i_col as usize;
        delta_offset = multiplier.mul_add_c(d.original_col_cost[i_col], delta_offset);
        for el in start[i_col] as usize..start[i_col + 1] as usize {
            let c = &mut lp.col_cost[index[el] as usize];
            *c = (-multiplier).mul_add_c(value[el], *c);
        }
    }
    if extra.num_col != 0 {
        let mut primal_bound = vec![0.0f64; original_num_col];
        for (&i_col, &v) in primal_bound_index.iter().zip(&primal_bound_value) {
            primal_bound[i_col as usize] = v;
        }
        for i_col in 0..extra.num_col as usize {
            let mut cost = lp.col_cost[original_num_row + i_col];
            for el in extra.start[i_col] as usize..extra.start[i_col + 1] as usize {
                cost = (-primal_bound[extra.index[el] as usize]).mul_add_c(extra.value[el], cost);
            }
            lp.col_cost[original_num_row + i_col] = cost;
        }
    }
    lp.offset += delta_offset;
    lp.a = dual_matrix;
    lp.a.ensure_colwise();
    let num_extra = extra.num_col;
    lp.a.add_cols(&extra.start, &extra.index, &extra.value, num_extra);
    let dual_num_col = d.original_num_row + num_upper_bound_col as i32 + num_upper_bound_row as i32;
    let dual_num_row = d.original_num_col;
    lp.sense = -lp.sense;
    lp.num_col = dual_num_col;
    lp.num_row = dual_num_row;
    let s = &mut lps.sh.status;
    s.is_dualized = true;
    // The LP's dimensions change, so the dual edge weights no longer fit
    // it (the C++ HEkk kept them and read past the end of the weight
    // vector on the clean-up solve after undualize)
    s.has_dual_steepest_edge_weights = false;
    s.has_basis = false;
    s.has_ar_matrix = false;
    s.has_nla = false;
    // The factor is of the primal LP's basis matrix, of other dimensions:
    // INVERT must be redone (undualize clears it too). The C++ HEkk kept
    // has_invert and solved with the stale factor (vol1 with presolve:
    // undefined behaviour in C++, model status Not Set; a panic in Rust)
    s.has_invert = false;
    s.has_fresh_invert = false;
    log_user!(log, LogType::Info, "Solving dual LP with %d columns", dual_num_col);
    if num_upper_bound_col + num_upper_bound_row != 0 {
        log_user!(log, LogType::Info, " [%d extra from", dual_num_col - d.original_num_row);
        if num_upper_bound_col != 0 {
            log_user!(log, LogType::Info, " %d boxed variable(s)", num_upper_bound_col as i32);
        }
        if num_upper_bound_col != 0 && num_upper_bound_row != 0 {
            log_user!(log, LogType::Info, " and");
        }
        if num_upper_bound_row != 0 {
            log_user!(log, LogType::Info, " %d boxed constraint(s)", num_upper_bound_row as i32);
        }
        log_user!(log, LogType::Info, "]");
    }
    log_user!(log, LogType::Info, " and %d rows\n", dual_num_row);
}

/// HEkk::undualize: the LP and basis of the primal, then a solve from the
/// basis (an `op`), or from a logical basis if `from_basis` is false;
/// returns the status of that solve if it undualized
fn undualize(h: &CSimplexApp, from_basis: bool) -> Option<Status> {
    let lps = h.lps();
    if !lps.sh.status.is_dualized {
        return None;
    }
    let dual_num_col = lps.lp.num_col;
    let d = std::mem::take(&mut lps.dz);
    let num_basic_variables = lps.undualize_basis(
        dual_num_col,
        &d.original_col_lower,
        &d.original_col_upper,
        &d.original_row_lower,
        &d.original_row_upper,
    );
    let lp = &mut lps.lp.g;
    lp.sense = -lp.sense;
    lp.num_col = d.original_num_col;
    lp.num_row = d.original_num_row;
    lp.offset = d.original_offset;
    lp.col_cost = d.original_col_cost;
    lp.col_lower = d.original_col_lower;
    lp.col_upper = d.original_col_upper;
    lp.row_lower = d.original_row_lower;
    lp.row_upper = d.original_row_upper;
    // The primal constraint matrix is available row-wise as the first
    // original_num_row vectors of the dual constraint matrix
    let nr = d.original_num_row as usize;
    let nz = d.original_num_nz as usize;
    let mut primal_matrix = SparseMatrix::default();
    primal_matrix.start = lp.a.start[..nr + 1].to_vec();
    primal_matrix.index = lp.a.index[..nz].to_vec();
    primal_matrix.value = lp.a.value[..nz].to_vec();
    primal_matrix.num_col = d.original_num_col;
    primal_matrix.num_row = d.original_num_row;
    primal_matrix.format = matrix_format::ROWWISE;
    lp.a = primal_matrix;
    lp.a.ensure_colwise();
    if num_basic_variables != d.original_num_row {
        print!("HEkk::undualize: Have {} basic variables, not {}\n", num_basic_variables, d.original_num_row);
    }
    let s = &mut lps.sh.status;
    s.is_dualized = false;
    s.has_dual_steepest_edge_weights = false;
    s.has_basis = from_basis;
    s.has_ar_matrix = false;
    s.has_nla = false;
    s.has_invert = false;
    let iteration_count0 = lps.sh.iteration_count;
    let solve_status = h.status(OP_SOLVE, 0);
    let primal_solve_iteration_count = h.lps().sh.iteration_count - iteration_count0;
    let name = String::from_utf8_lossy(&h.lps().lp.model_name).into_owned();
    log_user!(
        h.log,
        LogType::Info,
        "Solving the primal LP (%s) using the optimal basis of its dual required %d simplex iterations\n",
        name.as_str(),
        primal_solve_iteration_count
    );
    Some(solve_status)
}

/// undualize after the solve of the dual LP: whether it undualized. The
/// C++ ignores the status of the solve from the undualized basis; when the
/// dual LP's solve failed (vol1 with presolve: a rank deficient basis in
/// its primal clean-up) its basis is not used: the primal LP is solved
/// from a logical basis, and that solve decides the status (the C++
/// solves from the failed basis and returns Not Set)
fn undualized(h: &CSimplexApp, return_status: &mut Status) -> bool {
    match undualize(h, *return_status != Status::Error) {
        Some(s) => {
            if *return_status == Status::Error {
                *return_status = s;
            }
            true
        }
        None => false,
    }
}

/// HEkk::unpermute (permuting is not done)
fn unpermute(h: &CSimplexApp) {
    debug_assert!(!h.lps().sh.status.is_permuted);
}

/// solveLpSimplex
pub fn solve_lp_simplex(h: &mut CSimplexApp) -> Status {
    let mut return_status;
    let log = h.log;
    let mut scaled_model_status = MS_UNKNOWN;
    // SAFETY: the solver object's basis flags
    let (basis_valid, basis_useful) = unsafe { (*h.basis_valid, *h.basis_useful) };
    h.op0(OP_PROFILE_START, basis_valid as i64);
    // Copy the simplex iteration count from highs_info_ to ekk_instance
    h.lps().sh.iteration_count = h.info().simplex_iteration_count;
    reset_model_status_and_info(h.model_status(), h.info());
    h.op0(OP_STATS_INIT, 0);
    if h.incumbent.num_row <= 0 {
        log_user!(
            log,
            LogType::Error,
            "solveLpSimplex called for LP with non-positive (%d) number of constraints\n",
            h.incumbent.num_row
        );
        return h.return_from(Status::Error);
    }
    // Copy the incumbent LP: scaling, dualization and the solves work
    // on the copy
    // SAFETY: the incumbent's view and name
    unsafe { h.lps().lp.import(&h.incumbent, h.model_name.get()) };
    if consider_scaling(&h.lp_options, &mut h.lps().lp) {
        let lps = h.lps();
        lps.sh.status.has_ar_matrix = false;
        lps.sh.dual_values_valid = false;
    }
    let has_basis = h.lps().sh.status.has_basis;
    if !has_basis && !basis_valid && basis_useful {
        // There is no simplex basis, but there is a useful HiGHS basis
        // that is not validated: formSimplexLpBasisAndFactor (the LP is
        // scaled, so only the basis is checked and completed)
        let passed_scaled = h.lps().lp.is_scaled;
        if !passed_scaled {
            consider_scaling(&h.lp_options, &mut h.lps().lp);
        }
        // SAFETY: the basis' flag
        unsafe { *h.basis_alien = true };
        let lp = h.lps().lp.view();
        accommodate_alien_basis(&h.factor_log, &lp, h.col_status.as_mut_slice(), h.row_status.as_mut_slice());
        // SAFETY: as above
        unsafe { *h.basis_alien = false };
        if !passed_scaled {
            h.lps().lp.unapply_scale();
        }
        // formSimplexLpBasisAndFactor may introduce variables with
        // HighsBasisStatus::kNonbasic, so refine it
        let lp = &h.lps().lp;
        // SAFETY: the solution's flag
        let have_solution = unsafe { *h.value_valid };
        let (cv, rv) = if have_solution {
            (h.col_value.as_slice().to_vec(), h.row_value.as_slice().to_vec())
        } else {
            (Vec::new(), Vec::new())
        };
        refine_basis(&lp.col_lower, &lp.col_upper, &cv, h.col_status.as_mut_slice());
        refine_basis(&lp.row_lower, &lp.row_upper, &rv, h.row_status.as_mut_slice());
        // SAFETY: the basis' flag
        unsafe { *h.basis_valid = true };
    }
    // SAFETY: as above
    let basis_valid = unsafe { *h.basis_valid };
    // The LP is moved to EKK: the pointers and checks
    h.op0(OP_MOVE_LP, 0);
    if !h.lps().sh.status.has_basis {
        if basis_valid {
            let call_status = h.status(OP_SET_BASIS, 0);
            if call_status == Status::Error {
                h.move_back(false);
                return h.return_from(call_status);
            }
        } else {
            // Starting from a logical basis, so consider dualizing
            if h.simplex_dualize_strategy == OPTION_CHOOSE || h.simplex_dualize_strategy == OPTION_ON {
                let mut dualize_lp = true;
                if h.simplex_dualize_strategy == OPTION_CHOOSE && h.incumbent.num_row < 10 * h.incumbent.num_col {
                    dualize_lp = false;
                }
                if dualize_lp {
                    dualize(h.lps(), &log);
                }
            }
            if h.simplex_permute_strategy == OPTION_CHOOSE || h.simplex_permute_strategy == OPTION_ON {
                // HEkk::permute asserts: permuting is not implemented
                debug_assert!(false);
            }
        }
    }
    let mut num_unscaled_primal_infeasibilities = ILLEGAL_COUNT;
    let solve_unscaled_lp;
    let mut solved_unscaled_lp = false;
    let mut dualized = false;
    if !h.lps().lp.scale.has_scaling {
        // Solve the unscaled LP with unscaled NLA
        return_status = h.status(OP_SOLVE, 0);
        solved_unscaled_lp = true;
        unpermute(h);
        dualized = undualized(h, &mut return_status);
        if h.cost_scale_factor != 0 {
            let cost_scale_factor = 2f64.powi(-h.cost_scale_factor);
            log_dev!(log, LogType::Info, "Objective = %11.4g\n", cost_scale_factor * h.lps().sh.info.dual_objective_value);
            h.lps().sh.model_status = MS_NOTSET;
            return_status = Status::Error;
        }
        solve_unscaled_lp = true;
    } else {
        let mut refine_solution = false;
        if h.simplex_unscaled_solution_strategy == UNSCALED_NONE
            || h.simplex_unscaled_solution_strategy == UNSCALED_REFINE
        {
            // Solve the scaled LP!
            return_status = h.status(OP_SOLVE, 0);
            unpermute(h);
            dualized = undualized(h, &mut return_status);
            if h.cost_scale_factor != 0 {
                let cost_scale_factor = 2f64.powi(-h.cost_scale_factor);
                log_dev!(log, LogType::Info, "Objective = %11.4g\n", cost_scale_factor * h.lps().sh.info.dual_objective_value);
                h.lps().sh.model_status = MS_NOTSET;
                return_status = Status::Error;
            }
            if return_status == Status::Error {
                h.move_back(dualized);
                return h.return_from(return_status);
            }
            // Copy solution data from the EKK instance
            scaled_model_status = h.lps().sh.model_status;
            {
                let primal_objective_value = h.lps().sh.info.primal_objective_value;
                let iteration_count = h.lps().sh.iteration_count;
                let info = h.info();
                info.objective_function_value = primal_objective_value;
                info.simplex_iteration_count = iteration_count;
            }
            h.get_solution();
            h.get_highs_basis();
            h.info().basis_validity = BASIS_VALIDITY_VALID;
            h.move_back(dualized);
            dualized = false;
            // Now that the LP is unscaled, the simplex NLA applies the
            // scaling
            h.lps().set_nla_rust(true);
            {
                let s = &h.lps().lp.scale;
                let (col, row, cost) = (s.col.clone(), s.row.clone(), s.cost);
                unscale_solution(
                    &col[..s.num_col as usize],
                    &row[..s.num_row as usize],
                    cost,
                    h.col_value.as_mut_slice(),
                    h.col_dual.as_mut_slice(),
                    h.row_value.as_mut_slice(),
                    h.row_dual.as_mut_slice(),
                );
            }
            // Determine whether the unscaled LP has been solved
            {
                let env = h.env();
                let lp = env.lp;
                let mut u = super::lp_solver::UnscaledInfeasibilities::default();
                h.lps().get_unscaled_infeasibilities(&env, &lp, &mut u);
                let info = h.info();
                info.num_primal_infeasibilities = u.num_primal_infeasibilities;
                info.max_primal_infeasibility = u.max_primal_infeasibility;
                info.sum_primal_infeasibilities = u.sum_primal_infeasibilities;
                info.num_dual_infeasibilities = u.num_dual_infeasibilities;
                info.max_dual_infeasibility = u.max_dual_infeasibility;
                info.sum_dual_infeasibilities = u.sum_dual_infeasibilities;
                set_solution_status(info);
            }
            let info = *h.info();
            num_unscaled_primal_infeasibilities = info.num_primal_infeasibilities;
            let num_unscaled_dual_infeasibilities = info.num_dual_infeasibilities;
            let scaled_optimality_but_unscaled_infeasibilities = scaled_model_status == MS_OPTIMAL
                && (num_unscaled_primal_infeasibilities != 0 || num_unscaled_dual_infeasibilities != 0);
            let scaled_objective_target_but_unscaled_primal_infeasibilities =
                scaled_model_status == MS_OBJECTIVE_TARGET && info.num_primal_infeasibilities > 0;
            let scaled_objective_bound_but_unscaled_dual_infeasibilities =
                scaled_model_status == MS_OBJECTIVE_BOUND && info.num_dual_infeasibilities > 0;
            if scaled_optimality_but_unscaled_infeasibilities
                || scaled_objective_target_but_unscaled_primal_infeasibilities
                || scaled_objective_bound_but_unscaled_dual_infeasibilities
            {
                log_dev!(
                    log,
                    LogType::Info,
                    "After unscaling with status %s, have num/max/sum primal (%d/%g/%g) and dual (%d/%g/%g) unscaled infeasibilities\n",
                    super::hekk::model_status_string(scaled_model_status),
                    info.num_primal_infeasibilities,
                    info.max_primal_infeasibility,
                    info.sum_primal_infeasibilities,
                    info.num_dual_infeasibilities,
                    info.max_dual_infeasibility,
                    info.sum_dual_infeasibilities
                );
            }
            refine_solution = h.simplex_unscaled_solution_strategy == UNSCALED_REFINE
                && (scaled_optimality_but_unscaled_infeasibilities
                    || scaled_model_status == MS_INFEASIBLE
                    || scaled_model_status == MS_UNBOUNDED_OR_INFEASIBLE
                    || scaled_model_status == MS_UNBOUNDED
                    || scaled_objective_bound_but_unscaled_dual_infeasibilities
                    || scaled_objective_target_but_unscaled_primal_infeasibilities
                    || scaled_model_status == MS_UNKNOWN);
            if !refine_solution {
                *h.model_status() = scaled_model_status;
                return_status = status_from_model_status(scaled_model_status);
                return h.return_from(return_status);
            }
        } else {
            // The LP is scaled but solved directly unscaled: unscale it
            debug_assert!(h.simplex_unscaled_solution_strategy == UNSCALED_DIRECT);
            h.move_back(false);
        }
        debug_assert!(h.simplex_unscaled_solution_strategy == UNSCALED_DIRECT || refine_solution);
        // Solve the unscaled LP using scaled NLA
        h.op0(OP_MOVE_LP, 0);
        let mut solve_unscaled = true;
        let dual_ray_index = h.lps().sh.dual_ray_index;
        if scaled_model_status == MS_INFEASIBLE && h.lps().sh.exit_algorithm == ALGORITHM_DUAL {
            debug_assert!(dual_ray_index != NO_RAY_INDEX);
        }
        if scaled_model_status == MS_INFEASIBLE && dual_ray_index != NO_RAY_INDEX {
            h.lps().set_nla_rust(true);
            if h.op0(OP_PROOF, 0) != 0 {
                solve_unscaled = false;
            }
        }
        solve_unscaled_lp = solve_unscaled;
        return_status = Status::Ok;
        if solve_unscaled {
            // Save options/strategies that may be changed
            let simplex_strategy = *h.options_simplex_strategy();
            // SAFETY: the option
            let dual_simplex_cost_perturbation_multiplier = unsafe { *h.dual_simplex_cost_perturbation_multiplier };
            let simplex_dual_edge_weight_strategy = h.lps().sh.info.dual_edge_weight_strategy;
            if num_unscaled_primal_infeasibilities == 0 || scaled_model_status == MS_OBJECTIVE_TARGET {
                *h.options_simplex_strategy() = STRATEGY_PRIMAL;
                if scaled_model_status == MS_OBJECTIVE_TARGET {
                    let sh = &h.lps().sh;
                    log_dev!(
                        log,
                        LogType::Info,
                        "solveLpSimplex: Calling primal simplex after scaled_model_status == HighsModelStatus::kObjectiveTarget: solve = %d; tick = %d; iter = %d\n",
                        sh.debug_solve_call_num,
                        sh.debug_initial_build_synthetic_tick,
                        sh.iteration_count
                    );
                }
            } else {
                if *h.options_simplex_strategy() != STRATEGY_DUAL {
                    log_dev!(
                        log,
                        LogType::Info,
                        "Forcing change from %s to %s\n",
                        simplex_strategy_to_string(*h.options_simplex_strategy()),
                        simplex_strategy_to_string(STRATEGY_DUAL)
                    );
                }
                *h.options_simplex_strategy() = STRATEGY_DUAL;
                let st = &h.lps().sh.status;
                // SAFETY: the basis flag
                let valid = unsafe { *h.basis_valid };
                if (st.has_basis || valid) && !st.has_dual_steepest_edge_weights {
                    h.lps().sh.info.dual_edge_weight_strategy = EDGE_WEIGHT_DEVEX;
                }
            }
            let force_phase2 = h.simplex_unscaled_solution_strategy != UNSCALED_DIRECT
                && scaled_model_status != MS_OBJECTIVE_TARGET;
            return_status = h.status(OP_SOLVE, force_phase2 as i64);
            solved_unscaled_lp = true;
            if scaled_model_status != MS_OBJECTIVE_BOUND && h.lps().sh.model_status == MS_OBJECTIVE_BOUND {
                let objective_bound_refinement = h.lps().sh.info.num_dual_infeasibilities > 0;
                if objective_bound_refinement {
                    *h.options_simplex_strategy() = STRATEGY_PRIMAL;
                    return_status = h.status(OP_SOLVE, force_phase2 as i64);
                }
            }
            *h.options_simplex_strategy() = simplex_strategy;
            // SAFETY: the option
            unsafe { *h.dual_simplex_cost_perturbation_multiplier = dual_simplex_cost_perturbation_multiplier };
            h.lps().sh.info.dual_edge_weight_strategy = simplex_dual_edge_weight_strategy;
        }
    }
    if solved_unscaled_lp {
        scaled_model_status = h.lps().sh.model_status;
        {
            let primal_objective_value = h.lps().sh.info.primal_objective_value;
            let iteration_count = h.lps().sh.iteration_count;
            let info = h.info();
            info.objective_function_value = primal_objective_value;
            info.simplex_iteration_count = iteration_count;
        }
        h.get_solution();
        h.get_highs_basis();
        h.info().basis_validity = BASIS_VALIDITY_VALID;
    }
    // Move the incumbent LP back from Ekk
    h.move_back(dualized);
    h.lps().set_nla_rust(true);
    if return_status == Status::Error {
        *h.model_status() = scaled_model_status;
        return h.return_from(Status::Error);
    }
    if solved_unscaled_lp {
        debug_assert!(solve_unscaled_lp);
        let si = h.lps().sh.info;
        let info = h.info();
        info.num_primal_infeasibilities = si.num_primal_infeasibilities;
        info.max_primal_infeasibility = si.max_primal_infeasibility;
        info.sum_primal_infeasibilities = si.sum_primal_infeasibilities;
        info.num_dual_infeasibilities = si.num_dual_infeasibilities;
        info.max_dual_infeasibility = si.max_dual_infeasibility;
        info.sum_dual_infeasibilities = si.sum_dual_infeasibilities;
    } else {
        debug_assert!(!solve_unscaled_lp);
        debug_assert!(scaled_model_status == MS_INFEASIBLE);
    }
    set_solution_status(h.info());
    *h.model_status() = scaled_model_status;
    return_status = status_from_model_status(scaled_model_status);
    h.return_from(return_status)
}

/// # Safety
/// The host's views, pointers and op valid for the call
#[no_mangle]
pub unsafe extern "C" fn highs_rs_solve_lp_simplex(h: *mut CSimplexApp) -> i32 {
    solve_lp_simplex(&mut *h) as i32
}

