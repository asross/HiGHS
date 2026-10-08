//! HighsSolve.cpp: solveLp (the choice between simplex, IPX, HiPO and
//! PDLP, and the simplex clean-up of an unwelcome IPM status),
//! solveUnconstrainedLp and assessExcessiveObjectiveBoundScaling.
//!
//! The solvers stay behind C++ callbacks (highs/lp_data/HighsSolveRust.cpp),
//! which also catch their exceptions; the model status is the solver
//! object's, read and written in place.

use super::ffi::CLp;
use super::run::{use_ipm, MS_NOTSET, MS_UNBOUNDED_OR_INFEASIBLE, MS_UNKNOWN};
use super::solution::{Info, BASIS_BASIC, SOLUTION_STATUS_FEASIBLE, SOLUTION_STATUS_INFEASIBLE, SOLUTION_STATUS_NONE};
use super::{var_type, Log, LogType, Status, INF};
use crate::log_user;
use crate::simplex::hekk::model_status_string;
use crate::util::printf::sprintf;
use std::ffi::c_void;

// HighsBasisStatus
const BASIS_LOWER: u8 = 0;
const BASIS_UPPER: u8 = 2;
const BASIS_ZERO: u8 = 3;
const BASIS_NONBASIC: u8 = 4;

// HighsModelStatus
const MS_OPTIMAL: i32 = 7;
const MS_INFEASIBLE: i32 = 8;
const MS_UNBOUNDED: i32 = 10;
const BASIS_VALIDITY_VALID: i32 = 1;

/// setSolutionStatus (HSimplex.cpp)
pub fn set_solution_status(info: &mut Info) {
    info.primal_solution_status = match info.num_primal_infeasibilities {
        n if n < 0 => SOLUTION_STATUS_NONE,
        0 => SOLUTION_STATUS_FEASIBLE,
        _ => SOLUTION_STATUS_INFEASIBLE,
    };
    info.dual_solution_status = match info.num_dual_infeasibilities {
        n if n < 0 => SOLUTION_STATUS_NONE,
        0 => SOLUTION_STATUS_FEASIBLE,
        _ => SOLUTION_STATUS_INFEASIBLE,
    };
}

/// resetModelStatusAndHighsInfo
pub fn reset_model_status_and_info(model_status: &mut i32, info: &mut Info) {
    *model_status = MS_NOTSET;
    info.objective_function_value = 0.0;
    info.primal_solution_status = SOLUTION_STATUS_NONE;
    info.dual_solution_status = SOLUTION_STATUS_NONE;
    info.invalidate_kkt();
}

/// The solution and basis written by solveUnconstrainedLp, sized by C++
/// to the LP (the row values, duals and statuses are set here)
pub struct Unconstrained<'a> {
    pub col_value: &'a mut [f64],
    pub col_dual: &'a mut [f64],
    pub row_value: &'a mut [f64],
    pub row_dual: &'a mut [f64],
    pub col_status: &'a mut [u8],
    pub row_status: &'a mut [u8],
}

/// solveUnconstrainedLp(options, lp, ...) after the checks that the LP
/// has no nonzeros: returns the model status. The objective `objective +=
/// value * cost` is a separate statement, which clang contracts.
#[allow(clippy::too_many_arguments)]
pub fn solve_unconstrained_lp(
    log: &Log,
    on: bool,
    pft: f64,
    dft: f64,
    lp: &CLp,
    info: &mut Info,
    s: Unconstrained,
) -> i32 {
    use crate::util::fma::ClangFma;
    let num_col = lp.num_col;
    if on {
        log_user!(log, LogType::Info, "Solving an unconstrained LP with %d columns\n", num_col);
    }
    // SAFETY: the LP's arrays, read only
    let (cost, col_lower, col_upper, row_lower, row_upper) = unsafe {
        (lp.col_cost.get(), lp.col_lower.get(), lp.col_upper.get(), lp.row_lower.get(), lp.row_upper.get())
    };
    let mut objective = lp.offset;
    info.num_primal_infeasibilities = 0;
    info.max_primal_infeasibility = 0.0;
    info.sum_primal_infeasibilities = 0.0;
    info.num_dual_infeasibilities = 0;
    info.max_dual_infeasibility = 0.0;
    info.sum_dual_infeasibilities = 0.0;
    let cmax = |a: f64, b: f64| if a < b { b } else { a };
    for i in 0..lp.num_row as usize {
        let mut primal_infeasibility = 0.0;
        let lower = row_lower[i];
        let upper = row_upper[i];
        if lower > pft {
            primal_infeasibility = lower;
        } else if upper < -pft {
            primal_infeasibility = -upper;
        }
        s.row_value[i] = 0.0;
        s.row_dual[i] = 0.0;
        s.row_status[i] = BASIS_BASIC;
        if primal_infeasibility > pft {
            info.num_primal_infeasibilities += 1;
        }
        info.sum_primal_infeasibilities += primal_infeasibility;
        info.max_primal_infeasibility = cmax(primal_infeasibility, info.max_primal_infeasibility);
    }
    let sense = lp.sense as f64;
    for j in 0..num_col as usize {
        let cost_j = cost[j];
        let dual = sense * cost_j;
        let lower = col_lower[j];
        let upper = col_upper[j];
        let value;
        let mut primal_infeasibility = 0.0;
        let dual_infeasibility;
        let status;
        if lower > upper {
            if lower >= INF {
                if -upper >= INF {
                    value = 0.0;
                    status = BASIS_ZERO;
                    primal_infeasibility = INF;
                    dual_infeasibility = dual.abs();
                } else {
                    value = upper;
                    status = BASIS_UPPER;
                    primal_infeasibility = lower - value;
                    dual_infeasibility = cmax(dual, 0.0);
                }
            } else {
                value = lower;
                status = BASIS_LOWER;
                primal_infeasibility = value - upper;
                dual_infeasibility = cmax(-dual, 0.0);
            }
        } else if -lower >= INF && upper >= INF {
            value = 0.0;
            status = BASIS_ZERO;
            dual_infeasibility = dual.abs();
        } else if dual >= dft {
            if -lower < INF {
                value = lower;
                status = BASIS_LOWER;
                dual_infeasibility = 0.0;
            } else {
                value = upper;
                status = BASIS_UPPER;
                dual_infeasibility = dual;
            }
        } else if dual <= -dft {
            if upper < INF {
                value = upper;
                status = BASIS_UPPER;
                dual_infeasibility = 0.0;
            } else {
                value = lower;
                status = BASIS_LOWER;
                dual_infeasibility = -dual;
            }
        } else {
            if -lower >= INF {
                value = upper;
                status = BASIS_UPPER;
            } else {
                value = lower;
                status = BASIS_LOWER;
            }
            dual_infeasibility = dual.abs();
        }
        debug_assert!(status != BASIS_NONBASIC);
        s.col_value[j] = value;
        s.col_dual[j] = sense * dual;
        s.col_status[j] = status;
        objective = value.mul_add_c(cost_j, objective);
        if primal_infeasibility > pft {
            info.num_primal_infeasibilities += 1;
        }
        info.sum_primal_infeasibilities += primal_infeasibility;
        info.max_primal_infeasibility = cmax(primal_infeasibility, info.max_primal_infeasibility);
        if dual_infeasibility > dft {
            info.num_dual_infeasibilities += 1;
        }
        info.sum_dual_infeasibilities += dual_infeasibility;
        info.max_dual_infeasibility = cmax(dual_infeasibility, info.max_dual_infeasibility);
    }
    info.objective_function_value = objective;
    info.basis_validity = BASIS_VALIDITY_VALID;
    set_solution_status(info);
    if info.num_primal_infeasibilities != 0 {
        MS_INFEASIBLE
    } else if info.num_dual_infeasibilities != 0 {
        MS_UNBOUNDED
    } else {
        MS_OPTIMAL
    }
}

/// The steps of solveLp on C++ objects (HighsSolveRust.cpp)
#[repr(i32)]
#[derive(Clone, Copy)]
pub enum SolveOp {
    /// assessLp (highs_debug_level > min) -> status
    DebugAssess = 1,
    /// solveUnconstrainedLp -> status
    Unconstrained,
    /// solveLpIpx / solveLpCupdlp, exceptions caught -> status (HiPO and
    /// HiPDLP are not in Crestline)
    Ipx,
    Pdlp,
    /// solveLpSimplex -> status
    Simplex,
    /// isSolutionRightSize -> bool
    SolutionRightSize,
    /// debugHighsLpSolution(message) -> logical error
    DebugSolution,
}

/// The solver object and options of solveLp
#[repr(C)]
pub struct CSolve {
    pub log: Log,
    pub ctx: *mut c_void,
    pub op: unsafe extern "C" fn(*mut c_void, i32, *const u8, usize) -> i64,
    pub model_status: *mut i32,
    pub info: *mut Info,
    pub value_valid: *const bool,
    pub basis_valid: *const bool,
    pub num_row: i32,
    pub num_nz: i32,
    pub solver: super::options::RsStr,
    pub run_crossover: super::options::RsStr,
    pub run_centring: bool,
    pub allow_unbounded_or_infeasible: bool,
    pub highs_debug_level: i32,
    pub output_flag: *const bool,
    pub log_dev_level: *const i32,
    /// Set by C++ when a step threw: the exception is rethrown once Rust
    /// has returned, so no further step (or log message) is made
    pub aborted: *const bool,
}

fn status_of(v: i64) -> Status {
    match v {
        0 => Status::Ok,
        1 => Status::Warning,
        _ => Status::Error,
    }
}

impl CSolve {
    fn op(&self, op: SolveOp, msg: &str) -> i64 {
        // SAFETY: the C++ steps, called with their context
        unsafe { (self.op)(self.ctx, op as i32, msg.as_ptr(), msg.len()) }
    }
    fn on(&self) -> bool {
        // SAFETY: the C++ log options' flag
        !self.ab() && unsafe { *self.output_flag }
    }
    fn ab(&self) -> bool {
        // SAFETY: the C++ flag
        unsafe { *self.aborted }
    }
    fn interpret(&self, call: Status, from: Status, message: &str) -> Status {
        // SAFETY: as on
        if call != Status::Ok && self.on() && unsafe { *self.log_dev_level } != 0 {
            self.log.interpret(call, from, message)
        } else {
            call.worse(from)
        }
    }
    fn ms(&self) -> i32 {
        // SAFETY: the solver object's model status
        unsafe { *self.model_status }
    }

    /// The simplexSolve lambda of solveLp
    fn simplex_solve(&self, message: &str) -> Status {
        let call_status = status_of(self.op(SolveOp::Simplex, message));
        if self.ab() {
            return Status::Error;
        }
        let return_status = self.interpret(call_status, Status::Ok, "solveLpSimplex");
        if return_status == Status::Error {
            return return_status;
        }
        if self.op(SolveOp::SolutionRightSize, message) == 0 {
            if self.on() {
                log_user!(self.log, LogType::Error, "Inconsistent solution returned from solver\n");
            }
            return Status::Error;
        }
        return_status
    }

    /// solveLp(solver_object, message)
    pub fn solve_lp(&self, message: &str) -> Status {
        let mut return_status = Status::Ok;
        // SAFETY: the solver object's model status and HighsInfo
        unsafe { reset_model_status_and_info(&mut *self.model_status, &mut *self.info) };
        if self.on() {
            // The C++ passes the message as the format
            let fmt = format!("{}\n", message);
            self.log.user(LogType::Info, &sprintf(&fmt, &[]));
        }
        if self.highs_debug_level > 0 {
            let call_status = status_of(self.op(SolveOp::DebugAssess, message));
            return_status = self.interpret(call_status, return_status, "assessLp");
            if return_status == Status::Error {
                return return_status;
            }
        }
        // SAFETY: the C++ strings live for the call and are not changed
        let (solver, run_crossover) = unsafe { (self.solver.get(), self.run_crossover.get()) };
        let use_only_ipm = use_ipm(solver) || self.run_centring;
        let use_pdlp = solver == b"pdlp" || solver == b"hipdlp";
        if self.num_row == 0 || self.num_nz == 0 {
            let call_status = status_of(self.op(SolveOp::Unconstrained, message));
            if self.ab() {
                return Status::Error;
            }
            return_status = self.interpret(call_status, return_status, "solveUnconstrainedLp");
            if return_status == Status::Error {
                return return_status;
            }
        } else if use_only_ipm || use_pdlp {
            if use_only_ipm {
                let call_status = status_of(self.op(SolveOp::Ipx, message));
                if self.ab() {
                    return Status::Error;
                }
                return_status = self.interpret(call_status, return_status, "solveLpIpx");
            } else {
                let call_status = status_of(self.op(SolveOp::Pdlp, message));
                if self.ab() {
                    return Status::Error;
                }
                return_status = self.interpret(call_status, return_status, "solveLp-Pdlp");
            }
            if return_status == Status::Error {
                return return_status;
            }
            if use_ipm(solver) || self.run_centring {
                let ms = self.ms();
                let unwelcome_ipx_status = ms == MS_UNKNOWN
                    || (ms == MS_UNBOUNDED_OR_INFEASIBLE && !self.allow_unbounded_or_infeasible);
                if unwelcome_ipx_status {
                    if self.on() {
                        let crossover = if self.run_centring { "off".to_string() } else {
                            String::from_utf8_lossy(run_crossover).into_owned()
                        };
                        // SAFETY: the solver object's flags
                        let (basis_valid, value_valid) = unsafe { (*self.basis_valid, *self.value_valid) };
                        log_user!(
                            self.log,
                            LogType::Warning,
                            "Unwelcome IPX status of %s: basis is %svalid; solution is %svalid; run_crossover is \"%s\"\n",
                            model_status_string(ms),
                            if basis_valid { "" } else { "not " },
                            if value_valid { "" } else { "not " },
                            &*crossover
                        );
                    }
                    let allow_simplex_cleanup = run_crossover != b"off" && !self.run_centring;
                    if allow_simplex_cleanup {
                        if self.on() {
                            log_user!(
                                self.log,
                                LogType::Warning,
                                "IPM solution is imprecise, so clean up with simplex\n"
                            );
                        }
                        return_status = self.simplex_solve(message);
                        if return_status == Status::Error {
                            return return_status;
                        }
                    }
                }
            }
        } else {
            return_status = self.simplex_solve(message);
            if return_status == Status::Error {
                return return_status;
            }
        }
        if self.op(SolveOp::DebugSolution, message) != 0 {
            return_status = Status::Error;
        }
        return_status
    }
}

/// The extremes of finite nonzero absolute values (the
/// assessFiniteNonzero lambda)
fn assess_finite_nonzero(value: f64, min_value: &mut f64, max_value: &mut f64) {
    let abs_value = value.abs();
    if abs_value > 0.0 && abs_value < INF {
        // std::min / std::max
        if abs_value < *min_value {
            *min_value = abs_value;
        }
        if *max_value < abs_value {
            *max_value = abs_value;
        }
    }
}

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

/// The user scaling of assessExcessiveObjectiveBoundScaling (of HighsUserScaleData)
pub struct UserScale {
    pub user_objective_scale: i32,
    pub user_bound_scale: i32,
    pub suggested_user_objective_scale: i32,
    pub suggested_user_bound_scale: i32,
}

// kExcessivelySmall/LargeObjectiveCoefficient, kExcessivelySmall/LargeBoundValue
const SMALL_OBJECTIVE_COEFFICIENT: f64 = 1e-4;
const LARGE_OBJECTIVE_COEFFICIENT: f64 = 1e6;
const SMALL_BOUND: f64 = 1e-4;
const LARGE_BOUND: f64 = 1e6;

/// assessExcessiveObjectiveBoundScaling: the coefficient ranges of the
/// model, warnings about extreme values, and the suggested user scaling
pub fn assess_excessive_objective_bound_scaling(log: &Log, lp: &CLp, hessian_value: &[f64], d: &mut UserScale) {
    if lp.num_col == 0 || lp.num_row == 0 {
        return;
    }
    let user_cost_or_bound_scale = d.user_objective_scale != 0 || d.user_bound_scale != 0;
    if user_cost_or_bound_scale {
        let mut message = String::new();
        if d.user_objective_scale != 0 {
            message += &sprintf(" user_objective_scale option value of %d", &[d.user_objective_scale.into()]);
        }
        if d.user_bound_scale != 0 {
            if d.user_objective_scale != 0 {
                message += " and";
            }
            message += &sprintf(" user_bound_scale option value of %d", &[d.user_bound_scale.into()]);
        }
        log_user!(log, LogType::Info, "Assessing costs and bounds after applying%s\n", &*message);
    }
    // SAFETY: the LP's arrays, read only
    let (cost, col_lower, col_upper, row_lower, row_upper, integrality, value) = unsafe {
        (
            lp.col_cost.get(),
            lp.col_lower.get(),
            lp.col_upper.get(),
            lp.row_lower.get(),
            lp.row_upper.get(),
            lp.integrality.get(),
            lp.a.value.get(),
        )
    };
    let mut min_continuous_col_cost = INF;
    let mut min_noncontinuous_col_cost = INF;
    let mut max_continuous_col_cost = -INF;
    let mut max_noncontinuous_col_cost = -INF;
    let mut min_continuous_col_bound = INF;
    let mut min_noncontinuous_col_bound = INF;
    let mut max_continuous_col_bound = -INF;
    let mut max_noncontinuous_col_bound = -INF;
    let is_mip = !integrality.is_empty();
    for j in 0..lp.num_col as usize {
        if is_mip && integrality[j] != var_type::CONTINUOUS {
            assess_finite_nonzero(cost[j], &mut min_noncontinuous_col_cost, &mut max_noncontinuous_col_cost);
            assess_finite_nonzero(col_lower[j], &mut min_noncontinuous_col_bound, &mut max_noncontinuous_col_bound);
            assess_finite_nonzero(col_upper[j], &mut min_noncontinuous_col_bound, &mut max_noncontinuous_col_bound);
        } else {
            assess_finite_nonzero(cost[j], &mut min_continuous_col_cost, &mut max_continuous_col_cost);
            assess_finite_nonzero(col_lower[j], &mut min_continuous_col_bound, &mut max_continuous_col_bound);
            assess_finite_nonzero(col_upper[j], &mut min_continuous_col_bound, &mut max_continuous_col_bound);
        }
    }
    let mut min_col_cost = cmin(min_continuous_col_cost, min_noncontinuous_col_cost);
    let mut max_col_cost = cmax(max_continuous_col_cost, max_noncontinuous_col_cost);
    let mut min_col_bound = cmin(min_continuous_col_bound, min_noncontinuous_col_bound);
    let mut max_col_bound = cmax(max_continuous_col_bound, max_noncontinuous_col_bound);

    let mut min_matrix_value = INF;
    let mut max_matrix_value = -INF;
    // HighsSparseMatrix::numNz of the column-wise or row-wise matrix
    let num_matrix_nz = lp_num_nz(lp);
    for &v in &value[..num_matrix_nz] {
        assess_finite_nonzero(v, &mut min_matrix_value, &mut max_matrix_value);
    }
    let mut min_row_bound = INF;
    let mut max_row_bound = -INF;
    for i in 0..lp.num_row as usize {
        assess_finite_nonzero(row_lower[i], &mut min_row_bound, &mut max_row_bound);
        assess_finite_nonzero(row_upper[i], &mut min_row_bound, &mut max_row_bound);
    }
    let mut min_continuous_hessian_value = INF;
    let mut max_continuous_hessian_value = -INF;
    let num_hessian_nz = hessian_value.len();
    for &v in hessian_value {
        assess_finite_nonzero(v, &mut min_continuous_hessian_value, &mut max_continuous_hessian_value);
    }
    let mut min_scalable_bound = cmin(min_continuous_col_bound, min_row_bound);
    let mut max_scalable_bound = cmax(max_continuous_col_bound, max_row_bound);
    if min_scalable_bound == INF {
        min_scalable_bound = 0.0;
    }
    if max_scalable_bound == -INF {
        max_scalable_bound = 0.0;
    }
    if min_col_cost == INF {
        min_col_cost = 0.0;
    }
    if max_col_cost == -INF {
        max_col_cost = 0.0;
    }
    if min_col_bound == INF {
        min_col_bound = 0.0;
    }
    if max_col_bound == -INF {
        max_col_bound = 0.0;
    }
    if min_row_bound == INF {
        min_row_bound = 0.0;
    }
    if max_row_bound == -INF {
        max_row_bound = 0.0;
    }
    let mut min_hessian_value = min_continuous_hessian_value;
    let mut max_hessian_value = max_continuous_hessian_value;
    if min_hessian_value == INF {
        min_hessian_value = 0.0;
    }
    if max_hessian_value == -INF {
        max_hessian_value = 0.0;
    }
    log_user!(log, LogType::Info, "Coefficient ranges:\n");
    if num_matrix_nz != 0 {
        log_user!(log, LogType::Info, "  Matrix  [%5.0e, %5.0e]\n", min_matrix_value, max_matrix_value);
    }
    log_user!(log, LogType::Info, "  Cost    [%5.0e, %5.0e]\n", min_col_cost, max_col_cost);
    if num_hessian_nz != 0 {
        log_user!(log, LogType::Info, "  Hessian [%5.0e, %5.0e]\n", min_hessian_value, max_hessian_value);
    }
    log_user!(log, LogType::Info, "  Bound   [%5.0e, %5.0e]\n", min_col_bound, max_col_bound);
    log_user!(log, LogType::Info, "  RHS     [%5.0e, %5.0e]\n", min_row_bound, max_row_bound);

    let problem = if user_cost_or_bound_scale { "User-scaled problem" } else { "Problem" };
    let warn = |what: &str| log_user!(log, LogType::Warning, "%s has some excessively %s\n", problem, what);
    if 0.0 < min_col_cost && min_col_cost < SMALL_OBJECTIVE_COEFFICIENT {
        warn("small costs");
    }
    if max_col_cost > LARGE_OBJECTIVE_COEFFICIENT {
        warn("large costs");
    }
    if 0.0 < min_hessian_value && min_hessian_value < SMALL_OBJECTIVE_COEFFICIENT {
        warn("small Hessian values");
    }
    if max_hessian_value > LARGE_OBJECTIVE_COEFFICIENT {
        warn("large Hessian values");
    }
    if 0.0 < min_col_bound && min_col_bound < SMALL_BOUND {
        warn("small column bounds");
    }
    if max_col_bound > LARGE_BOUND {
        warn("large column bounds");
    }
    if 0.0 < min_row_bound && min_row_bound < SMALL_BOUND {
        warn("small row bounds");
    }
    if max_row_bound > LARGE_BOUND {
        warn("large row bounds");
    }
    let _ = min_scalable_bound;

    // The suggestScaling lambda
    let suggest_scaling = |max_value: f64, small_value: f64, large_value: f64| -> f64 {
        if max_value > large_value {
            large_value / max_value
        } else if 0.0 < max_value && max_value < small_value {
            small_value / max_value
        } else {
            1.0
        }
    };
    // The outerRoundedLog lambda (HighsInt of the rounded double)
    let outer_rounded_log = |value: f64, base: i32| -> i32 {
        let l = if base == 2 { value.log2() } else { value.log10() };
        (if value < 1.0 { l.floor() } else { l.ceil() }) as i32
    };
    let suggested_bound_scaling = suggest_scaling(max_scalable_bound, SMALL_BOUND, LARGE_BOUND);
    let dl_user_bound_scale = outer_rounded_log(suggested_bound_scaling, 2);
    d.suggested_user_bound_scale = d.user_bound_scale + dl_user_bound_scale;
    let suggested_bound_scale_order_of_magnitude = outer_rounded_log(suggested_bound_scaling, 10);
    let suggested_user_bound_scale_value = 2f64.powf(d.suggested_user_bound_scale as f64);
    min_noncontinuous_col_cost *= suggested_user_bound_scale_value;
    max_noncontinuous_col_cost *= suggested_user_bound_scale_value;
    min_col_cost = cmin(min_continuous_col_cost, min_noncontinuous_col_cost);
    max_col_cost = cmax(max_continuous_col_cost, max_noncontinuous_col_cost);
    let mut min_objective_coefficient = cmin(min_col_cost, min_continuous_hessian_value);
    let mut max_objective_coefficient = cmax(max_col_cost, max_continuous_hessian_value);
    if min_objective_coefficient == INF {
        min_objective_coefficient = 0.0;
    }
    if max_objective_coefficient == -INF {
        max_objective_coefficient = 0.0;
    }
    let _ = min_objective_coefficient;
    let suggested_objective_scaling =
        suggest_scaling(max_objective_coefficient, SMALL_OBJECTIVE_COEFFICIENT, LARGE_OBJECTIVE_COEFFICIENT);
    let dl_user_objective_scale = outer_rounded_log(suggested_objective_scaling, 2);
    d.suggested_user_objective_scale = d.user_objective_scale + dl_user_objective_scale;
    let suggested_objective_scale_order_of_magnitude = outer_rounded_log(suggested_objective_scaling, 10);

    let suggestion = |order: i32, user_scale: i32, dl: i32, suggested: i32, what: &str, option: &str| {
        let order_of_magnitude_message = order != 0 && user_scale == 0;
        let mut message = String::new();
        if order_of_magnitude_message {
            message += &sprintf("   Consider scaling the %s by 1e%+1d", &[what.into(), order.into()]);
        }
        if dl != 0 {
            message += if order_of_magnitude_message { ", or" } else { "   Consider" };
            message += &sprintf(" setting the %s option to %d", &[option.into(), suggested.into()]);
        }
        if order_of_magnitude_message || dl != 0 {
            log_user!(log, LogType::Warning, "%s\n", &*message);
        }
    };
    suggestion(
        suggested_objective_scale_order_of_magnitude,
        d.user_objective_scale,
        dl_user_objective_scale,
        d.suggested_user_objective_scale,
        "objective",
        "user_objective_scale",
    );
    suggestion(
        suggested_bound_scale_order_of_magnitude,
        d.user_bound_scale,
        dl_user_bound_scale,
        d.suggested_user_bound_scale,
        "   bounds",
        "user_bound_scale",
    );
}

/// HighsSparseMatrix::numNz
fn lp_num_nz(lp: &CLp) -> usize {
    use super::matrix_format::COLWISE;
    // SAFETY: the LP's matrix starts
    let start = unsafe { lp.a.start.get() };
    let n = if lp.a.format == COLWISE { lp.a.num_col } else { lp.a.num_row } as usize;
    if start.len() > n {
        start[n] as usize
    } else {
        0
    }
}

// The C++ entry points (highs/lp_data/HighsSolveRust.cpp)

/// # Safety
/// `c` valid, `message` valid for `len` bytes
#[no_mangle]
pub unsafe extern "C" fn highs_rs_solve_lp(c: *const CSolve, message: *const u8, len: usize) -> i32 {
    let message = std::str::from_utf8_unchecked(std::slice::from_raw_parts(message, len));
    (*c).solve_lp(message) as i32
}

/// # Safety
/// The arrays sized to the LP by C++
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn highs_rs_solve_unconstrained_lp(
    log: *const Log,
    on: bool,
    pft: f64,
    dft: f64,
    lp: *const CLp,
    info: *mut Info,
    col_value: super::ffi::RsMut<f64>,
    col_dual: super::ffi::RsMut<f64>,
    row_value: super::ffi::RsMut<f64>,
    row_dual: super::ffi::RsMut<f64>,
    col_status: super::ffi::RsMut<u8>,
    row_status: super::ffi::RsMut<u8>,
) -> i32 {
    solve_unconstrained_lp(
        &*log,
        on,
        pft,
        dft,
        &*lp,
        &mut *info,
        Unconstrained {
            col_value: col_value.get_mut(),
            col_dual: col_dual.get_mut(),
            row_value: row_value.get_mut(),
            row_dual: row_dual.get_mut(),
            col_status: col_status.get_mut(),
            row_status: row_status.get_mut(),
        },
    )
}

/// # Safety
/// `lp` and `hessian_value` valid
#[no_mangle]
pub unsafe extern "C" fn highs_rs_assess_excessive_objective_bound_scaling(
    log: *const Log,
    lp: *const CLp,
    hessian_value: super::ffi::RsMut<f64>,
    user_objective_scale: i32,
    user_bound_scale: i32,
    suggested_user_objective_scale: *mut i32,
    suggested_user_bound_scale: *mut i32,
) {
    let mut d = UserScale {
        user_objective_scale,
        user_bound_scale,
        suggested_user_objective_scale: *suggested_user_objective_scale,
        suggested_user_bound_scale: *suggested_user_bound_scale,
    };
    assess_excessive_objective_bound_scaling(&*log, &*lp, hessian_value.get(), &mut d);
    *suggested_user_objective_scale = d.suggested_user_objective_scale;
    *suggested_user_bound_scale = d.suggested_user_bound_scale;
}
