//! The IPX glue of highs/ipm/IpxWrapper.cpp and presolve/ICrashX.cpp:
//! solveLpIpx (IPX parameters from the options, the LP in IPX form, the
//! solve, the status reports and checks, and the interior or basic
//! solution in HiGHS form), fillInIpxData and callCrossover. IPX itself is
//! the Rust port (crate::ipx).
//!
//! The caller (an LP run on Rust data, lp_run.rs, or a handle's crossover)
//! passes the options, the LP view, the HighsInfo, model status and
//! validity flags in place, and callbacks for the timer and for sizing the
//! solution and basis vectors at the points the C++ resized them; the IPX
//! hooks (logging, task and user interrupt) are the handle's. A cancelled
//! task returns `CANCELLED` and C++ throws HighsTask::Interrupt.

use super::basis::{ipx_basic_solution_to_highs_basic_solution, ipx_solution_to_highs_solution, COut, IpxSolution};
use super::ffi::CLp;
use super::run::{MS_INFEASIBLE, MS_INTERRUPT, MS_ITERATION_LIMIT, MS_OPTIMAL, MS_SOLVE_ERROR, MS_TIME_LIMIT, MS_UNBOUNDED_OR_INFEASIBLE, MS_UNKNOWN};
use super::solution::Info;
use super::solve::reset_model_status_and_info;
use super::{Log, LogType, Status, INF};
use crate::ipx::{self, Hooks, LpSolver, Parameters};
use crate::util::fma::ClangFma;
use crate::{log_dev, log_user};
use std::ffi::c_void;

/// Returned when the HiGHS task running IPX was cancelled
pub const CANCELLED: i32 = 2;

// IPX status codes (ipx_status.h)
const NOT_RUN: i32 = 0;
const SOLVED: i32 = 1000;
const STOPPED: i32 = 1005;
const NO_MODEL: i32 = 1006;
const OUT_OF_MEMORY: i32 = 1003;
const INTERNAL_ERROR: i32 = 1004;
const OPTIMAL: i32 = 1;
const IMPRECISE: i32 = 2;
const PRIMAL_INFEAS: i32 = 3;
const DUAL_INFEAS: i32 = 4;
const USER_INTERRUPT: i32 = 5;
const TIME_LIMIT: i32 = 6;
const ITER_LIMIT: i32 = 7;
const NO_PROGRESS: i32 = 8;
const FAILED: i32 = 9;
const DEBUG: i32 = 10;

const KKT_TOLERANCE_DEFAULT: f64 = 1e-7;
const ANALYSIS_SOLVER_SUMMARY: i32 = 2;
const ANALYSIS_NLA_DATA: i32 = 16;
const BASIS_VALIDITY_INVALID: i32 = 0;
const BASIS_VALIDITY_VALID: i32 = 1;

/// The LP in IPX form (fillInIpxData's outputs). The capacities are those
/// of the C++ vectors, so an empty one is a null pointer to IPX as there.
pub struct IpxData {
    pub num_col: i32,
    pub num_row: i32,
    pub offset: f64,
    pub obj: Vec<f64>,
    pub col_lb: Vec<f64>,
    pub col_ub: Vec<f64>,
    pub ap: Vec<i32>,
    pub ai: Vec<i32>,
    pub ax: Vec<f64>,
    pub rhs: Vec<f64>,
    pub constraint_type: Vec<u8>,
}

/// fillInRhsAndConstraints
fn fill_in_rhs_and_constraints(row_lower: &[f64], row_upper: &[f64], rhs: &mut Vec<f64>, types: &mut Vec<u8>) {
    let num_row = row_lower.len();
    rhs.reserve(num_row);
    types.reserve(num_row);
    for row in 0..num_row {
        let (lo, up) = (row_lower[row], row_upper[row]);
        if lo > -INF && up >= INF {
            rhs.push(lo);
            types.push(b'>');
        } else if lo <= -INF && up < INF {
            rhs.push(up);
            types.push(b'<');
        } else if lo == up {
            rhs.push(up);
            types.push(b'=');
        } else if lo > -INF && up < INF {
            rhs.push(0.0);
            types.push(b'=');
        }
    }
}

/// fillInIpxData: a slack for each boxed row, free rows dropped
pub fn fill_in_ipx_data(lp: &CLp) -> IpxData {
    // SAFETY: the LP's arrays, read only
    let (start, index, value, col_cost, col_lower, col_upper, row_lower, row_upper) = unsafe {
        (
            lp.a.start.get(),
            lp.a.index.get(),
            lp.a.value.get(),
            lp.col_cost.get(),
            lp.col_lower.get(),
            lp.col_upper.get(),
            &lp.row_lower.get()[..lp.num_row as usize],
            &lp.row_upper.get()[..lp.num_row as usize],
        )
    };
    let lp_num_col = lp.num_col as usize;
    let lp_num_row = lp.num_row as usize;
    let mut general_bounded_rows = Vec::new();
    let mut free_rows = Vec::new();
    for row in 0..lp_num_row {
        let (lo, up) = (row_lower[row], row_upper[row]);
        if lo < up && lo > -INF && up < INF {
            general_bounded_rows.push(row);
        } else if lo <= -INF && up >= INF {
            free_rows.push(row);
        }
    }
    let num_slack = general_bounded_rows.len();
    let mut rhs = Vec::new();
    let mut constraint_type = Vec::new();
    fill_in_rhs_and_constraints(row_lower, row_upper, &mut rhs, &mut constraint_type);

    let mut reduced_rowmap = vec![-1i32; lp_num_row];
    if !free_rows.is_empty() {
        let (mut counter, mut findex) = (0, 0);
        for (row, r) in reduced_rowmap.iter_mut().enumerate() {
            // The C++ reads free_rows[findex] past its end after the last
            // free row; such a read is taken as no match
            if free_rows.get(findex) == Some(&row) {
                findex += 1;
            } else {
                *r = counter;
                counter += 1;
            }
        }
    } else {
        for (k, r) in reduced_rowmap.iter_mut().enumerate() {
            *r = k as i32;
        }
    }
    let num_row = lp_num_row - free_rows.len();
    let num_col = lp_num_col + num_slack;
    let kept = |row: usize| row_lower[row] > -INF || row_upper[row] < INF;

    let mut sizes = vec![0i32; num_col];
    for col in 0..lp_num_col {
        for k in start[col] as usize..start[col + 1] as usize {
            if kept(index[k] as usize) {
                sizes[col] += 1;
            }
        }
    }
    let nnz = index.len();
    let mut ap = vec![0i32; num_col + 1];
    let mut ai = Vec::with_capacity(nnz + num_slack);
    let mut ax = Vec::with_capacity(nnz + num_slack);
    for col in 0..lp_num_col {
        ap[col + 1] = ap[col] + sizes[col];
    }
    for col in lp_num_col..num_col {
        ap[col + 1] = ap[col] + 1;
    }
    for k in 0..nnz {
        let row = index[k] as usize;
        if kept(row) {
            ai.push(reduced_rowmap[row]);
            ax.push(value[k]);
        }
    }
    // As the C++: the slack's row index is not mapped past free rows
    for &row in &general_bounded_rows {
        ai.push(row as i32);
        ax.push(-1.0);
    }

    let mut col_lb = vec![0.0; num_col];
    let mut col_ub = vec![0.0; num_col];
    for col in 0..lp_num_col {
        col_lb[col] = if col_lower[col] <= -INF { -INF } else { col_lower[col] };
        col_ub[col] = if col_upper[col] >= INF { INF } else { col_upper[col] };
    }
    for (slack, &row) in general_bounded_rows.iter().enumerate() {
        col_lb[lp_num_col + slack] = row_lower[row];
        col_ub[lp_num_col + slack] = row_upper[row];
    }
    let sense = lp.sense as f64;
    let mut obj = vec![0.0; num_col];
    for col in 0..lp_num_col {
        obj[col] = sense * col_cost[col];
    }
    IpxData {
        num_col: num_col as i32,
        num_row: num_row as i32,
        offset: sense * lp.offset,
        obj,
        col_lb,
        col_ub,
        ap,
        ai,
        ax,
        rhs,
        constraint_type,
    }
}

/// A C++ vector's data(): null when nothing was ever allocated
fn data<T>(v: &[T], cap: usize) -> *const T {
    if cap == 0 {
        std::ptr::null()
    } else {
        v.as_ptr()
    }
}

/// The options solveLpIpx reads
#[repr(C)]
pub struct CIpxOptions {
    pub log: Log,
    pub log_options: *const c_void,
    pub output_flag: bool,
    pub log_to_console: bool,
    pub timeless_log: bool,
    pub run_centring: bool,
    pub log_dev_level: i32,
    pub ipx_dualize_strategy: i32,
    pub highs_analysis_level: i32,
    pub ipm_iteration_limit: i32,
    /// 1 on, 0 off, -1 choose
    pub run_crossover: i32,
    pub max_centring_steps: i32,
    pub primal_feasibility_tolerance: f64,
    pub dual_feasibility_tolerance: f64,
    pub ipm_optimality_tolerance: f64,
    pub start_crossover_tolerance: f64,
    pub kkt_tolerance: f64,
    pub time_limit: f64,
    pub centring_ratio_tolerance: f64,
}

/// What solveLpIpx works on
#[repr(C)]
pub struct CIpxHost {
    pub ctx: *mut c_void,
    /// timer.read()
    pub timer_read: unsafe extern "C" fn(*mut c_void) -> f64,
    /// Resize the solution (and with_basis the basis) to the LP and
    /// return the views
    pub resize: unsafe extern "C" fn(*mut c_void, bool, *mut COut),
    pub hooks: Hooks,
    pub lp: CLp,
    pub options: CIpxOptions,
    pub info: *mut Info,
    pub model_status: *mut i32,
    pub value_valid: *mut bool,
    pub dual_valid: *mut bool,
    pub basis_valid: *mut bool,
    pub basis_useful: *mut bool,
}

// The C layouts
const _: () = assert!(std::mem::size_of::<CIpxOptions>() == 112);
const _: () = assert!(std::mem::size_of::<CIpxHost>() == 64 + std::mem::size_of::<CLp>() + 112 + 48);

extern "C" {
    fn fflush(f: *mut c_void) -> i32;
}

/// reportIpxSolveStatus
fn report_solve_status(log: &Log, solve_status: i32, error_flag: i32) -> Status {
    let e = LogType::Error;
    match solve_status {
        SOLVED => {
            log_user!(log, LogType::Info, "Ipx: Solved\n");
            return Status::Ok;
        }
        STOPPED => {
            log_user!(log, LogType::Warning, "Ipx: Stopped\n");
            return Status::Warning;
        }
        NO_MODEL => log_user!(
            log,
            e,
            match error_flag {
                102 => "Ipx: Invalid input - argument_null\n",
                103 => "Ipx: Invalid input - invalid dimension\n",
                104 => "Ipx: Invalid input - invalid matrix\n",
                105 => "Ipx: Invalid input - invalid vector\n",
                107 => "Ipx: Invalid input - invalid basis\n",
                _ => "Ipx: Invalid input - unrecognised error\n",
            }
        ),
        OUT_OF_MEMORY => log_user!(log, e, "Ipx: Out of memory\n"),
        INTERNAL_ERROR => log_user!(log, e, "Ipx: Internal error %d\n", error_flag),
        _ => log_user!(log, e, "Ipx: unrecognised solve status = %d\n", solve_status),
    }
    Status::Error
}

/// reportIpxIpmCrossoverStatus
fn report_ipm_crossover_status(log: &Log, run_crossover_on: bool, status: i32, ipm_status: bool) -> Status {
    let name = if ipm_status { "IPM      " } else { "Crossover" };
    let (t, what, ret) = match status {
        NOT_RUN => {
            if ipm_status || run_crossover_on {
                log_user!(log, LogType::Warning, "Ipx: %s not run\n", name);
                return Status::Warning;
            }
            return Status::Ok;
        }
        OPTIMAL => (LogType::Info, "optimal", Status::Ok),
        IMPRECISE => (LogType::Warning, "imprecise", Status::Warning),
        PRIMAL_INFEAS => (LogType::Warning, "primal infeasible", Status::Warning),
        DUAL_INFEAS => (LogType::Warning, "dual infeasible", Status::Warning),
        USER_INTERRUPT => (LogType::Warning, "user interrupt", Status::Ok),
        TIME_LIMIT => (LogType::Warning, "reached time limit", Status::Warning),
        ITER_LIMIT => (LogType::Warning, "reached iteration limit", Status::Warning),
        NO_PROGRESS => (LogType::Warning, "no progress", Status::Warning),
        FAILED => (LogType::Error, "failed", Status::Error),
        DEBUG => (LogType::Error, "debug", Status::Error),
        _ => (LogType::Error, "unrecognised status", Status::Error),
    };
    log_user!(log, t, "Ipx: %s %s\n", name, what);
    ret
}

/// ipxStatusError (value < 0: no value)
fn status_error(log: &Log, error: bool, message: &str, value: i32) -> bool {
    if error {
        if value < 0 {
            log_user!(log, LogType::Error, "%s: %s\n", "Ipx", message);
        } else {
            log_user!(log, LogType::Error, "%s: %s %d\n", "Ipx", message, value);
        }
        // SAFETY: fflush(NULL) flushes all C streams
        unsafe { fflush(std::ptr::null_mut()) };
    }
    error
}

/// illegalIpxSolvedStatus, illegalIpxStoppedIpmStatus and
/// illegalIpxStoppedCrossoverStatus: the first status (ipm or crossover)
/// in `checks` that is set is reported
fn illegal(log: &Log, prefix: &str, checks: &[(bool, i32, &str)], info: &ipx::Info) -> bool {
    for &(ipm, status, name) in checks {
        let value = if ipm { info.status_ipm } else { info.status_crossover };
        let which = if ipm { "status_ipm" } else { "status_crossover" };
        if status_error(log, value == status, &format!("{prefix} {which} should not be IPX_STATUS_{name}"), -1) {
            return true;
        }
    }
    false
}

fn illegal_solved(log: &Log, info: &ipx::Info) -> bool {
    illegal(
        log,
        "solved ",
        &[
            (true, TIME_LIMIT, "time_limit"),
            (true, ITER_LIMIT, "iter_limit"),
            (true, NO_PROGRESS, "no_progress"),
            (true, FAILED, "failed"),
            (true, DEBUG, "debug"),
            (false, PRIMAL_INFEAS, "primal_infeas"),
            (false, DUAL_INFEAS, "dual_infeas"),
            (false, TIME_LIMIT, "time_limit"),
            (false, ITER_LIMIT, "iter_limit"),
            (false, NO_PROGRESS, "no_progress"),
            (false, FAILED, "failed"),
            (false, DEBUG, "debug"),
        ],
        info,
    )
}

fn illegal_stopped_ipm(log: &Log, info: &ipx::Info) -> bool {
    illegal(
        log,
        "stopped",
        &[
            (true, OPTIMAL, "optimal"),
            (true, IMPRECISE, "imprecise"),
            (true, PRIMAL_INFEAS, "primal_infeas"),
            (true, DUAL_INFEAS, "dual_infeas"),
            (true, FAILED, "failed"),
            (true, DEBUG, "debug"),
        ],
        info,
    )
}

fn illegal_stopped_crossover(log: &Log, info: &ipx::Info) -> bool {
    illegal(
        log,
        "stopped",
        &[
            (false, OPTIMAL, "optimal"),
            (false, IMPRECISE, "imprecise"),
            (false, PRIMAL_INFEAS, "primal_infeas"),
            (false, DUAL_INFEAS, "dual_infeas"),
            (false, ITER_LIMIT, "iter_limit"),
            (false, NO_PROGRESS, "no_progress"),
            (false, FAILED, "failed"),
            (false, DEBUG, "debug"),
        ],
        info,
    )
}

/// reportSolveData (highs_analysis_level has kHighsAnalysisLevelSolverSummaryData)
fn report_solve_data(log: &Log, i: &ipx::Info) {
    let t = LogType::Info;
    log_dev!(log, t, "\nIPX Solve data\n");
    log_dev!(log, t, "    IPX       status = %4d\n", i.status);
    log_dev!(log, t, "    IPM       status = %4d\n", i.status_ipm);
    log_dev!(log, t, "    Crossover status = %4d\n", i.status_crossover);
    log_dev!(log, t, "    IPX errflag      = %4d\n\n", i.errflag);
    log_dev!(log, t, "    LP variables   = %8d\n", i.num_var);
    log_dev!(log, t, "    LP constraints = %8d\n", i.num_constr);
    log_dev!(log, t, "    LP entries     = %8d\n\n", i.num_entries);
    log_dev!(log, t, "    Solver columns = %8d\n", i.num_cols_solver);
    log_dev!(log, t, "    Solver rows    = %8d\n", i.num_rows_solver);
    log_dev!(log, t, "    Solver entries = %8d\n\n", i.num_entries_solver);
    log_dev!(log, t, "    Dualized = %d\n", i.dualized);
    log_dev!(log, t, "    Number of dense columns detected = %d\n\n", i.dense_cols);
    log_dev!(log, t, "    Dependent rows    = %d\n", i.dependent_rows);
    log_dev!(log, t, "    Dependent cols    = %d\n", i.dependent_cols);
    log_dev!(log, t, "    Inconsistent rows = %d\n", i.rows_inconsistent);
    log_dev!(log, t, "    Inconsistent cols = %d\n", i.cols_inconsistent);
    log_dev!(log, t, "    Primal dropped    = %d\n", i.primal_dropped);
    log_dev!(log, t, "    Dual   dropped    = %d\n\n", i.dual_dropped);
    log_dev!(log, t, "    |Absolute primal residual| = %11.4g\n", i.abs_presidual);
    log_dev!(log, t, "    |Absolute   dual residual| = %11.4g\n", i.abs_dresidual);
    log_dev!(log, t, "    |Relative primal residual| = %11.4g\n", i.rel_presidual);
    log_dev!(log, t, "    |Relative   dual residual| = %11.4g\n\n", i.rel_dresidual);
    log_dev!(log, t, "    Primal objective value     = %11.4g\n", i.pobjval);
    log_dev!(log, t, "    Dual   objective value     = %11.4g\n", i.dobjval);
    log_dev!(log, t, "    Relative objective gap     = %11.4g\n", i.rel_objgap);
    log_dev!(log, t, "    Complementarity            = %11.4g\n\n", i.complementarity);
    log_dev!(log, t, "    |x| = %11.4g\n", i.normx);
    log_dev!(log, t, "    |y| = %11.4g\n", i.normy);
    log_dev!(log, t, "    |z| = %11.4g\n\n", i.normz);
    log_dev!(log, t, "    Objective value       = %11.4g\n", i.objval);
    log_dev!(log, t, "    Primal infeasibility = %11.4g\n", i.primal_infeas);
    log_dev!(log, t, "    Dual infeasibility   = %11.4g\n\n", i.dual_infeas);
    log_dev!(log, t, "    IPM iter   = %d\n", i.iter);
    log_dev!(log, t, "    KKT iter 1 = %d\n", i.kktiter1);
    log_dev!(log, t, "    KKT iter 2 = %d\n", i.kktiter2);
    log_dev!(log, t, "    Basis repairs = %d\n", i.basis_repairs);
    log_dev!(log, t, "    Updates start     = %d\n", i.updates_start);
    log_dev!(log, t, "    Updates ipm       = %d\n", i.updates_ipm);
    log_dev!(log, t, "    Updates crossover = %d\n\n", i.updates_crossover);
    log_dev!(log, t, "    Time total          = %8.2f\n\n", i.time_total);
    let sum = 0.0 + i.time_ipm1 + i.time_ipm2 + i.time_starting_basis;
    log_dev!(log, t, "    Time IPM 1          = %8.2f\n", i.time_ipm1);
    log_dev!(log, t, "    Time IPM 2          = %8.2f\n", i.time_ipm2);
    log_dev!(log, t, "    Time starting basis = %8.2f\n", i.time_starting_basis);
    log_dev!(log, t, "    Time crossover      = %8.2f\n", i.time_crossover);
    log_dev!(log, t, "    Sum                 = %8.2f\n\n", sum);
    log_dev!(log, t, "    Time kkt_factorize  = %8.2f\n", i.time_kkt_factorize);
    log_dev!(log, t, "    Time kkt_solve      = %8.2f\n", i.time_kkt_solve);
    log_dev!(log, t, "    Sum                 = %8.2f\n\n", 0.0 + i.time_kkt_factorize + i.time_kkt_solve);
    log_dev!(log, t, "    Time maxvol         = %8.2f\n", i.time_maxvol);
    log_dev!(log, t, "    Time cr1            = %8.2f\n", i.time_cr1);
    log_dev!(log, t, "    Time cr2            = %8.2f\n", i.time_cr2);
    log_dev!(log, t, "    Sum                 = %8.2f\n\n", 0.0 + i.time_maxvol + i.time_cr1 + i.time_cr2);
    log_dev!(log, t, "    Time cr1_AAt        = %8.2f\n", i.time_cr1_aat);
    log_dev!(log, t, "    Time cr1_pre        = %8.2f\n", i.time_cr1_pre);
    log_dev!(log, t, "    Sum  cr1            = %8.2f\n\n", 0.0 + i.time_cr1_aat + i.time_cr1_pre);
    log_dev!(log, t, "    Time cr2_NNt        = %8.2f\n", i.time_cr2_nnt);
    log_dev!(log, t, "    Time cr2_B          = %8.2f\n", i.time_cr2_b);
    log_dev!(log, t, "    Time cr2_Bt         = %8.2f\n", i.time_cr2_bt);
    log_dev!(log, t, "    Sum  cr2            = %8.2f\n\n", 0.0 + i.time_cr2_nnt + i.time_cr2_b + i.time_cr2_bt);
    log_dev!(log, t, "    Proportion of sparse FTRAN = %11.4g\n", i.ftran_sparse);
    log_dev!(log, t, "    Proportion of sparse BTRAN = %11.4g\n\n", i.btran_sparse);
    log_dev!(log, t, "    Time FTRAN       = %8.2f\n", i.time_ftran);
    log_dev!(log, t, "    Time BTRAN       = %8.2f\n", i.time_btran);
    log_dev!(log, t, "    Time LU INVERT   = %8.2f\n", i.time_lu_invert);
    log_dev!(log, t, "    Time LU UPDATE   = %8.2f\n", i.time_lu_update);
    log_dev!(log, t, "    Mean fill-in     = %11.4g\n", i.mean_fill);
    log_dev!(log, t, "    Max fill-in      = %11.4g\n", i.max_fill);
    log_dev!(log, t, "    Time symb INVERT = %11.4g\n\n", i.time_symb_invert);
    log_dev!(log, t, "    Maxvol updates       = %d\n", i.maxvol_updates);
    log_dev!(log, t, "    Maxvol skipped       = %d\n", i.maxvol_skipped);
    log_dev!(log, t, "    Maxvol passes        = %d\n", i.maxvol_passes);
    log_dev!(log, t, "    Tableau num nonzeros = %d\n", i.tbl_nnz);
    log_dev!(log, t, "    Tbl max?             = %11.4g\n", i.tbl_max);
    log_dev!(log, t, "    Frobnorm squared     = %11.4g\n", i.frobnorm_squared);
    log_dev!(log, t, "    Lambda max           = %11.4g\n", i.lambdamax);
    log_dev!(log, t, "    Volume increase      = %11.4g\n\n", i.volume_increase);
}

/// The IPX parameters from the options (solveLpIpx)
fn parameters(o: &CIpxOptions, ipm_iteration_count: i32) -> Parameters {
    let mut p = Parameters::default();
    p.display = 1;
    if !o.output_flag | !o.log_to_console {
        p.display = 0;
    }
    p.debug = match o.log_dev_level {
        1 => 2,
        3 => 4,
        _ => 0,
    };
    p.highs_logging = true;
    p.timeless_log = o.timeless_log;
    p.log_options = o.log_options;
    p.dualize = match o.ipx_dualize_strategy {
        1 => 1,
        -1 => 0,
        2 => -1,
        3 => -2,
        _ => p.dualize,
    };
    let (pft, dft) = (o.primal_feasibility_tolerance, o.dual_feasibility_tolerance);
    p.ipm_feasibility_tol = if dft < pft { dft } else { pft };
    p.ipm_optimality_tol = o.ipm_optimality_tolerance;
    p.start_crossover_tol = o.start_crossover_tolerance;
    if o.kkt_tolerance != KKT_TOLERANCE_DEFAULT {
        p.ipm_feasibility_tol = o.kkt_tolerance;
        p.ipm_optimality_tol = 1e-1 * o.kkt_tolerance;
        p.start_crossover_tol = 1e-1 * o.kkt_tolerance;
    }
    p.analyse_basis_data = ANALYSIS_NLA_DATA & o.highs_analysis_level != 0;
    p.time_limit = o.time_limit;
    p.ipm_maxiter = o.ipm_iteration_limit - ipm_iteration_count;
    p.run_crossover = if o.run_centring { 0 } else { o.run_crossover };
    if p.run_crossover == 0 {
        p.start_crossover_tol = -1.0;
    }
    p.run_centring = o.run_centring as i32;
    p.max_centring_steps = o.max_centring_steps;
    p.centring_ratio_tolerance = o.centring_ratio_tolerance;
    p
}

/// solveLpIpx: returns the HighsStatus, or `CANCELLED`
///
/// # Safety
/// The host's views and pointers valid, the callbacks with their context
pub unsafe fn solve_lp_ipx(h: &CIpxHost) -> i32 {
    let o = &h.options;
    let log = &o.log;
    let lp = &h.lp;
    let info = &mut *h.info;
    let model_status = &mut *h.model_status;
    *h.basis_valid = false;
    *h.value_valid = false;
    *h.dual_valid = false;
    reset_model_status_and_info(model_status, info);
    let mut lps = LpSolver::new();
    lps.set_hooks(h.hooks);
    lps.set_timer_offset((h.timer_read)(h.ctx));
    lps.set_parameters(parameters(o, info.ipm_iteration_count));

    let d = fill_in_ipx_data(lp);
    log_user!(
        log,
        LogType::Info,
        "IPX model has %d rows, %d columns and %d nonzeros\n",
        d.num_row,
        d.num_col,
        d.ap[d.num_col as usize]
    );
    // SAFETY: lps is an LpSolver; the arrays have the sizes IPX documents
    let load_status = ipx::ffi::ipx_rs_load_model(
        &mut lps as *mut LpSolver as *mut c_void,
        d.num_col,
        d.offset,
        data(&d.obj, d.obj.capacity()),
        data(&d.col_lb, d.col_lb.capacity()),
        data(&d.col_ub, d.col_ub.capacity()),
        d.num_row,
        data(&d.ap, d.ap.capacity()),
        data(&d.ai, d.ai.capacity()),
        data(&d.ax, d.ax.capacity()),
        data(&d.rhs, d.rhs.capacity()),
        data(&d.constraint_type, d.constraint_type.capacity()) as *const std::ffi::c_char,
    );
    if load_status != 0 {
        *model_status = MS_SOLVE_ERROR;
        return Status::Error as i32;
    }
    let solve_status = lps.solve();
    if lps.cancelled() {
        return CANCELLED;
    }
    let ii = lps.get_info();
    if ANALYSIS_SOLVER_SUMMARY & o.highs_analysis_level != 0 {
        report_solve_data(log, &ii);
    }
    info.ipm_iteration_count += ii.iter;
    info.crossover_iteration_count += ii.updates_crossover;

    if solve_status != SOLVED && report_solve_status(log, solve_status, ii.errflag) == Status::Error {
        *model_status = MS_SOLVE_ERROR;
        return Status::Error as i32;
    }
    let crossover_on = o.run_crossover == 1;
    let ipm_return = report_ipm_crossover_status(log, crossover_on, ii.status_ipm, true);
    let crossover_return = report_ipm_crossover_status(log, crossover_on, ii.status_crossover, false);
    if ipm_return == Status::Error || crossover_return == Status::Error {
        *model_status = MS_SOLVE_ERROR;
        return Status::Error as i32;
    }
    if status_error(
        log,
        solve_status != SOLVED && solve_status != STOPPED,
        "solve_status should be solved or stopped here but value is",
        solve_status,
    ) {
        return Status::Error as i32;
    }

    // The interior solution (getHighsNonVertexSolution)
    let non_vertex = || {
        let (n, m) = (d.num_col as usize, d.num_row as usize);
        let (mut x, mut xl, mut xu, mut zl, mut zu) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let (mut slack, mut y) = (vec![0.0; m], vec![0.0; m]);
        lps.get_interior_solution([
            Some(&mut x),
            Some(&mut xl),
            Some(&mut xu),
            Some(&mut slack),
            Some(&mut y),
            Some(&mut zl),
            Some(&mut zu),
        ]);
        let mut out = std::mem::zeroed::<COut>();
        (h.resize)(h.ctx, false, &mut out);
        ipx_solution_to_highs_solution(lp, &d.rhs, d.num_row, &x, &slack, &y, &zl, &zu, &mut out.view());
        *h.value_valid = true;
        *h.dual_valid = true;
    };

    if solve_status == STOPPED {
        non_vertex();
        if illegal_stopped_crossover(log, &ii) {
            return Status::Error as i32;
        }
        if ii.status_crossover == TIME_LIMIT {
            *model_status = MS_TIME_LIMIT;
            return Status::Warning as i32;
        }
        if illegal_stopped_ipm(log, &ii) {
            return Status::Error as i32;
        }
        *model_status = match ii.status_ipm {
            USER_INTERRUPT => MS_INTERRUPT,
            TIME_LIMIT => MS_TIME_LIMIT,
            ITER_LIMIT => MS_ITERATION_LIMIT,
            _ => {
                let w = LogType::Warning;
                log_user!(log, w, "No progress: primal objective value       = %11.4g\n", ii.pobjval);
                log_user!(log, w, "No progress: max absolute primal residual = %11.4g\n", ii.abs_presidual);
                log_user!(log, w, "No progress: max absolute   dual residual = %11.4g\n", ii.abs_dresidual);
                MS_UNKNOWN
            }
        };
        return Status::Warning as i32;
    }
    if status_error(log, solve_status != SOLVED, "solve_status should be solved here but value is", solve_status) {
        return Status::Error as i32;
    }
    if illegal_solved(log, &ii) {
        return Status::Error as i32;
    }
    if ii.status_ipm == PRIMAL_INFEAS || ii.status_ipm == DUAL_INFEAS {
        *model_status = if ii.status_ipm == PRIMAL_INFEAS { MS_INFEASIBLE } else { MS_UNBOUNDED_OR_INFEASIBLE };
        non_vertex();
        return Status::Ok as i32;
    }
    if status_error(
        log,
        ii.status_ipm != OPTIMAL && ii.status_ipm != IMPRECISE,
        "ipm status should be not run, optimal or imprecise but value is",
        ii.status_ipm,
    ) {
        return Status::Error as i32;
    }
    if status_error(
        log,
        ii.status_crossover != NOT_RUN && ii.status_crossover != OPTIMAL && ii.status_crossover != IMPRECISE,
        "crossover status should be not run, optimal or imprecise but value is",
        ii.status_crossover,
    ) {
        return Status::Error as i32;
    }
    let have_basic_solution = ii.status_crossover != NOT_RUN;
    let imprecise = ii.status_crossover == IMPRECISE || ii.status_ipm == IMPRECISE;
    if have_basic_solution {
        let (n, m) = (d.num_col as usize, d.num_row as usize);
        let (mut cv, mut rv, mut cd, mut rd) = (vec![0.0; n], vec![0.0; m], vec![0.0; n], vec![0.0; m]);
        let (mut rs, mut cs) = (vec![0i32; m], vec![0i32; n]);
        let errflag = lps.get_basic_solution(
            Some(&mut cv),
            Some(&mut rv),
            Some(&mut rd),
            Some(&mut cd),
            Some(&mut rs),
            Some(&mut cs),
        );
        if errflag != 0 {
            log_user!(log, LogType::Error, "IPX crossover getting basic solution: flag = %d\n", errflag);
            return Status::Error as i32;
        }
        let ipx_solution = IpxSolution {
            num_col: d.num_col,
            num_row: d.num_row,
            col_value: &cv,
            row_value: &rv,
            col_dual: &cd,
            row_dual: &rd,
            col_status: &cs,
            row_status: &rs,
        };
        let mut out = std::mem::zeroed::<COut>();
        (h.resize)(h.ctx, true, &mut out);
        let status =
            ipx_basic_solution_to_highs_basic_solution(log, lp, &d.rhs, &d.constraint_type, &ipx_solution, &mut out.view());
        if status != Status::Ok {
            // ipxBasicSolutionToHighsBasicSolution returns OK or Error
            log_user!(log, LogType::Error, "Failed to convert IPX basic solution to Highs basic solution\n");
            return Status::Error as i32;
        }
        *h.value_valid = true;
        *h.dual_valid = true;
        *h.basis_valid = true;
        *h.basis_useful = true;
    } else {
        non_vertex();
    }
    info.basis_validity = if *h.basis_valid { BASIS_VALIDITY_VALID } else { BASIS_VALIDITY_INVALID };
    if imprecise {
        *model_status = MS_UNKNOWN;
        Status::Warning as i32
    } else {
        *model_status = MS_OPTIMAL;
        Status::Ok as i32
    }
}

/// # Safety
/// As solve_lp_ipx
#[no_mangle]
pub unsafe extern "C" fn highs_rs_solve_lp_ipx(h: *const CIpxHost) -> i32 {
    solve_lp_ipx(&*h)
}

/// std::max and std::min of doubles
fn cpp_max(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}
fn cpp_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// A C++ vector's data() as IPX reads it: null when empty
fn opt(v: &[f64]) -> Option<&[f64]> {
    (!v.is_empty()).then_some(v)
}

/// callCrossover (ICrashX.cpp): IPX crossover from the primal point
/// `col_value` (and the duals `(row_dual, col_dual)` if given), the basic
/// solution in HiGHS form; returns the HighsStatus, or `CANCELLED`. Of the
/// host's options only the log, log options, output_flag and
/// log_dev_level are read.
///
/// # Safety
/// As solve_lp_ipx
pub unsafe fn call_crossover(h: &CIpxHost, col_value: &[f64], duals: Option<(&[f64], &[f64])>) -> i32 {
    let o = &h.options;
    let log = &o.log;
    let lp = &h.lp;
    let d = fill_in_ipx_data(lp);
    let mut p = Parameters::default();
    p.run_crossover = 1;
    p.crash_basis = 1;
    p.display = o.output_flag as i32;
    p.debug = match o.log_dev_level {
        1 => 2,
        3 => 4,
        _ => 0,
    };
    p.highs_logging = true;
    p.log_options = o.log_options;
    let mut lps = LpSolver::new();
    lps.set_hooks(h.hooks);
    lps.set_parameters(p);
    // SAFETY: lps is an LpSolver; the arrays have the sizes IPX documents
    let load_status = ipx::ffi::ipx_rs_load_model(
        &mut lps as *mut LpSolver as *mut c_void,
        d.num_col,
        d.offset,
        data(&d.obj, d.obj.capacity()),
        data(&d.col_lb, d.col_lb.capacity()),
        data(&d.col_ub, d.col_ub.capacity()),
        d.num_row,
        data(&d.ap, d.ap.capacity()),
        data(&d.ai, d.ai.capacity()),
        data(&d.ax, d.ax.capacity()),
        data(&d.rhs, d.rhs.capacity()),
        data(&d.constraint_type, d.constraint_type.capacity()) as *const std::ffi::c_char,
    );
    if load_status != 0 {
        log_user!(log, LogType::Error, "Error loading ipx model\n");
        return Status::Error as i32;
    }
    let (n, m) = (d.num_col as usize, d.num_row as usize);
    // x within its bounds (a short point is padded with zeros)
    let mut x = col_value.to_vec();
    x.resize(x.len().max(n), 0.0);
    for i in 0..n {
        x[i] = cpp_min(cpp_max(x[i], d.col_lb[i]), d.col_ub[i]);
    }
    // The slacks rhs - A*x, subject to the sign conditions
    let mut slack = d.rhs.clone();
    for i in 0..n {
        for p in d.ap[i] as usize..d.ap[i + 1] as usize {
            // clang fuses `slack[Ai[p]] -= Av[p] * x[i]`
            let r = d.ai[p] as usize;
            slack[r] = (-d.ax[p]).mul_add_c(x[i], slack[r]);
        }
    }
    for i in 0..m {
        match d.constraint_type[i] {
            b'=' => slack[i] = 0.0,
            b'<' => slack[i] = cpp_max(slack[i], 0.0),
            b'>' => slack[i] = cpp_min(slack[i], 0.0),
            _ => {}
        }
    }
    let crossover_status = match duals {
        Some((row_dual, col_dual)) => {
            log_user!(log, LogType::Info, "Calling IPX crossover with primal and dual values\n");
            lps.crossover_from_starting_point(opt(&x[..n]), opt(&slack), opt(row_dual), opt(col_dual))
        }
        None => {
            log_user!(log, LogType::Info, "Calling IPX crossover with only primal values\n");
            lps.crossover_from_starting_point(opt(&x[..n]), opt(&slack), None, None)
        }
    };
    if lps.cancelled() {
        return CANCELLED;
    }
    if crossover_status != 0 {
        log_user!(log, LogType::Error, "IPX crossover error: flag = %d\n", crossover_status);
        return Status::Error as i32;
    }
    let ii = lps.get_info();
    let info = &mut *h.info;
    info.crossover_iteration_count += ii.updates_crossover;
    let imprecise = ii.status_crossover == IMPRECISE;
    if ii.status_crossover != OPTIMAL && ii.status_crossover != IMPRECISE && ii.status_crossover != TIME_LIMIT {
        log_user!(log, LogType::Error, "IPX crossover failed: status = %d\n", ii.status_crossover);
        return Status::Error as i32;
    }
    if ii.status_crossover == TIME_LIMIT {
        *h.model_status = MS_TIME_LIMIT;
        return Status::Warning as i32;
    }
    let (mut cv, mut rv, mut cd, mut rd) = (vec![0.0; n], vec![0.0; m], vec![0.0; n], vec![0.0; m]);
    let (mut rs, mut cs) = (vec![0i32; m], vec![0i32; n]);
    let errflag =
        lps.get_basic_solution(Some(&mut cv), Some(&mut rv), Some(&mut rd), Some(&mut cd), Some(&mut rs), Some(&mut cs));
    if errflag != 0 {
        log_user!(log, LogType::Error, "IPX crossover getting basic solution: flag = %d\n", errflag);
        return Status::Error as i32;
    }
    let ipx_solution = IpxSolution {
        num_col: d.num_col,
        num_row: d.num_row,
        col_value: &cv,
        row_value: &rv,
        col_dual: &cd,
        row_dual: &rd,
        col_status: &cs,
        row_status: &rs,
    };
    let mut out = std::mem::zeroed::<COut>();
    (h.resize)(h.ctx, true, &mut out);
    if ipx_basic_solution_to_highs_basic_solution(log, lp, &d.rhs, &d.constraint_type, &ipx_solution, &mut out.view())
        == Status::Error
    {
        log_user!(log, LogType::Error, "Failed to convert IPX basic solution to Highs basic solution\n");
        return Status::Error as i32;
    }
    *h.value_valid = true;
    *h.dual_valid = true;
    *h.basis_valid = true;
    *h.basis_useful = true;
    info.basis_validity = BASIS_VALIDITY_VALID;
    if imprecise {
        *h.model_status = MS_UNKNOWN;
        Status::Warning as i32
    } else {
        *h.model_status = MS_OPTIMAL;
        Status::Ok as i32
    }
}
