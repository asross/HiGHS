//! The QP glue: Highs::callSolveQp up to the objective and KKT check
//! (the Hessian check, the Instance with the square Hessian, the Settings
//! from the options and their logging), solveqp's quass2highs and phase 1
//! (computeStartingPointHighs: the hot start check, or a feasibility LP
//! solved by a separate Highs, then the starting active set).
//!
//! C++ (highs/qpsolver/QpRust.cpp) passes the LP and Hessian views, the
//! options, the Highs solution, basis, model status and info in place, and
//! one `op` per step on a C++ object: the profiling clock, the timer,
//! sizing the solution and basis, and the phase 1 LP solve.

use super::vector::{MatrixBase, QpVector};
use super::{solve, BasisStatus, Callbacks, Instance, ModelStatus, Phase1Start, Settings};
use crate::lp_data::basis::COut;
use crate::lp_data::ffi::{CLp, RsMut};
use crate::lp_data::solution::Info;
use crate::lp_data::{Log, LogType, Status, INF};
use crate::util::printf::sprintf;
use crate::{log_dev, log_user};
use std::ffi::c_void;

// The ops of QpRust.cpp
const OP_PROFILING_START: i32 = 0;
const OP_PROFILING_STOP: i32 = 1;
const OP_TIMER_READ: i32 = 2;
/// Size the solution and basis to the LP, out: COut
const OP_RESIZE: i32 = 3;
/// The phase 1 LP, out: CPhase1
const OP_PHASE1: i32 = 4;

// HighsModelStatus
const MS_NOTSET: i32 = 0;
const MS_MODEL_ERROR: i32 = 2;
const MS_SOLVE_ERROR: i32 = 4;
const MS_OPTIMAL: i32 = 7;
const MS_INFEASIBLE: i32 = 8;
const MS_UNBOUNDED: i32 = 10;
const MS_TIME_LIMIT: i32 = 13;
const MS_ITERATION_LIMIT: i32 = 14;
const MS_INTERRUPT: i32 = 17;

// HighsBasisStatus
const LOWER: u8 = 0;
const BASIC: u8 = 1;
const UPPER: u8 = 2;
const ZERO: u8 = 3;
const NONBASIC: u8 = 4;

/// The phase 1 LP solve: its time limit in, the outcome out (the basis
/// and solution copied into the Rust buffers, at most their lengths)
#[repr(C)]
pub struct CPhase1 {
    pub time_limit: f64,
    pub run_error: bool,
    pub model_status: i32,
    pub simplex_iteration_count: i32,
    pub col_status: RsMut<u8>,
    pub row_status: RsMut<u8>,
    pub col_value: RsMut<f64>,
    pub row_value: RsMut<f64>,
    /// The sizes of the basis' vectors
    pub num_col_status: usize,
    pub num_row_status: usize,
}

/// What callSolveQp works on (QpRust.cpp: RsQpHost)
#[repr(C)]
pub struct CQpHost {
    pub ctx: *mut c_void,
    pub op: unsafe extern "C" fn(*mut c_void, i32, *mut c_void) -> f64,
    pub log: Log,
    pub lp: CLp,
    pub hessian_dim: i32,
    pub hessian_start: RsMut<i32>,
    pub hessian_index: RsMut<i32>,
    pub hessian_value: RsMut<f64>,
    pub qp_iteration_limit: i32,
    pub qp_nullspace_limit: i32,
    pub simplex_primal_edge_weight_strategy: i32,
    pub qp_allow_hot_start: bool,
    pub timeless_log: bool,
    pub qp_regularization_value: f64,
    pub time_limit: f64,
    pub dual_feasibility_tolerance: f64,
    /// The Highs solution and basis (read by the hot start check)
    pub col_value: RsMut<f64>,
    pub row_value: RsMut<f64>,
    pub col_status: RsMut<u8>,
    pub row_status: RsMut<u8>,
    pub value_valid: *mut bool,
    pub dual_valid: *mut bool,
    pub basis_valid: *mut bool,
    pub basis_alien: *mut bool,
    pub basis_useful: *mut bool,
    pub model_status: *mut i32,
    pub info: *mut Info,
}

impl CQpHost {
    fn op(&self, code: i32, out: *mut c_void) -> f64 {
        // SAFETY: the host's op with its context
        unsafe { (self.op)(self.ctx, code, out) }
    }
}

/// triangularToSquareHessian: the columns in row order
fn triangular_to_square(dim: usize, t_start: &[i32], t_index: &[i32], t_value: &[f64]) -> (Vec<usize>, Vec<usize>, Vec<f64>) {
    if dim == 0 {
        return (vec![0], Vec::new(), Vec::new());
    }
    let nnz = t_start[dim] as usize;
    let square_nnz = nnz + (nnz - dim);
    let mut start = vec![0usize; dim + 1];
    let mut index = vec![0usize; square_nnz];
    let mut value = vec![0.0; square_nnz];
    let mut length = vec![0usize; dim];
    for col in 0..dim {
        length[col] += 1;
        for el in t_start[col] as usize + 1..t_start[col + 1] as usize {
            length[t_index[el] as usize] += 1;
            length[col] += 1;
        }
    }
    for col in 0..dim {
        start[col + 1] = start[col] + length[col];
    }
    for col in 0..dim {
        let el = t_start[col] as usize;
        let to = start[col];
        index[to] = t_index[el] as usize;
        value[to] = t_value[el];
        start[col] += 1;
        for el in el + 1..t_start[col + 1] as usize {
            let row = t_index[el] as usize;
            let to = start[row];
            index[to] = col;
            value[to] = t_value[el];
            start[row] += 1;
            let to = start[col];
            index[to] = row;
            value[to] = t_value[el];
            start[col] += 1;
        }
    }
    start[0] = 0;
    for col in 0..dim {
        start[col + 1] = start[col] + length[col];
    }
    (start, index, value)
}

/// qpModelStatusToString
fn model_status_string(s: ModelStatus) -> &'static str {
    match s {
        ModelStatus::NotSet => "Not set",
        ModelStatus::Undetermined => "Undetermined",
        ModelStatus::Optimal => "Optimal",
        ModelStatus::Unbounded => "Unbounded",
        ModelStatus::Infeasible => "Infeasible",
        ModelStatus::IterationLimit => "Iteration limit",
        ModelStatus::TimeLimit => "Time ;limit",
        ModelStatus::LargeNullspace => "Large nullspace",
        ModelStatus::NonConvex => "Non-convex",
        ModelStatus::Error => "Error",
        ModelStatus::Interrupt => "Unidentified QP model status",
    }
}

/// The bounds phase 1 reads (the Instance's, which the solver consumes)
struct Bounds<'a> {
    var_lo: &'a [f64],
    var_up: &'a [f64],
    con_lo: &'a [f64],
    con_up: &'a [f64],
}

struct Glue<'a> {
    h: &'a CQpHost,
    b: Bounds<'a>,
    settings: &'a Settings,
    phase1_iterations: i32,
}

impl Callbacks for Glue<'_> {
    fn time(&mut self) -> f64 {
        self.h.op(OP_TIMER_READ, std::ptr::null_mut())
    }

    fn iteration_log(&mut self, iteration: i32, objval: f64, nullspace_dim: i32, time: f64) {
        let t = if self.h.timeless_log { String::new() } else { sprintf(" %9.2fs", &[time.into()]) };
        log_user!(&self.h.log, LogType::Info, "%11d  %15.8g           %6d%s\n", iteration, objval, nullspace_dim, &t);
    }

    fn nullspace_limit_log(&mut self, nullspace_limit: i32) {
        log_user!(&self.h.log, LogType::Error, "QP solver has exceeded nullspace limit of %d\n", nullspace_limit);
    }

    fn degeneracy_fail_log(&mut self, maxabsd: i32, log_d: f64) {
        log_user!(
            &self.h.log,
            LogType::Error,
            "QP solver has failed due to degeneracy: cannot find non-active constraint to leave basis. max: log(d[%d]) = %lf\n",
            maxabsd,
            log_d
        );
    }

    fn phase1(&mut self) -> Phase1Start {
        let (start, iterations) = compute_starting_point(self.h, &self.b, self.settings);
        self.phase1_iterations = iterations;
        start
    }
}

/// assessQpPrimalFeasibility, its infeasibility counts only (the rest is
/// only reported in debugging)
fn infeasibility_count(lower: &[f64], upper: &[f64], value: &[f64], tol: f64) -> usize {
    let mut num = 0;
    for i in 0..lower.len() {
        let (l, u, x) = (lower[i], upper[i], value[i]);
        let inf = if x < l - tol {
            l - x
        } else if x > u + tol {
            x - u
        } else {
            0.0
        };
        if inf > 0.0 && inf > tol {
            num += 1;
        }
    }
    num
}

/// computeStartingPointHighs: the start and the phase 1 iterations
fn compute_starting_point(h: &CQpHost, inst: &Bounds, settings: &Settings) -> (Phase1Start, i32) {
    let (n, m) = (inst.var_lo.len(), inst.con_lo.len());
    let empty = |status| Phase1Start {
        status,
        active: Vec::new(),
        status_active: Vec::new(),
        inactive: Vec::new(),
        primal: vec![0.0; n],
        rowact: vec![0.0; m],
    };
    // SAFETY: the Highs solution and basis, read only until resized
    let (h_col_value, h_row_value, h_col_status, h_row_status) =
        unsafe { (h.col_value.get(), h.row_value.get(), h.col_status.get(), h.row_status.get()) };
    let mut have_starting_point = false;
    // SAFETY: the flags of the Highs solution and basis
    if settings_allow_hot_start(h) && unsafe { *h.value_valid } {
        let tol = settings.lambda_zero_threshold;
        let num_var_inf = infeasibility_count(&inst.var_lo, &inst.var_up, h_col_value, tol);
        let num_con_inf = infeasibility_count(&inst.con_lo, &inst.con_up, h_row_value, tol);
        have_starting_point = num_var_inf == 0 && num_con_inf == 0 && unsafe { *h.basis_valid };
    }
    let mut status = ModelStatus::NotSet;
    let mut iterations = 0;
    let (mut lp_col_status, mut lp_row_status, mut lp_col_value, mut lp_row_value);
    let (col_status, row_status, col_value, row_value): (&[u8], &[u8], &[f64], &[f64]) = if have_starting_point {
        (h_col_status, h_row_status, h_col_value, h_row_value)
    } else {
        let time = h.op(OP_TIMER_READ, std::ptr::null_mut());
        let left = settings.time_limit - time;
        lp_col_status = vec![0u8; n];
        lp_row_status = vec![0u8; m];
        lp_col_value = vec![0.0; n];
        lp_row_value = vec![0.0; m];
        let mut p = CPhase1 {
            time_limit: if left < 0.001 { 0.001 } else { left },
            run_error: false,
            model_status: 0,
            simplex_iteration_count: 0,
            col_status: RsMut { ptr: lp_col_status.as_mut_ptr(), len: n },
            row_status: RsMut { ptr: lp_row_status.as_mut_ptr(), len: m },
            col_value: RsMut { ptr: lp_col_value.as_mut_ptr(), len: n },
            row_value: RsMut { ptr: lp_row_value.as_mut_ptr(), len: m },
            num_col_status: 0,
            num_row_status: 0,
        };
        h.op(OP_PHASE1, &mut p as *mut CPhase1 as *mut c_void);
        if p.run_error {
            return (empty(ModelStatus::Error), iterations);
        }
        status = match p.model_status {
            MS_OPTIMAL => ModelStatus::NotSet,
            MS_INFEASIBLE => ModelStatus::Infeasible,
            MS_TIME_LIMIT => ModelStatus::TimeLimit,
            MS_INTERRUPT => ModelStatus::Interrupt,
            _ => ModelStatus::Error,
        };
        iterations = p.simplex_iteration_count;
        if status != ModelStatus::NotSet {
            return (empty(status), iterations);
        }
        lp_col_status.truncate(p.num_col_status);
        lp_row_status.truncate(p.num_row_status);
        (&lp_col_status, &lp_row_status, &lp_col_value, &lp_row_value)
    };
    let tol = if have_starting_point { 0.0 } else { 1e-4 };
    let mut x0 = vec![0.0; n];
    let mut ra = vec![0.0; m];
    for i in 0..n {
        if col_value[i].abs() > tol {
            x0[i] = col_value[i];
        }
    }
    for i in 0..m {
        if row_value[i].abs() > tol {
            ra[i] = row_value[i];
        }
    }
    let is_free = |i: usize| inst.var_lo[i] == -INF && inst.var_up[i] == INF;
    let (mut active, mut status_active, mut inactive) = (Vec::new(), Vec::new(), Vec::new());
    for (i, &s) in row_status.iter().enumerate() {
        match s {
            LOWER => {
                active.push(i);
                status_active.push(BasisStatus::ActiveAtLower);
            }
            UPPER => {
                active.push(i);
                status_active.push(BasisStatus::ActiveAtUpper);
            }
            ZERO | NONBASIC => inactive.push(i),
            _ => {}
        }
    }
    for (i, &s) in col_status.iter().enumerate() {
        match s {
            LOWER | UPPER if !is_free(i) => {
                active.push(m + i);
                status_active.push(if s == LOWER { BasisStatus::ActiveAtLower } else { BasisStatus::ActiveAtUpper });
            }
            LOWER | UPPER | ZERO | NONBASIC => inactive.push(m + i),
            _ => {}
        }
    }
    if active.len() + inactive.len() != n {
        return (empty(ModelStatus::Error), iterations);
    }
    (Phase1Start { status, active, status_active, inactive, primal: x0, rowact: ra }, iterations)
}

fn settings_allow_hot_start(h: &CQpHost) -> bool {
    h.qp_allow_hot_start
}

/// callSolveQp until the QP solver's result is in the Highs solution and
/// basis: returns the HighsStatus
///
/// # Safety
/// The host's views and pointers valid, the op with its context
pub unsafe fn call_solve_qp(h: &CQpHost) -> Status {
    let lp = &h.lp;
    let log = &h.log;
    let (n, m) = (lp.num_col as usize, lp.num_row as usize);
    if h.hessian_dim > lp.num_col {
        log_dev!(
            log,
            LogType::Error,
            "Hessian dimension = %d is incompatible with matrix dimension = %d\n",
            h.hessian_dim,
            lp.num_col
        );
        *h.model_status = MS_MODEL_ERROR;
        *h.value_valid = false;
        *h.dual_valid = false;
        return Status::Error;
    }
    h.op(OP_PROFILING_START, std::ptr::null_mut());
    let (q_start, q_index, q_value) =
        triangular_to_square(h.hessian_dim as usize, h.hessian_start.get(), h.hessian_index.get(), h.hessian_value.get());
    let a_start = lp.a.start.get();
    let nnz = a_start[n] as usize;
    let mut c = lp.col_cost.get()[..n].to_vec();
    let mut q_value = q_value;
    if lp.sense == -1 {
        for v in c.iter_mut().chain(q_value.iter_mut()) {
            *v *= -1.0;
        }
    }
    let inst = Instance {
        num_var: n,
        num_con: m,
        offset: lp.offset,
        c: QpVector::from_dense(&c),
        q: MatrixBase { num_row: n, num_col: n, start: q_start, index: q_index, value: q_value },
        a: MatrixBase {
            num_row: m,
            num_col: n,
            start: a_start[..n + 1].iter().map(|&s| s as usize).collect(),
            index: lp.a.index.get()[..nnz].iter().map(|&i| i as usize).collect(),
            value: lp.a.value.get()[..nnz].to_vec(),
        },
        con_lo: lp.row_lower.get()[..m].to_vec(),
        con_up: lp.row_upper.get()[..m].to_vec(),
        var_lo: lp.col_lower.get()[..n].to_vec(),
        var_up: lp.col_upper.get()[..n].to_vec(),
    };
    let mut reinvertfrequency = 1000;
    let qp_update_limit = 1000;
    if qp_update_limit != reinvertfrequency {
        log_user!(log, LogType::Info, "Changing QP reinversion frequency from %d to %d\n", reinvertfrequency, qp_update_limit);
        reinvertfrequency = qp_update_limit;
    }
    let settings = Settings {
        ratiotest: 0,
        pricing: match h.simplex_primal_edge_weight_strategy {
            0 => 1,
            2 => 0,
            _ => 2,
        },
        reportingfequency: if h.qp_iteration_limit <= 10 {
            1
        } else if h.qp_iteration_limit <= 100 {
            10
        } else {
            100
        },
        nullspace_limit: h.qp_nullspace_limit,
        reinvertfrequency,
        gradientrecomputefrequency: 100,
        iteration_limit: h.qp_iteration_limit,
        ratiotest_t: 1e-9,
        ratiotest_d: 1e-8,
        pnorm_zero_threshold: 1e-11,
        d_zero_threshold: 1e-12,
        lambda_zero_threshold: h.dual_feasibility_tolerance,
        pqp_zero_threshold: 1e-7,
        hessian_regularization_value: h.qp_regularization_value,
        time_limit: h.time_limit,
    };
    log_user!(log, LogType::Info, "  Iteration        Objective     NullspaceDim\n");
    // solveqp
    let b = Bounds {
        var_lo: &lp.col_lower.get()[..n],
        var_up: &lp.col_upper.get()[..n],
        con_lo: &lp.row_lower.get()[..m],
        con_up: &lp.row_upper.get()[..m],
    };
    let mut glue = Glue { h, b, settings: &settings, phase1_iterations: 0 };
    let out = solve(inst, &settings, &mut glue);
    let phase1_iterations = glue.phase1_iterations;
    // quass2highs
    if matches!(
        out.status,
        ModelStatus::Undetermined | ModelStatus::LargeNullspace | ModelStatus::NonConvex | ModelStatus::Error | ModelStatus::NotSet
    ) {
        log_user!(log, LogType::Info, "QP solver model status: %s\n", model_status_string(out.status));
    }
    let (ms, status) = match out.status {
        ModelStatus::Optimal => (MS_OPTIMAL, Status::Ok),
        ModelStatus::Unbounded => (MS_UNBOUNDED, Status::Ok),
        ModelStatus::Infeasible => (MS_INFEASIBLE, Status::Ok),
        ModelStatus::IterationLimit => (MS_ITERATION_LIMIT, Status::Warning),
        ModelStatus::TimeLimit => (MS_TIME_LIMIT, Status::Warning),
        ModelStatus::Interrupt => (MS_INTERRUPT, Status::Warning),
        ModelStatus::Undetermined | ModelStatus::LargeNullspace | ModelStatus::Error => (MS_SOLVE_ERROR, Status::Error),
        ModelStatus::NotSet | ModelStatus::NonConvex => (MS_NOTSET, Status::Error),
    };
    *h.model_status = ms;
    if status != Status::Error {
        let mut o = std::mem::zeroed::<COut>();
        h.op(OP_RESIZE, &mut o as *mut COut as *mut c_void);
        let s = o.view();
        let sense = lp.sense as f64;
        for i in 0..n {
            s.col_value[i] = out.primal[i];
            s.col_dual[i] = sense * out.dualvar[i];
        }
        for i in 0..m {
            s.row_value[i] = out.rowact[i];
            s.row_dual[i] = sense * out.dualcon[i];
        }
        *h.value_valid = true;
        *h.dual_valid = true;
        let basis = |b: BasisStatus| match b {
            BasisStatus::ActiveAtLower => LOWER,
            BasisStatus::ActiveAtUpper => UPPER,
            BasisStatus::InactiveInBasis => NONBASIC,
            BasisStatus::Inactive => BASIC,
        };
        for i in 0..n {
            s.col_status[i] = basis(out.status_var[i]);
        }
        for i in 0..m {
            s.row_status[i] = basis(out.status_con[i]);
        }
        *h.basis_valid = true;
        *h.basis_alien = false;
        *h.basis_useful = true;
    }
    h.op(OP_PROFILING_STOP, std::ptr::null_mut());
    if status == Status::Error {
        return status;
    }
    let info = &mut *h.info;
    info.simplex_iteration_count += phase1_iterations;
    info.qp_iteration_count += out.num_iterations;
    status
}

/// # Safety
/// As call_solve_qp
#[no_mangle]
pub unsafe extern "C" fn highs_rs_call_solve_qp(h: *const CQpHost) -> i32 {
    call_solve_qp(&*h) as i32
}
