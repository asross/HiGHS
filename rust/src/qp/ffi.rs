//! `extern "C"` entry of the QP solver, called by solveqp in
//! highs/qpsolver/a_quass.cpp under HIGHS_RUST

use super::quass::{Callbacks, Phase1Start};
use super::vector::{MatrixBase, QpVector};
use super::{solve, BasisStatus, Instance, ModelStatus, Settings};
use crate::ffi::{sl, sl_mut};
use std::ffi::c_void;

/// The QP, column-wise (Q square)
#[repr(C)]
pub struct CQpModel {
    num_var: i32,
    num_con: i32,
    offset: f64,
    c: *const f64,
    a_start: *const i32,
    a_index: *const i32,
    a_value: *const f64,
    q_start: *const i32,
    q_index: *const i32,
    q_value: *const f64,
    con_lo: *const f64,
    con_up: *const f64,
    var_lo: *const f64,
    var_up: *const f64,
}

/// The C++ side: timer, logging and phase 1
#[repr(C)]
pub struct CQpCallbacks {
    ctx: *mut c_void,
    time: unsafe extern "C" fn(*mut c_void) -> f64,
    iteration_log: unsafe extern "C" fn(*mut c_void, i32, f64, i32, f64),
    nullspace_limit_log: unsafe extern "C" fn(*mut c_void, i32),
    degeneracy_fail_log: unsafe extern "C" fn(*mut c_void, i32, f64),
    /// Fills active (with status) and inactive (capacity num_var +
    /// num_con each, counts returned) and the dense x0 (num_var) and ra
    /// (num_con); returns the QpModelStatus
    #[allow(clippy::type_complexity)]
    phase1: unsafe extern "C" fn(
        *mut c_void,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut i32,
        *mut f64,
        *mut f64,
    ) -> i32,
}

/// The QpSolution arrays to fill (num_var or num_con long)
#[repr(C)]
pub struct CQpSolution {
    num_iterations: i32,
    primal: *mut f64,
    rowact: *mut f64,
    dualvar: *mut f64,
    dualcon: *mut f64,
    status_var: *mut i32,
    status_con: *mut i32,
}

struct CCallbacks<'a> {
    c: &'a CQpCallbacks,
    n: usize,
    m: usize,
}

impl Callbacks for CCallbacks<'_> {
    fn time(&mut self) -> f64 {
        // SAFETY: the C++ callback with its own context
        unsafe { (self.c.time)(self.c.ctx) }
    }

    fn iteration_log(&mut self, iteration: i32, objval: f64, nullspace_dim: i32, time: f64) {
        // SAFETY: as above
        unsafe { (self.c.iteration_log)(self.c.ctx, iteration, objval, nullspace_dim, time) }
    }

    fn nullspace_limit_log(&mut self, nullspace_limit: i32) {
        // SAFETY: as above
        unsafe { (self.c.nullspace_limit_log)(self.c.ctx, nullspace_limit) }
    }

    fn degeneracy_fail_log(&mut self, maxabsd: i32, log_d: f64) {
        // SAFETY: as above
        unsafe { (self.c.degeneracy_fail_log)(self.c.ctx, maxabsd, log_d) }
    }

    fn phase1(&mut self) -> Phase1Start {
        let cap = self.n + self.m;
        let (mut active, mut status) = (vec![0i32; cap], vec![0i32; cap]);
        let mut inactive = vec![0i32; cap];
        let (mut num_active, mut num_inactive) = (0i32, 0i32);
        let mut primal = vec![0.0; self.n];
        let mut rowact = vec![0.0; self.m];
        // SAFETY: every buffer has the capacity the callback is told of
        let qp_status = unsafe {
            (self.c.phase1)(
                self.c.ctx,
                active.as_mut_ptr(),
                status.as_mut_ptr(),
                &mut num_active,
                inactive.as_mut_ptr(),
                &mut num_inactive,
                primal.as_mut_ptr(),
                rowact.as_mut_ptr(),
            )
        };
        let (na, ni) = (num_active as usize, num_inactive as usize);
        assert!(na <= cap && ni <= cap);
        Phase1Start {
            status: ModelStatus::from_i32(qp_status),
            active: active[..na].iter().map(|&i| i as usize).collect(),
            status_active: status[..na].iter().map(|&s| BasisStatus::from_i32(s)).collect(),
            inactive: inactive[..ni].iter().map(|&i| i as usize).collect(),
            primal,
            rowact,
        }
    }
}

/// # Safety
/// The pointers must be valid for the lengths given by the model's
/// dimensions (and the matrices' start arrays).
unsafe fn csc(num_row: usize, num_col: usize, start: *const i32, index: *const i32, value: *const f64) -> MatrixBase {
    let start = sl(start, num_col as i32 + 1);
    let nnz = start[num_col];
    MatrixBase {
        num_row,
        num_col,
        start: start.iter().map(|&s| s as usize).collect(),
        index: sl(index, nnz).iter().map(|&i| i as usize).collect(),
        value: sl(value, nnz).to_vec(),
    }
}

/// Solve the QP; returns the QpModelStatus
///
/// # Safety
/// The arrays of `model` and `sol` must have the lengths of the QP's
/// dimensions, and the callbacks must be valid with their context.
#[no_mangle]
pub unsafe extern "C" fn highs_rs_qp_solve(
    model: *const CQpModel,
    settings: *const Settings,
    callbacks: *const CQpCallbacks,
    sol: *mut CQpSolution,
) -> i32 {
    let (md, sol) = (&*model, &mut *sol);
    let (n, m) = (md.num_var.max(0) as usize, md.num_con.max(0) as usize);
    let (ni, mi) = (n as i32, m as i32);
    let inst = Instance {
        num_var: n,
        num_con: m,
        offset: md.offset,
        c: QpVector::from_dense(sl(md.c, ni)),
        q: csc(n, n, md.q_start, md.q_index, md.q_value),
        a: csc(m, n, md.a_start, md.a_index, md.a_value),
        con_lo: sl(md.con_lo, mi).to_vec(),
        con_up: sl(md.con_up, mi).to_vec(),
        var_lo: sl(md.var_lo, ni).to_vec(),
        var_up: sl(md.var_up, ni).to_vec(),
    };
    let mut cb = CCallbacks { c: &*callbacks, n, m };
    let out = solve(inst, &*settings, &mut cb);
    sol.num_iterations = out.num_iterations;
    sl_mut(sol.primal, ni).copy_from_slice(&out.primal);
    sl_mut(sol.rowact, mi).copy_from_slice(&out.rowact);
    sl_mut(sol.dualvar, ni).copy_from_slice(&out.dualvar);
    sl_mut(sol.dualcon, mi).copy_from_slice(&out.dualcon);
    for (d, s) in sl_mut(sol.status_var, ni).iter_mut().zip(&out.status_var) {
        *d = *s as i32;
    }
    for (d, s) in sl_mut(sol.status_con, mi).iter_mut().zip(&out.status_con) {
        *d = *s as i32;
    }
    out.status as i32
}
