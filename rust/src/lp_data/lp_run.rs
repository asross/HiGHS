//! The data of an LP run owned by Rust ([`LpRun`], in the simplex engine
//! [`LpSolver::run`]): the solution, the HiGHS basis, HighsInfo and the
//! model status. The LP part of Highs::calledOptimizeModel (run.rs
//! `optimize_lp`) runs on it: when it is reached, the Highs object's
//! solution, basis, info and model status are copied in (`LpRustBegin`),
//! the run's steps on them are made here ([`lp_op`]) and the solves write
//! them (solveLp's simplex, IPX, PDLP and unconstrained solves on Rust
//! data, [`solve_lp`]), and at its end they are copied back
//! (`LpRustEnd`) before returnFromOptimizeModel. Steps on other C++
//! objects (the options, the clocks, the HEkk shell) are passed through to
//! the Highs object's `op`. LP presolve and postsolve work on Rust data
//! too (lp_presolve.rs).

use super::ffi::{rs_vec_test::rs_vec, CLp, RsMut, RsVec};
use super::ipx_glue::{solve_lp_ipx, CIpxHost};
use super::basis::COut;
use super::interface::BasisG;
use super::lp_handle::LpHandle;
use super::lp_presolve::PresolveData;
use super::run::{CHighs, Op, MS_ITERATION_LIMIT, MS_NOTSET, MS_TIME_LIMIT, MS_UNKNOWN};
use super::solution::{lp_kkt_check, CKktOptions, CSolution, Info, LpRef, SolRef};
use super::solve::{reset_model_status_and_info, solve_unconstrained_lp, CSolve, Unconstrained};
use crate::simplex::app::{solve_lp_simplex, CSimplexApp};
use crate::simplex::lp_solver::LpSolver;
use std::cell::Cell;
use std::ffi::c_void;

const BASIC: u8 = 1;
const NONBASIC: u8 = 4;
const BASIS_VALIDITY_INVALID: i32 = 0;

/// HighsSolution
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Solution {
    pub value_valid: bool,
    pub dual_valid: bool,
    pub col_value: Vec<f64>,
    pub col_dual: Vec<f64>,
    pub row_value: Vec<f64>,
    pub row_dual: Vec<f64>,
}

impl Solution {
    /// The view the C++-shaped Rust functions take
    pub fn view(&mut self) -> CSolution {
        CSolution {
            value_valid: self.value_valid,
            dual_valid: self.dual_valid,
            col_value: rm(&mut self.col_value),
            col_dual: rm(&mut self.col_dual),
            row_value: rm(&mut self.row_value),
            row_dual: rm(&mut self.row_dual),
        }
    }
}

/// HighsBasis
#[derive(Clone, Debug, PartialEq)]
pub struct Basis {
    pub b: BasisG<Vec<u8>>,
    pub origin: String,
}

impl Default for Basis {
    fn default() -> Self {
        Basis {
            b: BasisG {
                valid: false,
                alien: true,
                useful: false,
                was_alien: true,
                debug_id: -1,
                debug_update_count: -1,
                col_status: Vec::new(),
                row_status: Vec::new(),
            },
            origin: "None".into(),
        }
    }
}

impl Basis {
    /// HighsBasis::invalidate
    pub fn invalidate(&mut self) {
        let b = &mut self.b;
        b.valid = false;
        b.alien = true;
        b.useful = false;
        b.was_alien = true;
        b.debug_id = -1;
        b.debug_update_count = -1;
        self.origin = "None".into();
    }
}

/// The solution, basis, info and model status of an LP run
#[derive(Clone)]
pub struct LpRun {
    pub solution: Solution,
    pub basis: Basis,
    pub info: Info,
    pub model_status: i32,
    /// PresolveComponent's data
    pub presolve: PresolveData,
}

impl Default for LpRun {
    fn default() -> Self {
        // SAFETY: Info is plain data, all set by the first import
        let info = unsafe { std::mem::zeroed::<Info>() };
        LpRun {
            solution: Solution::default(),
            basis: Basis::default(),
            info,
            model_status: MS_NOTSET,
            presolve: PresolveData::default(),
        }
    }
}

fn rm<T>(v: &mut Vec<T>) -> RsMut<T> {
    RsMut { ptr: v.as_mut_ptr(), len: v.len() }
}

// ---- The C++ Highs object's data, copied in and out

/// The Highs object's solution, basis, info and model status in place
/// (HighsRunRust.cpp: RsRunData)
#[repr(C)]
pub struct CRunData {
    pub col_value: RsVec<f64>,
    pub col_dual: RsVec<f64>,
    pub row_value: RsVec<f64>,
    pub row_dual: RsVec<f64>,
    pub value_valid: *mut bool,
    pub dual_valid: *mut bool,
    pub basis: *mut BasisG<RsVec<u8>>,
    pub origin: RsMut<u8>,
    pub origin_ctx: *mut c_void,
    pub set_origin: unsafe extern "C" fn(*mut c_void, *const u8, usize),
    pub info: *mut Info,
    pub model_status: *mut i32,
}

fn copy_in<T: Copy + Default>(dst: &mut Vec<T>, src: &RsVec<T>) {
    dst.clear();
    dst.extend_from_slice(src.as_slice());
}

fn copy_out<T: Copy + Default>(dst: &mut RsVec<T>, src: &[T]) {
    dst.resize(src.len());
    dst.as_mut_slice().copy_from_slice(src);
}

impl LpRun {
    /// Copy in the Highs object's data
    ///
    /// # Safety
    /// The views valid
    pub unsafe fn import(&mut self, d: &CRunData) {
        let s = &mut self.solution;
        copy_in(&mut s.col_value, &d.col_value);
        copy_in(&mut s.col_dual, &d.col_dual);
        copy_in(&mut s.row_value, &d.row_value);
        copy_in(&mut s.row_dual, &d.row_dual);
        s.value_valid = *d.value_valid;
        s.dual_valid = *d.dual_valid;
        let cb = &*d.basis;
        let b = &mut self.basis.b;
        b.valid = cb.valid;
        b.alien = cb.alien;
        b.useful = cb.useful;
        b.was_alien = cb.was_alien;
        b.debug_id = cb.debug_id;
        b.debug_update_count = cb.debug_update_count;
        copy_in(&mut b.col_status, &cb.col_status);
        copy_in(&mut b.row_status, &cb.row_status);
        self.basis.origin = String::from_utf8_lossy(d.origin.get()).into_owned();
        self.info = *d.info;
        self.model_status = *d.model_status;
        // Presolve data of this run only
        self.presolve = PresolveData::default();
    }

    /// Copy out into the Highs object's data
    ///
    /// # Safety
    /// The views valid
    pub unsafe fn export(&self, d: &mut CRunData) {
        let s = &self.solution;
        copy_out(&mut d.col_value, &s.col_value);
        copy_out(&mut d.col_dual, &s.col_dual);
        copy_out(&mut d.row_value, &s.row_value);
        copy_out(&mut d.row_dual, &s.row_dual);
        *d.value_valid = s.value_valid;
        *d.dual_valid = s.dual_valid;
        let cb = &mut *d.basis;
        let b = &self.basis.b;
        cb.valid = b.valid;
        cb.alien = b.alien;
        cb.useful = b.useful;
        cb.was_alien = b.was_alien;
        cb.debug_id = b.debug_id;
        cb.debug_update_count = b.debug_update_count;
        copy_out(&mut cb.col_status, &b.col_status);
        copy_out(&mut cb.row_status, &b.row_status);
        (d.set_origin)(d.origin_ctx, self.basis.origin.as_ptr(), self.basis.origin.len());
        *d.info = self.info;
        *d.model_status = self.model_status;
    }
}

/// # Safety
/// `lps` a solver, `d` valid views
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lps_run_import(lps: *mut LpSolver, d: *const CRunData) {
    (*lps).run.import(&*d);
}

/// # Safety
/// As highs_rs_lps_run_import
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lps_run_export(lps: *mut LpSolver, d: *mut CRunData) {
    (*lps).run.export(&mut *d);
}

// ---- The LP part of the run on the Rust data

/// The Rust LP run: the Highs object's steps and the engine
pub struct LpMode<'a> {
    pub orig: &'a CHighs,
    /// The LP solver whose engine and model the run uses: the Highs
    /// object's (an LpHandle, or the C++ Highs object's engine)
    pub handle: *mut LpHandle,
    pub lps: *mut LpSolver,
    /// The run is a C++ Highs object's: the steps on the engine, the
    /// model and the solvers' options are its handle's
    host: bool,
    /// A step threw or IPX was cancelled: no further step
    pub aborted: Cell<bool>,
}

/// What a step returns when it threw (run.rs ABORT)
const ABORT: i64 = i64::MIN;

/// The steps of a C++ Highs object's LP run made by its handle: the
/// simplex engine and shell, the solvers' options and templates (built
/// from the handle's copy of the options, which the run changes and
/// restores), and the model (the handle's copy of the Highs object's LP,
/// whose scale factors the simplex sets)
fn is_handle_step(op: i32, arg: i64) -> bool {
    const STEPS: [Op; 18] = [
        Op::SetEkkLpName,
        Op::EkkClear,
        Op::EkkInvalidate,
        Op::EkkPivotThreshold,
        Op::SaveOptions,
        Op::RestoreOptions,
        Op::OptionsPrimalSimplex,
        Op::OptionsCleanup,
        Op::KktOptions,
        Op::SolveTemplate,
        Op::UnconstrainedTemplate,
        Op::IpxTemplate,
        Op::SimplexTemplate,
        Op::SimplexShell,
        Op::PresolveOptions,
        Op::LpOptions,
        Op::EkkFactorCompatible,
        Op::RetainedEkkDataOk,
    ];
    STEPS.iter().any(|&s| s as i32 == op) || ((op == Op::LpView as i32 || op == Op::Facts as i32) && arg == 0)
}

impl LpMode<'_> {
    /// The run on `handle`'s engine (`orig` a C++ Highs object, or the
    /// handle's own run)
    pub fn new(orig: &CHighs, handle: *mut LpHandle) -> LpMode<'_> {
        // SAFETY: the run's handle
        let lps: *mut LpSolver = unsafe { &mut *(*handle).lps };
        let host = orig.ctx != handle as *mut c_void;
        LpMode { orig, handle, lps, host, aborted: Cell::new(false) }
    }
    #[allow(clippy::mut_from_ref)]
    pub(super) fn lps<'b>(&self) -> &'b mut LpSolver {
        // SAFETY: the handle's engine; borrowed between steps
        unsafe { &mut *self.lps }
    }
    #[allow(clippy::mut_from_ref)]
    pub(super) fn run<'b>(&self) -> &'b mut LpRun {
        &mut self.lps().run
    }
    /// A step of the Highs object (or its handle's)
    pub fn fwd(&self, op: Op, arg: i64, p: *mut c_void, msg: &[u8]) -> i64 {
        self.fwd_raw(op as i32, arg, p, msg.as_ptr(), msg.len())
    }
    fn fwd_raw(&self, op: i32, arg: i64, p: *mut c_void, msg: *const u8, len: usize) -> i64 {
        let (f, ctx) = if self.host && is_handle_step(op, arg) {
            (super::lp_handle::HANDLE_OP, self.handle as *mut c_void)
        } else {
            (self.orig.op, self.orig.ctx)
        };
        // SAFETY: the Highs object's (or handle's) op with its context
        let r = unsafe { f(ctx, op, arg, p, msg, len) };
        if r == ABORT {
            self.aborted.set(true);
        }
        r
    }

    /// A struct the Highs object's op fills
    pub fn fill<T>(&self, op: Op) -> T {
        self.fill_arg(op, 0)
    }
    pub fn fill_arg<T>(&self, op: Op, arg: i64) -> T {
        let mut x = std::mem::MaybeUninit::<T>::uninit();
        self.fwd(op, arg, x.as_mut_ptr() as *mut c_void, b"");
        // SAFETY: filled by the C++ step
        unsafe { x.assume_init() }
    }

    /// The view of the Highs object on the Rust data: the run's steps go
    /// through [`lp_op`]
    pub fn view(&self) -> CHighs {
        let o = self.orig;
        let r = self.run();
        let mut ro = o.o;
        if self.host {
            // The options the LP part of the run changes and restores are
            // the handle's copy, which the solvers read
            // SAFETY: the run's handle
            let opts = unsafe { &mut (*self.handle).opts };
            ro.objective_bound = &mut opts.objective_bound;
            ro.lp_presolve_requires_basis_postsolve = &mut opts.lp_presolve_requires_basis_postsolve;
        }
        CHighs {
            log: o.log,
            ctx: self as *const LpMode as *mut c_void,
            op: lp_op,
            clock: lp_clock,
            model_status: &mut r.model_status,
            presolve_status: o.presolve_status,
            info: &mut r.info,
            run_data: o.run_data,
            value_valid: &mut r.solution.value_valid,
            dual_valid: &mut r.solution.dual_valid,
            basis_valid: &mut r.basis.b.valid,
            basis_alien: &mut r.basis.b.alien,
            basis_useful: &mut r.basis.b.useful,
            basis_was_alien: &mut r.basis.b.was_alien,
            called_return: o.called_return,
            o: ro,
        }
    }

    /// The view of the model LP (0) or the reduced LP of presolve (1),
    /// valid until the C++ LP changes
    pub fn lp_view(&self, which: i64) -> CLp {
        if which == 1 {
            self.reduced_view()
        } else {
            self.fill_arg(Op::LpView, 0)
        }
    }

    /// refineBasis(model, solution, basis)
    fn refine_basis(&self) {
        let r = self.run();
        let (has, s, b) = (r.solution.value_valid, &r.solution, &mut r.basis.b);
        let m = &self.lp_view(0);
        // SAFETY: the model's bounds
        let (cl, cu, rl, ru) = unsafe { (m.col_lower.get(), m.col_upper.get(), m.row_lower.get(), m.row_upper.get()) };
        let (cv, rv): (&[f64], &[f64]) = if has { (&s.col_value, &s.row_value) } else { (&[], &[]) };
        super::basis::refine_basis(cl, cu, cv, &mut b.col_status);
        super::basis::refine_basis(rl, ru, rv, &mut b.row_status);
    }

    /// callLpKktCheck(lp, message) of the model (0) or reduced LP (1)
    fn kkt_check(&self, which: i64, message: &str) {
        let lp = self.lp_view(which);
        // SAFETY: zeroed then filled by the Highs object's op
        let mut o = unsafe { std::mem::zeroed::<CKktOptions>() };
        self.fwd(Op::KktOptions, 0, &mut o as *mut CKktOptions as *mut c_void, b"");
        let r = self.run();
        let basis_valid = r.basis.b.valid;
        let sv = r.solution.view();
        // SAFETY: the model's and solution's arrays live for the call
        unsafe {
            lp_kkt_check(
                &mut r.model_status,
                &mut r.info,
                &LpRef::new(&lp),
                &SolRef::new(&sv),
                basis_valid,
                &o,
                message,
            )
        };
    }

    /// Highs::callSolveLp(lp, message) of the model (0) or reduced LP
    /// (1): the model status is the solver object's
    fn call_solve_lp(&self, which: i64, message: &str) -> i64 {
        let mut ms = MS_NOTSET;
        let status = self.solve_lp(which, &mut ms, message);
        if self.aborted.get() {
            return ABORT;
        }
        self.run().model_status = ms;
        status as i64
    }

    /// solveLp(solver_object, message) on the Rust data
    fn solve_lp(&self, which: i64, ms: &mut i32, message: &str) -> super::Status {
        let mut c: CSolve = self.fill(Op::SolveTemplate);
        let r = self.run();
        let lp = self.lp_view(which);
        let sc = SolveCtx { m: self, ms: ms as *mut i32, lp, which };
        c.ctx = &sc as *const SolveCtx as *mut c_void;
        c.op = solve_op;
        c.model_status = ms;
        c.info = &mut r.info;
        c.value_valid = &r.solution.value_valid;
        c.basis_valid = &r.basis.b.valid;
        c.num_row = lp.num_row;
        c.num_nz = if lp.num_row != 0 { model_num_nz(&lp) } else { 0 };
        c.aborted = self.aborted.as_ptr();
        c.solve_lp(message)
    }
}

fn model_num_nz(m: &CLp) -> i32 {
    let k = if m.a.format == super::matrix_format::COLWISE { m.a.num_col } else { m.a.num_row };
    // SAFETY: the matrix's starts
    unsafe { m.a.start.get() }.get(k as usize).copied().unwrap_or(0)
}

fn text(msg: *const u8, len: usize) -> String {
    // SAFETY: the C++ message lives for the call
    String::from_utf8_lossy(unsafe { crate::ffi::sl(msg, len as i32) }).into_owned()
}

/// The run's steps in the Rust LP run (CHighs::op)
unsafe extern "C" fn lp_op(ctx: *mut c_void, op: i32, arg: i64, p: *mut c_void, msg: *const u8, len: usize) -> i64 {
    let m = &*(ctx as *const LpMode);
    if m.aborted.get() {
        return 0;
    }
    if let Some(r) = m.presolve_op(op, arg, p) {
        return r;
    }
    match op {
        x if x == Op::RefineBasis as i32 => {
            m.refine_basis();
            0
        }
        x if x == Op::CallSolveLp as i32 => m.call_solve_lp(arg, &text(msg, len)),
        x if x == Op::KktCheck as i32 => {
            m.kkt_check(arg, &text(msg, len));
            0
        }
        x if x == Op::ReducedToEmpty as i32 => {
            let r = m.run();
            r.solution = Solution::default();
            r.basis = Basis::default();
            let b = &mut r.basis;
            b.origin = "Presolve to empty".into();
            b.b.valid = true;
            b.b.alien = false;
            b.b.useful = true;
            b.b.was_alien = false;
            r.solution.value_valid = true;
            r.solution.dual_valid = true;
            0
        }
        // A no-op in this build (debugging is left out)
        x if x == Op::DebugPostsolveSolution as i32 => 0,
        x if x == Op::InvalidateBasis as i32 => {
            let r = m.run();
            r.info.basis_validity = BASIS_VALIDITY_INVALID;
            r.basis.invalidate();
            0
        }
        _ => m.fwd_raw(op, arg, p, msg, len),
    }
}

unsafe extern "C" fn lp_clock(ctx: *mut c_void, which: i32, action: i32) -> f64 {
    let m = &*(ctx as *const LpMode);
    (m.orig.clock)(m.orig.ctx, which, action)
}

// ---- solveLp's solvers on the Rust data

struct SolveCtx<'a, 'b> {
    m: &'a LpMode<'b>,
    /// The solver object's model status
    ms: *mut i32,
    /// The solver object's LP: the model (which 0) or reduced LP (1)
    lp: CLp,
    which: i64,
}

// solve.rs: SolveOp
const SOLVE_DEBUG_ASSESS: i32 = 1;
const SOLVE_UNCONSTRAINED: i32 = 2;
const SOLVE_IPX: i32 = 3;
const SOLVE_PDLP: i32 = 4;
const SOLVE_SIMPLEX: i32 = 5;
const SOLVE_SOLUTION_RIGHT_SIZE: i32 = 6;
const SOLVE_DEBUG_SOLUTION: i32 = 7;

unsafe extern "C" fn solve_op(ctx: *mut c_void, op: i32, msg: *const u8, len: usize) -> i64 {
    let sc = &*(ctx as *const SolveCtx);
    let m = sc.m;
    if m.aborted.get() {
        return 0;
    }
    let r = m.run();
    let model = &sc.lp;
    match op {
        SOLVE_DEBUG_ASSESS => m.fwd(Op::AssessLp, 0, std::ptr::null_mut(), b""),
        SOLVE_UNCONSTRAINED => {
            // solveUnconstrainedLp(options, lp, model_status, info, solution, basis)
            reset_model_status_and_info(&mut *sc.ms, &mut r.info);
            if model.num_row > 0 && model_num_nz(model) > 0 {
                return super::Status::Error as i64;
            }
            let (nc, nr) = (model.num_col as usize, model.num_row as usize);
            let s = &mut r.solution;
            let b = &mut r.basis.b;
            s.col_value.clear();
            s.col_value.resize(nc, 0.0);
            s.col_dual.clear();
            s.col_dual.resize(nc, 0.0);
            b.col_status.clear();
            b.col_status.resize(nc, NONBASIC);
            s.row_value.clear();
            s.row_value.resize(nr, 0.0);
            s.row_dual.clear();
            s.row_dual.resize(nr, 0.0);
            b.row_status.clear();
            b.row_status.resize(nr, BASIC);
            let mut t = UnconTemplate::default();
            m.fwd(Op::UnconstrainedTemplate, 0, &mut t as *mut UnconTemplate as *mut c_void, b"");
            *sc.ms = solve_unconstrained_lp(
                &m.orig.log,
                t.on,
                t.primal_feasibility_tolerance,
                t.dual_feasibility_tolerance,
                model,
                &mut r.info,
                Unconstrained {
                    col_value: &mut s.col_value,
                    col_dual: &mut s.col_dual,
                    row_value: &mut s.row_value,
                    row_dual: &mut s.row_dual,
                    col_status: &mut b.col_status,
                    row_status: &mut b.row_status,
                },
            );
            s.value_valid = true;
            s.dual_valid = true;
            b.valid = true;
            b.useful = true;
            super::Status::Ok as i64
        }
        SOLVE_IPX => solve_ipx(sc),
        SOLVE_PDLP => solve_pdlp(sc),
        SOLVE_SIMPLEX => solve_simplex(sc),
        SOLVE_SOLUTION_RIGHT_SIZE => {
            let s = &r.solution;
            let (nc, nr) = (model.num_col as usize, model.num_row as usize);
            (s.col_value.len() == nc && s.row_value.len() == nr && s.col_dual.len() == nc && s.row_dual.len() == nr)
                as i64
        }
        SOLVE_DEBUG_SOLUTION => 0,
        _ => {
            let _ = (msg, len);
            0
        }
    }
}

/// The option values of solveUnconstrainedLp (HighsRunRust.cpp:
/// RsUnconTemplate)
#[repr(C)]
#[derive(Default)]
pub struct UnconTemplate {
    pub on: bool,
    pub primal_feasibility_tolerance: f64,
    pub dual_feasibility_tolerance: f64,
}

/// The context of IPX's host functions on the Rust data
pub(crate) struct IpxCtx {
    pub(crate) timer_read: unsafe extern "C" fn(*mut c_void) -> f64,
    pub(crate) timer_ctx: *mut c_void,
    pub(crate) lps: *mut LpSolver,
    pub(crate) num_col: usize,
    pub(crate) num_row: usize,
}

pub(crate) unsafe extern "C" fn ipx_timer(ctx: *mut c_void) -> f64 {
    let c = &*(ctx as *const IpxCtx);
    (c.timer_read)(c.timer_ctx)
}

pub(crate) unsafe extern "C" fn ipx_resize(ctx: *mut c_void, with_basis: bool, out: *mut COut) {
    let c = &*(ctx as *const IpxCtx);
    let r = &mut (*c.lps).run;
    let s = &mut r.solution;
    s.col_value.resize(c.num_col, 0.0);
    s.row_value.resize(c.num_row, 0.0);
    s.col_dual.resize(c.num_col, 0.0);
    s.row_dual.resize(c.num_row, 0.0);
    let b = &mut r.basis.b;
    if with_basis {
        b.col_status.resize(c.num_col, 0);
        b.row_status.resize(c.num_row, 0);
    }
    let none = RsMut { ptr: std::ptr::null_mut(), len: 0 };
    *out = COut {
        col_value: rm(&mut s.col_value),
        col_dual: rm(&mut s.col_dual),
        row_value: rm(&mut s.row_value),
        row_dual: rm(&mut s.row_dual),
        col_status: if with_basis { rm(&mut b.col_status) } else { none },
        row_status: if with_basis { RsMut { ptr: b.row_status.as_mut_ptr(), len: b.row_status.len() } } else { none },
    };
}

/// solveLpIpx on the Rust data
unsafe fn solve_ipx(sc: &SolveCtx) -> i64 {
    let m = sc.m;
    let mut h: CIpxHost = m.fill(Op::IpxTemplate);
    let r = m.run();
    let c = IpxCtx {
        timer_read: h.timer_read,
        timer_ctx: h.ctx,
        lps: m.lps,
        num_col: sc.lp.num_col as usize,
        num_row: sc.lp.num_row as usize,
    };
    h.ctx = &c as *const IpxCtx as *mut c_void;
    h.timer_read = ipx_timer;
    h.resize = ipx_resize;
    h.lp = sc.lp;
    h.info = &mut r.info;
    h.model_status = sc.ms;
    h.value_valid = &mut r.solution.value_valid;
    h.dual_valid = &mut r.solution.dual_valid;
    h.basis_valid = &mut r.basis.b.valid;
    h.basis_useful = &mut r.basis.b.useful;
    let status = solve_lp_ipx(&h);
    if status == 2 {
        // A cancelled task: C++ rethrows HighsTask::Interrupt
        m.fwd(Op::SetInterrupt, 0, std::ptr::null_mut(), b"");
        m.aborted.set(true);
        return super::Status::Error as i64;
    }
    status as i64
}

/// The PDLP parameters and print function (HighsRunRust.cpp:
/// RsPdlpTemplate)
#[repr(C)]
pub struct PdlpTemplate {
    pub params: crate::pdlp::Params,
    pub print: Option<extern "C" fn(*const std::ffi::c_char)>,
}

// cupdlp termination codes
const PDLP_OPTIMAL: i32 = 0;
const PDLP_INFEASIBLE: i32 = 1;
const PDLP_UNBOUNDED: i32 = 2;
const PDLP_INFEASIBLE_OR_UNBOUNDED: i32 = 3;
const PDLP_TIMELIMIT_OR_ITERLIMIT: i32 = 4;

/// solveLpCupdlp on the Rust data (with the profiling clock of the C++
/// solveLp step)
unsafe fn solve_pdlp(sc: &SolveCtx) -> i64 {
    use super::run::{MS_INFEASIBLE, MS_OPTIMAL, MS_UNBOUNDED, MS_UNBOUNDED_OR_INFEASIBLE};
    let m = sc.m;
    m.fwd(Op::PdlpProfiling, 1, std::ptr::null_mut(), b"");
    let r = m.run();
    reset_model_status_and_info(&mut *sc.ms, &mut r.info);
    let mut t = std::mem::zeroed::<PdlpTemplate>();
    m.fwd(Op::PdlpTemplate, 0, &mut t as *mut PdlpTemplate as *mut c_void, b"");
    let lp = &sc.lp;
    let (nc, nr) = (lp.num_col as usize, lp.num_row as usize);
    let rs_lp = crate::pdlp::ffi::PdlpRsLp {
        num_col: lp.num_col,
        num_row: lp.num_row,
        a_start: lp.a.start.ptr,
        a_index: lp.a.index.ptr,
        a_value: lp.a.value.ptr,
        col_cost: lp.col_cost.ptr,
        col_lower: lp.col_lower.ptr,
        col_upper: lp.col_upper.ptr,
        row_lower: lp.row_lower.ptr,
        row_upper: lp.row_upper.ptr,
        offset: lp.offset,
        sense: if lp.sense == -1 { -1.0 } else { 1.0 },
    };
    let s = &mut r.solution;
    s.col_value.resize(nc, 0.0);
    s.row_value.resize(nr, 0.0);
    s.col_dual.resize(nc, 0.0);
    s.row_dual.resize(nr, 0.0);
    let mut value_valid = s.value_valid as i32;
    let mut dual_valid = s.dual_valid as i32;
    let mut num_iter = 0;
    let code = crate::pdlp::ffi::pdlp_rs_solve(
        &rs_lp,
        &t.params,
        t.print,
        s.col_value.as_mut_ptr(),
        s.col_dual.as_mut_ptr(),
        s.row_value.as_mut_ptr(),
        s.row_dual.as_mut_ptr(),
        &mut value_valid,
        &mut dual_valid,
        &mut num_iter,
    );
    r.info.pdlp_iteration_count = num_iter;
    s.value_valid = value_valid != 0;
    s.dual_valid = dual_valid != 0;
    r.basis.b.valid = false;
    *sc.ms = match code {
        PDLP_OPTIMAL => MS_OPTIMAL,
        PDLP_INFEASIBLE => MS_INFEASIBLE,
        PDLP_UNBOUNDED => MS_UNBOUNDED,
        PDLP_INFEASIBLE_OR_UNBOUNDED => MS_UNBOUNDED_OR_INFEASIBLE,
        PDLP_TIMELIMIT_OR_ITERLIMIT => {
            if num_iter >= t.params.iter_lim - 1 {
                MS_ITERATION_LIMIT
            } else {
                MS_TIME_LIMIT
            }
        }
        _ => MS_UNKNOWN,
    };
    m.fwd(Op::PdlpProfiling, 0, std::ptr::null_mut(), b"");
    super::Status::Ok as i64
}

/// The simplex host functions on the Rust data: the basis' origin and
/// setBasis are here, the HEkk shell's steps are the Highs object's
struct SimplexCtx<'a, 'b> {
    m: &'a LpMode<'b>,
    which: i64,
}

// app.rs ops
const APP_SET_BASIS: i32 = 5;
const APP_LP_BACK: i32 = 10;
const APP_ENV: i32 = 11;
const APP_BASIS_ORIGIN: i32 = 12;

unsafe extern "C" fn simplex_op(ctx: *mut c_void, code: i32, arg: i64, p: *mut c_void) -> i64 {
    let c = &*(ctx as *const SimplexCtx);
    let m = c.m;
    match code {
        APP_BASIS_ORIGIN => {
            let s = crate::ffi::sl(p as *const u8, arg as i32);
            m.run().basis.origin = String::from_utf8_lossy(s).into_owned();
            0
        }
        APP_SET_BASIS => {
            // HEkk::setBasis(basis)
            let mut env = std::mem::MaybeUninit::<crate::simplex::lp_solver::LpsEnv>::uninit();
            simplex_op(ctx, APP_ENV, 0, env.as_mut_ptr() as *mut c_void);
            let env = env.assume_init();
            let lps = m.lps();
            let env = lps.env_of(&env);
            let b = &m.run().basis;
            let (nc, nr) = (env.lp.num_col as usize, env.lp.num_row as usize);
            lps.set_basis(
                &env,
                &b.b.col_status[..nc],
                &b.b.row_status[..nr],
                b.b.debug_id,
                b.b.debug_update_count,
                &b.origin,
            );
            0
        }
        // The reduced LP takes the engine LP's scale (and matrix)
        APP_LP_BACK if c.which == 1 => {
            m.reduced_lp_back(arg != 0);
            0
        }
        _ => m.fwd(Op::SimplexShell, ((code as i64) << 32) | (c.which << 16) | (arg & 0xffff), p, b""),
    }
}

/// solveLpSimplex on the Rust data
unsafe fn solve_simplex(sc: &SolveCtx) -> i64 {
    let m = sc.m;
    let mut h: CSimplexApp = m.fill_arg(Op::SimplexTemplate, sc.which);
    let r = m.run();
    let c = SimplexCtx { m, which: sc.which };
    h.ctx = &c as *const SimplexCtx as *mut c_void;
    h.op = simplex_op;
    h.lps = m.lps;
    if sc.which == 1 {
        h.incumbent = m.reduced_view();
        h.model_name = m.reduced_name();
    }
    h.model_status = sc.ms;
    h.info = &mut r.info;
    let s = &mut r.solution;
    h.value_valid = &mut s.value_valid;
    h.dual_valid = &mut s.dual_valid;
    h.col_value = rs_vec(&mut s.col_value);
    h.col_dual = rs_vec(&mut s.col_dual);
    h.row_value = rs_vec(&mut s.row_value);
    h.row_dual = rs_vec(&mut s.row_dual);
    let b = &mut r.basis.b;
    h.basis_valid = &mut b.valid;
    h.basis_alien = &mut b.alien;
    h.basis_useful = &mut b.useful;
    h.basis_was_alien = &mut b.was_alien;
    h.basis_debug_id = &mut b.debug_id;
    h.basis_debug_update_count = &mut b.debug_update_count;
    h.col_status = rs_vec(&mut b.col_status);
    h.row_status = rs_vec(&mut b.row_status);
    solve_lp_simplex(&mut h) as i64
}
