//! LP presolve and postsolve of the LP run on Rust data (lp_run.rs):
//! PresolveComponent's data is Rust's ([`PresolveData`]: the reduced LP an
//! [`Lp`], the postsolve stack's storage a [`PostsolveStack`] in the
//! layout presolve/postsolve.rs reads, the recovered solution and basis),
//! and the presolve (rust/src/presolve/hpresolve) runs with a host whose
//! model and stack callbacks are here. The option values, the timer and
//! the factorization of the dependent equations are reached through the
//! run's steps. At the end of the run the reduced LP, stack, status and
//! log are copied to the C++ PresolveComponent
//! ([`highs_rs_lps_presolve_export`]), whose API (the index maps,
//! postsolve, the presolved model) reads them.

use super::ffi::{CLp, CLpOptions, RsMut};
use super::lp::{CppLp, Lp};
use super::lp_run::{Basis, LpMode, Solution};
use super::lp_utils::clean_bounds;
use super::run::{Facts, Op};
use super::sparse::Buf;
use super::LogType;
use crate::presolve::hpresolve::ffi::{CModel, CSlice, Host, LiftOpp, MipEnv};
use crate::presolve::hpresolve::{Input, Options, Presolve};
use crate::presolve::postsolve::{self, compress_index_maps, Reduction, Stack, Tolerances};
use crate::simplex::lp_solver::LpSolver;
use std::ffi::{c_char, c_void, CStr};

/// HighsPostsolveStack's data
#[derive(Clone)]
pub struct PostsolveStack {
    pub data: Vec<u8>,
    pub reductions: Vec<Reduction>,
    pub orig_col_index: Vec<i32>,
    pub orig_row_index: Vec<i32>,
    pub linearly_transformable: Vec<u8>,
    pub orig_num_col: i32,
    pub orig_num_row: i32,
}

impl Default for PostsolveStack {
    fn default() -> Self {
        PostsolveStack {
            data: Vec::new(),
            reductions: Vec::new(),
            orig_col_index: Vec::new(),
            orig_row_index: Vec::new(),
            linearly_transformable: Vec::new(),
            orig_num_col: -1,
            orig_num_row: -1,
        }
    }
}

impl PostsolveStack {
    /// HighsPostsolveStack::initializeIndexMaps
    pub fn initialize_index_maps(&mut self, num_row: i32, num_col: i32) {
        self.orig_num_row = num_row;
        self.orig_num_col = num_col;
        self.orig_row_index = (0..num_row).collect();
        self.orig_col_index = (0..num_col).collect();
        self.linearly_transformable.resize(num_col as usize, 1);
    }

    /// HighsPostsolveStack::undo (undoRust): the solution and basis are
    /// expanded to the original space and the reductions undone
    pub fn undo(&self, tol: &Tolerances, solution: &mut Solution, basis: &mut Basis) {
        let (nc, nr) = (self.orig_num_col as usize, self.orig_num_row as usize);
        let dual_valid = solution.dual_valid;
        let valid = basis.b.valid;
        solution.col_value.resize(nc, 0.0);
        solution.row_value.resize(nr, 0.0);
        if dual_valid {
            solution.col_dual.resize(nc, 0.0);
            solution.row_dual.resize(nr, 0.0);
        }
        if valid {
            basis.b.col_status.resize(nc, 0);
            basis.b.row_status.resize(nr, 0);
        }
        let (dc, dr) = if dual_valid { (nc, nr) } else { (0, 0) };
        let (bc, br) = if valid { (nc, nr) } else { (0, 0) };
        let mut sol = postsolve::Solution {
            col_value: &mut solution.col_value[..nc],
            row_value: &mut solution.row_value[..nr],
            col_dual: &mut solution.col_dual[..dc],
            row_dual: &mut solution.row_dual[..dr],
            dual_valid,
        };
        let mut bs = postsolve::Basis {
            col_status: &mut basis.b.col_status[..bc],
            row_status: &mut basis.b.row_status[..br],
            valid,
        };
        let stack = Stack {
            data: &self.data,
            reductions: &self.reductions,
            orig_col_index: &self.orig_col_index,
            orig_row_index: &self.orig_row_index,
        };
        postsolve::undo(&stack, tol, &mut sol, &mut bs, 0, -1);
    }
}

/// PresolveComponent's data
#[derive(Clone, Default)]
pub struct PresolveData {
    /// init was called since the last clear: the data go to C++ at the end
    pub active: bool,
    pub reduced: Lp,
    pub stack: PostsolveStack,
    pub recovered_solution: Solution,
    pub recovered_basis: Basis,
    pub status: i32,
    /// (call, col_removed, row_removed) per rule
    pub log: Vec<[i32; 3]>,
    /// PrepareReducedLp ran (the reduced LP's origin name)
    pub prepared: bool,
}

/// The option values of the LP presolve (HPresolveRust.cpp:
/// rsLpPresolveOptions)
#[repr(C)]
pub(crate) struct PresolveOptions {
    pub(crate) o: Options,
    /// presolve_reduction_limit, -1 for none
    pub(crate) reduction_limit: i32,
}

fn mode<'a>(ctx: *mut c_void) -> &'a LpMode<'a> {
    // SAFETY: the host's context is the run
    unsafe { &*(ctx as *const LpMode) }
}

fn data<'a>(ctx: *mut c_void) -> &'a mut PresolveData {
    mode(ctx).presolve()
}

/// `v = s` for Rust or C++ vectors
fn set<T: Copy + Default, B: Buf<T>>(v: &mut B, s: &[T]) {
    v.resize(s.len());
    v.sl_mut().copy_from_slice(s);
}

/// A C++ array of n entries
unsafe fn sl<'a, T>(p: *const T, n: usize) -> &'a [T] {
    crate::ffi::sl(p, n as i32)
}

extern "C" fn h_log(ctx: *mut c_void, channel: i32, t: i32, msg: *const c_char) {
    let log = &mode(ctx).orig.log;
    // SAFETY: a NUL-terminated message
    let s = unsafe { CStr::from_ptr(msg) }.to_string_lossy();
    let t = match t {
        2 => LogType::Detailed,
        3 => LogType::Verbose,
        4 => LogType::Warning,
        5 => LogType::Error,
        _ => LogType::Info,
    };
    match channel {
        0 => log.user(t, &s),
        1 => log.dev(t, &s),
        _ => crate::io::log::c_stdout_flush(s.as_bytes()),
    }
}

extern "C" fn h_timer_read(ctx: *mut c_void) -> f64 {
    // HighsTimer::read() (the run clock)
    let o = mode(ctx).orig;
    // SAFETY: the Highs object's clock
    unsafe { (o.clock)(o.ctx, 0, 0) }
}

extern "C" fn h_time_string(_: *mut c_void, t: f64, buf: *mut c_char, len: usize) {
    // highsTimeSecondToString
    let s = format!("{}s", t as i32);
    let n = s.len().min(len - 1);
    // SAFETY: the buffer holds len bytes
    unsafe {
        std::ptr::copy_nonoverlapping(s.as_ptr() as *const c_char, buf, n);
        *buf.add(n) = 0;
    }
}

extern "C" fn h_sync_model(ctx: *mut c_void, m: *const CModel, _resize_row_names: bool) {
    // SAFETY: the presolve's arrays of the model's dimensions
    let m = unsafe { &*m };
    let lp = &mut data(ctx).reduced.g;
    let (nc, nr) = (m.num_col as usize, m.num_row as usize);
    lp.num_col = m.num_col;
    lp.num_row = m.num_row;
    // SAFETY: as above
    unsafe {
        set(&mut lp.col_cost, sl(m.col_cost, nc));
        set(&mut lp.col_lower, sl(m.col_lower, nc));
        set(&mut lp.col_upper, sl(m.col_upper, nc));
        set(&mut lp.row_lower, sl(m.row_lower, nr));
        set(&mut lp.row_upper, sl(m.row_upper, nr));
        set(&mut lp.integrality, sl(m.integrality, nc));
    }
    lp.offset = m.offset;
    lp.sense = if m.maximize { -1 } else { 1 };
}

extern "C" fn h_set_matrix(ctx: *mut c_void, start: *const i32, ns: usize, index: *const i32, value: *const f64, nnz: usize) {
    let a = &mut data(ctx).reduced.g.a;
    // SAFETY: the presolve's arrays of these lengths
    unsafe {
        set(&mut a.start, sl(start, ns));
        set(&mut a.index, sl(index, nnz));
        set(&mut a.value, sl(value, nnz));
    }
}

#[allow(clippy::too_many_arguments)]
extern "C" fn h_flush(
    ctx: *mut c_void,
    bytes: *const u8,
    len: usize,
    types: *const u8,
    pos: *const usize,
    n: usize,
    not_transformable: *const i32,
    nnt: usize,
) {
    let s = &mut data(ctx).stack;
    // SAFETY: the presolve's arrays of these lengths
    unsafe {
        s.data.extend_from_slice(sl(bytes, len));
        let (t, p) = (sl(types, n), sl(pos, n));
        s.reductions.extend(t.iter().zip(p).map(|(&kind, &position)| Reduction { kind, position }));
        for &c in sl(not_transformable, nnt) {
            s.linearly_transformable[c as usize] = 0;
        }
    }
}

extern "C" fn h_shrink(ctx: *mut c_void, new_col: *const i32, nc: usize, new_row: *const i32, nr: usize) {
    let d = data(ctx);
    // SAFETY: the presolve's index maps of these lengths
    let (new_col, new_row) = unsafe { (sl(new_col, nc), sl(new_row, nr)) };
    let s = &mut d.stack;
    let (num_row, num_col) = compress_index_maps(&mut s.orig_row_index, &mut s.orig_col_index, new_row, new_col);
    s.orig_row_index.truncate(num_row);
    s.orig_col_index.truncate(num_col);
    // setMatrixDimensions (the names follow the index maps at the export)
    let lp = &mut d.reduced.g;
    lp.a.num_col = lp.num_col;
    lp.a.num_row = lp.num_row;
}

// The MIP callbacks: never called for an LP
extern "C" fn h_profiling(_: *mut c_void, _: bool, _: i32) {
    unreachable!("MIP presolve only")
}
extern "C" fn h_probing_prepare(_: *mut c_void, _: i32, _: *mut bool) -> bool {
    unreachable!("MIP presolve only")
}
extern "C" fn h_mip_env(_: *mut c_void, _: *mut MipEnv) {
    unreachable!("MIP presolve only")
}
extern "C" fn h_probe(_: *mut c_void, _: i32, _: *mut i32) -> bool {
    unreachable!("MIP presolve only")
}
extern "C" fn h_set_lifting(_: *mut c_void, _: bool) {
    unreachable!("MIP presolve only")
}
extern "C" fn h_lifting_opps(_: *mut c_void, _: *mut CSlice<LiftOpp>) {
    unreachable!("MIP presolve only")
}
extern "C" fn h_finalise_begin(_: *mut c_void, _: bool, _: *mut CSlice<i32>, _: *mut CSlice<i32>) {
    unreachable!("MIP presolve only")
}
extern "C" fn h_domain_bounds(_: *mut c_void, _: *mut CSlice<f64>, _: *mut CSlice<f64>) {
    unreachable!("MIP presolve only")
}
extern "C" fn h_mip_finish_presolve(_: *mut c_void, _: i32) {
    unreachable!("MIP presolve only")
}
extern "C" fn h_add_cut(_: *mut c_void, _: *const i32, _: *const f64, _: usize, _: f64, _: bool) {
    unreachable!("MIP presolve only")
}
extern "C" fn h_upper_limit(_: *mut c_void) -> f64 {
    unreachable!("MIP presolve only")
}
extern "C" fn h_set_lower_bound_zero(_: *mut c_void) {
    unreachable!("MIP presolve only")
}

impl LpMode<'_> {
    #[allow(clippy::mut_from_ref)]
    fn presolve<'b>(&self) -> &'b mut PresolveData {
        &mut self.run().presolve
    }

    /// The view of the reduced LP
    pub(super) fn reduced_view(&self) -> CLp {
        self.presolve().reduced.view()
    }

    /// The reduced LP's model name
    pub(super) fn reduced_name(&self) -> RsMut<u8> {
        let n = &mut self.presolve().reduced.model_name;
        RsMut { ptr: n.as_mut_ptr(), len: n.len() }
    }

    /// The reduced LP takes the engine LP's scale and, with `matrix`, its
    /// matrix (HEkk::lpBack)
    pub(super) fn reduced_lp_back(&self, matrix: bool) {
        let e = &self.lps().lp.g;
        let r = &mut self.presolve().reduced.g;
        r.scale.clone_from(&e.scale);
        r.is_scaled = e.is_scaled;
        if matrix {
            r.a.clone_from(&e.a);
        }
    }

    /// The steps of the presolve branch on the Rust data (`None`: not one
    /// of them, or the C++ step follows)
    pub(super) fn presolve_op(&self, op: i32, arg: i64, p: *mut c_void) -> Option<i64> {
        let d = self.presolve();
        let r = match op {
            x if x == Op::PresolveClear as i32 => {
                // and the C++ PresolveComponent
                *d = PresolveData::default();
                return None;
            }
            x if x == Op::PresolveInit as i32 => {
                // PresolveComponent::init: the C++ one too (its reduced LP
                // takes the model's other members, its data are
                // overwritten at the end)
                self.fwd(Op::PresolveInit, 0, std::ptr::null_mut(), b"");
                let model: CLp = self.fill_arg(Op::LpView, 0);
                let mut facts = Facts::default();
                self.fwd(Op::Facts, 0, &mut facts as *mut Facts as *mut c_void, b"");
                d.active = true;
                d.stack.initialize_index_maps(model.num_row, model.num_col);
                // SAFETY: the views live for the call
                unsafe { d.reduced.import(&model, facts.model_name.get()) };
                0
            }
            x if x == Op::PresolveRun as i32 => self.presolve_run() as i64,
            x if x == Op::PresolveLog as i32 => d.status as i64,
            x if x == Op::PresolveRemoved as i32 => {
                if arg != 0 {
                    d.reduced.clear_scale();
                }
                return None;
            }
            x if x == Op::ClearReducedIntegrality as i32 => {
                d.reduced.g.integrality.clear();
                0
            }
            x if x == Op::Facts as i32 && arg == 1 => {
                let lp = &d.reduced;
                let nc = lp.num_col as usize;
                let f = Facts {
                    num_col: lp.num_col,
                    num_row: lp.num_row,
                    num_nz: if lp.is_colwise() { lp.a.start.get(nc).copied().unwrap_or(0) } else { lp.a.value.len() as i32 },
                    is_mip: lp.integrality.iter().any(|&t| t != 0),
                    is_qp: false,
                    is_empty: lp.num_col == 0 && lp.num_row == 0,
                    has_infinite_cost: lp.has_infinite_cost,
                    model_name: super::options::RsStr::of(&lp.model_name),
                };
                // SAFETY: the run's Facts
                unsafe { *(p as *mut Facts) = f };
                0
            }
            x if x == Op::LpView as i32 && arg == 1 => {
                // SAFETY: the run's CLp
                unsafe { *(p as *mut CLp) = self.reduced_view() };
                0
            }
            x if x == Op::PrepareReducedLp as i32 => self.prepare_reduced_lp() as i64,
            x if x == Op::CopyToPresolve as i32 => {
                let r = self.run();
                r.presolve.recovered_solution.clone_from(&r.solution);
                r.presolve.recovered_basis.clone_from(&r.basis);
                0
            }
            x if x == Op::RecoveredValidity as i32 => {
                let s = &d.recovered_solution;
                s.value_valid as i64 | (s.dual_valid as i64) << 1
            }
            x if x == Op::PostsolveUndo as i32 => {
                self.postsolve_undo(arg != 0);
                0
            }
            x if x == Op::TakeRecoveredSolution as i32 => {
                let r = self.run();
                r.solution.clone_from(&r.presolve.recovered_solution);
                0
            }
            x if x == Op::TakeRecoveredBasis as i32 => {
                let r = self.run();
                let (b, rb) = (&mut r.basis, &r.presolve.recovered_basis);
                b.b.col_status.clone_from(&rb.b.col_status);
                b.b.row_status.clone_from(&rb.b.row_status);
                b.origin.push_str(": after postsolve");
                0
            }
            _ => return None,
        };
        Some(r)
    }

    /// PresolveComponent::run (HPresolve::okSetInput and run, without a
    /// MIP solver): the presolve status
    fn presolve_run(&self) -> i32 {
        let po: PresolveOptions = self.fill(Op::PresolveOptions);
        let d = self.presolve();
        let lp = &mut d.reduced.g;
        // okSetInput: an LP's integrality is all continuous
        lp.integrality.clear();
        lp.integrality.resize(lp.num_col as usize, 0);
        lp.a.ensure_colwise();
        let host = Host {
            ctx: self as *const LpMode as *mut c_void,
            log: h_log,
            timer_read: h_timer_read,
            time_string: h_time_string,
            sync_model: h_sync_model,
            set_matrix: h_set_matrix,
            flush: h_flush,
            shrink: h_shrink,
            profiling: h_profiling,
            probing_prepare: h_probing_prepare,
            mip_env: h_mip_env,
            probe: h_probe,
            set_lifting: h_set_lifting,
            lifting_opps: h_lifting_opps,
            finalise_begin: h_finalise_begin,
            domain_bounds: h_domain_bounds,
            mip_finish_presolve: h_mip_finish_presolve,
            add_cut: h_add_cut,
            upper_limit: h_upper_limit,
            set_lower_bound_zero: h_set_lower_bound_zero,
        };
        let mut p = {
            // The presolve copies its input: the borrows end before the
            // callbacks write the reduced LP and stack
            let (lp, s) = (&d.reduced, &d.stack);
            let input = Input {
                num_col: lp.num_col,
                num_row: lp.num_row,
                col_cost: &lp.col_cost,
                col_lower: &lp.col_lower,
                col_upper: &lp.col_upper,
                row_lower: &lp.row_lower,
                row_upper: &lp.row_upper,
                integrality: &lp.integrality,
                offset: lp.offset,
                maximize: lp.sense == -1,
                a_start: &lp.a.start[..lp.num_col as usize + 1],
                a_index: &lp.a.index,
                a_value: &lp.a.value,
                orig_col_index: &s.orig_col_index,
                orig_row_index: &s.orig_row_index,
                stack_data_size: s.data.len(),
                num_reductions: s.reductions.len(),
                presolve_reduction_limit: po.reduction_limit,
                model_name: String::from_utf8_lossy(&lp.model_name).into_owned(),
            };
            Presolve::new(&host, po.o, None, &input)
        };
        p.run();
        let d = self.presolve();
        d.status = p.presolve_status;
        d.log = p.analysis.log.iter().map(|l| [l.0, l.1, l.2]).collect();
        d.status
    }

    /// Op::PrepareReducedLp: the reduced LP's origin name and matrix
    /// dimensions, assessSmallValues of its matrix and cleanBounds
    fn prepare_reduced_lp(&self) -> super::Status {
        let o: CLpOptions = self.fill(Op::LpOptions);
        let d = self.presolve();
        d.prepared = true;
        let lp = &mut d.reduced.g;
        lp.a.num_col = lp.num_col;
        lp.a.num_row = lp.num_row;
        let min = lp.a.value.iter().fold(f64::INFINITY, |m, v| m.min(v.abs()));
        if min <= o.small_matrix_value {
            // analyseVectorValues reports them
            let mut v = RsMut { ptr: lp.a.value.as_mut_ptr(), len: lp.a.value.len() };
            self.fwd(Op::AssessSmallValues, 0, &mut v as *mut RsMut<f64> as *mut c_void, b"");
        }
        clean_bounds(&mut d.reduced.view(), &o)
    }

    /// Op::PostsolveUndo: undo, the row values of the model, the duals
    /// negated for a maximization if `dual`
    fn postsolve_undo(&self, dual: bool) {
        let po: PresolveOptions = self.fill(Op::PresolveOptions);
        let tol = Tolerances {
            primal_feasibility: po.o.primal_feasibility_tolerance,
            dual_feasibility: po.o.dual_feasibility_tolerance,
            mip_feasibility: po.o.mip_feasibility_tolerance,
        };
        let model: CLp = self.fill_arg(Op::LpView, 0);
        let d = self.presolve();
        d.stack.undo(&tol, &mut d.recovered_solution, &mut d.recovered_basis);
        // calculateRowValuesQuad(model, recovered)
        let s = &mut d.recovered_solution;
        s.row_value.resize(model.num_row as usize, 0.0);
        // SAFETY: the model's column-wise matrix lives for the call
        let (start, index, value) = unsafe { (model.a.start.get(), model.a.index.get(), model.a.value.get()) };
        super::edit::calculate_row_values_quad(start, index, value, &s.col_value, &mut s.row_value);
        if dual && model.sense == -1 {
            // negateReducedLpColDuals: over the reduced LP's columns
            for c in 0..d.reduced.num_col as usize {
                s.col_dual[c] = -s.col_dual[c];
            }
        }
    }
}

/// The stack, status and log of an LP run's presolve for the C++
/// PresolveComponent (HighsRunRust.cpp: kLpRustEnd)
#[repr(C)]
pub struct CPresolveExport {
    pub status: i32,
    pub prepared: bool,
    pub log: *const [i32; 3],
    pub num_log: usize,
    pub data: *const u8,
    pub data_len: usize,
    pub reductions: *const Reduction,
    pub num_reductions: usize,
    pub orig_col_index: *const i32,
    pub num_col: usize,
    pub orig_row_index: *const i32,
    pub num_row: usize,
    pub linearly_transformable: *const u8,
    pub num_lt: usize,
    pub orig_num_col: i32,
    pub orig_num_row: i32,
}

/// The LP run's reduced LP into `lp` (the C++ reduced LP, its scalars
/// copied back by C++) and views of the rest of its presolve data in
/// `out`; false if presolve was not initialised
///
/// # Safety
/// `lps` a solver, `lp` a valid view; the views in `out` are valid until
/// the solver's run changes
#[no_mangle]
pub unsafe extern "C" fn highs_rs_lps_presolve_export(lps: *mut LpSolver, lp: *mut CppLp, out: *mut CPresolveExport) -> bool {
    let d = &(*lps).run.presolve;
    if !d.active {
        return false;
    }
    let (r, c) = (&d.reduced.g, &mut *lp);
    c.num_col = r.num_col;
    c.num_row = r.num_row;
    set(&mut c.col_cost, &r.col_cost);
    set(&mut c.col_lower, &r.col_lower);
    set(&mut c.col_upper, &r.col_upper);
    set(&mut c.row_lower, &r.row_lower);
    set(&mut c.row_upper, &r.row_upper);
    let (a, ca) = (&r.a, &mut c.a);
    ca.format = a.format;
    ca.num_col = a.num_col;
    ca.num_row = a.num_row;
    set(&mut ca.start, &a.start);
    set(&mut ca.p_end, &a.p_end);
    set(&mut ca.index, &a.index);
    set(&mut ca.value, &a.value);
    c.sense = r.sense;
    c.offset = r.offset;
    set(&mut c.integrality, &r.integrality);
    let (s, cs) = (&r.scale, &mut c.scale);
    cs.strategy = s.strategy;
    cs.has_scaling = s.has_scaling;
    cs.num_col = s.num_col;
    cs.num_row = s.num_row;
    cs.cost = s.cost;
    set(&mut cs.col, &s.col);
    set(&mut cs.row, &s.row);
    c.is_scaled = r.is_scaled;
    c.has_infinite_cost = r.has_infinite_cost;
    let s = &d.stack;
    *out = CPresolveExport {
        status: d.status,
        prepared: d.prepared,
        log: d.log.as_ptr(),
        num_log: d.log.len(),
        data: s.data.as_ptr(),
        data_len: s.data.len(),
        reductions: s.reductions.as_ptr(),
        num_reductions: s.reductions.len(),
        orig_col_index: s.orig_col_index.as_ptr(),
        num_col: s.orig_col_index.len(),
        orig_row_index: s.orig_row_index.as_ptr(),
        num_row: s.orig_row_index.len(),
        linearly_transformable: s.linearly_transformable.as_ptr(),
        num_lt: s.linearly_transformable.len(),
        orig_num_col: s.orig_num_col,
        orig_num_row: s.orig_num_row,
    };
    true
}
