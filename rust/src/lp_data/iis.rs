//! The IIS (highs/lp_data/HighsIis.cpp and the drivers of
//! HighsInterface.cpp): getIisInterface with its return, the elasticity
//! filter (also Highs::feasibilityRelaxation), and HighsIis's trivial,
//! rowValueBounds, deduce, compute (the deletion filter),
//! processBoundRelaxation, setLp, setStatus, indexStatusOk, lpDataOk, lpOk
//! and the logging.
//!
//! The `Highs` objects stay C++ (highs/lp_data/HighsIisRust.cpp): the
//! incumbent (handle 0) and the ones the IIS search creates for its LP
//! solves; each step on one of them is an `Op`. Rust holds the HighsIis
//! data for the whole call (loaded at the start, stored at the end), so
//! the C++ copies and restores of `iis_` around model edits that clear it
//! are not needed. The IIS LP (`HighsIis::model_`, with names) is built by
//! C++ from Rust's arrays and kept aside until the end of the call.
//! The developer reports (kIisDevReport) and the unused sensitivity filter
//! and dual ray options are left out.

use super::ffi::{CLp, RsMut};
use super::lp_handle::LpHandle;
use super::opts::OptValue;
use super::options::RsStr;
use super::run::{MS_INFEASIBLE, MS_INTERRUPT, MS_ITERATION_LIMIT, MS_NOTSET, MS_OBJECTIVE_BOUND, MS_OPTIMAL,
                 MS_TIME_LIMIT, MS_UNBOUNDED, MS_UNBOUNDED_OR_INFEASIBLE};
use super::{Log, LogType, Status, INF};
use crate::log_user;
use crate::simplex::hekk::model_status_string;
use crate::util::fma::ClangFma;
use std::ffi::c_void;

// IisBoundStatus
const BOUND_DROPPED: i32 = -1;
const BOUND_FREE: i32 = 1;
const BOUND_LOWER: i32 = 2;
const BOUND_UPPER: i32 = 3;
const BOUND_BOXED: i32 = 4;

// IisModelStatus
const IIS_FEASIBLE: i32 = -1;
const IIS_UNKNOWN: i32 = 0;
const IIS_TIME_LIMIT: i32 = 1;
const IIS_REDUCIBLE: i32 = 2;
const IIS_IRREDUCIBLE: i32 = 3;

// IisStatus
const NOT_IN_CONFLICT: i32 = -1;
const MAYBE_IN_CONFLICT: i32 = 0;
const IN_CONFLICT: i32 = 1;

// IisStrategy
const STRATEGY_LIGHT: i32 = 0;
const STRATEGY_FROM_LP: i32 = 2;
const STRATEGY_IRREDUCIBLE: i32 = 4;
const STRATEGY_COL_PRIORITY: i32 = 8;
const STRATEGY_RELAXATION: i32 = 16;

const CALLBACK_LOGGING: i32 = 0;
const CALLBACK_SIMPLEX_INTERRUPT: i32 = 1;
const CALLBACK_MAX: i32 = 9;

const IINF: i32 = i32::MAX;
const VAR_CONTINUOUS: u8 = 0;

/// std::min and std::max
fn cmin<T: PartialOrd>(a: T, b: T) -> T {
    if b < a { b } else { a }
}
fn cmax<T: PartialOrd>(a: T, b: T) -> T {
    if a < b { b } else { a }
}

/// HighsIisInfo
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IisInfo {
    pub num_lp_solved: i32,
    pub sum_simplex_iteration_counts: i32,
    pub min_simplex_iteration_count: i32,
    pub max_simplex_iteration_count: i32,
    pub sum_simplex_times: f64,
    pub min_simplex_time: f64,
    pub max_simplex_time: f64,
    pub iis_last_disptime: f64,
    pub iis_num_disp_lines: i32,
}

impl IisInfo {
    fn clear(&mut self) {
        *self = IisInfo {
            num_lp_solved: 0,
            sum_simplex_iteration_counts: 0,
            min_simplex_iteration_count: 0,
            max_simplex_iteration_count: 0,
            sum_simplex_times: 0.0,
            min_simplex_time: 0.0,
            max_simplex_time: 0.0,
            iis_last_disptime: -INF,
            iis_num_disp_lines: 0,
        }
    }
    fn update(&mut self, time: f64, iterations: i32) {
        if self.num_lp_solved == 0 {
            self.min_simplex_iteration_count = IINF;
            self.min_simplex_time = INF;
        }
        self.num_lp_solved += 1;
        self.sum_simplex_times += time;
        self.min_simplex_time = cmin(time, self.min_simplex_time);
        self.max_simplex_time = cmax(time, self.max_simplex_time);
        self.sum_simplex_iteration_counts = self.sum_simplex_iteration_counts.wrapping_add(iterations);
        self.min_simplex_iteration_count = cmin(iterations, self.min_simplex_iteration_count);
        self.max_simplex_iteration_count = cmax(iterations, self.max_simplex_iteration_count);
    }
}

/// HighsIis's data but the IIS LP, as C++ loads and stores it
#[repr(C)]
pub struct IisState {
    pub valid: bool,
    pub status: i32,
    pub strategy: i32,
    pub col_index: RsMut<i32>,
    pub row_index: RsMut<i32>,
    pub col_bound: RsMut<i32>,
    pub row_bound: RsMut<i32>,
    pub col_status: RsMut<i32>,
    pub row_status: RsMut<i32>,
    pub info: IisInfo,
}

/// An LP for C++ to build: the IIS LP (`model_name` 1: the incumbent's
/// name with "_IIS") or deduce's sub-LP (0: no name). Names are those of
/// the incumbent's columns `col_map` and rows `row_map`, when it has
/// names.
#[repr(C)]
pub struct LpArrays {
    pub num_col: i32,
    pub num_row: i32,
    pub format: i32,
    pub model_name: i32,
    pub col_cost: RsMut<f64>,
    pub col_lower: RsMut<f64>,
    pub col_upper: RsMut<f64>,
    pub row_lower: RsMut<f64>,
    pub row_upper: RsMut<f64>,
    pub start: RsMut<i32>,
    pub index: RsMut<i32>,
    pub value: RsMut<f64>,
    pub col_map: RsMut<i32>,
    pub row_map: RsMut<i32>,
}

/// The arguments of an op
#[repr(C)]
pub struct Args {
    pub h: i32,
    pub i: i32,
    pub j: i32,
    pub x: f64,
    pub y: f64,
    pub s: RsStr,
    pub p: [*const c_void; 6],
}

/// The incumbent `Highs` (with its HighsIis) as Rust sees it
#[repr(C)]
pub struct IisHost {
    pub log: Log,
    pub ctx: *mut c_void,
    pub op: unsafe extern "C" fn(*mut c_void, i32, *const Args) -> f64,
    /// A view of the incumbent's model_.lp_, with its numbers of column
    /// and row names
    pub lp: unsafe extern "C" fn(*mut c_void, *mut CLp, *mut usize, *mut usize),
    /// The incumbent's column (row) name i, or its model name (i < 0)
    pub name: unsafe extern "C" fn(*mut c_void, bool, i32) -> RsStr,
    /// The incumbent's solution_.col_value
    pub col_value: unsafe extern "C" fn(*mut c_void) -> RsMut<f64>,
}

/// The steps on C++ objects (HighsIisRust.cpp: IisOp). `h` is the Highs
/// handle (0: the incumbent).
#[repr(i32)]
#[derive(Clone, Copy)]
enum Op {
    /// The incumbent's iis_ into p0 (IisState, views)
    LoadIis = 1,
    /// p0 (IisState) into the incumbent's iis_, with the kept IIS LP
    StoreIis,
    /// HighsIis::model_.clear() of the kept IIS LP
    ClearIisModel,
    /// The kept IIS LP from p0 (LpArrays)
    SetIisLp,
    /// A new Highs -> its handle
    NewHighs,
    DeleteHighs,
    /// passOptions(the incumbent's options_), with i = 1 the time,
    /// iteration and objective bound limits removed
    PassOptions,
    /// setOptionValue(s, bool i / int i / double x / string p0 (RsStr))
    SetOptionBool,
    SetOptionInt,
    SetOptionDouble,
    SetOptionString,
    /// The incumbent's logging and simplex interrupt callbacks to h
    PropagateCallbacks,
    /// passModel(the LP p0 (LpArrays)) -> status
    PassModelArrays,
    /// passModel(the kept IIS LP) -> status
    PassIisModel,
    /// changeColsCost(i, j, p0) -> status
    ChangeColsCost,
    /// optimizeModel() -> status
    OptimizeModel,
    RunTime,
    SimplexIterations,
    ModelStatus,
    /// changeColBounds / changeRowBounds(i, x, y) -> status
    ChangeColBounds,
    ChangeRowBounds,
    /// writeModel("")
    WriteModel,
    ZeroAllClocks,
    /// callback_.active[i]
    CallbackActive,
    StopCallback,
    StartCallback,
    /// Save / restore the incumbent's options_
    SaveOptions,
    RestoreOptions,
    /// model_.lp_.a_matrix_.ensureColwise()
    EnsureColwise,
    /// An option: i = 0 iis_strategy, 1 primal_feasibility_tolerance, 2
    /// iis_time_limit, 3 output_flag, 5 log_dev_level
    GetOption,
    /// options_.output_flag = i
    SetOutputFlag,
    InvalidateSolverData,
    /// passModelName(s)
    PassModelName,
    /// changeColsIntegrality / changeColsBounds(i, j, p0[, p1]) -> status
    ChangeColsIntegrality,
    ChangeColsBounds,
    /// addCols(i, p0, p1, p2, j, p3, p4, p5) / addRows(i, p0, p1, j, p2,
    /// p3, p4) -> status
    AddCols,
    AddRows,
    /// passColName / passRowName(i, s)
    PassColName,
    PassRowName,
    /// deleteRows / deleteCols(i, j) -> status
    DeleteRows,
    DeleteCols,
    /// basis_.valid = false
    BasisInvalid,
    /// The elasticity filter's solution: row values, objective x, KKT
    /// failures
    ElasticSolution,
    SetModelStatus,
    /// info_.objective_function_value
    ObjectiveValue,
    /// The incumbent's engine (an LpHandle, its option values current)
    /// into p0 (a *mut *mut LpHandle)
    Engine,
    /// Whether the incumbent's logging or simplex interrupt callbacks are
    /// to be propagated (a user log callback, or those callbacks active)
    CallbacksToPropagate,
}

struct H<'a> {
    host: &'a IisHost,
    /// The LP solvers of the IIS search (handles 1, 2, ...): LpHandles
    /// with the incumbent's option values and its simplex interrupt
    /// callback
    locals: std::cell::RefCell<Vec<Option<Box<LpHandle>>>>,
}

impl H<'_> {
    fn new(host: &IisHost) -> H<'_> {
        H { host, locals: std::cell::RefCell::new(Vec::new()) }
    }
    fn call(&self, op: Op, a: Args) -> f64 {
        if a.h >= 1 || matches!(op, Op::NewHighs) {
            if let Some(r) = self.local(op, &a) {
                return r;
            }
        }
        // SAFETY: the host's op takes its ctx and the arguments it lists
        unsafe { (self.host.op)(self.host.ctx, op as i32, &a) }
    }
    /// The incumbent's engine
    fn engine(&self) -> &'static LpHandle {
        let mut p: *mut LpHandle = std::ptr::null_mut();
        let mut a = args(0);
        a.p[0] = &mut p as *mut *mut LpHandle as *const c_void;
        // SAFETY: the op writes the pointer
        unsafe { (self.host.op)(self.host.ctx, Op::Engine as i32, &a) };
        // SAFETY: the incumbent's engine outlives the IIS call
        unsafe { &*p }
    }
    /// The op of the IIS search's LP solver `a.h` (None: the incumbent's)
    fn local(&self, op: Op, a: &Args) -> Option<f64> {
        let mut locals = self.locals.borrow_mut();
        if let Op::NewHighs = op {
            locals.push(Some(LpHandle::new()));
            return Some(locals.len() as f64);
        }
        let k = a.h as usize - 1;
        let x = locals[k].as_mut().expect("a live LP solver of the IIS");
        let st = |s: Status| s as i32 as f64;
        // SAFETY: the option name and string value live for the call
        let name = || unsafe { std::str::from_utf8(a.s.get()).unwrap_or("") };
        let r = match op {
            Op::DeleteHighs => {
                locals[k] = None;
                0.0
            }
            Op::PassOptions => {
                x.opts.assign(&self.engine().opts);
                if a.i != 0 {
                    x.opts.time_limit = INF;
                    x.opts.simplex_iteration_limit = IINF;
                    x.opts.objective_bound = INF;
                }
                0.0
            }
            Op::SetOptionBool => st(set_ok(x.set_option(name(), OptValue::Bool(a.i != 0)))),
            Op::SetOptionInt => st(set_ok(x.set_option(name(), OptValue::Int(a.i)))),
            Op::SetOptionDouble => st(set_ok(x.set_option(name(), OptValue::Double(a.x)))),
            Op::SetOptionString => {
                // SAFETY: p0 is the value's RsStr
                let v = unsafe { (*(a.p[0] as *const RsStr)).get() };
                st(set_ok(x.set_option(name(), OptValue::Str(v))))
            }
            Op::PropagateCallbacks => {
                drop(locals);
                let propagate = self.call(Op::CallbacksToPropagate, args(0)) != 0.0;
                let cb = if propagate { self.engine().host_simplex_callback() } else { None };
                self.locals.borrow_mut()[k].as_mut().expect("a live LP solver").simplex_callback = cb;
                return Some(0.0);
            }
            Op::PassModelArrays => {
                // SAFETY: p0 is an LpArrays of the caller's vectors
                let lp = unsafe { lp_of(&*(a.p[0] as *const LpArrays), b"") };
                st(x.pass_model(lp))
            }
            Op::ChangeColsCost => {
                let n = (a.j - a.i + 1).max(0) as usize;
                // SAFETY: p0 holds the interval's costs
                let cost = unsafe { std::slice::from_raw_parts(a.p[0] as *const f64, n) };
                st(x.change_col_costs_interval(a.i, a.j, cost))
            }
            Op::OptimizeModel => st(x.optimize_lp()),
            Op::RunTime => x.run_time(),
            Op::SimplexIterations => x.info().simplex_iteration_count as f64,
            Op::ModelStatus => x.model_status() as f64,
            Op::ChangeColBounds => st(x.change_col_bounds_set(&[a.i], &[a.x], &[a.y])),
            Op::ChangeRowBounds => st(x.change_row_bounds_set(&[a.i], &[a.x], &[a.y])),
            // writeModel("") of a silent solver writes nothing
            Op::WriteModel => {
                debug_assert!(!x.opts.output_flag);
                0.0
            }
            _ => return None,
        };
        Some(r)
    }
    /// passModel of an LP of the IIS (the IIS LP: its model name the
    /// incumbent's + "_IIS")
    fn pass_lp(&self, k: i32, a: &LpArrays) -> Status {
        let name = if a.model_name != 0 {
            let mut n = self.name(true, usize::MAX).into_bytes();
            n.extend_from_slice(b"_IIS");
            n
        } else {
            Vec::new()
        };
        let mut locals = self.locals.borrow_mut();
        let x = locals[k as usize - 1].as_mut().expect("a live LP solver of the IIS");
        // SAFETY: the arrays are the caller's vectors
        x.pass_model(unsafe { lp_of(a, &name) })
    }
    fn on(&self, op: Op, h: i32) -> f64 {
        self.call(op, args(h))
    }
    fn status(&self, op: Op, a: Args) -> Status {
        status_of(self.call(op, a))
    }
    fn log(&self) -> &Log {
        &self.host.log
    }
    fn opt(&self, i: i32) -> f64 {
        self.call(Op::GetOption, Args { i, ..args(0) })
    }
    fn strategy(&self) -> i32 {
        self.opt(0) as i32
    }
    fn tol(&self) -> f64 {
        self.opt(1)
    }
    fn model_status(&self, h: i32) -> i32 {
        self.on(Op::ModelStatus, h) as i32
    }
    fn set_bool(&self, h: i32, name: &[u8], v: bool) {
        self.call(Op::SetOptionBool, Args { i: v as i32, s: RsStr::of(name), ..args(h) });
    }
    fn set_int(&self, h: i32, name: &[u8], v: i32) {
        self.call(Op::SetOptionInt, Args { i: v, s: RsStr::of(name), ..args(h) });
    }
    fn set_double(&self, h: i32, name: &[u8], v: f64) {
        self.call(Op::SetOptionDouble, Args { x: v, s: RsStr::of(name), ..args(h) });
    }
    fn bounds(&self, h: i32, row: bool, i: i32, lower: f64, upper: f64) -> Status {
        let op = if row { Op::ChangeRowBounds } else { Op::ChangeColBounds };
        self.status(op, Args { i, x: lower, y: upper, ..args(h) })
    }
    /// A view of the incumbent's LP, valid until the next op that
    /// changes it
    fn lp(&self) -> Lp<'static> {
        let mut c = std::mem::MaybeUninit::<CLp>::uninit();
        let (mut nc, mut nr) = (0usize, 0usize);
        // SAFETY: the host fills the view; its arrays live until the LP
        // changes, and each caller refetches the view after such an op
        unsafe {
            (self.host.lp)(self.host.ctx, c.as_mut_ptr(), &mut nc, &mut nr);
            let c = c.assume_init();
            Lp {
                num_col: c.num_col as usize,
                num_row: c.num_row as usize,
                col_lower: c.col_lower.get(),
                col_upper: c.col_upper.get(),
                row_lower: c.row_lower.get(),
                row_upper: c.row_upper.get(),
                format: c.a.format,
                start: c.a.start.get(),
                index: c.a.index.get(),
                value: c.a.value.get(),
                has_col_names: nc > 0,
                has_row_names: nr > 0,
            }
        }
    }
    fn name(&self, is_col: bool, i: usize) -> String {
        // SAFETY: the name lives until the incumbent's names change
        let s = unsafe { (self.host.name)(self.host.ctx, is_col, i as i32).get() };
        // ponytail: a non-UTF-8 name is logged lossily
        String::from_utf8_lossy(s).into_owned()
    }
    fn col_value(&self) -> &'static [f64] {
        // SAFETY: valid until the next solve, before which it is refetched
        unsafe { (self.host.col_value)(self.host.ctx).get() }
    }
    fn run_time(&self, h: i32) -> f64 {
        self.on(Op::RunTime, h)
    }
    fn iterations(&self, h: i32) -> i32 {
        self.on(Op::SimplexIterations, h) as i32
    }
}

/// setOptionValue's status of Opts::set
fn set_ok(ok: bool) -> Status {
    if ok {
        Status::Ok
    } else {
        Status::Error
    }
}

/// The LP of an LpArrays (as HighsIisRust.cpp's buildLp, without names)
///
/// # Safety
/// The arrays valid
unsafe fn lp_of(a: &LpArrays, model_name: &[u8]) -> super::lp::Lp {
    let mut lp = super::lp::Lp::default();
    let g = &mut lp.g;
    g.num_col = a.num_col;
    g.num_row = a.num_row;
    g.col_cost = a.col_cost.get().to_vec();
    g.col_lower = a.col_lower.get().to_vec();
    g.col_upper = a.col_upper.get().to_vec();
    g.row_lower = a.row_lower.get().to_vec();
    g.row_upper = a.row_upper.get().to_vec();
    g.a.format = a.format;
    g.a.num_col = a.num_col;
    g.a.num_row = a.num_row;
    g.a.start = a.start.get().to_vec();
    g.a.index = a.index.get().to_vec();
    g.a.value = a.value.get().to_vec();
    lp.model_name = model_name.to_vec();
    lp
}

fn args(h: i32) -> Args {
    Args { h, i: 0, j: 0, x: 0.0, y: 0.0, s: RsStr::of(&[]), p: [std::ptr::null(); 6] }
}

fn status_of(v: f64) -> Status {
    match v as i32 {
        0 => Status::Ok,
        1 => Status::Warning,
        _ => Status::Error,
    }
}

/// data() of a std::vector holding `v` (null when empty, as for a vector
/// that never allocated)
fn ptr<T>(v: &[T]) -> *const c_void {
    if v.is_empty() { std::ptr::null() } else { v.as_ptr() as *const c_void }
}

fn rs<T>(v: &[T]) -> RsMut<T> {
    RsMut { ptr: v.as_ptr() as *mut T, len: v.len() }
}

/// The parts of a HighsLp the IIS reads
struct Lp<'a> {
    num_col: usize,
    num_row: usize,
    col_lower: &'a [f64],
    col_upper: &'a [f64],
    row_lower: &'a [f64],
    row_upper: &'a [f64],
    format: i32,
    start: &'a [i32],
    index: &'a [i32],
    value: &'a [f64],
    has_col_names: bool,
    has_row_names: bool,
}

impl Lp<'_> {
    fn colwise(&self) -> bool {
        self.format == super::matrix_format::COLWISE
    }
    fn range(&self, i: usize) -> std::ops::Range<usize> {
        self.start[i] as usize..self.start[i + 1] as usize
    }
}

/// An LP built here (deduce's sub-LP, the IIS LP)
#[derive(Default)]
struct OwnLp {
    format: i32,
    col_cost: Vec<f64>,
    col_lower: Vec<f64>,
    col_upper: Vec<f64>,
    row_lower: Vec<f64>,
    row_upper: Vec<f64>,
    start: Vec<i32>,
    index: Vec<i32>,
    value: Vec<f64>,
    col_map: Vec<i32>,
    row_map: Vec<i32>,
}

impl OwnLp {
    fn new() -> OwnLp {
        OwnLp { format: super::matrix_format::COLWISE, start: vec![0], ..Default::default() }
    }
    fn view(&self) -> Lp<'_> {
        Lp {
            num_col: self.col_cost.len(),
            num_row: self.row_lower.len(),
            col_lower: &self.col_lower,
            col_upper: &self.col_upper,
            row_lower: &self.row_lower,
            row_upper: &self.row_upper,
            format: self.format,
            start: &self.start,
            index: &self.index,
            value: &self.value,
            has_col_names: false,
            has_row_names: false,
        }
    }
    fn arrays(&self, model_name: i32) -> LpArrays {
        LpArrays {
            num_col: self.col_cost.len() as i32,
            num_row: self.row_lower.len() as i32,
            format: self.format,
            model_name,
            col_cost: rs(&self.col_cost),
            col_lower: rs(&self.col_lower),
            col_upper: rs(&self.col_upper),
            row_lower: rs(&self.row_lower),
            row_upper: rs(&self.row_upper),
            start: rs(&self.start),
            index: rs(&self.index),
            value: rs(&self.value),
            col_map: rs(&self.col_map),
            row_map: rs(&self.row_map),
        }
    }
}

/// HighsIis (without the IIS LP, which C++ keeps)
struct Iis {
    valid: bool,
    status: i32,
    strategy: i32,
    col_index: Vec<i32>,
    row_index: Vec<i32>,
    col_bound: Vec<i32>,
    row_bound: Vec<i32>,
    col_status: Vec<i32>,
    row_status: Vec<i32>,
    info: IisInfo,
    /// The IIS LP as built by setLp, for lpDataOk and lpOk
    lp: OwnLp,
}

impl Iis {
    fn load(h: &H) -> Iis {
        let mut s = std::mem::MaybeUninit::<IisState>::uninit();
        h.call(Op::LoadIis, Args { p: [s.as_mut_ptr() as *const c_void, std::ptr::null(), std::ptr::null(),
                                       std::ptr::null(), std::ptr::null(), std::ptr::null()], ..args(0) });
        // SAFETY: C++ filled the state with views of its vectors, read
        // here before any other op
        unsafe {
            let s = s.assume_init();
            Iis {
                valid: s.valid,
                status: s.status,
                strategy: s.strategy,
                col_index: s.col_index.get().to_vec(),
                row_index: s.row_index.get().to_vec(),
                col_bound: s.col_bound.get().to_vec(),
                row_bound: s.row_bound.get().to_vec(),
                col_status: s.col_status.get().to_vec(),
                row_status: s.row_status.get().to_vec(),
                info: s.info,
                lp: OwnLp::new(),
            }
        }
    }
    fn store(&self, h: &H) {
        let s = IisState {
            valid: self.valid,
            status: self.status,
            strategy: self.strategy,
            col_index: rs(&self.col_index),
            row_index: rs(&self.row_index),
            col_bound: rs(&self.col_bound),
            row_bound: rs(&self.row_bound),
            col_status: rs(&self.col_status),
            row_status: rs(&self.row_status),
            info: self.info,
        };
        let mut a = args(0);
        a.p[0] = &s as *const IisState as *const c_void;
        h.call(Op::StoreIis, a);
    }
    /// clearData
    fn clear_data(&mut self, h: &H) {
        self.valid = false;
        self.status = IIS_UNKNOWN;
        self.strategy = STRATEGY_LIGHT;
        self.col_index.clear();
        self.row_index.clear();
        self.col_bound.clear();
        self.row_bound.clear();
        self.col_status.clear();
        self.row_status.clear();
        h.on(Op::ClearIisModel, 0);
    }
    fn clear(&mut self, h: &H) {
        self.clear_data(h);
        self.info.clear();
    }
    fn clear_log_info(&mut self) {
        self.info.iis_last_disptime = -INF;
        self.info.iis_num_disp_lines = 0;
    }
    fn add_col(&mut self, col: i32, status: i32) {
        self.col_index.push(col);
        self.col_bound.push(status);
    }
    fn add_row(&mut self, row: i32, status: i32) {
        self.row_index.push(row);
        self.row_bound.push(status);
    }

    fn report_iteration(&mut self, h: &H, iter: i32, num_rows_remaining: i32, force: bool) {
        if h.opt(3) == 0.0 {
            return;
        }
        let min_interval = 5.0;
        let runtime = self.info.sum_simplex_times;
        if !force && self.info.iis_last_disptime > -0.5 * INF && runtime - self.info.iis_last_disptime < min_interval {
            return;
        }
        self.info.iis_last_disptime = runtime;
        // Widths: "Iteration" + 2, "Rows" + 17, "Runtime" + 17
        if self.info.iis_num_disp_lines % 20 == 0 {
            log_user!(h.log(), LogType::Info, "%11s%21s%24s\n", "Iteration", "Rows", "Runtime");
        }
        self.info.iis_num_disp_lines += 1;
        let time_string = crate::util::printf::sprintf("%.2fs", &[runtime.into()]);
        log_user!(h.log(), LogType::Info, "%11d%21d%24s\n", iter, num_rows_remaining, time_string.as_str());
    }

    fn report_final(&self, h: &H) {
        if h.opt(3) == 0.0 {
            return;
        }
        let log = h.log();
        log_user!(log, LogType::Info, "\n");
        log_user!(log, LogType::Info, "%-19s : %s\n", "IIS status", model_status_str(self.status));
        log_user!(log, LogType::Info, "%-19s : %d\n", "Rows", self.row_index.len() as i32);
        log_user!(log, LogType::Info, "%-19s : %d\n", "Columns", self.col_index.len() as i32);
        log_user!(log, LogType::Info, "%-19s : %.2fs\n", "HiGHS run time", self.info.sum_simplex_times);
    }

    fn found(&mut self, h: &H, status: i32) {
        self.valid = true;
        self.status = status;
        self.strategy = h.strategy();
    }

    /// HighsIis::trivial
    fn trivial(&mut self, h: &H, lp: &Lp) -> bool {
        self.clear(h);
        let tol = h.tol();
        let col_priority = h.strategy() & STRATEGY_COL_PRIORITY != 0;
        for k in 0..2 {
            if (col_priority && k == 0) || (!col_priority && k == 1) {
                if let Some(c) = (0..lp.num_col).find(|&c| lp.col_lower[c] - lp.col_upper[c] > 2.0 * tol) {
                    self.add_col(c as i32, BOUND_BOXED);
                }
                if !self.col_index.is_empty() {
                    break;
                }
            } else {
                if let Some(r) = (0..lp.num_row).find(|&r| lp.row_lower[r] - lp.row_upper[r] > 2.0 * tol) {
                    self.add_row(r as i32, BOUND_BOXED);
                }
                if !self.row_index.is_empty() {
                    break;
                }
            }
        }
        if !self.col_index.is_empty() || !self.row_index.is_empty() {
            self.found(h, IIS_IRREDUCIBLE);
            return true;
        }
        // Empty rows that cannot have zero activity
        let count: Vec<i32> = if lp.colwise() {
            let mut count = vec![0; lp.num_row];
            for &r in &lp.index[..lp.start[lp.num_col] as usize] {
                count[r as usize] += 1;
            }
            count
        } else {
            (0..lp.num_row).map(|r| lp.start[r + 1] - lp.start[r]).collect()
        };
        for r in 0..lp.num_row {
            if count[r] > 0 {
                continue;
            }
            if lp.row_lower[r] > tol {
                self.add_row(r as i32, BOUND_LOWER);
            } else if lp.row_upper[r] < -tol {
                self.add_row(r as i32, BOUND_UPPER);
            }
            if !self.row_index.is_empty() {
                self.found(h, IIS_IRREDUCIBLE);
                return true;
            }
        }
        false
    }

    /// HighsIis::rowValueBounds
    fn row_value_bounds(&mut self, h: &H, lp: &Lp) -> bool {
        self.clear(h);
        let tol = h.tol();
        let mut lower_value = vec![0.0; lp.num_row];
        let mut upper_value = vec![0.0; lp.num_row];
        // clang fuses the sums
        if lp.colwise() {
            for c in 0..lp.num_col {
                let (lower, upper) = (lp.col_lower[c], lp.col_upper[c]);
                for el in lp.range(c) {
                    let r = lp.index[el] as usize;
                    let value = lp.value[el];
                    if value > 0.0 {
                        lower_value[r] = value.mul_add_c(lower, lower_value[r]);
                        upper_value[r] = value.mul_add_c(upper, upper_value[r]);
                    } else {
                        lower_value[r] = value.mul_add_c(upper, lower_value[r]);
                        upper_value[r] = value.mul_add_c(lower, upper_value[r]);
                    }
                }
            }
        } else {
            for r in 0..lp.num_row {
                let (mut lo, mut up) = (0.0f64, 0.0f64);
                for el in lp.range(r) {
                    let c = lp.index[el] as usize;
                    let (lower, upper) = (lp.col_lower[c], lp.col_upper[c]);
                    let value = lp.value[el];
                    if value > 0.0 {
                        lo = value.mul_add_c(lower, lo);
                        up = value.mul_add_c(upper, up);
                    } else {
                        lo = value.mul_add_c(upper, lo);
                        up = value.mul_add_c(lower, up);
                    }
                }
                lower_value[r] = lo;
                upper_value[r] = up;
            }
        }
        let mut below_lower = false;
        for r in 0..lp.num_row {
            below_lower = upper_value[r] < lp.row_lower[r] - tol;
            let above_upper = lower_value[r] > lp.row_upper[r] + tol;
            if below_lower || above_upper {
                self.row_index.push(r as i32);
                self.row_bound.push(if below_lower { BOUND_LOWER } else { BOUND_UPPER });
                break;
            }
        }
        if self.row_index.is_empty() {
            // Nothing found, but the IIS data are still valid
            self.clear(h);
            self.found(h, IIS_UNKNOWN);
            return false;
        }
        let r = self.row_index[0] as usize;
        let row_name = if lp.has_row_names { format!("({})", h.name(false, r)) } else { String::new() };
        if below_lower {
            log_user!(h.log(), LogType::Info, "LP row %d %shas maximum row value of %g, below lower bound of %g\n",
                      r as i32, row_name.as_str(), upper_value[r], lp.row_lower[r]);
        } else {
            log_user!(h.log(), LogType::Info, "LP row %d %shas minimum row value of %g, above upper bound of %g\n",
                      r as i32, row_name.as_str(), lower_value[r], lp.row_upper[r]);
        }
        let col_bound = |value: f64| {
            if (value > 0.0) == below_lower { BOUND_UPPER } else { BOUND_LOWER }
        };
        if lp.colwise() {
            for c in 0..lp.num_col {
                for el in lp.range(c) {
                    let value = lp.value[el];
                    if lp.index[el] as usize == r && value != 0.0 {
                        self.add_col(c as i32, col_bound(value));
                    }
                }
            }
        } else {
            for el in lp.range(r) {
                let value = lp.value[el];
                if value != 0.0 {
                    self.add_col(lp.index[el], col_bound(value));
                }
            }
        }
        self.found(h, IIS_IRREDUCIBLE);
        true
    }

    /// HighsIis::deduce: the LP of the infeasible rows (the incumbent's
    /// LP is column-wise)
    fn deduce(&mut self, h: &H) -> Status {
        let lp = h.lp();
        let from_row = std::mem::take(&mut self.row_index);
        self.clear_data(h);
        let mut to_row = vec![-1i32; lp.num_row];
        for (k, &r) in from_row.iter().enumerate() {
            to_row[r as usize] = k as i32;
        }
        let from_col: Vec<i32> =
            (0..lp.num_col).filter(|&c| lp.range(c).any(|el| to_row[lp.index[el] as usize] >= 0)).map(|c| c as i32).collect();
        let mut to = OwnLp::new();
        for &c in &from_col {
            let c = c as usize;
            to.col_cost.push(0.0);
            to.col_lower.push(lp.col_lower[c]);
            to.col_upper.push(lp.col_upper[c]);
            for el in lp.range(c) {
                let t = to_row[lp.index[el] as usize];
                if t >= 0 {
                    to.index.push(t);
                    to.value.push(lp.value[el]);
                }
            }
            to.start.push(to.index.len() as i32);
        }
        for &r in &from_row {
            to.row_lower.push(lp.row_lower[r as usize]);
            to.row_upper.push(lp.row_upper[r as usize]);
        }
        to.col_map = from_col.clone();
        to.row_map = from_row.clone();
        let status = self.compute(h, &to);
        for c in self.col_index.iter_mut() {
            *c = from_col[*c as usize];
        }
        for r in self.row_index.iter_mut() {
            *r = from_row[*r as usize];
        }
        status
    }

    /// HighsIis::setLp: the IIS LP, built by C++ (with names) and kept
    /// here for lpDataOk and lpOk
    fn set_lp(&mut self, h: &H, lp: &Lp) {
        let colwise = lp.colwise();
        let mut iis_lp = OwnLp::new();
        iis_lp.format = lp.format;
        let mut iis_row = Vec::new();
        let mut iis_col = Vec::new();
        if colwise {
            iis_row = vec![-1i32; lp.num_row];
            for (k, &r) in self.row_index.iter().enumerate() {
                iis_row[r as usize] = k as i32;
            }
        } else {
            iis_col = vec![-1i32; lp.num_col];
            for (k, &c) in self.col_index.iter().enumerate() {
                iis_col[c as usize] = k as i32;
            }
        }
        for (k, &r) in self.row_index.iter().enumerate() {
            let r = r as usize;
            let b = self.row_bound[k];
            iis_lp.row_lower.push(if b == BOUND_LOWER || b == BOUND_BOXED { lp.row_lower[r] } else { -INF });
            iis_lp.row_upper.push(if b == BOUND_UPPER || b == BOUND_BOXED { lp.row_upper[r] } else { INF });
            if !colwise {
                for el in lp.range(r) {
                    let k = iis_col[lp.index[el] as usize];
                    if k >= 0 {
                        iis_lp.index.push(k);
                        iis_lp.value.push(lp.value[el]);
                    }
                }
            }
        }
        for (k, &c) in self.col_index.iter().enumerate() {
            let c = c as usize;
            iis_lp.col_cost.push(0.0);
            let b = self.col_bound[k];
            iis_lp.col_lower.push(if b == BOUND_LOWER || b == BOUND_BOXED { lp.col_lower[c] } else { -INF });
            iis_lp.col_upper.push(if b == BOUND_UPPER || b == BOUND_BOXED { lp.col_upper[c] } else { INF });
            if colwise {
                for el in lp.range(c) {
                    let k = iis_row[lp.index[el] as usize];
                    if k >= 0 {
                        iis_lp.index.push(k);
                        iis_lp.value.push(lp.value[el]);
                    }
                }
            }
            // As the C++, one start per column even when row-wise
            iis_lp.start.push(iis_lp.index.len() as i32);
        }
        iis_lp.col_map = self.col_index.clone();
        iis_lp.row_map = self.row_index.clone();
        let arrays = iis_lp.arrays(1);
        let mut a = args(0);
        a.p[0] = &arrays as *const LpArrays as *const c_void;
        h.call(Op::SetIisLp, a);
        self.lp = iis_lp;
    }

    /// nonIsStatus
    fn non_is_status(&self) -> i32 {
        let is_feasible = self.status == IIS_FEASIBLE;
        let has_is = !self.col_index.is_empty() || !self.row_index.is_empty();
        if is_feasible || has_is { NOT_IN_CONFLICT } else { MAYBE_IN_CONFLICT }
    }

    /// setStatus
    fn set_status(&mut self, lp: &Lp) {
        if !self.valid {
            return;
        }
        let non_is = self.non_is_status();
        let in_is = if self.status == IIS_IRREDUCIBLE { IN_CONFLICT } else { MAYBE_IN_CONFLICT };
        self.col_status = vec![non_is; lp.num_col];
        self.row_status = vec![non_is; lp.num_row];
        for &c in &self.col_index {
            self.col_status[c as usize] = in_is;
        }
        for &r in &self.row_index {
            self.row_status[r as usize] = in_is;
        }
    }

    /// indexStatusOk
    fn index_status_ok(&self, lp: &Lp) -> bool {
        if self.col_status.len() != lp.num_col || self.row_status.len() != lp.num_row {
            return false;
        }
        let true_iis = self.col_status.contains(&IN_CONFLICT) || self.row_status.contains(&IN_CONFLICT);
        let in_is = if true_iis { IN_CONFLICT } else { MAYBE_IN_CONFLICT };
        const ILLEGAL: i32 = -99;
        let mut col_status = self.col_status.clone();
        let mut row_status = self.row_status.clone();
        for &c in &self.col_index {
            if self.col_status[c as usize] != in_is {
                return false;
            }
            col_status[c as usize] = ILLEGAL;
        }
        for &r in &self.row_index {
            if self.row_status[r as usize] != in_is {
                return false;
            }
            row_status[r as usize] = ILLEGAL;
        }
        let non_is = self.non_is_status();
        col_status.iter().chain(&row_status).all(|&s| s <= ILLEGAL || s == non_is)
    }

    /// lpDataOk: the IIS LP is the incumbent's reduced to the IIS
    fn lp_data_ok(&self, lp: &Lp) -> bool {
        let il = self.lp.view();
        let (n_col, n_row) = (self.col_index.len(), self.row_index.len());
        if il.num_col != n_col || il.num_row != n_row {
            return false;
        }
        let colwise = lp.colwise();
        let mut iis_row = vec![-1i32; lp.num_row];
        let mut iis_col = vec![-1i32; lp.num_col];
        for (k, &r) in self.row_index.iter().enumerate() {
            let r = r as usize;
            iis_row[r] = k as i32;
            let b = self.row_bound[k];
            let lower = if b == BOUND_LOWER || b == BOUND_BOXED { lp.row_lower[r] } else { -INF };
            let upper = if b == BOUND_UPPER || b == BOUND_BOXED { lp.row_upper[r] } else { INF };
            if il.row_lower[k] != lower || il.row_upper[k] != upper {
                return false;
            }
        }
        for (k, &c) in self.col_index.iter().enumerate() {
            let c = c as usize;
            iis_col[c] = k as i32;
            if self.lp.col_cost[k] != 0.0 {
                return false;
            }
            let b = self.col_bound[k];
            let lower = if b == BOUND_LOWER || b == BOUND_BOXED { lp.col_lower[c] } else { -INF };
            let upper = if b == BOUND_UPPER || b == BOUND_BOXED { lp.col_upper[c] } else { INF };
            if il.col_lower[k] != lower || il.col_upper[k] != upper {
                return false;
            }
        }
        // The vectors (columns if column-wise) of both matrices agree
        let (vecs, inner_map, iis_of, n_inner_iis, n_inner) = if colwise {
            (&self.col_index, &self.row_index, &iis_row, n_row, lp.num_row)
        } else {
            (&self.row_index, &self.col_index, &iis_col, n_col, lp.num_col)
        };
        for (k, &v) in vecs.iter().enumerate() {
            let v = v as usize;
            // The IIS vector scattered against the LP's
            let mut index = vec![-1i32; n_inner_iis];
            let mut value = vec![INF; n_inner_iis];
            for el in il.range(k) {
                let ki = il.index[el] as usize;
                index[ki] = inner_map[ki];
                value[ki] = il.value[el];
            }
            for el in lp.range(v) {
                let i = lp.index[el];
                let ki = iis_of[i as usize];
                if ki >= 0 {
                    let ki = ki as usize;
                    if index[ki] != i || value[ki] != lp.value[el] {
                        return false;
                    }
                    index[ki] = -1;
                    value[ki] = INF;
                }
            }
            // The LP vector scattered against the IIS's
            let mut index = vec![-1i32; n_inner];
            let mut value = vec![INF; n_inner];
            for el in lp.range(v) {
                let i = lp.index[el] as usize;
                index[i] = iis_of[i];
                value[i] = lp.value[el];
            }
            for el in il.range(k) {
                let ki = il.index[el];
                let i = inner_map[ki as usize] as usize;
                if index[i] != ki || value[i] != il.value[el] {
                    return false;
                }
            }
        }
        true
    }

    /// lpOk: the IIS LP is infeasible, and feasible if any bound of an
    /// irreducible one is relaxed
    fn lp_ok(&self, h: &H) -> bool {
        if !self.valid {
            return true;
        }
        if self.col_index.is_empty() {
            return true;
        }
        let il = &self.lp;
        let k = h.on(Op::NewHighs, 0) as i32;
        h.call(Op::PassOptions, Args { i: 1, ..args(k) });
        h.set_bool(k, b"output_flag", false);
        h.pass_lp(k, &il.arrays(1));
        h.on(Op::WriteModel, k);
        let status = h.status(Op::OptimizeModel, args(k));
        let result = (|| {
            if status != Status::Ok || h.model_status(k) != MS_INFEASIBLE {
                log_user!(h.log(), LogType::Warning, "HighsIis: Failed to prove infeasibility for IIS LP\n");
                return false;
            }
            if self.status != IIS_IRREDUCIBLE {
                return true;
            }
            let optimal = || {
                if h.opt(5) > 0.0 {
                    h.on(Op::WriteModel, k);
                }
                h.on(Op::OptimizeModel, k);
                h.model_status(k) == MS_OPTIMAL
            };
            for row in [false, true] {
                let (index, bound, lower, upper, t) = if row {
                    (&self.row_index, &self.row_bound, &il.row_lower, &il.row_upper, LogType::Error)
                } else {
                    (&self.col_index, &self.col_bound, &il.col_lower, &il.col_upper, LogType::Warning)
                };
                for (ki, &i) in index.iter().enumerate() {
                    for drop_lower in [true, false] {
                        if bound[ki] != if drop_lower { BOUND_LOWER } else { BOUND_UPPER } {
                            continue;
                        }
                        let (lo, up) = if drop_lower { (-INF, upper[ki]) } else { (lower[ki], INF) };
                        h.bounds(k, row, ki as i32, lo, up);
                        if !optimal() {
                            let which = if drop_lower { "lower" } else { "upper" };
                            let b = if drop_lower { lower[ki] } else { upper[ki] };
                            let ms = model_status_string(h.model_status(k));
                            let fmt = if row {
                                "HighsIis: IIS row %d (LP row %d): relaxing %s bound of %g yield IIS LP with status %s\n"
                            } else {
                                "HighsIis: IIS column %d (LP column %d): relaxing %s bound of %g yield IIS LP with status %s\n"
                            };
                            log_user!(h.log(), t, fmt, ki as i32, i, which, b, ms);
                            return false;
                        }
                        h.bounds(k, row, ki as i32, lower[ki], upper[ki]);
                    }
                }
            }
            true
        })();
        h.on(Op::DeleteHighs, k);
        result
    }

    /// HighsIis::compute: the deletion filter on `lp`, whose LP is
    /// infeasible
    fn compute(&mut self, h: &H, lp: &OwnLp) -> Status {
        let v = lp.view();
        let col_priority = h.strategy() & STRATEGY_COL_PRIORITY != 0;
        let row_priority = !col_priority;
        let mut num_rows = 0;
        for c in 0..v.num_col {
            let s = determine_bound_status(v.col_lower[c], v.col_upper[c], false);
            self.add_col(c as i32, s);
        }
        for r in 0..v.num_row {
            let s = determine_bound_status(v.row_lower[r], v.row_upper[r], true);
            self.add_row(r as i32, s);
            if s != BOUND_DROPPED {
                num_rows += 1;
            }
        }
        let k = h.on(Op::NewHighs, 0) as i32;
        h.call(Op::PassOptions, Args { i: 0, ..args(k) });
        h.set_bool(k, b"output_flag", false);
        let mut a = args(k);
        let off = RsStr::of(b"off");
        a.p[0] = &off as *const RsStr as *const c_void;
        a.s = RsStr::of(b"presolve");
        h.call(Op::SetOptionString, a);
        h.set_double(k, b"time_limit", cmax(h.opt(2) - self.info.sum_simplex_times, 0.0));
        h.on(Op::PropagateCallbacks, k);
        let arrays = lp.arrays(0);
        let mut a = args(k);
        a.p[0] = &arrays as *const LpArrays as *const c_void;
        h.call(Op::PassModelArrays, a);
        let cost = vec![0.0; v.num_col];
        let mut a = Args { i: 0, j: v.num_col as i32 - 1, ..args(k) };
        a.p[0] = ptr(&cost);
        h.call(Op::ChangeColsCost, a);
        let result = self.deletion_filter(h, k, &v, row_priority, num_rows);
        h.on(Op::DeleteHighs, k);
        result
    }

    fn solve(&mut self, h: &H, k: i32) -> Status {
        let time = -h.run_time(k);
        let iterations = -h.iterations(k);
        let status = h.status(Op::OptimizeModel, args(k));
        let time = time + h.run_time(k);
        let iterations = iterations + h.iterations(k);
        self.info.update(time, iterations);
        status
    }

    fn deletion_filter(&mut self, h: &H, k: i32, lp: &Lp, row_priority: bool, mut num_rows: i32) -> Status {
        if self.solve(h, k) != Status::Ok {
            // Infeasibility not established: the initial subset
            self.valid = true;
            self.strategy = h.strategy();
            self.status = if h.model_status(k) == MS_TIME_LIMIT { IIS_TIME_LIMIT } else { IIS_REDUCIBLE };
            return Status::Warning;
        }
        let mut iis_status = IIS_IRREDUCIBLE;
        let mut search_status = Status::Ok;
        self.clear_log_info();
        let mut iter = 0;
        log_user!(h.log(), LogType::Info, "\nRunning deletion filter to identify an IIS\n");
        self.report_iteration(h, iter, num_rows, true);
        for pass in 0..2 {
            let row = (row_priority && pass == 0) || (!row_priority && pass == 1);
            let num_index = if row { lp.num_row } else { lp.num_col };
            for x in 0..num_index {
                iter += 1;
                let force = row && x == num_index - 1;
                let x_status = if row { self.row_bound[x] } else { self.col_bound[x] };
                if x_status == BOUND_DROPPED || x_status == BOUND_FREE {
                    self.report_iteration(h, iter, num_rows, force);
                    continue;
                }
                let mut lower = if row { lp.row_lower[x] } else { lp.col_lower[x] };
                let mut upper = if row { lp.row_upper[x] } else { lp.col_upper[x] };
                for drop_lower in [true, false] {
                    let relax = if drop_lower { lower > -INF } else { upper < INF };
                    if !relax || iis_status == IIS_TIME_LIMIT {
                        continue;
                    }
                    let (lo, up) = if drop_lower { (-INF, upper) } else { (lower, INF) };
                    h.bounds(k, row, x as i32, lo, up);
                    self.solve(h, k);
                    // processBoundRelaxation
                    match h.model_status(k) {
                        MS_OPTIMAL => {
                            // Now feasible, so restore the bound, and if the
                            // lower bound must be kept, then drop any finite
                            // upper bound
                            h.bounds(k, row, x as i32, lower, upper);
                            if drop_lower && upper < INF {
                                upper = INF;
                                h.bounds(k, row, x as i32, lower, upper);
                            }
                        }
                        MS_INFEASIBLE => {
                            if drop_lower {
                                lower = -INF;
                            } else {
                                upper = INF;
                            }
                        }
                        ms => {
                            h.bounds(k, row, x as i32, lower, upper);
                            iis_status = if ms == MS_TIME_LIMIT { IIS_TIME_LIMIT } else { IIS_REDUCIBLE };
                            search_status = Status::Warning;
                        }
                    }
                }
                let s = determine_bound_status(lower, upper, row);
                if row {
                    self.row_bound[x] = s;
                    if s == BOUND_DROPPED {
                        num_rows -= 1;
                    }
                } else {
                    self.col_bound[x] = s;
                }
                self.report_iteration(h, iter, num_rows, force);
            }
            if pass == 0 && row {
                // Mark empty columns as dropped
                for c in 0..lp.num_col {
                    if lp.range(c).all(|el| self.row_bound[lp.index[el] as usize] == BOUND_DROPPED) {
                        self.col_bound[c] = BOUND_DROPPED;
                        h.bounds(k, false, c as i32, -INF, INF);
                    }
                }
            }
        }
        let keep = |index: &mut Vec<i32>, bound: &mut Vec<i32>| {
            let mut n = 0;
            for i in 0..bound.len() {
                if bound[i] != BOUND_DROPPED {
                    index[n] = index[i];
                    bound[n] = bound[i];
                    n += 1;
                }
            }
            index.truncate(n);
            bound.truncate(n);
        };
        keep(&mut self.col_index, &mut self.col_bound);
        keep(&mut self.row_index, &mut self.row_bound);
        self.valid = true;
        self.status = iis_status;
        self.strategy = h.strategy();
        search_status
    }
}

/// determineBoundStatus
fn determine_bound_status(lower: f64, upper: f64, is_row: bool) -> i32 {
    if lower <= -INF {
        if upper >= INF {
            // Free rows can be dropped, free columns only if empty
            if is_row { BOUND_DROPPED } else { BOUND_FREE }
        } else {
            BOUND_UPPER
        }
    } else if upper >= INF {
        BOUND_LOWER
    } else {
        BOUND_BOXED
    }
}

/// iisModelStatusToString
fn model_status_str(status: i32) -> &'static str {
    match status {
        IIS_FEASIBLE => "Feasible",
        IIS_UNKNOWN => "Unknown",
        IIS_TIME_LIMIT => "Time limit reached",
        IIS_REDUCIBLE => "Reducible",
        IIS_IRREDUCIBLE => "Irreducible",
        _ => "*****",
    }
}

/// getIisInterfaceReturn: restore the options and callbacks (`restore`:
/// the saved ones, else the current ones), check and report the IIS
fn iis_return(h: &H, iis: &mut Iis, status: Status, restore: Option<&[bool]>) -> Status {
    let active: Vec<bool> = match restore {
        Some(a) => {
            h.on(Op::RestoreOptions, 0);
            a.to_vec()
        }
        None => callback_active(h),
    };
    for (i, &a) in active.iter().enumerate() {
        if a {
            h.call(Op::StartCallback, Args { i: i as i32, ..args(0) });
        }
    }
    if status == Status::Error {
        iis.report_final(h);
        return status;
    }
    if iis.status >= IIS_TIME_LIMIT {
        h.call(Op::SetModelStatus, Args { i: MS_INFEASIBLE, ..args(0) });
    }
    let lp = h.lp();
    let has_is = !iis.col_index.is_empty() || !iis.row_index.is_empty();
    if has_is {
        iis.set_lp(h, &lp);
        if !iis.lp_data_ok(&lp) {
            iis.report_final(h);
            return Status::Error;
        }
        if !iis.lp_ok(h) {
            // Infeasibility not proved: the candidate set is invalid
            iis.valid = false;
            if iis.status != IIS_TIME_LIMIT {
                iis.status = IIS_UNKNOWN;
            }
            iis.report_final(h);
            return Status::Warning;
        }
    }
    iis.set_status(&lp);
    if !iis.index_status_ok(&lp) {
        iis.report_final(h);
        return Status::Error;
    }
    iis.report_final(h);
    status
}

fn callback_active(h: &H) -> Vec<bool> {
    (0..=CALLBACK_MAX).map(|i| h.call(Op::CallbackActive, Args { i, ..args(0) }) != 0.0).collect()
}

/// getIisInterface
fn get_iis(h: &H, iis: &mut Iis) -> Status {
    let ms = h.model_status(0);
    if ms == MS_OPTIMAL || ms == MS_UNBOUNDED {
        log_user!(h.log(), LogType::Info, "Calling Highs::getIis for a model that is known to be feasible\n");
        iis.clear(h);
        iis.valid = true;
        iis.status = IIS_FEASIBLE;
        return iis_return(h, iis, Status::Ok, None);
    }
    if iis.valid {
        return iis_return(h, iis, Status::Ok, None);
    }
    iis.clear(h);
    let lp = h.lp();
    if iis.trivial(h, &lp) {
        return iis_return(h, iis, Status::Ok, None);
    }
    if lp.num_row == 0 {
        // Only inconsistent column bounds could make it infeasible
        iis.valid = true;
        return iis_return(h, iis, Status::Ok, None);
    }
    if iis.row_value_bounds(h, &lp) {
        return iis_return(h, iis, Status::Ok, None);
    }
    if h.strategy() == STRATEGY_LIGHT {
        return iis_return(h, iis, Status::Ok, None);
    }
    iis.clear(h);
    h.on(Op::SaveOptions, 0);
    // Disable the callbacks but logging and simplex interrupt
    let original_active = callback_active(h);
    for (i, &a) in original_active.iter().enumerate() {
        let i = i as i32;
        if i != CALLBACK_LOGGING && i != CALLBACK_SIMPLEX_INTERRUPT && a {
            h.call(Op::StopCallback, Args { i, ..args(0) });
        }
    }
    let restore = Some(original_active.as_slice());
    h.on(Op::ZeroAllClocks, 0);
    h.set_double(0, b"time_limit", h.opt(2));
    h.set_int(0, b"simplex_iteration_limit", IINF);
    h.set_double(0, b"objective_bound", INF);
    h.set_bool(0, b"allow_unbounded_or_infeasible", false);
    let ms = h.model_status(0);
    if matches!(ms, MS_NOTSET | MS_ITERATION_LIMIT | MS_OBJECTIVE_BOUND | MS_INTERRUPT | MS_TIME_LIMIT
                    | MS_UNBOUNDED_OR_INFEASIBLE) {
        log_user!(h.log(), LogType::Info,
                  "Model status is %s. Resolving to establish infeasibility before computing IIS\n\n",
                  model_status_string(ms));
        iis.solve(h, 0);
        log_user!(h.log(), LogType::Info, "\n");
    }
    let ms = h.model_status(0);
    if ms == MS_OPTIMAL || ms == MS_UNBOUNDED {
        log_user!(h.log(), LogType::Info, "Model became feasible\n");
        iis.found(h, IIS_FEASIBLE);
        return iis_return(h, iis, Status::Ok, restore);
    } else if ms == MS_TIME_LIMIT {
        log_user!(h.log(), LogType::Error, "Time limit reached prior to establishing infeasibility\n");
        iis.status = IIS_TIME_LIMIT;
        iis.strategy = h.strategy();
        return iis_return(h, iis, Status::Error, restore);
    } else if ms != MS_INFEASIBLE {
        log_user!(h.log(), LogType::Error, "Can not compute IIS for a model with status %s\n",
                  model_status_string(ms));
        iis.strategy = h.strategy();
        return iis_return(h, iis, Status::Error, restore);
    }
    // An infeasible subset of rows from the elasticity filter (the dual
    // ray option is disabled in HiGHS)
    let mut status = Status::Ok;
    if h.strategy() & STRATEGY_FROM_LP != 0 {
        status = elasticity_filter(h, iis, -1.0, -1.0, 1.0, None, None, None, true);
    }
    if h.strategy() & STRATEGY_IRREDUCIBLE == 0 {
        return iis_return(h, iis, status, restore);
    }
    // Without a valid IS, use all the rows
    if !iis.valid || iis.status != IIS_REDUCIBLE {
        iis.valid = true;
        iis.status = IIS_REDUCIBLE;
        iis.row_index.extend(0..h.lp().num_row as i32);
    }
    h.on(Op::EnsureColwise, 0);
    let status = iis.deduce(h);
    iis_return(h, iis, status, restore)
}

/// elasticityFilter (Highs.cpp's feasibilityRelaxation and the
/// elasticity filter of getIisInterface when `get_iis`)
#[allow(clippy::too_many_arguments)]
fn elasticity_filter(h: &H, iis: &mut Iis, global_lower_penalty: f64, global_upper_penalty: f64,
                     global_rhs_penalty: f64, local_lower_penalty: Option<&[f64]>,
                     local_upper_penalty: Option<&[f64]>, local_rhs_penalty: Option<&[f64]>, get_iis: bool)
                     -> Status {
    let lp = h.lp();
    let original_status = h.model_status(0);
    let original_num_col = lp.num_col;
    let original_num_row = lp.num_row;
    // The original model name and column data, restored on return
    let mut original = OwnLp::new();
    // SAFETY: the incumbent's col_cost_, read before any change
    let cost = unsafe {
        let mut c = std::mem::MaybeUninit::<CLp>::uninit();
        let (mut a, mut b) = (0, 0);
        (h.host.lp)(h.host.ctx, c.as_mut_ptr(), &mut a, &mut b);
        let c = c.assume_init();
        (c.col_cost.get().to_vec(), c.integrality.get().to_vec())
    };
    original.col_cost = cost.0;
    let original_integrality = cost.1;
    original.col_lower = lp.col_lower.to_vec();
    original.col_upper = lp.col_upper.to_vec();
    let original_name = h.name(true, usize::MAX);
    let is_mip = original_integrality.iter().any(|&t| t != VAR_CONTINUOUS);
    h.on(Op::InvalidateSolverData, 0);
    let elastic_name = format!("{original_name}_elastic");
    h.call(Op::PassModelName, Args { s: RsStr::of(elastic_name.as_bytes()), ..args(0) });
    let zero = vec![0.0; original_num_col];
    let mut a = Args { i: 0, j: original_num_col as i32 - 1, ..args(0) };
    a.p[0] = ptr(&zero);
    h.call(Op::ChangeColsCost, a);
    if get_iis && is_mip && h.strategy() & STRATEGY_RELAXATION != 0 {
        let continuous = vec![VAR_CONTINUOUS; original_num_col];
        let mut a = Args { i: 0, j: original_num_col as i32 - 1, ..args(0) };
        a.p[0] = ptr(&continuous);
        h.call(Op::ChangeColsIntegrality, a);
    }
    let has_elastic_lower = local_lower_penalty.is_some() || global_lower_penalty >= 0.0;
    let has_elastic_upper = local_upper_penalty.is_some() || global_upper_penalty >= 0.0;
    let has_elastic_columns = has_elastic_lower || has_elastic_upper;
    let has_elastic_rows = local_rhs_penalty.is_some() || global_rhs_penalty >= 0.0;
    let mut col_of_ecol: Vec<i32> = Vec::new();
    let mut row_of_ecol: Vec<i32> = Vec::new();
    let mut bound_of_row_of_ecol_is_lower: Vec<bool> = Vec::new();
    let col_ecol_offset = original_num_col;
    if has_elastic_columns {
        let lp = h.lp();
        let mut col_lower = Vec::with_capacity(lp.num_col);
        let mut col_upper = Vec::with_capacity(lp.num_col);
        let (mut erow_lower, mut erow_upper) = (Vec::new(), Vec::new());
        let (mut erow_start, mut erow_index, mut erow_value) = (vec![0i32], Vec::new(), Vec::new());
        let (mut erow_name, mut ecol_name, mut ecol_cost) = (Vec::new(), Vec::new(), Vec::new());
        let mut evar_ix = lp.num_col as i32;
        for c in 0..lp.num_col {
            let (lower, upper) = (lp.col_lower[c], lp.col_upper[c]);
            col_lower.push(lower);
            col_upper.push(upper);
            if lower <= -INF && upper >= INF {
                continue;
            }
            let lower_penalty = local_lower_penalty.map_or(global_lower_penalty, |p| p[c]);
            if lower_penalty < 0.0 && upper >= INF {
                continue;
            }
            let upper_penalty = local_upper_penalty.map_or(global_upper_penalty, |p| p[c]);
            if lower <= -INF && upper_penalty < 0.0 {
                continue;
            }
            erow_lower.push(lower);
            erow_upper.push(upper);
            let name = if lp.has_col_names { h.name(true, c) } else { String::new() };
            if lp.has_col_names {
                erow_name.push(format!("row_{c}_{name}_erow"));
            }
            erow_index.push(c as i32);
            erow_value.push(1.0);
            if lower > -INF && lower_penalty >= 0.0 {
                col_of_ecol.push(c as i32);
                if lp.has_col_names {
                    ecol_name.push(format!("col_{c}_{name}_lower"));
                }
                col_lower[c] = -INF;
                erow_index.push(evar_ix);
                erow_value.push(1.0);
                ecol_cost.push(lower_penalty);
                evar_ix += 1;
            }
            if upper < INF && upper_penalty >= 0.0 {
                col_of_ecol.push(c as i32);
                if lp.has_col_names {
                    ecol_name.push(format!("col_{c}_{name}_upper"));
                }
                col_upper[c] = INF;
                erow_index.push(evar_ix);
                erow_value.push(-1.0);
                ecol_cost.push(upper_penalty);
                evar_ix += 1;
            }
            erow_start.push(erow_index.len() as i32);
        }
        let has_col_names = lp.has_col_names;
        let num_col = lp.num_col as i32;
        let num_new_col = col_of_ecol.len();
        let num_new_row = erow_start.len() - 1;
        let mut a = Args { i: 0, j: num_col - 1, ..args(0) };
        a.p[0] = ptr(&col_lower);
        a.p[1] = ptr(&col_upper);
        h.call(Op::ChangeColsBounds, a);
        let ecol_lower = vec![0.0; num_new_col];
        let ecol_upper = vec![INF; num_new_col];
        let mut a = Args { i: num_new_col as i32, j: 0, ..args(0) };
        a.p[..3].copy_from_slice(&[ptr(&ecol_cost), ptr(&ecol_lower), ptr(&ecol_upper)]);
        h.call(Op::AddCols, a);
        let mut a = Args { i: num_new_row as i32, j: erow_start[num_new_row], ..args(0) };
        a.p[..5].copy_from_slice(&[ptr(&erow_lower), ptr(&erow_upper), ptr(&erow_start), ptr(&erow_index),
                                   ptr(&erow_value)]);
        h.call(Op::AddRows, a);
        if has_col_names {
            for (k, name) in ecol_name.iter().enumerate() {
                h.call(Op::PassColName,
                       Args { i: (col_ecol_offset + k) as i32, s: RsStr::of(name.as_bytes()), ..args(0) });
            }
            for (k, name) in erow_name.iter().enumerate() {
                h.call(Op::PassRowName,
                       Args { i: (original_num_row + k) as i32, s: RsStr::of(name.as_bytes()), ..args(0) });
            }
        }
    }
    let row_ecol_offset = h.lp().num_col;
    if has_elastic_rows {
        let lp = h.lp();
        let (mut ecol_name, mut ecol_cost) = (Vec::new(), Vec::new());
        let (mut ecol_start, mut ecol_index, mut ecol_value) = (vec![0i32], Vec::new(), Vec::new());
        for r in 0..original_num_row {
            let penalty = local_rhs_penalty.map_or(global_rhs_penalty, |p| p[r]);
            if penalty < 0.0 {
                continue;
            }
            let (lower, upper) = (lp.row_lower[r], lp.row_upper[r]);
            let name = if lp.has_row_names { h.name(false, r) } else { String::new() };
            for (is_lower, finite, sign) in [(true, lower > -INF, 1.0), (false, upper < INF, -1.0)] {
                if !finite {
                    continue;
                }
                row_of_ecol.push(r as i32);
                if lp.has_row_names {
                    ecol_name.push(format!("row_{r}_{name}_{}", if is_lower { "lower" } else { "upper" }));
                }
                bound_of_row_of_ecol_is_lower.push(is_lower);
                ecol_index.push(r as i32);
                ecol_value.push(sign);
                ecol_start.push(ecol_index.len() as i32);
                ecol_cost.push(penalty);
            }
        }
        let has_row_names = lp.has_row_names;
        let num_new_col = ecol_start.len() - 1;
        let ecol_lower = vec![0.0; num_new_col];
        let ecol_upper = vec![INF; num_new_col];
        let mut a = Args { i: num_new_col as i32, j: ecol_start[num_new_col], ..args(0) };
        a.p.copy_from_slice(&[ptr(&ecol_cost), ptr(&ecol_lower), ptr(&ecol_upper), ptr(&ecol_start),
                              ptr(&ecol_index), ptr(&ecol_value)]);
        h.call(Op::AddCols, a);
        if has_row_names {
            for (k, name) in ecol_name.iter().enumerate() {
                h.call(Op::PassColName,
                       Args { i: (row_ecol_offset + k) as i32, s: RsStr::of(name.as_bytes()), ..args(0) });
            }
        }
    }
    if get_iis {
        log_user!(h.log(), LogType::Info, "Running elasticity filter to identify an infeasible subset of rows\n");
        iis.report_iteration(h, 0, 0, true);
    }
    let original = ElasticOriginal {
        name: original_name,
        status: original_status,
        num_col: original_num_col,
        num_row: original_num_row,
        lp: original,
        integrality: original_integrality,
    };
    let solve = |iis: &mut Iis| {
        let output_flag = h.opt(3) != 0.0;
        h.call(Op::SetOutputFlag, Args { i: 0, ..args(0) });
        let status = iis.solve(h, 0);
        h.call(Op::SetOutputFlag, Args { i: output_flag as i32, ..args(0) });
        status
    };
    let failed = |iis: &mut Iis| {
        if h.model_status(0) == MS_TIME_LIMIT {
            iis.status = IIS_TIME_LIMIT;
            log_user!(h.log(), LogType::Error, "Elasticity filter failed because time limit was reached\n");
        } else {
            iis.status = IIS_UNKNOWN;
            log_user!(h.log(), LogType::Error,
                      "Elasticity filter failed because it encountered an unknown model status\n");
        }
        iis.valid = false;
        elastic_return(h, Status::Error, &original)
    };
    if solve(iis) != Status::Ok {
        if get_iis {
            return failed(iis);
        }
        // The same without the messages
        iis.status = if h.model_status(0) == MS_TIME_LIMIT { IIS_TIME_LIMIT } else { IIS_UNKNOWN };
        iis.valid = false;
        return elastic_return(h, Status::Error, &original);
    }
    iis.valid = true;
    iis.status = if h.on(Op::ObjectiveValue, 0) > 0.0 { IIS_REDUCIBLE } else { IIS_FEASIBLE };
    if !get_iis {
        return elastic_return(h, Status::Ok, &original);
    }
    // Getting an IIS: no elastic columns. Fix the positive e-variables
    // and re-solve until the e-LP is infeasible
    let tol = h.tol();
    let mut loop_k = 0;
    let mut row_set = std::collections::HashSet::new();
    loop {
        loop_k += 1;
        let mut num_fixed = 0;
        let col_value = h.col_value();
        let fix: Vec<usize> = (0..row_of_ecol.len()).filter(|&e| col_value[row_ecol_offset + e] > tol).collect();
        for e in fix {
            h.bounds(0, false, (row_ecol_offset + e) as i32, 0.0, 0.0);
            num_fixed += 1;
            row_set.insert(row_of_ecol[e]);
        }
        if num_fixed == 0 {
            // No positive e-variable: feasible
            iis.status = IIS_FEASIBLE;
            iis.report_iteration(h, loop_k, row_set.len() as i32, true);
            break;
        }
        if solve(iis) != Status::Ok {
            return failed(iis);
        }
        let terminate = h.model_status(0) == MS_INFEASIBLE;
        iis.report_iteration(h, loop_k, row_set.len() as i32, terminate);
        if terminate {
            break;
        }
    }
    let lp = h.lp();
    let enforced = |e: usize| lp.col_upper[row_ecol_offset + e] == 0.0;
    for (e, &r) in row_of_ecol.iter().enumerate() {
        if enforced(e) && iis.row_index.last() != Some(&r) {
            iis.row_index.push(r);
        }
    }
    let num_iis_row = iis.row_index.len();
    if iis.status == IIS_FEASIBLE {
        log_user!(h.log(), LogType::Info, "Elasticity filter failed to reproduce infeasibility\n");
    } else {
        log_user!(h.log(), LogType::Info,
                  "Elasticity filter after %d passes found an infeasible subset of %d rows\n",
                  loop_k, row_set.len() as i32);
    }
    iis.valid = true;
    iis.strategy = h.strategy();
    if iis.status == IIS_FEASIBLE {
        return elastic_return(h, Status::Ok, &original);
    }
    // The columns with nonzeros in the infeasible rows, and the row bounds
    let mut in_row_index = vec![-1i32; original_num_row];
    for (k, &r) in iis.row_index.iter().enumerate() {
        in_row_index[r as usize] = k as i32;
    }
    let mut nonzero = vec![false; original_num_col];
    if lp.colwise() {
        for (c, nz) in nonzero.iter_mut().enumerate() {
            *nz = lp.range(c).any(|el| in_row_index[lp.index[el] as usize] >= 0);
        }
    } else {
        for &r in &iis.row_index {
            for el in lp.range(r as usize) {
                nonzero[lp.index[el] as usize] = true;
            }
        }
    }
    for c in (0..original_num_col).filter(|&c| nonzero[c]) {
        let b = if lp.col_lower[c] > -INF {
            if lp.col_upper[c] < INF { BOUND_BOXED } else { BOUND_LOWER }
        } else if lp.col_upper[c] < INF {
            BOUND_UPPER
        } else {
            BOUND_FREE
        };
        iis.add_col(c as i32, b);
    }
    iis.row_bound = vec![-1; num_iis_row];
    for (e, &r) in row_of_ecol.iter().enumerate() {
        if enforced(e) {
            let k = in_row_index[r as usize] as usize;
            iis.row_bound[k] = if iis.row_bound[k] == -1 {
                if bound_of_row_of_ecol_is_lower[e] { BOUND_LOWER } else { BOUND_UPPER }
            } else {
                BOUND_BOXED
            };
        }
    }
    elastic_return(h, Status::Ok, &original)
}

struct ElasticOriginal {
    name: String,
    status: i32,
    num_col: usize,
    num_row: usize,
    lp: OwnLp,
    integrality: Vec<u8>,
}

/// elasticityFilterReturn: restore the incumbent's model
fn elastic_return(h: &H, status: Status, o: &ElasticOriginal) -> Status {
    let objective = h.on(Op::ObjectiveValue, 0);
    let lp = h.lp();
    h.call(Op::DeleteRows, Args { i: o.num_row as i32, j: lp.num_row as i32 - 1, ..args(0) });
    let lp = h.lp();
    h.call(Op::DeleteCols, Args { i: o.num_col as i32, j: lp.num_col as i32 - 1, ..args(0) });
    h.on(Op::BasisInvalid, 0);
    let last = o.num_col as i32 - 1;
    let mut a = Args { i: 0, j: last, ..args(0) };
    a.p[0] = ptr(&o.lp.col_cost);
    h.call(Op::ChangeColsCost, a);
    let mut a = Args { i: 0, j: last, ..args(0) };
    a.p[0] = ptr(&o.lp.col_lower);
    a.p[1] = ptr(&o.lp.col_upper);
    h.call(Op::ChangeColsBounds, a);
    if h.strategy() & STRATEGY_RELAXATION != 0 && !o.integrality.is_empty() {
        let mut a = Args { i: 0, j: last, ..args(0) };
        a.p[0] = ptr(&o.integrality);
        h.call(Op::ChangeColsIntegrality, a);
    }
    h.call(Op::PassModelName, Args { s: RsStr::of(o.name.as_bytes()), ..args(0) });
    h.on(Op::InvalidateSolverData, 0);
    if status == Status::Ok {
        h.call(Op::ElasticSolution, Args { x: objective, ..args(0) });
    }
    h.call(Op::SetModelStatus, Args { i: o.status, ..args(0) });
    status
}

/// Highs::getIisInterface
///
/// # Safety
/// `host` must be valid, its functions callable with its ctx
#[no_mangle]
pub unsafe extern "C" fn highs_rs_get_iis(host: *const IisHost) -> i32 {
    let h = H::new(&*host);
    let mut iis = Iis::load(&h);
    let status = get_iis(&h, &mut iis);
    iis.store(&h);
    status as i32
}

/// Highs::elasticityFilter (get_iis false: feasibilityRelaxation); the
/// local penalties are null or of the incumbent's numbers of columns
/// (lower, upper) and rows (rhs)
///
/// # Safety
/// `host` must be valid, the penalties null or of those lengths
#[no_mangle]
pub unsafe extern "C" fn highs_rs_elasticity_filter(host: *const IisHost, global_lower_penalty: f64,
                                                    global_upper_penalty: f64, global_rhs_penalty: f64,
                                                    local_lower_penalty: *const f64,
                                                    local_upper_penalty: *const f64,
                                                    local_rhs_penalty: *const f64, get_iis: bool) -> i32 {
    let h = H::new(&*host);
    let lp = h.lp();
    let slice = |p: *const f64, n: usize| {
        if p.is_null() { None } else { Some(std::slice::from_raw_parts(p, n)) }
    };
    let (ll, lu, lr) = (slice(local_lower_penalty, lp.num_col), slice(local_upper_penalty, lp.num_col),
                        slice(local_rhs_penalty, lp.num_row));
    let mut iis = Iis::load(&h);
    let status = elasticity_filter(&h, &mut iis, global_lower_penalty, global_upper_penalty, global_rhs_penalty,
                                   ll, lu, lr, get_iis);
    iis.store(&h);
    status as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bound_status_and_info() {
        assert_eq!(determine_bound_status(-INF, INF, true), BOUND_DROPPED);
        assert_eq!(determine_bound_status(-INF, INF, false), BOUND_FREE);
        assert_eq!(determine_bound_status(0.0, INF, false), BOUND_LOWER);
        assert_eq!(determine_bound_status(-INF, 1.0, true), BOUND_UPPER);
        assert_eq!(determine_bound_status(0.0, 1.0, true), BOUND_BOXED);
        let mut i = IisInfo { num_lp_solved: 0, sum_simplex_iteration_counts: 0, min_simplex_iteration_count: 0,
                              max_simplex_iteration_count: 0, sum_simplex_times: 0.0, min_simplex_time: 0.0,
                              max_simplex_time: 0.0, iis_last_disptime: 0.0, iis_num_disp_lines: 0 };
        i.clear();
        i.update(0.5, 7);
        i.update(0.25, 3);
        assert_eq!((i.num_lp_solved, i.sum_simplex_iteration_counts, i.min_simplex_iteration_count,
                    i.max_simplex_iteration_count), (2, 10, 3, 7));
        assert_eq!((i.min_simplex_time, i.max_simplex_time), (0.25, 0.5));
        assert_eq!(std::mem::size_of::<IisInfo>(), 56);
    }
}
