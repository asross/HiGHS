//! A Rust LP solver without a C++ `Highs` object ([`LpHandle`]): the LP
//! solver of the MIP's LP relaxation (and of its IPX race and IPM basis).
//! It owns what the `Highs` object held for an LP: the option values
//! ([`Opts`]), the model LP (an [`Lp`]), the simplex engine ([`LpSolver`],
//! with the run's solution, basis, info, model status and LP presolve
//! data), the run data and the presolve status, the four clocks of a run
//! ([`Timer`]) and what the C++ HEkk shell kept (the LP's name, the
//! analysis' report data, the simplex NLA's LP, the factor's log). The
//! Highs methods it needs are here, mirroring Highs.cpp: passModel,
//! clearSolver, the column bound and cost changes, adding and deleting
//! rows, setBasis, optimizeLp and run, putIterate / getIterate, the dual
//! ray and the basis inverse rows.
//!
//! The run is run.rs's calledOptimizeModel (and drivers.rs' passModel and
//! setBasis) on a [`CHighs`] whose `op` makes every step on this data in
//! Rust ([`handle_op`]); the simplex shell's steps are [`LpHandle::shell`].
//! The only C++ left is the MIP's profiling clocks (a C++ HighsProfiling,
//! reached through a function registered by C++). Logging: the internal
//! LP solvers are silent (output_flag false); with output on a handle
//! logs to the console only (no log file or callbacks: a `Highs` made by
//! the MIP has neither) and does not print the version header or model
//! statistics.

use super::basis::{basis_consistent, refine_basis};
use super::ffi::{CLp, RsMut};
use super::interface::{self, BasisG, IfaceHost, IfaceOptions, Invalidate};
use super::ipx_glue::CIpxHost;
use super::lp::Lp;
use super::lp_presolve::PresolveOptions;
use super::lp_run::{Basis, LpRun, Solution, UnconTemplate};
use super::lp_utils::{assess_lp, lp_dimensions_ok, IndexCollection};
use super::opts::{OptValue, Opts};
use super::run::{CHighs, Facts, Op, Run, RunData, MS_NOTSET, MS_UNKNOWN};
use super::solution::Info;
use super::solve::CSolve;
use super::{Log, LogType, Status, INF};
use crate::simplex::app::CSimplexApp;
use crate::simplex::dual_row::WorkPair;
use crate::simplex::hekk::Host;
use crate::simplex::lp_solver::{LpSolver, LpsEnv};
use crate::simplex::report::SimplexReport;
use crate::util::fma::ClangFma;
use std::ffi::{c_char, c_void, CStr};
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering};

// HighsBasisStatus
const BASIC: u8 = 1;
const NONBASIC: u8 = 4;
/// kSimplexStrategyPrimal
const STRATEGY_PRIMAL: i32 = 4;
/// kSimplexStrategyChoose
const STRATEGY_CHOOSE: i32 = 0;
const BASIS_VALIDITY_INVALID: i32 = 0;
const SOLUTION_STATUS_NONE: i32 = 0;
const SOLUTION_STATUS_FEASIBLE: i32 = 2;

// ---- The clocks of a run (HighsTimer's run, solve, presolve and
// postsolve clocks)

/// HighsTimer::initial_clock_start
const INITIAL_CLOCK_START: f64 = 1.0;

fn wall_time() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs_f64()).unwrap_or(0.0)
}

/// HighsTimer's four clocks of a run: 0 run, 1 solve, 2 presolve, 3
/// postsolve
#[derive(Clone, Debug)]
pub struct Timer {
    start: [f64; 4],
    time: [f64; 4],
}

impl Default for Timer {
    fn default() -> Self {
        Timer { start: [INITIAL_CLOCK_START; 4], time: [0.0; 4] }
    }
}

impl Timer {
    pub fn start(&mut self, c: usize) {
        self.start[c] = -wall_time();
    }
    pub fn stop(&mut self, c: usize) {
        let w = wall_time();
        self.time[c] += w + self.start[c];
        self.start[c] = w;
    }
    pub fn read(&self, c: usize) -> f64 {
        if self.start[c] < 0.0 {
            self.time[c] + wall_time() + self.start[c]
        } else {
            self.time[c]
        }
    }
    pub fn running(&self, c: usize) -> bool {
        self.start[c] < 0.0
    }
}

// ---- Logging

/// The log flags a handle's `Log` points to (its options', or the
/// factor's copy of them)
#[repr(C)]
struct LogHead {
    output_flag: *const bool,
    log_to_console: *const bool,
    log_dev_level: *const i32,
    /// The log file, if any (null: none)
    log_stream: *const *mut crate::io::log::File,
}

impl LogHead {
    fn none() -> LogHead {
        LogHead {
            output_flag: std::ptr::null(),
            log_to_console: std::ptr::null(),
            log_dev_level: std::ptr::null(),
            log_stream: std::ptr::null(),
        }
    }
    fn of(f: &FactorLog) -> LogHead {
        LogHead {
            output_flag: &f.output_flag,
            log_to_console: &f.log_to_console,
            log_dev_level: &f.log_dev_level,
            log_stream: &f.log_stream,
        }
    }
}

/// The factor's log flags: a copy of the options' made when the simplex
/// NLA is set up (HFactor::setupGeneral), with the options' log file (as
/// a C++ HighsLogOptions without callbacks)
#[derive(Clone, Copy)]
struct FactorLog {
    output_flag: bool,
    log_to_console: bool,
    log_dev_level: i32,
    log_stream: *mut crate::io::log::File,
}

impl Default for FactorLog {
    fn default() -> Self {
        FactorLog { output_flag: false, log_to_console: true, log_dev_level: 0, log_stream: std::ptr::null_mut() }
    }
}

const LOG_DETAILED: i32 = 2;
const LOG_VERBOSE: i32 = 3;
const LOG_WARNING: i32 = 4;
const LOG_ERROR: i32 = 5;

/// highsLogUser / highsLogDev without callbacks (io/log.rs): to the log
/// file, if any, and the console
unsafe extern "C" fn handle_log(opts: *const c_void, dev: i32, t: i32, msg: *const u8, len: usize) {
    let o = &*(opts as *const LogHead);
    let dev = dev != 0;
    let stream = if o.log_stream.is_null() { std::ptr::null_mut() } else { *o.log_stream };
    if !*o.output_flag || (!*o.log_to_console && stream.is_null()) {
        return;
    }
    let level = *o.log_dev_level;
    if dev && (level == 0 || (t == LOG_DETAILED && level < LOG_DETAILED) || (t == LOG_VERBOSE && level < LOG_VERBOSE)) {
        return;
    }
    let mut m = if len == 0 { &[][..] } else { std::slice::from_raw_parts(msg, len) };
    if let Some(n) = m.iter().position(|&c| c == 0) {
        m = &m[..n];
    }
    let prefix: &[u8] = match (dev, t) {
        (false, LOG_WARNING) => b"WARNING: ",
        (false, LOG_ERROR) => b"ERROR:   ",
        _ => b"",
    };
    crate::io::log::to_log_file(stream, prefix, m);
    if *o.log_to_console {
        crate::io::log::c_stdout(prefix);
        crate::io::log::c_stdout_flush(m);
    }
}

fn log_of(head: &LogHead) -> Log {
    Log { opts: head as *const LogHead as *const c_void, log: Some(handle_log) }
}

// ---- The C++ profiling clocks

/// The C++ function making the simplex solve's profiling steps on a
/// HighsProfiling (HEkkRust.cpp: rsSimplexProfiling): (profiling, code 1
/// start / 2 stop / 3 PDLP start / 4 PDLP stop, simplex_strategy, arg)
type ProfilingFn = unsafe extern "C" fn(*mut c_void, i32, i32, i64);
static PROFILING_FN: AtomicUsize = AtomicUsize::new(0);

/// Registers the C++ profiling function (once, by HighsLpRelaxation)
#[no_mangle]
pub extern "C" fn highs_rs_lph_register(f: ProfilingFn) {
    PROFILING_FN.store(f as usize, Ordering::Relaxed);
}

fn profiling_step(profiling: *mut c_void, code: i32, simplex_strategy: i32, arg: i64) {
    let f = PROFILING_FN.load(Ordering::Relaxed);
    if profiling.is_null() || f == 0 {
        return;
    }
    // SAFETY: stored from a ProfilingFn by highs_rs_lph_register; the
    // profiling object outlives the handle's use of it
    unsafe { std::mem::transmute::<usize, ProfilingFn>(f)(profiling, code, simplex_strategy, arg) }
}

// ---- Interrupts (the IPX race)

/// When a solve is interrupted, as the callbacks of the race between the
/// dual simplex and IPX did: with `simplex`, the simplex when `flag` is 2
/// (IPX won); with `ipm`, IPX when it is not zero (the race is decided)
#[derive(Clone, Copy)]
pub struct Interrupt {
    pub flag: *const AtomicI32,
    pub simplex: bool,
    pub ipm: bool,
}

// SAFETY: the flag is an atomic that outlives the solve
unsafe impl Send for Interrupt {}

// ---- The engine of a C++ Highs object

/// What a handle that is a C++ Highs object's simplex engine (and its LP
/// runs' solver) calls in C++: mirrored by RsLphHost in
/// highs/lp_data/HighsLpHandle.h. The handle logs through the Highs
/// object's log options (file and callbacks), reads its run clock and
/// calls its user interrupt callbacks.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CHost {
    /// The Highs object
    pub ctx: *mut c_void,
    /// Its options_.log_options (a C++ HighsLogOptions)
    pub log_options: *const c_void,
    /// timer_.read()
    pub timer_read: unsafe extern "C" fn(*mut c_void) -> f64,
    /// The simplex interrupt callback: with iteration_count < 0 whether it
    /// is active, else whether the user interrupts (HEkk's bailout, with
    /// its "User interrupt" dev log)
    pub simplex_interrupt: unsafe extern "C" fn(*mut c_void, i32) -> bool,
    /// The IPM interrupt callback (ipx::LpSolver's user interrupt hook)
    pub ipm_interrupt: unsafe extern "C" fn(*mut c_void, crate::ipx::Int) -> crate::ipx::Int,
}

/// HighsSimplexStats (lp_data/HStruct.h)
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct SimplexStats {
    pub valid: bool,
    pub iteration_count: i32,
    pub num_invert: i32,
    pub last_invert_num_el: i32,
    pub last_factored_basis_num_el: i32,
    pub col_aq_density: f64,
    pub row_ep_density: f64,
    pub row_ap_density: f64,
    pub row_dse_density: f64,
}

const _: () = assert!(std::mem::size_of::<SimplexStats>() == 56);

impl SimplexStats {
    /// HighsSimplexStats::initialise
    fn initialise(&mut self, iteration_count: i32) {
        *self = SimplexStats { iteration_count: -iteration_count, ..SimplexStats::default() };
    }
}

/// The simplex NLA's LP when it is a C++ LP (setNlaPointersForLpAndScale
/// of a Highs object's model): its dimensions and the scale factors that
/// apply, valid until the C++ LP changes
#[derive(Clone, Copy)]
struct NlaLp {
    num_col: i32,
    num_row: i32,
    has_scale: bool,
    col_scale: RsMut<f64>,
    row_scale: RsMut<f64>,
}

// ---- What HEkk's C++ shell kept

struct Shell {
    lp_name: Vec<u8>,
    /// The factor's log of solveLpSimplex (accommodateAlienBasis)
    app_factor_log: FactorLog,
    app_factor_head: LogHead,
    report: SimplexReport,
    /// simplex_stats_ (the simplex counts num_invert)
    stats: SimplexStats,
    factor_log: FactorLog,
    factor_head: LogHead,
    /// The simplex NLA's LP is the model (setNlaPointersForLpAndScale(model))
    /// with its scale factors if `nla_model_scale`
    nla_model: bool,
    nla_model_scale: bool,
    /// The simplex NLA's LP is a C++ LP (a Highs object's model)
    nla_cpp: Option<NlaLp>,
    /// A Highs object's engine: what the API returns (hot_start_ and
    /// primal_phase1_dual_), kept from the last solve that set them
    hot_start: Option<crate::simplex::basis_records::HotStart>,
    primal_phase1_dual: Vec<f64>,
}

fn zero_report() -> SimplexReport {
    SimplexReport {
        simplex_strategy: 0,
        solve_phase: 0,
        simplex_iteration_count: 0,
        pivotal_row_index: 0,
        entering_variable: 0,
        rebuild_reason: 0,
        num_primal_infeasibility: 0,
        num_dual_infeasibility: 0,
        num_iteration_report_since_last_header: 0,
        num_invert_report_since_last_header: 0,
        objective_value: 0.0,
        sum_primal_infeasibility: 0.0,
        sum_dual_infeasibility: 0.0,
        highs_run_time: 0.0,
        last_user_log_time: 0.0,
        delta_user_log_time: 0.0,
        col_aq_density: 0.0,
        row_ep_density: 0.0,
        row_ap_density: 0.0,
        row_dse_density: 0.0,
        timeless_log: false,
    }
}

/// HighsInfo() (the records' defaults)
fn info_default() -> Info {
    // SAFETY: Info is plain data
    let mut i: Info = unsafe { std::mem::zeroed() };
    i.num_primal_infeasibilities = -1;
    i.num_dual_infeasibilities = -1;
    i.num_semi_infeasibilities = -1;
    i.num_relative_primal_infeasibilities = -1;
    i.num_relative_dual_infeasibilities = -1;
    i.num_primal_residual_errors = -1;
    i.num_dual_residual_errors = -1;
    i.num_relative_primal_residual_errors = -1;
    i.num_relative_dual_residual_errors = -1;
    i.num_complementarity_violations = -1;
    i
}

/// The records of handleInfCost (HighsLpMods' inf cost vectors)
#[derive(Default)]
struct InfCostMods {
    index: Vec<i32>,
    cost: Vec<f64>,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

/// The LP solver: see the module comment
pub struct LpHandle {
    pub opts: Opts,
    saved_opts: Option<Box<Opts>>,
    /// The model LP (model_.lp_)
    pub model: Lp,
    /// The simplex engine and the run's data
    pub lps: Box<LpSolver>,
    pub run_data: RunData,
    presolve_status: i32,
    called_return: bool,
    pub timer: Timer,
    head: LogHead,
    shell: Shell,
    inf_cost: InfCostMods,
    /// The C++ HighsProfiling (setProfiling), or null
    pub profiling: *mut c_void,
    /// The interrupts of the IPX race, if any
    pub interrupt: Option<Interrupt>,
    /// A cancelled task interrupted IPX: the C++ caller throws
    /// HighsTask::Interrupt
    pub task_interrupted: bool,
    /// What the set_basis and pass_model steps take
    user_basis: Option<Basis>,
    user_model: Option<Lp>,
    /// The variables with no pivot of the presolve's dependent equations
    /// A C++ Highs object's engine: its log, clock and callbacks
    host: Option<CHost>,
    /// A Highs object's run changed the model's matrix (the undualized
    /// LP's, lpBack): C++ takes it back at the run's end
    pub model_matrix_back: bool,
}

// SAFETY: a handle is used by one thread at a time (the race's IPX handle
// is made and dropped on its thread); its raw pointers point into itself
// or to C++ objects that outlive it
unsafe impl Send for LpHandle {}

fn rm<T>(v: &mut Vec<T>) -> RsMut<T> {
    RsMut { ptr: v.as_mut_ptr(), len: v.len() }
}

fn status_of(v: i64) -> Status {
    match v {
        0 => Status::Ok,
        1 => Status::Warning,
        _ => Status::Error,
    }
}

impl LpHandle {
    /// Highs()
    pub fn new() -> Box<LpHandle> {
        let mut lps: Box<LpSolver> = Box::default();
        lps.run = LpRun { info: info_default(), ..LpRun::default() };
        let mut h = Box::new(LpHandle {
            opts: Opts::default(),
            saved_opts: None,
            model: Lp::default(),
            lps,
            // SAFETY: plain data, invalidated below
            run_data: unsafe { std::mem::zeroed() },
            presolve_status: super::run::PS_NOT_PRESOLVED,
            called_return: true,
            timer: Timer::default(),
            head: LogHead::none(),
            shell: Shell {
                lp_name: Vec::new(),
                app_factor_log: FactorLog::default(),
                app_factor_head: LogHead::none(),
                report: zero_report(),
                stats: SimplexStats::default(),
                factor_log: FactorLog::default(),
                factor_head: LogHead::none(),
                nla_model: false,
                nla_model_scale: false,
                nla_cpp: None,
                hot_start: None,
                primal_phase1_dual: Vec::new(),
            },
            inf_cost: InfCostMods::default(),
            profiling: std::ptr::null_mut(),
            interrupt: None,
            task_interrupted: false,
            user_basis: None,
            user_model: None,
            host: None,
            model_matrix_back: false,
        });
        h.run_data.invalidate();
        let o = &h.opts;
        h.head = LogHead {
            output_flag: &o.output_flag,
            log_to_console: &o.log_to_console,
            log_dev_level: &o.log_dev_level,
            log_stream: std::ptr::null(),
        };
        h.shell.factor_head = LogHead::of(&h.shell.factor_log);
        h.shell.app_factor_head = LogHead::of(&h.shell.app_factor_log);
        h
    }

    /// The engine of a C++ Highs object (HEkk's place in it)
    pub fn new_host(host: CHost) -> Box<LpHandle> {
        let mut h = LpHandle::new();
        h.host = Some(host);
        h
    }

    pub fn log(&self) -> Log {
        match &self.host {
            Some(c) => Log { opts: c.log_options, log: Some(crate::io::log::highs_rs_log) },
            None => log_of(&self.head),
        }
    }

    /// A copy of the options' log flags (and the log file of a Highs
    /// object's), as HFactor's log options are made
    fn factor_log_copy(&self) -> FactorLog {
        FactorLog {
            output_flag: self.opts.output_flag,
            log_to_console: self.opts.log_to_console,
            log_dev_level: self.opts.log_dev_level,
            // SAFETY: the Highs object's log options
            log_stream: match &self.host {
                Some(c) => unsafe { crate::io::log::log_options_stream(c.log_options) },
                None => std::ptr::null_mut(),
            },
        }
    }

    /// The run clock (timer_.read() of a Highs object)
    fn clock_read(&self) -> f64 {
        match &self.host {
            // SAFETY: the Highs object's timer
            Some(c) => unsafe { (c.timer_read)(c.ctx) },
            None => self.timer.read(0),
        }
    }

    fn factor_log(&self) -> Log {
        log_of(&self.shell.factor_head)
    }

    // ---- Options

    /// setOptionValue: false for an unknown name or a value of the wrong
    /// type
    pub fn set_option(&mut self, name: &str, value: OptValue) -> bool {
        self.opts.set(name, value)
    }

    /// passOptions
    pub fn pass_options(&mut self, from: &Opts) {
        self.opts.assign(from);
    }

    // ---- The view of the Highs object for run.rs and drivers.rs

    fn chighs(&mut self) -> CHighs {
        let log = self.log();
        let ctx = self as *mut LpHandle as *mut c_void;
        let o = self.opts.r_options();
        let r = &mut self.lps.run;
        CHighs {
            log,
            ctx,
            op: handle_op,
            clock: handle_clock,
            model_status: &mut r.model_status,
            presolve_status: &mut self.presolve_status,
            info: &mut r.info,
            run_data: &mut self.run_data,
            value_valid: &mut r.solution.value_valid,
            dual_valid: &mut r.solution.dual_valid,
            basis_valid: &mut r.basis.b.valid,
            basis_alien: &mut r.basis.b.alien,
            basis_useful: &mut r.basis.b.useful,
            basis_was_alien: &mut r.basis.b.was_alien,
            called_return: &mut self.called_return,
            o,
        }
    }

    /// Run `f` on a run of this handle
    fn with_run<R>(&mut self, f: impl FnOnce(&Run) -> R) -> R {
        let c = self.chighs();
        // SAFETY: c's pointers are into this handle, which outlives the run
        let run = unsafe { Run::new(&c) };
        f(&run)
    }

    // ---- Model status, info, solution and basis (Highs' invalidations)

    fn run(&mut self) -> &mut LpRun {
        &mut self.lps.run
    }

    /// Highs::invalidateSolution
    fn invalidate_solution(&mut self) {
        let r = self.run();
        let i = &mut r.info;
        i.primal_solution_status = SOLUTION_STATUS_NONE;
        i.dual_solution_status = SOLUTION_STATUS_NONE;
        i.num_primal_infeasibilities = -1;
        i.max_primal_infeasibility = INF;
        i.sum_primal_infeasibilities = INF;
        i.num_dual_infeasibilities = -1;
        i.max_dual_infeasibility = INF;
        i.sum_dual_infeasibilities = INF;
        r.solution.value_valid = false;
        r.solution.dual_valid = false;
    }

    /// Highs::invalidateBasis
    fn invalidate_basis(&mut self) {
        let r = self.run();
        r.info.basis_validity = BASIS_VALIDITY_INVALID;
        r.basis.invalidate();
    }

    /// Highs::invalidateModelStatusAndInfo (the ranging and IIS are not
    /// kept)
    fn invalidate_status_and_info(&mut self) {
        let r = self.run();
        r.model_status = MS_NOTSET;
        r.info.invalidate();
        self.run_data.invalidate();
    }

    /// Highs::invalidateSolverData
    fn invalidate_solver_data(&mut self) {
        self.invalidate_status_and_info();
        self.invalidate_solution();
        self.invalidate_basis();
        // invalidateEkk: HEkk::invalidate (and the simplex stats)
        self.ekk_invalidate();
    }

    /// Highs::clearDerivedModelProperties: the presolve data and the ray
    /// records
    fn clear_derived_model_properties(&mut self) {
        self.presolve_status = super::run::PS_NOT_PRESOLVED;
        self.lps.run.presolve = Default::default();
        self.lps.clear_ray_records();
    }

    /// HEkk::clear on the shell's side (clearCpp)
    pub(crate) fn clear_shell(&mut self) {
        self.shell.lp_name.clear();
        self.clear_nla_lp();
        self.shell.primal_phase1_dual.clear();
    }

    /// The simplex NLA has no LP of the shell (the model or a C++ LP)
    fn clear_nla_lp(&mut self) {
        self.shell.nla_model = false;
        self.shell.nla_model_scale = false;
        self.shell.nla_cpp = None;
    }

    /// HEkk::invalidate (and the simplex stats)
    fn ekk_invalidate(&mut self) {
        self.lps.invalidate();
        self.shell.stats.initialise(0);
    }

    /// HEkk::clear
    fn ekk_clear(&mut self) {
        self.clear_shell();
        self.lps.clear();
    }

    /// Highs::clearSolver
    pub fn clear_solver(&mut self) -> Status {
        self.clear_derived_model_properties();
        self.invalidate_solver_data();
        self.ekk_clear();
        self.with_run(|r| r.return_from_highs(Status::Ok))
    }

    /// Highs::clearModel
    pub fn clear_model(&mut self) -> Status {
        self.model.clear();
        self.inf_cost = InfCostMods::default();
        self.clear_solver()
    }

    // ---- The simplex shell (HEkk's C++ shell)

    /// HEkk::setNlaPointersForLpAndScale(model)
    pub(crate) fn set_nla_model(&mut self) {
        let s = &self.model.scale;
        self.shell.nla_model = true;
        self.shell.nla_model_scale = s.has_scaling && !self.model.is_scaled;
        self.shell.nla_cpp = None;
        self.lps.set_nla_rust(false);
    }

    /// HEkk::setNlaPointersForLpAndScale(lp) of a C++ LP (a Highs
    /// object's model): its view's dimensions and scale factors
    fn set_nla_cpp(&mut self, lp: &CLp) {
        let has_scale = lp.scale_has_scaling && !lp.is_scaled;
        let none = RsMut { ptr: std::ptr::null_mut(), len: 0 };
        self.shell.nla_model = false;
        self.shell.nla_model_scale = false;
        self.shell.nla_cpp = Some(NlaLp {
            num_col: lp.num_col,
            num_row: lp.num_row,
            has_scale,
            col_scale: if has_scale { lp.scale_col } else { none },
            row_scale: if has_scale { lp.scale_row } else { none },
        });
        self.lps.set_nla_rust(false);
    }

    /// HEkk::rsEnv: the option values, the simplex NLA's LP (the model or
    /// a C++ LP) and the host functions; the engine adds its own LP
    fn env(&mut self) -> LpsEnv {
        let none = RsMut { ptr: std::ptr::null_mut(), len: 0 };
        let nla = self.lps.sh.nla_lp_set && self.shell.nla_model;
        let has_scale = nla && self.shell.nla_model_scale;
        let interrupt_callback = match &self.host {
            // SAFETY: the Highs object's callback
            Some(c) => unsafe { (c.simplex_interrupt)(c.ctx, -1) },
            None => self.interrupt.is_some_and(|i| i.simplex),
        };
        let m = &mut self.model;
        let mut env = LpsEnv {
            lp: m.view(),
            model_name: none_u8(),
            nla_num_col: if nla { m.num_col } else { 0 },
            nla_num_row: if nla { m.num_row } else { 0 },
            nla_has_scale: has_scale,
            nla_col_scale: if has_scale { rm(&mut m.g.scale.col) } else { none },
            nla_row_scale: if has_scale { rm(&mut m.g.scale.row) } else { RsMut { ptr: std::ptr::null_mut(), len: 0 } },
            opt: self.opts.lps(),
            host: Host {
                ctx: self as *mut LpHandle as *mut c_void,
                log: host_log,
                timer_read: host_timer_read,
                interrupt: host_interrupt,
                chuzc_fail: host_chuzc_fail,
            },
            report: &mut self.shell.report,
            num_invert: &mut self.shell.stats.num_invert,
            num_threads: num_threads(),
            interrupt_callback,
        };
        if let (true, Some(c)) = (self.lps.sh.nla_lp_set, self.shell.nla_cpp) {
            env.nla_num_col = c.num_col;
            env.nla_num_row = c.num_row;
            env.nla_has_scale = c.has_scale;
            env.nla_col_scale = c.col_scale;
            env.nla_row_scale = c.row_scale;
        }
        env.lp = self.lps.lp.view();
        env
    }

    /// HEkk::movedLp: the engine's checks of its LP (and initialiseEkk if
    /// needed, which clears the simplex NLA)
    fn moved_lp(&mut self) {
        let env = self.env();
        let env = self.lps.env_of(&env);
        if self.lps.move_lp(&env) {
            self.clear_nla_lp();
        }
    }

    /// HEkk::solve
    fn ekk_solve(&mut self, force_phase2: bool) -> i64 {
        // initialiseAnalysis: HighsSimplexAnalysis::setup's report data
        let r = &mut self.shell.report;
        r.highs_run_time = 0.0;
        r.last_user_log_time = -INF;
        r.delta_user_log_time = 5.0;
        r.timeless_log = self.opts.timeless_log;
        r.num_iteration_report_since_last_header = -1;
        r.num_invert_report_since_last_header = -1;
        r.entering_variable = -1;
        r.pivotal_row_index = -1;
        r.col_aq_density = 0.0;
        r.row_ep_density = 0.0;
        r.row_ap_density = 0.0;
        r.row_dse_density = 0.0;
        // snapshotFactorLog: the solve sets up the simplex NLA for its LP
        if !self.lps.sh.status.has_nla {
            self.shell.factor_log = self.factor_log_copy();
        }
        // setNlaEngineLp
        self.clear_nla_lp();
        self.lps.set_nla_rust(true);
        let env = self.env();
        let env = self.lps.env_of(&env);
        let out = self.lps.solve(&env, force_phase2);
        self.take_rust_out();
        // returnFromEkkSolve
        self.lps.return_from_ekk_solve();
        let st = &mut self.shell.stats;
        st.valid = true;
        st.iteration_count += self.lps.sh.iteration_count;
        st.last_invert_num_el = out.invert_num_el;
        st.last_factored_basis_num_el = out.basis_matrix_num_el;
        let r = &self.shell.report;
        st.col_aq_density = r.col_aq_density;
        st.row_ep_density = r.row_ep_density;
        st.row_ap_density = r.row_ap_density;
        st.row_dse_density = r.row_dse_density;
        out.status as i64
    }

    /// HEkk::takeRustOut: the hot start and primal phase 1 duals that a
    /// solve or INVERT left are only the API's (a Highs object's)
    fn take_rust_out(&mut self) {
        let rec = std::mem::take(&mut self.lps.records.out);
        if self.host.is_some() {
            if rec.hot_start.is_some() {
                self.shell.hot_start = rec.hot_start;
            }
            if let Some(d) = rec.primal_phase1_dual {
                self.shell.primal_phase1_dual = d;
            }
        }
    }

    /// A step of solveLpSimplex on the shell (app.rs ops): code, the
    /// step's arg and pointer
    fn shell_op(&mut self, code: i32, arg: i64, p: *mut c_void) -> i64 {
        match code {
            // The profiling clocks of the simplex solve
            1 | 2 => {
                profiling_step(self.profiling, code, self.opts.simplex_strategy, arg);
                0
            }
            // initialiseSimplexStats
            3 => {
                self.shell.stats.initialise(self.lps.sh.iteration_count);
                0
            }
            4 => {
                self.moved_lp();
                0
            }
            6 => self.ekk_solve(arg != 0),
            7 => {
                self.ekk_clear();
                0
            }
            8 => {
                let env = self.env();
                let env = self.lps.env_of(&env);
                self.lps.proof_of_primal_infeasibility(&env) as i64
            }
            9 => {
                self.set_nla_model();
                0
            }
            10 => {
                // lpBack: the model takes the engine LP's scale (and its
                // rebuilt matrix)
                let e = &self.lps.lp.g;
                let m = &mut self.model.g;
                m.scale.clone_from(&e.scale);
                m.is_scaled = e.is_scaled;
                if arg != 0 {
                    m.a.clone_from(&e.a);
                    self.model_matrix_back = true;
                }
                0
            }
            11 => {
                let env = self.env();
                // SAFETY: an LpsEnv to fill
                unsafe { (p as *mut LpsEnv).write(env) };
                0
            }
            _ => unreachable!("simplex shell op {code}"),
        }
    }

    // ---- The model

    /// The facts of the model
    fn facts(&mut self) -> Facts {
        let m = &self.model;
        let nc = m.num_col as usize;
        Facts {
            num_col: m.num_col,
            num_row: m.num_row,
            num_nz: if m.is_colwise() { m.a.start.get(nc).copied().unwrap_or(0) } else { m.a.num_nz() },
            is_mip: m.integrality.iter().any(|&t| t != 0),
            is_qp: false,
            is_empty: m.num_col == 0 && m.num_row == 0,
            has_infinite_cost: m.has_infinite_cost,
            model_name: super::options::RsStr::of(&m.model_name),
        }
    }

    /// HighsLp::exactResize
    fn exact_resize_model(&mut self) {
        let m = &mut self.model.g;
        let (nc, nr) = (m.num_col as usize, m.num_row as usize);
        m.col_cost.resize(nc, 0.0);
        m.col_lower.resize(nc, 0.0);
        m.col_upper.resize(nc, 0.0);
        m.row_lower.resize(nr, 0.0);
        m.row_upper.resize(nr, 0.0);
        m.a.exact_resize();
        if !m.integrality.is_empty() {
            m.integrality.resize(nc, 0);
        }
    }

    /// Highs::handleInfCost
    fn handle_inf_cost(&mut self) -> Status {
        if !self.model.has_infinite_cost {
            return Status::Ok;
        }
        let n = self.model.num_col as usize;
        let (mut index, mut cost, mut lower, mut upper) = (vec![0i32; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        let mut m = super::model::CInfCostMods {
            index: rm(&mut index),
            cost: rm(&mut cost),
            lower: rm(&mut lower),
            upper: rm(&mut upper),
            num: 0,
        };
        let log = self.log();
        let lp = &mut self.model.g;
        let is_mip = lp.integrality.iter().any(|&t| t != 0);
        // SAFETY: the buffers hold num_col entries
        let s = unsafe {
            super::model::handle_inf_cost(
                &log,
                self.opts.infinite_cost,
                lp.sense == 1,
                is_mip,
                &lp.integrality,
                &mut lp.col_cost,
                &mut lp.col_lower,
                &mut lp.col_upper,
                &mut m,
            )
        };
        if s == Status::Error {
            return Status::Error;
        }
        let k = m.num as usize;
        let c = &mut self.inf_cost;
        c.index.extend_from_slice(&index[..k]);
        c.cost.extend_from_slice(&cost[..k]);
        c.lower.extend_from_slice(&lower[..k]);
        c.upper.extend_from_slice(&upper[..k]);
        self.model.has_infinite_cost = false;
        Status::Ok
    }

    /// Highs::restoreInfCost (and HighsLp::unapplyMods, which has nothing
    /// else to undo without semi-variables)
    fn undo_mods(&mut self, mut return_status: Status) -> Status {
        if self.inf_cost.index.is_empty() {
            return return_status;
        }
        let c = std::mem::take(&mut self.inf_cost);
        let r = &mut self.lps.run;
        let lp = &mut self.model.g;
        let col_value: &[f64] = if r.solution.value_valid { &r.solution.col_value } else { &[] };
        let col_status: &mut [u8] = if r.basis.b.valid { &mut r.basis.b.col_status } else { &mut [] };
        super::model::restore_inf_cost(
            &c.index,
            &c.cost,
            &c.lower,
            &c.upper,
            col_value,
            col_status,
            &mut lp.col_cost,
            &mut lp.col_lower,
            &mut lp.col_upper,
            &mut r.info.objective_function_value,
        );
        lp.has_infinite_cost = true;
        if r.model_status == super::run::MS_INFEASIBLE {
            r.model_status = MS_UNKNOWN;
            // setHighsModelStatusAndClearSolutionAndBasis
            self.invalidate_solution();
            self.invalidate_basis();
            self.lps.run.info.valid = true;
            return_status = super::run::status_from_model_status(MS_UNKNOWN);
        }
        // unapplyMods clears the records
        return_status
    }

    /// Highs::basisForSolution: the statuses of the solution, set as an
    /// alien basis
    fn basis_for_solution(&mut self) -> Status {
        self.invalidate_basis();
        let log = self.log();
        let (nc, nr) = (self.model.num_col as usize, self.model.num_row as usize);
        let mut basis = Basis::default();
        basis.b.col_status.resize(nc, 0);
        basis.b.row_status.resize(nr, 0);
        let s = &self.lps.run.solution;
        let m = &self.model;
        super::model::basis_for_solution(
            &log,
            self.opts.primal_feasibility_tolerance,
            &m.col_lower,
            &m.col_upper,
            &s.col_value,
            &m.row_lower,
            &m.row_upper,
            &s.row_value,
            &mut basis.b.col_status,
            &mut basis.b.row_status,
        );
        self.set_basis(basis, "")
    }

    // ---- The Highs API

    /// Highs::passModel(HighsLp): the model's checks and a cleared solver
    pub fn pass_model(&mut self, lp: Lp) -> Status {
        self.user_model = Some(lp);
        self.with_run(|r| r.pass_model())
    }

    /// Highs::setBasis(basis, origin)
    pub fn set_basis(&mut self, basis: Basis, origin: &str) -> Status {
        let alien = basis.b.alien;
        self.user_basis = Some(basis);
        let s = self.with_run(|r| r.set_basis(alien, origin));
        self.user_basis = None;
        s
    }

    /// Highs::optimizeLp (calledOptimizeModel); a task interrupt of IPX is
    /// in `task_interrupted`
    pub fn optimize_lp(&mut self) -> Status {
        self.task_interrupted = false;
        self.with_run(|r| r.called_optimize_model())
    }

    /// Highs::run for an LP without file options or user scaling:
    /// optimizeModel (the scheduler must have been initialized)
    pub fn run_lp(&mut self) -> Status {
        // initializeMultiThreading: the scheduler has this many threads
        if self.opts.threads != 0 && num_threads() != self.opts.threads {
            return Status::Error;
        }
        debug_assert!(self.opts.user_objective_scale == 0 && self.opts.user_bound_scale == 0);
        let status = self.optimize_lp();
        if status == Status::Error {
            return status;
        }
        self.run_data.valid = true;
        status
    }

    /// The end of a Highs model edit (interpretCallStatus and
    /// returnFromHighs)
    fn edit_return(&mut self, call: Status, what: &str) -> Status {
        let s = self.log().interpret(call, Status::Ok, what);
        if s == Status::Error {
            return Status::Error;
        }
        self.with_run(|r| r.return_from_highs(s))
    }

    fn iface_options(&self) -> IfaceOptions {
        let o = &self.opts;
        IfaceOptions {
            log: self.log(),
            infinite_cost: o.infinite_cost,
            infinite_bound: o.infinite_bound,
            small_matrix_value: o.small_matrix_value,
            large_matrix_value: o.large_matrix_value,
            allowed_matrix_scale_factor: o.allowed_matrix_scale_factor,
        }
    }

    /// The interfaces' view of the model, basis and host
    fn iface<R>(&mut self, f: impl FnOnce(&mut Lp, &mut BasisG<Vec<u8>>, &mut Iface, &IfaceOptions) -> R) -> R {
        let o = self.iface_options();
        let mut h = Iface(self as *mut LpHandle);
        // SAFETY: the interfaces borrow the model and basis, and reach the
        // rest of the handle through `h` (no field is borrowed twice)
        let me = unsafe { &mut *(self as *mut LpHandle) };
        f(&mut self.model, &mut me.lps.run.basis.b, &mut h, &o)
    }

    /// Highs::changeColsBounds over an index collection (after its
    /// creation); `set` collections are sorted with their bounds
    fn change_col_bounds_ic(&mut self, ic: &IndexCollection, lower: Vec<f64>, upper: Vec<f64>) -> Status {
        self.clear_derived_model_properties();
        let num = interface_data_size(ic);
        let call = if num <= 0 {
            Status::Ok
        } else {
            self.iface(|lp, b, h, o| interface::change_bounds_iface(lp, b, h, o, ic, true, lower, upper))
        };
        self.edit_return(call, "changeColBounds")
    }

    /// Highs::changeColsBounds(from, to, lower, upper)
    pub fn change_col_bounds_interval(&mut self, from: i32, to: i32, lower: &[f64], upper: &[f64]) -> Status {
        let ic = IndexCollection::interval(self.model.num_col, from, to);
        if from < 0 || to >= self.model.num_col {
            return Status::Error;
        }
        let n = (to - from + 1).max(0) as usize;
        self.change_col_bounds_ic(&ic, lower[..n].to_vec(), upper[..n].to_vec())
    }

    /// Highs::changeColsBounds(num, set, lower, upper) (changeColBounds
    /// for one column)
    pub fn change_col_bounds_set(&mut self, set: &[i32], lower: &[f64], upper: &[f64]) -> Status {
        if set.is_empty() {
            return Status::Ok;
        }
        let mut local_set = set.to_vec();
        // sortSetData, then the interface's own sort of the sorted set
        let (l1, u1) = interface::sort_set_bounds(&mut local_set, lower, upper);
        let (l2, u2) = interface::sort_set_bounds(&mut local_set, &l1, &u1);
        let ic = IndexCollection {
            dimension: self.model.num_col,
            is_interval: false,
            from: -1,
            to: -2,
            is_set: true,
            set_num_entries: local_set.len() as i32,
            set: &local_set,
            is_mask: false,
            mask: &[],
        };
        self.change_col_bounds_ic(&ic, l2, u2)
    }

    /// Highs::changeColsCost(mask, cost)
    pub fn change_col_costs_mask(&mut self, mask: &[i32], cost: &[f64]) -> Status {
        self.clear_derived_model_properties();
        let n = self.model.num_col as usize;
        let ic = mask_collection(self.model.num_col, &mask[..n]);
        let call = if interface_data_size(&ic) <= 0 {
            Status::Ok
        } else {
            self.iface(|lp, _b, h, o| interface::change_costs_iface(lp, h, o, &ic, &cost[..n]))
        };
        self.edit_return(call, "changeCosts")
    }

    /// Highs::addRows
    #[allow(clippy::too_many_arguments)]
    pub fn add_rows(
        &mut self,
        num_new_row: i32,
        lower: &[f64],
        upper: &[f64],
        num_new_nz: i32,
        start: &[i32],
        index: &[i32],
        value: &[f64],
    ) -> Status {
        self.clear_derived_model_properties();
        let call = if num_new_row < 0 || num_new_nz < 0 {
            Status::Error
        } else if num_new_row == 0 {
            Status::Ok
        } else {
            self.iface(|lp, b, h, o| {
                interface::add_rows(lp, b, h, o, num_new_row, lower, upper, num_new_nz, start, index, value)
            })
        };
        self.edit_return(call, "addRows")
    }

    /// Highs::deleteRows(from, to)
    pub fn delete_rows_interval(&mut self, from: i32, to: i32) -> Status {
        self.clear_derived_model_properties();
        if from < 0 || to >= self.model.num_row {
            return Status::Error;
        }
        let ic = IndexCollection::interval(self.model.num_row, from, to);
        self.iface(|lp, b, h, _o| interface::delete_rows(lp, b, h, &ic));
        self.with_run(|r| r.return_from_highs(Status::Ok))
    }

    /// Highs::deleteRows(mask): the mask becomes the new row indices (-1
    /// for deleted rows)
    pub fn delete_rows_mask(&mut self, mask: &mut [i32]) -> Status {
        self.clear_derived_model_properties();
        let n = self.model.num_row as usize;
        let local = mask[..n].to_vec();
        let ic = mask_collection(self.model.num_row, &local);
        let renumber = self.iface(|lp, b, h, _o| interface::delete_rows(lp, b, h, &ic));
        mask[..n].copy_from_slice(&local);
        if renumber {
            interface::renumber_mask(&mut mask[..n]);
        }
        self.with_run(|r| r.return_from_highs(Status::Ok))
    }

    /// Highs::putIterate
    pub fn put_iterate(&mut self) -> Status {
        if !self.lps.sh.status.has_invert {
            let log = self.log();
            log.user(LogType::Error, "putIterate: no simplex iterate to put\n");
            return Status::Error;
        }
        self.lps.put_iterate();
        self.with_run(|r| r.return_from_highs(Status::Ok))
    }

    /// Highs::getIterate
    pub fn get_iterate(&mut self) -> Status {
        if !self.lps.sh.status.initialised_for_new_lp {
            let log = self.log();
            log.user(LogType::Error, "getIterate: no simplex iterate to get\n");
            return Status::Error;
        }
        if !self.lps.get_iterate() {
            return Status::Error;
        }
        // basis_ = ekk_instance_.getHighsBasis(model_.lp_)
        self.get_highs_basis();
        // invalidateModelStatusSolutionAndInfo
        self.invalidate_status_and_info();
        self.invalidate_solution();
        self.with_run(|r| r.return_from_highs(Status::Ok))
    }

    /// basis_ = HEkk::getHighsBasis(model_.lp_)
    fn get_highs_basis(&mut self) {
        let (nc, nr) = (self.model.num_col as usize, self.model.num_row as usize);
        let lp = self.model.view();
        let lps = &mut self.lps;
        let sense = lps.lp.sense;
        let b = &mut lps.run.basis;
        b.b.col_status.resize(nc, 0);
        b.b.row_status.resize(nr, 0);
        let mut col = std::mem::take(&mut b.b.col_status);
        let mut row = std::mem::take(&mut b.b.row_status);
        let (id, count) = lps.get_highs_basis(&lp, sense, &mut col, &mut row);
        let origin = String::from_utf8_lossy(lps.basis.debug_origin_name.as_bytes()).into_owned();
        let b = &mut lps.run.basis;
        b.b.col_status = col;
        b.b.row_status = row;
        b.b.valid = true;
        b.b.alien = false;
        b.b.useful = true;
        b.b.was_alien = false;
        b.b.debug_id = id;
        b.b.debug_update_count = count;
        b.origin = origin;
    }

    /// Highs::getBasisInverseRowSparse
    ///
    /// # Safety
    /// `rhs` a valid HVector view of at least num_row entries
    pub unsafe fn basis_inverse_row_sparse(&mut self, row: i32, rhs: &mut crate::ffi::CHVec) {
        self.set_nla_model();
        let mut v = rhs.view();
        v.clear();
        v.count = 1;
        v.index[0] = row;
        v.array[row as usize] = 1.0;
        v.pack_flag = true;
        rhs.store(&v);
        let density = self.lps.sh.info.row_ep_density;
        let env = self.env();
        let env = self.lps.env_of(&env);
        self.lps.nla_solve(&env, rhs, density, true);
    }

    /// Highs::getDualRaySparse: whether there is a dual ray
    ///
    /// # Safety
    /// As basis_inverse_row_sparse
    pub unsafe fn dual_ray_sparse(&mut self, rhs: &mut crate::ffi::CHVec) -> bool {
        let row = self.lps.sh.dual_ray_index;
        if row == -1 {
            return false;
        }
        self.set_nla_model();
        let mut v = rhs.view();
        v.clear();
        v.count = 1;
        v.pack_flag = true;
        v.index[0] = row;
        v.array[row as usize] = self.lps.sh.dual_ray_sign as f64;
        rhs.store(&v);
        let density = self.lps.sh.info.row_ep_density;
        let env = self.env();
        let env = self.lps.env_of(&env);
        self.lps.nla_solve(&env, rhs, density, true);
        true
    }

    pub fn has_invert(&self) -> bool {
        self.lps.sh.status.has_invert
    }

    /// Highs::getBasicVariablesArray
    pub fn basic_index(&self) -> &[i32] {
        &self.lps.basis.basic_index
    }

    /// Highs::getDualEdgeWeights: null without DSE weights
    pub fn dual_edge_weights(&self) -> *const f64 {
        if self.lps.sh.status.has_dual_steepest_edge_weights {
            self.lps.dual_edge_weight.as_ptr()
        } else {
            std::ptr::null()
        }
    }

    /// Highs::getRunTime
    pub fn run_time(&self) -> f64 {
        self.timer.read(0)
    }

    pub fn model_status(&self) -> i32 {
        self.lps.run.model_status
    }
    pub fn info(&self) -> &Info {
        &self.lps.run.info
    }
    pub fn solution(&self) -> &Solution {
        &self.lps.run.solution
    }
    pub fn basis(&self) -> &Basis {
        &self.lps.run.basis
    }

    // ---- The steps of the run on this handle (handle_op)

    fn op(&mut self, op: i32, arg: i64, p: *mut c_void, msg: &[u8]) -> i64 {
        macro_rules! is {
            ($o:ident) => {
                op == Op::$o as i32
            };
        }
        if is!(ClearSolver) || is!(ClearSolver2) {
            return self.clear_solver() as i64;
        }
        if is!(HandleInfCost) {
            return self.handle_inf_cost() as i64;
        }
        if is!(ExactResizeModel) {
            self.exact_resize_model();
            return 0;
        }
        if is!(InvalidateInfo) {
            self.lps.run.info.invalidate();
            return 0;
        }
        if is!(InvalidateRunData) {
            self.run_data.invalidate();
            return 0;
        }
        if is!(InvalidateBasis) {
            self.invalidate_basis();
            return 0;
        }
        if is!(Facts) {
            debug_assert_eq!(arg, 0);
            let f = self.facts();
            // SAFETY: the run's Facts
            unsafe { (p as *mut Facts).write(f) };
            return 0;
        }
        if is!(EnsureColwise) {
            self.model.a.ensure_colwise();
            return 0;
        }
        if is!(HasLargeValue) {
            return self.model.a.has_large_value(self.opts.large_matrix_value) as i64;
        }
        if is!(DebugAssess) {
            let o = self.opts.lp_options(self.log());
            let mut v = self.model.view();
            let s = assess_lp(&mut v, &o);
            self.model.take_scalars(&v);
            return s as i64;
        }
        if is!(AssessSemiVariables) {
            // An LP relaxation has no semi-variables
            // SAFETY: the run's flag
            unsafe { *(p as *mut bool) = false };
            debug_assert!(!self.model.integrality.iter().any(|&t| t == 2 || t == 3));
            return 0;
        }
        if is!(BasisForSolution) {
            return self.basis_for_solution() as i64;
        }
        if is!(BasisClear) {
            let b = &mut self.lps.run.basis;
            b.invalidate();
            b.b.col_status.clear();
            b.b.row_status.clear();
            return 0;
        }
        if is!(RefineBasis) {
            let r = &mut self.lps.run;
            let m = &self.model;
            let (cv, rv): (&[f64], &[f64]) =
                if r.solution.value_valid { (&r.solution.col_value, &r.solution.row_value) } else { (&[], &[]) };
            refine_basis(&m.col_lower, &m.col_upper, cv, &mut r.basis.b.col_status);
            refine_basis(&m.row_lower, &m.row_upper, rv, &mut r.basis.b.row_status);
            return 0;
        }
        if is!(SetEkkLpName) {
            self.shell.lp_name.clear();
            self.shell.lp_name.extend_from_slice(msg);
            return 0;
        }
        if is!(EkkClear) {
            self.ekk_clear();
            return 0;
        }
        if is!(EkkInvalidate) {
            self.ekk_invalidate();
            return 0;
        }
        if is!(EkkPivotThreshold) {
            if !self.lps.sh.status.initialised_for_solve {
                return 0;
            }
            // SAFETY: the run's f64
            unsafe { *(p as *mut f64) = self.lps.sh.info.factor_pivot_threshold };
            return 1;
        }
        if is!(SaveOptions) {
            self.saved_opts = Some(Box::new(self.opts.clone()));
            return 0;
        }
        if is!(RestoreOptions) {
            if let Some(s) = self.saved_opts.take() {
                self.opts.assign(&s);
            }
            return 0;
        }
        if is!(OptionsPrimalSimplex) {
            self.opts.set("solver", OptValue::Str(b"simplex"));
            self.opts.simplex_strategy = STRATEGY_PRIMAL;
            return 0;
        }
        if is!(OptionsCleanup) {
            self.opts.set("solver", OptValue::Str(b"simplex"));
            self.opts.simplex_strategy = STRATEGY_CHOOSE;
            self.opts.simplex_min_concurrency = 1;
            self.opts.simplex_max_concurrency = 1;
            if arg != 0 {
                // SAFETY: the run's f64
                self.opts.factor_pivot_threshold = unsafe { *(p as *const f64) };
            }
            return 0;
        }
        if is!(SetPostsolveStatus) || is!(PresolveTime) || is!(PresolveRemoved) || is!(PresolveInit) || is!(PresolveClear)
        {
            // The C++ PresolveComponent's records: none here
            return 0;
        }
        if is!(UndoMods) {
            return self.undo_mods(status_of(arg)) as i64;
        }
        if is!(DebugReturn) || is!(DebugPostsolveSolution) {
            // Debugging is left out of this build
            return 0;
        }
        if is!(ForceSolutionBasisSize) {
            self.with_run(|r| r.force_solution_basis_size());
            return 0;
        }
        if is!(BasisConsistent) || is!(RetainedEkkDataOk) {
            // debugHighsBasisConsistent and debugRetainedDataOk are not
            // checked in this build
            return 1;
        }
        if is!(LpDimensionsOk) {
            let log = self.log();
            let v = self.model.view();
            return lp_dimensions_ok(&log, "returnFromHighs", &v) as i64;
        }
        if is!(EkkFactorCompatible) {
            if !self.lps.sh.status.has_nla {
                return -1;
            }
            let env = self.env();
            let env = self.lps.env_of(&env);
            return self.lps.lp_factor_row_compatible(&env, self.model.num_row) as i64;
        }
        if is!(LpView) {
            if arg == 2 {
                // The view was changed: it is the handle's model
                return 0;
            }
            debug_assert_eq!(arg, 0);
            let v = self.model.view();
            // SAFETY: the run's CLp
            unsafe { (p as *mut CLp).write(v) };
            return 0;
        }
        if is!(SolutionBasisSizes) {
            let r = &mut self.lps.run;
            let (s, b) = (&mut r.solution, &mut r.basis.b);
            if arg == 0 {
                let sizes = [
                    s.col_value.len() as i64,
                    s.row_value.len() as i64,
                    s.col_dual.len() as i64,
                    s.row_dual.len() as i64,
                    b.col_status.len() as i64,
                    b.row_status.len() as i64,
                ];
                // SAFETY: the run's [i64; 6]
                unsafe { (p as *mut [i64; 6]).write(sizes) };
            } else {
                let (nc, nr) = (self.model.num_col as usize, self.model.num_row as usize);
                s.col_value.resize(nc, 0.0);
                s.row_value.resize(nr, 0.0);
                s.col_dual.resize(nc, 0.0);
                s.row_dual.resize(nr, 0.0);
                b.col_status.resize(nc, NONBASIC);
                b.row_status.resize(nr, BASIC);
            }
            return 0;
        }
        // drivers.rs: passModel
        if is!(LogHeader) || is!(MatrixImages) || is!(ReportModelStats) {
            return 0;
        }
        if is!(ClearModel) {
            self.clear_model();
            return 0;
        }
        if is!(TakeModel) {
            let lp = self.user_model.take().expect("passModel's model");
            self.model = lp;
            return 0;
        }
        if is!(EmptyMatrix) {
            let n = self.model.num_col as usize;
            let a = &mut self.model.g.a;
            a.format = super::matrix_format::COLWISE;
            a.start.clear();
            a.start.resize(n + 1, 0);
            a.index.clear();
            a.value.clear();
            return 0;
        }
        if is!(FormatOk) {
            return if arg != 0 { 1 } else { (self.model.a.is_colwise() || self.model.a.is_rowwise()) as i64 };
        }
        if is!(PrepareModelLp) {
            let m = &mut self.model.g;
            m.a.num_col = m.num_col;
            m.a.num_row = m.num_row;
            m.clear_scale();
            return 0;
        }
        if is!(AssessLp) {
            let o = self.opts.lp_options(self.log());
            let mut v = self.model.view();
            let s = assess_lp(&mut v, &o);
            self.model.take_scalars(&v);
            let m = &mut self.model.g;
            if s != Status::Error && m.num_col > 0 {
                // Entries may have been removed from the matrix
                let nnz = m.a.num_nz() as usize;
                m.a.index.truncate(nnz);
                m.a.value.truncate(nnz);
            }
            return s as i64;
        }
        if is!(AssessHessian) || is!(HessianClear) || is!(CompleteHessian) {
            return 0;
        }
        if is!(HessianDims) {
            // SAFETY: the run's [i32; 2]
            unsafe { (p as *mut [i32; 2]).write([0, 0]) };
            return 0;
        }
        // drivers.rs: setBasis
        if is!(SetBasis) {
            return self.set_basis_step(arg, p);
        }
        if is!(SetBasisOrigin) {
            self.lps.run.basis.origin = String::from_utf8_lossy(msg).into_owned();
            return 0;
        }
        if is!(BasisDebug) {
            let b = &self.lps.run.basis;
            let d = super::drivers::BasisDebug {
                id: b.b.debug_id,
                update_count: b.b.debug_update_count,
                origin: super::options::RsStr::of(b.origin.as_bytes()),
            };
            // SAFETY: the run's BasisDebug
            unsafe { (p as *mut super::drivers::BasisDebug).write(d) };
            return 0;
        }
        if is!(NewHighsBasis) {
            // newHighsBasis: HEkk::updateStatus(kNewBasis)
            if self.lps.update_status(LP_NEW_BASIS) {
                self.clear_shell();
            }
            return 0;
        }
        // lp_run.rs
        if is!(LpRustBegin) {
            // SAFETY: the run's handle pointer
            unsafe { *(p as *mut *mut LpHandle) = self };
            return 0;
        }
        if is!(LpRustEnd) {
            return 0;
        }
        if is!(KktOptions) {
            let o = self.opts.kkt(self.log());
            // SAFETY: the run's CKktOptions
            unsafe { (p as *mut super::solution::CKktOptions).write(o) };
            return 0;
        }
        if is!(SolveTemplate) {
            let c = CSolve {
                log: self.log(),
                ctx: std::ptr::null_mut(),
                op: no_solve_op,
                model_status: std::ptr::null_mut(),
                info: std::ptr::null_mut(),
                value_valid: std::ptr::null(),
                basis_valid: std::ptr::null(),
                num_row: 0,
                num_nz: 0,
                solver: super::options::RsStr::of(&self.opts.solver),
                run_crossover: super::options::RsStr::of(&self.opts.run_crossover),
                run_centring: self.opts.run_centring,
                allow_unbounded_or_infeasible: self.opts.allow_unbounded_or_infeasible,
                highs_debug_level: self.opts.highs_debug_level,
                output_flag: &self.opts.output_flag,
                log_dev_level: &self.opts.log_dev_level,
                aborted: std::ptr::null(),
            };
            // SAFETY: the run's CSolve
            unsafe { (p as *mut CSolve).write(c) };
            return 0;
        }
        if is!(UnconstrainedTemplate) {
            let t = UnconTemplate {
                on: self.opts.output_flag,
                primal_feasibility_tolerance: self.opts.primal_feasibility_tolerance,
                dual_feasibility_tolerance: self.opts.dual_feasibility_tolerance,
            };
            // SAFETY: the run's UnconTemplate
            unsafe { (p as *mut UnconTemplate).write(t) };
            return 0;
        }
        if is!(IpxTemplate) {
            let h = self.ipx_template();
            // SAFETY: the run's CIpxHost
            unsafe { (p as *mut CIpxHost).write(h) };
            return 0;
        }
        if is!(PdlpProfiling) {
            profiling_step(self.profiling, if arg != 0 { 3 } else { 4 }, 0, 0);
            return 0;
        }
        if is!(SimplexTemplate) {
            let h = self.simplex_template();
            // SAFETY: the run's CSimplexApp
            unsafe { (p as *mut CSimplexApp).write(h) };
            return 0;
        }
        if is!(SimplexShell) {
            let code = (arg >> 32) as i32;
            let which = (arg >> 16) & 1;
            if which == 1 && code == 9 {
                // The reduced LP is the engine's: setNlaEngineLp
                self.shell.nla_model = false;
                self.shell.nla_model_scale = false;
                self.lps.set_nla_rust(true);
                return 0;
            }
            return self.shell_op(code, arg & 0xffff, p);
        }
        if is!(SetInterrupt) {
            self.task_interrupted = true;
            return 0;
        }
        if is!(PresolveOptions) {
            let o = PresolveOptions {
                o: self.opts.presolve(),
                reduction_limit: if self.opts.presolve_reduction_limit < 0 {
                    -1
                } else {
                    self.opts.presolve_reduction_limit
                },
            };
            // SAFETY: the run's PresolveOptions
            unsafe { (p as *mut PresolveOptions).write(o) };
            return 0;
        }
        if is!(AssessSmallValues) {
            // analyseVectorValues' report of the small values (a log only)
            if self.opts.output_flag {
                let log = self.log();
                log.dev(LogType::Info, "Small values in matrix\n");
            }
            return 0;
        }
        if is!(LpOptions) {
            let o = self.opts.lp_options(self.log());
            // SAFETY: the run's CLpOptions
            unsafe { (p as *mut super::ffi::CLpOptions).write(o) };
            return 0;
        }
        unreachable!("LpHandle: run step {op} is not one of an LP")
    }

    /// The steps of setBasis with the user's basis (drivers.rs)
    fn set_basis_step(&mut self, arg: i64, p: *mut c_void) -> i64 {
        let user = self.user_basis.as_ref().expect("setBasis' basis");
        let (nc, nr) = (self.model.num_col as usize, self.model.num_row as usize);
        match arg {
            0 => {
                let b = &mut self.lps.run.basis.b;
                for (s, &u) in b.col_status.iter_mut().zip(&user.b.col_status).take(nc) {
                    *s = if u == BASIC { NONBASIC } else { u };
                }
                b.alien = false;
                0
            }
            1 => {
                let b = &self.lps.run.basis.b;
                let sizes = [b.col_status.len() as i64, b.row_status.len() as i64, nc as i64, nr as i64];
                // SAFETY: the run's [i64; 4]
                unsafe { (p as *mut [i64; 4]).write(sizes) };
                (user.b.col_status.len() == nc && user.b.row_status.len() == nr) as i64
            }
            2 => {
                let mut basis = user.clone();
                basis.b.was_alien = true;
                let s = self.form_basis(&mut basis);
                if s != Status::Ok {
                    return s as i64;
                }
                self.lps.run.basis = basis;
                0
            }
            3 => {
                let ok = user.b.col_status.len() == nc
                    && user.b.row_status.len() == nr
                    && basis_consistent(&user.b.col_status, &user.b.row_status);
                ok as i64
            }
            _ => {
                self.lps.run.basis = user.clone();
                0
            }
        }
    }

    /// formSimplexLpBasisAndFactor of an alien basis on the model
    fn form_basis(&mut self, basis: &mut Basis) -> Status {
        let b = FormBasis {
            valid: basis.b.valid,
            useful: basis.b.useful,
            alien: &mut basis.b.alien,
            col_status: rm(&mut basis.b.col_status),
            row_status: rm(&mut basis.b.row_status),
            debug_id: basis.b.debug_id,
            debug_update_count: basis.b.debug_update_count,
            origin: RsMut { ptr: basis.origin.as_ptr() as *mut u8, len: basis.origin.len() },
        };
        // SAFETY: the views are of the basis, which outlives the call
        unsafe { self.form_basis_of(b, false) }
    }

    /// formSimplexLpBasisAndFactor on the model with a basis' views
    ///
    /// # Safety
    /// The views valid for the call
    unsafe fn form_basis_of(&mut self, b: FormBasis, only_from_known_basis: bool) -> Status {
        let lp_options = self.opts.lp_options(self.log());
        let mut ctx = FormCtx { h: self as *mut LpHandle, b };
        // the factor's log: a copy of the options' log flags
        self.shell.app_factor_log = self.factor_log_copy();
        let (log, factor_log) = (self.log(), log_of(&self.shell.app_factor_head));
        let m = &mut self.model;
        let host = super::form_basis::CFormHost {
            ctx: &mut ctx as *mut FormCtx as *mut c_void,
            op: form_op,
            log,
            factor_log,
            lps: &mut *self.lps,
            incumbent: m.view(),
            model_name: rm(&mut m.model_name),
            lp_options,
            basis_valid: b.valid,
            basis_useful: b.useful,
            basis_alien: b.alien,
            col_status: b.col_status,
            row_status: b.row_status,
        };
        super::form_basis::form_simplex_lp_basis_and_factor(&host, only_from_known_basis)
    }

    /// solveLpSimplex's options and logs (rsSimplexAppTemplate), the model
    /// as incumbent
    fn simplex_template(&mut self) -> CSimplexApp {
        let lp_options = self.opts.lp_options(self.log());
        // The factor's log: a copy of the options' log flags
        self.shell.app_factor_log = self.factor_log_copy();
        let log = self.log();
        let m = &mut self.model;
        CSimplexApp {
            log,
            factor_log: log_of(&self.shell.app_factor_head),
            ctx: std::ptr::null_mut(),
            op: no_app_op,
            lps: std::ptr::null_mut(),
            incumbent: m.view(),
            model_name: rm(&mut m.model_name),
            model_status: std::ptr::null_mut(),
            info: std::ptr::null_mut(),
            value_valid: std::ptr::null_mut(),
            dual_valid: std::ptr::null_mut(),
            col_value: empty_rs_vec(),
            col_dual: empty_rs_vec(),
            row_value: empty_rs_vec(),
            row_dual: empty_rs_vec(),
            basis_valid: std::ptr::null_mut(),
            basis_alien: std::ptr::null_mut(),
            basis_useful: std::ptr::null_mut(),
            basis_was_alien: std::ptr::null_mut(),
            basis_debug_id: std::ptr::null_mut(),
            basis_debug_update_count: std::ptr::null_mut(),
            col_status: empty_rs_vec(),
            row_status: empty_rs_vec(),
            simplex_strategy: &mut self.opts.simplex_strategy,
            dual_simplex_cost_perturbation_multiplier: &mut self.opts.dual_simplex_cost_perturbation_multiplier,
            simplex_unscaled_solution_strategy: self.opts.simplex_unscaled_solution_strategy,
            cost_scale_factor: self.opts.cost_scale_factor,
            simplex_dualize_strategy: self.opts.simplex_dualize_strategy,
            simplex_permute_strategy: self.opts.simplex_permute_strategy,
            lp_options,
        }
    }

    /// callCrossover (Highs::crossover) on the model and the run's data:
    /// IPX crossover from the run's solution; the HighsStatus, or
    /// ipx_glue's `CANCELLED`
    pub fn crossover(&mut self) -> i32 {
        use super::lp_run::{ipx_resize, ipx_timer, IpxCtx};
        let mut h = self.ipx_template();
        let lps: *mut LpSolver = &mut *self.lps;
        let c = IpxCtx {
            timer_read: h.timer_read,
            timer_ctx: h.ctx,
            lps,
            num_col: self.model.num_col as usize,
            num_row: self.model.num_row as usize,
        };
        h.ctx = &c as *const IpxCtx as *mut c_void;
        h.timer_read = ipx_timer;
        h.resize = ipx_resize;
        // SAFETY: the run's data, not otherwise borrowed during the call
        unsafe {
            let r = &mut (*lps).run;
            h.info = &mut r.info;
            h.model_status = &mut r.model_status;
            h.value_valid = &mut r.solution.value_valid;
            h.dual_valid = &mut r.solution.dual_valid;
            h.basis_valid = &mut r.basis.b.valid;
            h.basis_useful = &mut r.basis.b.useful;
            let s = &r.solution;
            let (nc, nr) = (c.num_col, c.num_row);
            let x = s.col_value.clone();
            let duals = (s.dual_valid && s.col_dual.len() == nc && s.row_dual.len() == nr)
                .then(|| (s.row_dual.clone(), s.col_dual.clone()));
            super::ipx_glue::call_crossover(&h, &x, duals.as_ref().map(|(r, c)| (&r[..], &c[..])))
        }
    }

    /// solveLpIpx's options, hooks and timer (rsIpxHostTemplate)
    fn ipx_template(&mut self) -> CIpxHost {
        let log = self.log();
        // IPX logs through the log options it is given: a Highs object's
        // (highsLogUser) or this handle's
        let (log_options, ipx_log_fn): (*const c_void, unsafe extern "C" fn(*const c_void, *const c_char)) =
            match &self.host {
                Some(c) => (c.log_options, ipx_log_cpp),
                None => (&self.head as *const LogHead as *const c_void, ipx_log),
            };
        let options = self.opts.ipx(log, log_options);
        CIpxHost {
            ctx: self as *mut LpHandle as *mut c_void,
            timer_read: ipx_timer_read,
            resize: no_resize,
            hooks: crate::ipx::Hooks {
                log: Some(ipx_log_fn),
                print: Some(ipx_print),
                task_interrupt: Some(ipx_task_interrupt),
                user_interrupt: Some(ipx_user_interrupt),
                ctx: self as *mut LpHandle as *mut c_void,
            },
            lp: self.model.view(),
            options,
            info: std::ptr::null_mut(),
            model_status: std::ptr::null_mut(),
            value_valid: std::ptr::null_mut(),
            dual_valid: std::ptr::null_mut(),
            basis_valid: std::ptr::null_mut(),
            basis_useful: std::ptr::null_mut(),
        }
    }
}

/// HEkk::updateStatus' LpAction::kNewBasis
const LP_NEW_BASIS: i32 = 3;

fn none_u8() -> RsMut<u8> {
    RsMut { ptr: std::ptr::null_mut(), len: 0 }
}

fn empty_rs_vec<T>() -> super::ffi::RsVec<T> {
    super::ffi::RsVec::null()
}

/// highs::parallel::num_threads() of this thread's scheduler
pub fn num_threads() -> i32 {
    let d = crate::parallel::this_worker_deque();
    if d.is_null() {
        return 1;
    }
    // SAFETY: the thread's deque
    unsafe { (*d).num_workers() }
}

/// dataSize of an index collection
fn interface_data_size(ic: &IndexCollection) -> i32 {
    if ic.is_interval {
        ic.to - ic.from + 1
    } else if ic.is_set {
        ic.set_num_entries
    } else {
        ic.dimension
    }
}

fn mask_collection(dimension: i32, mask: &[i32]) -> IndexCollection<'_> {
    IndexCollection {
        dimension,
        is_interval: false,
        from: -1,
        to: -2,
        is_set: false,
        set_num_entries: -1,
        set: &[],
        is_mask: true,
        mask,
    }
}

// ---- The interfaces' host (IfaceHost)

struct Iface(*mut LpHandle);

impl Iface {
    #[allow(clippy::mut_from_ref)]
    fn h(&self) -> &mut LpHandle {
        // SAFETY: the handle outlives the interface call; the interface
        // holds the model and basis, which the host functions do not touch
        unsafe { &mut *self.0 }
    }
}

impl IfaceHost for Iface {
    fn lps(&self) -> &mut LpSolver {
        &mut self.h().lps
    }
    fn names_resize(&mut self, _cols: bool, _num: i32) {}
    fn names_delete(&mut self, _cols: bool, _kept: &[i32], _new_num: i32) {}
    fn names_hash_clear(&mut self, _cols: bool) {}
    fn invalidate(&mut self, what: Invalidate) {
        let h = self.h();
        match what {
            Invalidate::StatusSolutionAndInfo => {
                h.invalidate_status_and_info();
                h.invalidate_solution();
            }
            Invalidate::StatusAndInfo => h.invalidate_status_and_info(),
            Invalidate::Status | Invalidate::StatusNotset => h.lps.run.model_status = MS_NOTSET,
        }
    }
    fn feasible_wrt_bounds(&self, columns: bool) -> bool {
        let h = self.h();
        let r = &h.lps.run;
        if r.info.primal_solution_status != SOLUTION_STATUS_FEASIBLE {
            return false;
        }
        let m = &h.model;
        let (v, l, u) = if columns {
            (&r.solution.col_value, &m.col_lower, &m.col_upper)
        } else {
            (&r.solution.row_value, &m.row_lower, &m.row_upper)
        };
        super::edit::feasible_wrt_bounds(v, l, u, h.opts.primal_feasibility_tolerance)
    }
    fn ekk_nla_lp(&mut self) {
        self.h().set_nla_model();
    }
    fn ekk_clear_shell(&mut self) {
        self.h().clear_shell();
    }
    fn hessian_complete(&mut self, _num_col: i32) {}
    fn hessian_delete_cols(&mut self, _ic: &IndexCollection) {}
}

// ---- formSimplexLpBasisAndFactor's host

/// The views of a basis that formSimplexLpBasisAndFactor reads and
/// changes (a Rust Basis or a C++ HighsBasis): mirrored by RsFormBasis in
/// highs/lp_data/HighsLpHandle.h
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FormBasis {
    valid: bool,
    useful: bool,
    alien: *mut bool,
    col_status: RsMut<u8>,
    row_status: RsMut<u8>,
    debug_id: i32,
    debug_update_count: i32,
    origin: RsMut<u8>,
}

struct FormCtx {
    h: *mut LpHandle,
    b: FormBasis,
}

unsafe extern "C" fn form_op(ctx: *mut c_void, code: i32, arg: i32, _out: *mut c_void) -> i32 {
    let c = &*(ctx as *const FormCtx);
    let h = &mut *c.h;
    match code {
        // HEkk::moveLp's checks after the copy
        4 => {
            h.moved_lp();
            0
        }
        // HEkk::setBasis(basis)
        6 => {
            let b = c.b;
            let env = h.env();
            let env = h.lps.env_of(&env);
            let (nc, nr) = (env.lp.num_col as usize, env.lp.num_row as usize);
            let col = std::slice::from_raw_parts(b.col_status.ptr, b.col_status.len);
            let row = std::slice::from_raw_parts(b.row_status.ptr, b.row_status.len);
            let origin = String::from_utf8_lossy(if b.origin.len == 0 {
                &[][..]
            } else {
                std::slice::from_raw_parts(b.origin.ptr, b.origin.len)
            })
            .into_owned();
            h.lps.set_basis(&env, &col[..nc], &row[..nr], b.debug_id, b.debug_update_count, &origin);
            0
        }
        // HEkk::initialiseSimplexLpBasisAndFactor(arg)
        7 => {
            // snapshotFactorLog and setNlaEngineLp
            if !h.lps.sh.status.has_nla {
                h.shell.factor_log = h.factor_log_copy();
            }
            h.clear_nla_lp();
            h.lps.set_nla_rust(true);
            let env = h.env();
            let env = h.lps.env_of(&env);
            let s = h.lps.initialise_simplex_lp_basis_and_factor(&env, arg != 0);
            h.take_rust_out();
            s
        }
        // The model takes the engine LP's scale
        8 => {
            h.shell_op(10, 0, std::ptr::null_mut());
            0
        }
        _ => unreachable!("form basis op {code}"),
    }
}

// ---- The host functions of the run, the simplex and IPX

/// The run's op on a handle (its context)
pub(super) const HANDLE_OP: unsafe extern "C" fn(*mut c_void, i32, i64, *mut c_void, *const u8, usize) -> i64 = handle_op;

unsafe extern "C" fn handle_op(ctx: *mut c_void, op: i32, arg: i64, p: *mut c_void, msg: *const u8, len: usize) -> i64 {
    let h = &mut *(ctx as *mut LpHandle);
    let m = if len == 0 { &[][..] } else { std::slice::from_raw_parts(msg, len) };
    h.op(op, arg, p, m)
}

unsafe extern "C" fn handle_clock(ctx: *mut c_void, which: i32, action: i32) -> f64 {
    let t = &mut (*(ctx as *mut LpHandle)).timer;
    let c = which as usize;
    match action {
        0 => t.read(c),
        1 => {
            t.start(c);
            0.0
        }
        2 => {
            t.stop(c);
            0.0
        }
        _ => t.running(c) as i32 as f64,
    }
}

unsafe extern "C" fn no_solve_op(_: *mut c_void, _: i32, _: *const u8, _: usize) -> i64 {
    unreachable!("set by the run")
}

unsafe extern "C" fn no_app_op(_: *mut c_void, _: i32, _: i64, _: *mut c_void) -> i64 {
    unreachable!("set by the run")
}

unsafe extern "C" fn no_resize(_: *mut c_void, _: bool, _: *mut super::basis::COut) {
    unreachable!("set by the run")
}

fn handle_of<'a>(ctx: *mut c_void) -> &'a mut LpHandle {
    // SAFETY: the host functions' context is the handle
    unsafe { &mut *(ctx as *mut LpHandle) }
}

extern "C" fn host_log(ctx: *mut c_void, channel: i32, t: i32, msg: *const c_char) {
    let h = handle_of(ctx);
    // SAFETY: a NUL-terminated message
    let s = unsafe { CStr::from_ptr(msg) }.to_bytes();
    let t = match t {
        2 => LogType::Detailed,
        3 => LogType::Verbose,
        4 => LogType::Warning,
        5 => LogType::Error,
        _ => LogType::Info,
    };
    let text = String::from_utf8_lossy(s);
    match channel {
        0 => h.log().user(t, &text),
        1 => h.log().dev(t, &text),
        2 => h.factor_log().dev(t, &text),
        _ => crate::io::log::c_stdout_flush(s),
    }
}

extern "C" fn host_timer_read(ctx: *mut c_void) -> f64 {
    handle_of(ctx).clock_read()
}

/// The user interrupt of HEkk::bailout: the race's flag, or a Highs
/// object's simplex interrupt callback
extern "C" fn host_interrupt(ctx: *mut c_void) -> bool {
    let h = handle_of(ctx);
    if let Some(c) = &h.host {
        // SAFETY: the Highs object's callback
        return unsafe { (c.simplex_interrupt)(c.ctx, h.lps.sh.iteration_count) };
    }
    let Some(i) = h.interrupt.filter(|i| i.simplex) else { return false };
    // SAFETY: the flag outlives the solve
    if unsafe { (*i.flag).load(Ordering::Relaxed) } == 2 {
        h.log().dev(LogType::Info, "User interrupt\n");
        return true;
    }
    false
}

/// debugDualChuzcFailQuad0 (kind 1) / Quad1: dev log reports
extern "C" fn host_chuzc_fail(
    ctx: *mut c_void,
    kind: i32,
    work_count: i32,
    work_data: *const WorkPair,
    select_theta: f64,
    remain_theta: f64,
) {
    let h = handle_of(ctx);
    let log = h.log();
    let data = if work_count <= 0 {
        &[][..]
    } else {
        // SAFETY: the simplex's work data of work_count entries
        unsafe { std::slice::from_raw_parts(work_data, work_count as usize) }
    };
    let mut work_data_norm = 0.0f64;
    for p in data {
        work_data_norm = p.value.mul_add_c(p.value, work_data_norm);
    }
    let work_data_norm = work_data_norm.sqrt();
    let mut work_dual_norm = 0.0f64;
    for &v in &h.lps.work_dual {
        work_dual_norm = v.mul_add_c(v, work_dual_norm);
    }
    let work_dual_norm = work_dual_norm.sqrt();
    if kind == 1 {
        crate::log_dev!(log, LogType::Info, "DualChuzC:     No change in loop 2 so return error\n");
        crate::log_dev!(
            log,
            LogType::Info,
            "DualChuzC:     workCount = %d; selectTheta=%g; remainTheta=%g\n",
            work_count,
            select_theta,
            remain_theta
        );
    } else {
        crate::log_dev!(log, LogType::Info, "DualChuzC:     No group identified in quad search so return error\n");
        crate::log_dev!(log, LogType::Info, "DualChuzC:     workCount = %d; selectTheta=%g\n", work_count, select_theta);
    }
    crate::log_dev!(
        log,
        LogType::Info,
        "DualChuzC:     workDataNorm = %g; workDualNorm = %g\n",
        work_data_norm,
        work_dual_norm
    );
}

unsafe extern "C" fn ipx_timer_read(ctx: *mut c_void) -> f64 {
    handle_of(ctx).clock_read()
}

/// IPX's log hook: highsLogUser(kInfo, "%s", msg) on the handle's log
unsafe extern "C" fn ipx_log(log_options: *const c_void, msg: *const c_char) {
    let log = Log { opts: log_options, log: Some(handle_log) };
    log.user(LogType::Info, &CStr::from_ptr(msg).to_string_lossy());
}

/// IPX's log hook on a Highs object's log options
unsafe extern "C" fn ipx_log_cpp(log_options: *const c_void, msg: *const c_char) {
    let log = Log { opts: log_options, log: Some(crate::io::log::highs_rs_log) };
    log.user(LogType::Info, &CStr::from_ptr(msg).to_string_lossy());
}

unsafe extern "C" fn ipx_print(msg: *const c_char) {
    crate::io::log::c_stdout(CStr::from_ptr(msg).to_bytes());
}

/// checkInterrupt of the thread's deque
unsafe extern "C" fn ipx_task_interrupt(_: *mut c_void) -> crate::ipx::Int {
    let d = crate::parallel::this_worker_deque();
    (!d.is_null() && (*d).check_interrupt()) as crate::ipx::Int
}

/// The IPM interrupt of the race, or a Highs object's IPM interrupt
/// callback
unsafe extern "C" fn ipx_user_interrupt(ctx: *mut c_void, iter: crate::ipx::Int) -> crate::ipx::Int {
    let h = handle_of(ctx);
    if let Some(c) = &h.host {
        return (c.ipm_interrupt)(c.ctx, iter);
    }
    match h.interrupt {
        Some(i) if i.ipm => ((*i.flag).load(Ordering::Relaxed) != 0) as crate::ipx::Int,
        _ => 0,
    }
}

impl LpHandle {
    /// HighsLpRelaxation::optimizeRacingIpx: the LP, which has no basis,
    /// solved by the dual simplex here and by IPX with crossover on a
    /// helper thread; the first to finish stops the other. If IPX wins,
    /// the dual simplex goes on from its basis, and its iterations so far
    /// are added to `extra_iterations`. Returns the status and whether IPX
    /// won.
    pub fn optimize_racing_ipx(&mut self, seed: i32, extra_iterations: &mut i64) -> (Status, bool) {
        let winner = std::sync::Arc::new(AtomicI32::new(0));
        let lp = self.model.clone();
        let time_limit = self.opts.time_limit;
        let w = std::sync::Arc::clone(&winner);
        let helper = std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(move || {
                crate::parallel::initialize_thread(1);
                let mut ipx = LpHandle::new();
                ipx.set_option("output_flag", OptValue::Bool(false));
                ipx.set_option("solver", OptValue::Str(b"ipx"));
                ipx.set_option("run_crossover", OptValue::Str(b"on"));
                ipx.set_option("threads", OptValue::Int(1));
                ipx.set_option("time_limit", OptValue::Double(time_limit));
                ipx.set_option("random_seed", OptValue::Int(seed));
                ipx.pass_model(lp);
                ipx.interrupt = Some(Interrupt { flag: &*w, simplex: false, ipm: true });
                let mut basis = None;
                if ipx.run_lp() == Status::Ok
                    && ipx.model_status() == super::run::MS_OPTIMAL
                    && ipx.basis().b.valid
                    && w.compare_exchange(0, 2, Ordering::SeqCst, Ordering::SeqCst).is_ok()
                {
                    basis = Some(ipx.basis().clone());
                }
                basis
            })
            .expect("the IPX race's helper thread");
        self.interrupt = Some(Interrupt { flag: &*winner, simplex: true, ipm: false });
        let callstatus = self.optimize_lp();
        self.interrupt = None;
        // the dual simplex is only interrupted once IPX has won
        let ipx_won = self.model_status() == super::run::MS_INTERRUPT;
        if !ipx_won {
            let _ = winner.compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst);
        }
        let basis = helper.join().expect("the IPX race's helper");
        if !ipx_won {
            return (callstatus, false);
        }
        *extra_iterations += self.info().simplex_iteration_count.max(0) as i64;
        self.set_basis(basis.expect("IPX's basis"), "HighsLpRelaxation::optimizeRacingIpx");
        (self.optimize_lp(), true)
    }

    /// HighsLpRelaxation::ipmBasisAfterIterationLimit (HiPO is not in this
    /// build): the basis of IPX (presolved unless only the root may be),
    /// limited to 200 IPM iterations and this LP's simplex iterations, set
    /// as this LP's
    pub fn ipm_basis_after_iteration_limit(&mut self, use_presolve: bool, profiling: *mut c_void) {
        let mut ipm = LpHandle::new();
        ipm.profiling = profiling;
        ipm.set_option("output_flag", OptValue::Bool(false));
        ipm.set_option("presolve", OptValue::Str(if use_presolve { b"choose" } else { b"off" }));
        ipm.set_option("solver", OptValue::Str(b"ipx"));
        ipm.set_option("ipm_iteration_limit", OptValue::Int(200));
        ipm.pass_model(self.model.clone());
        ipm.set_option("simplex_iteration_limit", OptValue::Int(self.info().simplex_iteration_count));
        ipm.optimize_lp();
        let basis = ipm.basis().clone();
        self.set_basis(basis, "HighsLpRelaxation::run IPM basis");
    }
}

impl Default for Box<LpHandle> {
    fn default() -> Self {
        LpHandle::new()
    }
}

/// The C++ interface of a handle (HighsLpRelaxation.h)
pub mod ffi {
    use super::super::options::COptionRecord;
    use super::*;
    use crate::ffi::{sl, sl_mut, CHVec};

    /// What C++ reads of a handle (valid until the handle changes):
    /// mirrored by highs_rs::LphView in HighsLpRelaxation.h
    #[repr(C)]
    pub struct LphView {
        pub num_col: i32,
        pub num_row: i32,
        pub num_nz: i32,
        pub model_status: i32,
        pub col_cost: *const f64,
        pub col_lower: *const f64,
        pub col_upper: *const f64,
        pub row_lower: *const f64,
        pub row_upper: *const f64,
        pub a_start: *const i32,
        pub a_index: *const i32,
        pub a_value: *const f64,
        pub col_value: *const f64,
        pub col_dual: *const f64,
        pub row_value: *const f64,
        pub row_dual: *const f64,
        pub n_col_value: i32,
        pub n_col_dual: i32,
        pub n_row_value: i32,
        pub n_row_dual: i32,
        pub col_status: *const u8,
        pub row_status: *const u8,
        pub n_col_status: i32,
        pub n_row_status: i32,
        pub value_valid: bool,
        pub dual_valid: bool,
        pub basis_valid: bool,
        pub info: *const Info,
    }

    /// A HighsBasis' fields (C++ copies the statuses)
    #[repr(C)]
    pub struct LphBasis {
        pub valid: bool,
        pub alien: bool,
        pub useful: bool,
        pub was_alien: bool,
        pub debug_id: i32,
        pub debug_update_count: i32,
        pub col_status: *const u8,
        pub n_col: i32,
        pub row_status: *const u8,
        pub n_row: i32,
        pub origin: *const u8,
        pub origin_len: usize,
    }

    fn h<'a>(p: *mut LpHandle) -> &'a mut LpHandle {
        // SAFETY: a handle of highs_rs_lph_new or of an LP relaxation,
        // used by one thread at a time
        unsafe { &mut *p }
    }

    unsafe fn text<'a>(p: *const u8, n: usize) -> &'a str {
        std::str::from_utf8_unchecked(sl(p, n as i32))
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_new() -> *mut LpHandle {
        Box::into_raw(LpHandle::new())
    }

    /// # Safety
    /// a pointer of highs_rs_lph_new, or null
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_free(p: *mut LpHandle) {
        if !p.is_null() {
            drop(Box::from_raw(p));
        }
    }

    /// # Safety
    /// a live handle; the output writable
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_view(p: *mut LpHandle, v: *mut LphView) {
        let h = h(p);
        let m = &h.model;
        let r = &h.lps.run;
        let (s, b) = (&r.solution, &r.basis.b);
        let nc = m.num_col as usize;
        let num_nz = if nc > 0 { m.a.start.get(nc).copied().unwrap_or(0) } else { 0 };
        let out = LphView {
            num_col: m.num_col,
            num_row: m.num_row,
            num_nz,
            model_status: r.model_status,
            col_cost: m.col_cost.as_ptr(),
            col_lower: m.col_lower.as_ptr(),
            col_upper: m.col_upper.as_ptr(),
            row_lower: m.row_lower.as_ptr(),
            row_upper: m.row_upper.as_ptr(),
            a_start: m.a.start.as_ptr(),
            a_index: m.a.index.as_ptr(),
            a_value: m.a.value.as_ptr(),
            col_value: s.col_value.as_ptr(),
            col_dual: s.col_dual.as_ptr(),
            row_value: s.row_value.as_ptr(),
            row_dual: s.row_dual.as_ptr(),
            n_col_value: s.col_value.len() as i32,
            n_col_dual: s.col_dual.len() as i32,
            n_row_value: s.row_value.len() as i32,
            n_row_dual: s.row_dual.len() as i32,
            col_status: b.col_status.as_ptr(),
            row_status: b.row_status.as_ptr(),
            n_col_status: b.col_status.len() as i32,
            n_row_status: b.row_status.len() as i32,
            value_valid: s.value_valid,
            dual_valid: s.dual_valid,
            basis_valid: b.valid,
            info: &r.info,
        };
        // SAFETY: C++'s view to fill
        unsafe { v.write(out) };
    }

    /// # Safety
    /// a live handle; the output writable
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_basis(p: *mut LpHandle, out: *mut LphBasis) {
        let b = &h(p).lps.run.basis;
        let v = LphBasis {
            valid: b.b.valid,
            alien: b.b.alien,
            useful: b.b.useful,
            was_alien: b.b.was_alien,
            debug_id: b.b.debug_id,
            debug_update_count: b.b.debug_update_count,
            col_status: b.b.col_status.as_ptr(),
            n_col: b.b.col_status.len() as i32,
            row_status: b.b.row_status.as_ptr(),
            n_row: b.b.row_status.len() as i32,
            origin: b.origin.as_ptr(),
            origin_len: b.origin.len(),
        };
        // SAFETY: C++'s struct to fill
        unsafe { out.write(v) };
    }

    /// Highs::setBasis(basis, origin) of a C++ HighsBasis
    ///
    /// # Safety
    /// The basis' arrays and strings valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_set_basis(
        p: *mut LpHandle,
        b: *const LphBasis,
        origin: *const u8,
        origin_len: usize,
    ) -> i32 {
        let b = &*b;
        let basis = Basis {
            b: BasisG {
                valid: b.valid,
                alien: b.alien,
                useful: b.useful,
                was_alien: b.was_alien,
                debug_id: b.debug_id,
                debug_update_count: b.debug_update_count,
                col_status: sl(b.col_status, b.n_col).to_vec(),
                row_status: sl(b.row_status, b.n_row).to_vec(),
            },
            origin: String::from_utf8_lossy(sl(b.origin, b.origin_len as i32)).into_owned(),
        };
        h(p).set_basis(basis, text(origin, origin_len)) as i32
    }

    /// setOptionValue: kind 0 bool, 1 int, 2 double, 3 string (`s` of
    /// `len` bytes); false if not set
    ///
    /// # Safety
    /// the name and string of their lengths
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_lph_set_option(
        p: *mut LpHandle,
        name: *const u8,
        name_len: usize,
        kind: i32,
        i: i64,
        d: f64,
        s: *const u8,
        len: usize,
    ) -> bool {
        let v = match kind {
            0 => OptValue::Bool(i != 0),
            1 => OptValue::Int(i as i32),
            2 => OptValue::Double(d),
            _ => OptValue::Str(sl(s, len as i32)),
        };
        h(p).set_option(text(name, name_len), v)
    }

    /// getOptionValue of a double (`d`) or a string (`s`, valid until the
    /// option changes); false for an unknown name or another type
    ///
    /// # Safety
    /// the name of its length
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_get_option(
        p: *mut LpHandle,
        name: *const u8,
        name_len: usize,
        d: *mut f64,
        s: *mut *const u8,
        len: *mut usize,
    ) -> bool {
        match h(p).opts.get(text(name, name_len)) {
            Some(OptValue::Double(x)) => {
                *d = x;
                true
            }
            Some(OptValue::Str(v)) => {
                *s = v.as_ptr();
                *len = v.len();
                true
            }
            _ => false,
        }
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_pass_options(p: *mut LpHandle, from: *mut LpHandle) {
        let from = h(from).opts.clone();
        h(p).pass_options(&from);
    }

    /// passModel of a C++ HighsLp (its view and model name)
    ///
    /// # Safety
    /// the view's arrays valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_pass_model(p: *mut LpHandle, lp: *const CLp, name: *const u8, len: usize) -> i32 {
        let mut m = Lp::default();
        m.import(&*lp, sl(name, len as i32));
        h(p).pass_model(m) as i32
    }

    /// passModel(from's model)
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_pass_model_of(p: *mut LpHandle, from: *mut LpHandle) -> i32 {
        let m = h(from).model.clone();
        h(p).pass_model(m) as i32
    }

    /// The model as a C++ HighsLp view (valid until the model changes)
    /// # Safety
    /// a live handle; the output writable
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_model(p: *mut LpHandle, out: *mut CLp, name: *mut *const u8, len: *mut usize) {
        let m = &mut h(p).model;
        // SAFETY: C++'s outputs
        unsafe {
            out.write(m.view());
            *name = m.model_name.as_ptr();
            *len = m.model_name.len();
        }
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_clear_solver(p: *mut LpHandle) -> i32 {
        h(p).clear_solver() as i32
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_clear_model(p: *mut LpHandle) -> i32 {
        h(p).clear_model() as i32
    }

    /// changeColsBounds(from, to, lower, upper)
    ///
    /// # Safety
    /// to - from + 1 bounds
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_change_col_bounds_interval(
        p: *mut LpHandle,
        from: i32,
        to: i32,
        lower: *const f64,
        upper: *const f64,
    ) -> i32 {
        let n = (to - from + 1).max(0);
        h(p).change_col_bounds_interval(from, to, sl(lower, n), sl(upper, n)) as i32
    }

    /// changeColsBounds(num, set, lower, upper)
    ///
    /// # Safety
    /// `num` entries each
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_change_col_bounds_set(
        p: *mut LpHandle,
        num: i32,
        set: *const i32,
        lower: *const f64,
        upper: *const f64,
    ) -> i32 {
        h(p).change_col_bounds_set(sl(set, num), sl(lower, num), sl(upper, num)) as i32
    }

    /// changeColsCost(mask, cost)
    ///
    /// # Safety
    /// num_col entries each
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_change_col_costs_mask(p: *mut LpHandle, mask: *const i32, cost: *const f64) -> i32 {
        let n = h(p).model.num_col;
        h(p).change_col_costs_mask(sl(mask, n), sl(cost, n)) as i32
    }

    /// addRows
    ///
    /// # Safety
    /// the rows' data of their lengths
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_lph_add_rows(
        p: *mut LpHandle,
        num: i32,
        lower: *const f64,
        upper: *const f64,
        num_nz: i32,
        start: *const i32,
        index: *const i32,
        value: *const f64,
    ) -> i32 {
        let nz = num_nz.max(0);
        let (start, index, value) =
            if nz > 0 { (sl(start, num), sl(index, nz), sl(value, nz)) } else { (&[][..], &[][..], &[][..]) };
        h(p).add_rows(num, sl(lower, num), sl(upper, num), num_nz, start, index, value) as i32
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_delete_rows_interval(p: *mut LpHandle, from: i32, to: i32) -> i32 {
        h(p).delete_rows_interval(from, to) as i32
    }

    /// deleteRows(mask): the mask (num_row entries) becomes the new
    /// indices
    ///
    /// # Safety
    /// num_row entries
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_delete_rows_mask(p: *mut LpHandle, mask: *mut i32) -> i32 {
        let n = h(p).model.num_row;
        h(p).delete_rows_mask(sl_mut(mask, n)) as i32
    }

    /// optimizeLp; `interrupted` a cancelled task's interrupt of IPX (C++
    /// throws HighsTask::Interrupt)
    ///
    /// # Safety
    /// `interrupted` writable
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_optimize_lp(p: *mut LpHandle, interrupted: *mut bool) -> i32 {
        let s = h(p).optimize_lp();
        *interrupted = h(p).task_interrupted;
        s as i32
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_put_iterate(p: *mut LpHandle) -> i32 {
        h(p).put_iterate() as i32
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_get_iterate(p: *mut LpHandle) -> i32 {
        h(p).get_iterate() as i32
    }

    /// getBasisInverseRowSparse into an HVector
    ///
    /// # Safety
    /// a valid view of an HVector of at least num_row entries
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_basis_inverse_row(p: *mut LpHandle, row: i32, v: *mut CHVec) {
        h(p).basis_inverse_row_sparse(row, &mut *v);
    }

    /// getDualRaySparse into an HVector: whether there is a dual ray
    ///
    /// # Safety
    /// as highs_rs_lph_basis_inverse_row
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_dual_ray(p: *mut LpHandle, v: *mut CHVec) -> bool {
        h(p).dual_ray_sparse(&mut *v)
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_has_invert(p: *mut LpHandle) -> bool {
        h(p).has_invert()
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_basic_index(p: *mut LpHandle) -> *const i32 {
        h(p).basic_index().as_ptr()
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_dual_edge_weights(p: *mut LpHandle) -> *const f64 {
        h(p).dual_edge_weights()
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_run_time(p: *mut LpHandle) -> f64 {
        h(p).run_time()
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_set_profiling(p: *mut LpHandle, profiling: *mut c_void) {
        h(p).profiling = profiling;
    }

    /// optimizeRacingIpx: the status, `ipx_won`
    ///
    /// # Safety
    /// the outputs writable
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_race_ipx(p: *mut LpHandle, seed: i32, extra: *mut i64, ipx_won: *mut bool) -> i32 {
        let (s, won) = h(p).optimize_racing_ipx(seed, &mut *extra);
        *ipx_won = won;
        s as i32
    }

    /// Highs::crossover's callCrossover on the handle's model and run data
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_crossover(p: *mut LpHandle) -> i32 {
        h(p).crossover()
    }

    #[no_mangle]
    pub extern "C" fn highs_rs_lph_ipm_basis(p: *mut LpHandle, use_presolve: bool, profiling: *mut c_void) {
        h(p).ipm_basis_after_iteration_limit(use_presolve, profiling);
    }

    // ---- The engine of a C++ Highs object (highs/lp_data/HighsLpHandle.h)

    /// A handle that is a Highs object's simplex engine
    ///
    /// # Safety
    /// The host's context and functions live as long as the handle
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_new_host(host: *const CHost) -> *mut LpHandle {
        Box::into_raw(LpHandle::new_host(*host))
    }

    /// The simplex engine (for the calls that need no environment)
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_lps(p: *mut LpHandle) -> *mut LpSolver {
        &mut *h(p).lps
    }

    /// The option values from the Highs object's records
    ///
    /// # Safety
    /// The records' views valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_sync_options(p: *mut LpHandle, recs: *const COptionRecord, n: usize) {
        let recs = if n == 0 { &[][..] } else { std::slice::from_raw_parts(recs, n) };
        h(p).opts.sync(recs);
    }

    /// The model is a copy of the Highs object's LP (a run on the handle,
    /// or a basis formed for it)
    ///
    /// # Safety
    /// The view's arrays valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_import_model(p: *mut LpHandle, lp: *const CLp, name: *const u8, len: usize) {
        let h = h(p);
        h.model.import(&*lp, sl(name, len as i32));
        h.model_matrix_back = false;
    }

    /// The model's LP data into a C++ LP (Highs::syncLpFromRust), and
    /// the model name's bytes
    ///
    /// # Safety
    /// `lp` a C++ LP's views, the outputs writable
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_export_model(
        p: *mut LpHandle,
        lp: *mut crate::lp_data::lp::CppLp,
        name: *mut *const u8,
        len: *mut usize,
    ) {
        let m = &h(p).model;
        m.export(&mut *lp);
        *name = m.model_name.as_ptr();
        *len = m.model_name.len();
    }

    /// Whether the model's LP data are a C++ LP's (the sync check:
    /// otherwise the differences are printed to stderr)
    ///
    /// # Safety
    /// As highs_rs_lph_import_model
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_model_matches(p: *mut LpHandle, lp: *const CLp, name: *const u8, len: usize) -> bool {
        let d = h(p).model.differences(&*lp, sl(name, len as i32));
        if !d.is_empty() {
            eprintln!("HIGHS_RS_CHECK_SYNC: the engine's model differs from the C++ model in: {d}");
        }
        d.is_empty()
    }

    /// The model's sense (ObjSense) and offset
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_set_model_scalars(p: *mut LpHandle, sense: i32, offset: f64) {
        let m = &mut h(p).model;
        m.sense = sense;
        m.offset = offset;
    }

    /// The model's name
    ///
    /// # Safety
    /// `name` holds `len` bytes
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_set_model_name(p: *mut LpHandle, name: *const u8, len: usize) {
        let m = &mut h(p).model;
        m.model_name.clear();
        m.model_name.extend_from_slice(sl(name, len as i32));
    }

    /// HighsLp::exactResize of the model
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_exact_resize_model(p: *mut LpHandle) {
        h(p).exact_resize_model();
    }

    /// Whether the run rebuilt the model's matrix (an undualized LP's),
    /// which the Highs object takes back with the scale factors
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_take_matrix_back(p: *mut LpHandle) -> bool {
        std::mem::take(&mut h(p).model_matrix_back)
    }

    /// HEkk::clear
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_ekk_clear(p: *mut LpHandle) {
        h(p).ekk_clear();
    }

    /// HEkk::invalidate
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_ekk_invalidate(p: *mut LpHandle) {
        h(p).ekk_invalidate();
    }

    /// HEkk::updateStatus(action)
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_update_status(p: *mut LpHandle, action: i32) {
        let h = h(p);
        if h.lps.update_status(action) {
            h.clear_shell();
        }
    }

    /// What HEkk::clear clears of the shell
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_clear_shell(p: *mut LpHandle) {
        h(p).clear_shell();
    }

    /// HEkk::lp_name_
    ///
    /// # Safety
    /// `len` bytes
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_set_lp_name(p: *mut LpHandle, name: *const u8, len: usize) {
        let n = &mut h(p).shell.lp_name;
        n.clear();
        n.extend_from_slice(sl(name, len as i32));
    }

    /// HEkk::setNlaPointersForLpAndScale(lp) of a C++ LP
    ///
    /// # Safety
    /// The view's scale vectors valid until the next call that sets the
    /// NLA's LP or solves with it
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_set_nla_lp(p: *mut LpHandle, lp: *const CLp) {
        h(p).set_nla_cpp(&*lp);
    }

    /// HEkk::btran (transposed) / ftran, with the simplex NLA's LP set to
    /// `lp` first unless it is null
    ///
    /// # Safety
    /// A valid HVector view; `lp` as for highs_rs_lph_set_nla_lp
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_nla_solve(
        p: *mut LpHandle,
        lp: *const CLp,
        rhs: *mut CHVec,
        expected_density: f64,
        transposed: bool,
    ) {
        let h = h(p);
        if !lp.is_null() {
            h.set_nla_cpp(&*lp);
        }
        let env = h.env();
        let env = h.lps.env_of(&env);
        h.lps.nla_solve(&env, &mut *rhs, expected_density, transposed);
    }

    /// HEkk::computeBasisCondition(lp, exact, report)
    ///
    /// # Safety
    /// The view and name valid
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_basis_condition(
        p: *mut LpHandle,
        lp: *const CLp,
        name: *const u8,
        len: usize,
        exact: bool,
        report: bool,
    ) -> f64 {
        let h = h(p);
        let env = h.env();
        let env = h.lps.env_of(&env);
        let name = String::from_utf8_lossy(sl(name, len as i32)).into_owned();
        h.lps.compute_basis_condition(&env, &*lp, &name, exact, report)
    }

    /// -1 without a simplex NLA, else HEkk::lpFactorRowCompatible
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_factor_row_compatible(p: *mut LpHandle, expected_num_row: i32) -> i32 {
        let h = h(p);
        if !h.lps.sh.status.has_nla {
            return -1;
        }
        let env = h.env();
        let env = h.lps.env_of(&env);
        h.lps.lp_factor_row_compatible(&env, expected_num_row) as i32
    }

    /// formSimplexLpBasisAndFactor of a basis for the model (imported
    /// first): the basis' views are changed in place
    ///
    /// # Safety
    /// The views valid for the call
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_form_basis(p: *mut LpHandle, b: *const FormBasis, only_from_known_basis: bool) -> i32 {
        h(p).form_basis_of(*b, only_from_known_basis) as i32
    }

    /// HEkk::simplex_stats_
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_simplex_stats(p: *mut LpHandle) -> *mut SimplexStats {
        &mut h(p).shell.stats
    }

    /// HEkk::initialiseSimplexStats
    #[no_mangle]
    pub extern "C" fn highs_rs_lph_initialise_simplex_stats(p: *mut LpHandle) {
        let h = h(p);
        h.shell.stats.initialise(h.lps.sh.iteration_count);
    }

    /// HEkk::hot_start_: false if no solve or INVERT set it
    ///
    /// # Safety
    /// The outputs writable; the pointers valid until the next solve
    #[no_mangle]
    #[allow(clippy::too_many_arguments)]
    pub unsafe extern "C" fn highs_rs_lph_hot_start(
        p: *mut LpHandle,
        refactor_use: *mut bool,
        pivot_row: *mut *const i32,
        pivot_var: *mut *const i32,
        pivot_type: *mut *const i8,
        num_pivot: *mut i32,
        build_synthetic_tick: *mut f64,
        nonbasic_move: *mut *const i8,
        num_tot: *mut i32,
    ) -> bool {
        let Some(s) = &h(p).shell.hot_start else { return false };
        *refactor_use = s.refactor_use;
        *pivot_row = s.pivot_row.as_ptr();
        *pivot_var = s.pivot_var.as_ptr();
        *pivot_type = s.pivot_type.as_ptr();
        *num_pivot = s.pivot_row.len().min(s.pivot_var.len()).min(s.pivot_type.len()) as i32;
        *build_synthetic_tick = s.build_synthetic_tick;
        *nonbasic_move = s.nonbasic_move.as_ptr();
        *num_tot = s.nonbasic_move.len() as i32;
        true
    }

    /// HEkk::primal_phase1_dual_
    ///
    /// # Safety
    /// `n` writable; the values valid until the next solve
    #[no_mangle]
    pub unsafe extern "C" fn highs_rs_lph_primal_phase1_dual(p: *mut LpHandle, n: *mut usize) -> *const f64 {
        let d = &h(p).shell.primal_phase1_dual;
        *n = d.len();
        d.as_ptr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timer_accumulates() {
        let mut t = Timer::default();
        assert!(!t.running(0));
        t.start(0);
        assert!(t.running(0));
        t.stop(0);
        assert!(t.read(0) >= 0.0);
    }

    /// min -x - y st x + 2y <= 4, 3x + y <= 6, 0 <= x, y <= 10:
    /// optimal at (1.6, 1.2), objective -2.8
    fn small_lp() -> Lp {
        use super::super::lp::LpG;
        use super::super::sparse::Mat;
        Lp {
            g: LpG {
                num_col: 2,
                num_row: 2,
                col_cost: vec![-1.0, -1.0],
                col_lower: vec![0.0, 0.0],
                col_upper: vec![10.0, 10.0],
                row_lower: vec![-INF, -INF],
                row_upper: vec![4.0, 6.0],
                a: Mat {
                    format: super::super::matrix_format::COLWISE,
                    num_col: 2,
                    num_row: 2,
                    start: vec![0, 2, 4],
                    p_end: vec![],
                    index: vec![0, 1, 0, 1],
                    value: vec![1.0, 3.0, 2.0, 1.0],
                },
                ..Default::default()
            },
            model_name: b"small".to_vec(),
        }
    }

    #[test]
    fn solves_edits_and_resolves() {
        let mut h = LpHandle::new();
        h.set_option("output_flag", OptValue::Bool(false));
        h.set_option("presolve", OptValue::Str(b"off"));
        assert_eq!(h.pass_model(small_lp()), Status::Ok);
        assert_eq!(h.optimize_lp(), Status::Ok);
        assert_eq!(h.model_status(), super::super::run::MS_OPTIMAL);
        assert!((h.info().objective_function_value + 2.8).abs() < 1e-9);
        assert!(h.basis().b.valid && h.has_invert());
        // tighten x <= 1: optimum (1, 1.5), objective -2.5
        assert_eq!(h.change_col_bounds_set(&[0], &[0.0], &[1.0]), Status::Ok);
        assert_eq!(h.optimize_lp(), Status::Ok);
        assert!((h.info().objective_function_value + 2.5).abs() < 1e-9);
        // a cut x + y <= 2
        assert_eq!(h.add_rows(1, &[-INF], &[2.0], 2, &[0], &[0, 1], &[1.0, 1.0]), Status::Ok);
        assert_eq!(h.optimize_lp(), Status::Ok);
        assert!((h.info().objective_function_value + 2.0).abs() < 1e-9);
        // and gone again
        let mut mask = vec![0, 0, 1];
        assert_eq!(h.delete_rows_mask(&mut mask), Status::Ok);
        assert_eq!(mask, vec![0, 1, -1]);
        assert_eq!(h.optimize_lp(), Status::Ok);
        assert!((h.info().objective_function_value + 2.5).abs() < 1e-9);
        // with presolve from scratch
        h.set_option("presolve", OptValue::Str(b"on"));
        assert_eq!(h.clear_solver(), Status::Ok);
        assert_eq!(h.optimize_lp(), Status::Ok);
        assert!((h.info().objective_function_value + 2.5).abs() < 1e-9);
    }

    #[test]
    fn options_default_to_highs_options() {
        let h = LpHandle::new();
        assert_eq!(h.opts.presolve, b"choose");
        assert_eq!(h.model_status(), MS_NOTSET);
        assert!(!h.has_invert());
    }
}
