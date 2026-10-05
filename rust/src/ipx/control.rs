//! Control (control.h/.cc): user parameters, solver output and solver
//! interruption. Output and interrupt checks that belong to HiGHS (logging
//! through highsLogUser, std::cout, the task executor and the user
//! callback) go back to C++ through the function pointers in `Hooks`.

use super::{Int, Parameters, ERROR_TIME_INTERRUPT, ERROR_USER_INTERRUPT};
use std::cell::{Cell, RefCell};
use std::ffi::{c_char, c_void, CString};
use std::io::Write;
use std::time::Instant;

/// Function pointers into C++, all optional (the Rust tests run without)
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Hooks {
    /// highsLogUser(*log_options, HighsLogType::kInfo, "%s", msg)
    pub log: Option<unsafe extern "C" fn(log_options: *const c_void, msg: *const c_char)>,
    /// std::cout << msg
    pub print: Option<unsafe extern "C" fn(msg: *const c_char)>,
    /// HighsTaskExecutor::getThisWorkerDeque()->checkInterrupt(); returns
    /// nonzero if the task was cancelled (the C++ side then rethrows
    /// HighsTask::Interrupt once Solve() returns)
    pub task_interrupt: Option<unsafe extern "C" fn(ctx: *mut c_void) -> Int>,
    /// The kCallbackIpmInterrupt user callback; nonzero requests interrupt
    pub user_interrupt: Option<unsafe extern "C" fn(ctx: *mut c_void, iter: Int) -> Int>,
    /// Passed to the interrupt hooks
    pub ctx: *mut c_void,
}

impl Default for Hooks {
    fn default() -> Self {
        Hooks {
            log: None,
            print: None,
            task_interrupt: None,
            user_interrupt: None,
            ctx: std::ptr::null_mut(),
        }
    }
}

/// ipx::Timer: wall clock since construction or Reset(). Reset() keeps
/// offset + Elapsed() continuous, Elapsed() alone restarts.
struct Timer {
    t0: Cell<Instant>,
    offset: Cell<f64>,
}

impl Timer {
    fn new() -> Self {
        Timer {
            t0: Cell::new(Instant::now()),
            offset: Cell::new(0.0),
        }
    }
    fn elapsed(&self) -> f64 {
        self.t0.get().elapsed().as_secs_f64()
    }
    fn reset(&self) {
        let now = Instant::now();
        let gone = now.duration_since(self.t0.get()).as_secs_f64();
        self.offset.set(self.offset.get() + gone);
        self.t0.set(now);
    }
}

pub struct Control {
    params: RefCell<Parameters>,
    hooks: Cell<Hooks>,
    logfile: RefCell<Option<std::fs::File>>,
    timer: Timer,    // total runtime
    interval: Timer, // time since last interval log
    // set when the HiGHS task running IPX was cancelled: from then on every
    // interrupt check fails and nothing is logged, as C++ would have thrown
    cancelled: Cell<bool>,
}

macro_rules! accessors {
    ($($name:ident: $t:ty),*) => {
        $(pub fn $name(&self) -> $t { self.params.borrow().$name })*
    };
}

impl Control {
    pub fn new() -> Self {
        Control {
            params: RefCell::new(Parameters::default()),
            hooks: Cell::new(Hooks::default()),
            logfile: RefCell::new(None),
            timer: Timer::new(),
            interval: Timer::new(),
            cancelled: Cell::new(false),
        }
    }

    accessors!(dualize: Int, scale: Int, ipm_maxiter: Int, ipm_feasibility_tol: f64,
        ipm_optimality_tol: f64, ipm_drop_primal: f64, ipm_drop_dual: f64, kkt_tol: f64,
        crash_basis: Int, dependency_tol: f64, volume_tol: f64, rows_per_slice: Int,
        maxskip_updates: Int, lu_kernel: Int, lu_pivottol: f64, run_crossover: Int,
        start_crossover_tol: f64, pfeasibility_tol: f64, dfeasibility_tol: f64,
        switchiter: Int, stop_at_switch: Int, update_heuristic: Int, maxpasses: Int,
        run_centring: Int, max_centring_steps: Int, centring_ratio_tolerance: f64,
        centring_ratio_reduction: f64, centring_alpha_scaling: f64, timeless_log: bool,
        analyse_basis_data: bool);

    pub fn parameters(&self) -> Parameters {
        *self.params.borrow()
    }

    pub fn set_parameters(&self, p: Parameters) {
        *self.params.borrow_mut() = p;
    }

    pub fn set_hooks(&self, hooks: Hooks) {
        self.hooks.set(hooks);
    }

    pub fn cancelled(&self) -> bool {
        self.cancelled.get()
    }

    pub fn set_timer_offset(&self, offset: f64) {
        self.timer.offset.set(offset);
    }

    /// Returns IPX_ERROR_* if interrupt is requested, 0 otherwise.
    pub fn interrupt_check(&self, ipm_iteration_count: Int) -> Int {
        let hooks = self.hooks.get();
        if let Some(f) = hooks.task_interrupt {
            // SAFETY: the C++ wrapper sets the hooks with a ctx that is
            // valid for the lifetime of the solver
            if unsafe { f(hooks.ctx) } != 0 {
                self.cancelled.set(true);
            }
        }
        if self.cancelled.get() {
            return ERROR_USER_INTERRUPT;
        }
        let time_limit = self.params.borrow().time_limit;
        if time_limit >= 0.0 && time_limit < self.elapsed() {
            return ERROR_TIME_INTERRUPT;
        }
        if let Some(f) = hooks.user_interrupt {
            // SAFETY: as above
            if unsafe { f(hooks.ctx, ipm_iteration_count) } != 0 {
                return ERROR_USER_INTERRUPT;
            }
        }
        0
    }

    /// printf to stdout (regardless of display)
    pub fn print(&self, s: &str) {
        match self.hooks.get().print {
            Some(f) => {
                let c = cstring(s);
                // SAFETY: c is a valid NUL-terminated string
                unsafe { f(c.as_ptr()) }
            }
            None => print!("{s}"),
        }
    }

    /// Sends text to the log stream (stdout if display, and the logfile)
    fn output(&self, s: &str) {
        if self.cancelled.get() {
            return;
        }
        if self.params.borrow().display != 0 {
            match self.hooks.get().print {
                Some(f) => {
                    let c = cstring(s);
                    // SAFETY: c is a valid NUL-terminated string
                    unsafe { f(c.as_ptr()) }
                }
                None => print!("{s}"),
            }
        }
        if let Some(file) = self.logfile.borrow_mut().as_mut() {
            let _ = file.write_all(s.as_bytes());
        }
    }

    /// Sends text to HiGHS logging or the log stream according to
    /// parameters.highs_logging
    fn emit(&self, s: &str) {
        let p = *self.params.borrow();
        if p.highs_logging {
            assert!(!p.log_options.is_null());
            if self.cancelled.get() {
                return;
            }
            if let Some(f) = self.hooks.get().log {
                let c = cstring(s);
                // SAFETY: log_options points to the HighsLogOptions given
                // with the parameters, c is NUL-terminated
                unsafe { f(p.log_options, c.as_ptr()) }
            }
        } else {
            self.output(s);
        }
    }

    /// hLog
    pub fn log(&self, s: &str) {
        self.emit(s);
        // Reset interval-based logging since something has been logged
        self.interval.reset();
    }

    /// hIntervalLog: logs if >= parameters.print_interval seconds have
    /// passed since the last log or ResetPrintInterval()
    pub fn interval_log(&self, s: &str) {
        let print_interval = self.params.borrow().print_interval;
        if print_interval >= 0.0 && self.interval.elapsed() >= print_interval {
            self.interval.reset();
            self.emit(s);
        }
    }

    pub fn reset_print_interval(&self) {
        self.interval.reset();
    }

    /// Debug(level) evaluates to true
    pub fn debug(&self, level: Int) -> bool {
        self.params.borrow().debug >= level
    }

    /// Debug(level) << s
    pub fn debug_out(&self, level: Int, s: &str) {
        if self.debug(level) {
            self.output(s);
        }
    }

    /// Total runtime
    pub fn elapsed(&self) -> f64 {
        self.timer.offset.get() + self.timer.elapsed()
    }

    pub fn reset_timer(&self) {
        self.timer.reset();
    }

    /// Opens the log file defined in parameters.logfile, if any.
    pub fn open_logfile(&self) {
        let p = self.params.borrow().logfile;
        let mut file = self.logfile.borrow_mut();
        *file = None;
        if !p.is_null() {
            // SAFETY: logfile is NULL or a NUL-terminated string from the
            // caller (as in C++)
            let name = unsafe { std::ffi::CStr::from_ptr(p) };
            if !name.to_bytes().is_empty() {
                if let Ok(name) = name.to_str() {
                    *file = std::fs::OpenOptions::new()
                        .append(true)
                        .create(true)
                        .open(name)
                        .ok();
                }
            }
        }
    }

    pub fn close_logfile(&self) {
        *self.logfile.borrow_mut() = None;
    }
}

fn cstring(s: &str) -> CString {
    CString::new(s.replace('\0', "")).unwrap()
}
