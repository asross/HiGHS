//! crest's Highs object, all Rust (no C++ `Highs`): its engine ([`LpHandle`]
//! on a host that is this object) holds the model, its names, the option
//! values, the solution, basis, info and model status; this object adds
//! what the C++ `Highs` held besides them for the app's steps: the log
//! file, the HighsProfiling, the presolved model, the IIS and the
//! ranging, the model file readers (with the gzip/zlib detection of
//! zstr) and the writers' glue, the option records (option_records.rs),
//! the MIP solver's callback host and the scheduler's start-up.
//!
//! The steps mirror the C++ methods: Highs.cpp (readModel, presolve, run,
//! the file steps of run, writeModel, writeSolution, writeBasis,
//! logHeader, openLogFile, the option calls), HighsRunRust.cpp (topOp and
//! the drivers' steps), HighsIisRust.cpp (the IIS's incumbent),
//! HighsMipHost.cpp, FilereaderMps.cpp / FilereaderLp.cpp,
//! HighsWritersRust.cpp, HMPSIO.cpp's writeModelAsMps, HighsRanging.cpp's
//! getRangingData and HighsAppRust.cpp (the app's objects, [`app_create`]).
//! What only a library caller reaches (user callbacks, the rays' re-solve
//! in a run's file steps) is not here: crest has no user callbacks.

use super::top::{self, CTop};
use super::{CHost, LpHandle, Names};
use crate::io::log::{cfile, highs_rs_log, File, LogOptionsHead};
use crate::io::write::{COut, Model as WModel, Out};
use crate::lp_data::app::AppHost;
use crate::lp_data::ffi::{CLp, RsMut, RsName};
use crate::lp_data::hessian::{self, Hessian, HessianView};
use crate::lp_data::iis::{Args, IisHost, IisInfo, IisState, LpArrays, Op as IisOp};
use crate::lp_data::lp::Lp;
use crate::lp_data::lp_utils::IndexCollection;
use crate::lp_data::model_utils;
use crate::lp_data::option_records::RECORDS;
use crate::lp_data::options::{self, COptionHost, COptionRecord, RsStr};
use crate::lp_data::opts::Opts;
use crate::lp_data::profiling::{self, Profiling};
use crate::lp_data::run::{Op, Run};
use crate::lp_data::{interface, writers, Log, LogType, Status, INF};
use crate::{log_dev, log_user};
use std::collections::HashMap;
use std::ffi::c_void;

// ---- The version and notices (HConfig.h, HighsExternalApi)

pub const VERSION: [i32; 3] = [1, 15, 1];
/// `git describe --always` of the build (build.rs), as CMake's HIGHS_GITHASH
pub const GITHASH: &str = env!("CREST_GITHASH");
/// kHighsCopyrightStatement
pub const COPYRIGHT: &str = "Copyright (c) 2026 under MIT licence terms";
/// HighsExternalApi::thirdPartyNoticeHeader
pub const NOTICE_HEADER: &str = "Includes third-party software components, see THIRD_PARTY_NOTICES.md for full details";
/// The app's getThirdPartyNotice<HighsExtras::appAll> (kept as the C++
/// app prints it)
pub const APP_NOTICE: &str = "Third-party components:\n\n\
key      name     version      license     \n\
-------  -------  -----------  ------------\n\
cli11    CLI11    2.5.0        BSD-3-Clause\n\
pdqsort  pdqsort  git:b1ef26a  Zlib        \n\
zlib     ZLIB     1.2.12       Zlib        \n\
zstr     zstr     1.0.6        MIT         ";
/// CLI11_VERSION (options_cli.rs parses as CLI11 2.5.0)
pub const CLI11_VERSION: &str = "2.5.0";

// HighsFileType
const FILE_MINIMAL: i32 = 0;
const FILE_FULL: i32 = 1;
const FILE_MPS: i32 = 2;
const FILE_LP: i32 = 3;
const FILE_MD: i32 = 4;

// kSolutionStyle*
const STYLE_RAW: i32 = 0;
const STYLE_SPARSE: i32 = 4;

// HighsPresolveStatus
const PS_NOT_PRESOLVED: i32 = -1;

/// A model the object holds besides its own: the read model, the
/// presolved model, the IIS LP
#[derive(Clone)]
pub struct Model {
    pub lp: Lp,
    pub hessian: Hessian,
    pub names: Names,
}

fn empty_hessian() -> Hessian {
    let mut h = Hessian::default();
    h.clear();
    h
}

impl Model {
    /// HighsModel()
    fn empty() -> Model {
        Model { lp: Lp::default(), hessian: empty_hessian(), names: Names::default() }
    }
}

/// HighsIis's data
struct IisStore {
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
    model: Model,
}

impl IisStore {
    /// HighsIis() (clear)
    fn new() -> IisStore {
        IisStore {
            valid: false,
            status: 0,
            strategy: 0,
            col_index: Vec::new(),
            row_index: Vec::new(),
            col_bound: Vec::new(),
            row_bound: Vec::new(),
            col_status: Vec::new(),
            row_status: Vec::new(),
            info: IisInfo {
                num_lp_solved: 0,
                sum_simplex_iteration_counts: 0,
                min_simplex_iteration_count: 0,
                max_simplex_iteration_count: 0,
                sum_simplex_times: 0.0,
                min_simplex_time: 0.0,
                max_simplex_time: 0.0,
                iis_last_disptime: -INF,
                iis_num_disp_lines: 0,
            },
            model: Model::empty(),
        }
    }
}

/// HighsRanging's records (value, objective; in_var and ou_var unread)
#[derive(Default)]
struct Ranging {
    valid: bool,
    rec: [(Vec<f64>, Vec<f64>, Vec<i32>, Vec<i32>); 6],
}

/// The model a write step works on (writeLocalModel's argument)
#[derive(Clone, Copy, PartialEq)]
enum Target {
    Main,
    Presolved,
    Iis,
}

/// The Highs object: see the module comment
pub struct Highs {
    /// The engine: the store of the model, names, options, solution,
    /// basis, info and model status
    pub e: Box<LpHandle>,
    /// options_.log_options (the log flags point into the engine's
    /// options)
    log_head: Box<LogOptionsHead>,
    written_log_header: bool,
    /// profiling_ (a Box's pointer, or null)
    profiling: *mut Profiling,
    /// presolved_model_
    presolved: Model,
    /// presolve_.data_.reduced_lp_ with its names (X_PRESOLVE)
    reduced: Option<Model>,
    iis: IisStore,
    /// The IIS LP of an IIS call, kept aside until its end
    iis_kept: Model,
    ranging: Ranging,
    /// The name lists of the H_NAMES top op
    name_views: [Vec<RsName>; 2],
    /// passModel's model names (taken at X_TAKE_MODEL)
    pending_names: Option<Names>,
    /// The drivers' state between steps: the read model and basis, the
    /// model a write step works on, writeBasis' file
    read_model: Option<Model>,
    read_basis: Option<crate::lp_data::lp_run::Basis>,
    target: Target,
    write_file: *mut File,
    /// The options kSaveOptions saved
    saved_opts: Option<Box<Opts>>,
    max_threads: i32,
}

// ---- Option records and hosts

/// HighsOptions::records over the values of `opts`
fn records(opts: &mut Opts) -> Vec<COptionRecord> {
    let tmpl: Vec<COptionRecord> = RECORDS
        .iter()
        .map(|m| COptionRecord {
            type_: m.type_,
            advanced: m.advanced,
            name: RsStr::of(m.name.as_bytes()),
            description: RsStr::of(m.description.as_bytes()),
            value: std::ptr::null_mut(),
            str_value: RsStr::of(m.str_default.as_bytes()),
            str_default: RsStr::of(m.str_default.as_bytes()),
            bool_default: m.bool_default,
            int_lower: m.int.0,
            int_default: m.int.1,
            int_upper: m.int.2,
            dbl_lower: m.dbl.0,
            dbl_default: m.dbl.1,
            dbl_upper: m.dbl.2,
        })
        .collect();
    // SAFETY: the metadata strings are static; the table is used while
    // `opts` is not moved
    unsafe { opts.records_on(&tmpl) }
}

/// What highsOpenLogFile changes: the log options' stream and the
/// log_file option's value
struct LogFileCtx {
    head: *mut LogOptionsHead,
    log_file: *mut Vec<u8>,
}

/// highsOpenLogFile's stream part: the open stream closed, `name`
/// opened for appending (none if empty or it cannot be opened)
fn reopen_log_stream(head: &mut LogOptionsHead, name: &[u8]) {
    if !head.log_stream.is_null() {
        cfile::flush(head.log_stream);
        // SAFETY: the stream this object opened
        unsafe { cfile::close(head.log_stream) };
        head.log_stream = std::ptr::null_mut();
    }
    if !name.is_empty() {
        head.log_stream = cfile::open(name, "a");
    }
}

unsafe extern "C" fn open_log_file_cb(ctx: *mut c_void, p: *const u8, n: usize) {
    let c = &*(ctx as *const LogFileCtx);
    let name = crate::ffi::sl(p, n as i32).to_vec();
    reopen_log_stream(&mut *c.head, &name);
    // (the value keeps its buffer, which the option table views)
    let v = &mut *c.log_file;
    v.clear();
    v.extend_from_slice(&name);
}

unsafe extern "C" fn write_file_cb(file: *mut c_void, p: *const u8, n: usize) {
    cfile::write(file as *mut File, crate::ffi::sl(p, n as i32));
}

/// The log of a HighsLogOptions head
fn log_of(head: &LogOptionsHead) -> Log {
    Log { opts: head as *const LogOptionsHead as *const c_void, log: Some(highs_rs_log) }
}

/// A HighsLogOptions head on the log flags of `opts` (no stream)
fn head_on(opts: &Opts) -> LogOptionsHead {
    LogOptionsHead {
        log_stream: std::ptr::null_mut(),
        output_flag: &opts.output_flag,
        log_to_console: &opts.log_to_console,
        log_dev_level: &opts.log_dev_level,
        user_log_callback: None,
        user_log_callback_data: std::ptr::null_mut(),
    }
}

/// The option host of a HighsOptions (`ctx` its LogFileCtx), reporting
/// to `log`
fn option_host(log: Log, ctx: &mut LogFileCtx) -> COptionHost {
    COptionHost {
        log,
        ctx: ctx as *mut LogFileCtx as *mut c_void,
        set_string: Some(crate::lp_data::opts::set_string),
        open_log_file: Some(open_log_file_cb),
        write: Some(write_file_cb),
    }
}

// ---- The engine's host

unsafe extern "C" fn host_timer_read(ctx: *mut c_void) -> f64 {
    (*(ctx as *mut Highs)).e.timer.read(0)
}

unsafe extern "C" fn host_simplex_interrupt(_ctx: *mut c_void, _iteration_count: i32) -> bool {
    // No user callback
    false
}

unsafe extern "C" fn host_ipm_interrupt(_ctx: *mut c_void, _count: crate::ipx::Int) -> crate::ipx::Int {
    0
}

unsafe extern "C" fn host_top_op(ctx: *mut c_void, code: i32, arg: i64, p: *mut c_void) -> i64 {
    (*(ctx as *mut Highs)).top_op(code, arg, p)
}

unsafe extern "C" fn host_clock(ctx: *mut c_void, which: i32, action: i32) -> f64 {
    let t = &mut (*(ctx as *mut Highs)).e.timer;
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

// The top ops (top.rs H_*)
const H_PROFILING_BEGIN: i32 = 1;
const H_PROFILING_END: i32 = 2;
const H_PROFILING_RESET: i32 = 3;
const H_MULTITHREADING: i32 = 4;
const H_SUBSOLVER: i32 = 5;
const H_MIP_HOST: i32 = 6;
const H_SMALL_VALUES: i32 = 7;
const H_NAMES: i32 = 8;
const H_PROFILING_SINGLE_BEGIN: i32 = 9;
const H_PROFILING_SINGLE_END: i32 = 10;
const H_LOG_HEADER: i32 = 11;
const H_MATRIX_IMAGES: i32 = 12;
const H_RUN: i32 = 13;
const H_FILE: i32 = 14;

/// The names as a list of views (c_str and strlen: cut at a NUL)
fn name_views(names: &[Vec<u8>]) -> Vec<RsName> {
    names
        .iter()
        .map(|n| {
            let len = n.iter().position(|&c| c == 0).unwrap_or(n.len());
            RsName { ptr: n.as_ptr(), len }
        })
        .collect()
}

/// A string as "%s" prints it (cut at a NUL)
fn cstr(s: &[u8]) -> &[u8] {
    &s[..s.iter().position(|&c| c == 0).unwrap_or(s.len())]
}

fn lossy(s: &[u8]) -> String {
    String::from_utf8_lossy(s).into_owned()
}

impl Highs {
    /// Highs(): an engine with default options on this object as its host
    pub fn new() -> Box<Highs> {
        let mut h = Box::new(Highs {
            e: LpHandle::new(),
            log_head: Box::new(LogOptionsHead {
                log_stream: std::ptr::null_mut(),
                output_flag: std::ptr::null(),
                log_to_console: std::ptr::null(),
                log_dev_level: std::ptr::null(),
                user_log_callback: None,
                user_log_callback_data: std::ptr::null_mut(),
            }),
            written_log_header: false,
            profiling: std::ptr::null_mut(),
            presolved: Model::empty(),
            reduced: None,
            iis: IisStore::new(),
            iis_kept: Model::empty(),
            ranging: Ranging::default(),
            name_views: [Vec::new(), Vec::new()],
            pending_names: None,
            read_model: None,
            read_basis: None,
            target: Target::Main,
            write_file: std::ptr::null_mut(),
            saved_opts: None,
            max_threads: 0,
        });
        let ctx = &mut *h as *mut Highs as *mut c_void;
        let host = CHost {
            ctx,
            log_options: &*h.log_head as *const LogOptionsHead as *const c_void,
            timer_read: host_timer_read,
            simplex_interrupt: host_simplex_interrupt,
            ipm_interrupt: host_ipm_interrupt,
            top: CTop { top_op: host_top_op, clock: host_clock },
        };
        h.e = LpHandle::new_host(host);
        h.e.names = Some(Box::default());
        *h.log_head = head_on(&h.e.opts);
        // The profiling steps of the simplex and PDLP, and the MIP
        // solver's functions of this object
        super::highs_rs_lph_register(profiling::fns::simplex);
        // SAFETY: a static table
        unsafe { crate::mip::host::highs_rs_mip_register(&MIP_FNS) };
        h
    }

    /// options_.log_options
    pub fn log(&self) -> Log {
        log_of(&self.log_head)
    }

    fn names(&mut self) -> &mut Names {
        self.e.names.as_mut().expect("the object's names")
    }

    /// The option host of the object's options (reporting to `log`)
    fn with_options<R>(&mut self, log: Log, f: impl FnOnce(&COptionHost, &[COptionRecord]) -> R) -> R {
        let recs = records(&mut self.e.opts);
        let mut ctx = LogFileCtx { head: &mut *self.log_head, log_file: &mut self.e.opts.log_file };
        let host = option_host(log, &mut ctx);
        f(&host, &recs)
    }

    /// optionChangeAction
    fn option_change_action(&mut self) -> Status {
        if self.iis.valid && self.e.opts.iis_strategy != self.iis.strategy {
            self.iis = IisStore::new();
        }
        Status::Ok
    }

    /// setOptionValue(name, value) of each type
    pub fn set_option_bool(&mut self, name: &[u8], v: bool) -> Status {
        let log = self.log();
        let s = self.with_options(log, |h, r| options::set_bool(&h.log, name, r, v));
        self.option_status(s)
    }
    pub fn set_option_int(&mut self, name: &[u8], v: i32) -> Status {
        let log = self.log();
        let s = self.with_options(log, |h, r| options::set_int(&h.log, name, r, v));
        self.option_status(s)
    }
    pub fn set_option_double(&mut self, name: &[u8], v: f64) -> Status {
        let log = self.log();
        let s = self.with_options(log, |h, r| options::set_double(&h.log, name, r, v));
        self.option_status(s)
    }
    pub fn set_option_string(&mut self, name: &[u8], v: &[u8]) -> Status {
        let log = self.log();
        let s = self.with_options(log, |h, r| options::set_from_string(h, name, r, v));
        self.option_status(s)
    }
    fn option_status(&mut self, s: i32) -> Status {
        if s == options::OK {
            self.option_change_action()
        } else {
            Status::Error
        }
    }

    /// passOptions
    pub fn pass_options(&mut self, from: &mut Opts) -> Status {
        let log = self.log();
        let from_recs = records(from);
        let s = self.with_options(log, |h, r| options::pass_options(h, &from_recs, r));
        self.option_status(s)
    }

    /// writeOptions(filename, report_only_deviations)
    pub fn write_options(&mut self, filename: &[u8], only_deviations: bool) -> Status {
        self.log_header();
        let log = self.log();
        let (file, mut file_type) = match self.open_write_file(filename, "writeOptions") {
            Ok(f) => f,
            Err(s) => return log.interpret(s, Status::Ok, "openWriteFile"),
        };
        if filename.is_empty() {
            file_type = FILE_MINIMAL;
        } else {
            log_user!(log, LogType::Info, "Writing the option values to %s\n", lossy(filename).as_str());
        }
        let stdout = cfile::stdout();
        self.with_options(log, |h, r| {
            options::write_options(h, file as *mut c_void, file == stdout, r, only_deviations, file_type)
        });
        if file != stdout {
            // SAFETY: the file opened above
            unsafe { cfile::close(file) };
        }
        Status::Ok
    }

    /// openWriteFile: stdout for an empty name, else the file (written)
    /// and its type by extension
    fn open_write_file(&self, filename: &[u8], method: &str) -> Result<(*mut File, i32), Status> {
        if filename.is_empty() {
            return Ok((cfile::stdout(), FILE_FULL));
        }
        let file = cfile::open(filename, "w");
        if file.is_null() {
            log_user!(
                self.log(),
                LogType::Error,
                "Cannot open writable file \"%s\" in %s\n",
                lossy(cstr(filename)).as_str(),
                method
            );
            return Err(Status::Error);
        }
        let name = cstr(filename);
        let mut file_type = FILE_FULL;
        if let Some(k) = name.iter().rposition(|&c| c == b'.') {
            if k != 0 {
                file_type = match &name[k + 1..] {
                    b"mps" => FILE_MPS,
                    b"lp" => FILE_LP,
                    b"md" => FILE_MD,
                    _ => FILE_FULL,
                };
            }
        }
        Ok((file, file_type))
    }

    /// logHeader
    pub fn log_header(&mut self) {
        if self.written_log_header || !self.e.opts.output_flag {
            return;
        }
        let log = self.log();
        let githash = if self.e.opts.log_githash { format!(" (git hash: {GITHASH})") } else { String::new() };
        log_user!(
            log,
            LogType::Info,
            "Running HiGHS %d.%d.%d%s: %s\n",
            VERSION[0],
            VERSION[1],
            VERSION[2],
            githash.as_str(),
            COPYRIGHT
        );
        log_user!(log, LogType::Info, "%s\n", NOTICE_HEADER);
        self.written_log_header = true;
    }

    /// openLogFile
    pub fn open_log_file(&mut self, name: &[u8]) {
        reopen_log_stream(&mut self.log_head, name);
        self.e.opts.log_file.clear();
        self.e.opts.log_file.extend_from_slice(name);
    }

    /// closeLogFile
    pub fn close_log_file(&mut self) {
        let f = self.log_head.log_stream;
        if !f.is_null() {
            cfile::flush(f);
            // SAFETY: the stream this object opened
            unsafe { cfile::close(f) };
            self.log_head.log_stream = std::ptr::null_mut();
        }
    }

    /// initializeMultiThreading
    fn initialize_multithreading(&mut self) -> Status {
        let threads = self.e.opts.threads;
        let n = if threads == 0 {
            // (std::thread::hardware_concurrency() + 1) / 2
            (std::thread::available_parallelism().map_or(0, |n| n.get()) as i32 + 1) / 2
        } else {
            threads
        };
        crate::parallel::initialize_thread(n);
        self.max_threads = super::num_threads();
        let log = self.log();
        if threads != 0 && self.max_threads != threads {
            log_user!(
                log,
                LogType::Error,
                "Option 'threads' is set to %d but global scheduler has already been initialized to use %d threads. The previous scheduler instance can be destroyed by calling Highs::resetGlobalScheduler().\n",
                threads,
                self.max_threads
            );
            return Status::Error;
        }
        if self.max_threads <= 0 {
            log_dev!(log, LogType::Error, "max_threads() returns %d\n", self.max_threads);
            return Status::Error;
        }
        log_dev!(log, LogType::Detailed, "Running with %d thread(s)\n", self.max_threads);
        Status::Ok
    }

    // ---- Profiling (HighsProfiling)

    fn initialize_profiling(&mut self, p: *mut Profiling) {
        if p.is_null() || !self.profiling.is_null() {
            return;
        }
        let sub_solver = self.e.opts.log_dev_level > 0;
        let mip = sub_solver && profiling::ANALYSIS_LEVEL_MIP_TIME & self.e.opts.highs_analysis_level != 0;
        // SAFETY: a live Profiling of this object
        unsafe { (*p).initialize(&self.e.timer, sub_solver, mip) };
        self.profiling = p;
    }

    fn clear_profiling(&mut self) {
        if !self.profiling.is_null() {
            // SAFETY: as initialize_profiling
            unsafe { (*self.profiling).clear() };
            self.profiling = std::ptr::null_mut();
        }
    }

    /// The end of a profiling this object made: cleared and freed
    fn end_profiling(&mut self, report: bool) {
        let p = self.profiling;
        if report && !p.is_null() {
            // SAFETY: a live Profiling of this object
            unsafe { (*p).report(&self.log()) };
        }
        self.clear_profiling();
        if !p.is_null() {
            // SAFETY: made by Box::into_raw in top_op
            unsafe { drop(Box::from_raw(p)) };
        }
    }

    // ---- The engine's calls and the copies' updates

    /// What a call on the engine changed of this object's data (the
    /// topOut of HighsRunRust.cpp for what is not the engine's)
    fn take_changes(&mut self) {
        // SAFETY: this object's engine
        let changed = unsafe { top::highs_rs_lph_top_changed(&mut *self.e) };
        if changed & top::X_CLEAR_MODEL != 0 {
            *self.names() = Names::default();
        }
        if changed & top::X_CLEAR_IIS != 0 {
            self.iis = IisStore::new();
            self.ranging.valid = false;
        }
        if changed & (top::X_CLEAR_DERIVED | top::X_CLEAR_PRESOLVE) != 0 {
            self.presolved = Model::empty();
            self.reduced = None;
        }
        if changed & top::X_TAKE_MODEL != 0 {
            let mut names = self.pending_names.take().unwrap_or_default();
            names.origin_name = b"Original".to_vec();
            *self.names() = names;
        }
        if changed & top::X_PRESOLVE_CLEAR != 0 {
            self.reduced = None;
        }
        if changed & top::X_PRESOLVE != 0 {
            self.reduced = Some(self.reduced_model());
        }
        if changed & top::X_PRESOLVED_MODEL != 0 {
            if self.e.top().presolved_which == 0 {
                let hessian = self.e.top().hessian.clone();
                self.presolved =
                    Model { lp: self.e.model.clone(), hessian, names: self.e.names.as_deref().cloned().unwrap_or_default() };
            } else {
                let r = self.reduced.clone().unwrap_or_else(Model::empty);
                self.presolved.lp = r.lp;
                self.presolved.names = r.names;
                let lp = &mut self.presolved.lp;
                lp.g.a.num_col = lp.num_col;
                lp.g.a.num_row = lp.num_row;
            }
        }
    }

    /// presolve_'s reduced LP (presolveExport): the presolve data's LP
    /// with the model's other members, its names following the index
    /// maps
    fn reduced_model(&self) -> Model {
        let d = &self.e.lps.run.presolve;
        let mut lp = d.reduced.clone();
        lp.model_name = self.e.model.model_name.clone();
        lp.g.is_moved = self.e.model.is_moved;
        let mut names = self.e.names.as_deref().cloned().unwrap_or_default();
        if !names.col.is_empty() {
            names.col = d.stack.orig_col_index.iter().map(|&i| names.col[i as usize].clone()).collect();
        }
        if !names.row.is_empty() {
            names.row = d.stack.orig_row_index.iter().map(|&i| names.row[i as usize].clone()).collect();
        }
        if d.prepared {
            names.origin_name = b"Reduced LP".to_vec();
        }
        Model { lp, hessian: self.presolved.hessian.clone(), names }
    }

    /// passModel of a model (its names taken)
    fn pass_model(&mut self, m: Model) -> Status {
        self.pending_names = Some(m.names);
        let s = self.e.top_pass_model(m.lp, m.hessian);
        self.take_changes();
        s
    }

    /// presolve
    pub fn presolve(&mut self) -> Status {
        self.e.task_interrupted = false;
        let s = self.e.with_run(|r| r.presolve());
        self.take_changes();
        s
    }

    /// run
    pub fn run(&mut self) -> Status {
        self.e.task_interrupted = false;
        let s = self.e.top_run();
        self.take_changes();
        s
    }

    /// The model presolve status
    pub fn presolve_status(&self) -> i32 {
        self.e.presolve_status
    }

    // ---- The top level's steps (top.rs H_*)

    fn top_op(&mut self, code: i32, arg: i64, p: *mut c_void) -> i64 {
        // SAFETY (the pointer arguments): as top.rs passes them
        unsafe {
            match code {
                H_PROFILING_BEGIN | H_PROFILING_SINGLE_BEGIN => {
                    let already = !self.profiling.is_null();
                    if !already {
                        let mut prof = Box::new(Profiling::new());
                        prof.multi_threaded = code == H_PROFILING_BEGIN;
                        self.initialize_profiling(Box::into_raw(prof));
                    }
                    *(p as *mut *mut c_void) = self.profiling as *mut c_void;
                    already as i64
                }
                H_PROFILING_END | H_PROFILING_SINGLE_END => {
                    if arg == 0 {
                        self.end_profiling(code == H_PROFILING_END);
                    }
                    0
                }
                H_PROFILING_RESET => {
                    let prof = self.profiling;
                    if !prof.is_null() {
                        self.clear_profiling();
                        self.initialize_profiling(prof);
                    }
                    0
                }
                H_MULTITHREADING => self.initialize_multithreading() as i64,
                H_SUBSOLVER => {
                    let prof = self.profiling;
                    if !prof.is_null() {
                        let k = if (arg >> 1) != 0 { profiling::SUB_SOLVER_QP_ASM } else { profiling::SUB_SOLVER_MIP };
                        if arg & 1 != 0 {
                            Profiling::start(prof, k, false);
                        } else {
                            Profiling::stop(prof, k);
                        }
                    }
                    0
                }
                H_MIP_HOST => {
                    if arg & 1 != 0 {
                        *(p as *mut *mut c_void) = Box::into_raw(self.mip_host(arg & 2 != 0)) as *mut c_void;
                    } else if !p.is_null() {
                        drop(Box::from_raw(p as *mut MipHost));
                    }
                    0
                }
                H_SMALL_VALUES => {
                    let v = &*(p as *const RsMut<f64>);
                    analyse_vector_values(&self.log(), "Small values in matrix", v.get());
                    0
                }
                H_NAMES => {
                    let n = self.e.names.as_deref().expect("the object's names");
                    self.name_views = [name_views(&n.col), name_views(&n.row)];
                    let out = p as *mut RsMut<RsName>;
                    for k in 0..2 {
                        let v = &mut self.name_views[k];
                        *out.add(k) = RsMut { ptr: v.as_mut_ptr(), len: v.len() };
                    }
                    0
                }
                H_LOG_HEADER => {
                    self.log_header();
                    0
                }
                H_MATRIX_IMAGES => {
                    let log = self.log();
                    if self.e.opts.write_matrix_image {
                        let m = &self.e.model;
                        write_matrix_pic(&log, b"LpMatrix", m.num_row, m.num_col, &m.a.start, &m.a.index);
                    }
                    if self.e.opts.write_hessian_image {
                        let h = &self.e.top().hessian;
                        let (dim, start, index) = (h.dim, h.start.clone(), h.index.clone());
                        write_matrix_pic(&log, b"Hessian", dim, dim, &start, &index);
                    }
                    0
                }
                // Highs::run of this object (the rays' re-solve)
                H_RUN => self.run() as i64,
                H_FILE => {
                    self.take_changes();
                    let o = &self.e.opts;
                    let s = match arg {
                        0 => {
                            let f = o.read_solution_file.clone();
                            self.read_solution(&f)
                        }
                        1 => {
                            let f = lossy(&o.read_basis_file);
                            self.file_run(|r| r.read_basis(&f))
                        }
                        2 => {
                            let f = lossy(&o.write_model_file);
                            self.write_local_model(Target::Main, &f)
                        }
                        3 => {
                            let f = lossy(&o.write_iis_model_file);
                            self.write_iis_model(&f)
                        }
                        4 => {
                            let (f, style) = (o.solution_file.clone(), o.write_solution_style);
                            self.write_solution(&f, style)
                        }
                        _ => {
                            let f = lossy(&o.write_basis_file);
                            self.file_run(|r| r.write_basis(&f))
                        }
                    };
                    s as i64
                }
                _ => unreachable!("top op {code}"),
            }
        }
    }

    // ---- The drivers' steps (drivers.rs on this object)

    /// `f` on a run whose op makes the drivers' file steps on this object
    /// and the other steps on the engine
    fn file_run<R>(&mut self, f: impl FnOnce(&Run) -> R) -> R {
        let mut c = self.e.chighs();
        c.ctx = self as *mut Highs as *mut c_void;
        c.op = file_op;
        // SAFETY: c's pointers are into the engine, which outlives the run
        let run = unsafe { Run::new(&c) };
        let r = f(&run);
        self.take_changes();
        r
    }

    /// readModel
    pub fn read_model(&mut self, filename: &str) -> Status {
        let s = self.file_run(|r| r.read_model(filename));
        self.read_model = None;
        s
    }

    /// writePresolvedModel
    pub fn write_presolved_model(&mut self, filename: &str) -> Status {
        self.write_local_model(Target::Presolved, filename)
    }

    /// writeLocalModel of a target
    fn write_local_model(&mut self, target: Target, filename: &str) -> Status {
        self.target = target;
        self.file_run(|r| r.write_local_model(filename))
    }

    /// writeIisModel
    fn write_iis_model(&mut self, filename: &str) -> Status {
        let s = self.get_iis_interface();
        if s == Status::Error {
            return s;
        }
        self.write_local_model(Target::Iis, filename)
    }

    /// The model of a target: its LP, Hessian and names
    fn target_model(&mut self) -> (&mut Lp, &Hessian, &mut Names) {
        match self.target {
            Target::Main => {
                let e = &mut *self.e;
                let t = e.top.as_deref().expect("a Highs object's engine");
                (&mut e.model, &t.hessian, e.names.as_deref_mut().expect("the object's names"))
            }
            Target::Presolved => {
                let m = &mut self.presolved;
                (&mut m.lp, &m.hessian, &mut m.names)
            }
            Target::Iis => {
                let m = &mut self.iis.model;
                (&mut m.lp, &m.hessian, &mut m.names)
            }
        }
    }

    fn file_step(&mut self, which: i32, arg: i64, p: *mut c_void, msg: &[u8]) -> Option<i64> {
        let is = |o: Op| which == o as i32;
        let log = self.log();
        let st = |s: Status| s as i64;
        if is(Op::LogHeader) {
            self.log_header();
            return Some(0);
        }
        if is(Op::ReadModelFile) {
            return Some(self.read_model_file(msg));
        }
        if is(Op::ReadModelPass) {
            if arg == 0 {
                if let Some(m) = self.read_model.as_mut() {
                    m.lp.model_name = msg.to_vec();
                }
                return Some(0);
            }
            let m = self.read_model.take().unwrap_or_else(Model::empty);
            return Some(st(self.pass_model(m)));
        }
        if is(Op::ReadBasis) {
            return Some(match arg {
                0 => {
                    let mut b = self.e.lps.run.basis.clone();
                    let s = self.read_basis_file(msg, &mut b);
                    self.read_basis = Some(b);
                    st(s)
                }
                1 => {
                    let b = &self.read_basis.as_ref().expect("the read basis").b;
                    let m = &self.e.model;
                    let ok = b.col_status.len() == m.num_col as usize
                        && b.row_status.len() == m.num_row as usize
                        && crate::lp_data::basis::basis_consistent(&b.col_status, &b.row_status);
                    ok as i64
                }
                _ => {
                    let mut b = self.read_basis.take().expect("the read basis");
                    b.b.valid = true;
                    b.b.useful = true;
                    self.e.lps.run.basis = b;
                    // newHighsBasis: HEkk::updateStatus(kNewBasis)
                    if self.e.lps.update_status(3) {
                        self.e.clear_shell();
                    }
                    0
                }
            });
        }
        if is(Op::WriteModelPrepare) {
            let (lp, _, names) = self.target_model();
            lp.g.a.num_col = lp.num_col;
            lp.g.a.num_row = lp.num_row;
            let (nc, nr) = (lp.num_col, lp.num_row);
            let s = normalise_names(&log, nc, nr, names, arg as i32);
            lp.g.a.ensure_colwise();
            return Some(st(s));
        }
        if is(Op::WriteModelLpView) {
            let (lp, _, _) = self.target_model();
            // SAFETY: the run's CLp
            unsafe { (p as *mut CLp).write(lp.view()) };
            return Some(0);
        }
        if is(Op::WriteModelCheck) {
            let o = self.e.opts.clone();
            let (lp, hessian, names) = self.target_model();
            return Some(match arg {
                0 => {
                    if hessian.dim > 0 {
                        let _ = &o;
                        st(hessian::assess_hessian_dimensions(&log, hessian))
                    } else {
                        0
                    }
                }
                1 => st(assess_start(&log, &lp.a)),
                2 => st(assess_index_bounds(&log, &lp.a)),
                3 => has_duplicate(&names.col, &mut names.col_hash) as i64,
                _ => has_duplicate(&names.row, &mut names.row_hash) as i64,
            });
        }
        if is(Op::ReportWrittenModel) {
            let (lp, hessian, names) = self.target_model();
            let v = lp.view();
            let (c, r) = (name_views(&names.col), name_views(&names.row));
            crate::lp_data::report::report_lp(&log, &v, &c, &r, LogType::Verbose as i32);
            if hessian.dim != 0 {
                let d = hessian.dim;
                hessian::report_hessian(&log, d, hessian.start[d as usize], &hessian.start, &hessian.index, &hessian.value);
            }
            return Some(0);
        }
        if is(Op::WriteModelFile) {
            let kind = model_utils::file_reader_kind(&log, msg, true);
            if arg == 0 {
                return Some((kind != 0) as i64);
            }
            return Some(st(self.write_model_file(msg, kind)));
        }
        if is(Op::WriteBasis) {
            return Some(match arg {
                0 => match self.open_write_file(msg, "writeBasis") {
                    Ok((f, _)) => {
                        self.write_file = f;
                        0
                    }
                    Err(s) => st(s),
                },
                1 => {
                    let (nc, nr) = (self.e.model.num_col, self.e.model.num_row);
                    st(normalise_names(&log, nc, nr, self.names(), FILE_FULL))
                }
                _ => {
                    let f = std::mem::replace(&mut self.write_file, std::ptr::null_mut());
                    self.write_basis_file(f);
                    if f != cfile::stdout() {
                        // SAFETY: the file opened by step 0
                        unsafe { cfile::close(f) };
                    }
                    0
                }
            });
        }
        None
    }

    // ---- Model files: reading

    /// Filereader::getFilereader and readModelFromFile: the reader's
    /// FilereaderRetcode (-1: no reader); the model in self.read_model
    fn read_model_file(&mut self, filename: &[u8]) -> i64 {
        let log = self.log();
        match model_utils::file_reader_kind(&log, filename, true) {
            1 => self.read_mps(filename) as i64,
            2 => self.read_lp(filename) as i64,
            _ => -1,
        }
    }

    /// FilereaderMps::readModelFromFile (free format)
    fn read_mps(&mut self, filename: &[u8]) -> i32 {
        use crate::io::mps::{read, Status as M};
        let log = self.log();
        let o = &self.e.opts;
        if !o.mps_parser_type_free {
            log_user!(
                log,
                LogType::Warning,
                "The fixed format MPS reader is not available in this build: using the free format reader\n"
            );
        }
        let time_limit = if o.time_limit < INF && o.time_limit > 0.0 { o.time_limit } else { INF };
        log_dev!(log, LogType::Info, "readMPS: Trying to open file %s\n", lossy(filename).as_str());
        let buf = match crate::io::model_file::read(filename) {
            Ok(b) => b,
            Err(e) => {
                log_dev!(log, LogType::Info, "%s", e.as_str());
                return 2;
            }
        };
        let v = read(&buf, time_limit);
        for (kind, text) in &v.messages {
            if *kind == 0 {
                log_dev!(log, LogType::Info, "%s", text.as_str());
            } else {
                log.userf(log_type(*kind), "%s", &[text.as_str().into()]);
            }
        }
        match v.status {
            M::Success => {
                let mut lp = Lp::default();
                lp.num_row = v.num_row as i32;
                lp.num_col = v.num_col as i32;
                lp.sense = if v.maximize { -1 } else { 1 };
                lp.offset = v.offset;
                if !v.a_start.is_empty() {
                    lp.a.start = v.a_start.clone();
                    lp.a.index = v.a_index.clone();
                    lp.a.value = v.a_value.clone();
                }
                lp.col_cost = v.col_cost.clone();
                lp.col_lower = v.col_lower.clone();
                lp.col_upper = v.col_upper.clone();
                lp.row_lower = v.row_lower.clone();
                lp.row_upper = v.row_upper.clone();
                if !v.integrality.is_empty() {
                    lp.integrality = v.integrality.clone();
                }
                let mut hessian = empty_hessian();
                if !v.q_start.is_empty() {
                    hessian = Hessian {
                        dim: v.q_dim as i32,
                        format: hessian::SQUARE,
                        start: v.q_start.clone(),
                        index: v.q_index.clone(),
                        value: v.q_value.clone(),
                    };
                }
                let mut names = Names {
                    col: v.col_names.iter().map(|n| n.to_vec()).collect(),
                    row: v.row_names.iter().map(|n| n.to_vec()).collect(),
                    objective_name: v.objective_name.to_vec(),
                    cost_row_location: v.cost_row_location,
                    ..Names::default()
                };
                if names.objective_name.is_empty() {
                    names.objective_name = objective_name(&lp, hessian.dim, &names).into_bytes();
                }
                lp.a.ensure_colwise();
                let warning = v.warning_issued;
                self.read_model = Some(Model { lp, hessian, names });
                warning as i32
            }
            M::ParserError => 3,
            M::FixedFormat => {
                log_user!(
                    log,
                    LogType::Error,
                    "Free format reader has detected row/col names with spaces: the fixed format MPS reader is not available in this build\n"
                );
                3
            }
            M::Timeout => {
                log_user!(log, LogType::Warning, "Free format reader reached time_limit while parsing the input file\n");
                5
            }
        }
    }

    /// FilereaderLp::readModelFromFile
    fn read_lp(&mut self, filename: &[u8]) -> i32 {
        let log = self.log();
        let buf = match crate::io::model_file::read(filename) {
            Ok(b) => b,
            Err(_) => {
                let f = cfile::open(filename, "r");
                if f.is_null() {
                    return 2;
                }
                // SAFETY: just opened
                unsafe { cfile::close(f) };
                return 3;
            }
        };
        let v = crate::io::lp::read(&buf);
        for (kind, text) in &v.messages {
            if *kind < 0 {
                crate::io::log::c_stdout(text);
            } else {
                log.userf(log_type(*kind), "%s", &[lossy(text).as_str().into()]);
            }
        }
        let status = v.status as i32;
        if status != 3 {
            let mut lp = Lp::default();
            lp.num_row = v.row_lower.len() as i32;
            lp.num_col = v.col_lower.len() as i32;
            lp.sense = if v.maximize { -1 } else { 1 };
            lp.offset = v.offset;
            lp.col_cost = v.col_cost.clone();
            lp.col_lower = v.col_lower.clone();
            lp.col_upper = v.col_upper.clone();
            lp.row_lower = v.row_lower.clone();
            lp.row_upper = v.row_upper.clone();
            lp.integrality = v.integrality.clone();
            lp.a.start = v.a_start.clone();
            lp.a.index = v.a_index.clone();
            lp.a.value = v.a_value.clone();
            let mut hessian = empty_hessian();
            if !v.q_start.is_empty() {
                hessian = Hessian {
                    dim: lp.num_col,
                    format: hessian::SQUARE,
                    start: v.q_start.clone(),
                    index: v.q_index.clone(),
                    value: v.q_value.clone(),
                };
            }
            let names = Names {
                col: v.col_names.iter().map(|n| n.to_vec()).collect(),
                row: v.row_names.iter().map(|n| n.to_vec()).collect(),
                objective_name: v.objective_name.to_vec(),
                ..Names::default()
            };
            lp.a.ensure_colwise();
            self.read_model = Some(Model { lp, hessian, names });
        }
        status
    }

    // ---- Solution and basis files

    /// The readers' view of the model's names
    fn names_host(&mut self) -> crate::lp_data::readers::CNames {
        let (nc, nr) = (self.e.model.num_col as usize, self.e.model.num_row as usize);
        let n = self.names();
        crate::lp_data::readers::CNames {
            ctx: n as *mut Names as *mut c_void,
            op: names_op,
            have_col: n.col.len() == nc,
            have_row: n.row.len() == nr,
        }
    }

    /// readBasisFile into `b`
    fn read_basis_file(&mut self, filename: &[u8], b: &mut crate::lp_data::lp_run::Basis) -> Status {
        let r = crate::lp_data::readers::CRead {
            log: self.log(),
            names: self.names_host(),
            filename: RsMut { ptr: filename.as_ptr() as *mut u8, len: filename.len() },
            basis_valid: &mut b.b.valid,
            col_status: RsMut { ptr: b.b.col_status.as_mut_ptr(), len: b.b.col_status.len() },
            row_status: RsMut { ptr: b.b.row_status.as_mut_ptr(), len: b.b.row_status.len() },
        };
        // SAFETY: the views live for the call
        status_of(unsafe { crate::lp_data::readers::highs_rs_read_basis_file(&r) } as i64)
    }

    /// readSolution(filename, kSolutionStyleRaw)
    fn read_solution(&mut self, filename: &[u8]) -> Status {
        let (nc, nr) = (self.e.model.num_col as usize, self.e.model.num_row as usize);
        let mut sol = crate::lp_data::lp_run::Solution {
            value_valid: false,
            dual_valid: false,
            col_value: vec![0.0; nc],
            col_dual: vec![0.0; nc],
            row_value: vec![0.0; nr],
            row_dual: vec![0.0; nr],
        };
        let mut basis = crate::lp_data::lp_run::Basis::default();
        basis.b.col_status = vec![0; nc];
        basis.b.row_status = vec![0; nr];
        let r = crate::lp_data::readers::CRead {
            log: self.log(),
            names: self.names_host(),
            filename: RsMut { ptr: filename.as_ptr() as *mut u8, len: filename.len() },
            basis_valid: &mut basis.b.valid,
            col_status: RsMut { ptr: basis.b.col_status.as_mut_ptr(), len: nc },
            row_status: RsMut { ptr: basis.b.row_status.as_mut_ptr(), len: nr },
        };
        let a = &mut self.e.model.g.a;
        let c = crate::lp_data::readers::CReadSolution {
            style_sparse: false,
            colwise: a.is_colwise(),
            a_start: RsMut { ptr: a.start.as_mut_ptr(), len: a.start.len() },
            a_index: RsMut { ptr: a.index.as_mut_ptr(), len: a.index.len() },
            a_value: RsMut { ptr: a.value.as_mut_ptr(), len: a.value.len() },
            value_valid: &mut sol.value_valid,
            col_value: RsMut { ptr: sol.col_value.as_mut_ptr(), len: nc },
            col_dual: RsMut { ptr: sol.col_dual.as_mut_ptr(), len: nc },
            row_value: RsMut { ptr: sol.row_value.as_mut_ptr(), len: nr },
            row_dual: RsMut { ptr: sol.row_dual.as_mut_ptr(), len: nr },
        };
        // SAFETY: the views live for the call
        let s = unsafe { crate::lp_data::readers::highs_rs_read_solution_file(&r, &c) };
        if s == -2 {
            return Status::Error;
        }
        let s = status_of(s as i64);
        if s != Status::Ok {
            return s;
        }
        self.e.lps.run.solution = sol;
        self.e.lps.run.basis = basis;
        Status::Ok
    }

    /// A writer's output to `file` (chunked: stdout's text through the
    /// log, as highsFprintfString)
    fn out(&self, file: *mut File, chunked: bool) -> COut {
        COut {
            file: file as *mut c_void,
            log_options: &*self.log_head as *const LogOptionsHead as *const c_void,
            chunked: chunked && file == cfile::stdout(),
            emit: Some(out_emit),
        }
    }

    /// writeBasisFile of the model's basis
    fn write_basis_file(&mut self, file: *mut File) {
        let out = self.out(file, true);
        let names = self.e.names.as_deref().expect("the object's names");
        let m = write_model(&self.e.model, None, names, None);
        let b = &self.e.lps.run.basis.b;
        let basis = writers::Basis { valid: b.valid, col_status: &b.col_status, row_status: &b.row_status };
        writers::write_basis_file(&mut Out::new(Some(&out)), &m, &basis);
    }

    /// writeSolution(filename, style)
    fn write_solution(&mut self, filename: &[u8], style: i32) -> Status {
        let log = self.log();
        let mut return_status = Status::Ok;
        let (file, _) = match self.open_write_file(filename, "writeSolution") {
            Ok(f) => f,
            Err(s) => return log.interpret(s, return_status, "openWriteFile"),
        };
        let (nc, nr) = (self.e.model.num_col, self.e.model.num_row);
        let call = normalise_names(&log, nc, nr, self.names(), FILE_FULL);
        return_status = log.interpret(call, return_status, "normaliseNames");
        if !filename.is_empty() {
            log_user!(log, LogType::Info, "Writing the solution to %s\n", lossy(cstr(filename)).as_str());
        }
        self.write_solution_file(file, style);
        let close = |f: *mut File| {
            if f != cfile::stdout() {
                // SAFETY: the file opened above
                unsafe { cfile::close(f) };
            }
        };
        if style == STYLE_SPARSE {
            close(file);
            return return_status;
        }
        if style == STYLE_RAW {
            cfile::write(file, b"\n# Basis\n");
            self.write_basis_file(file);
        }
        if self.e.opts.ranging == b"on" {
            let is_mip = self.e.model.integrality.iter().any(|&t| t != 0);
            if is_mip || self.e.top().hessian.dim != 0 {
                log_user!(log, LogType::Error, "Cannot determine ranging information for MIP or QP\n");
                close(file);
                return Status::Error;
            }
            let call = self.get_ranging();
            return_status = log.interpret(call, return_status, "getRangingInterface");
            if return_status == Status::Error {
                close(file);
                return return_status;
            }
            cfile::write(file, b"\n# Ranging\n");
            self.write_ranging_file(file, style == 1);
        }
        close(file);
        return_status
    }

    /// writeSolutionFile
    fn write_solution_file(&mut self, file: *mut File, style: i32) {
        let out = self.out(file, true);
        let e = &mut *self.e;
        let (kkt, dual_valid) = {
        let names = e.names.as_deref().expect("the object's names");
        let hessian = &e.top.as_deref().expect("a Highs object's engine").hessian;
        let m = write_model(&e.model, Some(hessian), names, None);
        let r = &e.lps.run;
        let s = &r.solution;
        let model_solution = matches!(style, STYLE_RAW | STYLE_SPARSE | 5 | 6);
        let mut objective = 0.0;
        if model_solution && s.value_valid && r.info.primal_solution_status != 0 {
            let lp = &e.model;
            let mut v = model_utils::lp_objective_cdouble(lp.offset, &lp.col_cost[..lp.num_col as usize], &s.col_value);
            v += hessian::objective_cdouble_value(&hessian_view(hessian), &s.col_value);
            objective = v.into();
        }
        let status_string = crate::simplex::hekk::model_status_string(r.model_status);
        let sf = writers::SolutionFile {
            style,
            info: &r.info,
            model_status: r.model_status,
            model_status_string: status_string.as_bytes(),
            objective,
            num_nz: e.model.a.num_nz() as usize,
            glpsol_cost_row_location: e.opts.glpsol_cost_row_location,
        };
        let sol = writers::Solution {
            value_valid: s.value_valid,
            dual_valid: s.dual_valid,
            col_value: &s.col_value,
            col_dual: &s.col_dual,
            row_value: &s.row_value,
            row_dual: &s.row_dual,
        };
        let b = &r.basis.b;
        let basis = writers::Basis { valid: b.valid, col_status: &b.col_status, row_status: &b.row_status };
        (writers::write_solution_file(&mut Out::new(Some(&out)), &m, &sol, &basis, &sf), s.dual_valid)
        };
        if !kkt {
            return;
        }
        // The KKT report of a pretty Glpsol file
        let mut info = e.lps.run.info;
        let errors = kkt_errors(e, &mut info);
        let is_mip = e.model.integrality.iter().any(|&t| t != 0);
        writers::write_glpsol_kkt(&mut Out::new(Some(&out)), &errors, e.model.num_col, is_mip, dual_valid);
    }

    /// getRangingInterface: getRangingData of the model's simplex solve
    fn get_ranging(&mut self) -> Status {
        use crate::lp_data::ranging::{CRanging, CRecord};
        let log = self.log();
        self.ranging = Ranging::default();
        let e = &mut *self.e;
        let optimal = e.lps.run.model_status == crate::lp_data::run::MS_OPTIMAL;
        let initialised = e.lps.sh.status.initialised_for_solve;
        let lp = e.model.view();
        if optimal && initialised {
            e.lps.unscale_simplex(&lp);
        }
        let (nc, nr) = (e.model.num_col as usize, e.model.num_row as usize);
        let rm = |v: &mut Vec<f64>| RsMut { ptr: v.as_mut_ptr(), len: v.len() };
        let ranging = &mut self.ranging;
        let mut out: [CRecord; 6] = std::array::from_fn(|_| CRecord {
            value: RsMut { ptr: std::ptr::null_mut(), len: 0 },
            objective: RsMut { ptr: std::ptr::null_mut(), len: 0 },
            in_var: RsMut { ptr: std::ptr::null_mut(), len: 0 },
            ou_var: RsMut { ptr: std::ptr::null_mut(), len: 0 },
        });
        if optimal && initialised {
            let sizes = [nc + nr, nc + nr, nc, nc, nr, nr];
            for k in 0..6 {
                let r = &mut ranging.rec[k];
                *r = (vec![0.0; sizes[k]], vec![0.0; sizes[k]], vec![0; sizes[k]], vec![0; sizes[k]]);
                out[k] = CRecord {
                    value: rm(&mut r.0),
                    objective: rm(&mut r.1),
                    in_var: RsMut { ptr: r.2.as_mut_ptr(), len: r.2.len() },
                    ou_var: RsMut { ptr: r.3.as_mut_ptr(), len: r.3.len() },
                };
            }
        }
        let mut f = RangingFtran { e: e as *mut LpHandle, column: crate::hvector::OwnedHVec::new(nr as i32) };
        let objective = e.lps.run.info.objective_function_value;
        let sense = if e.model.sense == -1 { -1 } else { 1 };
        let sl = e.lps.ranging_slices();
        let c = CRanging {
            log,
            optimal,
            initialised_for_solve: initialised,
            num_col: nc as i32,
            num_row: nr as i32,
            sense,
            objective,
            work_value: sl.work_value,
            work_dual: sl.work_dual,
            work_cost: sl.work_cost,
            work_lower: sl.work_lower,
            work_upper: sl.work_upper,
            base_value: sl.base_value,
            base_lower: sl.base_lower,
            base_upper: sl.base_upper,
            nonbasic_flag: sl.nonbasic_flag,
            nonbasic_move: sl.nonbasic_move,
            basic_index: sl.basic_index,
            ftran: ranging_ftran,
            ctx: &mut f as *mut RangingFtran as *mut c_void,
            out,
        };
        // SAFETY: the views live for the call
        let status = status_of(unsafe { crate::lp_data::ranging::highs_rs_get_ranging_data(&c) } as i64);
        if status != Status::Ok {
            return status;
        }
        self.ranging.valid = true;
        if self.e.opts.log_dev_level != 0 {
            self.write_ranging_file(cfile::stdout(), true);
        }
        Status::Ok
    }

    /// writeRangingFile
    fn write_ranging_file(&mut self, file: *mut File, pretty: bool) {
        // No messages: fprintf's to the file
        let out = self.out(file, false);
        let e = &*self.e;
        let names = e.names.as_deref().expect("the object's names");
        let m = write_model(&e.model, None, names, None);
        let r = &e.lps.run;
        let s = &r.solution;
        let sol = writers::Solution {
            value_valid: s.value_valid,
            dual_valid: s.dual_valid,
            col_value: &s.col_value,
            col_dual: &s.col_dual,
            row_value: &s.row_value,
            row_dual: &s.row_dual,
        };
        let b = &r.basis.b;
        let basis = writers::Basis { valid: b.valid, col_status: &b.col_status, row_status: &b.row_status };
        let g = &self.ranging;
        let rec = |k: usize| (&g.rec[k].0[..], &g.rec[k].1[..]);
        let ranging = writers::Ranging { valid: g.valid, rec: [rec(0), rec(1), rec(2), rec(3), rec(4), rec(5)] };
        writers::write_ranging_file(
            &mut Out::new(Some(&out)),
            &m,
            r.info.objective_function_value,
            &basis,
            &sol,
            &ranging,
            pretty,
        );
    }

    /// The writer of a model file kind (1 MPS, 2 LP) for the target
    fn write_model_file(&mut self, filename: &[u8], kind: i32) -> Status {
        let log = self.log();
        let free = self.e.opts.mps_parser_type_free;
        let log_options = &*self.log_head as *const LogOptionsHead as *const c_void;
        let (lp, hessian, names) = self.target_model();
        if names.col.len() != lp.num_col as usize
            || names.row.len() != lp.num_row as usize
            || names.col.iter().chain(&names.row).any(|n| n.is_empty() || n.contains(&b' '))
        {
            // okNames
            return Status::Error;
        }
        if kind == 1 {
            // writeModelAsMps
            let max_name_length =
                names.col.iter().chain(&names.row).map(|n| cstr(n).len()).max().unwrap_or(0) as i32;
            let mut use_free = free;
            let mut warning = false;
            if !free && max_name_length > 8 {
                log_user!(
                    log,
                    LogType::Warning,
                    "Maximum name length is %d so using free format rather than fixed format\n",
                    max_name_length
                );
                use_free = true;
                warning = true;
            }
            let objective = if names.objective_name.is_empty() {
                objective_name(lp, hessian.dim, names).into_bytes()
            } else {
                names.objective_name.clone()
            };
            // writeMps
            log_dev!(log, LogType::Info, "writeMPS: Trying to open file %s\n", lossy(cstr(filename)).as_str());
            let file = cfile::open(filename, "w");
            if file.is_null() {
                log_user!(log, LogType::Error, "Cannot open file %s\n", lossy(cstr(filename)).as_str());
                return Status::Error;
            }
            log_dev!(log, LogType::Info, "writeMPS: Opened file  OK\n");
            if !use_free && max_name_length > 8 {
                log_user!(log, LogType::Error, "Cannot write fixed MPS with names of length (up to) %d\n", max_name_length);
                // SAFETY: just opened
                unsafe { cfile::close(file) };
                return Status::Error;
            }
            let mut m = write_model(lp, Some(hessian), names, None);
            m.objective_name = &objective;
            m.cost_row_location = -1;
            let out = COut { file: file as *mut c_void, log_options, chunked: false, emit: Some(out_emit) };
            crate::io::model_write::write_mps(&mut Out::new(Some(&out)), &m);
            // SAFETY: opened above
            unsafe { cfile::close(file) };
            if warning {
                return Status::Warning;
            }
            return Status::Ok;
        }
        // FilereaderLp::writeModelToFile: the matrix row-wise
        let mut ar = lp.a.clone();
        ar.ensure_rowwise();
        let file = cfile::open(filename, "w");
        let is_qp = hessian.dim != 0;
        let m = write_model(lp, if is_qp { Some(hessian) } else { None }, names, Some(&ar));
        let out = COut { file: file as *mut c_void, log_options, chunked: false, emit: Some(out_emit) };
        crate::io::model_write::write_lp(&mut Out::new(Some(&out)), &m);
        if !file.is_null() {
            // SAFETY: opened above
            unsafe { cfile::close(file) };
        }
        Status::Ok
    }

    // ---- The MIP solver's callback host (HighsMipHost.cpp)

    fn mip_host(&mut self, semi: bool) -> Box<MipHost> {
        let m = &self.e.model;
        let names = self.e.names.as_deref().expect("the object's names");
        let n = m.num_col as usize;
        let mut cost = m.col_cost[..n].to_vec();
        let mut col_names = names.col.clone();
        if semi {
            // withoutSemiVariables' LP: a binary of zero cost per
            // semi-variable, named if the columns are
            let have = names.col.len() == n;
            let mut k = 0;
            for &t in m.integrality.iter().take(n) {
                if t == crate::lp_data::var_type::SEMI_CONTINUOUS || t == crate::lp_data::var_type::SEMI_INTEGER {
                    cost.push(0.0);
                    if have {
                        col_names.push(format!("semi_binary_{k}").into_bytes());
                    }
                    k += 1;
                }
            }
        }
        Box::new(MipHost {
            log_head: &*self.log_head,
            file_name: self.e.opts.mip_improving_solution_file.clone(),
            sparse: self.e.opts.mip_improving_solution_report_sparse,
            offset: m.offset,
            cost,
            names: col_names,
            improving: std::ptr::null_mut(),
        })
    }

    // ---- The IIS (HighsIisRust.cpp): the incumbent's steps

    /// getIisInterface
    fn get_iis_interface(&mut self) -> Status {
        let host = IisHost {
            log: self.log(),
            ctx: self as *mut Highs as *mut c_void,
            op: iis_op,
            lp: iis_lp,
            name: iis_name,
            col_value: iis_col_value,
        };
        // SAFETY: the host's functions take this object
        status_of(unsafe { crate::lp_data::iis::highs_rs_get_iis(&host) } as i64)
    }

    fn iis_step(&mut self, which: i32, a: &Args) -> f64 {
        let is = |o: IisOp| which == o as i32;
        let st = |s: Status| s as i32 as f64;
        // SAFETY: the option name lives for the call
        let name = unsafe { a.s.get() };
        let d = |k: usize, n: i32| -> &[f64] {
            // SAFETY: p[k] holds n values, as iis.rs passes them
            unsafe { crate::ffi::sl(a.p[k] as *const f64, n.max(0)) }
        };
        let i32s = |k: usize, n: i32| -> &[i32] {
            // SAFETY: as d
            unsafe { crate::ffi::sl(a.p[k] as *const i32, n.max(0)) }
        };
        if is(IisOp::LoadIis) {
            let iis = &mut self.iis;
            let v = |x: &mut Vec<i32>| RsMut { ptr: x.as_mut_ptr(), len: x.len() };
            let s = IisState {
                valid: iis.valid,
                status: iis.status,
                strategy: iis.strategy,
                col_index: v(&mut iis.col_index),
                row_index: v(&mut iis.row_index),
                col_bound: v(&mut iis.col_bound),
                row_bound: v(&mut iis.row_bound),
                col_status: v(&mut iis.col_status),
                row_status: v(&mut iis.row_status),
                info: iis.info,
            };
            // SAFETY: p0 is the caller's IisState
            unsafe { (a.p[0] as *mut IisState).write(s) };
            self.iis_kept = iis.model.clone();
            return 0.0;
        }
        if is(IisOp::StoreIis) {
            // SAFETY: p0 is the caller's IisState
            let s = unsafe { &*(a.p[0] as *const IisState) };
            let iis = &mut self.iis;
            iis.valid = s.valid;
            iis.status = s.status;
            iis.strategy = s.strategy;
            // SAFETY: Rust's vectors
            unsafe {
                iis.col_index = s.col_index.get().to_vec();
                iis.row_index = s.row_index.get().to_vec();
                iis.col_bound = s.col_bound.get().to_vec();
                iis.row_bound = s.row_bound.get().to_vec();
                iis.col_status = s.col_status.get().to_vec();
                iis.row_status = s.row_status.get().to_vec();
            }
            iis.info = s.info;
            iis.model = std::mem::replace(&mut self.iis_kept, Model::empty());
            return 0.0;
        }
        if is(IisOp::ClearIisModel) {
            self.iis_kept = Model::empty();
            return 0.0;
        }
        if is(IisOp::SetIisLp) {
            // SAFETY: p0 is the caller's LpArrays
            let x = unsafe { &*(a.p[0] as *const LpArrays) };
            self.iis_kept.lp = self.build_iis_lp(x);
            return 0.0;
        }
        if is(IisOp::SetOptionBool) {
            return st(self.set_option_bool(name, a.i != 0));
        }
        if is(IisOp::SetOptionInt) {
            return st(self.set_option_int(name, a.i));
        }
        if is(IisOp::SetOptionDouble) {
            return st(self.set_option_double(name, a.x));
        }
        if is(IisOp::SetOptionString) {
            // SAFETY: p0 is the value's RsStr
            let v = unsafe { (*(a.p[0] as *const RsStr)).get() };
            return st(self.set_option_string(name, v));
        }
        if is(IisOp::ChangeColBounds) {
            return st(self.e.change_col_bounds_set(&[a.i], &[a.x], &[a.y]));
        }
        if is(IisOp::ChangeRowBounds) {
            return st(self.e.change_row_bounds_set(&[a.i], &[a.x], &[a.y]));
        }
        if is(IisOp::ChangeColsCost) {
            return st(self.e.change_col_costs_interval(a.i, a.j, d(0, a.j - a.i + 1)));
        }
        if is(IisOp::OptimizeModel) {
            let s = self.e.optimize_model_steps(false);
            return st(s);
        }
        if is(IisOp::RunTime) {
            return self.e.timer.read(0);
        }
        if is(IisOp::SimplexIterations) {
            return self.e.lps.run.info.simplex_iteration_count as f64;
        }
        if is(IisOp::ModelStatus) {
            return self.e.lps.run.model_status as f64;
        }
        if is(IisOp::ZeroAllClocks) {
            self.e.timer = super::Timer::default();
            return 0.0;
        }
        if is(IisOp::CallbackActive) || is(IisOp::CallbacksToPropagate) {
            return 0.0;
        }
        if is(IisOp::StopCallback) || is(IisOp::StartCallback) {
            // Only called for an active callback
            return 0.0;
        }
        if is(IisOp::SaveOptions) {
            self.saved_opts = Some(Box::new(self.e.opts.clone()));
            return 0.0;
        }
        if is(IisOp::RestoreOptions) {
            let saved = self.saved_opts.take().expect("the saved options");
            self.e.opts.assign(&saved);
            return 0.0;
        }
        if is(IisOp::EnsureColwise) {
            self.e.model.a.ensure_colwise();
            return 0.0;
        }
        if is(IisOp::GetOption) {
            let o = &self.e.opts;
            return match a.i {
                0 => o.iis_strategy as f64,
                1 => o.primal_feasibility_tolerance,
                2 => o.iis_time_limit,
                3 => o.output_flag as i32 as f64,
                _ => o.log_dev_level as f64,
            };
        }
        if is(IisOp::SetOutputFlag) {
            self.e.opts.output_flag = a.i != 0;
            return 0.0;
        }
        if is(IisOp::InvalidateSolverData) {
            self.e.invalidate_solver_data();
            return 0.0;
        }
        if is(IisOp::PassModelName) {
            self.e.model.model_name = name.to_vec();
            return 0.0;
        }
        if is(IisOp::ChangeColsIntegrality) {
            let n = a.j - a.i + 1;
            // SAFETY: p0 holds the interval's types
            let t = unsafe { crate::ffi::sl(a.p[0] as *const u8, n.max(0)) };
            return st(self.e.change_cols_integrality_interval(a.i, a.j, t));
        }
        if is(IisOp::ChangeColsBounds) {
            let n = a.j - a.i + 1;
            return st(self.e.change_col_bounds_interval(a.i, a.j, d(0, n), d(1, n)));
        }
        if is(IisOp::AddCols) {
            self.log_header();
            let s = self.e.add_cols(a.i, d(0, a.i), d(1, a.i), d(2, a.i), a.j, i32s(3, a.i), i32s(4, a.j), d(5, a.j));
            return st(s);
        }
        if is(IisOp::AddRows) {
            self.log_header();
            let s = self.e.add_rows(a.i, d(0, a.i), d(1, a.i), a.j, i32s(2, a.i), i32s(3, a.j), d(4, a.j));
            return st(s);
        }
        if is(IisOp::PassColName) || is(IisOp::PassRowName) {
            return st(self.pass_name(is(IisOp::PassColName), a.i, name));
        }
        if is(IisOp::DeleteRows) {
            return st(self.e.delete_rows_interval(a.i, a.j));
        }
        if is(IisOp::DeleteCols) {
            return st(self.e.delete_cols_interval(a.i, a.j));
        }
        if is(IisOp::BasisInvalid) {
            self.e.lps.run.basis.b.valid = false;
            return 0.0;
        }
        if is(IisOp::ElasticSolution) {
            let e = &mut *self.e;
            let s = &mut e.lps.run.solution;
            s.row_value.resize(e.model.num_row as usize, 0.0);
            e.model.a.product_quad(&mut s.row_value, &s.col_value);
            s.value_valid = true;
            e.lps.run.info.objective_function_value = a.x;
            e.kkt_failures();
            e.lps.run.info.valid = true;
            return 0.0;
        }
        if is(IisOp::SetModelStatus) {
            self.e.lps.run.model_status = a.i;
            return 0.0;
        }
        if is(IisOp::ObjectiveValue) {
            return self.e.lps.run.info.objective_function_value;
        }
        if is(IisOp::Engine) {
            // SAFETY: p0 is the caller's *mut *mut LpHandle
            unsafe { *(a.p[0] as *mut *mut LpHandle) = &mut *self.e };
            return 0.0;
        }
        unreachable!("IIS op {which} is not the incumbent's")
    }

    /// buildLp: an LP of Rust's arrays, with the incumbent's names of its
    /// columns and rows
    fn build_iis_lp(&mut self, x: &LpArrays) -> Lp {
        // SAFETY: the arrays are iis.rs's vectors
        let g = |v: &RsMut<f64>| unsafe { v.get().to_vec() };
        let gi = |v: &RsMut<i32>| unsafe { v.get().to_vec() };
        let mut lp = Lp::default();
        lp.num_col = x.num_col;
        lp.num_row = x.num_row;
        lp.col_cost = g(&x.col_cost);
        lp.col_lower = g(&x.col_lower);
        lp.col_upper = g(&x.col_upper);
        lp.row_lower = g(&x.row_lower);
        lp.row_upper = g(&x.row_upper);
        lp.a.format = x.format;
        lp.a.num_col = x.num_col;
        lp.a.num_row = x.num_row;
        lp.a.start = gi(&x.start);
        lp.a.index = gi(&x.index);
        lp.a.value = g(&x.value);
        let from = self.e.names.as_deref().expect("the object's names");
        let mut names = Names::default();
        if !from.col.is_empty() {
            names.col = gi(&x.col_map).iter().map(|&k| from.col[k as usize].clone()).collect();
        }
        if !from.row.is_empty() {
            names.row = gi(&x.row_map).iter().map(|&k| from.row[k as usize].clone()).collect();
        }
        if x.model_name != 0 {
            let mut n = self.e.model.model_name.clone();
            n.extend_from_slice(b"_IIS");
            lp.model_name = n;
        }
        self.iis_kept.names = names;
        lp
    }

    /// passColName / passRowName
    fn pass_name(&mut self, col: bool, i: i32, name: &[u8]) -> Status {
        let log = self.log();
        let num = if col { self.e.model.num_col } else { self.e.model.num_row };
        let what = if col { "column" } else { "row" };
        if i < 0 || i >= num {
            log_user!(
                log,
                LogType::Error,
                "Index %d for %s name %s is outside the range [0, num_%s = %d)\n",
                i,
                what,
                lossy(name).as_str(),
                if col { "col" } else { "row" },
                num
            );
            return Status::Error;
        }
        if name.is_empty() {
            log_user!(log, LogType::Error, "Cannot define empty %s names\n", what);
            return Status::Error;
        }
        let n = self.names();
        let (names, hash) = if col { (&mut n.col, &mut n.col_hash) } else { (&mut n.row, &mut n.row_hash) };
        names.resize(num as usize, Vec::new());
        // HighsNameHash::update
        hash.remove(&names[i as usize]);
        match hash.entry(name.to_vec()) {
            std::collections::hash_map::Entry::Occupied(mut o) => {
                o.insert(-1);
            }
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert(i);
            }
        }
        names[i as usize] = name.to_vec();
        Status::Ok
    }
}

impl Drop for Highs {
    fn drop(&mut self) {
        self.close_log_file();
        self.end_profiling(false);
    }
}

// ---- LpHandle's API calls that only crest's object makes

impl LpHandle {
    /// Highs::addCols (after logHeader)
    #[allow(clippy::too_many_arguments)]
    pub fn add_cols(
        &mut self,
        num_new_col: i32,
        cost: &[f64],
        lower: &[f64],
        upper: &[f64],
        num_new_nz: i32,
        start: &[i32],
        index: &[i32],
        value: &[f64],
    ) -> Status {
        self.clear_derived_model_properties();
        let call = if num_new_col < 0 || num_new_nz < 0 {
            Status::Error
        } else if num_new_col == 0 {
            Status::Ok
        } else {
            self.iface(|lp, b, h, o| {
                interface::add_cols(lp, b, h, o, num_new_col, cost, lower, upper, num_new_nz, start, index, value)
            })
        };
        self.edit_return(call, "addCols")
    }

    /// Highs::deleteCols(from, to)
    pub fn delete_cols_interval(&mut self, from: i32, to: i32) -> Status {
        self.clear_derived_model_properties();
        if from < 0 || to >= self.model.num_col {
            return Status::Error;
        }
        let ic = IndexCollection::interval(self.model.num_col, from, to);
        self.iface(|lp, b, h, _o| interface::delete_cols(lp, b, h, &ic));
        self.with_run(|r| r.return_from_highs(Status::Ok))
    }

    /// Highs::changeColsIntegrality(from, to, integrality)
    pub fn change_cols_integrality_interval(&mut self, from: i32, to: i32, integrality: &[u8]) -> Status {
        // clearPresolve
        self.presolve_status = PS_NOT_PRESOLVED;
        self.lps.run.presolve = Default::default();
        if let Some(t) = self.top.as_mut() {
            t.changed |= top::X_CLEAR_PRESOLVE;
        }
        if from < 0 || to >= self.model.num_col {
            return Status::Error;
        }
        let ic = IndexCollection::interval(self.model.num_col, from, to);
        let call = if to - from + 1 <= 0 {
            Status::Ok
        } else {
            self.iface(|lp, _b, h, _o| interface::change_integrality_iface(lp, h, &ic, integrality))
        };
        self.edit_return(call, "changeIntegrality")
    }
}

// ---- Helpers of the steps

fn status_of(v: i64) -> Status {
    match v {
        0 => Status::Ok,
        1 => Status::Warning,
        _ => Status::Error,
    }
}

fn log_type(kind: i32) -> LogType {
    match kind {
        2 => LogType::Detailed,
        3 => LogType::Verbose,
        4 => LogType::Warning,
        5 => LogType::Error,
        _ => LogType::Info,
    }
}

fn hessian_view(h: &Hessian) -> HessianView {
    HessianView { dim: h.dim, format: h.format, start: h.start.as_ptr(), index: h.index.as_ptr(), value: h.value.as_ptr() }
}

/// findModelObjectiveName of a model without one
fn objective_name(lp: &Lp, q_dim: i32, names: &Names) -> String {
    let has_objective = lp.col_cost.iter().take(lp.num_col as usize).any(|&c| c != 0.0) || q_dim != 0;
    let rows = name_views(&names.row);
    model_utils::find_model_objective_name(has_objective, &rows, lp.num_row.max(0) as usize)
}

/// normaliseNames of the columns, then the rows
fn normalise_names(log: &Log, num_col: i32, num_row: i32, n: &mut Names, file: i32) -> Status {
    let mut one = |column: bool, num: i32| {
        let (names, prefix, suffix, hash) = if column {
            (&mut n.col, &mut n.col_prefix, &mut n.col_suffix, &mut n.col_hash)
        } else {
            (&mut n.row, &mut n.row_prefix, &mut n.row_suffix, &mut n.row_hash)
        };
        let views = name_views(names);
        let p = String::from_utf8_lossy(cstr(prefix.as_bytes())).into_owned();
        let r = model_utils::normalise_names(log, column, num.max(0) as usize, &p, *suffix, &views, file);
        if let Some(v) = r.names {
            *names = v;
        }
        if let Some(p) = r.prefix {
            *prefix = p.to_string();
        }
        *suffix = r.suffix;
        hash.clear();
        r.status
    };
    let col = one(true, num_col);
    let row = one(false, num_row);
    if row != Status::Ok {
        return row;
    }
    col
}

/// HighsNameHash::hasDuplicate (the hash is left clear)
fn has_duplicate(names: &[Vec<u8>], hash: &mut HashMap<Vec<u8>, i32>) -> bool {
    hash.clear();
    let mut seen = std::collections::HashSet::new();
    names.iter().any(|n| !seen.insert(n.as_slice()))
}

/// HighsSparseMatrix::assessStart
fn assess_start(log: &Log, a: &crate::lp_data::lp::SparseMatrix) -> Status {
    let num_vec = if a.is_colwise() { a.num_col } else { a.num_row } as usize;
    if a.start[0] != 0 {
        log_user!(log, LogType::Error, "Matrix start[0] = %d, not 0\n", a.start[0]);
        return Status::Error;
    }
    let num_nz = a.num_nz();
    for k in 1..num_vec {
        if a.start[k] < a.start[k - 1] {
            log_user!(log, LogType::Error, "Matrix start[%d] = %d > %d = start[%d]\n", k as i32, a.start[k], a.start[k - 1], k as i32 - 1);
            return Status::Error;
        }
        if a.start[k] > num_nz {
            log_user!(log, LogType::Error, "Matrix start[%d] = %d > %d = number of nonzeros\n", k as i32, a.start[k], num_nz);
            return Status::Error;
        }
    }
    Status::Ok
}

/// HighsSparseMatrix::assessIndexBounds
fn assess_index_bounds(log: &Log, a: &crate::lp_data::lp::SparseMatrix) -> Status {
    let vec_dim = if a.is_colwise() { a.num_row } else { a.num_col };
    for el in 0..a.num_nz().max(0) as usize {
        let i = a.index[el];
        if i < 0 || i >= vec_dim {
            log_user!(log, LogType::Error, "Matrix index[%d] = %d is not in legal range of [0, %d)\n", el as i32, i, vec_dim);
            return Status::Error;
        }
    }
    Status::Ok
}

/// The writers' view of a model (`ar`: the row-wise matrix of the LP
/// writer; no matrix otherwise but for the MPS writer's column-wise one)
fn write_model<'a>(lp: &'a Lp, hessian: Option<&'a Hessian>, names: &'a Names, ar: Option<&'a crate::lp_data::lp::SparseMatrix>) -> WModel<'a> {
    let a = ar.unwrap_or(&lp.a);
    let (q_dim, q_start, q_index, q_value) = match hessian {
        Some(h) => (h.dim.max(0) as usize, &h.start[..], &h.index[..], &h.value[..]),
        None => (0, &[][..], &[][..], &[][..]),
    };
    WModel {
        num_col: lp.num_col as usize,
        num_row: lp.num_row as usize,
        col_cost: &lp.col_cost,
        col_lower: &lp.col_lower,
        col_upper: &lp.col_upper,
        row_lower: &lp.row_lower,
        row_upper: &lp.row_upper,
        a_start: &a.start,
        a_index: &a.index,
        a_value: &a.value,
        sense: lp.sense,
        offset: lp.offset,
        integrality: &lp.integrality,
        q_dim,
        q_start,
        q_index,
        q_value,
        col_names: names.col.iter().map(|n| cstr(n)).collect(),
        row_names: names.row.iter().map(|n| cstr(n)).collect(),
        model_name: cstr(&lp.model_name),
        objective_name: cstr(&names.objective_name),
        cost_row_location: names.cost_row_location,
    }
}

/// A writer's text (kind -1) and messages (0 highsLogDev, else
/// highsLogUser of that type): HighsWritersRust.cpp's rsOutEmit
unsafe extern "C" fn out_emit(out: *const COut, kind: i32, text: *const u8, len: usize) {
    let o = &*out;
    if kind < 0 {
        if o.chunked {
            // highsFprintfString to stdout: highsLogUser(kInfo, "%s")
            highs_rs_log(o.log_options, 0, LogType::Info as i32, text, len);
        } else if !o.file.is_null() {
            cfile::write(o.file as *mut File, crate::ffi::sl(text, len as i32));
        }
        return;
    }
    if kind == 0 {
        highs_rs_log(o.log_options, 1, LogType::Info as i32, text, len);
    } else {
        highs_rs_log(o.log_options, 0, kind, text, len);
    }
}

/// The readers' name lookups on Names (HighsLpUtilsRust.cpp rsNamesOp)
unsafe extern "C" fn names_op(ctx: *mut c_void, code: i32, name: *const u8, len: usize) -> i32 {
    let n = &mut *(ctx as *mut Names);
    let form = |names: &[Vec<u8>], hash: &mut HashMap<Vec<u8>, i32>| {
        // HighsNameHash::form
        hash.clear();
        for (i, s) in names.iter().enumerate() {
            match hash.entry(s.clone()) {
                std::collections::hash_map::Entry::Occupied(mut o) => {
                    o.insert(-1);
                }
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(i as i32);
                }
            }
        }
    };
    if code == 0 {
        // (have_col/have_row as the CNames says, sized by the model)
        if !n.col.is_empty() && n.col_hash.is_empty() {
            let c = n.col.clone();
            form(&c, &mut n.col_hash);
        }
        if !n.row.is_empty() && n.row_hash.is_empty() {
            let r = n.row.clone();
            form(&r, &mut n.row_hash);
        }
        return 0;
    }
    let key = crate::ffi::sl(name, len as i32);
    let hash = if code == 1 { &n.col_hash } else { &n.row_hash };
    hash.get(key).copied().unwrap_or(-2)
}

/// The column of a ranging step: collectAj then FTRAN on the engine
struct RangingFtran {
    e: *mut LpHandle,
    column: crate::hvector::OwnedHVec,
}

unsafe extern "C" fn ranging_ftran(ctx: *mut c_void, j: i32, out: *mut crate::lp_data::ranging::CColumn) {
    use crate::util::fma::ClangFma;
    const TINY: f64 = 1e-14;
    const ZERO: f64 = 1e-50;
    let f = &mut *(ctx as *mut RangingFtran);
    let e = &mut *f.e;
    let v = &mut f.column;
    v.clear();
    // collectAj(column, j, 1)
    let a = &e.model.a;
    let nc = a.num_col as usize;
    let j = j as usize;
    let mut add = |row: usize, x: f64| {
        let value0 = v.array[row];
        let value1 = 1.0f64.mul_add_c(x, value0);
        if value0 == 0.0 {
            v.index[v.count as usize] = row as i32;
            v.count += 1;
        }
        v.array[row] = if value1.abs() < TINY { ZERO } else { value1 };
    };
    if j < nc {
        for el in a.start[j] as usize..a.start[j + 1] as usize {
            add(a.index[el] as usize, a.value[el]);
        }
    } else {
        add(j - nc, 1.0);
    }
    let density = e.lps.sh.info.col_aq_density;
    let mut c = crate::ffi::CHVec::of(v);
    let env = e.env();
    let env = e.lps.env_of(&env);
    e.lps.nla_solve(&env, &mut c, density, false);
    c.store_into(v);
    *out = crate::lp_data::ranging::CColumn { count: v.count, index: v.index.as_ptr(), array: v.array.as_ptr() };
}

/// The KKT errors of a pretty Glpsol solution file
/// (getKktFailures(options, model, solution, basis, info, errors, true))
fn kkt_errors(e: &mut LpHandle, info: &mut crate::lp_data::solution::Info) -> crate::lp_data::solution::PrimalDualErrors {
    use crate::lp_data::solution::{get_kkt_failures, get_primal_dual_basis_errors, get_primal_dual_glpsol_errors, LpRef, SolRef};
    let o = e.opts.kkt(e.log());
    let h = &e.top.as_deref().expect("a Highs object's engine").hessian;
    let hv = hessian_view(h);
    let is_qp = h.dim != 0;
    let r = &mut e.lps.run;
    let m = &mut e.model;
    let n = m.num_col as usize;
    let mut gradient = vec![0.0; n];
    if h.dim > 0 {
        hessian::product(&hv, &r.solution.col_value, &mut gradient);
    }
    for j in 0..n {
        gradient[j] += m.col_cost[j];
    }
    let v = m.view();
    let sv = r.solution.view();
    // SAFETY: the views live for the call
    let (lp, sol) = unsafe { (LpRef::new(&v), SolRef::new(&sv)) };
    get_kkt_failures(&o, is_qp, &lp, &gradient, &sol, info, true);
    // SAFETY: plain data
    let mut errors: crate::lp_data::solution::PrimalDualErrors = unsafe { std::mem::zeroed() };
    let b = &r.basis.b;
    get_primal_dual_basis_errors(&o, &lp, &sol, b.valid, &b.col_status, &b.row_status, &mut errors);
    get_primal_dual_glpsol_errors(&o, &lp, &sol, &mut errors);
    errors
}

/// analyseVectorValues (without the value list) through highsLogDev
fn analyse_vector_values(log: &Log, message: &str, vec: &[f64]) {
    let vec_dim = vec.len();
    if vec_dim == 0 {
        return;
    }
    const NVK: usize = 20;
    let (mut nnz, mut n_pos_inf, mut n_neg_inf) = (0i32, 0i32, 0i32);
    let mut pos_vk = [0i32; NVK + 1];
    let mut neg_vk = [0i32; NVK + 1];
    let mut min_abs = INF;
    let mut max_abs = 0.0f64;
    let log10 = 10f64.ln();
    for &v in vec {
        let abs_v = v.abs();
        if abs_v != 0.0 {
            min_abs = if abs_v < min_abs { abs_v } else { min_abs };
            max_abs = if max_abs < abs_v { abs_v } else { max_abs };
            nnz += 1;
            if -v >= INF {
                n_neg_inf += 1;
            } else if v >= INF {
                n_pos_inf += 1;
            } else {
                let log10_v: i32 = if abs_v == 1.0 {
                    0
                } else if abs_v == 10.0 {
                    1
                } else if abs_v == 100.0 {
                    2
                } else if abs_v == 1000.0 {
                    3
                } else {
                    (abs_v.ln() / log10) as i32
                };
                if log10_v >= 0 {
                    pos_vk[(log10_v as usize).min(NVK)] += 1;
                } else {
                    neg_vk[((-log10_v) as usize).min(NVK)] += 1;
                }
            }
        }
    }
    if nnz == 0 {
        min_abs = 0.0;
    }
    let pct = (1e2 * nnz as f64 / vec_dim as f64) as i32;
    log_dev!(
        log,
        LogType::Info,
        "%s of dimension %d with %d nonzeros (%3d%%) in [%11.4g, %11.4g]\n",
        message,
        vec_dim as i32,
        nnz,
        pct,
        min_abs,
        max_abs
    );
    if n_neg_inf > 0 {
        log_dev!(log, LogType::Info, "%12d values are -Inf\n", n_neg_inf);
    }
    if n_pos_inf > 0 {
        log_dev!(log, LogType::Info, "%12d values are +Inf\n", n_pos_inf);
    }
    if pos_vk[NVK] > 0 {
        log_dev!(log, LogType::Info, "%12d values satisfy 10^(%3d) <= v < Inf\n", pos_vk[NVK], NVK as i32);
    }
    for k in (0..NVK).rev() {
        if pos_vk[k] > 0 {
            log_dev!(log, LogType::Info, "%12d values satisfy 10^(%3d) <= v < 10^(%3d)\n", pos_vk[k], k as i32 - 1, k as i32);
        }
    }
    for k in 1..=NVK {
        if neg_vk[k] > 0 {
            log_dev!(log, LogType::Info, "%12d values satisfy 10^(%3d) <= v < 10^(%3d)\n", neg_vk[k], -(k as i32) - 1, -(k as i32));
        }
    }
    let zeros = vec_dim as i32 - nnz;
    if zeros > 0 {
        log_dev!(log, LogType::Info, "%12d values are zero\n", zeros);
    }
}

/// writeMatrixPicToFile: the sparsity pattern of a column-wise matrix as
/// a .pbm image
fn write_matrix_pic(log: &Log, prefix: &[u8], num_row: i32, num_col: i32, start: &[i32], index: &[i32]) -> Status {
    if num_row == 0 || num_col == 0 {
        log_user!(log, LogType::Error, "Cannot generate image of matrix with a zero dimension\n");
        return Status::Error;
    }
    let (nr, nc) = (num_row as usize, num_col as usize);
    let num_nz = start[nc] as usize;
    let mut ar_length = vec![0usize; nr];
    for &i in &index[..num_nz] {
        ar_length[i as usize] += 1;
    }
    let mut ar_start = vec![0usize; nr + 1];
    for r in 0..nr {
        ar_start[r + 1] = ar_start[r] + ar_length[r];
    }
    let mut ar_index = vec![0usize; num_nz];
    for c in 0..nc {
        for el in start[c] as usize..start[c + 1] as usize {
            let r = index[el] as usize;
            ar_index[ar_start[r]] = c;
            ar_start[r] += 1;
        }
    }
    ar_start[0] = 0;
    for r in 0..nr {
        ar_start[r + 1] = ar_start[r] + ar_length[r];
    }
    // writeRmatrixPicToFile
    let mut filename = prefix.to_vec();
    filename.extend_from_slice(b".pbm");
    const BORDER: usize = 1;
    const MAX_WIDE: usize = 1600 - 2 * BORDER;
    const MAX_DEEP: usize = 900 - 2 * BORDER;
    let per = |n: usize, max: usize| if n > max { n / max + usize::from((n / max) * max < n) } else { 1 };
    let dim_per_pixel = per(nc, MAX_WIDE).max(per(nr, MAX_DEEP));
    let mut wide = nc / dim_per_pixel;
    if dim_per_pixel * wide < nc {
        wide += 1;
    }
    let mut deep = nr / dim_per_pixel;
    if dim_per_pixel * deep < nr {
        deep += 1;
    }
    wide += 2;
    deep += 2;
    log_user!(
        log,
        LogType::Info,
        "Representing matrix sparsity pattern %dx%d .pbm file, mapping entries in square of size %d onto one pixel\n",
        wide as i32,
        deep as i32,
        dim_per_pixel as i32
    );
    let mut f = String::new();
    f.push_str(&format!("P1\n{} {}\n", wide, deep));
    f.push_str(&"1 ".repeat(wide));
    f.push('\n');
    let mut value = vec![0u8; wide];
    let mut from_row = 0;
    loop {
        let to_row = (from_row + dim_per_pixel).min(nr);
        for r in from_row..to_row {
            for &c in &ar_index[ar_start[r]..ar_start[r + 1]] {
                value[c / dim_per_pixel] = 1;
            }
        }
        f.push_str("1 ");
        for v in value.iter_mut().take(wide - 2) {
            f.push_str(if *v != 0 { "1 " } else { "0 " });
            *v = 0;
        }
        f.push_str("1 \n");
        if to_row == nr {
            break;
        }
        from_row = to_row;
    }
    f.push_str(&"1 ".repeat(wide));
    f.push('\n');
    use std::os::unix::ffi::OsStrExt;
    let _ = std::fs::write(std::ffi::OsStr::from_bytes(&filename), f);
    Status::Ok
}

// ---- The MIP solver's host (HighsFns)

/// The context of a MIP solve's callbacks: the improving solution file
/// with the model's (or semi-variable LP's) objective and column names
pub struct MipHost {
    log_head: *const LogOptionsHead,
    file_name: Vec<u8>,
    sparse: bool,
    offset: f64,
    cost: Vec<f64>,
    names: Vec<Vec<u8>>,
    improving: *mut File,
}

unsafe extern "C" fn mip_callback(
    _ctx: *mut c_void,
    _which: i32,
    _type: i32,
    _out: *const crate::mip::host::CallbackOut,
    _sol: *const f64,
    _n: i32,
    _message: *const u8,
    _len: i32,
) -> bool {
    // No user callback
    false
}

unsafe extern "C" fn mip_user_solution(_ctx: *mut c_void, n: *mut i32) -> *const f64 {
    *n = 0;
    std::ptr::null()
}

unsafe extern "C" fn mip_cut_pool_output(
    _ctx: *mut c_void,
    _num_col: i32,
    _num_cut: i32,
    _lower: *const f64,
    _upper: *const f64,
    _start: *const i32,
    _index: *const i32,
    _value: *const f64,
    _nnz: i32,
) {
}

unsafe extern "C" fn mip_improving_file(ctx: *mut c_void, op: i32, sol: *const f64, n: i32) {
    let h = &mut *(ctx as *mut MipHost);
    match op {
        0 => h.improving = cfile::open(&h.file_name, "w"),
        1 => {
            if h.improving.is_null() {
                return;
            }
            let solution = crate::ffi::sl(sol, n);
            let out = COut {
                file: h.improving as *mut c_void,
                log_options: h.log_head as *const c_void,
                chunked: h.improving == cfile::stdout(),
                emit: Some(out_emit),
            };
            // writeLpObjective, then writePrimalSolution
            let objective: f64 = model_utils::lp_objective_cdouble(h.offset, &h.cost, solution).into();
            writers::write_objective_value(&mut Out::new(Some(&out)), objective);
            let names: Vec<&[u8]> = h.names.iter().map(|n| cstr(n)).collect();
            writers::write_primal_solution(&mut Out::new(Some(&out)), &names, solution, h.cost.len(), h.sparse);
            cfile::flush(h.improving);
        }
        _ => {
            if !h.improving.is_null() {
                cfile::close(h.improving);
            }
            h.improving = std::ptr::null_mut();
        }
    }
}

static MIP_FNS: crate::mip::host::HighsFns = crate::mip::host::HighsFns {
    profiling: profiling::fns::mip,
    callback: mip_callback,
    user_solution: mip_user_solution,
    cut_pool_output: mip_cut_pool_output,
    improving_file: mip_improving_file,
};

// ---- The drivers' op and the IIS host functions

unsafe extern "C" fn file_op(ctx: *mut c_void, which: i32, arg: i64, p: *mut c_void, msg: *const u8, len: usize) -> i64 {
    let h = &mut *(ctx as *mut Highs);
    let m = if len == 0 { &[][..] } else { std::slice::from_raw_parts(msg, len) };
    match h.file_step(which, arg, p, m) {
        Some(r) => r,
        None => super::handle_op(&mut *h.e as *mut LpHandle as *mut c_void, which, arg, p, msg, len),
    }
}

unsafe extern "C" fn iis_op(ctx: *mut c_void, which: i32, a: *const Args) -> f64 {
    (*(ctx as *mut Highs)).iis_step(which, &*a)
}

unsafe extern "C" fn iis_lp(ctx: *mut c_void, out: *mut CLp, nc: *mut usize, nr: *mut usize) {
    let h = &mut *(ctx as *mut Highs);
    *out = h.e.model.view();
    let n = h.e.names.as_deref().expect("the object's names");
    *nc = n.col.len();
    *nr = n.row.len();
}

unsafe extern "C" fn iis_name(ctx: *mut c_void, is_col: bool, i: i32) -> RsStr {
    let h = &*(ctx as *const Highs);
    let n = h.e.names.as_deref().expect("the object's names");
    let s = if i < 0 {
        &h.e.model.model_name
    } else if is_col {
        &n.col[i as usize]
    } else {
        &n.row[i as usize]
    };
    RsStr::of(s)
}

unsafe extern "C" fn iis_col_value(ctx: *mut c_void) -> RsMut<f64> {
    let v = &mut (*(ctx as *mut Highs)).e.lps.run.solution.col_value;
    RsMut { ptr: v.as_mut_ptr(), len: v.len() }
}

// ---- The app's objects (HighsAppRust.cpp)

/// The app's Highs object and loaded options
struct App {
    highs: Box<Highs>,
    loaded: Box<Opts>,
    /// loaded.log_options
    loaded_head: Box<LogOptionsHead>,
}

impl App {
    /// The option host of the loaded options, reporting to the Highs
    /// object's log
    fn loaded<R>(&mut self, f: impl FnOnce(&COptionHost, &[COptionRecord]) -> R) -> R {
        let log = self.highs.log();
        let recs = records(&mut self.loaded);
        let mut ctx = LogFileCtx { head: &mut *self.loaded_head, log_file: &mut self.loaded.log_file };
        let host = option_host(log, &mut ctx);
        f(&host, &recs)
    }
}

impl Drop for App {
    fn drop(&mut self) {
        let f = self.loaded_head.log_stream;
        if !f.is_null() {
            // SAFETY: the stream the loaded options opened
            unsafe { cfile::close(f) };
        }
    }
}

unsafe extern "C" fn app_op(
    ctx: *mut c_void,
    code: i32,
    arg: i64,
    x: f64,
    s: *const u8,
    n: usize,
    s2: *const u8,
    n2: usize,
    out: *mut c_void,
) -> i64 {
    let a = &mut *(ctx as *mut App);
    let s = crate::ffi::sl(s, n as i32);
    let s2 = crate::ffi::sl(s2, n2 as i32);
    let h = &mut a.highs;
    match code {
        0 => {
            let f = if arg == 2 { cfile::stderr() } else { cfile::stdout() };
            cfile::write(f, s);
            cfile::flush(f);
            0
        }
        1 => {
            h.log_header();
            0
        }
        2 => {
            h.close_log_file();
            0
        }
        3 => {
            let name = a.loaded.log_file.clone();
            a.highs.open_log_file(&name);
            0
        }
        4 => {
            let mut loaded = a.loaded.clone();
            a.highs.pass_options(&mut loaded) as i64
        }
        5 => h.write_options(b"", true) as i64,
        6 => h.read_model(&lossy(s)) as i64,
        7 => h.presolve() as i64,
        8 => h.presolve_status() as i64,
        9 => {
            let f = lossy(&h.e.opts.write_presolved_model_file);
            h.write_presolved_model(&f) as i64
        }
        10 => !h.e.opts.write_presolved_model_file.is_empty() as i64,
        11 => h.run() as i64,
        12 => {
            crate::parallel::shutdown(true);
            0
        }
        13 => a.loaded.output_flag as i64,
        14 => a.loaded(|host, r| options::load_options_from_file(host, r, s)) as i64,
        15 => {
            let log = log_of(&a.loaded_head);
            let recs = records(&mut a.loaded);
            let mut ctx = LogFileCtx { head: &mut *a.loaded_head, log_file: &mut a.loaded.log_file };
            let host = option_host(log, &mut ctx);
            options::write_options(&host, cfile::stdout() as *mut c_void, true, &recs, false, FILE_FULL);
            0
        }
        16 => a.loaded(|host, r| options::set_from_string(host, s, r, s2)) as i64,
        17 => a.loaded(|host, r| options::set_int(&host.log, s, r, arg as i32)) as i64,
        18 => a.loaded(|host, r| options::set_double(&host.log, s, r, x)) as i64,
        19 => {
            let text: String = match arg {
                0 => format!("{}.{}.{}", VERSION[0], VERSION[1], VERSION[2]),
                1 => GITHASH.into(),
                2 => COPYRIGHT.into(),
                3 => NOTICE_HEADER.into(),
                4 => APP_NOTICE.into(),
                _ => CLI11_VERSION.into(),
            };
            (*(out as *mut Vec<u8>)).extend_from_slice(text.as_bytes());
            0
        }
        _ => 0,
    }
}

/// highs_app_create of crest: the app's Highs object and loaded options
/// (log_file kHighsRunLogFile, the app's default) in a host for app.rs;
/// free the returned context with [`app_destroy`]
pub fn app_create() -> (AppHost, *mut c_void) {
    let highs = Highs::new();
    let mut loaded = Box::<Opts>::default();
    loaded.log_file = b"Highs.log".to_vec();
    let loaded_head = Box::new(head_on(&loaded));
    let log = highs.log();
    let app = Box::new(App { highs, loaded, loaded_head });
    let ctx = Box::into_raw(app) as *mut c_void;
    (AppHost { ctx, op: app_op, log }, ctx)
}

/// highs_app_destroy
///
/// # Safety
/// `ctx` from [`app_create`], not used after
pub unsafe fn app_destroy(ctx: *mut c_void) {
    drop(Box::from_raw(ctx as *mut App));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The records' metadata are the fields of Opts, in order, and their
    /// defaults are Opts::default()'s values
    #[test]
    fn option_records_match_opts() {
        let names: Vec<&str> = RECORDS.iter().map(|m| m.name).collect();
        assert_eq!(names, crate::lp_data::opts::NAMES);
        let mut o = Opts::default();
        for r in &records(&mut o) {
            // SAFETY: the table's values point into `o`
            let same = unsafe {
                match r.type_ {
                    options::BOOL => *(r.value as *const bool) == r.bool_default,
                    options::INT => *(r.value as *const i32) == r.int_default,
                    options::DOUBLE => (*(r.value as *const f64)).to_bits() == r.dbl_default.to_bits(),
                    _ => r.str_value.get() == r.str_default.get(),
                }
            };
            assert!(same, "{}", String::from_utf8_lossy(r.name()));
        }
    }
}
