//! The highs app (app/RunHighs.cpp's main and HighsRuntimeOptions.h's
//! loadOptions): parse the command line (options_cli.rs), load the options
//! into a separate HighsOptions, open the log file, pass the options, read
//! the model, presolve and write the presolved model or run, and close.
//! The `Highs` object and the loaded options are the host's (`AppHost`):
//! the C++ ones for the C++ `highs` app under HIGHS_RUST (highs_app_create
//! in lp_data/HighsAppRust.cpp), the Rust ones for `crest`
//! (lp_data/highs.rs `app_create`); each step on them is one `op`.
//! Output to stdout/stderr goes through C's stdio (`Print`), so it
//! interleaves with the logging as before.

use super::options_cli::{parse, CommandLine, Outcome};
use super::{Log, LogType};
use std::ffi::{c_char, c_void, CStr};

/// The steps on the C++ objects (codes of highs_app_op)
#[derive(Clone, Copy)]
#[repr(i32)]
enum Op {
    /// fwrite s to stdout (arg 1) or stderr (arg 2), then fflush
    Print = 0,
    LogHeader,
    CloseLogFile,
    /// highs.openLogFile(loaded.log_file)
    OpenLoadedLogFile,
    /// highs.passOptions(loaded)
    PassOptions,
    /// highs.writeOptions("", true)
    WriteChangedOptions,
    ReadModel,
    Presolve,
    PresolveStatus,
    /// highs.writePresolvedModel(options.write_presolved_model_file)
    WritePresolvedModel,
    /// options.write_presolved_model_file != ""
    HasPresolvedModelFile,
    Run,
    /// Highs::resetGlobalScheduler(true)
    ResetScheduler,
    /// loaded.output_flag
    LoadedOutputFlag,
    /// loadOptionsFromFile(report_log_options, loaded, s)
    LoadOptionsFile,
    /// writeOptionsToFile(stdout, loaded.log_options, loaded.records)
    WriteLoadedOptions,
    /// setLocalOptionValue(report_log_options, s, loaded.log_options, loaded.records, s2)
    SetString,
    /// setLocalOptionValue(report_log_options, s, loaded.records, HighsInt(arg))
    SetInt,
    /// setLocalOptionValue(report_log_options, s, loaded.records, x)
    SetDouble,
    /// A text into `out` (highs_rs_app_put): arg 0 the version, 1 the
    /// githash, 2 kHighsCopyrightStatement, 3 the third-party notice
    /// header, 4 the app's third-party notice, 5 CLI11_VERSION
    Text,
}

/// What C++ hands the app (RsAppHost)
#[repr(C)]
pub struct AppHost {
    pub ctx: *mut c_void,
    #[allow(clippy::type_complexity)]
    pub op: unsafe extern "C" fn(
        ctx: *mut c_void,
        code: i32,
        arg: i64,
        x: f64,
        s: *const u8,
        n: usize,
        s2: *const u8,
        n2: usize,
        out: *mut c_void,
    ) -> i64,
    /// highs.getOptions().log_options
    pub log: Log,
}

/// Appends bytes to the Vec<u8> `out` (Op::Text's result)
///
/// # Safety
/// `out` is the `*mut Vec<u8>` passed to the op; `p` holds `n` bytes
#[no_mangle]
pub unsafe extern "C" fn highs_rs_app_put(out: *mut c_void, p: *const u8, n: usize) {
    if n > 0 {
        (*(out as *mut Vec<u8>)).extend_from_slice(std::slice::from_raw_parts(p, n));
    }
}

const ERROR: i32 = -1;

extern "C" {
    fn exit(code: i32) -> !;
}

struct App<'a> {
    h: &'a AppHost,
    written_cli_copyright_line: bool,
}

impl App<'_> {
    fn call(&self, op: Op, arg: i64, x: f64, s: &[u8], s2: &[u8], out: *mut c_void) -> i64 {
        // SAFETY: the host's op takes its ctx; the slices outlive the call
        unsafe { (self.h.op)(self.h.ctx, op as i32, arg, x, s.as_ptr(), s.len(), s2.as_ptr(), s2.len(), out) }
    }
    fn op(&self, op: Op) -> i64 {
        self.call(op, 0, 0.0, &[], &[], std::ptr::null_mut())
    }
    fn op_s(&self, op: Op, s: &[u8]) -> i64 {
        self.call(op, 0, 0.0, s, &[], std::ptr::null_mut())
    }
    fn print(&self, stream: i64, s: &[u8]) {
        self.call(Op::Print, stream, 0.0, s, &[], std::ptr::null_mut());
    }
    fn text(&self, which: i64) -> Vec<u8> {
        let mut v: Vec<u8> = Vec::new();
        self.call(Op::Text, which, 0.0, &[], &[], &mut v as *mut Vec<u8> as *mut c_void);
        v
    }

    fn cli_copyright_line(&mut self) {
        if self.written_cli_copyright_line {
            return;
        }
        let version = String::from_utf8_lossy(&self.text(5)).into_owned();
        crate::log_user!(
            self.h.log,
            LogType::Info,
            "Command line parsed using CLI11 %s: Copyright (c) 2017-2025 University of Cincinnati\n",
            version.as_str()
        );
        self.written_cli_copyright_line = true;
    }

    /// runHighsReturn
    fn ret(&mut self, status: i32) -> i32 {
        self.op(Op::LogHeader);
        self.cli_copyright_line();
        self.op(Op::CloseLogFile);
        status
    }

    /// loadOptions
    fn load_options(&self, c: &CommandLine) -> bool {
        if c.version || c.notice {
            let mut s = Vec::new();
            s.extend_from_slice(b"HiGHS version ");
            s.extend(self.text(0));
            s.extend_from_slice(b" Githash ");
            s.extend(self.text(1));
            s.extend_from_slice(b". ");
            s.extend(self.text(2));
            s.push(b'\n');
            s.extend(self.text(3));
            s.push(b'\n');
            if c.notice {
                s.push(b'\n');
                s.extend(self.text(4));
                s.push(b'\n');
            }
            self.print(1, &s);
            // SAFETY: C's exit, as the C++ app
            unsafe { exit(0) }
        }
        let options_file = &c.strings[1];
        if !options_file.is_empty() {
            match self.op_s(Op::LoadOptionsFile, options_file) {
                -1 => return false,
                1 => {
                    self.op(Op::WriteLoadedOptions);
                    return false;
                }
                _ => {}
            }
        }
        let set = |name: &str, value: &[u8]| {
            value.is_empty() || self.call(Op::SetString, 0, 0.0, name.as_bytes(), value, std::ptr::null_mut()) == 0
        };
        let set_int =
            |name: &str, v: i32| self.call(Op::SetInt, v as i64, 0.0, name.as_bytes(), &[], std::ptr::null_mut()) == 0;
        let ok = set("read_solution_file", &c.strings[2])
            && set("read_basis_file", &c.strings[3])
            && set("write_model_file", &c.strings[4])
            && set("solution_file", &c.strings[5])
            && set("write_basis_file", &c.strings[6])
            && set("presolve", &c.strings[7])
            && set("solver", &c.strings[8])
            && set("parallel", &c.strings[9])
            && (c.count[super::options_cli::THREADS] == 0 || set_int("threads", c.threads))
            && set("run_crossover", &c.strings[11])
            && (c.count[super::options_cli::TIME_LIMIT] == 0
                || self.call(Op::SetDouble, 0, c.time_limit, b"time_limit", &[], std::ptr::null_mut()) == 0)
            && (c.count[super::options_cli::RANDOM_SEED] == 0 || set_int("random_seed", c.random_seed))
            && set("ranging", &c.strings[14]);
        if !ok {
            return false;
        }
        if c.strings[0].is_empty() {
            self.print(1, b"Please specify filename in .mps|.lp|.ems format.\n");
            return false;
        }
        true
    }

    fn main(&mut self, args: &[Vec<u8>]) -> i32 {
        let program = args.first().cloned().unwrap_or_default();
        let mut c = CommandLine::default();
        let line = |m: &[u8], tail: &[u8]| [m, b"\n", tail].concat();
        match parse(args.get(1..).unwrap_or(&[]), &program, &mut c) {
            Outcome::Ok => {}
            Outcome::Help(m) => {
                self.print(1, &line(&m, b""));
                return self.ret(0);
            }
            Outcome::Extras(m) => {
                self.print(1, &line(&m, b"Multiple files not supported.\n"));
                return self.ret(ERROR);
            }
            Outcome::Mismatch(m) => {
                self.print(1, &line(&m, b"Too many arguments provided. Please provide only one.\n"));
                return self.ret(ERROR);
            }
            Outcome::Error(m, code) => {
                self.print(1, &line(&m, b""));
                // CLI::App::exit's message
                self.print(2, &line(&m, b"Run with --help for more information.\n"));
                return self.ret(code);
            }
        }
        if !self.load_options(&c) {
            return self.ret(ERROR);
        }
        // Open the app log file - unless output_flag is false, to avoid
        // creating an empty file
        if self.op(Op::LoadedOutputFlag) != 0 {
            self.op(Op::OpenLoadedLogFile);
        }
        self.op(Op::PassOptions);
        self.op(Op::LogHeader);
        self.cli_copyright_line();
        self.op(Op::WriteChangedOptions);

        let read_status = self.op_s(Op::ReadModel, &c.strings[0]) as i32;
        if read_status == ERROR {
            crate::log_user!(self.h.log, LogType::Info, "Error loading file\n");
            return self.ret(read_status);
        }
        if self.op(Op::HasPresolvedModelFile) != 0 {
            let status = self.op(Op::Presolve) as i32;
            if status == ERROR {
                return self.ret(status);
            }
            // kNotReduced, kReduced, kReducedToEmpty, kTimeout
            if !matches!(self.op(Op::PresolveStatus), 0 | 3 | 4 | 5) {
                crate::log_user!(self.h.log, LogType::Info, "No presolved model to write to file\n");
                return self.ret(status);
            }
            let status = self.op(Op::WritePresolvedModel) as i32;
            return self.ret(status);
        }
        let run_status = self.op(Op::Run) as i32;
        if run_status == ERROR {
            // The C++ app calls runHighsReturn here and ignores its result
            self.ret(run_status);
        }
        self.op(Op::ResetScheduler);
        self.ret(run_status)
    }
}

/// The app's main on `args` (argv as bytes): its exit status
pub fn app_main(args: &[Vec<u8>], host: &AppHost) -> i32 {
    App { h: host, written_cli_copyright_line: false }.main(args)
}

/// # Safety
/// `argv` holds `argc` NUL-terminated strings; `host` comes from
/// highs_app_create
#[no_mangle]
pub unsafe extern "C" fn highs_rs_app_main(argc: i32, argv: *const *const c_char, host: *const AppHost) -> i32 {
    let args: Vec<Vec<u8>> = (0..argc.max(0) as usize).map(|k| CStr::from_ptr(*argv.add(k)).to_bytes().to_vec()).collect();
    app_main(&args, &*host)
}
